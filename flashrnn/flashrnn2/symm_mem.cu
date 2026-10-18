// Symmetric exchange buffers for the multi-GPU recurrence (multi_gpu.py, alloc="nccl_symm"): NCCL 2.32 memory
// (ncclMemAlloc, cuMem-backed) registered as symmetric windows on one communicator per GPU of this process
// (ncclCommInitAll), with every peer's address taken from the window's LSA (load/store-accessible) mapping on the
// device (ncclGetLsaPointer). The recurrence kernels are unchanged: they read the int64 peer-address tables.
//
// The process has already loaded PyTorch's own (older) libnccl.so.2, with the same soname. NCCL 2.32 is therefore
// opened by its full path with RTLD_LOCAL | RTLD_DEEPBIND and called through dlsym'd pointers only; nothing here links
// against libnccl. The device functions (ncclGetLsaPointer) are header-inline.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct Api {
  void* lib = nullptr;
  decltype(&ncclGetVersion) getVersion;
  decltype(&ncclCommInitAll) commInitAll;
  decltype(&ncclCommDestroy) commDestroy;
  decltype(&ncclMemAlloc) memAlloc;
  decltype(&ncclMemFree) memFree;
  decltype(&ncclCommWindowRegister) winRegister;
  decltype(&ncclCommWindowDeregister) winDeregister;
  decltype(&ncclGroupStart) groupStart;
  decltype(&ncclGroupEnd) groupEnd;
  decltype(&ncclGetErrorString) errorString;
};

struct Buffer {
  size_t bytes = 0;
  std::vector<void*> ptrs;
  std::vector<ncclWindow_t> wins;
};

Api api;
std::vector<int> devs;
std::vector<ncclComm_t> comms;
std::vector<Buffer> buffers;
char last_error[512] = "";

int fail(const char* what, int code) {
  snprintf(last_error, sizeof(last_error), "%s: %d", what, code);
  return code ? code : -1;
}

#define NCCL_OR_FAIL(call)                                                                                     \
  do {                                                                                                         \
    ncclResult_t r_ = (call);                                                                                  \
    if (r_ != ncclSuccess) {                                                                                   \
      snprintf(last_error, sizeof(last_error), "%s: %s", #call, api.errorString ? api.errorString(r_) : "?"); \
      return 1000 + (int)r_;                                                                                   \
    }                                                                                                          \
  } while (0)
#define CUDA_OR_FAIL(call)                                                                       \
  do {                                                                                           \
    cudaError_t e_ = (call);                                                                     \
    if (e_ != cudaSuccess) {                                                                     \
      snprintf(last_error, sizeof(last_error), "%s: %s", #call, cudaGetErrorString(e_));         \
      return 2000 + (int)e_;                                                                     \
    }                                                                                            \
  } while (0)

template <class F>
bool sym(F& f, const char* name) {
  f = reinterpret_cast<F>(dlsym(api.lib, name));
  return f != nullptr;
}

__global__ void lsa_table(ncclWindow_t win, size_t offset, int n, long long* out) {
  int j = threadIdx.x;
  if (j < n) out[j] = (long long)(uintptr_t)ncclGetLsaPointer(win, offset, j);
}

}  // namespace

extern "C" const char* frnn_symm_last_error() { return last_error; }

// Opens NCCL at `path` (no GPU needed) and reports its version: proves that the 2.32 library, not PyTorch's, answers.
extern "C" int frnn_symm_version(const char* path, int* version) {
  void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (!lib) {
    snprintf(last_error, sizeof(last_error), "dlopen: %s", dlerror());
    return 1;
  }
  auto get = reinterpret_cast<decltype(&ncclGetVersion)>(dlsym(lib, "ncclGetVersion"));
  if (!get) return 2;
  return get(version) == ncclSuccess ? 0 : 3;
}

// Opens NCCL at `path` and creates one communicator per device (ranks in the order given). Returns 0 or an error.
extern "C" int frnn_symm_init(const char* path, int ndev, const int* devices, int* version) {
  if (!api.lib) {
    api.lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!api.lib) {
      snprintf(last_error, sizeof(last_error), "dlopen: %s", dlerror());
      return 1;
    }
    bool ok = sym(api.getVersion, "ncclGetVersion") && sym(api.commInitAll, "ncclCommInitAll") &&
              sym(api.commDestroy, "ncclCommDestroy") && sym(api.memAlloc, "ncclMemAlloc") &&
              sym(api.memFree, "ncclMemFree") && sym(api.winRegister, "ncclCommWindowRegister") &&
              sym(api.winDeregister, "ncclCommWindowDeregister") && sym(api.groupStart, "ncclGroupStart") &&
              sym(api.groupEnd, "ncclGroupEnd") && sym(api.errorString, "ncclGetErrorString");
    if (!ok) {
      snprintf(last_error, sizeof(last_error), "dlsym: missing NCCL symbol");
      return 2;
    }
  }
  NCCL_OR_FAIL(api.getVersion(version));
  if (!comms.empty()) {
    snprintf(last_error, sizeof(last_error), "already initialized");
    return 3;
  }
  devs.assign(devices, devices + ndev);
  comms.resize(ndev);
  NCCL_OR_FAIL(api.commInitAll(comms.data(), ndev, devs.data()));
  return 0;
}

// Allocates `bytes` on every device and registers the allocations as one symmetric window. Returns the buffer id
// (>= 0) or a negative error.
extern "C" int frnn_symm_alloc(size_t bytes) {
  Buffer b;
  b.bytes = bytes;
  b.ptrs.assign(devs.size(), nullptr);
  b.wins.assign(devs.size(), nullptr);
  for (size_t i = 0; i < devs.size(); ++i) {
    if (cudaSetDevice(devs[i]) != cudaSuccess) return -fail("cudaSetDevice", 1);
    if (api.memAlloc(&b.ptrs[i], bytes) != ncclSuccess) return -fail("ncclMemAlloc", 2);
    if (cudaMemset(b.ptrs[i], 0, bytes) != cudaSuccess) return -fail("cudaMemset", 3);
  }
  // Window registration is collective over the communicators: one group.
  if (api.groupStart() != ncclSuccess) return -fail("ncclGroupStart", 4);
  for (size_t i = 0; i < devs.size(); ++i) {
    cudaSetDevice(devs[i]);
    ncclResult_t r = api.winRegister(comms[i], b.ptrs[i], bytes, &b.wins[i], NCCL_WIN_COLL_SYMMETRIC);
    if (r != ncclSuccess) {
      api.groupEnd();
      snprintf(last_error, sizeof(last_error), "ncclCommWindowRegister: %s", api.errorString(r));
      return -5;
    }
  }
  ncclResult_t r = api.groupEnd();
  if (r != ncclSuccess) {
    snprintf(last_error, sizeof(last_error), "ncclGroupEnd (window register): %s", api.errorString(r));
    return -6;
  }
  for (size_t i = 0; i < devs.size(); ++i) {
    cudaSetDevice(devs[i]);
    if (cudaDeviceSynchronize() != cudaSuccess) return -fail("cudaDeviceSynchronize", 7);
  }
  buffers.push_back(b);
  return (int)buffers.size() - 1;
}

// The device's own (local) address of buffer `id` on rank `rank`.
extern "C" long long frnn_symm_local(int id, int rank) { return (long long)(uintptr_t)buffers[id].ptrs[rank]; }

// Writes, on rank `rank`'s device, the LSA address of every rank's copy of buffer `id` (+ offset) into the
// device array `table` (int64, one entry per rank). Synchronous.
extern "C" int frnn_symm_peer_table(int id, int rank, size_t offset, long long* table) {
  CUDA_OR_FAIL(cudaSetDevice(devs[rank]));
  lsa_table<<<1, 32>>>(buffers[id].wins[rank], offset, (int)devs.size(), table);
  CUDA_OR_FAIL(cudaGetLastError());
  CUDA_OR_FAIL(cudaDeviceSynchronize());
  return 0;
}

// Zeroes bytes [offset, offset + bytes) of rank `rank`'s copy of buffer `id` on `stream` (0: the default stream).
extern "C" int frnn_symm_zero(int id, int rank, size_t offset, size_t bytes, void* stream) {
  CUDA_OR_FAIL(cudaSetDevice(devs[rank]));
  CUDA_OR_FAIL(cudaMemsetAsync(static_cast<char*>(buffers[id].ptrs[rank]) + offset, 0, bytes, (cudaStream_t)stream));
  return 0;
}

// Copies bytes of rank `rank`'s copy of buffer `id` to host memory (diagnostics and the address probe).
extern "C" int frnn_symm_read(int id, int rank, size_t offset, size_t bytes, void* host) {
  CUDA_OR_FAIL(cudaSetDevice(devs[rank]));
  CUDA_OR_FAIL(cudaDeviceSynchronize());
  CUDA_OR_FAIL(cudaMemcpy(host, static_cast<char*>(buffers[id].ptrs[rank]) + offset, bytes, cudaMemcpyDeviceToHost));
  return 0;
}

extern "C" int frnn_symm_finalize() {
  for (auto& b : buffers) {
    api.groupStart();
    for (size_t i = 0; i < devs.size(); ++i)
      if (b.wins[i]) api.winDeregister(comms[i], b.wins[i]);
    api.groupEnd();
    for (size_t i = 0; i < devs.size(); ++i) {
      cudaSetDevice(devs[i]);
      if (b.ptrs[i]) api.memFree(b.ptrs[i]);
    }
  }
  buffers.clear();
  for (auto c : comms) api.commDestroy(c);
  comms.clear();
  devs.clear();
  return 0;
}
