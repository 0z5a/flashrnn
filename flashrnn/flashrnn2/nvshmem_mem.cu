// NVSHMEM 3.9 symmetric memory for the multi-GPU recurrence, one process per GPU (tools/scale_nvshmem.py): unique-ID
// bootstrap (rank 0 creates the ID, the caller passes it to the others), nvshmem_malloc (collective, same order on
// every PE), and the address of a symmetric object on another PE as seen from this GPU (nvshmem_ptr: non-null for PEs
// reachable by load/store, i.e. NVLink peers). Host API only; the recurrence kernels take the int64 address tables.
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdint>
#include <cstring>

extern "C" int frnn_shm_uid_bytes() { return (int)sizeof(nvshmemx_uniqueid_t); }

extern "C" int frnn_shm_uid(void* out) {
  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  int rc = nvshmemx_get_uniqueid(&id);
  memcpy(out, &id, sizeof(id));
  return rc;
}

// The caller has set the CUDA device of this PE.
extern "C" int frnn_shm_init(int rank, int nranks, const void* uid) {
  nvshmemx_uniqueid_t id;
  memcpy(&id, uid, sizeof(id));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  int rc = nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
  if (rc) return 100 + rc;
  rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  if (rc) return 200 + rc;
  return nvshmem_my_pe() == rank && nvshmem_n_pes() == nranks ? 0 : 300;
}

extern "C" long long frnn_shm_malloc(size_t bytes) {
  void* p = nvshmem_malloc(bytes);
  if (p && cudaMemset(p, 0, bytes) != cudaSuccess) return 0;
  cudaDeviceSynchronize();
  return (long long)(uintptr_t)p;
}

extern "C" long long frnn_shm_ptr(long long local, int pe) {
  return (long long)(uintptr_t)nvshmem_ptr(reinterpret_cast<void*>(local), pe);
}

extern "C" int frnn_shm_zero(long long local, size_t bytes) {
  return (int)cudaMemset(reinterpret_cast<void*>(local), 0, bytes);
}

extern "C" void frnn_shm_barrier() {
  cudaDeviceSynchronize();
  nvshmem_barrier_all();
  cudaDeviceSynchronize();
}

extern "C" void frnn_shm_finalize() { nvshmem_finalize(); }
