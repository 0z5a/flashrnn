// Scale-out phase B exchange skeleton (docs/flashrnn2/scale-out-plan.md, section 7).
//
// One MPI rank per GPU. Every step: a compute spin of gamma ns (the measured per-step compute of the recurrence),
// write this rank's slice of h_t (pattern f(rank, step, i)), exchange the slices with a schedule, wait until every
// slice of step t is present, next step. The schedule is a host-built table (rounds of sends with block ranges), so
// one kernel runs every schedule:
//   flat      one round: own block to every peer, visited in stride order (r + k*s) mod P
//   cyclic    P-1 rounds: round k sends own block to (r + k*s) mod P (gcd(s, P) = 1: every round a permutation)
//   ring      P-1 rounds of store-and-forward to r+1
//   doubling  log2 P rounds: partner r ^ 2^k, the 2^k blocks owned so far (P a power of two)
//   hier      round 0: own block to the same local rank on every other node (network); round 1: the N blocks of
//             this local column to every local peer (NVLink)
// Transports
//   --backend gin    NCCL 2.32 device API: same-node peers by LSA pointer stores + a system-scope flag add; other
//                    nodes by gin.put with a weak GIN signal increment
//   --backend host   (same binary) stream-ordered spin kernel + ncclAllGather per step, optionally a CUDA graph
//   BACKEND_SHMEM    NVSHMEM 3.9: nvshmemx_putmem_signal_nbi_block to every peer (P2P inside the node, IB across)
// Arrival counters are separate per (step parity, round): a peer can be at most one step ahead, so the counter of
// parity p only ever counts step-t arrivals while step t is being waited for. The expected counts are absolute:
// before each launch, after a barrier at which every earlier arrival has landed, a one-thread kernel snapshots them.
// Output: one JSON line per case on rank 0 (per-step microseconds from the slope between T/4 and T steps).

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

#ifdef BACKEND_SHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#else
#include <nccl.h>
#include <nccl_device.h>
#endif

#define CUDACHECK(cmd) do { cudaError_t e = (cmd); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
#ifndef BACKEND_SHMEM
#define NCCLCHECK(cmd) do { ncclResult_t r = (cmd); if (r != ncclSuccess) { \
  fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD, 3); } } while (0)
#endif

#include "schedule.h"

constexpr int NCOUNT = 2 * MAX_ROUNDS;
constexpr int THREADS = 512;
constexpr unsigned long long TIMEOUT_NS = 5000000000ull;

__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__host__ __device__ __forceinline__ uint32_t pattern(int rank, int step, int i) {
  return (uint32_t)(rank * 2654435761u) ^ (uint32_t)(step * 40503u) ^ (uint32_t)i;
}
__device__ void spin(uint64_t ns) { uint64_t t0 = gtime(); while (gtime() - t0 < ns) {} }
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

struct Args {
  int rank, P, steps, lsa0; size_t slice; uint64_t gamma_ns; int L, node, has_gin;
  uint32_t* recv;              // local [2][P][slice/4], registered / symmetric
  uint32_t* flags;             // local [NCOUNT] same-node arrival counters (registered / symmetric)
  const uint64_t* base;        // [2][NCOUNT]: snapshot of flags (row 0) and network signals (row 1) at launch
  unsigned long long* timing;  // [3]: compute, publish, wait (ns sums)
  int* err;
};

__device__ __forceinline__ void write_own(const Args& a, uint32_t* buf, int step) {
  size_t words = a.slice / 4;
  for (size_t i = threadIdx.x; i < words; i += blockDim.x) buf[(size_t)a.rank * words + i] = pattern(a.rank, step, (int)i);
}

#ifndef BACKEND_SHMEM
// ---- GIN backend -------------------------------------------------------------------------------------------------
__global__ void snapshot_gin(Args a, ncclDevComm dc, uint64_t* out) {
  for (int i = 0; i < NCOUNT; ++i) out[i] = ld_acquire_sys(a.flags + i);
  if (a.has_gin) { ncclGin gin{dc, 0}; for (int i = 0; i < NCOUNT; ++i) out[NCOUNT + i] = gin.readSignal(i); }
  else for (int i = 0; i < NCOUNT; ++i) out[NCOUNT + i] = 0;
}

__global__ void __launch_bounds__(THREADS) step_kernel_gin(Args a, Table tb, ncclWindow_t rwin, ncclWindow_t fwin,
                                                           ncclDevComm dc) {
  __shared__ uint64_t base[2 * NCOUNT];
  if (threadIdx.x < 2 * NCOUNT) base[threadIdx.x] = a.base[threadIdx.x];
  __syncthreads();
  size_t words = a.slice / 4;
  unsigned long long tc = 0, tp = 0, tw = 0;
  for (int step = 0; step < a.steps; ++step) {
    int par = step & 1;
    uint32_t* buf = a.recv + (size_t)par * a.P * words;
    uint64_t t0 = gtime();
    if (threadIdx.x == 0) spin(a.gamma_ns);
    __syncthreads();
    write_own(a, buf, step);
    __syncthreads();
    uint64_t t1 = gtime(), tw0 = t1;
    for (int r = 0; r < tb.nround; ++r) {
      const Round& rd = tb.round[r];
      int ci = par * MAX_ROUNDS + r;
      for (int i = 0; i < rd.nsend; ++i) {
        const Send& s = rd.send[i];
        size_t off = ((size_t)par * a.P + s.block) * a.slice;
        size_t bytes = (size_t)s.nblocks * a.slice;
        if (s.peer / a.L == a.node) {
          uint4* dst = (uint4*)ncclGetLsaPointer(rwin, off, s.peer - a.lsa0);
          const uint4* src = (const uint4*)((const char*)a.recv + off);
          for (size_t w = threadIdx.x; w < bytes / 16; w += blockDim.x) dst[w] = src[w];
          __threadfence_system();
          __syncthreads();
          if (threadIdx.x == 0) {
            uint32_t* f = (uint32_t*)ncclGetLsaPointer(fwin, ci * sizeof(uint32_t), s.peer - a.lsa0);
            atomicAdd_system(f, 1u);
          }
        } else if (threadIdx.x == 0) {
          ncclGin gin{dc, 0};
          gin.put(ncclTeamWorld(dc), s.peer, rwin, off, rwin, off, bytes, ncclGin_WeakSignalInc{(ncclGinSignal_t)ci});
        }
      }
      if (r == 0) { __syncthreads(); tw0 = gtime(); }
      if (threadIdx.x == 0) {
        uint64_t k = (uint64_t)(step / 2 + 1);
        uint32_t want_l = (uint32_t)(base[ci] + k * rd.arrive_local);
        uint64_t want_r = base[NCOUNT + ci] + k * rd.arrive_remote;
        uint64_t ts = gtime();
        while (true) {
          bool ok_l = rd.arrive_local == 0 || (int32_t)(ld_acquire_sys(a.flags + ci) - want_l) >= 0;
          bool ok_r = rd.arrive_remote == 0;
          if (!ok_r) { ncclGin gin{dc, 0}; ok_r = gin.readSignal(ci) >= want_r; }
          if (ok_l && ok_r) break;
          if (gtime() - ts > TIMEOUT_NS) { atomicExch(a.err, 1 + step); break; }
        }
      }
      __syncthreads();
    }
    if (a.has_gin && threadIdx.x == 0) { ncclGin gin{dc, 0}; gin.flush(ncclCoopThread()); }
    __syncthreads();
    uint64_t t2 = gtime();
    tc += t1 - t0; tp += tw0 - t1; tw += t2 - tw0;
  }
  if (threadIdx.x == 0) { a.timing[0] = tc; a.timing[1] = tp; a.timing[2] = tw; }
}

__global__ void spin_kernel(Args a, uint32_t* buf, int step) {
  if (threadIdx.x == 0) spin(a.gamma_ns);
  __syncthreads();
  write_own(a, buf, step);
}
#else
// ---- NVSHMEM backend ---------------------------------------------------------------------------------------------
__global__ void snapshot_shmem(Args a, uint64_t* sig, uint64_t* out) {
  for (int i = 0; i < NCOUNT; ++i) { out[i] = 0; out[NCOUNT + i] = nvshmem_signal_fetch(sig + i); }
}

__global__ void __launch_bounds__(THREADS) step_kernel_shmem(Args a, Table tb, uint64_t* sig) {
  __shared__ uint64_t base[NCOUNT];
  if (threadIdx.x < NCOUNT) base[threadIdx.x] = a.base[NCOUNT + threadIdx.x];
  __syncthreads();
  size_t words = a.slice / 4;
  unsigned long long tc = 0, tp = 0, tw = 0;
  for (int step = 0; step < a.steps; ++step) {
    int par = step & 1;
    uint32_t* buf = a.recv + (size_t)par * a.P * words;
    uint64_t t0 = gtime();
    if (threadIdx.x == 0) spin(a.gamma_ns);
    __syncthreads();
    write_own(a, buf, step);
    __syncthreads();
    uint64_t t1 = gtime(), tw0 = t1;
    for (int r = 0; r < tb.nround; ++r) {
      const Round& rd = tb.round[r];
      int ci = par * MAX_ROUNDS + r;
      for (int i = 0; i < rd.nsend; ++i) {
        const Send& s = rd.send[i];
        uint32_t* p = a.recv + ((size_t)par * a.P + s.block) * words;
        nvshmemx_putmem_signal_nbi_block(p, p, (size_t)s.nblocks * a.slice, sig + ci, 1, NVSHMEM_SIGNAL_ADD, s.peer);
      }
      if (r == 0) { __syncthreads(); tw0 = gtime(); }
      if (threadIdx.x == 0) {
        uint64_t want = base[ci] + (uint64_t)(step / 2 + 1) * (rd.arrive_local + rd.arrive_remote);
        uint64_t ts = gtime();
        while (nvshmem_signal_fetch(sig + ci) < want)
          if (gtime() - ts > TIMEOUT_NS) { atomicExch(a.err, 1 + step); break; }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) nvshmem_quiet();
    __syncthreads();
    uint64_t t2 = gtime();
    tc += t1 - t0; tp += tw0 - t1; tw += t2 - tw0;
  }
  if (threadIdx.x == 0) { a.timing[0] = tc; a.timing[1] = tp; a.timing[2] = tw; }
}
#endif

static const char* arg(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out; size_t a = 0;
  while (a <= s.size()) { size_t b = s.find(',', a); if (b == std::string::npos) b = s.size(); if (b > a) out.push_back(s.substr(a, b - a)); a = b + 1; }
  return out;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int wrank, wsize; MPI_Comm_rank(MPI_COMM_WORLD, &wrank); MPI_Comm_size(MPI_COMM_WORLD, &wsize);
#ifdef BACKEND_SHMEM
  std::string backend = "shmem";
#else
  std::string backend = arg(argc, argv, "--backend", "gin");
#endif
  std::vector<std::string> sched_list = split(arg(argc, argv, "--schedules", "flat"));
  std::vector<size_t> slice_list; for (auto& s : split(arg(argc, argv, "--slice-bytes", "8192"))) slice_list.push_back(std::stoull(s));
  int P = atoi(arg(argc, argv, "--ranks", "0")); if (P <= 0 || P > wsize) P = wsize;
  int stride = atoi(arg(argc, argv, "--stride", "1"));
  int steps = atoi(arg(argc, argv, "--steps", "256"));
  long long gamma_ns = atoll(arg(argc, argv, "--gamma-ns", "3000"));
  int reps = atoi(arg(argc, argv, "--reps", "5"));
  int graph = atoi(arg(argc, argv, "--graph", "0"));
  const char* tag = arg(argc, argv, "--tag", "");
  for (size_t s : slice_list) if (s % 16 || s == 0) { if (!wrank) fprintf(stderr, "slice must be a multiple of 16\n"); MPI_Abort(MPI_COMM_WORLD, 5); }

  MPI_Comm shm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, wrank, MPI_INFO_NULL, &shm);
  int L, lrank; MPI_Comm_size(shm, &L); MPI_Comm_rank(shm, &lrank);
  CUDACHECK(cudaSetDevice(lrank));
  MPI_Comm comm; MPI_Comm_split(MPI_COMM_WORLD, wrank < P ? 0 : 1, wrank, &comm);
  if (wrank >= P) { MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); return 0; }
  Topo t = make_topo(P, L, wrank);

  size_t max_slice = *std::max_element(slice_list.begin(), slice_list.end());
  size_t buf_bytes = 2 * (size_t)P * max_slice;
  uint32_t* recv = nullptr; uint32_t* flags = nullptr;
  unsigned long long* timing; int* err; uint64_t* base;
  CUDACHECK(cudaMalloc(&timing, 3 * sizeof(unsigned long long)));
  CUDACHECK(cudaMalloc(&err, sizeof(int)));
  CUDACHECK(cudaMalloc(&base, 2 * NCOUNT * sizeof(uint64_t)));
  int lsa0 = wrank - lrank, has_gin = 0;
#ifdef BACKEND_SHMEM
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER; attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  recv = (uint32_t*)nvshmem_malloc(buf_bytes);
  uint64_t* sig = (uint64_t*)nvshmem_calloc(NCOUNT, sizeof(uint64_t));
  CUDACHECK(cudaMalloc(&flags, NCOUNT * sizeof(uint32_t)));
  if (wrank == 0) printf("{\"event\":\"init\",\"backend\":\"shmem\",\"P\":%d,\"L\":%d,\"nodes\":%d}\n", P, L, t.nodes);
#else
  ncclUniqueId id; if (wrank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
  ncclComm_t nc; NCCLCHECK(ncclCommInitRank(&nc, P, id, wrank));
  NCCLCHECK(ncclMemAlloc((void**)&recv, buf_bytes));
  NCCLCHECK(ncclMemAlloc((void**)&flags, 4096));
  CUDACHECK(cudaMemset(flags, 0, 4096));
  ncclWindow_t rwin = nullptr, fwin = nullptr;
  ncclDevComm dc{};
  if (backend == "gin") {
    NCCLCHECK(ncclCommWindowRegister(nc, recv, buf_bytes, &rwin, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(nc, flags, 4096, &fwin, NCCL_WIN_COLL_SYMMETRIC));
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCLCHECK(ncclCommQueryProperties(nc, &props));
    ncclTeam_t lsa = ncclTeamLsa(nc);
    lsa0 = wrank - lsa.rank;
    has_gin = t.nodes > 1;
    if (wrank == 0)
      printf("{\"event\":\"init\",\"backend\":\"gin\",\"P\":%d,\"L\":%d,\"nodes\":%d,\"deviceApiSupport\":%d,\"ginType\":%d,"
             "\"lsaSize\":%d,\"lsaRank\":%d}\n", P, L, t.nodes, (int)props.deviceApiSupport, (int)props.ginType, lsa.nRanks, lsa.rank);
    fflush(stdout);
    if (lsa.nRanks != L) { fprintf(stderr, "rank %d: LSA team %d != node size %d\n", wrank, lsa.nRanks, L); MPI_Abort(MPI_COMM_WORLD, 6); }
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (has_gin) {
      reqs.ginContextCount = 1; reqs.ginSignalCount = NCOUNT;
      reqs.ginStrongSignalsRequired = false; reqs.ginVaSignalsRequired = false;
      reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    } else {
      reqs.ginContextCount = 0;
    }
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclDevCommCreate(nc, &reqs, &dc));
    NCCLCHECK(ncclGroupEnd());
  }
#endif
  CUDACHECK(cudaMemset(recv, 0, buf_bytes));
  CUDACHECK(cudaDeviceSynchronize());
  MPI_Barrier(comm);
  cudaStream_t st; CUDACHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaEvent_t e0, e1; CUDACHECK(cudaEventCreate(&e0)); CUDACHECK(cudaEventCreate(&e1));

  for (size_t slice : slice_list) {
    for (const std::string& sched : sched_list) {
      Table tb = build(sched, stride, t, wrank);
      if (tb.nround < 0) {
        if (wrank == 0) printf("{\"event\":\"skip\",\"schedule\":\"%s\",\"stride\":%d,\"P\":%d,\"L\":%d}\n", sched.c_str(), stride, P, L);
        continue;
      }
      bool complete = schedule_complete(sched, stride, t);
      arrivals(sched, stride, t, tb);
      std::vector<double> per_step, ph_c, ph_p, ph_w;
      int bad = 0;
      int T0 = std::max(4, steps / 4);
      for (int rep = 0; rep < reps; ++rep) {
        double ms[2];
        for (int pass = 0; pass < 2; ++pass) {
          int T = pass == 0 ? T0 : steps;
          Args a{wrank, P, T, lsa0, slice, (uint64_t)gamma_ns, L, t.node, has_gin, recv, flags, base, timing, err};
          CUDACHECK(cudaMemset(err, 0, sizeof(int)));
          CUDACHECK(cudaDeviceSynchronize());
          MPI_Barrier(comm);   // every arrival of the previous launch has landed everywhere
#ifdef BACKEND_SHMEM
          snapshot_shmem<<<1, 1, 0, st>>>(a, sig, base);
#else
          if (backend == "gin") snapshot_gin<<<1, 1, 0, st>>>(a, dc, base);
#endif
          CUDACHECK(cudaStreamSynchronize(st));
          MPI_Barrier(comm);   // no rank sends this launch's data before every rank has its snapshot
          float m = 0;
#ifdef BACKEND_SHMEM
          CUDACHECK(cudaEventRecord(e0, st));
          step_kernel_shmem<<<1, THREADS, 0, st>>>(a, tb, sig);
          CUDACHECK(cudaEventRecord(e1, st));
#else
          if (backend == "gin") {
            CUDACHECK(cudaEventRecord(e0, st));
            step_kernel_gin<<<1, THREADS, 0, st>>>(a, tb, rwin, fwin, dc);
            CUDACHECK(cudaEventRecord(e1, st));
          } else {  // host: spin kernel + ncclAllGather per step
            size_t words = slice / 4;
            auto body = [&](int s) {
              uint32_t* buf = recv + (size_t)(s & 1) * P * words;
              spin_kernel<<<1, THREADS, 0, st>>>(a, buf, s);
              NCCLCHECK(ncclAllGather(buf + (size_t)wrank * words, buf, slice, ncclUint8, nc, st));
            };
            if (graph) {
              cudaGraph_t g; cudaGraphExec_t ge;
              CUDACHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
              for (int s = 0; s < T; ++s) body(s);
              CUDACHECK(cudaStreamEndCapture(st, &g));
              CUDACHECK(cudaGraphInstantiate(&ge, g, 0));
              CUDACHECK(cudaGraphUpload(ge, st));
              CUDACHECK(cudaStreamSynchronize(st));
              MPI_Barrier(comm);
              CUDACHECK(cudaEventRecord(e0, st));
              CUDACHECK(cudaGraphLaunch(ge, st));
              CUDACHECK(cudaEventRecord(e1, st));
              CUDACHECK(cudaStreamSynchronize(st));
              cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
            } else {
              CUDACHECK(cudaEventRecord(e0, st));
              for (int s = 0; s < T; ++s) body(s);
              CUDACHECK(cudaEventRecord(e1, st));
            }
          }
#endif
          CUDACHECK(cudaStreamSynchronize(st));
          CUDACHECK(cudaEventElapsedTime(&m, e0, e1));
          int e = 0; CUDACHECK(cudaMemcpy(&e, err, sizeof(int), cudaMemcpyDeviceToHost)); if (e) bad = e;
          double md = m, mx; MPI_Allreduce(&md, &mx, 1, MPI_DOUBLE, MPI_MAX, comm);
          ms[pass] = mx;
          if (pass == 1 && backend != "host") {
            unsigned long long h[3]; CUDACHECK(cudaMemcpy(h, timing, sizeof(h), cudaMemcpyDeviceToHost));
            double v[3] = {h[0] / 1e3 / T, h[1] / 1e3 / T, h[2] / 1e3 / T}, s[3];
            MPI_Allreduce(v, s, 3, MPI_DOUBLE, MPI_SUM, comm);
            ph_c.push_back(s[0] / P); ph_p.push_back(s[1] / P); ph_w.push_back(s[2] / P);
          }
        }
        per_step.push_back((ms[1] - ms[0]) * 1000.0 / (steps - T0));
      }
      // Verification: the buffer of the last step holds every rank's pattern.
      size_t words = slice / 4; int last = steps - 1;
      std::vector<uint32_t> hb((size_t)P * words);
      CUDACHECK(cudaMemcpy(hb.data(), recv + (size_t)(last & 1) * P * words, hb.size() * 4, cudaMemcpyDeviceToHost));
      long wrong = 0;
      for (int b = 0; b < P; ++b) for (size_t i = 0; i < words; ++i) wrong += hb[(size_t)b * words + i] != pattern(b, last, (int)i);
      long wrong_all; MPI_Allreduce(&wrong, &wrong_all, 1, MPI_LONG, MPI_SUM, comm);
      int bad_all; MPI_Allreduce(&bad, &bad_all, 1, MPI_INT, MPI_MAX, comm);
      if (wrank == 0) {
        auto med = [](std::vector<double> v) { if (v.empty()) return -1.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        int al = 0, ar = 0; for (int r = 0; r < tb.nround; ++r) { al += tb.round[r].arrive_local; ar += tb.round[r].arrive_remote; }
        printf("{\"event\":\"case\",\"tag\":\"%s\",\"backend\":\"%s\",\"graph\":%d,\"schedule\":\"%s\",\"stride\":%d,\"P\":%d,"
               "\"L\":%d,\"nodes\":%d,\"slice_bytes\":%zu,\"steps\":%d,\"gamma_ns\":%lld,\"rounds\":%d,\"rank0_arrivals_local\":%d,"
               "\"rank0_arrivals_remote\":%d,\"schedule_complete\":%s,\"per_step_us\":%.4f,\"per_step_us_all\":[",
               tag, backend.c_str(), graph, sched.c_str(), stride, P, L, t.nodes, slice, steps, gamma_ns, tb.nround, al, ar,
               complete ? "true" : "false", med(per_step));
        for (size_t i = 0; i < per_step.size(); ++i) printf("%s%.4f", i ? "," : "", per_step[i]);
        printf("],\"phase_compute_us\":%.4f,\"phase_publish_us\":%.4f,\"phase_wait_us\":%.4f,\"wrong_words\":%ld,\"timeout_step\":%d}\n",
               med(ph_c), med(ph_p), med(ph_w), wrong_all, bad_all);
        fflush(stdout);
      }
      MPI_Barrier(comm);
    }
  }
#ifdef BACKEND_SHMEM
  nvshmem_free(recv); nvshmem_free(sig); nvshmem_finalize();
#else
  if (backend == "gin") {
    NCCLCHECK(ncclDevCommDestroy(nc, &dc));
    NCCLCHECK(ncclCommWindowDeregister(nc, rwin)); NCCLCHECK(ncclCommWindowDeregister(nc, fwin));
  }
  NCCLCHECK(ncclMemFree(recv)); NCCLCHECK(ncclMemFree(flags)); NCCLCHECK(ncclCommDestroy(nc));
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
