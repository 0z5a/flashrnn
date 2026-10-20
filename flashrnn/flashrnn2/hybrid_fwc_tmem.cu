// BETA, UNVERIFIED (docs/flashrnn2/max-width.md section 81): the cluster forward of hybrid_fwc.cu with the GEMM on
// Blackwell's fifth-generation tensor cores (tcgen05.mma, accumulator in tensor memory). Target sm_100a / sm_103a.
// Written without access to a Blackwell GPU; see the handoff checklist in docs/flashrnn2/tmem-beta-handoff.md.
//
// Differences to hybrid_fwc.cu:
// * Operands come from shared memory in the canonical K-major no-swizzle layout: element (m, k) of a [M x K] matrix
//   lives at (m / 8) * SBO + (k / 8) * LBO + (m % 8) * 16 + (k % 8) * 2 bytes, with LBO = 128 (the two 8-column core
//   matrices of a K = 16 slab adjacent) and SBO = K * 16 (all core matrices of an 8-row group contiguous). R (the B
//   operand, [N = NC x K = D]) is loaded into this layout once; round(h) (the A operand, [M x K = D]) is written into it
//   by the pointwise owners. The whole R block must therefore fit shared memory: D = 512 with C = 16 (128 KB), not
//   D = 768.
// * M = 64 is the smallest M of tcgen05.mma with cta_group::1; only rows 0..15 carry data. The A buffer holds 16 rows
//   (two 8-row core groups); the descriptor's row-group stride points the other six groups past the buffer into the
//   following shared memory (R), which is valid memory whose products land in accumulator lanes 16..63 that are never
//   read. The two parity buffers are therefore placed before R.
// * The accumulator [64 lanes x NC columns] FP32 is allocated in tensor memory once (tcgen05.alloc by one warp,
//   NC columns, a power of two >= 32). Per step one thread issues the K / 16 MMAs (the first with enable_input_d = 0),
//   then tcgen05.commit arrives on an mbarrier that all threads wait on; warp 0's lanes 0..15 read their row with
//   tcgen05.ld 32x32b (lane i = row i, x NC columns in chunks of 32) and spill it to the shared partials, after which
//   the pointwise update runs on all threads as in hybrid_fwc.cu.
// * Instruction descriptor (kind::f16): D format FP32, A/B format FP16 or BF16, both K-major, N and M encoded as
//   N >> 3 and M >> 4. Bit positions follow the PTX ISA 8.7 table and are UNVERIFIED here.
//
// Build flags: FRNN_D, FRNN_C, FRNN_SLSTM, FRNN_BF16, FRNN_MODE, FRNN_SNAP32, FRNN_VERIFY (bit 16: phase timers), and
// -arch=sm_100a (or sm_103a).
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace cg = cooperative_groups;

#if FRNN_BF16
typedef __nv_bfloat16 elem_t;
__device__ __forceinline__ elem_t to_elem(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float from_elem(elem_t x) { return __bfloat162float(x); }
constexpr unsigned AB_FORMAT = 1;  // kind::f16: 0 F16, 1 BF16
#else
typedef __half elem_t;
__device__ __forceinline__ elem_t to_elem(float x) { return __float2half_rn(x); }
__device__ __forceinline__ float from_elem(elem_t x) { return __half2float(x); }
constexpr unsigned AB_FORMAT = 0;
#endif

#ifndef FRNN_VERIFY
#define FRNN_VERIFY 0
#endif
#ifndef FRNN_MODE
#define FRNN_MODE 1
#endif
#ifndef FRNN_SNAP32
#define FRNN_SNAP32 0
#endif
#if FRNN_SNAP32
typedef float snap_t;
__device__ __forceinline__ snap_t to_snap(float x) { return x; }
#else
typedef elem_t snap_t;
__device__ __forceinline__ snap_t to_snap(float x) { return to_elem(x); }
#endif
constexpr int VERIFY = FRNN_VERIFY, MODE = FRNN_MODE, V_PROF = 16;
constexpr int D = FRNN_D, C = FRNN_C;
constexpr bool SLSTM = FRNN_SLSTM != 0;
constexpr int S = SLSTM ? 4 : 2;
constexpr int THREADS = 256, BM = 16, MMA_M = 64, MMA_K = 16;
constexpr int NU = D / C, NC = 4 * NU, KS = D / MMA_K;
constexpr int PE = (BM * NU + THREADS - 1) / THREADS;
// Canonical K-major no-swizzle layout: LBO between the two k-cores of a K = 16 slab, SBO between 8-row groups.
constexpr int LBO = 128, SBO = D * 16;
constexpr int SMEM_H = BM * D * 2;       // one parity buffer: 16 rows (two row groups) of D elements
constexpr int SMEM_R = NC * D * 2;       // R block: NC rows (N) of D elements (K)
constexpr int SMEM_P = BM * NC * 4;      // gate preactivations [BM][NC] FP32
constexpr int TMEM_COLS = NC >= 32 ? NC : 32;
static_assert(D % C == 0 && NU % 16 == 0, "NU must be a multiple of 16");
static_assert(NC % 16 == 0 && NC <= 256, "tcgen05.mma N must be a multiple of 16 and at most 256");
static_assert((TMEM_COLS & (TMEM_COLS - 1)) == 0, "tensor-memory columns must be a power of two");
static_assert(2 * SMEM_H + SMEM_R + SMEM_P <= 227 * 1024, "R and the buffers must fit shared memory");
static_assert(MODE >= 0 && MODE <= 2, "output mode");

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

// Byte offset of element (m, k) in the canonical K-major no-swizzle layout.
__device__ __forceinline__ unsigned canon(int m, int k) {
  return (unsigned)((m >> 3) * SBO + (k >> 3) * LBO + (m & 7) * 16 + (k & 7) * 2);
}

// Shared-memory matrix descriptor (64 bits): [0,14) start >> 4, [16,30) LBO >> 4, [32,46) SBO >> 4, [49,52) base
// offset 0, [61,64) layout type 0 = SWIZZLE_NONE. UNVERIFIED bit layout; see the handoff.
__device__ __forceinline__ unsigned long long smem_desc(unsigned addr) {
  unsigned long long d = 0;
  d |= (unsigned long long)((addr >> 4) & 0x3FFF);
  d |= (unsigned long long)((LBO >> 4) & 0x3FFF) << 16;
  d |= (unsigned long long)((SBO >> 4) & 0x3FFF) << 32;
  return d;
}

// Instruction descriptor (32 bits) for kind::f16: [4,6) D format (1 = F32), [7,10) A format, [10,13) B format,
// bit 15 A major (0 = K), bit 16 B major (0 = K), [17,23) N >> 3, [24,29) M >> 4. UNVERIFIED bit layout.
__device__ __forceinline__ unsigned inst_desc() {
  unsigned d = 0;
  d |= 1u << 4;
  d |= AB_FORMAT << 7;
  d |= AB_FORMAT << 10;
  d |= (unsigned)(NC >> 3) << 17;
  d |= (unsigned)(MMA_M >> 4) << 24;
  return d;
}

__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

struct Args {
  const elem_t* wx;      // [B, T, 4, H, D]
  const elem_t* R;       // [4, H, D, D]
  const elem_t* bias;    // [4, H, D]
  const float* initial;  // [S, B, 1, H, D] FP32
  snap_t* out;
  float* final;
  int* err;
  const bool* zero_first;
  int B, T, H, tiles;
};

__global__ void __launch_bounds__(THREADS, 1) frnn_hybrid_fwc_tmem_kernel(Args a) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* hbuf = smem;                                   // [2][SMEM_H] canonical A operands by parity
  unsigned char* rbuf = smem + 2 * SMEM_H;                      // canonical B operand (R block)
  float* spart = reinterpret_cast<float*>(smem + 2 * SMEM_H + SMEM_R);  // [BM][NC]
  __shared__ __align__(8) unsigned long long mma_bar;
  __shared__ unsigned tmem_base;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cb = (int)cluster.block_rank();
  const int g = blockIdx.y, head = g / a.tiles, bt = g % a.tiles;
  const bool zero_first = SLSTM && a.zero_first != nullptr && *a.zero_first;
  const size_t hd = (size_t)a.H * D, state_stride = (size_t)a.B * hd, out_stride = (size_t)a.B * a.T * hd;

  // Tensor memory: NC columns for the [64 x NC] FP32 accumulator (one warp allocates; the address lands in shared).
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(smem_u32(&tmem_base)), "r"((unsigned)TMEM_COLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (tid == 0) {
    mbar_init(&mma_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // R block into the canonical layout: row n = gate g * NU + unit u of this CTA, column k.
  for (int i = tid; i < NC * D; i += THREADS) {
    const int n = i / D, k = i % D, gq = n / NU, o = cb * NU + n % NU;
    *reinterpret_cast<elem_t*>(rbuf + canon(n, k)) = a.R[(((size_t)gq * a.H + head) * D + o) * D + k];
  }
  // Both parity buffers zero (rows beyond B and the padding stay zero).
  for (int i = tid; i < 2 * SMEM_H / 4; i += THREADS) reinterpret_cast<unsigned*>(hbuf)[i] = 0u;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic writes of R before the async-proxy MMA reads

  auto elem = [&](int e, int& lrow, int& u) -> bool {
    const int idx = tid + THREADS * e;
    lrow = idx / NU;
    u = idx % NU;
    return idx < BM * NU;
  };
  float hs[PE], cs[PE], ns[PE], ms[PE], bias[PE][4];
  elem_t wxr[PE][4];
  unsigned char* peer[C];
#pragma unroll
  for (int r = 0; r < C; ++r) peer[r] = cluster.map_shared_rank(hbuf, r);
#pragma unroll
  for (int e = 0; e < PE; ++e) {
    int lrow, u;
    const bool own = elem(e, lrow, u);
    const int row = bt * BM + lrow, o = cb * NU + u;
    const bool valid = own && row < a.B;
    const size_t unit = ((size_t)row * a.H + head) * D + o;
    hs[e] = valid ? a.initial[unit] : 0.f;
    cs[e] = valid ? a.initial[state_stride + unit] : 0.f;
    ns[e] = (SLSTM && valid) ? a.initial[2 * state_stride + unit] : 0.f;
    ms[e] = (SLSTM && valid) ? a.initial[3 * state_stride + unit] : 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) bias[e][q] = own ? from_elem(a.bias[((size_t)q * a.H + head) * D + o]) : 0.f;
    if (own) {
      const elem_t v = to_elem(hs[e]);
#pragma unroll
      for (int r = 0; r < C; ++r) *reinterpret_cast<elem_t*>(peer[r] + canon(lrow, o)) = v;
    }
  }

  unsigned mma_phase = 0;
  for (int t = 0; t < a.T; ++t) {
    const bool last = t == a.T - 1;
    unsigned long long t0 = (VERIFY & V_PROF) ? gtime() : 0;
    cluster_arrive();
#pragma unroll
    for (int e = 0; e < PE; ++e) {
      int lrow, u;
      const bool own = elem(e, lrow, u);
      const int row = bt * BM + lrow;
      if (own && row < a.B) {
        const size_t xo = (((size_t)row * a.T + t) * 4 * a.H + head) * D + cb * NU + u;
#pragma unroll
        for (int q = 0; q < 4; ++q) wxr[e][q] = a.wx[xo + (size_t)q * hd];
      }
    }
    cluster_wait();
    // The peers' generic-proxy stores into this CTA's A buffer precede the async-proxy MMA reads.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long t1 = (VERIFY & V_PROF) ? gtime() : 0;

    // GEMM: one thread issues KS MMAs into the tensor-memory accumulator, then commits to the mbarrier.
    if (tid == 0) {
      const unsigned abase = smem_u32(hbuf + (t & 1) * SMEM_H), bbase = smem_u32(rbuf), idesc = inst_desc();
      for (int c = 0; c < KS; ++c) {
        const unsigned long long adesc = smem_desc(abase + c * 2 * LBO), bdesc = smem_desc(bbase + c * 2 * LBO);
        const unsigned accumulate = c > 0 ? 1u : 0u;
        asm volatile(
            "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
            " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            ::"r"(tmem_base), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate) : "memory");
      }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                   ::"r"(smem_u32(&mma_bar)) : "memory");
    }
    mbar_wait(&mma_bar, mma_phase);
    mma_phase ^= 1;
    // Warp 0 reads lanes 0..31 (rows; only 0..15 are data): 32 columns per load, NC / 32 loads, into spart.
    if (warp == 0) {
#pragma unroll
      for (int c0 = 0; c0 < NC; c0 += 32) {
        unsigned v[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,"
            "%28,%29,%30,%31}, [%32];"
            : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
              "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
              "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
              "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
            : "r"(tmem_base + (unsigned)c0));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (lane < BM) {
#pragma unroll
          for (int j = 0; j < 32; ++j) spart[(size_t)lane * NC + c0 + j] = __uint_as_float(v[j]);
        }
      }
    }
    __syncthreads();
    unsigned long long t2 = (VERIFY & V_PROF) ? gtime() : 0;

    const int nb = (t + 1) & 1;
#pragma unroll
    for (int e = 0; e < PE; ++e) {
      int lrow, u;
      const bool own = elem(e, lrow, u);
      if (!own) continue;
      const int row = bt * BM + lrow, o = cb * NU + u;
      float h = 0.f;
      if (row < a.B) {
        float gate[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) gate[q] = spart[(size_t)lrow * NC + (size_t)q * NU + u] + from_elem(wxr[e][q]) + bias[e][q];
        const float gi = gate[0], gf = gate[1], gz = gate[2], go = gate[3];
        const float z = 2.0f * sigmoid(2.0f * gz) - 1.0f;
        if (SLSTM) {
          const float logf_ = fminf(gf, 0.0f) - logf(1.0f + expf(-fabsf(gf))) + ms[e];
          const float m_new = (t == 0 && zero_first) ? gi : fmaxf(gi, logf_);
          const float igate = expf(gi - m_new), fgate = expf(logf_ - m_new);
          cs[e] = fgate * cs[e] + igate * z;
          ns[e] = fmaxf(fgate * ns[e] + igate, 1.0f);
          h = sigmoid(go) * cs[e] / ns[e];
          ms[e] = m_new;
        } else {
          cs[e] = sigmoid(gf) * cs[e] + sigmoid(gi) * z;
          h = sigmoid(go) * (2.0f * sigmoid(2.0f * cs[e]) - 1.0f);
        }
        hs[e] = h;
        const size_t here = (((size_t)row * a.T + t) * a.H + head) * D + o;
        if (MODE == 0) {
          a.out[here] = to_snap(h);
          a.out[out_stride + here] = to_snap(cs[e]);
          if (SLSTM) {
            a.out[2 * out_stride + here] = to_snap(ns[e]);
            a.out[3 * out_stride + here] = to_snap(ms[e]);
          }
        } else if (MODE == 1) {
          a.out[here] = to_snap(h);
        } else if (last) {
          a.out[((size_t)row * a.H + head) * D + o] = to_snap(h);
        }
      }
      if (!last) {
        const elem_t v = to_elem(h);
#pragma unroll
        for (int r = 0; r < C; ++r) *reinterpret_cast<elem_t*>(peer[r] + nb * SMEM_H + canon(lrow, o)) = v;
      }
    }
    if ((VERIFY & V_PROF) && tid == 0) {
      const unsigned long long t3 = gtime();
      atomicAdd(a.err + 4, (int)((t1 - t0) >> 6));
      atomicAdd(a.err + 5, (int)((t2 - t1) >> 6));
      atomicAdd(a.err + 6, (int)((t3 - t2) >> 6));
      atomicAdd(a.err + 9, 1);
    }
  }
#pragma unroll
  for (int e = 0; e < PE; ++e) {
    int lrow, u;
    const bool own = elem(e, lrow, u);
    const int row = bt * BM + lrow;
    const size_t unit = ((size_t)row * a.H + head) * D + cb * NU + u;
    if (own && row < a.B) {
      a.final[unit] = hs[e];
      a.final[state_stride + unit] = cs[e];
      if (SLSTM) {
        a.final[2 * state_stride + unit] = ns[e];
        a.final[3 * state_stride + unit] = ms[e];
      }
    }
  }
  cluster.sync();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"((unsigned)TMEM_COLS)
                 : "memory");
  }
}

extern "C" int frnn_hybrid_fwc_tmem_smem() { return 2 * SMEM_H + SMEM_R + SMEM_P; }

extern "C" int frnn_hybrid_fwc_tmem(const void* wx, const void* R, const void* bias, const float* initial, void* out,
                                    float* final, int* err, const void* zero_first, int B, int T, int H, int groups,
                                    int tiles, void* stream) {
  Args a{reinterpret_cast<const elem_t*>(wx), reinterpret_cast<const elem_t*>(R),
         reinterpret_cast<const elem_t*>(bias), initial, reinterpret_cast<snap_t*>(out), final, err,
         reinterpret_cast<const bool*>(zero_first), B, T, H, tiles};
  const int smem = frnn_hybrid_fwc_tmem_smem();
  cudaError_t e = cudaFuncSetAttribute(frnn_hybrid_fwc_tmem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return (int)e;
  if (C > 8) {
    e = cudaFuncSetAttribute(frnn_hybrid_fwc_tmem_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return (int)e;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C, groups);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = C;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  e = cudaLaunchKernelEx(&cfg, frnn_hybrid_fwc_tmem_kernel, a);
  return (int)e;
}
