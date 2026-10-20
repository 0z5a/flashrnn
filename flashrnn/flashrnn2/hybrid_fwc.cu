// Forward for small batches (B <= 16) and D <= 768 in one thread-block cluster (docs/flashrnn2/max-width.md section
// 80): no exchange through L2, one cluster barrier per step.
//
// The C CTAs of a cluster split the output units: CTA c keeps R[g, o, k] resident (register rows + shared memory) for
// its NU = D / C units o, all four gates g (NC = 4 NU output columns) and the whole hidden dimension k. The hidden state
// round(h_{t-1}) [BM = 16 rows x D] lives in every CTA's shared memory, double buffered by step parity: the pointwise
// owner of a unit writes round(h_t) straight into the next-parity buffer of every CTA of the cluster (distributed shared
// memory), and one cluster barrier per step (release arrival, acquire wait) orders those writes before the next
// step's GEMM. Per step:
//   arrive -> (this step's Wx loads) -> wait -> GEMM (local h x resident R -> gate preactivations into shared memory)
//   -> __syncthreads -> pointwise update (Wx, hoisted bias; cell state in registers) -> round(h_t) into every peer.
// Double buffering is race-free with one barrier: a CTA writes buffer (t + 1) % 2 after it passed barrier t, which
// every peer reaches only after its GEMM of step t - 1, the last reader of that buffer.
//
// The GEMM runs KSPLIT interleaved accumulators per n-tile (k-step c into acc[c % KSPLIT]) to shorten the dependent
// mma.sync chain, summed at the end in fixed order; the result therefore differs from a single-chain accumulation in
// the order of FP32 additions only.
//
// Exchange modes: push (default) -- the owner stores round(h_t) into every CTA's buffer, the GEMM reads local shared
// memory; FRNN_FWC_PULL = 1 -- symmetric buffers: the owner stores only into its own buffer, and the GEMM's A fragments
// for hidden indices of unit block c are read from CTA c's shared memory through distributed shared memory.
//
// Build flags: FRNN_D, FRNN_C (cluster size; 16 is non-portable and needs the launch attribute), FRNN_CR (k-steps of
// R in registers per warp), FRNN_KSPLIT, FRNN_SLSTM, FRNN_BF16, FRNN_MODE (0 full / 1 hidden / 2 last), FRNN_SNAP32,
// FRNN_VERIFY (bit 16: phase timers; globaltimer, thread 0, units of 64 ns: err[4] barrier, err[5] GEMM, err[6]
// pointwise + exchange, err[9] steps).
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace cg = cooperative_groups;

#if FRNN_BF16
typedef __nv_bfloat16 elem_t;
#define FRNN_MMA "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
__device__ __forceinline__ elem_t to_elem(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float from_elem(elem_t x) { return __bfloat162float(x); }
#else
typedef __half elem_t;
#define FRNN_MMA "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
__device__ __forceinline__ elem_t to_elem(float x) { return __float2half_rn(x); }
__device__ __forceinline__ float from_elem(elem_t x) { return __half2float(x); }
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
#ifndef FRNN_KSPLIT
#define FRNN_KSPLIT 2
#endif
#ifndef FRNN_FWC_PULL
#define FRNN_FWC_PULL 0
#endif
constexpr bool PULL = FRNN_FWC_PULL != 0;
#if FRNN_SNAP32
typedef float snap_t;
__device__ __forceinline__ snap_t to_snap(float x) { return x; }
#else
typedef elem_t snap_t;
__device__ __forceinline__ snap_t to_snap(float x) { return to_elem(x); }
#endif
constexpr int VERIFY = FRNN_VERIFY, MODE = FRNN_MODE, V_PROF = 16;
constexpr int D = FRNN_D, C = FRNN_C, CR = FRNN_CR, KSPLIT = FRNN_KSPLIT;
constexpr bool SLSTM = FRNN_SLSTM != 0;
constexpr int S = SLSTM ? 4 : 2;
constexpr int THREADS = 256, WARPS = 8, BM = 16;
constexpr int NU = D / C, NC = 4 * NU, NCW = NC / WARPS, NTW = NCW / 8, KS = D / 16, CSM = KS - CR;
constexpr int LDH = D + 8;  // padded row stride of the hidden-state buffers (elements)
constexpr int PE = (BM * NU + THREADS - 1) / THREADS;
constexpr int SMEM_R = WARPS * CSM * NTW * 32 * 8, SMEM_H = 2 * BM * LDH * 2, SMEM_P = BM * NC * 4;
static_assert(D % C == 0 && NU % 16 == 0, "a warp's columns must lie within one gate (NU a multiple of 16)");
static_assert(NU % NCW == 0, "a warp's columns must not cross a gate");
static_assert(!PULL || NU % 16 == 0, "pull mode reads one owner per 16-wide k-step");
static_assert(CR >= 0 && CR <= KS, "register k-steps");
static_assert(KSPLIT >= 1 && KSPLIT <= 4, "accumulator split");
static_assert(SMEM_R + SMEM_H + SMEM_P <= 227 * 1024, "R, the two h buffers and the partials must fit shared memory");
static_assert(MODE >= 0 && MODE <= 2, "output mode");

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint32_t pack(elem_t lo, elem_t hi) {
  uint16_t a = *reinterpret_cast<uint16_t*>(&lo), b = *reinterpret_cast<uint16_t*>(&hi);
  return (uint32_t)a | ((uint32_t)b << 16);
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile(FRNN_MMA " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// B fragment register for k (even) and k + 1, output column (g, o): R[g, head, o, k .. k + 1] (contiguous).
__device__ __forceinline__ uint32_t r_pair(const elem_t* R, int H, int head, int g, int o, int k) {
  const size_t base = (((size_t)g * H + head) * D + o) * D + k;
  return pack(R[base], R[base + 1]);
}

__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

struct Args {
  const elem_t* wx;      // [B, T, 4, H, D]
  const elem_t* R;       // [4, H, D, D]
  const elem_t* bias;    // [4, H, D]
  const float* initial;  // [S, B, 1, H, D] FP32
  snap_t* out;           // full [S, B, T, H, D]; hidden [B, T, H, D]; last [B, 1, H, D]
  float* final;          // [S, B, 1, H, D] FP32 carry
  int* err;              // [12] phase timers
  const bool* zero_first;  // sLSTM: normalizer zero at the call's first step (device flag; nullptr: false)
  int B, T, H, tiles;
};

__global__ void __launch_bounds__(THREADS, 1) frnn_hybrid_fwc_kernel(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint2* sr = reinterpret_cast<uint2*>(smem);                              // [WARPS][CSM][NTW][32] B fragments
  elem_t* hbuf = reinterpret_cast<elem_t*>(smem + SMEM_R);                 // [2][BM][LDH] round(h), by step parity
  float* spart = reinterpret_cast<float*>(smem + SMEM_R + SMEM_H);         // [BM][NC] gate preactivations
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, fg = lane >> 2, ft = lane & 3;
  const int cb = (int)cluster.block_rank();  // unit block of this CTA
  const int g = blockIdx.y;                  // group: (head, batch tile)
  const int head = g / a.tiles, bt = g % a.tiles;
  const bool zero_first = SLSTM && a.zero_first != nullptr && *a.zero_first;
  const int c_warp = warp * NCW, g_warp = c_warp / NU, o_warp = cb * NU + c_warp % NU;
  const size_t hd = (size_t)a.H * D, state_stride = (size_t)a.B * hd, out_stride = (size_t)a.B * a.T * hd;

  // Resident R: the first CR k-steps of every warp in registers, the rest in shared memory (fragment order).
  uint32_t rreg[CR > 0 ? CR : 1][NTW][2];
#pragma unroll
  for (int c = 0; c < CR; ++c)
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt) {
      const int k = c * 16 + 2 * ft, o = o_warp + nt * 8 + fg;
      rreg[c][nt][0] = r_pair(a.R, a.H, head, g_warp, o, k);
      rreg[c][nt][1] = r_pair(a.R, a.H, head, g_warp, o, k + 8);
    }
  for (int c = 0; c < CSM; ++c)
    for (int nt = 0; nt < NTW; ++nt) {
      const int k = (CR + c) * 16 + 2 * ft, o = o_warp + nt * 8 + fg;
      sr[(((size_t)warp * CSM + c) * NTW + nt) * 32 + lane] =
          make_uint2(r_pair(a.R, a.H, head, g_warp, o, k), r_pair(a.R, a.H, head, g_warp, o, k + 8));
    }

  // Pointwise ownership: element e of this thread is (row idx / NU, unit cb * NU + idx % NU), fixed for the call.
  auto elem = [&](int e, int& lrow, int& u) -> bool {
    const int idx = tid + THREADS * e;
    lrow = idx / NU;
    u = idx % NU;
    return idx < BM * NU;
  };
  float hs[PE], cs[PE], ns[PE], ms[PE], bias[PE][4];
  elem_t wxr[PE][4];
  // Peer buffers: round(h) rows of this CTA's units go to every CTA of the cluster (own rank included).
  elem_t* peer[C];
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
    if (own) {  // round(h_0) into buffer 0 (of every peer, or only the own buffer in pull mode); rows >= B: zeros
      const elem_t v = to_elem(hs[e]);
      if (PULL) {
        hbuf[(size_t)lrow * LDH + o] = v;
      } else {
#pragma unroll
        for (int r = 0; r < C; ++r) peer[r][(size_t)lrow * LDH + o] = v;
      }
    }
  }

  for (int t = 0; t < a.T; ++t) {
    const bool last = t == a.T - 1;
    unsigned long long t0 = (VERIFY & V_PROF) ? gtime() : 0;
    cluster_arrive();  // my round(h_{t-1}) stores (and the prologue's h_0) are released to the cluster
#pragma unroll
    for (int e = 0; e < PE; ++e) {  // this step's input projections, loaded while the cluster arrives
      int lrow, u;
      const bool own = elem(e, lrow, u);
      const int row = bt * BM + lrow;
      if (own && row < a.B) {
        const size_t xo = (((size_t)row * a.T + t) * 4 * a.H + head) * D + cb * NU + u;
#pragma unroll
        for (int q = 0; q < 4; ++q) wxr[e][q] = a.wx[xo + (size_t)q * hd];
      }
    }
    cluster_wait();  // every peer's round(h_{t-1}) is in hbuf[t % 2]
    unsigned long long t1 = (VERIFY & V_PROF) ? gtime() : 0;

    // GEMM: hbuf[t % 2] (A, 16 x D) times the resident R block (B) -> gate preactivations of the warp's columns.
    {
      const elem_t* hb = hbuf + (size_t)(t & 1) * BM * LDH;
      const size_t par = (size_t)(t & 1) * BM * LDH;
      float acc[KSPLIT][NTW][4];
#pragma unroll
      for (int s = 0; s < KSPLIT; ++s)
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[s][nt][q] = 0.f;
      const uint2* sb = sr + (size_t)warp * CSM * NTW * 32 + lane;
#pragma unroll
      for (int c = 0; c < KS; ++c) {
        // A fragment rows fg / fg + 8, hidden indices c * 16 + 2 ft (+ 8): local, or the owner's buffer (pull mode).
        const elem_t* src = PULL ? peer[(c * 16) / NU] + par : hb;
        const elem_t* r0 = src + (size_t)fg * LDH + c * 16 + 2 * ft;
        const elem_t* r8 = r0 + 8 * LDH;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(r0), a1 = *reinterpret_cast<const uint32_t*>(r8);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(r0 + 8), a3 = *reinterpret_cast<const uint32_t*>(r8 + 8);
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
          uint32_t b0, b1;
          if (c < CR) {
            b0 = rreg[c < CR ? c : 0][nt][0];
            b1 = rreg[c < CR ? c : 0][nt][1];
          } else {
            const uint2 x = sb[((size_t)(c - CR) * NTW + nt) * 32];
            b0 = x.x;
            b1 = x.y;
          }
          mma(acc[c % KSPLIT][nt], a0, a1, a2, a3, b0, b1);
        }
      }
#pragma unroll
      for (int nt = 0; nt < NTW; ++nt) {
#pragma unroll
        for (int s = 1; s < KSPLIT; ++s)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[0][nt][q] += acc[s][nt][q];
        const int col = c_warp + nt * 8 + 2 * ft;
        *reinterpret_cast<float2*>(spart + (size_t)fg * NC + col) = make_float2(acc[0][nt][0], acc[0][nt][1]);
        *reinterpret_cast<float2*>(spart + (size_t)(fg + 8) * NC + col) = make_float2(acc[0][nt][2], acc[0][nt][3]);
      }
    }
    __syncthreads();  // the warps' preactivations are complete in spart
    unsigned long long t2 = (VERIFY & V_PROF) ? gtime() : 0;

    // Pointwise update of the owned elements; round(h_t) into every peer's next-parity buffer.
    elem_t* nb_off = hbuf + (size_t)((t + 1) & 1) * BM * LDH;
    const size_t nb_delta = (size_t)(nb_off - hbuf);
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
      if (!last) {  // rows beyond B publish zeros
        const elem_t v = to_elem(h);
        if (PULL) {
          hbuf[nb_delta + (size_t)lrow * LDH + o] = v;
        } else {
#pragma unroll
          for (int r = 0; r < C; ++r) peer[r][nb_delta + (size_t)lrow * LDH + o] = v;
        }
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
  // Final FP32 carry.
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
  cluster.sync();  // no CTA exits while a peer may still store into its buffers
}

extern "C" int frnn_hybrid_fwc_smem() { return SMEM_R + SMEM_H + SMEM_P; }
extern "C" int frnn_hybrid_fwc_ctas() { return C; }

extern "C" int frnn_hybrid_fwc(const void* wx, const void* R, const void* bias, const float* initial, void* out,
                               float* final, int* err, const void* zero_first, int B, int T, int H, int groups,
                               int tiles, void* stream) {
  Args a{reinterpret_cast<const elem_t*>(wx), reinterpret_cast<const elem_t*>(R),
         reinterpret_cast<const elem_t*>(bias), initial, reinterpret_cast<snap_t*>(out), final, err,
         reinterpret_cast<const bool*>(zero_first), B, T, H, tiles};
  const int smem = frnn_hybrid_fwc_smem();
  cudaError_t e = cudaFuncSetAttribute(frnn_hybrid_fwc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return (int)e;
  if (C > 8) {
    e = cudaFuncSetAttribute(frnn_hybrid_fwc_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
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
  e = cudaLaunchKernelEx(&cfg, frnn_hybrid_fwc_kernel, a);
  return (int)e;
}
