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
// FRNN_FWC_BULKX = 1 (bulk exchange, section 82): the exchange buffers are unit-block-major ([2 parities][C blocks]
// [BM rows][NUP = NU + 8 padded columns]); the owner writes round(h_t) into its own block locally, and one thread then
// issues one bulk asynchronous copy of that block (cp.async.bulk shared::cta -> shared::cluster) to every peer, each
// completing on the peer's mbarrier of the target parity (expect_tx = C blocks per step). The per-step cluster
// barrier is replaced by the mbarrier wait: a CTA's GEMM of step t + 1 starts only when every peer's round(h_t) has
// landed, and every peer wrote round(h_t) only after its GEMM of step t, the last reader of the buffer the next
// writes go to, so one buffer per parity is race-free without a barrier. Phase timers are finer in this mode:
// err[4] wait (and the Wx loads), err[5] MMA chain, err[6] partial store + CTA barrier, err[7] pointwise + outputs,
// err[8] exchange (local stores + copy issue).
//
// FRNN_FWC_STASYNC = 1 (needs FRNN_FWC_BULKX, section 82): the owners write round(h_t) into every peer's block with
// st.async (4 bytes = two adjacent units per store), each completing on the peer's mbarrier of the target parity; no
// local staging, no proxy fence, no copy issue. Ownership is then by unit pairs.
// FRNN_FASTMATH = 1: __expf / __logf / __fdividef in the cell (approximate transcendentals; the FP64 qualification
// decides whether the error budget holds).
// The GEMM loads the fragments of k-step c + 1 while k-step c multiplies (explicit software pipeline).
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
#ifndef FRNN_FWC_BULKX
#define FRNN_FWC_BULKX 0
#endif
constexpr bool BULKX = FRNN_FWC_BULKX != 0;
#if FRNN_FWC_BULKX && FRNN_FWC_PULL
#error "FRNN_FWC_BULKX excludes FRNN_FWC_PULL"
#endif
#ifndef FRNN_FWC_STASYNC
#define FRNN_FWC_STASYNC 0
#endif
constexpr bool STASYNC = FRNN_FWC_STASYNC != 0;
#if FRNN_FWC_STASYNC && !FRNN_FWC_BULKX
#error "FRNN_FWC_STASYNC needs the FRNN_FWC_BULKX buffer layout"
#endif
#ifndef FRNN_FASTMATH
#define FRNN_FASTMATH 0
#endif
constexpr int UP = STASYNC ? 2 : 1;  // units per owned slot (st.async moves two adjacent units per store)
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
constexpr int NUP = NU + 8;  // FRNN_FWC_BULKX: padded columns of one unit block
constexpr int XBLK = BM * NUP;  // elements of one exchanged block
constexpr int PE = (BM * NU / UP + THREADS - 1) / THREADS;
constexpr int SMEM_R = WARPS * CSM * NTW * 32 * 8, SMEM_P = BM * NC * 4;
constexpr int SMEM_H = BULKX ? 2 * C * XBLK * 2 : 2 * BM * LDH * 2;
static_assert(!BULKX || (XBLK * 2) % 16 == 0, "bulk copies move multiples of 16 bytes");
static_assert(D % C == 0 && NU % 16 == 0, "a warp's columns must lie within one gate (NU a multiple of 16)");
static_assert(NU % NCW == 0, "a warp's columns must not cross a gate");
static_assert(!PULL || NU % 16 == 0, "pull mode reads one owner per 16-wide k-step");
static_assert(CR >= 0 && CR <= KS, "register k-steps");
static_assert(KSPLIT >= 1 && KSPLIT <= 4, "accumulator split");
static_assert(SMEM_R + SMEM_H + SMEM_P <= 227 * 1024, "R, the two h buffers and the partials must fit shared memory");
static_assert(MODE >= 0 && MODE <= 2, "output mode");

#if FRNN_FASTMATH
__device__ __forceinline__ float fexp(float x) { return __expf(x); }
__device__ __forceinline__ float flog(float x) { return __logf(x); }
__device__ __forceinline__ float sigmoid(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
#else
__device__ __forceinline__ float fexp(float x) { return expf(x); }
__device__ __forceinline__ float flog(float x) { return logf(x); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
#endif

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
  asm(FRNN_MMA " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// B fragment register for k (even) and k + 1, output column (g, o): R[g, head, o, k .. k + 1] (contiguous).
__device__ __forceinline__ uint32_t r_pair(const elem_t* R, int H, int head, int g, int o, int k) {
  const size_t base = (((size_t)g * H + head) * D + o) * D + k;
  return pack(R[base], R[base + 1]);
}

#if FRNN_FWC_BULKX
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect(unsigned long long* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
// Bulk copy of a local shared block into the same offset of peer CTA's shared memory, completing on the peer's copy
// of the given local mbarrier (both addresses mapped with mapa).
__device__ __forceinline__ void bulk_to_peer(const void* local_src, unsigned bytes, unsigned long long* local_bar,
                                             int peer) {
  const unsigned src = smem_u32(local_src), bar = smem_u32(local_bar);
  unsigned rdst, rbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(src), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar), "r"(peer));
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(rdst), "r"(src), "r"(bytes), "r"(rbar) : "memory");
}
#if FRNN_FWC_STASYNC
// 4-byte store into peer CTA's shared memory (same offset as local_dst), completing 4 transaction bytes on the peer's
// copy of local_bar.
__device__ __forceinline__ void st_async_peer(void* local_dst, uint32_t value, unsigned long long* local_bar, int peer) {
  const unsigned dst = smem_u32(local_dst), bar = smem_u32(local_bar);
  unsigned rdst, rbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(dst), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar), "r"(peer));
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
               ::"r"(rdst), "r"(value), "r"(rbar) : "memory");
}
#endif
#endif

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
#if FRNN_FWC_BULKX
  __shared__ __align__(8) unsigned long long xbar[2];  // one per buffer parity
  unsigned xphase[2] = {0u, 0u};
  if (threadIdx.x == 0) {
    mbar_init(&xbar[0], 1);
    mbar_init(&xbar[1], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster.sync();  // barriers initialized in every CTA before any peer's copy can target them
#endif
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

  // Pointwise ownership: slot e of this thread is (row idx / (NU / UP), units cb * NU + UP * (idx % (NU / UP)) ..
  // + UP - 1), fixed for the call.
  constexpr int NUS = NU / UP;
  auto elem = [&](int e, int& lrow, int& u) -> bool {
    const int idx = tid + THREADS * e;
    lrow = idx / NUS;
    u = UP * (idx % NUS);
    return idx < BM * NUS;
  };
  float hs[PE][UP], cs[PE][UP], ns[PE][UP], ms[PE][UP], bias[PE][UP][4];
  elem_t wxr[PE][UP][4];
  // Peer buffers: round(h) rows of this CTA's units go to every CTA of the cluster (own rank included).
  elem_t* peer[C];
#pragma unroll
  for (int r = 0; r < C; ++r) peer[r] = cluster.map_shared_rank(hbuf, r);
#if FRNN_FWC_STASYNC
  constexpr unsigned XBYTES = (unsigned)(C * BM * NU * sizeof(elem_t));  // st.async bytes landing per step
  if (tid == 0) mbar_expect(&xbar[0], XBYTES);
#endif
#pragma unroll
  for (int e = 0; e < PE; ++e) {
    int lrow, u0;
    const bool own = elem(e, lrow, u0);
    const int row = bt * BM + lrow;
    const bool valid = own && row < a.B;
#pragma unroll
    for (int j = 0; j < UP; ++j) {
      const int o = cb * NU + u0 + j;
      const size_t unit = ((size_t)row * a.H + head) * D + o;
      hs[e][j] = valid ? a.initial[unit] : 0.f;
      cs[e][j] = valid ? a.initial[state_stride + unit] : 0.f;
      ns[e][j] = (SLSTM && valid) ? a.initial[2 * state_stride + unit] : 0.f;
      ms[e][j] = (SLSTM && valid) ? a.initial[3 * state_stride + unit] : 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) bias[e][j][q] = own ? from_elem(a.bias[((size_t)q * a.H + head) * D + o]) : 0.f;
    }
    if (own) {  // round(h_0) into buffer 0 (of every peer, or only the own buffer in pull / bulk mode); rows >= B: 0
      if (STASYNC) {
#if FRNN_FWC_STASYNC
        const uint32_t v = pack(to_elem(hs[e][0]), to_elem(hs[e][UP - 1]));
#pragma unroll
        for (int r = 0; r < C; ++r) st_async_peer(hbuf + (size_t)cb * XBLK + (size_t)lrow * NUP + u0, v, &xbar[0], r);
#endif
      } else if (BULKX) {
        hbuf[(size_t)cb * XBLK + (size_t)lrow * NUP + u0] = to_elem(hs[e][0]);
      } else if (PULL) {
        hbuf[(size_t)lrow * LDH + cb * NU + u0] = to_elem(hs[e][0]);
      } else {
#pragma unroll
        for (int r = 0; r < C; ++r) peer[r][(size_t)lrow * LDH + cb * NU + u0] = to_elem(hs[e][0]);
      }
    }
  }
#if FRNN_FWC_BULKX && !FRNN_FWC_STASYNC
  // Prologue exchange: my block of round(h_0) to every peer's parity-0 buffer; arm my parity-0 barrier.
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (tid == 0) mbar_expect(&xbar[0], (unsigned)(C * XBLK * sizeof(elem_t)));
  if (tid < C) bulk_to_peer(hbuf + (size_t)cb * XBLK, XBLK * sizeof(elem_t), &xbar[0], tid);
#endif

  for (int t = 0; t < a.T; ++t) {
    const bool last = t == a.T - 1;
    unsigned long long t0 = (VERIFY & V_PROF) ? gtime() : 0;
    if (!BULKX) cluster_arrive();  // my round(h_{t-1}) stores (and the prologue's h_0) are released to the cluster
#pragma unroll
    for (int e = 0; e < PE; ++e) {  // this step's input projections, loaded while the cluster arrives
      int lrow, u0;
      const bool own = elem(e, lrow, u0);
      const int row = bt * BM + lrow;
      if (own && row < a.B) {
#pragma unroll
        for (int j = 0; j < UP; ++j) {
          const size_t xo = (((size_t)row * a.T + t) * 4 * a.H + head) * D + cb * NU + u0 + j;
#pragma unroll
          for (int q = 0; q < 4; ++q) wxr[e][j][q] = a.wx[xo + (size_t)q * hd];
        }
      }
    }
#if FRNN_FWC_BULKX
    mbar_wait(&xbar[t & 1], xphase[t & 1]);  // every peer's block of round(h_{t-1}) landed in parity t % 2
    xphase[t & 1] ^= 1u;
#else
    cluster_wait();  // every peer's round(h_{t-1}) is in hbuf[t % 2]
#endif
    unsigned long long t1 = (VERIFY & V_PROF) ? gtime() : 0;

    // GEMM: hbuf[t % 2] (A, 16 x D) times the resident R block (B) -> gate preactivations of the warp's columns.
    {
      const elem_t* hb = hbuf + (BULKX ? (size_t)(t & 1) * C * XBLK : (size_t)(t & 1) * BM * LDH);
      const size_t par = (size_t)(t & 1) * BM * LDH;
      float acc[KSPLIT][NTW][4];
#pragma unroll
      for (int s = 0; s < KSPLIT; ++s)
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[s][nt][q] = 0.f;
      const uint2* sb = sr + (size_t)warp * CSM * NTW * 32 + lane;
      // Fragments of k-step c: A rows fg / fg + 8, hidden indices c * 16 + 2 ft (+ 8), from the local buffer or the
      // owner's (pull mode); B from registers (c < CR) or shared memory.
      auto load_a = [&](int c, uint32_t (&af)[4]) {
        const elem_t* r0;
        const elem_t* r8;
        if (BULKX) {  // block (c * 16) / NU of the unit-block-major buffer, column (c * 16) % NU + 2 ft
          r0 = hb + (size_t)((c * 16) / NU) * XBLK + (size_t)fg * NUP + (c * 16) % NU + 2 * ft;
          r8 = r0 + 8 * NUP;
        } else {
          const elem_t* src = PULL ? peer[(c * 16) / NU] + par : hb;
          r0 = src + (size_t)fg * LDH + c * 16 + 2 * ft;
          r8 = r0 + 8 * LDH;
        }
        af[0] = *reinterpret_cast<const uint32_t*>(r0);
        af[1] = *reinterpret_cast<const uint32_t*>(r8);
        af[2] = *reinterpret_cast<const uint32_t*>(r0 + 8);
        af[3] = *reinterpret_cast<const uint32_t*>(r8 + 8);
      };
      auto load_b = [&](int c, uint32_t (&bf)[NTW][2]) {
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
          if (c < CR) {
            bf[nt][0] = rreg[c < CR ? c : 0][nt][0];
            bf[nt][1] = rreg[c < CR ? c : 0][nt][1];
          } else {
            const uint2 x = sb[((size_t)(c - CR) * NTW + nt) * 32];
            bf[nt][0] = x.x;
            bf[nt][1] = x.y;
          }
        }
      };
      uint32_t af[2][4], bf[2][NTW][2];
      load_a(0, af[0]);
      load_b(0, bf[0]);
#pragma unroll
      for (int c = 0; c < KS; ++c) {  // the next k-step's fragments load while this one multiplies
        if (c + 1 < KS) {
          load_a(c + 1, af[(c + 1) & 1]);
          load_b(c + 1, bf[(c + 1) & 1]);
        }
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt)
          mma(acc[c % KSPLIT][nt], af[c & 1][0], af[c & 1][1], af[c & 1][2], af[c & 1][3], bf[c & 1][nt][0],
              bf[c & 1][nt][1]);
      }
      if ((VERIFY & V_PROF) && tid == 0) atomicAdd(a.err + 5, (int)((gtime() - t1) >> 6));  // MMA chain
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
    const size_t nb_delta = BULKX ? (size_t)((t + 1) & 1) * C * XBLK + (size_t)cb * XBLK
                                  : (size_t)((t + 1) & 1) * BM * LDH;
    elem_t hv[PE][UP];
#pragma unroll
    for (int e = 0; e < PE; ++e) {
      int lrow, u0;
      const bool own = elem(e, lrow, u0);
      if (!own) continue;
      const int row = bt * BM + lrow;
#pragma unroll
      for (int j = 0; j < UP; ++j) {
        const int u = u0 + j, o = cb * NU + u;
        float h = 0.f;
        if (row < a.B) {
          float gate[4];
#pragma unroll
          for (int q = 0; q < 4; ++q)
            gate[q] = spart[(size_t)lrow * NC + (size_t)q * NU + u] + from_elem(wxr[e][j][q]) + bias[e][j][q];
          const float gi = gate[0], gf = gate[1], gz = gate[2], go = gate[3];
          const float z = 2.0f * sigmoid(2.0f * gz) - 1.0f;
          if (SLSTM) {
            const float logf_ = fminf(gf, 0.0f) - flog(1.0f + fexp(-fabsf(gf))) + ms[e][j];
            const float m_new = (t == 0 && zero_first) ? gi : fmaxf(gi, logf_);
            const float igate = fexp(gi - m_new), fgate = fexp(logf_ - m_new);
            cs[e][j] = fgate * cs[e][j] + igate * z;
            ns[e][j] = fmaxf(fgate * ns[e][j] + igate, 1.0f);
            h = sigmoid(go) * cs[e][j] / ns[e][j];
            ms[e][j] = m_new;
          } else {
            cs[e][j] = sigmoid(gf) * cs[e][j] + sigmoid(gi) * z;
            h = sigmoid(go) * (2.0f * sigmoid(2.0f * cs[e][j]) - 1.0f);
          }
          hs[e][j] = h;
          const size_t here = (((size_t)row * a.T + t) * a.H + head) * D + o;
          if (MODE == 0) {
            a.out[here] = to_snap(h);
            a.out[out_stride + here] = to_snap(cs[e][j]);
            if (SLSTM) {
              a.out[2 * out_stride + here] = to_snap(ns[e][j]);
              a.out[3 * out_stride + here] = to_snap(ms[e][j]);
            }
          } else if (MODE == 1) {
            a.out[here] = to_snap(h);
          } else if (last) {
            a.out[((size_t)row * a.H + head) * D + o] = to_snap(h);
          }
        }
        hv[e][j] = to_elem(h);
      }
    }
    unsigned long long t3a = (VERIFY & V_PROF) ? gtime() : 0;
    if (!last) {  // round(h_t) to the peers; rows beyond B publish zeros
#if FRNN_FWC_STASYNC
      if (tid == 0) mbar_expect(&xbar[(t + 1) & 1], XBYTES);  // arm my next-parity barrier for the peers' stores
#endif
#pragma unroll
      for (int e = 0; e < PE; ++e) {
        int lrow, u0;
        if (!elem(e, lrow, u0)) continue;
        const int o = cb * NU + u0;
        if (STASYNC) {
#if FRNN_FWC_STASYNC
          const uint32_t v = pack(hv[e][0], hv[e][UP - 1]);
#pragma unroll
          for (int r = 0; r < C; ++r) st_async_peer(hbuf + nb_delta + (size_t)lrow * NUP + u0, v, &xbar[(t + 1) & 1], r);
#endif
        } else if (BULKX) {
          hbuf[nb_delta + (size_t)lrow * NUP + u0] = hv[e][0];
        } else if (PULL) {
          hbuf[nb_delta + (size_t)lrow * LDH + o] = hv[e][0];
        } else {
#pragma unroll
          for (int r = 0; r < C; ++r) peer[r][nb_delta + (size_t)lrow * LDH + o] = hv[e][0];
        }
      }
#if FRNN_FWC_BULKX && !FRNN_FWC_STASYNC
      __syncthreads();  // the block is complete in local shared memory
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic stores before the async-proxy copy
      if (tid == 0) mbar_expect(&xbar[(t + 1) & 1], (unsigned)(C * XBLK * sizeof(elem_t)));
      if (tid < C) bulk_to_peer(hbuf + nb_delta, XBLK * sizeof(elem_t), &xbar[(t + 1) & 1], tid);
#endif
    }
    if ((VERIFY & V_PROF) && tid == 0) {
      const unsigned long long t3 = gtime();
      atomicAdd(a.err + 4, (int)((t1 - t0) >> 6));   // wait (with the Wx loads)
      atomicAdd(a.err + 6, (int)((t2 - t1) >> 6));   // MMA chain + partial store + CTA barrier (err[5]: chain alone)
      atomicAdd(a.err + 7, (int)((t3a - t2) >> 6));  // pointwise math + outputs
      atomicAdd(a.err + 8, (int)((t3 - t3a) >> 6));  // exchange
      atomicAdd(a.err + 9, 1);
    }
  }
  // Final FP32 carry.
#pragma unroll
  for (int e = 0; e < PE; ++e) {
    int lrow, u0;
    const bool own = elem(e, lrow, u0);
    const int row = bt * BM + lrow;
#pragma unroll
    for (int j = 0; j < UP; ++j) {
      const size_t unit = ((size_t)row * a.H + head) * D + cb * NU + u0 + j;
      if (own && row < a.B) {
        a.final[unit] = hs[e][j];
        a.final[state_stride + unit] = cs[e][j];
        if (SLSTM) {
          a.final[2 * state_stride + unit] = ns[e][j];
          a.final[3 * state_stride + unit] = ms[e][j];
        }
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
