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
// FRNN_FWC_WGMMA = 1 (section 82, needs FRNN_FWC_BULKX and NC = 128): the GEMM is transposed onto Hopper's
// warpgroup MMA, preact^T [NC x BM] = R_block [NC x D] x round(h)^T [D x BM], as wgmma.m64n16k16 with the CTA's R
// block as the M = 64 operand (warpgroup w: output columns 64 w .. 64 w + 63) and round(h) as the N = 16 operand.
// R k-steps c < CR live in registers (A fragments, 4 per k-step and thread), the rest in shared memory as canonical
// K-major core-matrix tiles (64 x 16 per k-step, 2 KB); round(h) is exchanged straight into the canonical K-major
// layout of the B operand (8 x 8 core matrices of 128 contiguous bytes; the two 8-row groups adjacent, SBO 128 B;
// K-adjacent core columns LBO = 288 B apart, 32 B of it padding), so a CTA's block is one contiguous piece: one bulk
// copy per peer, or 16-byte st.async stores (a quad of lanes gathers its 8 units) into the canonical positions. FRNN_WG_SWAP = 1 swaps the descriptors' leading / stride offsets (tools/wgmma_probe.cu
// decides the convention).
// FRNN_TRACE = 1 (section 82): thread 0 of every CTA of group 0 writes globaltimer stamps before and after the
// per-step exchange wait into trace[(t * C + cta) * 2 + {0, 1}] (arrival-jitter diagnostics).
// FRNN_FWC_MCAST = 1 (section 82, needs the bulk exchange without st.async): the owner writes its round(h) block to a
// global staging slot (gstage, double buffered by parity) instead of local shared memory and issues ONE bulk copy
// global -> shared::cluster with .multicast::cluster to all C CTAs (itself included), completing on every CTA's
// mbarrier of the target parity. One asynchronous request per CTA and step instead of C, at the price of an L2
// round trip (generic stores, fence.proxy.async.global, TMA read). Section 67 found the barrier-based form of this
// unprofitable on H100; this form has no barrier.
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
#ifndef FRNN_FWC_WGMMA
#define FRNN_FWC_WGMMA 0
#endif
constexpr bool WG = FRNN_FWC_WGMMA != 0;
#if FRNN_FWC_WGMMA && !FRNN_FWC_BULKX
#error "FRNN_FWC_WGMMA needs the mbarrier exchange (FRNN_FWC_BULKX, with or without FRNN_FWC_STASYNC)"
#endif
#ifndef FRNN_WG_SWAP
#define FRNN_WG_SWAP 0
#endif
#ifndef FRNN_TRACE
#define FRNN_TRACE 0
#endif
constexpr bool TRACE = FRNN_TRACE != 0;
#ifndef FRNN_FWC_MCAST
#define FRNN_FWC_MCAST 0
#endif
constexpr bool MCAST = FRNN_FWC_MCAST != 0;
#if FRNN_FWC_MCAST && (!FRNN_FWC_BULKX || FRNN_FWC_STASYNC)
#error "FRNN_FWC_MCAST needs the bulk exchange layout (FRNN_FWC_BULKX without FRNN_FWC_STASYNC)"
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
// Elements of one exchange buffer (per step parity): canonical K-major [BM x D] (wgmma), unit-block-major (bulk), or
// padded row-major.
constexpr int HK = 144;  // wgmma mode: elements per K core column of the h operand (2 x 64 + 16 padding)
constexpr int PBUF = WG ? (D / 8) * HK : (BULKX ? C * XBLK : BM * LDH);
constexpr int BLKE = WG ? (NU / 8) * HK : XBLK;  // bulk modes: elements of one CTA's contiguous exchange block
constexpr int SMEM_H = 2 * PBUF * 2;
static_assert(!BULKX || (XBLK * 2) % 16 == 0, "bulk copies move multiples of 16 bytes");
static_assert(!WG || NC == 128, "wgmma mode: two m64 tiles, one per warpgroup (D = 32 C)");
static_assert(!WG || WARPS * CSM * NTW * 32 * 8 == 2 * CSM * 2048, "wgmma mode: one 2 KB tile per warpgroup and k-step");
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
#if FRNN_FWC_MCAST
// One bulk copy of a global block into the same shared offset of every CTA in ctaMask, completing `bytes` on each
// CTA's copy of local_bar.
__device__ __forceinline__ void mcast_from_global(void* local_dst, const void* gsrc, unsigned bytes,
                                                  unsigned long long* local_bar, unsigned short mask) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
               ::"r"(smem_u32(local_dst)), "l"(gsrc), "r"(bytes), "r"(smem_u32(local_bar)), "h"(mask) : "memory");
}
#endif
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
// 16-byte store (four packed unit pairs) into peer CTA's shared memory, completing 16 transaction bytes.
__device__ __forceinline__ void st_async_peer_v4(void* local_dst, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
                                                 unsigned long long* local_bar, int peer) {
  const unsigned dst = smem_u32(local_dst), bar = smem_u32(local_bar);
  unsigned rdst, rbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(dst), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar), "r"(peer));
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
               ::"r"(rdst), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(rbar) : "memory");
}
#endif
#endif

#if FRNN_FWC_WGMMA
// Canonical K-major no-swizzle layouts (elements; a core matrix is 8 rows x 16 bytes, contiguous). h operand
// [BM x D]: the two 8-row groups of one K core column are adjacent (SBO = 128 B) and K-adjacent core columns are
// HK = 288 B apart (LBO; 256 B of data + 32 B of padding), so that a CTA's block of NU units is one contiguous piece
// (one bulk copy per peer) and a warp's exchange stores of 4 consecutive core columns hit distinct bank groups. R tile
// [64 x 16] of one k-step: core matrix (m / 8, k / 8) at (m / 8) * 128 + (k / 8) * 64 (SBO 256 B, LBO 128 B).
__device__ __forceinline__ int hoff(int n, int k) { return (k >> 3) * HK + (n >> 3) * 64 + (n & 7) * 8 + (k & 7); }
__device__ __forceinline__ int aoff(int m, int k) { return (m >> 3) * 128 + (k >> 3) * 64 + (m & 7) * 8 + (k & 7); }
constexpr unsigned WG_LBO_H = FRNN_WG_SWAP ? 128 : HK * 2, WG_SBO_H = FRNN_WG_SWAP ? HK * 2 : 128;
constexpr unsigned WG_LBO_A = FRNN_WG_SWAP ? 256 : 128, WG_SBO_A = FRNN_WG_SWAP ? 128 : 256;
__device__ __forceinline__ uint64_t make_desc(const void* p, unsigned lbo, unsigned sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_u32(p) & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo & 0x3FFFFu) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFFu) >> 4) << 32;
  return d;  // base offset 0, no swizzle
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
#if FRNN_BF16
#define FRNN_WGMMA "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"
#else
#define FRNN_WGMMA "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16"
#endif
__device__ __forceinline__ void wg_ss(float (&d)[8], uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n" FRNN_WGMMA
               " {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "l"(da), "l"(db), "r"(1));
}
__device__ __forceinline__ void wg_rs(float (&d)[8], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n" FRNN_WGMMA
               " {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}
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
  unsigned long long* trace;  // FRNN_TRACE: [T][C][2] globaltimer stamps of group 0 (before / after the wait)
  elem_t* gstage;             // FRNN_FWC_MCAST: [groups][C][2 parities][BLKE] global staging of round(h) blocks
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
#if FRNN_FWC_WGMMA
  // wgmma A operand: warpgroup wg owns output columns 64 wg .. + 63; warp wig of it rows 16 wig + fg (+ 8) of that
  // tile. Registers: A fragments of k-steps c < CR (k = 16 c + 2 ft, + 8); shared memory: canonical tiles of the rest.
  const int wg = warp >> 2, wig = warp & 3;
  elem_t* sa = reinterpret_cast<elem_t*>(smem);  // [2 warpgroups][CSM][64 x 16 canonical tile]
  uint32_t areg[CR > 0 ? CR : 1][4];
  {
    const int ol = wg * 64 + wig * 16 + fg;  // local output column (gate ol / NU, unit ol % NU)
#pragma unroll
    for (int c = 0; c < CR; ++c) {
      const int k = c * 16 + 2 * ft;
      areg[c][0] = r_pair(a.R, a.H, head, ol / NU, cb * NU + ol % NU, k);
      areg[c][1] = r_pair(a.R, a.H, head, (ol + 8) / NU, cb * NU + (ol + 8) % NU, k);
      areg[c][2] = r_pair(a.R, a.H, head, ol / NU, cb * NU + ol % NU, k + 8);
      areg[c][3] = r_pair(a.R, a.H, head, (ol + 8) / NU, cb * NU + (ol + 8) % NU, k + 8);
    }
    for (int i = tid; i < 2 * CSM * 1024; i += THREADS) {
      const int w = i / (CSM * 1024), j = (i / 1024) % CSM, m = (i % 1024) / 16, kk = i % 16;
      const int o = w * 64 + m, k = (CR + j) * 16 + kk;
      sa[(size_t)(w * CSM + j) * 1024 + aoff(m, kk)] =
          a.R[(((size_t)(o / NU) * a.H + head) * D + cb * NU + o % NU) * D + k];
    }
  }
#else
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
#endif
  // Element offset of (row lrow, unit u0 of this CTA) within one parity buffer of the exchange layout.
  auto xoff = [&](int lrow, int u0) -> int {
#if FRNN_FWC_WGMMA
    return hoff(lrow, cb * NU + u0);
#else
    return BULKX ? cb * XBLK + lrow * NUP + u0 : lrow * LDH + cb * NU + u0;
#endif
  };
#if FRNN_FWC_MCAST
  // My global staging slots (one per parity); the block's element (lrow, u0) lives at xoff - my block's base.
  elem_t* const gstage = a.gstage + ((size_t)g * C + cb) * 2 * BLKE;
  const int blk0 = WG ? cb * (NU / 8) * HK : cb * XBLK;
#endif
#if FRNN_FWC_STASYNC
  // st.async publication of one owned unit pair (packed) to every peer's buffer at parity offset pbase. In wgmma mode
  // the four lanes of a quad own 8 consecutive units of one row (one 16-byte core-matrix row piece): the quad gathers
  // its four pairs and lane j stores the 16 bytes to peers j, j + 4, ... (4 v4 stores per lane instead of C b32 stores).
  auto publish = [&](uint32_t v, int lrow, int u0, size_t pbase, unsigned long long* bar) {
#if FRNN_FWC_WGMMA
    static_assert(PE == 1 && NU / UP >= 4 && (NU / UP) % 4 == 0, "wgmma st.async: whole quads own 8-unit pieces");
    const int q = lane & 3, base = lane & ~3;
    const uint32_t v0 = __shfl_sync(0xffffffffu, v, base), v1 = __shfl_sync(0xffffffffu, v, base + 1),
                   v2 = __shfl_sync(0xffffffffu, v, base + 2), v3 = __shfl_sync(0xffffffffu, v, base + 3);
    elem_t* dst = hbuf + pbase + xoff(lrow, u0 & ~7);
    for (int r = q; r < C; r += 4) st_async_peer_v4(dst, v0, v1, v2, v3, bar, r);
#else
    elem_t* dst = hbuf + pbase + xoff(lrow, u0);
#pragma unroll
    for (int r = 0; r < C; ++r) st_async_peer(dst, v, bar, r);
#endif
  };
#endif

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
      const int off = xoff(lrow, u0);
      if (STASYNC) {
#if FRNN_FWC_STASYNC
        publish(pack(to_elem(hs[e][0]), to_elem(hs[e][UP - 1])), lrow, u0, 0, &xbar[0]);
#endif
      } else if (MCAST) {
#if FRNN_FWC_MCAST
        gstage[off - blk0] = to_elem(hs[e][0]);
#endif
      } else if (BULKX || PULL) {
        hbuf[off] = to_elem(hs[e][0]);
      } else {
#pragma unroll
        for (int r = 0; r < C; ++r) peer[r][off] = to_elem(hs[e][0]);
      }
    }
  }
#if FRNN_FWC_BULKX && !FRNN_FWC_STASYNC
  // Prologue exchange: my block of round(h_0) to every peer's parity-0 buffer; arm my parity-0 barrier. In wgmma
  // mode the block is two contiguous pieces (one per 8-row group), NU * 16 bytes each.
  // Bytes landing per step (all CTAs' blocks): the canonical layout has no padding; the unit-block-major layout
  // copies its NUP padded columns too.
  constexpr unsigned XBYTES = WG ? (unsigned)(PBUF * sizeof(elem_t)) : (unsigned)(C * XBLK * sizeof(elem_t));
  __syncthreads();
#if FRNN_FWC_MCAST
  if (tid == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");  // the CTA's generic stores before the async-proxy read
    mbar_expect(&xbar[0], XBYTES);
    mcast_from_global(hbuf + blk0, gstage, BLKE * sizeof(elem_t), &xbar[0], (unsigned short)((1u << C) - 1u));
  }
#else
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (tid == 0) mbar_expect(&xbar[0], XBYTES);
  if (tid < C) bulk_to_peer(hbuf + (size_t)(WG ? cb * (NU / 8) * HK : cb * XBLK), BLKE * sizeof(elem_t), &xbar[0], tid);
#endif
#endif
#if FRNN_FWC_WGMMA
  __syncthreads();  // the canonical R tiles are complete before any warpgroup MMA reads them
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif

  for (int t = 0; t < a.T; ++t) {
    const bool last = t == a.T - 1;
    unsigned long long t0 = ((VERIFY & V_PROF) || TRACE) ? gtime() : 0;
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
    unsigned long long t1 = ((VERIFY & V_PROF) || TRACE) ? gtime() : 0;
    if (TRACE && tid == 0 && blockIdx.y == 0) {
      a.trace[((size_t)t * C + cb) * 2] = t0;
      a.trace[((size_t)t * C + cb) * 2 + 1] = t1;
    }

#if FRNN_FWC_WGMMA
    // GEMM (wgmma): preact^T [64 x 16 per warpgroup] = R tile [64 x D] x round(h)^T; A from registers (c < CR) or the
    // canonical shared-memory tiles, B = hbuf[t % 2] through its descriptor (k-step c: + 256 bytes).
    {
      const elem_t* hb = hbuf + (size_t)(t & 1) * PBUF;
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // peers' generic-proxy stores before the async MMA
      float acc[KSPLIT][8];
#pragma unroll
      for (int s = 0; s < KSPLIT; ++s)
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[s][q] = 0.f;
      wg_fence();
#pragma unroll
      for (int c = 0; c < KS; ++c) {
        const uint64_t db = make_desc(hb + c * 2 * HK, WG_LBO_H, WG_SBO_H);  // k-step c: K core columns 2c, 2c + 1
        if (c < CR) {
          wg_rs(acc[c % KSPLIT], areg[c < CR ? c : 0], db);
        } else {
          wg_ss(acc[c % KSPLIT], make_desc(sa + (size_t)(wg * CSM + (c - CR)) * 1024, WG_LBO_A, WG_SBO_A), db);
        }
      }
      wg_commit();
      wg_wait0();
      if ((VERIFY & V_PROF) && tid == 0) atomicAdd(a.err + 5, (int)((gtime() - t1) >> 6));  // MMA chain
#pragma unroll
      for (int s = 1; s < KSPLIT; ++s)
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[0][q] += acc[s][q];
      // D fragment: rows (output columns) 16 wig + fg (+ 8), columns (batch rows) 2 ft (+ 1), + 8.
      const int ol = wg * 64 + wig * 16 + fg, n0 = 2 * ft;
      spart[(size_t)n0 * NC + ol] = acc[0][0];
      spart[(size_t)(n0 + 1) * NC + ol] = acc[0][1];
      spart[(size_t)n0 * NC + ol + 8] = acc[0][2];
      spart[(size_t)(n0 + 1) * NC + ol + 8] = acc[0][3];
      spart[(size_t)(n0 + 8) * NC + ol] = acc[0][4];
      spart[(size_t)(n0 + 9) * NC + ol] = acc[0][5];
      spart[(size_t)(n0 + 8) * NC + ol + 8] = acc[0][6];
      spart[(size_t)(n0 + 9) * NC + ol + 8] = acc[0][7];
    }
#else
    // GEMM: hbuf[t % 2] (A, 16 x D) times the resident R block (B) -> gate preactivations of the warp's columns.
    {
      const elem_t* hb = hbuf + (size_t)(t & 1) * PBUF;
      const size_t par = (size_t)(t & 1) * PBUF;
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
#endif
    __syncthreads();  // the warps' preactivations are complete in spart
    unsigned long long t2 = (VERIFY & V_PROF) ? gtime() : 0;

    // Pointwise update of the owned elements; round(h_t) into every peer's next-parity buffer.
    const size_t nb_delta = (size_t)((t + 1) & 1) * PBUF;
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
        const size_t off = nb_delta + xoff(lrow, u0);
        if (STASYNC) {
#if FRNN_FWC_STASYNC
          publish(pack(hv[e][0], hv[e][UP - 1]), lrow, u0, nb_delta, &xbar[(t + 1) & 1]);
#endif
        } else if (MCAST) {
#if FRNN_FWC_MCAST
          gstage[(size_t)((t + 1) & 1) * BLKE + xoff(lrow, u0) - blk0] = hv[e][0];
#endif
        } else if (BULKX || PULL) {
          hbuf[off] = hv[e][0];
        } else {
#pragma unroll
          for (int r = 0; r < C; ++r) peer[r][off] = hv[e][0];
        }
      }
#if FRNN_FWC_MCAST
      __syncthreads();  // the block is complete in my global staging slot
      if (tid == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mbar_expect(&xbar[(t + 1) & 1], XBYTES);
        mcast_from_global(hbuf + nb_delta + blk0, gstage + (size_t)((t + 1) & 1) * BLKE, BLKE * sizeof(elem_t),
                          &xbar[(t + 1) & 1], (unsigned short)((1u << C) - 1u));
      }
#elif FRNN_FWC_BULKX && !FRNN_FWC_STASYNC
      __syncthreads();  // the block is complete in local shared memory
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic stores before the async-proxy copy
      if (tid == 0) mbar_expect(&xbar[(t + 1) & 1], XBYTES);
      if (tid < C) bulk_to_peer(hbuf + nb_delta + (size_t)(WG ? cb * (NU / 8) * HK : cb * XBLK), BLKE * sizeof(elem_t), &xbar[(t + 1) & 1], tid);
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
extern "C" int frnn_hybrid_fwc_block_elems() { return BULKX ? BLKE : 0; }

extern "C" int frnn_hybrid_fwc(const void* wx, const void* R, const void* bias, const float* initial, void* out,
                               float* final, int* err, const void* zero_first, int B, int T, int H, int groups,
                               int tiles, void* stream, void* trace, void* gstage) {
  Args a{reinterpret_cast<const elem_t*>(wx), reinterpret_cast<const elem_t*>(R),
         reinterpret_cast<const elem_t*>(bias), initial, reinterpret_cast<snap_t*>(out), final, err,
         reinterpret_cast<const bool*>(zero_first), B, T, H, tiles, reinterpret_cast<unsigned long long*>(trace),
         reinterpret_cast<elem_t*>(gstage)};
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
