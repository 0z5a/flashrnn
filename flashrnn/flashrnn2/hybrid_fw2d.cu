// Large-D forward with R split in two dimensions (docs/flashrnn2/max-width.md section 56): the transpose of
// hybrid_bw2d.cu.
//
// The PC x Q CTAs of a group form a grid. CTA (i, j) keeps R[g, o, k] resident (register rows + shared memory) for the
// NU = D / PC units o of unit block i, all four gates g (NC = 4 NU output columns), and the K block j of the hidden
// state (KQ = D / Q indices k). Per step it reads only its K block of round(h_{t-1}) from the exchange buffer and writes
// FP32 partial gate preactivations for its NC columns. The partials of a unit's four gates are summed over the Q K
// blocks, in fixed order j = 0 .. Q - 1, by the CTA that owns the unit's pointwise update (unit block i, rows
// j * BM / Q ..), which also adds Wx and the bias, updates the cell state (kept in registers for the whole slot),
// stores the history and publishes round(h_t). Per step:
//   grid barrier B2 (h_{t-1} complete in xbuf) -> GEMM (xbuf K block x resident R -> partial) -> grid barrier B1 ->
//   sum of the Q partials + pointwise update (h_t into the output and, rounded, into xbuf).
// The prologue of every slot publishes round(h_0) from the initial state. The xbuf and partial buffers are single: every
// xbuf write (pointwise) is separated from every xbuf read (GEMM) by B1 before it and B2 after it, and every partial
// write (GEMM) from every partial read (sum) by B2 before it and B1 after it, as in the backward.
//
// Warps split the NC columns (8 warps x NCW = NC / 8 columns, all of one gate since NCW divides NU); every warp covers
// all BM rows and the whole K block. The first CR k-steps of every warp's R are mma B fragments in registers, the rest
// in shared memory (fragment order). The h K block is staged in shared memory in 64-column chunks with cp.async
// (double buffered).
//
// Verification (FRNN_VERIFY): bit 1 content check of every staged A element against the stored h_{t-1} (the output,
// modes full and hidden; err[1]) and a step stamp per (group, unit block, K block) written after each partial and
// checked by every partial reader (err[2]); bit 2 seeded delays before the pointwise update and before the GEMM; bit 4
// broken: no barrier B1; bit 8 broken: no barrier B2 (each keeps the barriers of the last step); bit 16 phase timers
// (globaltimer, thread 0 of every active CTA, units of 64 ns): err[4] B2, err[5] GEMM, err[6] B1, err[7] pointwise,
// err[9] steps.
//
// B = 16 variants (opt-in, compile-time, max-width.md section 76; the default builds compile as before):
// * Q = 1 (a 1-D split, every CTA the whole K): the partials stay within the CTA, so B1 is a __syncthreads.
// * FRNN_FW_PREFETCH = 1: B2 is split into its release arrival and acquire wait (the protocol of cooperative groups'
//   grid.sync, counter after the stamps), and this step's input projections Wx[t] are loaded between them, so their
//   latency overlaps the wait and the GEMM. Same values and arithmetic: bitwise equal outputs.
// * FRNN_STAGES = n > 2: the h K-block chunks are staged n deep (n - 1 cp.async groups in flight) instead of double
//   buffered. Same staged values: bitwise equal outputs.
// * FRNN_FW_BULK = 1 (section 78): the whole h K block is staged with one bulk asynchronous copy per row
//   (cp.async.bulk, one mbarrier with the block's byte count) into a padded shared layout, instead of 64-column
//   cp.async rounds. Same staged values: bitwise equal outputs.
// * FRNN_FW_FLAGBAR = 1 (section 78): B2 is a flat flag barrier: every CTA publishes its step count with a release
//   store to its own flag after the pointwise update, and every CTA's threads poll all flags with acquire loads in
//   parallel (no single atomic counter). With FRNN_FW_PREFETCH the Wx loads sit between the publish and the poll.
//   Same ordering guarantees as the grid barrier: every xbuf write precedes the flag, every staged read follows the
//   acquire. The last step and the no-B2 verification variant behave as for the grid barrier.
// * FRNN_FW_SMEMPART = 1 (section 78, Q = 1 only): the GEMM writes its gate partials to shared memory and the pointwise
//   update reads them there (no global round trip); B1 is the CTA barrier it already is at Q = 1.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

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
#define FRNN_MODE 0  // 0 full (all state components), 1 hidden (h every step), 2 last (h of the last step)
#endif
#ifndef FRNN_SNAP32
#define FRNN_SNAP32 1  // output (snapshot) dtype: 1 FP32, 0 the input dtype
#endif
#if FRNN_SNAP32
typedef float snap_t;
__device__ __forceinline__ snap_t to_snap(float x) { return x; }
__device__ __forceinline__ float from_snap(snap_t x) { return x; }
#else
typedef elem_t snap_t;
__device__ __forceinline__ snap_t to_snap(float x) { return to_elem(x); }
__device__ __forceinline__ float from_snap(snap_t x) { return from_elem(x); }
#endif
constexpr int VERIFY = FRNN_VERIFY, MODE = FRNN_MODE;
#ifndef FRNN_FW_PREFETCH
#define FRNN_FW_PREFETCH 0
#endif
#ifndef FRNN_STAGES
#define FRNN_STAGES 2
#endif
constexpr int STAGES = FRNN_STAGES;
static_assert(STAGES >= 2 && STAGES <= 8, "cp.async stages");
#ifndef FRNN_FW_BULK
#define FRNN_FW_BULK 0
#endif
#ifndef FRNN_FW_FLAGBAR
#define FRNN_FW_FLAGBAR 0
#endif
#ifndef FRNN_FW_SMEMPART
#define FRNN_FW_SMEMPART 0
#endif
constexpr bool BULK = FRNN_FW_BULK != 0, FLAGBAR = FRNN_FW_FLAGBAR != 0, SMEMPART = FRNN_FW_SMEMPART != 0;
#if FRNN_FW_SMEMPART && FRNN_Q != 1
#error "FRNN_FW_SMEMPART needs Q = 1"
#endif
constexpr int V_CHECK = 1, V_DELAY = 2, V_NO_B1 = 4, V_NO_B2 = 8, V_PROF = 16;
constexpr int D = FRNN_D, PC = FRNN_PC, Q = FRNN_Q, BM = FRNN_BM, CR = FRNN_CR;
constexpr bool SLSTM = FRNN_SLSTM != 0;
constexpr int S = SLSTM ? 4 : 2;  // state components h, c (, n, m)
constexpr int THREADS = 256, WARPS = 8;
constexpr int NU = D / PC, NC = 4 * NU, NCW = NC / WARPS, NTW = NCW / 8, KQ = D / Q, KS = KQ / 16, MT = BM / 16;
constexpr int RPQ = BM / Q, PE = (RPQ * NU + THREADS - 1) / THREADS;
constexpr int KC = 64, LDS = KC + 8, ROUNDS = KQ / KC, J = KC / 16, CSM = KS - CR;
constexpr int LDSB = KQ + 8;  // FRNN_FW_BULK: padded row stride of the staged K block (elements)
static_assert(!BULK || (KQ * 2) % 16 == 0, "bulk copies move multiples of 16 bytes");
constexpr int SMEM_R = WARPS * CSM * NTW * 32 * 8;
constexpr int SMEM_STAGE = BULK ? BM * LDSB * 2 : STAGES * BM * LDS * 2;
constexpr int SMEM_PART = SMEMPART ? BM * NC * 4 : 0;
static_assert(D % PC == 0 && NU % 16 == 0, "unit block must be a multiple of 16 (a warp's columns within one gate)");
static_assert(D % Q == 0 && KQ % KC == 0, "K block must be a multiple of 64");
static_assert(BM % Q == 0, "row ownership needs BM a multiple of Q");
static_assert(BM == 16 || BM == 32 || BM == 64, "BM must be 16, 32 or 64");
static_assert(CR >= 0 && CR <= KS, "register k-steps");
static_assert(MODE >= 0 && MODE <= 2, "output mode");

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void delay(int seed, int a, int b, int c) {
  uint32_t h = (uint32_t)seed * 2654435761u ^ (uint32_t)a * 40503u ^ (uint32_t)b * 2246822519u ^ (uint32_t)c * 3266489917u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  __nanosleep(h % 4000u);
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

#if FRNN_FW_PREFETCH
// Split grid barrier: the protocol of cooperative groups' sync_grids_arrive / sync_grids_wait.
__device__ __forceinline__ unsigned grid_arrive(unsigned* ctr) {
  __syncthreads();
  unsigned old = 0;
  if (threadIdx.x == 0) {
    const unsigned expected = gridDim.x * gridDim.y * gridDim.z;
    const unsigned nb = blockIdx.x + blockIdx.y + blockIdx.z == 0 ? 0x80000000u - (expected - 1) : 1u;
    asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;" : "=r"(old) : "l"(ctr), "r"(nb) : "memory");
  }
  return old;
}

__device__ __forceinline__ void grid_wait(unsigned old, unsigned* ctr) {
  if (threadIdx.x == 0) {
    unsigned cur;
    do {
      asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(ctr) : "memory");
    } while (((old ^ cur) & 0x80000000u) == 0);
  }
  __syncthreads();
}
#endif

#if FRNN_FW_BULK
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
__device__ __forceinline__ void bulk_copy(void* dst, const void* src, unsigned bytes, unsigned long long* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
#endif

struct Args {
  const elem_t* wx;      // [B, T, 4, H, D]
  const elem_t* R;       // [4, H, D, D]: R[g, head, o, k]
  const elem_t* bias;    // [4, H, D]
  const float* initial;  // [S, B, 1, H, D] FP32
  snap_t* out;           // full [S, B, T, H, D]; hidden [B, T, H, D]; last [B, 1, H, D]
  float* final;          // [S, B, 1, H, D] FP32 carry
  elem_t* xbuf;          // [RESIDENT, BM, D] exchanged round(h) (single buffer)
  float* part;           // [RESIDENT, Q, BM, 4D] FP32 partial gate preactivations (single buffer)
  int* stamp;            // [RESIDENT, PC, Q] verification step stamps (+ 1 B2 counter with FRNN_FW_PREFETCH,
                         // + one flag per CTA with FRNN_FW_FLAGBAR)
  int* err;              // [12] verification counters / phase timers, [3] delay seed
  int B, T, H, groups, resident, slots, tiles;
  const bool* zero_first;  // sLSTM: device flag, normalizer state zero at the call's first step (nullptr: false)
};

// B fragment register for k (even) and k + 1, output column (g, o): R[g, head, o, k .. k + 1] (contiguous).
__device__ __forceinline__ uint32_t r_pair(const elem_t* R, int H, int head, int g, int o, int k) {
  return *reinterpret_cast<const uint32_t*>(R + ((size_t)(g * H + head) * D + o) * D + k);
}

__global__ void __launch_bounds__(THREADS, 1) frnn_hybrid_fw2d_kernel(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint2* sr = reinterpret_cast<uint2*>(smem);  // [WARPS][CSM][NTW][32] B fragments
  elem_t* stage = reinterpret_cast<elem_t*>(smem + SMEM_R);  // [STAGES][BM][LDS], or [BM][LDSB] with FRNN_FW_BULK
  float* spart = reinterpret_cast<float*>(smem + SMEM_R + SMEM_STAGE);  // FRNN_FW_SMEMPART: [BM][NC] gate partials
#if FRNN_FW_BULK
  __shared__ __align__(8) unsigned long long bulk_bar;
  unsigned bulk_phase = 0;
  if (threadIdx.x == 0) {
    mbar_init(&bulk_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
#endif
  cg::grid_group grid = cg::this_grid();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, fg = lane >> 2, ft = lane & 3;
  const int cb = blockIdx.x / Q, kb = blockIdx.x % Q, gy = blockIdx.y;
  // Read on the device, so the caller never synchronizes on it (CUDA-graph capturable).
  const bool zero_first = SLSTM && a.zero_first != nullptr && *a.zero_first;
  const int c_warp = warp * NCW;     // first local output column of this warp's MMA tile
  const int g_warp = c_warp / NU;    // its gate
  const int o_warp = cb * NU + c_warp % NU;  // its first unit
  const int k0 = kb * KQ;            // first hidden index of the K block
  const size_t hd = (size_t)a.H * D, state_stride = (size_t)a.B * hd, out_stride = (size_t)a.B * a.T * hd;
  elem_t* xb = a.xbuf + (size_t)gy * BM * D;
  float* pt = a.part + (size_t)gy * Q * BM * 4 * D;
  int* const stamps = a.stamp + ((size_t)gy * PC + cb) * Q;
#if FRNN_FW_PREFETCH
  unsigned* const b2ctr = reinterpret_cast<unsigned*>(a.stamp + (size_t)a.resident * PC * Q);
#endif
#if FRNN_FW_FLAGBAR
  // One flag per CTA of the whole grid, after the stamps (and the B2 counter, if any).
  int* const flags = a.stamp + (size_t)a.resident * PC * Q + (FRNN_FW_PREFETCH ? 1 : 0);
  const int nblocks = gridDim.x * gridDim.y, me = blockIdx.y * gridDim.x + blockIdx.x;
#endif
  uint32_t rreg[CR > 0 ? CR : 1][NTW][2];
  int cnt = 0;  // steps so far over all slots (stamp value)

  for (int slot = 0; slot < a.slots; ++slot) {
    const int group = slot * a.resident + gy;
    const bool active = group < a.groups;
    const int g = active ? group : a.groups - 1;
    const int head = g / a.tiles, bt = g % a.tiles;
    if (active) {
#pragma unroll
      for (int c = 0; c < CR; ++c)
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
          int k = k0 + c * 16 + 2 * ft, o = o_warp + nt * 8 + fg;
          rreg[c][nt][0] = r_pair(a.R, a.H, head, g_warp, o, k);
          rreg[c][nt][1] = r_pair(a.R, a.H, head, g_warp, o, k + 8);
        }
      for (int c = 0; c < CSM; ++c)
        for (int nt = 0; nt < NTW; ++nt) {
          int k = k0 + (CR + c) * 16 + 2 * ft, o = o_warp + nt * 8 + fg;
          sr[(((size_t)warp * CSM + c) * NTW + nt) * 32 + lane] =
              make_uint2(r_pair(a.R, a.H, head, g_warp, o, k), r_pair(a.R, a.H, head, g_warp, o, k + 8));
        }
    }

    // Pointwise ownership: element e of this thread is (row kb * RPQ + idx / NU, unit cb * NU + idx % NU); fixed for
    // the whole slot, so the cell state stays in registers.
    auto elem = [&](int e, int& lrow, int& o) -> bool {
      const int idx = tid + THREADS * e;
      lrow = kb * RPQ + idx / NU;
      o = cb * NU + idx % NU;
      return idx < RPQ * NU;
    };
    float hs[PE], cs[PE], ns[PE], ms[PE];
#pragma unroll
    for (int e = 0; e < PE; ++e) {
      int lrow, o;
      const bool own = elem(e, lrow, o);
      const int row = bt * BM + lrow;
      const bool valid = active && own && row < a.B;
      const size_t unit = ((size_t)row * a.H + head) * D + o;
      hs[e] = valid ? a.initial[unit] : 0.f;
      cs[e] = valid ? a.initial[state_stride + unit] : 0.f;
      ns[e] = (SLSTM && valid) ? a.initial[2 * state_stride + unit] : 0.f;
      ms[e] = (SLSTM && valid) ? a.initial[3 * state_stride + unit] : 0.f;
      if (active && own) xb[(size_t)lrow * D + o] = to_elem(hs[e]);  // round(h_0); rows beyond B publish zeros
    }
    __syncthreads();

    // GEMM of the K block against the staged round(h_{t-1}) (rows of tile bt) -> partial gates; returns A-check count.
    auto gemm = [&](int t) -> int {
      float acc[MT][NTW][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[mt][nt][q] = 0.f;
      int bad = 0;
      auto load = [&](int r, int buf) {
        for (int i = tid; i < BM * KC / 8; i += THREADS) {
          int row = i / (KC / 8), col8 = i % (KC / 8);
          const uint4* src = reinterpret_cast<const uint4*>(xb + (size_t)row * D + k0 + r * KC) + col8;
          elem_t* dst = stage + ((size_t)buf * BM + row) * LDS + col8 * 8;
          const unsigned sa = (unsigned)__cvta_generic_to_shared(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(src) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      };
      // Expected A element: round(h_{t-1}[row, k]) from the stored output (t > 0) or the initial state.
      auto expect = [&](int row, int k) -> elem_t {
        if (row >= a.B) return to_elem(0.f);
        if (t == 0) return to_elem(a.initial[((size_t)row * a.H + head) * D + k]);
        return to_elem(from_snap(a.out[(((size_t)row * a.T + t - 1) * a.H + head) * D + k]));
      };
      auto check = [&](uint32_t v, int row, int k) -> int {
        return v != pack(expect(row, k), expect(row, k + 1));
      };
      const uint2* sb = sr + (size_t)warp * CSM * NTW * 32 + lane;
      auto round = [&](int r, auto in_regs) {
        if constexpr (STAGES == 2) {
          if (r + 1 < ROUNDS) load(r + 1, (r + 1) & 1);
        } else {
          if (r + STAGES - 1 < ROUNDS) load(r + STAGES - 1, (r + STAGES - 1) % STAGES);
          else asm volatile("cp.async.commit_group;" ::: "memory");  // empty group: keeps the wait count uniform
        }
        const elem_t* tile = stage + (size_t)(STAGES == 2 ? (r & 1) : (r % STAGES)) * BM * LDS;
#pragma unroll
        for (int jj = 0; jj < J; ++jj) {
          const int c = r * J + jj;
          uint32_t b[NTW][2];
#pragma unroll
          for (int nt = 0; nt < NTW; ++nt) {
            if (decltype(in_regs)::value && c < CR) {
              const int cc = decltype(in_regs)::value ? (c < CR ? c : 0) : 0;
              b[nt][0] = rreg[cc][nt][0];
              b[nt][1] = rreg[cc][nt][1];
            } else {
              uint2 x = sb[((size_t)(c - CR) * NTW + nt) * 32];
              b[nt][0] = x.x;
              b[nt][1] = x.y;
            }
          }
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const elem_t* r0 = tile + (size_t)(mt * 16 + fg) * LDS + jj * 16 + 2 * ft;
            const elem_t* r8 = r0 + 8 * LDS;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(r0), a1 = *reinterpret_cast<const uint32_t*>(r8);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(r0 + 8), a3 = *reinterpret_cast<const uint32_t*>(r8 + 8);
            if ((VERIFY & V_CHECK) && MODE != 2) {
              const int row0 = bt * BM + mt * 16 + fg, k = k0 + c * 16 + 2 * ft;
              bad += check(a0, row0, k) + check(a1, row0 + 8, k) + check(a2, row0, k + 8) +
                     check(a3, row0 + 8, k + 8);
            }
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) mma(acc[mt][nt], a0, a1, a2, a3, b[nt][0], b[nt][1]);
          }
        }
        if constexpr (STAGES == 2) {
          asm volatile("cp.async.wait_group 0;" ::: "memory");  // this thread's copies of round r + 1
        } else {
          asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");  // round r + 1 (later ones in flight)
        }
        __syncthreads();  // round r + 1 staged, every warp done with round r's buffer
      };
      constexpr int RREG = (CR + J - 1) / J;  // rounds with at least one register k-step
#if FRNN_FW_BULK
      // Whole K block in one asynchronous bulk transaction: BM row copies of KQ elements into the padded layout, one
      // mbarrier phase. Generic reads of the previous step ended before this step's B2 (bar.sync inside it).
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (tid == 0) mbar_expect(&bulk_bar, (unsigned)(BM * KQ * sizeof(elem_t)));
      __syncthreads();
      if (tid < BM) bulk_copy(stage + (size_t)tid * LDSB, xb + (size_t)tid * D + k0, KQ * sizeof(elem_t), &bulk_bar);
      mbar_wait(&bulk_bar, bulk_phase);
      bulk_phase ^= 1;
#pragma unroll
      for (int c = 0; c < KS; ++c) {
        uint32_t b[NTW][2];
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
          if (c < CR) {
            b[nt][0] = rreg[c < CR ? c : 0][nt][0];
            b[nt][1] = rreg[c < CR ? c : 0][nt][1];
          } else {
            uint2 x = sb[((size_t)(c - CR) * NTW + nt) * 32];
            b[nt][0] = x.x;
            b[nt][1] = x.y;
          }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const elem_t* r0 = stage + (size_t)(mt * 16 + fg) * LDSB + c * 16 + 2 * ft;
          const elem_t* r8 = r0 + 8 * LDSB;
          uint32_t a0 = *reinterpret_cast<const uint32_t*>(r0), a1 = *reinterpret_cast<const uint32_t*>(r8);
          uint32_t a2 = *reinterpret_cast<const uint32_t*>(r0 + 8), a3 = *reinterpret_cast<const uint32_t*>(r8 + 8);
          if ((VERIFY & V_CHECK) && MODE != 2) {
            const int row0 = bt * BM + mt * 16 + fg, k = k0 + c * 16 + 2 * ft;
            bad += check(a0, row0, k) + check(a1, row0 + 8, k) + check(a2, row0, k + 8) +
                   check(a3, row0 + 8, k + 8);
          }
#pragma unroll
          for (int nt = 0; nt < NTW; ++nt) mma(acc[mt][nt], a0, a1, a2, a3, b[nt][0], b[nt][1]);
        }
      }
#else
      if constexpr (STAGES == 2) {
        load(0, 0);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
      } else {
#pragma unroll
        for (int s0 = 0; s0 < STAGES - 1; ++s0) {
          if (s0 < ROUNDS) load(s0, s0);
          else asm volatile("cp.async.commit_group;" ::: "memory");
        }
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");  // round 0 staged
      }
      __syncthreads();
#pragma unroll
      for (int r = 0; r < RREG; ++r) round(r, std::true_type{});
#pragma unroll 1
      for (int r = RREG; r < ROUNDS; ++r) round(r, std::false_type{});
#endif
      // Partial gates of this K block for the warp's columns (gate g_warp, units o_warp ..), all BM rows.
      if constexpr (SMEMPART) {  // [BM][NC]: column g_warp * NU + (o - cb * NU) = c_warp + nt * 8 + 2 ft
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NTW; ++nt) {
            const int r = mt * 16 + fg, c = c_warp + nt * 8 + 2 * ft;
            *reinterpret_cast<float2*>(spart + (size_t)r * NC + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            *reinterpret_cast<float2*>(spart + (size_t)(r + 8) * NC + c) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
          }
      } else {
        float* dst = pt + (size_t)kb * BM * 4 * D + (size_t)g_warp * D;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NTW; ++nt) {
            const int r = mt * 16 + fg, o = o_warp + nt * 8 + 2 * ft;
            *reinterpret_cast<float2*>(dst + (size_t)r * 4 * D + o) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            *reinterpret_cast<float2*>(dst + (size_t)(r + 8) * 4 * D + o) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
          }
      }
      if (VERIFY & V_CHECK) {  // step stamp after the partial (release at GPU scope)
        __syncthreads();
        if (tid == 0) {
          __threadfence();
          asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(stamps + kb), "r"(cnt) : "memory");
        }
      }
      return bad;
    };

#if FRNN_FW_PREFETCH
    elem_t wxr[PE][4];  // this step's input projections of the owned elements, loaded during B2
#endif
    for (int t = 0; t < a.T; ++t) {
      const bool last = t == a.T - 1;
      unsigned long long t0 = (VERIFY & V_PROF) ? gtime() : 0;
#if FRNN_FW_FLAGBAR
      {
        const bool b2 = !(VERIFY & V_NO_B2) || last;
        __syncthreads();  // every thread's xbuf writes of the previous step
        if (b2 && tid == 0) {
          __threadfence();
          asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(flags + me), "r"(cnt + 1) : "memory");
        }
#if FRNN_FW_PREFETCH
#pragma unroll
        for (int e = 0; e < PE; ++e) {
          int lrow, o;
          const bool own = elem(e, lrow, o);
          const int row = bt * BM + lrow;
          if (active && own && row < a.B) {
            const size_t xo = (((size_t)row * a.T + t) * 4 * a.H + head) * D + o;
#pragma unroll
            for (int q = 0; q < 4; ++q) wxr[e][q] = a.wx[xo + (size_t)q * hd];
          }
        }
#endif
        if (b2) {
          for (int i = tid; i < nblocks; i += THREADS) {
            int v;
            do {
              asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flags + i) : "memory");
            } while (v < cnt + 1);
          }
          __syncthreads();  // B2: every CTA's round(h_{t-1}) complete in xbuf
        }
      }
#elif FRNN_FW_PREFETCH
      {
        const bool b2 = !(VERIFY & V_NO_B2) || last;
        const unsigned old = b2 ? grid_arrive(b2ctr) : 0u;
#pragma unroll
        for (int e = 0; e < PE; ++e) {
          int lrow, o;
          const bool own = elem(e, lrow, o);
          const int row = bt * BM + lrow;
          if (active && own && row < a.B) {
            const size_t xo = (((size_t)row * a.T + t) * 4 * a.H + head) * D + o;
#pragma unroll
            for (int q = 0; q < 4; ++q) wxr[e][q] = a.wx[xo + (size_t)q * hd];
          }
        }
        if (b2) grid_wait(old, b2ctr);  // B2: round(h_{t-1}) complete in xbuf
      }
#else
      if (!(VERIFY & V_NO_B2) || last) grid.sync();  // B2: round(h_{t-1}) complete in xbuf
#endif
      unsigned long long t1 = (VERIFY & V_PROF) ? gtime() : 0;
      if (VERIFY & V_DELAY) delay(a.err[3], blockIdx.x + 977, g, t);
      ++cnt;
      if (active) {
        int bad = gemm(t);
        if ((VERIFY & V_CHECK) && bad) atomicAdd(a.err + 1, bad);
      }
      unsigned long long t2 = (VERIFY & V_PROF) ? gtime() : 0;
      if constexpr (Q == 1) {
        __syncthreads();  // B1: the partials are this CTA's own (1-D split)
      } else {
        if (!(VERIFY & V_NO_B1) || last) grid.sync();  // B1: partials complete
      }
      unsigned long long t3 = (VERIFY & V_PROF) ? gtime() : 0;
      if (VERIFY & V_DELAY) delay(a.err[3], blockIdx.x, g, t);
      if (active) {
        if (VERIFY & V_CHECK) {
          int stale = 0;
          for (int jj = 0; jj < Q; ++jj) {
            int s;
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(s) : "l"(stamps + jj) : "memory");
            stale += s != cnt;
          }
          if (stale && tid == 0) atomicAdd(a.err + 2, stale);
        }
#pragma unroll
        for (int e = 0; e < PE; ++e) {
          int lrow, o;
          const bool own = elem(e, lrow, o);
          if (!own) continue;
          const int row = bt * BM + lrow;
          float gate[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            float sum = 0.f;
            if constexpr (SMEMPART) {
              sum = spart[(size_t)lrow * NC + (size_t)q * NU + (o - cb * NU)];
            } else {
              for (int jj = 0; jj < Q; ++jj) sum += __ldcg(pt + ((size_t)jj * BM + lrow) * 4 * D + (size_t)q * D + o);
            }
            gate[q] = sum;
          }
          float h = 0.f;
          if (row < a.B) {
            const size_t xo = (((size_t)row * a.T + t) * 4 * a.H + head) * D + o;
#pragma unroll
            for (int q = 0; q < 4; ++q)
#if FRNN_FW_PREFETCH
              gate[q] += from_elem(wxr[e][q]) + from_elem(a.bias[((size_t)q * a.H + head) * D + o]);
#else
              gate[q] += from_elem(a.wx[xo + (size_t)q * hd]) + from_elem(a.bias[((size_t)q * a.H + head) * D + o]);
#endif
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
          if (!last) xb[(size_t)lrow * D + o] = to_elem(h);  // rows beyond B publish zeros
        }
      }
      if ((VERIFY & V_PROF) && tid == 0 && active) {
        unsigned long long t4 = gtime();
        atomicAdd(a.err + 4, (int)((t1 - t0) >> 6));
        atomicAdd(a.err + 5, (int)((t2 - t1) >> 6));
        atomicAdd(a.err + 6, (int)((t3 - t2) >> 6));
        atomicAdd(a.err + 7, (int)((t4 - t3) >> 6));
        atomicAdd(a.err + 9, 1);
      }
    }
    // Final FP32 carry.
    if (active) {
#pragma unroll
      for (int e = 0; e < PE; ++e) {
        int lrow, o;
        const bool own = elem(e, lrow, o);
        const int row = bt * BM + lrow;
        const size_t unit = ((size_t)row * a.H + head) * D + o;
        if (own && row < a.B) {
          a.final[unit] = hs[e];
          a.final[state_stride + unit] = cs[e];
          if (SLSTM) {
            a.final[2 * state_stride + unit] = ns[e];
            a.final[3 * state_stride + unit] = ms[e];
          }
        }
      }
    }
    // The next slot's prologue writes xbuf: every GEMM read of this slot ended before the last B1.
    __syncthreads();
  }
}

extern "C" int frnn_hybrid_fw2d_smem() {
  return SMEM_R + SMEM_STAGE + SMEM_PART;
}

extern "C" int frnn_hybrid_fw2d_ctas() { return PC * Q; }

extern "C" int frnn_hybrid_fw2d(const void* wx, const void* R, const void* bias, const float* initial, void* out,
                                float* final, void* xbuf, float* part, int* stamp, int* err, int B, int T, int H,
                                int groups, int resident, int slots, int tiles, const void* zero_first,
                                void* stream) {
  Args a{reinterpret_cast<const elem_t*>(wx), reinterpret_cast<const elem_t*>(R),
         reinterpret_cast<const elem_t*>(bias), initial, reinterpret_cast<snap_t*>(out), final,
         reinterpret_cast<elem_t*>(xbuf), part, stamp, err, B, T, H, groups, resident, slots, tiles,
         reinterpret_cast<const bool*>(zero_first)};
  int smem = frnn_hybrid_fw2d_smem();
  cudaError_t e = cudaFuncSetAttribute(frnn_hybrid_fw2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return (int)e;
  void* args[] = {&a};
  e = cudaLaunchCooperativeKernel((void*)frnn_hybrid_fw2d_kernel, dim3(PC * Q, resident), dim3(THREADS), args, smem,
                                  (cudaStream_t)stream);
  return (int)e;
}
