// Large-D backward of the FlashRNN2 R-tiled recurrence (LSTM / sLSTM), register + shared-memory residency of R.
//
// Same contract and arithmetic as the Triton backward kernel (triton_rtiled_bw._rtiled_bw, persistent mode, plain
// variant): CTA s owns S = 16 input columns p of h; every step it reduces dh_t[:, p] = sum_{g,o} dG_{t+1}[:, g, o]
// R[g, o, p] over K = 4D, runs the pointwise cell backward for its columns, stores dG_t (FP32) and publishes the
// rounded dG_t into a double-buffered exchange buffer, then synchronizes the cooperative grid.
//
// The K = 4D reduction is split evenly over the 8 warps (KW = D / 2 rows each). Warp w keeps the first KRW of its
// rows of R[:, :, p] pre-arranged as mma.sync B fragments in registers and the rest in shared memory (fragment
// order, conflict-free 8-byte loads); the dG_{t+1} operand is read straight from the exchange buffer into A
// fragments with L2-only loads (ld.global.cg: the buffer is written by other SMs inside this kernel). The 8 partial
// [16 x 16] sums meet in shared memory. Compile-time parameters: FRNN_D, FRNN_KRW, FRNN_SLSTM, FRNN_BF16.
//
// Race verification (FRNN_VERIFY, compiled out when 0): bit 1 content check (every A-fragment element read from the
// exchange buffer is compared with the sender's stored FP32 dG, rounded the same way, after the read and again after
// the MMAs: counters err[1], err[2]); bit 2 seeded delays of 0-4 us before the publish and after the grid barrier;
// bit 4 broken: single buffer; bit 8 broken: no grid barrier (except before the tail); bit 16 phase timing
// (globaltimer, thread 0 of every CTA, units of 64 ns summed into err[4..7]: reduction = exchange loads + MMAs + partial
// sums, pointwise + dG stores + publish, grid barrier, steps).
#include <cooperative_groups.h>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace cg = cooperative_groups;

#if FRNN_BF16
typedef __nv_bfloat16 elem_t;
#define FRNN_MMA "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
__device__ __forceinline__ elem_t to_elem(float x) { return __float2bfloat16_rn(x); }
#else
typedef __half elem_t;
#define FRNN_MMA "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
__device__ __forceinline__ elem_t to_elem(float x) { return __float2half_rn(x); }
#endif

#ifndef FRNN_VERIFY
#define FRNN_VERIFY 0
#endif
constexpr int VERIFY = FRNN_VERIFY;
constexpr int V_CHECK = 1, V_DELAY = 2, V_SINGLE = 4, V_NO_SYNC = 8, V_PROF = 16, V_CL_SINGLE = 32, V_CL_NO_SYNC = 64;
constexpr int D = FRNN_D;
constexpr int KRW = FRNN_KRW;
constexpr bool SLSTM = FRNN_SLSTM != 0;
#ifndef FRNN_BM
#define FRNN_BM 16
#endif
// BM batch rows per CTA (16 / 32 / 64): MT = BM / 16 mma row tiles per warp share every R fragment; each of the 256
// threads runs PE = BM * 16 / 256 pointwise elements (rows pb + 16 e, column pj).
constexpr int S = 16, BM = FRNN_BM, WARPS = 8, THREADS = 256, MT = BM / 16, PE = BM * S / THREADS;
static_assert(BM == 16 || BM == 32 || BM == 64, "BM must be 16, 32 or 64");
#ifndef FRNN_CLUSTER
#define FRNN_CLUSTER 0
#endif
// FRNN_CLUSTER = 8: thread-block clusters of 8 neighbouring CTAs share the exchanged dG operand through distributed
// shared memory. Per round of KC_STAGE rows of every warp's K range, cluster rank q loads warp q's [BM x KC_STAGE]
// chunk from L2 with 128-bit loads into its shared stage (a symmetric buffer: same offset in every CTA, double
// buffered, aliasing the partial-sum buffer), the cluster synchronizes once, and warp w of every CTA reads its A
// fragments from rank w's stage. L2 reads of the exchange drop by 8; one cluster barrier per round.
constexpr int CLUSTER = FRNN_CLUSTER, KC_STAGE = 64, LDS_STAGE = KC_STAGE + 8;
static_assert(CLUSTER == 0 || CLUSTER == WARPS, "cluster rank q stages warp q's chunk");
constexpr int K = 4 * D;
constexpr int KW = K / WARPS;
constexpr int KSW = KW - KRW;
constexpr int CR = KRW / 16, CS = KSW / 16;
static_assert(D % 16 == 0 && KW % 16 == 0 && KRW % 16 == 0 && KSW >= 0, "bad tiling");

__device__ __forceinline__ uint32_t pack(elem_t lo, elem_t hi) {
  uint32_t r;
  uint16_t a = *reinterpret_cast<uint16_t*>(&lo), b = *reinterpret_cast<uint16_t*>(&hi);
  r = (uint32_t)a | ((uint32_t)b << 16);
  return r;
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile(FRNN_MMA " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

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

// Content check of one A-fragment register (two elements, row r, reduction index k) against the stored dG of step t+1.
__device__ __forceinline__ int check_pair(uint32_t v, const float* dg, int B, int T, int H, int head, int row, int t1,
                                          int k) {
  if (row >= B) return 0;
  int g = k / D, o = k % D;
  size_t at = (((size_t)row * T + t1) * 4 * H + (size_t)g * H + head) * D + o;
  uint32_t want = pack(to_elem(dg[at]), to_elem(dg[at + 1]));
  return v != want;
}

// B fragment register of R for k (even) and k + 1, column n: R[g, head, o, n] with k = g D + o.
__device__ __forceinline__ uint32_t r_pair(const elem_t* R, int H, int head, int k, int n) {
  int g = k / D, o = k % D;
  size_t base = ((size_t)(g * H + head) * D + o) * D + n;
  return pack(R[base], R[base + D]);
}

struct Args {
  const float* pre;      // [B, T, 4, H, D] FP32 gate preactivations
  const float* states;   // [S, B, T, H, D] FP32 forward state history
  const float* initial;  // [S, B, 1, H, D] FP32
  const float* dstates;  // [S, B, T, H, D] FP32 history gradient
  const float* dfinal;   // [S, B, 1, H, D] FP32 final-carry gradient
  const elem_t* R;       // [4, H, D, D]
  float* dg;             // [B, T, 4, H, D] FP32 out
  float* dinit;          // [S, B, 1, H, D] FP32 out
  elem_t* xbuf;          // [RESIDENT, 2, BM, 4D] exchange
  int* err;              // verification: [0] unused, [1] after read, [2] after MMA, [3] delay seed, [4..7] phase
                         // timers, [8] cluster stage reads (A operands read through distributed shared memory)
  int B, T, H, groups, resident, slots, tiles;
};

__global__ void __launch_bounds__(THREADS, 1) frnn_hybrid_bw_kernel(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint32_t* sr = reinterpret_cast<uint32_t*>(smem);                                   // [WARPS][CS][2][32][2]
  float* red = reinterpret_cast<float*>(smem + (size_t)WARPS * CS * 2 * 32 * 2 * 4);  // [WARPS][BM][16]
  cg::grid_group grid = cg::this_grid();
  const int s = blockIdx.x, gy = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int fg = lane >> 2, ft = lane & 3;  // mma fragment row group / thread-in-group
  const int pb = tid >> 4, pj = tid & 15;   // pointwise element (batch row pb + 16 e, column pj)
  const int p = s * S + pj;
  const size_t hd = (size_t)a.H * D, state_stride = (size_t)a.B * hd, history_stride = (size_t)a.B * a.T * hd;
  uint32_t rreg[CR > 0 ? CR : 1][2][2];
  int cnt = 0;  // publishes so far (parity of the exchange buffer)

  for (int slot = 0; slot < a.slots; ++slot) {
    const int group = slot * a.resident + gy;
    const bool active = group < a.groups;
    const int g = active ? group : a.groups - 1;
    const int head = g / a.tiles, bt = g % a.tiles;
    if (active) {
      // R[:, :, p] of this CTA: registers for the first KRW rows of every warp's K range, shared memory for the rest.
#pragma unroll
      for (int c = 0; c < CR; ++c)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          int k = warp * KW + c * 16 + 2 * ft, n = s * S + nt * 8 + fg;
          rreg[c][nt][0] = r_pair(a.R, a.H, head, k, n);
          rreg[c][nt][1] = r_pair(a.R, a.H, head, k + 8, n);
        }
      for (int c = 0; c < CS; ++c)
        for (int nt = 0; nt < 2; ++nt) {
          int k = warp * KW + KRW + c * 16 + 2 * ft, n = s * S + nt * 8 + fg;
          uint32_t* dst = sr + ((((size_t)warp * CS + c) * 2 + nt) * 32 + lane) * 2;
          dst[0] = r_pair(a.R, a.H, head, k, n);
          dst[1] = r_pair(a.R, a.H, head, k + 8, n);
        }
    }
    __syncthreads();

    float dc[PE], dn[PE], dh_in[PE];
#pragma unroll
    for (int e = 0; e < PE; ++e) dc[e] = dn[e] = dh_in[e] = 0.f;

    // Content check of the A elements this thread reads (rows mt*16 + fg / + 8 of the tile).
    auto check = [&](const elem_t* xa, int t1) -> int {
      int bad = 0;
      for (int mt = 0; mt < MT; ++mt) {
        const uint32_t* r0 = reinterpret_cast<const uint32_t*>(xa + (size_t)(mt * 16 + fg) * K + 2 * ft);
        const uint32_t* r8 = reinterpret_cast<const uint32_t*>(xa + (size_t)(mt * 16 + fg + 8) * K + 2 * ft);
        int row0 = bt * BM + mt * 16 + fg;
        for (int c = 0; c < CR + CS; ++c) {
          int k = warp * KW + c * 16 + 2 * ft;
          bad += check_pair(__ldcg(r0 + c * 8), a.dg, a.B, a.T, a.H, head, row0, t1, k);
          bad += check_pair(__ldcg(r8 + c * 8), a.dg, a.B, a.T, a.H, head, row0 + 8, t1, k);
          bad += check_pair(__ldcg(r0 + c * 8 + 4), a.dg, a.B, a.T, a.H, head, row0, t1, k + 8);
          bad += check_pair(__ldcg(r8 + c * 8 + 4), a.dg, a.B, a.T, a.H, head, row0 + 8, t1, k + 8);
        }
      }
      return bad;
    };

    // dh from the exchanged, rounded dG of every CTA of the group (parity par); all 256 threads take part.
    auto reduce = [&](int par, int t1) {
      float acc[MT][2][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[mt][nt][q] = 0.f;
      const elem_t* xa = a.xbuf + ((size_t)(gy * 2 + par) * BM) * K + warp * KW;
      if (VERIFY & V_CHECK) {
        int bad = check(xa, t1);
        if (bad) atomicAdd(a.err + 1, bad);
      }
#if FRNN_CLUSTER
      {
        cg::cluster_group cl = cg::this_cluster();
        const int rank = (int)cl.block_rank();
        elem_t* stage = reinterpret_cast<elem_t*>(red);  // [2][BM][LDS_STAGE], symmetric across the cluster
        const elem_t* src = a.xbuf + ((size_t)(gy * 2 + par) * BM) * K + rank * KW;  // warp `rank`'s K range
        constexpr int R = KW / KC_STAGE, J = KC_STAGE / 16;
        static_assert(KW % KC_STAGE == 0, "KW must be a multiple of the stage chunk");
        auto load = [&](int r, int buf) {
          for (int i = tid; i < BM * KC_STAGE / 8; i += THREADS) {
            int row = i / (KC_STAGE / 8), col8 = i % (KC_STAGE / 8);
            uint4 v = __ldcg(reinterpret_cast<const uint4*>(src + (size_t)row * K + r * KC_STAGE) + col8);
            *reinterpret_cast<uint4*>(stage + ((size_t)buf * BM + row) * LDS_STAGE + col8 * 8) = v;
          }
        };
        const uint2* sb = reinterpret_cast<const uint2*>(sr) + (size_t)warp * CS * 2 * 32 + lane;
        const elem_t* remote = cl.map_shared_rank(stage, warp);
        // One round: prefetch round r+1 into the other buffer, MMA round r from rank `warp`'s stage, cluster barrier
        // (round r+1 staged everywhere, and every CTA is done reading round r's buffer). Rounds that touch register
        // rows of R are unrolled (static register indices); the shared-memory rounds stay rolled (register pressure).
        constexpr bool ONE_BUF = (VERIFY & V_CL_SINGLE) != 0;  // broken variant: one stage buffer
        int stage_bad = 0;
        auto round = [&](int r, auto in_regs) {
          if (VERIFY & V_DELAY) delay(a.err[3], s + 3001 + 17 * r, gy, t1);
          if (r + 1 < R) load(r + 1, ONE_BUF ? 0 : (r + 1) & 1);
          if (VERIFY & V_DELAY) delay(a.err[3], s + 5003 + 17 * r, gy, t1);
          const elem_t* tile = remote + (size_t)(ONE_BUF ? 0 : (r & 1)) * BM * LDS_STAGE;
#pragma unroll
          for (int jj = 0; jj < J; ++jj) {
            const int c = r * J + jj;
            uint32_t b00, b01, b10, b11;
            if (decltype(in_regs)::value && c < CR) {
              const int cc = decltype(in_regs)::value ? (c < CR ? c : 0) : 0;
              b00 = rreg[cc][0][0]; b01 = rreg[cc][0][1]; b10 = rreg[cc][1][0]; b11 = rreg[cc][1][1];
            } else {
              uint2 x0 = sb[((c - CR) * 2) * 32], x1 = sb[((c - CR) * 2 + 1) * 32];
              b00 = x0.x; b01 = x0.y; b10 = x1.x; b11 = x1.y;
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              const elem_t* r0 = tile + (size_t)(mt * 16 + fg) * LDS_STAGE + jj * 16 + 2 * ft;
              const elem_t* r8 = r0 + 8 * LDS_STAGE;
              uint32_t a0 = *reinterpret_cast<const uint32_t*>(r0), a1 = *reinterpret_cast<const uint32_t*>(r8);
              uint32_t a2 = *reinterpret_cast<const uint32_t*>(r0 + 8), a3 = *reinterpret_cast<const uint32_t*>(r8 + 8);
              if (VERIFY & V_CHECK) {  // the operands as read through DSMEM against the stored dG of step t1
                const int row0 = bt * BM + mt * 16 + fg, k = warp * KW + c * 16 + 2 * ft;
                stage_bad += check_pair(a0, a.dg, a.B, a.T, a.H, head, row0, t1, k) +
                             check_pair(a1, a.dg, a.B, a.T, a.H, head, row0 + 8, t1, k) +
                             check_pair(a2, a.dg, a.B, a.T, a.H, head, row0, t1, k + 8) +
                             check_pair(a3, a.dg, a.B, a.T, a.H, head, row0 + 8, t1, k + 8);
              }
              mma(acc[mt][0], a0, a1, a2, a3, b00, b01);
              mma(acc[mt][1], a0, a1, a2, a3, b10, b11);
            }
          }
          if (!(VERIFY & V_CL_NO_SYNC) || r == R - 1) cl.sync();  // broken variant: only the last round's barrier
        };
        constexpr int RREG = (CR + J - 1) / J;  // rounds with at least one register row
        load(0, 0);
        cl.sync();
#pragma unroll
        for (int r = 0; r < RREG; ++r) round(r, std::true_type{});
#pragma unroll 1
        for (int r = RREG; r < R; ++r) round(r, std::false_type{});
        if ((VERIFY & V_CHECK) && stage_bad) atomicAdd(a.err + 8, stage_bad);
      }
#else
#pragma unroll
      for (int c = 0; c < CR; ++c) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint32_t* r0 = reinterpret_cast<const uint32_t*>(xa + (size_t)(mt * 16 + fg) * K + 2 * ft);
          const uint32_t* r8 = reinterpret_cast<const uint32_t*>(xa + (size_t)(mt * 16 + fg + 8) * K + 2 * ft);
          uint32_t a0 = __ldcg(r0 + c * 8), a1 = __ldcg(r8 + c * 8), a2 = __ldcg(r0 + c * 8 + 4), a3 = __ldcg(r8 + c * 8 + 4);
          mma(acc[mt][0], a0, a1, a2, a3, rreg[c][0][0], rreg[c][0][1]);
          mma(acc[mt][1], a0, a1, a2, a3, rreg[c][1][0], rreg[c][1][1]);
        }
      }
      const uint2* sb = reinterpret_cast<const uint2*>(sr) + (size_t)warp * CS * 2 * 32 + lane;
#pragma unroll 2
      for (int c = 0; c < CS; ++c) {
        int off = (CR + c) * 8;
        uint2 b0 = sb[(c * 2) * 32], b1 = sb[(c * 2 + 1) * 32];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint32_t* r0 = reinterpret_cast<const uint32_t*>(xa + (size_t)(mt * 16 + fg) * K + 2 * ft);
          const uint32_t* r8 = reinterpret_cast<const uint32_t*>(xa + (size_t)(mt * 16 + fg + 8) * K + 2 * ft);
          uint32_t a0 = __ldcg(r0 + off), a1 = __ldcg(r8 + off), a2 = __ldcg(r0 + off + 4), a3 = __ldcg(r8 + off + 4);
          mma(acc[mt][0], a0, a1, a2, a3, b0.x, b0.y);
          mma(acc[mt][1], a0, a1, a2, a3, b1.x, b1.y);
        }
      }
#endif
      if (VERIFY & V_CHECK) {
        int bad = check(xa, t1);
        if (bad) atomicAdd(a.err + 2, bad);
      }
      float* rw = red + warp * BM * 16;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          int col = nt * 8 + 2 * ft, r = mt * 16 + fg;
          rw[r * 16 + col] = acc[mt][nt][0];
          rw[r * 16 + col + 1] = acc[mt][nt][1];
          rw[(r + 8) * 16 + col] = acc[mt][nt][2];
          rw[(r + 8) * 16 + col + 1] = acc[mt][nt][3];
        }
      __syncthreads();
#pragma unroll
      for (int e = 0; e < PE; ++e) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += red[w * BM * 16 + (pb + 16 * e) * 16 + pj];
        dh_in[e] = sum;
      }
      __syncthreads();
    };

    for (int idx = 0; idx < a.T; ++idx) {
      const int t = a.T - 1 - idx;
      if (idx == 0) {
#pragma unroll
        for (int e = 0; e < PE; ++e) {
          const int row = bt * BM + pb + 16 * e;
          const size_t unit = ((size_t)row * a.H + head) * D + p;
          if (active && row < a.B) {
            dh_in[e] = a.dfinal[unit];
            dc[e] = a.dfinal[state_stride + unit];
            if (SLSTM) dn[e] = a.dfinal[2 * state_stride + unit];
          }
        }
      }
      unsigned long long t_a = (VERIFY & V_PROF) ? gtime() : 0;
      if (idx > 0 && active) {
        if (VERIFY & V_DELAY) delay(a.err[3], s, g, t);
        reduce((VERIFY & V_SINGLE) ? 0 : (cnt - 1) & 1, t + 1);
      }
      unsigned long long t_b = (VERIFY & V_PROF) ? gtime() : 0;
      if (VERIFY & V_DELAY) delay(a.err[3], s + 977, g, t);
#pragma unroll
      for (int e = 0; e < PE; ++e) {
        const int lrow = pb + 16 * e, row = bt * BM + lrow;
        const bool valid = active && row < a.B;
        const size_t unit = ((size_t)row * a.H + head) * D + p;
        float d_i = 0.f, d_f = 0.f, d_z = 0.f, d_o = 0.f;
        const size_t here = (((size_t)row * a.T + t) * a.H + head) * D + p;
        const size_t pre = (((size_t)row * a.T + t) * 4 * a.H + head) * D + p;
        if (valid) {
          float dh = dh_in[e] + a.dstates[here];
          dc[e] = dc[e] + a.dstates[history_stride + here];
          float gi = a.pre[pre], gf = a.pre[pre + hd], gz = a.pre[pre + 2 * hd], go = a.pre[pre + 3 * hd];
          float c_t = a.states[history_stride + here];
          float c_prev = t > 0 ? a.states[history_stride + here - hd] : a.initial[state_stride + unit];
          float z = 2.0f * sigmoid(2.0f * gz) - 1.0f;
          float o = sigmoid(go);
          if (SLSTM) {
            dn[e] = dn[e] + a.dstates[2 * history_stride + here];
            float n_t = a.states[2 * history_stride + here], m_t = a.states[3 * history_stride + here];
            float n_prev = t > 0 ? a.states[2 * history_stride + here - hd] : a.initial[2 * state_stride + unit];
            float m_prev = t > 0 ? a.states[3 * history_stride + here - hd] : a.initial[3 * state_stride + unit];
            float logf_ = fminf(gf, 0.0f) - logf(1.0f + expf(-fabsf(gf))) + m_prev;
            float fgate = expf(logf_ - m_t), igate = expf(gi - m_t);
            dc[e] = dc[e] + dh * o / n_t;
            dn[e] = dn[e] - dh * o * c_t / (n_t * n_t);
            d_o = dh * (c_t / n_t) * o * (1.0f - o);
            d_f = (dc[e] * c_prev + dn[e] * n_prev) * fgate * (1.0f - sigmoid(gf));
            d_i = (dc[e] * z + dn[e]) * igate;
            d_z = dc[e] * igate * (1.0f - z * z);
            dc[e] = dc[e] * fgate;
            dn[e] = dn[e] * fgate;
          } else {
            float igate = sigmoid(gi), fgate = sigmoid(gf);
            float tc = 2.0f * sigmoid(2.0f * c_t) - 1.0f;
            dc[e] = dc[e] + dh * o * (1.0f - tc * tc);
            d_o = dh * tc * o * (1.0f - o);
            d_f = dc[e] * c_prev * fgate * (1.0f - fgate);
            d_i = dc[e] * z * igate * (1.0f - igate);
            d_z = dc[e] * igate * (1.0f - z * z);
            dc[e] = dc[e] * fgate;
          }
          a.dg[pre] = d_i;
          a.dg[pre + hd] = d_f;
          a.dg[pre + 2 * hd] = d_z;
          a.dg[pre + 3 * hd] = d_o;
        }
        if (active) {
          elem_t* w = a.xbuf + ((size_t)(gy * 2 + ((VERIFY & V_SINGLE) ? 0 : (cnt & 1))) * BM + lrow) * K + p;
          w[0] = to_elem(d_i);
          w[D] = to_elem(d_f);
          w[2 * D] = to_elem(d_z);
          w[3 * D] = to_elem(d_o);
        }
      }
      ++cnt;
      unsigned long long t_c = (VERIFY & V_PROF) ? gtime() : 0;
      if (!(VERIFY & V_NO_SYNC) || idx == a.T - 1) grid.sync();
      if ((VERIFY & V_PROF) && tid == 0 && active) {
        unsigned long long t_d = gtime();
        atomicAdd(a.err + 4, (int)((t_b - t_a) >> 6));
        atomicAdd(a.err + 5, (int)((t_c - t_b) >> 6));
        atomicAdd(a.err + 6, (int)((t_d - t_c) >> 6));
        atomicAdd(a.err + 7, 1);
      }
    }
    // Tail: gradient of the initial state from dG_0.
    if (active) {
      reduce((VERIFY & V_SINGLE) ? 0 : (cnt - 1) & 1, 0);
#pragma unroll
      for (int e = 0; e < PE; ++e) {
        const int row = bt * BM + pb + 16 * e;
        const size_t unit = ((size_t)row * a.H + head) * D + p;
        if (row < a.B) {
          a.dinit[unit] = dh_in[e];
          a.dinit[state_stride + unit] = dc[e];
          if (SLSTM) {
            a.dinit[2 * state_stride + unit] = dn[e];
            a.dinit[3 * state_stride + unit] = 0.f;
          }
        }
      }
    }
    __syncthreads();
  }
}

extern "C" int frnn_hybrid_bw_smem() { return WARPS * CS * 2 * 32 * 2 * 4 + WARPS * BM * 16 * 4; }

// Clusters of FRNN_CLUSTER CTAs that can be co-resident (0 without clusters); the caller caps the resident groups.
extern "C" int frnn_hybrid_bw_max_clusters() {
#if FRNN_CLUSTER
  int smem = frnn_hybrid_bw_smem(), n = 0;
  if (cudaFuncSetAttribute(frnn_hybrid_bw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
    return -1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(D / S, 1);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = FRNN_CLUSTER;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&n, frnn_hybrid_bw_kernel, &cfg) != cudaSuccess) return -1;
  return n;
#else
  return 0;
#endif
}

extern "C" int frnn_hybrid_bw(const float* pre, const float* states, const float* initial, const float* dstates,
                              const float* dfinal, const void* R, float* dg, float* dinit, void* xbuf, int* err,
                              int B, int T, int H, int groups, int resident, int slots, int tiles, void* stream) {
  Args a{pre, states, initial, dstates, dfinal, reinterpret_cast<const elem_t*>(R), dg, dinit,
         reinterpret_cast<elem_t*>(xbuf), err, B, T, H, groups, resident, slots, tiles};
  int smem = frnn_hybrid_bw_smem();
  cudaError_t e = cudaFuncSetAttribute(frnn_hybrid_bw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return (int)e;
#if FRNN_CLUSTER
  static_assert((D / S) % (FRNN_CLUSTER > 0 ? FRNN_CLUSTER : 1) == 0, "D / 16 CTAs must split into whole clusters");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(D / S, resident);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = FRNN_CLUSTER;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeCooperative;
  attr[1].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  e = cudaLaunchKernelEx(&cfg, frnn_hybrid_bw_kernel, a);
#else
  void* args[] = {&a};
  e = cudaLaunchCooperativeKernel((void*)frnn_hybrid_bw_kernel, dim3(D / S, resident), dim3(THREADS), args, smem,
                                  (cudaStream_t)stream);
#endif
  return (int)e;
}
