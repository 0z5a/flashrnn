// Large-D backward with R split in two dimensions (docs/flashrnn2/max-width.md section 20).
//
// Same contract and arithmetic per element as hybrid_bw.cu, except the order of the K = 4D reduction. The PC x Q
// CTAs of a group form a grid: CTA (i, j) keeps R[k, n] for the K block j (KQ = 4D / Q rows) and the column block i
// (NC = D / PC columns) resident (register rows + shared memory), so per step it reads only its K block of the
// exchanged dG (FP16) and writes an FP32 partial dh[:, column block i] for that K block. The partials of the Q K blocks
// of a column are summed by the CTA that owns the element's pointwise update (column block i, rows j * BM / Q .., see
// FRNN_ROWOWN), in fixed order j = 0 .. Q - 1. Per step:
//   pointwise (dh of this step -> dG_t into dg (FP32) and xbuf (rounded)) -> grid barrier B2 ->
//   GEMM (xbuf K block x resident R -> partial) -> grid barrier B1 -> sum of the Q partials (dh of the next step).
// Bytes per step: 8 * PC * B * D (dG slices) + 8 * Q * B * D (partials written and read), against 8 * (D / 16) * B * D
// for the 1-D kernel. The xbuf and partial buffers are single: every xbuf write (pointwise) is separated from every
// xbuf read (GEMM) by B2 before it and by B1 after it (the next write follows the next B1), and every partial write
// (GEMM) is separated from every partial read (sum) by B1 before it and by B2 after it. The first step takes dh from
// d_final without a GEMM; the GEMM of step t = 0 produces the initial state's h gradient.
//
// Warps split the NC columns (8 warps x NCW = NC / 8 columns, NTW = NCW / 8 mma n-tiles); every warp covers all BM
// rows (MT = BM / 16 m-tiles) and the whole K block. The first CR k-steps (16 rows each) of every warp's R are mma B
// fragments in registers, the rest in shared memory (fragment order). The dG K block is staged in shared memory in
// chunks of 64 rows with 128-bit loads (double buffered).
//
// Verification (FRNN_VERIFY): bit 1 content check of every staged A element against the stored FP32 dG (err[1]) and a
// step stamp per (group, K block) written after each partial and checked by every partial reader (err[2]); bit 2
// seeded delays before the pointwise update and before the GEMM; bit 4 broken: no barrier B1; bit 8 broken: no barrier
// B2 (each keeps the barriers of the last step); bit 16 phase timers (globaltimer, thread 0 of every active CTA, units
// of 64 ns): err[4] pointwise, err[5] B2, err[6] GEMM, err[7] B1, err[8] partial sum, err[9] steps.
//
// Multi-GPU (FRNN_MG = 1, docs/flashrnn2/scale-out-plan.md section 17): the PC column blocks are split over np GPUs
// (GPU r runs column blocks r * PC / np .. with all Q K blocks each), so the partial sums of a column stay on one GPU
// and B1 stays a grid barrier of that GPU. Only the exchanged dG crosses GPUs: the pointwise update stores it into the
// exchange buffer of every GPU (address table xpeers), fences at system scope, and B2 becomes the GPU's grid barrier
// followed by one arrival on every GPU's counter (speers, CTA 0) and a wait for np arrivals per step. Verification:
// the A-operand content check reads the owning GPU's stored dG (dgpeers); bit 32 broken: no cross-GPU wait (except
// at the last step); a wait longer than 5 s counts err[10] and gives up.
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
#else
typedef __half elem_t;
#define FRNN_MMA "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
__device__ __forceinline__ elem_t to_elem(float x) { return __float2half_rn(x); }
#endif

#ifndef FRNN_VERIFY
#define FRNN_VERIFY 0
#endif
constexpr int VERIFY = FRNN_VERIFY;
constexpr int V_CHECK = 1, V_DELAY = 2, V_NO_B1 = 4, V_NO_B2 = 8, V_PROF = 16, V_MG_NO_WAIT = 32;
#ifndef FRNN_MG
#define FRNN_MG 0
#endif
constexpr int D = FRNN_D, PC = FRNN_PC, Q = FRNN_Q, BM = FRNN_BM, CR = FRNN_CR;
constexpr bool SLSTM = FRNN_SLSTM != 0;
constexpr int THREADS = 256, WARPS = 8, K = 4 * D;
constexpr int NC = D / PC, NCW = NC / WARPS, NTW = NCW / 8, KQ = K / Q, KS = KQ / 16, MT = BM / 16;
#ifndef FRNN_ROWOWN
#define FRNN_ROWOWN 1
#endif
// Pointwise ownership: FRNN_ROWOWN = 1 (default) gives CTA (i, j) the BM / Q rows j * BM / Q .. of all NC columns of
// column block i (each row's loads and stores contiguous over NC columns); 0 gives it all BM rows of the NC / Q columns
// of sub-block j (the first version: 12-16 contiguous columns per row). Same elements per CTA, same arithmetic.
constexpr bool ROWOWN = FRNN_ROWOWN != 0;
#ifndef FRNN_ASYNC
#define FRNN_ASYNC 1
#endif
// FRNN_ASYNC = 1 (default): the dG chunks are staged with cp.async (L2 -> shared memory without a register round
// trip), so the copy of chunk r + 1 overlaps the MMAs of chunk r; 0: load to registers, then store (the warp waits
// for the data before its MMAs). Same values staged, same arithmetic.
constexpr bool ASYNC = FRNN_ASYNC != 0;
constexpr int NCQ = NC / Q, RPQ = BM / Q, PE = (BM * NCQ + THREADS - 1) / THREADS;
static_assert(!ROWOWN || BM % Q == 0, "row ownership needs BM a multiple of Q");
constexpr int KC = 64, LDS = KC + 8, ROUNDS = KQ / KC, J = KC / 16, CSM = KS - CR;
static_assert(D % PC == 0 && NC % 64 == 0, "column block must be a multiple of 64 (8 warps x n8 tiles)");
static_assert(K % Q == 0 && KQ % KC == 0, "K block must be a multiple of 64");
static_assert(NC % Q == 0, "pointwise sub-block NC / Q must be whole");
static_assert(BM == 16 || BM == 32 || BM == 64, "BM must be 16, 32 or 64");
static_assert(CR >= 0 && CR <= KS, "register k-steps");

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

// A-operand register (two elements, row r, reduction index k even) against the stored dG of step t.
__device__ __forceinline__ int check_pair(uint32_t v, const float* dg, int B, int T, int H, int head, int row, int t,
                                          int k) {
  if (row >= B) return 0;
  int g = k / D, o = k % D;
  size_t at = (((size_t)row * T + t) * 4 * H + (size_t)g * H + head) * D + o;
  return v != pack(to_elem(dg[at]), to_elem(dg[at + 1]));
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
  elem_t* xbuf;          // [RESIDENT, BM, 4D] exchanged dG (single buffer)
  float* part;           // [RESIDENT, Q, BM, D] FP32 partial dh (single buffer)
  int* stamp;            // [RESIDENT, Q] verification step stamps
  int* err;              // [12] verification counters / phase timers, [3] delay seed
  int B, T, H, groups, resident, slots, tiles;
  const long long* xpeers;   // FRNN_MG: every GPU's exchange buffer, arrival counter, stored dG
  const long long* speers;
  const long long* dgpeers;
  int rank, np;
};

__global__ void __launch_bounds__(THREADS, 1) frnn_hybrid_bw2d_kernel(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint2* sr = reinterpret_cast<uint2*>(smem);  // [WARPS][CSM][NTW][32] B fragments
  elem_t* stage = reinterpret_cast<elem_t*>(smem + (size_t)WARPS * CSM * NTW * 32 * sizeof(uint2));  // [2][BM][LDS]
  cg::grid_group grid = cg::this_grid();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, fg = lane >> 2, ft = lane & 3;
  const int cb = blockIdx.x / Q + (FRNN_MG ? a.rank * (PC / a.np) : 0), kb = blockIdx.x % Q, gy = blockIdx.y;
  const int n_warp = cb * NC + warp * NCW;                                 // first column of this warp's MMA tile
  const int k0 = kb * KQ;                                                   // first reduction index of the K block
  const size_t hd = (size_t)a.H * D, state_stride = (size_t)a.B * hd, history_stride = (size_t)a.B * a.T * hd;
  elem_t* xb = a.xbuf + (size_t)gy * BM * K;
  float* pt = a.part + (size_t)gy * Q * BM * D;
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
          int k = k0 + c * 16 + 2 * ft, n = n_warp + nt * 8 + fg;
          rreg[c][nt][0] = r_pair(a.R, a.H, head, k, n);
          rreg[c][nt][1] = r_pair(a.R, a.H, head, k + 8, n);
        }
      for (int c = 0; c < CSM; ++c)
        for (int nt = 0; nt < NTW; ++nt) {
          int k = k0 + (CR + c) * 16 + 2 * ft, n = n_warp + nt * 8 + fg;
          sr[(((size_t)warp * CSM + c) * NTW + nt) * 32 + lane] =
              make_uint2(r_pair(a.R, a.H, head, k, n), r_pair(a.R, a.H, head, k + 8, n));
        }
    }
    __syncthreads();

    // Pointwise ownership (FRNN_ROWOWN): element e of this thread is one (row, column) of the CTA's share; fixed for
    // the whole slot, so the carries dc / dn stay in registers.
    float dc[PE], dn[PE], dh_in[PE];
#pragma unroll
    for (int e = 0; e < PE; ++e) dc[e] = dn[e] = dh_in[e] = 0.f;
    auto elem = [&](int e, int& lrow, int& p) -> bool {
      const int idx = tid + THREADS * e;
      if (ROWOWN) {
        lrow = kb * RPQ + idx / NC;
        p = cb * NC + idx % NC;
      } else {
        lrow = idx / NCQ;
        p = cb * NC + kb * NCQ + idx % NCQ;
      }
      return idx < BM * NCQ;
    };

    // Stored dG of reduction index k: this GPU's, or with FRNN_MG the GPU owning column k % D.
    auto dgsrc = [&](int k) -> const float* {
      return FRNN_MG ? reinterpret_cast<const float*>(a.dgpeers[(k % D) / (D / a.np)]) : a.dg;
    };
    // GEMM of the K block against the staged dG of step t (rows of tile bt) -> partial dh; returns A-check count.
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
          const uint4* src = reinterpret_cast<const uint4*>(xb + (size_t)row * K + k0 + r * KC) + col8;
          elem_t* dst = stage + ((size_t)buf * BM + row) * LDS + col8 * 8;
          if (ASYNC) {
            const unsigned sa = (unsigned)__cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(src) : "memory");
          } else {
            *reinterpret_cast<uint4*>(dst) = __ldcg(src);
          }
        }
        if (ASYNC) asm volatile("cp.async.commit_group;" ::: "memory");
      };
      const uint2* sb = sr + (size_t)warp * CSM * NTW * 32 + lane;
      auto round = [&](int r, auto in_regs) {
        if (r + 1 < ROUNDS) load(r + 1, (r + 1) & 1);
        const elem_t* tile = stage + (size_t)(r & 1) * BM * LDS;
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
            if (VERIFY & V_CHECK) {
              const int row0 = bt * BM + mt * 16 + fg, k = k0 + c * 16 + 2 * ft;
              bad += check_pair(a0, dgsrc(k), a.B, a.T, a.H, head, row0, t, k) +
                     check_pair(a1, dgsrc(k), a.B, a.T, a.H, head, row0 + 8, t, k) +
                     check_pair(a2, dgsrc(k), a.B, a.T, a.H, head, row0, t, k + 8) +
                     check_pair(a3, dgsrc(k), a.B, a.T, a.H, head, row0 + 8, t, k + 8);
            }
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) mma(acc[mt][nt], a0, a1, a2, a3, b[nt][0], b[nt][1]);
          }
        }
        if (ASYNC) asm volatile("cp.async.wait_group 0;" ::: "memory");  // this thread's copies of round r + 1
        __syncthreads();  // round r + 1 staged, every warp done with round r's buffer
      };
      constexpr int RREG = (CR + J - 1) / J;  // rounds with at least one register k-step
      load(0, 0);
      if (ASYNC) asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
#pragma unroll
      for (int r = 0; r < RREG; ++r) round(r, std::true_type{});
#pragma unroll 1
      for (int r = RREG; r < ROUNDS; ++r) round(r, std::false_type{});
      // Partial dh of this K block for the warp's columns, all BM rows.
      float* dst = pt + (size_t)kb * BM * D;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
          const int r = mt * 16 + fg, n = n_warp + nt * 8 + 2 * ft;
          *reinterpret_cast<float2*>(dst + (size_t)r * D + n) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
          *reinterpret_cast<float2*>(dst + (size_t)(r + 8) * D + n) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
        }
      if (VERIFY & V_CHECK) {  // step stamp after the partial (release at GPU scope)
        __syncthreads();
        if (tid == 0) {
          __threadfence();
          asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(a.stamp + gy * Q + kb), "r"(cnt) : "memory");
        }
      }
      return bad;
    };

    for (int idx = 0; idx < a.T; ++idx) {
      const int t = a.T - 1 - idx;
      const bool last = idx == a.T - 1;
      unsigned long long t0 = (VERIFY & V_PROF) ? gtime() : 0;
      if (VERIFY & V_DELAY) delay(a.err[3], blockIdx.x, g, t);
#pragma unroll
      for (int e = 0; e < PE; ++e) {
        int lrow, p;
        const bool own = elem(e, lrow, p);
        const int row = bt * BM + lrow;
        const bool valid = active && own && row < a.B;
        const size_t unit = ((size_t)row * a.H + head) * D + p;
        if (idx == 0 && valid) {
          dh_in[e] = a.dfinal[unit];
          dc[e] = a.dfinal[state_stride + unit];
          if (SLSTM) dn[e] = a.dfinal[2 * state_stride + unit];
        }
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
        if (active && own) {  // rows beyond B publish zeros
#if FRNN_MG
          for (int q = 0; q < a.np; ++q) {
            elem_t* w = reinterpret_cast<elem_t*>(a.xpeers[q]) + (size_t)gy * BM * K + (size_t)lrow * K + p;
            w[0] = to_elem(d_i);
            w[D] = to_elem(d_f);
            w[2 * D] = to_elem(d_z);
            w[3 * D] = to_elem(d_o);
          }
#else
          elem_t* w = xb + (size_t)lrow * K + p;
          w[0] = to_elem(d_i);
          w[D] = to_elem(d_f);
          w[2 * D] = to_elem(d_z);
          w[3 * D] = to_elem(d_o);
#endif
        }
      }
      unsigned long long t1 = (VERIFY & V_PROF) ? gtime() : 0;
#if FRNN_MG
      __threadfence_system();  // the peer stores above, before this GPU signals the step
#endif
      if (!(VERIFY & V_NO_B2) || last) grid.sync();  // B2: dG_t complete in xbuf
#if FRNN_MG
      if (!(VERIFY & V_MG_NO_WAIT) || last) {  // B2 across GPUs
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
          __threadfence_system();
          for (int q = 0; q < a.np; ++q) atomicAdd_system(reinterpret_cast<int*>(a.speers[q]), 1);
        }
        if (tid == 0) {
          const int* mine = reinterpret_cast<const int*>(a.speers[a.rank]);
          const int target = a.np * (cnt + 1);
          const unsigned long long w0 = gtime();
          int seen;
          while (true) {
            asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(seen) : "l"(mine) : "memory");
            if (seen >= target) break;
            if (gtime() - w0 > 5000000000ull) {
              atomicAdd(a.err + 10, 1);
              break;
            }
          }
        }
        __syncthreads();
      }
#endif
      unsigned long long t2 = (VERIFY & V_PROF) ? gtime() : 0;
      if (VERIFY & V_DELAY) delay(a.err[3], blockIdx.x + 977, g, t);
      ++cnt;
      if (active) {
        int bad = gemm(t);
        if ((VERIFY & V_CHECK) && bad) atomicAdd(a.err + 1, bad);
      }
      unsigned long long t3 = (VERIFY & V_PROF) ? gtime() : 0;
      if (!(VERIFY & V_NO_B1) || last) grid.sync();  // B1: partials complete
      unsigned long long t4 = (VERIFY & V_PROF) ? gtime() : 0;
      if (active) {
        int stale = 0;
        if (VERIFY & V_CHECK) {
          for (int jj = 0; jj < Q; ++jj) {
            int s;
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(s) : "l"(a.stamp + gy * Q + jj) : "memory");
            stale += s != cnt;
          }
        }
#pragma unroll
        for (int e = 0; e < PE; ++e) {
          int lrow, p;
          const bool own = elem(e, lrow, p);
          float sum = 0.f;
          if (own)
            for (int jj = 0; jj < Q; ++jj) sum += __ldcg(pt + ((size_t)jj * BM + lrow) * D + p);
          dh_in[e] = sum;
        }
        if ((VERIFY & V_CHECK) && stale) atomicAdd(a.err + 2, stale);
      }
      if ((VERIFY & V_PROF) && tid == 0 && active) {
        unsigned long long t5 = gtime();
        atomicAdd(a.err + 4, (int)((t1 - t0) >> 6));
        atomicAdd(a.err + 5, (int)((t2 - t1) >> 6));
        atomicAdd(a.err + 6, (int)((t3 - t2) >> 6));
        atomicAdd(a.err + 7, (int)((t4 - t3) >> 6));
        atomicAdd(a.err + 8, (int)((t5 - t4) >> 6));
        atomicAdd(a.err + 9, 1);
      }
    }
    // Initial-state gradient: dh from the GEMM of step 0, carries from the pointwise updates.
    if (active) {
#pragma unroll
      for (int e = 0; e < PE; ++e) {
        int lrow, p;
        const bool own = elem(e, lrow, p);
        const int row = bt * BM + lrow;
        const size_t unit = ((size_t)row * a.H + head) * D + p;
        if (own && row < a.B) {
          a.dinit[unit] = dh_in[e];
          a.dinit[state_stride + unit] = dc[e];
          if (SLSTM) {
            a.dinit[2 * state_stride + unit] = dn[e];
            a.dinit[3 * state_stride + unit] = 0.f;
          }
        }
      }
    }
    // The next slot's first pointwise update writes xbuf; every GEMM read of this slot ended before the last B1.
    __syncthreads();
  }
}

extern "C" int frnn_hybrid_bw2d_smem() {
  return WARPS * CSM * NTW * 32 * (int)sizeof(uint2) + 2 * BM * LDS * (int)sizeof(elem_t);
}

extern "C" int frnn_hybrid_bw2d_ctas() { return PC * Q; }

#if FRNN_MG
// One GPU's share (the caller's current device): column blocks rank * PC / np ..; tables as in Args.
extern "C" int frnn_hybrid_bw2d_mg(const float* pre, const float* states, const float* initial, const float* dstates,
                                   const float* dfinal, const void* R, float* dg, float* dinit, void* xbuf,
                                   float* part, int* stamp, int* err, int B, int T, int H, int groups, int resident,
                                   int slots, int tiles, const long long* xpeers, const long long* speers,
                                   const long long* dgpeers, int rank, int np, void* stream) {
  if (PC % np) return -1;
  Args a{pre, states, initial, dstates, dfinal, reinterpret_cast<const elem_t*>(R), dg, dinit,
         reinterpret_cast<elem_t*>(xbuf), part, stamp, err, B, T, H, groups, resident, slots, tiles,
         xpeers, speers, dgpeers, rank, np};
  int smem = frnn_hybrid_bw2d_smem();
  cudaError_t e = cudaFuncSetAttribute(frnn_hybrid_bw2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return (int)e;
  void* args[] = {&a};
  e = cudaLaunchCooperativeKernel((void*)frnn_hybrid_bw2d_kernel, dim3(PC / np * Q, resident), dim3(THREADS), args,
                                  smem, (cudaStream_t)stream);
  return (int)e;
}
#endif

extern "C" int frnn_hybrid_bw2d(const float* pre, const float* states, const float* initial, const float* dstates,
                                const float* dfinal, const void* R, float* dg, float* dinit, void* xbuf, float* part,
                                int* stamp, int* err, int B, int T, int H, int groups, int resident, int slots,
                                int tiles, void* stream) {
  Args a{pre, states, initial, dstates, dfinal, reinterpret_cast<const elem_t*>(R), dg, dinit,
         reinterpret_cast<elem_t*>(xbuf), part, stamp, err, B, T, H, groups, resident, slots, tiles,
         nullptr, nullptr, nullptr, 0, 1};
  int smem = frnn_hybrid_bw2d_smem();
  cudaError_t e = cudaFuncSetAttribute(frnn_hybrid_bw2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return (int)e;
  void* args[] = {&a};
  e = cudaLaunchCooperativeKernel((void*)frnn_hybrid_bw2d_kernel, dim3(PC * Q, resident), dim3(THREADS), args, smem,
                                  (cudaStream_t)stream);
  return (int)e;
}
