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
// bit 4 broken: single buffer; bit 8 broken: no grid barrier (except before the tail).
#include <cooperative_groups.h>
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
constexpr int V_CHECK = 1, V_DELAY = 2, V_SINGLE = 4, V_NO_SYNC = 8;
constexpr int D = FRNN_D;
constexpr int KRW = FRNN_KRW;
constexpr bool SLSTM = FRNN_SLSTM != 0;
constexpr int S = 16, BM = 16, WARPS = 8, THREADS = 256;
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
  int* err;              // verification: [0] unused, [1] after read, [2] after MMA, [3] delay seed
  int B, T, H, groups, resident, slots, tiles;
};

__global__ void __launch_bounds__(THREADS, 1) frnn_hybrid_bw_kernel(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint32_t* sr = reinterpret_cast<uint32_t*>(smem);                                   // [WARPS][CS][2][32][2]
  float* red = reinterpret_cast<float*>(smem + (size_t)WARPS * CS * 2 * 32 * 2 * 4);  // [WARPS][16][16]
  cg::grid_group grid = cg::this_grid();
  const int s = blockIdx.x, gy = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int fg = lane >> 2, ft = lane & 3;  // mma fragment row group / thread-in-group
  const int pb = tid >> 4, pj = tid & 15;   // pointwise element (batch row, column)
  const int p = s * S + pj;
  const size_t hd = (size_t)a.H * D, state_stride = (size_t)a.B * hd, history_stride = (size_t)a.B * a.T * hd;
  uint32_t rreg[CR > 0 ? CR : 1][2][2];
  int cnt = 0;  // publishes so far (parity of the exchange buffer)

  for (int slot = 0; slot < a.slots; ++slot) {
    const int group = slot * a.resident + gy;
    const bool active = group < a.groups;
    const int g = active ? group : a.groups - 1;
    const int head = g / a.tiles, bt = g % a.tiles;
    const int row = bt * BM + pb;
    const bool valid = active && row < a.B;
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

    const size_t unit = ((size_t)row * a.H + head) * D + p;
    float dc = 0.f, dn = 0.f, dh_in = 0.f;

    // dh from the exchanged, rounded dG of every CTA of the group (parity par); all 256 threads take part.
    auto reduce = [&](int par, int t1) -> float {
      float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      const elem_t* xa = a.xbuf + ((size_t)(gy * 2 + par) * BM) * K + warp * KW;
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(xa + (size_t)fg * K + 2 * ft);
      const uint32_t* r8 = reinterpret_cast<const uint32_t*>(xa + (size_t)(fg + 8) * K + 2 * ft);
      if (VERIFY & V_CHECK) {
        int bad = 0;
        for (int c = 0; c < CR + CS; ++c) {
          int k = warp * KW + c * 16 + 2 * ft;
          bad += check_pair(__ldcg(r0 + c * 8), a.dg, a.B, a.T, a.H, head, bt * BM + fg, t1, k);
          bad += check_pair(__ldcg(r8 + c * 8), a.dg, a.B, a.T, a.H, head, bt * BM + fg + 8, t1, k);
          bad += check_pair(__ldcg(r0 + c * 8 + 4), a.dg, a.B, a.T, a.H, head, bt * BM + fg, t1, k + 8);
          bad += check_pair(__ldcg(r8 + c * 8 + 4), a.dg, a.B, a.T, a.H, head, bt * BM + fg + 8, t1, k + 8);
        }
        if (bad) atomicAdd(a.err + 1, bad);
      }
#pragma unroll
      for (int c = 0; c < CR; ++c) {
        uint32_t a0 = __ldcg(r0 + c * 8), a1 = __ldcg(r8 + c * 8), a2 = __ldcg(r0 + c * 8 + 4), a3 = __ldcg(r8 + c * 8 + 4);
        mma(acc[0], a0, a1, a2, a3, rreg[c][0][0], rreg[c][0][1]);
        mma(acc[1], a0, a1, a2, a3, rreg[c][1][0], rreg[c][1][1]);
      }
      const uint2* sb = reinterpret_cast<const uint2*>(sr) + (size_t)warp * CS * 2 * 32 + lane;
#pragma unroll 4
      for (int c = 0; c < CS; ++c) {
        int off = (CR + c) * 8;
        uint32_t a0 = __ldcg(r0 + off), a1 = __ldcg(r8 + off), a2 = __ldcg(r0 + off + 4), a3 = __ldcg(r8 + off + 4);
        uint2 b0 = sb[(c * 2) * 32], b1 = sb[(c * 2 + 1) * 32];
        mma(acc[0], a0, a1, a2, a3, b0.x, b0.y);
        mma(acc[1], a0, a1, a2, a3, b1.x, b1.y);
      }
      if (VERIFY & V_CHECK) {
        // Re-read every A element this thread used, once now (exchange content) and once after the MMAs below.
        int bad = 0;
        for (int c = 0; c < CR + CS; ++c) {
          int k = warp * KW + c * 16 + 2 * ft;
          bad += check_pair(__ldcg(r0 + c * 8), a.dg, a.B, a.T, a.H, head, bt * BM + fg, t1, k);
          bad += check_pair(__ldcg(r8 + c * 8), a.dg, a.B, a.T, a.H, head, bt * BM + fg + 8, t1, k);
          bad += check_pair(__ldcg(r0 + c * 8 + 4), a.dg, a.B, a.T, a.H, head, bt * BM + fg, t1, k + 8);
          bad += check_pair(__ldcg(r8 + c * 8 + 4), a.dg, a.B, a.T, a.H, head, bt * BM + fg + 8, t1, k + 8);
        }
        if (bad) atomicAdd(a.err + 2, bad);
      }
      float* rw = red + warp * 256;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        int col = nt * 8 + 2 * ft;
        rw[fg * 16 + col] = acc[nt][0];
        rw[fg * 16 + col + 1] = acc[nt][1];
        rw[(fg + 8) * 16 + col] = acc[nt][2];
        rw[(fg + 8) * 16 + col + 1] = acc[nt][3];
      }
      __syncthreads();
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) sum += red[w * 256 + pb * 16 + pj];
      __syncthreads();
      return sum;
    };

    for (int idx = 0; idx < a.T; ++idx) {
      const int t = a.T - 1 - idx;
      if (idx == 0) {
        if (valid) {
          dh_in = a.dfinal[unit];
          dc = a.dfinal[state_stride + unit];
          if (SLSTM) dn = a.dfinal[2 * state_stride + unit];
        }
      } else if (active) {
        if (VERIFY & V_DELAY) delay(a.err[3], s, g, t);
        dh_in = reduce((VERIFY & V_SINGLE) ? 0 : (cnt - 1) & 1, t + 1);
      }
      float d_i = 0.f, d_f = 0.f, d_z = 0.f, d_o = 0.f;
      const size_t here = (((size_t)row * a.T + t) * a.H + head) * D + p;
      const size_t pre = (((size_t)row * a.T + t) * 4 * a.H + head) * D + p;
      if (valid) {
        float dh = dh_in + a.dstates[here];
        dc = dc + a.dstates[history_stride + here];
        float gi = a.pre[pre], gf = a.pre[pre + hd], gz = a.pre[pre + 2 * hd], go = a.pre[pre + 3 * hd];
        float c_t = a.states[history_stride + here];
        float c_prev = t > 0 ? a.states[history_stride + here - hd] : a.initial[state_stride + unit];
        float z = 2.0f * sigmoid(2.0f * gz) - 1.0f;
        float o = sigmoid(go);
        if (SLSTM) {
          dn = dn + a.dstates[2 * history_stride + here];
          float n_t = a.states[2 * history_stride + here], m_t = a.states[3 * history_stride + here];
          float n_prev = t > 0 ? a.states[2 * history_stride + here - hd] : a.initial[2 * state_stride + unit];
          float m_prev = t > 0 ? a.states[3 * history_stride + here - hd] : a.initial[3 * state_stride + unit];
          float logf_ = fminf(gf, 0.0f) - logf(1.0f + expf(-fabsf(gf))) + m_prev;
          float fgate = expf(logf_ - m_t), igate = expf(gi - m_t);
          dc = dc + dh * o / n_t;
          dn = dn - dh * o * c_t / (n_t * n_t);
          d_o = dh * (c_t / n_t) * o * (1.0f - o);
          d_f = (dc * c_prev + dn * n_prev) * fgate * (1.0f - sigmoid(gf));
          d_i = (dc * z + dn) * igate;
          d_z = dc * igate * (1.0f - z * z);
          dc = dc * fgate;
          dn = dn * fgate;
        } else {
          float igate = sigmoid(gi), fgate = sigmoid(gf);
          float tc = 2.0f * sigmoid(2.0f * c_t) - 1.0f;
          dc = dc + dh * o * (1.0f - tc * tc);
          d_o = dh * tc * o * (1.0f - o);
          d_f = dc * c_prev * fgate * (1.0f - fgate);
          d_i = dc * z * igate * (1.0f - igate);
          d_z = dc * igate * (1.0f - z * z);
          dc = dc * fgate;
        }
        a.dg[pre] = d_i;
        a.dg[pre + hd] = d_f;
        a.dg[pre + 2 * hd] = d_z;
        a.dg[pre + 3 * hd] = d_o;
      }
      if (VERIFY & V_DELAY) delay(a.err[3], s + 977, g, t);
      if (active) {
        elem_t* w = a.xbuf + ((size_t)(gy * 2 + ((VERIFY & V_SINGLE) ? 0 : (cnt & 1))) * BM + pb) * K + p;
        w[0] = to_elem(d_i);
        w[D] = to_elem(d_f);
        w[2 * D] = to_elem(d_z);
        w[3 * D] = to_elem(d_o);
      }
      ++cnt;
      if (!(VERIFY & V_NO_SYNC) || idx == a.T - 1) grid.sync();
    }
    // Tail: gradient of the initial state from dG_0.
    if (active) {
      dh_in = reduce((VERIFY & V_SINGLE) ? 0 : (cnt - 1) & 1, 0);
      if (valid) {
        a.dinit[unit] = dh_in;
        a.dinit[state_stride + unit] = dc;
        if (SLSTM) {
          a.dinit[2 * state_stride + unit] = dn;
          a.dinit[3 * state_stride + unit] = 0.f;
        }
      }
    }
    __syncthreads();
  }
}

extern "C" int frnn_hybrid_bw_smem() { return WARPS * CS * 2 * 32 * 2 * 4 + WARPS * 256 * 4; }

extern "C" int frnn_hybrid_bw(const float* pre, const float* states, const float* initial, const float* dstates,
                              const float* dfinal, const void* R, float* dg, float* dinit, void* xbuf, int* err,
                              int B, int T, int H, int groups, int resident, int slots, int tiles, void* stream) {
  Args a{pre, states, initial, dstates, dfinal, reinterpret_cast<const elem_t*>(R), dg, dinit,
         reinterpret_cast<elem_t*>(xbuf), err, B, T, H, groups, resident, slots, tiles};
  int smem = frnn_hybrid_bw_smem();
  cudaError_t e = cudaFuncSetAttribute(frnn_hybrid_bw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return (int)e;
  void* args[] = {&a};
  e = cudaLaunchCooperativeKernel((void*)frnn_hybrid_bw_kernel, dim3(D / S, resident), dim3(THREADS), args, smem,
                                  (cudaStream_t)stream);
  return (int)e;
}
