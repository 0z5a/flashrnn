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
//
// FRNN_P > 1 (multi-GPU over NVSwitch multicast; docs/flashrnn2/tmem-beta-handoff.md, "Multi-GPU"): P GPUs split the
// units, GPU `rank` owns units [rank D / P, (rank + 1) D / P) with its C CTAs owning NU = D / (P C) each; every CTA
// still holds the whole round(h) [BM x D] as its A operand. Per step the owner publishes its block twice: into the
// local peers' A buffers through DSMEM as before, and into a multicast-mapped global exchange buffer (one store per
// element; the NVSwitch replicates it into every GPU's copy, including the sender's). After its CTA's stores it
// releases a per-(parity, GPU, CTA) step flag through the same multicast mapping. Before the GEMM a CTA waits for
// the local cluster barrier, then polls the (P - 1) C remote flags of this step (ld.acquire.sys on its local copy),
// and pulls each remote GPU's slice of round(h) from its local copy of the exchange buffer into shared memory with one
// cp.async.bulk per remote GPU (mbarrier completion). For that pull to be one contiguous copy, the A operand uses the
// K-core-major canonical layout in multi-GPU builds: LBO_A = 256 (K-adjacent core matrices), SBO_A = 128 (the two
// 8-row groups adjacent), so a slice of D / P columns is (D / P) * 32 contiguous bytes; the exchange buffer uses the
// same layout. The flags carry flag_base + step + 1, flag_base rising per call, so nothing is reset between calls.
// A poll gives up after 2^26 spins and raises err[1] (the call then returns garbage, not a hang).
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
#ifndef FRNN_P
#define FRNN_P 1
#endif
constexpr int VERIFY = FRNN_VERIFY, MODE = FRNN_MODE, V_PROF = 16;
constexpr int D = FRNN_D, C = FRNN_C, P = FRNN_P;
constexpr bool MG = P > 1;
constexpr unsigned SPIN = 1u << 26;
constexpr bool SLSTM = FRNN_SLSTM != 0;
constexpr int S = SLSTM ? 4 : 2;
constexpr int THREADS = 256, BM = 16, MMA_M = 64, MMA_K = 16;
constexpr int DP = D / P;                // units per GPU
constexpr int NU = DP / C, NC = 4 * NU, KS = D / MMA_K;
constexpr int PE = (BM * NU + THREADS - 1) / THREADS;
// Canonical K-major no-swizzle layouts: LBO between the two k-cores of a K = 16 slab, SBO between 8-row groups.
// B (R block): all core matrices of an 8-row group contiguous. A (round(h)): the same in single-GPU builds; in
// multi-GPU builds K-core-major (SBO 128, LBO 256) so that a GPU's slice of columns is one contiguous piece.
constexpr int LBO_B = 128, SBO_B = D * 16;
constexpr int LBO_A = MG ? 256 : 128, SBO_A = MG ? 128 : D * 16;
constexpr int SMEM_H = BM * D * 2;       // one parity buffer: 16 rows (two row groups) of D elements
constexpr int SLICE = DP * BM * 2;       // multi-GPU: bytes of one GPU's slice of a parity buffer (contiguous)
constexpr int SMEM_R = NC * D * 2;       // R block: NC rows (N) of D elements (K)
constexpr int SMEM_P = BM * NC * 4;      // gate preactivations [BM][NC] FP32
constexpr int TMEM_COLS = NC >= 32 ? NC : 32;
static_assert(D % (P * C) == 0 && NU % 16 == 0, "NU must be a multiple of 16");
static_assert(!MG || (DP % 8 == 0 && SLICE % 16 == 0), "a GPU's slice must be whole core columns");
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

// Byte offset of element (m, k) in the canonical K-major no-swizzle layouts of the A (round(h)) and B (R) operands.
__device__ __forceinline__ unsigned canon_a(int m, int k) {
  return (unsigned)((m >> 3) * SBO_A + (k >> 3) * LBO_A + (m & 7) * 16 + (k & 7) * 2);
}
__device__ __forceinline__ unsigned canon_b(int m, int k) {
  return (unsigned)((m >> 3) * SBO_B + (k >> 3) * LBO_B + (m & 7) * 16 + (k & 7) * 2);
}

// Shared-memory matrix descriptor (64 bits): [0,14) start >> 4, [16,30) LBO >> 4, [32,46) SBO >> 4, [49,52) base
// offset 0, [61,64) layout type 0 = SWIZZLE_NONE. UNVERIFIED bit layout; see the handoff.
__device__ __forceinline__ unsigned long long smem_desc(unsigned addr, int lbo, int sbo) {
  unsigned long long d = 0;
  d |= (unsigned long long)((addr >> 4) & 0x3FFF);
  d |= (unsigned long long)((lbo >> 4) & 0x3FFF) << 16;
  d |= (unsigned long long)((sbo >> 4) & 0x3FFF) << 32;
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
#if FRNN_P > 1
__device__ __forceinline__ void mbar_expect(unsigned long long* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
// One bulk copy global -> this CTA's shared memory, completing `bytes` on the mbarrier.
__device__ __forceinline__ void bulk_g2s(void* smem_dst, const void* gsrc, unsigned bytes, unsigned long long* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(smem_dst)), "l"(gsrc), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
#endif

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
  // FRNN_P > 1: multicast-mapped exchange buffer ([2 parities][SMEM_H bytes], K-core-major) and flags
  // ([2][P][C] int32), each as the multicast address (stores) and this GPU's unicast copy (loads).
  unsigned char* mc_x;
  const unsigned char* uc_x;
  int* mc_flag;
  const int* uc_flag;
  int rank, flag_base;
};

__global__ void __launch_bounds__(THREADS, 1) frnn_hybrid_fwc_tmem_kernel(Args a) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* hbuf = smem;                                   // [2][SMEM_H] canonical A operands by parity
  unsigned char* rbuf = smem + 2 * SMEM_H;                      // canonical B operand (R block)
  float* spart = reinterpret_cast<float*>(smem + 2 * SMEM_H + SMEM_R);  // [BM][NC]
  __shared__ __align__(8) unsigned long long mma_bar;
  __shared__ __align__(8) unsigned long long xbar;  // FRNN_P > 1: remote slices landed (bulk copies)
  __shared__ unsigned tmem_base;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cb = (int)cluster.block_rank();
  const int ub = MG ? a.rank * DP : 0;  // first unit of this GPU
  unsigned xphase = 0;
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
    mbar_init(&xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // R block into the canonical layout: row n = gate g * NU + unit u of this CTA, column k.
  for (int i = tid; i < NC * D; i += THREADS) {
    const int n = i / D, k = i % D, gq = n / NU, o = ub + cb * NU + n % NU;
    *reinterpret_cast<elem_t*>(rbuf + canon_b(n, k)) = a.R[(((size_t)gq * a.H + head) * D + o) * D + k];
  }
  // No zeroing of the parity buffers: every element of both is written by its owner before it is read (owners of
  // rows beyond B publish zeros), and a zeroing loop here would race with faster peers' round(h_0) pushes into this
  // CTA's buffer (no cluster barrier separates them) -- the step-0 error that decayed over steps in the first B300
  // run (2026-10-20).
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
    const int row = bt * BM + lrow, o = ub + cb * NU + u;
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
      for (int r = 0; r < C; ++r) *reinterpret_cast<elem_t*>(peer[r] + canon_a(lrow, o)) = v;
#if FRNN_P > 1
      *reinterpret_cast<elem_t*>(a.mc_x + canon_a(lrow, o)) = v;  // multicast: lands in every GPU's parity-0 copy
#endif
    }
  }
#if FRNN_P > 1
  __syncthreads();  // this CTA's block of round(h_0) is stored; release its step flag (parity 0, value base + 1)
  if (tid == 0) {
    __threadfence_system();
    st_release_sys(a.mc_flag + a.rank * C + cb, a.flag_base + 1);
  }
#endif

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
        const size_t xo = (((size_t)row * a.T + t) * 4 * a.H + head) * D + ub + cb * NU + u;
#pragma unroll
        for (int q = 0; q < 4; ++q) wxr[e][q] = a.wx[xo + (size_t)q * hd];
      }
    }
    cluster_wait();
    unsigned long long t1a = (VERIFY & V_PROF) ? gtime() : 0;
#if FRNN_P > 1
    {
      // Remote GPUs' blocks of round(h_{t-1}) have landed in this GPU's copy when their flags reach base + t + 1.
      const int want = a.flag_base + t + 1;
      const int* flags = a.uc_flag + (size_t)(t & 1) * P * C;
      for (int i = tid; i < (P - 1) * C; i += THREADS) {
        const int q = i / C, qq = q < a.rank ? q : q + 1, c2 = i % C;
        unsigned spins = 0;
        while (ld_acquire_sys(flags + qq * C + c2) < want && ++spins < SPIN) {
        }
        if (spins >= SPIN) a.err[1] = 1;
      }
      __syncthreads();
      if (tid == 0) {  // pull every remote slice into this parity's A buffer: one bulk copy per remote GPU
        asm volatile("fence.proxy.async;" ::: "memory");
        mbar_expect(&xbar, (unsigned)((P - 1) * SLICE));
        for (int q = 0; q < P; ++q)
          if (q != a.rank)
            bulk_g2s(hbuf + (t & 1) * SMEM_H + q * SLICE, a.uc_x + (size_t)(t & 1) * SMEM_H + q * SLICE, SLICE, &xbar);
      }
      mbar_wait(&xbar, xphase);
      xphase ^= 1;
    }
#endif
    // The peers' generic-proxy stores into this CTA's A buffer precede the async-proxy MMA reads.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long t1 = (VERIFY & V_PROF) ? gtime() : 0;

    // GEMM: one thread issues KS MMAs into the tensor-memory accumulator, then commits to the mbarrier.
    if (tid == 0) {
      const unsigned abase = smem_u32(hbuf + (t & 1) * SMEM_H), bbase = smem_u32(rbuf), idesc = inst_desc();
      for (int c = 0; c < KS; ++c) {
        const unsigned long long adesc = smem_desc(abase + c * 2 * LBO_A, LBO_A, SBO_A),
                                 bdesc = smem_desc(bbase + c * 2 * LBO_B, LBO_B, SBO_B);
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
      const int row = bt * BM + lrow, o = ub + cb * NU + u;
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
        for (int r = 0; r < C; ++r) *reinterpret_cast<elem_t*>(peer[r] + nb * SMEM_H + canon_a(lrow, o)) = v;
#if FRNN_P > 1
        *reinterpret_cast<elem_t*>(a.mc_x + nb * SMEM_H + canon_a(lrow, o)) = v;
#endif
      }
    }
#if FRNN_P > 1
    if (!last) {  // this CTA's block of round(h_t) is in the multicast buffer: release its flag (parity nb, base + t + 2)
      __syncthreads();
      if (tid == 0) {
        __threadfence_system();
        st_release_sys(a.mc_flag + nb * P * C + a.rank * C + cb, a.flag_base + t + 2);
      }
    }
#endif
    if ((VERIFY & V_PROF) && tid == 0) {
      const unsigned long long t3 = gtime();
      atomicAdd(a.err + 4, (int)((t1a - t0) >> 6));  // cluster barrier (with the Wx loads)
      atomicAdd(a.err + 7, (int)((t1 - t1a) >> 6));  // FRNN_P > 1: remote flags + slice pull
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
    const size_t unit = ((size_t)row * a.H + head) * D + ub + cb * NU + u;
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

extern "C" int frnn_hybrid_fwc_tmem_peers() { return P; }
extern "C" int frnn_hybrid_fwc_tmem_xbytes() { return 2 * SMEM_H; }  // exchange buffer bytes per GPU (FRNN_P > 1)

extern "C" int frnn_hybrid_fwc_tmem(const void* wx, const void* R, const void* bias, const float* initial, void* out,
                                    float* final, int* err, const void* zero_first, int B, int T, int H, int groups,
                                    int tiles, void* stream, void* mc_x, const void* uc_x, void* mc_flag,
                                    const void* uc_flag, int rank, int flag_base) {
  Args a{reinterpret_cast<const elem_t*>(wx), reinterpret_cast<const elem_t*>(R),
         reinterpret_cast<const elem_t*>(bias), initial, reinterpret_cast<snap_t*>(out), final, err,
         reinterpret_cast<const bool*>(zero_first), B, T, H, tiles,
         reinterpret_cast<unsigned char*>(mc_x), reinterpret_cast<const unsigned char*>(uc_x),
         reinterpret_cast<int*>(mc_flag), reinterpret_cast<const int*>(uc_flag), rank, flag_base};
  if (MG && (mc_x == nullptr || uc_x == nullptr || mc_flag == nullptr || uc_flag == nullptr)) return -1;
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
