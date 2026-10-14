// ThunderKittens R tiling across a thread-block cluster (H100, sm_90a).
//
// Same split as triton_rtiled / tilelang_rtiled: cluster rank s owns the four gate
// slices R[g, head, s*S:(s+1)*S, :] in shared memory and the matching c/n/m in
// registers. Per step, one warpgroup computes the gate pre-activations as
//   acc[4S x 16] = Rslice[4S x D] * h^T[D x 16]          (WGMMA, weights as A)
// Each CTA writes its rounded h slice into its own next-step h^T buffer and copies
// those bytes into every peer's buffer with cp.async.bulk.shared::cluster, which
// completes on the peer's mbarrier. Double buffering plus that data dependency
// orders the step, so no per-step cluster barrier is needed.
//
// Contract: forward only, FP16, one full logical call from step 0; hidden history
// [B,T,H,D] and FP32 final carry [S,B,1,H,D]; FlashRNN generic LSTM/sLSTM.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kittens.cuh"

using namespace kittens;

namespace frnn_tk {

constexpr int BM = 16;
constexpr int THREADS = 128;

template <int D, int S>
struct Layout {
    static constexpr int K = D / S;            // cluster size
    static constexpr int NT = 4 * S / 64;      // 64-row WGMMA tiles of the R slice
    static constexpr int PPT = S * BM / THREADS;  // (unit, batch) pairs per thread
    using rtile = st_hf<64, D>;
    using htile = st_hf<D, BM>;
    using stile = st_fl<64, BM>;
    static constexpr size_t smem = NT * sizeof(rtile) + 2 * sizeof(htile) + NT * sizeof(stile) + 2048;
};

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ void bulk_to_peer(void *local_dst, const void *local_src, uint32_t bytes,
                                             semaphore &local_bar, int peer) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(local_dst));
    uint32_t src = static_cast<uint32_t>(__cvta_generic_to_shared(local_src));
    uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(&local_bar));
    uint32_t rdst, rbar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rdst) : "r"(dst), "r"(peer));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rbar) : "r"(bar), "r"(peer));
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(rdst), "r"(src), "r"(bytes), "r"(rbar) : "memory");
}

template <int D, int S, bool SLSTM>
__global__ void __launch_bounds__(THREADS, 1)
rtiled_kernel(const half *__restrict__ wx, const half *__restrict__ R, const half *__restrict__ bias,
              const float *__restrict__ init, const int *__restrict__ zero, half *__restrict__ out,
              float *__restrict__ fin, int B, int T, int H) {
    using L = Layout<D, S>;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int *)&__shm[0]);
    typename L::rtile(&rt)[L::NT] = al.allocate<typename L::rtile, L::NT>();
    typename L::htile(&hb)[2] = al.allocate<typename L::htile, 2>();
    typename L::stile(&stage)[L::NT] = al.allocate<typename L::stile, L::NT>();
    __shared__ semaphore ready[2];

    const int tid = threadIdx.x;
    const int s = blockIdx.x;  // cluster rank (cluster spans x)
    const int group = blockIdx.y;
    const int head = group % H;
    const int row0 = (group / H) * BM;
    const size_t HD = (size_t)H * D;

    if (tid == 0) {
        init_semaphore(ready[0], 1, 0);
        init_semaphore(ready[1], 1, 0);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    // R slice, gate-major rows: q = g*S + j  ->  R[g, head, s*S + j, :]
    for (int idx = tid; idx < L::NT * 64 * D; idx += THREADS) {
        const int q = idx / D, k = idx % D;
        const int g = q / S, j = q % S;
        rt[q / 64][{q % 64, k}] = R[((size_t)(g * H + head) * D + s * S + j) * D + k];
    }
    // h0^T into buffer 0 (every CTA loads the full initial hidden state)
    for (int idx = tid; idx < D * BM; idx += THREADS) {
        const int k = idx / BM, b = idx % BM;
        const int row = row0 + b;
        hb[0][{k, b}] = __float2half(row < B ? init[((size_t)row * H + head) * D + k] : 0.f);
    }
    float c[L::PPT], n[L::PPT], m[L::PPT], h[L::PPT];
    float bi[L::PPT], bfg[L::PPT], bz[L::PPT], bo[L::PPT];
    const size_t stride = (size_t)B * HD;
#pragma unroll
    for (int i = 0; i < L::PPT; ++i) {
        const int p = tid + THREADS * i, j = p % S, b = p / S, row = row0 + b, col = s * S + j;
        const bool v = row < B;
        const size_t o = ((size_t)row * H + head) * D + col;
        c[i] = v ? init[stride + o] : 0.f;
        n[i] = SLSTM && v ? init[2 * stride + o] : 1.f;
        m[i] = SLSTM && v ? init[3 * stride + o] : 0.f;
        h[i] = 0.f;
        bi[i] = __half2float(bias[(0 * H + head) * D + col]);
        bfg[i] = __half2float(bias[(1 * H + head) * D + col]);
        bz[i] = __half2float(bias[(2 * H + head) * D + col]);
        bo[i] = __half2float(bias[(3 * H + head) * D + col]);
    }
    const bool zero_normalizer = SLSTM && zero[0] != 0;
    const uint32_t slice_bytes = S * BM * sizeof(half);
    if (tid == 0) tma::expect_bytes(ready[1], (L::K - 1) * slice_bytes);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    everyone::tma::cluster::sync();  // barriers initialized and expected before any peer copy

    rt_fl<16, BM> acc[L::NT];
    for (int t = 0; t < T; ++t) {
        const int b_cur = t & 1, b_next = b_cur ^ 1;
        if (t > 0) wait(ready[b_cur], ((t - 1) >> 1) & 1);
#pragma unroll
        for (int i = 0; i < L::NT; ++i) {
            kittens::warp::zero(acc[i]);
            warpgroup::mma_AB(acc[i], rt[i], hb[b_cur]);
        }
        warpgroup::mma_async_wait();
#pragma unroll
        for (int i = 0; i < L::NT; ++i) warpgroup::store(stage[i], acc[i]);
        __syncthreads();
        // The buffer read at step t-1 is free again: arm its next phase before sending.
        if (tid == 0 && t + 1 < T && t >= 1) tma::expect_bytes(ready[b_next], (L::K - 1) * slice_bytes);
#pragma unroll
        for (int i = 0; i < L::PPT; ++i) {
            const int p = tid + THREADS * i, j = p % S, b = p / S, row = row0 + b, col = s * S + j;
            const bool v = row < B;
            const size_t xo = (((size_t)row * T + t) * 4) * HD + (size_t)head * D + col;
            const int q0 = j, q1 = S + j, q2 = 2 * S + j, q3 = 3 * S + j;
            const float gi = stage[q0 / 64][{q0 % 64, b}] + (v ? __half2float(wx[xo]) : 0.f) + bi[i];
            const float gf = stage[q1 / 64][{q1 % 64, b}] + (v ? __half2float(wx[xo + HD]) : 0.f) + bfg[i];
            const float gz = stage[q2 / 64][{q2 % 64, b}] + (v ? __half2float(wx[xo + 2 * HD]) : 0.f) + bz[i];
            const float go = stage[q3 / 64][{q3 % 64, b}] + (v ? __half2float(wx[xo + 3 * HD]) : 0.f) + bo[i];
            const float zv = 2.f * sigmoidf(2.f * gz) - 1.f;
            if constexpr (SLSTM) {
                const float logf_ = fminf(gf, 0.f) - __logf(1.f + __expf(-fabsf(gf))) + m[i];
                const float nm = (t == 0 && zero_normalizer) ? gi : fmaxf(gi, logf_);
                const float ig = __expf(gi - nm), fg = __expf(logf_ - nm);
                c[i] = fg * c[i] + ig * zv;
                n[i] = fmaxf(fg * n[i] + ig, 1.f);
                h[i] = sigmoidf(go) * c[i] / n[i];
                m[i] = nm;
            } else {
                c[i] = sigmoidf(gf) * c[i] + sigmoidf(gi) * zv;
                h[i] = sigmoidf(go) * (2.f * sigmoidf(2.f * c[i]) - 1.f);
            }
            if (v) out[(((size_t)row * T + t) * H + head) * D + col] = __float2half(h[i]);
            if (t + 1 < T) hb[b_next][{col, b}] = __float2half(h[i]);
        }
        if (t + 1 < T) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();
            if (tid == 0) {
                char *base = reinterpret_cast<char *>(&hb[b_next].data[0]) + (size_t)s * slice_bytes;
                for (int peer = 0; peer < L::K; ++peer)
                    if (peer != s) bulk_to_peer(base, base, slice_bytes, ready[b_next], peer);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < L::PPT; ++i) {
        const int p = tid + THREADS * i, j = p % S, b = p / S, row = row0 + b, col = s * S + j;
        if (row >= B) continue;
        const size_t o = ((size_t)row * H + head) * D + col;
        fin[o] = h[i];
        fin[stride + o] = c[i];
        if (SLSTM) {
            fin[2 * stride + o] = n[i];
            fin[3 * stride + o] = m[i];
        }
    }
    everyone::tma::cluster::sync();  // keep shared memory alive until every peer copy has landed
}

template <int D, int S, bool SLSTM>
int launch(const half *wx, const half *R, const half *bias, const float *init, const int *zero, half *out,
           float *fin, int B, int T, int H, cudaStream_t stream) {
    using L = Layout<D, S>;
    auto kernel = rtiled_kernel<D, S, SLSTM>;
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L::smem);
    if (e != cudaSuccess) return 1000 + (int)e;
    if (L::K > 8) {
        e = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (e != cudaSuccess) return 2000 + (int)e;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(L::K, ((B + BM - 1) / BM) * H, 1);
    cfg.blockDim = dim3(THREADS, 1, 1);
    cfg.dynamicSmemBytes = L::smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = L::K;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    e = cudaLaunchKernelEx(&cfg, kernel, wx, R, bias, init, zero, out, fin, B, T, H);
    return e == cudaSuccess ? 0 : 3000 + (int)e;
}

}  // namespace frnn_tk

#define FRNN_TK_CASE(D_, S_)                                                                              \
    if (D == D_ && S == S_)                                                                                \
        return slstm ? frnn_tk::launch<D_, S_, true>(wx, R, bias, init, zero, out, fin, B, T, H, stream)   \
                     : frnn_tk::launch<D_, S_, false>(wx, R, bias, init, zero, out, fin, B, T, H, stream);

extern "C" int frnn_tk_rtiled(int slstm, int D, int S, const half *wx, const half *R, const half *bias,
                              const float *init, const int *zero, half *out, float *fin, int B, int T, int H,
                              cudaStream_t stream) {
    FRNN_TK_CASE(128, 32)
    FRNN_TK_CASE(128, 64)
    FRNN_TK_CASE(192, 16)
    FRNN_TK_CASE(192, 32)
    FRNN_TK_CASE(192, 64)
    FRNN_TK_CASE(256, 16)
    FRNN_TK_CASE(256, 32)
    FRNN_TK_CASE(256, 64)
    FRNN_TK_CASE(384, 32)
    FRNN_TK_CASE(512, 32)
    return -1;  // unsupported (D, S)
}

extern "C" int frnn_tk_smem(int D, int S) {
#define FRNN_TK_SMEM(D_, S_) \
    if (D == D_ && S == S_) return (int)frnn_tk::Layout<D_, S_>::smem;
    FRNN_TK_SMEM(128, 32) FRNN_TK_SMEM(128, 64) FRNN_TK_SMEM(192, 16) FRNN_TK_SMEM(192, 32) FRNN_TK_SMEM(192, 64)
    FRNN_TK_SMEM(256, 16) FRNN_TK_SMEM(256, 32) FRNN_TK_SMEM(256, 64) FRNN_TK_SMEM(384, 32) FRNN_TK_SMEM(512, 32)
    return -1;
}
