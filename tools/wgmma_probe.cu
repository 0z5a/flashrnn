// Probe of the wgmma operand conventions used by hybrid_fwc.cu's FRNN_FWC_WGMMA mode (docs/flashrnn2/max-width.md
// section 82): D[64 x 16] = A[64 x K] * B[16 x K]^T with wgmma.m64n16k16 (f32 accumulate), A from shared memory
// (canonical K-major, no swizzle: 8 x 8 core matrices of 128 contiguous bytes) or from registers, B from shared
// memory in the same canonical layout. The leading / stride byte offsets are tried both ways; the run prints, per
// (type, A source, convention), the largest difference to a CPU reference. The convention with ~1e-3 error (half /
// bf16 rounding of random inputs) is the one the kernel must use.
// Build: nvcc -std=c++17 -O2 -arch=sm_90a tools/wgmma_probe.cu -o wgmma_probe
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int M = 64, N = 16, K = 64, KS = K / 16;

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

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

template <bool BF16>
__device__ __forceinline__ void wg_ss(float (&d)[8], uint64_t da, uint64_t db) {
  if (BF16)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "l"(da), "l"(db), "r"(1));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "l"(da), "l"(db), "r"(1));
}

template <bool BF16>
__device__ __forceinline__ void wg_rs(float (&d)[8], const uint32_t (&a)[4], uint64_t db) {
  if (BF16)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}

// Canonical K-major no-swizzle layout of a [rows x K] 16-bit matrix: core matrix (r / 8, k / 8) at
// (r / 8) * SBO + (k / 8) * LBO bytes, row r % 8 at + 16 bytes each, k % 8 at + 2 bytes each. LBO = 128 (K-adjacent
// core matrices contiguous), SBO = (K / 8) * 128.
__device__ __forceinline__ int canon(int r, int k, int K_) { return (r >> 3) * (K_ * 8) + (k >> 3) * 64 + (r & 7) * 8 + (k & 7); }

template <typename T, bool BF16>
__global__ void probe(const T* A, const T* B, float* D, int a_regs, int swap) {
  __shared__ __align__(128) T sa[M * K];
  __shared__ __align__(128) T sb[N * K];
  const int tid = threadIdx.x;
  for (int i = tid; i < M * K; i += 128) sa[canon(i / K, i % K, K)] = A[i];
  for (int i = tid; i < N * K; i += 128) sb[canon(i / K, i % K, K)] = B[i];
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const int warp = tid >> 5, lane = tid & 31, fg = lane >> 2, ft = lane & 3;
  uint32_t areg[KS][4];
  if (a_regs) {  // mma.m16n8k16-style A fragment per warp (rows 16 warp + fg, + 8; k 2 ft, + 8)
    for (int c = 0; c < KS; ++c) {
      const int r = warp * 16 + fg, k = c * 16 + 2 * ft;
      auto pk = [&](int rr, int kk) {
        const uint16_t lo = *reinterpret_cast<const uint16_t*>(&A[rr * K + kk]);
        const uint16_t hi = *reinterpret_cast<const uint16_t*>(&A[rr * K + kk + 1]);
        return (uint32_t)lo | ((uint32_t)hi << 16);
      };
      areg[c][0] = pk(r, k);
      areg[c][1] = pk(r + 8, k);
      areg[c][2] = pk(r, k + 8);
      areg[c][3] = pk(r + 8, k + 8);
    }
  }
  float d[8];
  for (int i = 0; i < 8; ++i) d[i] = 0.f;
  const unsigned lbo_a = swap ? (K / 8) * 128 : 128, sbo_a = swap ? 128 : (K / 8) * 128;
  const unsigned lbo_b = lbo_a, sbo_b = sbo_a;
  wg_fence();
  for (int c = 0; c < KS; ++c) {
    const uint64_t db = make_desc(sb + c * 128, lbo_b, sbo_b);  // k-step c: two K-core matrices on
    if (a_regs) {
      wg_rs<BF16>(d, areg[c], db);
    } else {
      const uint64_t da = make_desc(sa + c * 128, lbo_a, sbo_a);
      wg_ss<BF16>(d, da, db);
    }
  }
  wg_commit();
  wg_wait0();
  // D fragment: rows 16 warp + fg (+ 8), cols 2 ft (+ 1), + 8.
  const int r = warp * 16 + fg, cc = 2 * ft;
  D[r * N + cc] = d[0];
  D[r * N + cc + 1] = d[1];
  D[(r + 8) * N + cc] = d[2];
  D[(r + 8) * N + cc + 1] = d[3];
  D[r * N + cc + 8] = d[4];
  D[r * N + cc + 9] = d[5];
  D[(r + 8) * N + cc + 8] = d[6];
  D[(r + 8) * N + cc + 9] = d[7];
}

template <typename T, bool BF16>
void run(const char* name) {
  std::vector<float> fa(M * K), fb(N * K);
  std::vector<T> ha(M * K), hb(N * K);
  uint32_t s = 12345;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; };
  for (int i = 0; i < M * K; ++i) { ha[i] = (T)rnd(); fa[i] = (float)ha[i]; }
  for (int i = 0; i < N * K; ++i) { hb[i] = (T)rnd(); fb[i] = (float)hb[i]; }
  std::vector<float> ref(M * N, 0.f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)fa[m * K + k] * fb[n * K + k];
      ref[m * N + n] = (float)acc;
    }
  T *dA, *dB;
  float* dD;
  cudaMalloc(&dA, M * K * sizeof(T));
  cudaMalloc(&dB, N * K * sizeof(T));
  cudaMalloc(&dD, M * N * sizeof(float));
  cudaMemcpy(dA, ha.data(), M * K * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hb.data(), N * K * sizeof(T), cudaMemcpyHostToDevice);
  for (int a_regs = 0; a_regs < 2; ++a_regs)
    for (int swap = 0; swap < 2; ++swap) {
      cudaMemset(dD, 0, M * N * sizeof(float));
      probe<T, BF16><<<1, 128>>>(dA, dB, dD, a_regs, swap);
      cudaError_t e = cudaDeviceSynchronize();
      std::vector<float> out(M * N);
      cudaMemcpy(out.data(), dD, M * N * sizeof(float), cudaMemcpyDeviceToHost);
      double maxd = 0, maxr = 0;
      for (int i = 0; i < M * N; ++i) { maxd = fmax(maxd, fabs((double)out[i] - ref[i])); maxr = fmax(maxr, fabs((double)ref[i])); }
      printf("PROBE type=%s a=%s conv=%s err=%s max_abs_diff=%.3e ref_max=%.3e %s\n", name, a_regs ? "regs" : "smem",
             swap ? "lbo=SBO(swapped)" : "lbo=128,sbo=K*16", cudaGetErrorString(e), maxd, maxr,
             (e == cudaSuccess && maxd < 2e-2) ? "PASS" : "FAIL");
    }
  cudaFree(dA); cudaFree(dB); cudaFree(dD);
}

int main() {
  run<__half, false>("f16");
  run<__nv_bfloat16, true>("bf16");
  return 0;
}
