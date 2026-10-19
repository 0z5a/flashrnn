// TMA (bulk-copy) cluster multicast against per-CTA loads of the same exchange tile (report-only microbenchmark,
// max-width.md section 67). Every CTA of a 128-CTA grid needs, per step, the same BYTES of an L2-resident buffer in
// shared memory (as the 2-D backward stages its dG K block at B = 16: 8-16 KB per step).
//   mode 0 (own):       each CTA bulk-copies all BYTES itself (cp.async.bulk, one mbarrier), then __syncthreads.
//   mode 1 (multicast): CTA rank r of a C-CTA cluster copies slice r (BYTES / C) with .multicast::cluster to every CTA
//                       of the cluster; each CTA waits for all BYTES on its own mbarrier. Double-buffered shared tile;
//                       one cluster barrier per step keeps a slice write from overtaking a peer still reading the
//                       buffer of two steps ago.
//   mode 2 (own+sync):  mode 0 plus the same per-step cluster barrier (control: separates the barrier's cost from the
//                       multicast copy's).
// Every step reads a different (step % 64) tile of a 64-tile buffer, and every CTA sums its tile (checked against the
// host sum), so the copies cannot be elided. Output: median µs per step over 7 launches.
// Build: nvcc -std=c++17 -O3 -arch=sm_90a tma_multicast_bench.cu -o tma_mc
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect(unsigned long long* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_addr(bar)),
      "r"(parity)
      : "memory");
}

template <int MODE>
__global__ void bench(const uint4* __restrict__ src, int bytes, int steps, float* out) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) unsigned long long bar[2];
  unsigned char* buf[2] = {smem, smem + bytes};
  cg::cluster_group cluster = cg::this_cluster();
  const int csize = (int)cluster.num_blocks(), crank = (int)cluster.block_rank();
  if (threadIdx.x == 0) {
    mbar_init(&bar[0], 1);
    mbar_init(&bar[1], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster.sync();
  float acc = 0.f;
  const unsigned short mask = (unsigned short)((1u << csize) - 1);
  for (int s = 0; s < steps; ++s) {
    const int b = s & 1;
    const unsigned char* tile = reinterpret_cast<const unsigned char*>(src) + (size_t)(s % 64) * bytes;
    if (MODE >= 1) cluster.sync();  // every CTA done with buf[b] (read at step s - 2) before any peer refills it
    if (threadIdx.x == 0) {
      mbar_expect(&bar[b], bytes);
      if (MODE != 1) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         smem_addr(buf[b])),
                     "l"(tile), "r"(bytes), "r"(smem_addr(&bar[b]))
                     : "memory");
      } else {
        const int slice = bytes / csize;
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], "
            "%4;" ::"r"(smem_addr(buf[b] + crank * slice)),
            "l"(tile + crank * slice), "r"(slice), "r"(smem_addr(&bar[b])), "h"(mask)
            : "memory");
      }
    }
    mbar_wait(&bar[b], (s >> 1) & 1);
    // consume: every thread sums its share of the tile (as the GEMM reads its A operand)
    const float* f = reinterpret_cast<const float*>(buf[b]);
    for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x) acc += f[i];
    __syncthreads();
  }
  if (MODE >= 1) cluster.sync();  // no CTA exits while a peer may still multicast into it
  atomicAdd(out, acc);
}

int main(int argc, char** argv) {
  const int ctas = 128, steps = 2048;
  const int sizes[] = {8192, 16384, 32768, 65536};
  const int clusters[] = {1, 2, 4, 8};
  size_t total = (size_t)64 * 65536;
  std::vector<float> host(total / 4);
  for (size_t i = 0; i < host.size(); ++i) host[i] = (float)((i * 2654435761u) % 1000) * 1e-3f;
  uint4* src;
  float* out;
  cudaMalloc(&src, total);
  cudaMalloc(&out, sizeof(float));
  cudaMemcpy(src, host.data(), total, cudaMemcpyHostToDevice);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  printf("{\"rows\": [\n");
  bool first = true;
  for (int bytes : sizes) {
    double ref = 0;  // expected sum per launch: every CTA sums every visited tile
    for (int s = 0; s < steps; ++s)
      for (int i = 0; i < bytes / 4; ++i) ref += host[(size_t)(s % 64) * bytes / 4 + i];
    ref *= ctas;
    for (int c : clusters) {
      for (int mode = 0; mode < 3; ++mode) {
        if (mode >= 1 && c == 1) continue;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(ctas);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = 2 * bytes;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = c;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        auto kern = mode == 0 ? bench<0> : mode == 1 ? bench<1> : bench<2>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * bytes);
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        std::vector<float> us;
        float sum = 0;
        cudaError_t err = cudaSuccess;
        for (int rep = 0; rep < 8; ++rep) {
          cudaMemset(out, 0, sizeof(float));
          cudaEventRecord(e0);
          err = cudaLaunchKernelEx(&cfg, kern, (const uint4*)src, bytes, steps, out);
          cudaEventRecord(e1);
          if (err == cudaSuccess) err = cudaEventSynchronize(e1);
          if (err != cudaSuccess) break;
          float ms;
          cudaEventElapsedTime(&ms, e0, e1);
          if (rep) us.push_back(ms * 1000.f / steps);  // first launch is the warm-up
          cudaMemcpy(&sum, out, sizeof(float), cudaMemcpyDeviceToHost);
        }
        std::sort(us.begin(), us.end());
        const bool ok = err == cudaSuccess && std::abs(sum - ref) <= 1e-3 * ref;
        printf("%s {\"bytes\": %d, \"cluster\": %d, \"mode\": \"%s\", \"us_per_step\": %.4f, \"sum_ok\": %s, "
               "\"error\": \"%s\"}",
               first ? " " : ",\n ", bytes, c, mode == 0 ? "own" : mode == 1 ? "multicast" : "own+sync", us.empty() ? -1.0 : us[us.size() / 2],
               ok ? "true" : "false", cudaGetErrorString(err));
        first = false;
        if (err != cudaSuccess) cudaGetLastError();
      }
    }
  }
  printf("\n]}\n");
  return 0;
}
