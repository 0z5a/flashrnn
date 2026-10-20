// Probe of NVSwitch multicast for the multi-GPU cluster forward (docs/flashrnn2/tmem-beta-handoff.md, "Multi-GPU").
// P GPUs run one CTA each (256 threads) concurrently. Iteration i: the CTA writes `words` 32-bit words of a pattern
// (rank, i, word) into its slot of the exchange buffer, parity i % 2, then releases flag[parity][rank] = i + 1;
// then it waits until every GPU's flag of this parity is >= i + 1 (ld.acquire.sys on its LOCAL copy), and checks
// every slot of this parity against the pattern. Two modes:
//   0  multicast: one store per word into the multicast mapping (the switch delivers it to every GPU's copy,
//      including this one's) and one flag store through the multicast mapping;
//   1  unicast fan-out: the same stores repeated into every GPU's unicast copy, and P flag stores.
// Reports mismatched words (sum over iterations) and the elapsed nanoseconds over the timed iterations. The protocol
// is the one hybrid_fwc_tmem.cu uses (data stores -> __syncthreads -> fence.sys -> release flag; acquire poll).
// Layout: data [2 parities][P slots][words] u32 at the buffer start; flags [2][P] int at flag_off bytes.
#include <cstdint>

__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned pattern(int rank, int i, int w) {
  return (unsigned)(rank + 1) * 0x01000193u ^ (unsigned)i * 0x9E3779B9u ^ (unsigned)w * 0x85EBCA6Bu;
}

extern "C" __global__ void nvls_probe_kernel(unsigned char* mc_base, const unsigned long long* uc_table, int flag_off,
                                             int rank, int P, int words, int iters, int warm, int mode,
                                             unsigned long long* out) {
  const int tid = threadIdx.x;
  __shared__ unsigned long long t0;
  __shared__ unsigned long long bad_total;
  if (tid == 0) bad_total = 0;
  __syncthreads();
  const int* my_flags = reinterpret_cast<const int*>(reinterpret_cast<unsigned char*>(uc_table[rank]) + flag_off);
  const unsigned* my_data = reinterpret_cast<const unsigned*>(uc_table[rank]);
  unsigned bad = 0;
  constexpr unsigned SPIN = 1u << 28;
  for (int i = 0; i < warm + iters; ++i) {
    if (i == warm) {
      __syncthreads();
      if (tid == 0) t0 = gtime();
    }
    const int par = i & 1;
    const size_t slot_words = (size_t)P * words;
    const size_t base_w = (size_t)par * slot_words + (size_t)rank * words;
    if (mode == 0) {
      unsigned* dst = reinterpret_cast<unsigned*>(mc_base) + base_w;
      for (int w = tid; w < words; w += 256) dst[w] = pattern(rank, i, w);
      __syncthreads();
      if (tid == 0) {
        __threadfence_system();
        st_release_sys(reinterpret_cast<int*>(mc_base + flag_off) + par * P + rank, i + 1);
      }
    } else {
      for (int q = 0; q < P; ++q) {
        unsigned* dst = reinterpret_cast<unsigned*>(uc_table[q]) + base_w;
        for (int w = tid; w < words; w += 256) dst[w] = pattern(rank, i, w);
      }
      __syncthreads();
      if (tid == 0) {
        __threadfence_system();
        for (int q = 0; q < P; ++q)
          st_release_sys(reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(uc_table[q]) + flag_off) + par * P + rank,
                         i + 1);
      }
    }
    // Wait for every GPU's block of this iteration.
    if (tid < P) {
      unsigned spins = 0;
      while (ld_acquire_sys(my_flags + par * P + tid) < i + 1 && ++spins < SPIN) {
      }
      if (spins >= SPIN) atomicAdd(reinterpret_cast<unsigned long long*>(&bad_total), 1ull << 40);  // timeout marker
    }
    __syncthreads();
    for (int q = 0; q < P; ++q) {
      const unsigned* src = my_data + (size_t)par * slot_words + (size_t)q * words;
      for (int w = tid; w < words; w += 256) bad += src[w] != pattern(q, i, w);
    }
    __syncthreads();  // everyone finished reading this parity before anyone (two iterations on) rewrites it
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&bad_total), (unsigned long long)bad);
  __syncthreads();
  if (tid == 0) {
    out[0] = bad_total;
    out[1] = gtime() - t0;
  }
}

// Host shim: one CTA of 256 threads on the given stream (the current device's context).
extern "C" int nvls_probe_launch(void* mc_base, const void* uc_table, int flag_off, int rank, int P, int words, int iters,
                                 int warm, int mode, void* out, void* stream) {
  nvls_probe_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(
      reinterpret_cast<unsigned char*>(mc_base), reinterpret_cast<const unsigned long long*>(uc_table), flag_off, rank, P,
      words, iters, warm, mode, reinterpret_cast<unsigned long long*>(out));
  return (int)cudaGetLastError();
}
