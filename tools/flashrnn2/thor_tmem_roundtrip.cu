#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

__global__ void tmem_roundtrip(std::uint32_t* output) {
  __shared__ std::uint32_t address;
  const auto shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(&address));

  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
      : : "r"(shared_address) : "memory");
  __syncwarp();

  const std::uint32_t tmem_address = address;
  const std::uint32_t expected = 0x5a000000u ^ threadIdx.x;
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n\t"
      "tcgen05.wait::st.sync.aligned;"
      : : "r"(tmem_address), "r"(expected) : "memory");

  std::uint32_t actual;
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(actual) : "r"(tmem_address) : "memory");
  output[threadIdx.x] = actual;

  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n\t"
      "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
      : : "r"(tmem_address) : "memory");
}

int main() {
  int device = 0;
  cudaDeviceProp properties{};
  cudaError_t status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "device query: %s\n", cudaGetErrorString(status));
    return 2;
  }
  if (properties.major != 11 || properties.minor != 0) {
    std::fprintf(stderr, "expected SM110, found SM%d%d\n", properties.major,
                 properties.minor);
    return 2;
  }

  std::uint32_t* output = nullptr;
  status = cudaMallocManaged(&output, 32 * sizeof(*output));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "allocation: %s\n", cudaGetErrorString(status));
    return 2;
  }
  tmem_roundtrip<<<1, 32>>>(output);
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "kernel: %s\n", cudaGetErrorString(status));
    cudaFree(output);
    return 1;
  }

  for (std::uint32_t lane = 0; lane < 32; ++lane) {
    const std::uint32_t expected = 0x5a000000u ^ lane;
    if (output[lane] != expected) {
      std::fprintf(stderr, "lane %u: got %08x, expected %08x\n", lane,
                   output[lane], expected);
      cudaFree(output);
      return 1;
    }
  }
  cudaFree(output);
  std::printf("THOR_TMEM_ROUNDTRIP_PASS lanes=32 columns=32\n");
  return 0;
}
