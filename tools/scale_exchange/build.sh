#!/bin/bash
# Build the phase B exchange benchmark (two binaries) against copied NCCL 2.32.3 and NVSHMEM 3.9 builds.
# Usage: build.sh <deps dir with nccl/ and nvshmem/> <out dir>
set -euo pipefail
DEPS=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA=${CUDA_HOME:-/apps/ACC/CUDA/13.2}
MPI=${MPI_HOME:-/apps/ACC/OPENMPI/4.1.5/GCC}
NVCC="$CUDA/bin/nvcc"
MPI_INC="-I$MPI/include"
MPI_LIB="-L$MPI/lib -lmpi -Xlinker -rpath -Xlinker $MPI/lib"
ARCH="-gencode arch=compute_90,code=sm_90"
mkdir -p "$OUT"
$NVCC -O3 -std=c++17 $ARCH -lineinfo -I"$DEPS/nccl/include" $MPI_INC \
  "$HERE/exchange_bench.cu" -o "$OUT/exchange_gin" \
  -L"$DEPS/nccl/lib" -lnccl -Xlinker -rpath -Xlinker "$DEPS/nccl/lib" $MPI_LIB -Xcompiler -pthread
$NVCC -O3 -std=c++17 $ARCH -lineinfo -rdc=true -DBACKEND_SHMEM -I"$DEPS/nvshmem/include" $MPI_INC \
  "$HERE/exchange_bench.cu" -o "$OUT/exchange_shmem" \
  -L"$DEPS/nvshmem/lib" -lnvshmem_host -lnvshmem_device -lcuda -Xlinker -rpath -Xlinker "$DEPS/nvshmem/lib" \
  $MPI_LIB -Xcompiler -pthread
ls -la "$OUT"
