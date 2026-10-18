#!/bin/bash
# Build the phase B exchange benchmark (two binaries) against copied NCCL 2.32.3 and NVSHMEM 3.9 builds.
# Usage: build.sh <deps dir with nccl/ and nvshmem/> <out dir>
set -euo pipefail
DEPS=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda}
MPI=${MPI_HOME:?set MPI_HOME to the MPI installation}
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
g++ -O2 -std=c++17 -DSCHED_MAX_ROUNDS=64 -DSCHED_MAX_SENDS=64 "$HERE/schedule_check.cpp" -o "$OUT/schedule_check"
# Build stamp: which sources produced which binaries.
{
  echo '{"sources": {'
  first=1; for f in exchange_bench.cu schedule.h schedule_check.cpp build.sh; do
    [ $first = 1 ] || echo ','; first=0; printf '"%s": "%s"' "$f" "$(sha256sum "$HERE/$f" | cut -d' ' -f1)"; done
  echo '}, "binaries": {'
  first=1; for f in exchange_gin exchange_shmem schedule_check; do
    [ $first = 1 ] || echo ','; first=0; printf '"%s": "%s"' "$f" "$(sha256sum "$OUT/$f" | cut -d' ' -f1)"; done
  echo '}, "cuda": "'"$CUDA"'", "mpi": "'"$MPI"'", "built": "'"$(date -Is)"'"}'
} > "$OUT/build-stamp.json"
ls -la "$OUT"
