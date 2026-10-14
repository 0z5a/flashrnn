#!/usr/bin/env bash
set -euo pipefail
BASE=/gpfs/scratch/ehpc390/tueb043681/flashrnn2-rtiled-m6-20261014-v2
# The wrapper interpreter path must not leak into the module-built runtime.
unset LD_LIBRARY_PATH
source /gpfs/home/tueb/tueb043681/flashns-math376-job47032336/activate_flashns.sh
# Module activation sets CC=nvc; Triton builds its launcher modules with $CC and needs gcc.
export CC=/usr/bin/gcc CXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST=9.0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Pinned FlashRNN1 baseline Python deps (einops etc.); its CUDA builds use the module's CUDA 11.8 (torch cu118).
V1_DEPS=/gpfs/scratch/ehpc390/tueb043681/paper-baselines-20261013-v1/deps
export PYTHONPATH="$BASE/source:/gpfs/scratch/ehpc390/tueb043681/flashrnn2-distributed-20261011-v2/deps36:$V1_DEPS:${PYTHONPATH:-}"
# TileLang 0.1.15 (approved VPS wheel route, offline install) only for TileLang tasks, with CUDA 12.8.
if [ "${FRNN_TILELANG:-0}" = "1" ]; then
  export CUDA_HOME=/apps/ACC/CUDA/12.8 PATH="/apps/ACC/CUDA/12.8/bin:$PATH"
  export PYTHONPATH="/gpfs/scratch/ehpc390/tueb043681/flashrnn2-tilelang-assets-20261014-v1/site:$PYTHONPATH"
fi
# Compile caches stay on node-local disk: the group scratch quota is nearly full.
LOCAL=/scratch/tmp/tueb043681-rtiled-m6-${SLURM_JOB_ID:-dry}-${FRNN_CACHE_TAG:-default}
if ! mkdir -p "$LOCAL" 2>/dev/null; then LOCAL="$BASE/runtime-cache/${FRNN_CACHE_TAG:-default}"; fi
export TRITON_CACHE_DIR="$LOCAL/triton36" TMPDIR="$LOCAL/tmp" TILELANG_CACHE_DIR="$LOCAL/tilelang"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
exec ${FRNN_PREFIX:-} python3 -B "$@"
