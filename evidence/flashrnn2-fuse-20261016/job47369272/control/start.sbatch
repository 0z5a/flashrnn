#!/bin/bash
#SBATCH --job-name=209
#SBATCH --account=ehpc390
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --comment=flashrnn2-fuse-20261016-v2
#SBATCH --output=/gpfs/home/tueb/tueb043681/flashrnn2-fuse-20261016-v2/control/slurm-%j.stdout
#SBATCH --error=/gpfs/home/tueb/tueb043681/flashrnn2-fuse-20261016-v2/control/slurm-%j.stderr
set -euo pipefail
exec env -u PYTHONHOME -u PYTHONPATH -u LD_PRELOAD \
  LD_LIBRARY_PATH=/apps/ACC/PYTHON/3.12.1/GCC/lib:/usr/lib64 \
  /apps/ACC/PYTHON/3.12.1/GCC/bin/python3 -I -B -S \
  /gpfs/scratch/ehpc390/tueb043681/short-test-start-command-20261013-v1/offline_batch.py \
  --manifest /gpfs/home/tueb/tueb043681/flashrnn2-fuse-20261016-v2/control/start-command.json
