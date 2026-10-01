#!/bin/bash
#SBATCH --job-name=p4-convFix
#SBATCH --partition=rtx4090
#SBATCH --nodelist=bml-compute10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_convert_fixed.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_convert_fixed.err

# Rewrite the staged `cot_fixed` token files as bare `.npy`, the way the original CoT run's were.
#
# The first attempt at training this condition read the `.npz` files directly and spent 2,974 of
# every 3,288 seconds in the loader -- 55 minutes an epoch against the 17 the original run
# managed. The difference is the zip container: `np.load` on an uncompressed `.npz` goes through
# Python's `zipfile`, a directory parse and a chunked copy per member, measured here at 42 MB/s
# cold against 120 for the same bytes in a bare array.
#
# The conversion existed already for the first CoT run. Its output was deleted during a scratch
# cleanup on 2026-09-02 -- the embeddings themselves were the point of that cleanup, and the
# derived `.npy` went with them without anyone noting that the next training run would need them
# rebuilt.
#
# **Pinned to compute10 because that is where the staged copy is.** `/scratch` is local to a
# node; the 391 GB this reads exists nowhere else. A GPU is requested only because this cluster's
# partitions refuse a job without one -- the work is entirely disk and CPU.
#
# Every converted file is read back and compared against its source. One silently corrupted
# trajectory in 7,824 would not show in a loss curve.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

ROOT=/scratch/jonghoon/embeddings_cot_fixed/embeddings/train
echo "[slurm] $(hostname) | $ROOT"
df -BG /scratch | tail -1

python convert_npy.py --root "$ROOT" --workers 12
