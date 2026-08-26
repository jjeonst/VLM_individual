#!/bin/bash
#SBATCH --job-name=p4-convNpy
#SBATCH --partition=rtx3090
#SBATCH --nodelist=bml-compute08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_convert.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_convert.err

# Rewrite the staged tokens as plain arrays. Measured on this node with the cache dropped, the
# zip container reads at 42 MB/s where the same bytes as .npy read at 120 -- and reading is 58%
# of what it costs to build a batch, which is 86% of an epoch.
#
# Pinned to compute08 because /scratch is local to it. A GPU is requested only because the
# partition requires one; nothing here computes.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

df -h /scratch | tail -1
python convert_npy.py --workers 12
df -h /scratch | tail -1
