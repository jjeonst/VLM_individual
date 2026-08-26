#!/bin/bash
#SBATCH --job-name=p4-trainZero
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_train_zero.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_train_zero.err

# The blank control: the same policy, the same demonstrations, the same forty epochs, with the
# frame representation removed. What is left is the pose, the heading, the previous action and
# the target object -- everything an agent knows that is not the picture.
#
# It answers a question neither the paper nor our own numbers have addressed: how much of a
# success rate is available without seeing anything? The image encoder scores 25.2% on this
# benchmark against the paper's 11.6%, and until the blind floor is known there is no way to
# tell whether that gap is about representations at all.
#
# No staging and no embeddings. `--zero-tokens` now skips the token files rather than reading
# 365 GB an epoch to overwrite it with zeros, so a batch is 1.5 MB instead of 126 and an epoch
# is minutes. Everything else -- optimiser, schedule, loss weighting, epoch count -- is what the
# other conditions ran, because the point is a floor for *this* setup, not a different one.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536

nvidia-smi --query-gpu=name --format=csv,noheader

python train.py train --condition cot --run-name stageB_zero --zero-tokens
