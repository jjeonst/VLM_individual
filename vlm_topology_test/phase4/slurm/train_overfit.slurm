#!/bin/bash
#SBATCH --job-name=p4-overfit
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_overfit.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_overfit.err

# Ask whether the policy can memorise twenty demonstrations. Generalisation is not required, so
# a failure here cannot be blamed on having too little data -- it means something between the
# stored representation and the optimiser is not connected. This is the first time train.py,
# dataset.load_trajectory and the collate path run against real files at all, so it is also the
# cheapest place for a shape or dtype mistake to surface.
#
# What to read in the log:
#   - accuracy climbing past 95% by epoch 200 (the script exits non-zero below that)
#   - every action's own accuracy climbing, not just the overall figure. Twenty trajectories are
#     still two thirds "move forward", so an overall 70% with 정지 at 0% is a collapsed policy.
#   - the read/compute split on the first epochs, which is the measurement that decides whether
#     the full run needs its embeddings staged on /scratch.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py overfit --condition cot --run-name stageA_overfit
