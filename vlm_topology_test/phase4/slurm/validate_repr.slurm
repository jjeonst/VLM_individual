#!/bin/bash
#SBATCH --job-name=p4-repr3a
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:20:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_repr3a.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_repr3a.err

# Appendix W, experiment 3-A. Encode frames of a training trajectory through the path
# `evaluate.py` uses and compare against what `encode.py` stored, so that the encode-PCA-store
# chain is checked end to end rather than trusted.
#
# Run under legacy BOS: the stored files and the checkpoints all predate the `_assemble` fix,
# so legacy is the condition the policy actually learned. The script flips the switch itself to
# also report what the fix changes, which is why the variable is not exported here.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# The cluster points XDG_CACHE_HOME at a per-job scratch directory, and huggingface_hub
# resolves its hub cache through that unless told otherwise -- so every job re-downloaded the
# 3.5 GB SigLIP weights, and shard 16382_0 sat in that download for three and a half hours with
# the GPU idle before it was requeued. Naming the shared cache outright makes the weights a
# read from /data instead of a fetch from the network.
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python validate_representation.py \
    --condition cot \
    --trajectories 3 \
    --frames 4 \
    --out /data/topovlm/checkpoints/pr2l_phase4/validate_3a.json
