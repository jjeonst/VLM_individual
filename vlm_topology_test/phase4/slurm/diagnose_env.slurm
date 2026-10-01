#!/bin/bash
#SBATCH --job-name=p4-diag1a
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_diag1a.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_diag1a.err

# Follow-up to 1-A. One episode, driven through both loops with a single action sequence, so
# that a difference in what the policy sees can be told apart from a difference in what it
# chose. See the module docstring of diagnose_env.py.
#
#   INDEX  which episode of the subset to take (default 0)
#   STEPS  how many steps to compare (default 30)

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# The cluster points XDG_CACHE_HOME at a per-job scratch directory, and huggingface_hub
# resolves its hub cache through that unless told otherwise -- so every job re-downloaded the
# 3.5 GB SigLIP weights, and shard 16382_0 sat in that download for three and a half hours with
# the GPU idle before it was requeued. Naming the shared cache outright makes the weights a
# read from /data instead of a fetch from the network.
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

python diagnose_env.py --index "${INDEX:-0}" --steps "${STEPS:-30}" \
    --out "/data/topovlm/checkpoints/pr2l_phase4/diagnose_1a_${INDEX:-0}.json"
