#!/bin/bash
#SBATCH --job-name=p4-probeLen
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_probe_len.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_probe_len.err

# Appendix C.7's cheap check: how long the no-CoT answers run once the 32-token minimum is gone.
# Two rtx2080 cards in float16 -- see the docstring of `probe_answer_length.py` for why that is
# acceptable here and would not be for an encoding run.

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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python probe_answer_length.py "$@"
