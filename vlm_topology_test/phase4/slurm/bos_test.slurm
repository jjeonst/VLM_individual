#!/bin/bash
#SBATCH --job-name=p4-bos
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_bos.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_bos.err

# Is position 0 of the prefill the BOS token or the first image patch? Causal masking makes it
# measurable: position 0 attends to nothing but itself, so if it is BOS its hidden state cannot
# depend on the image. Two different images through the same prompt settle it, and with it
# whether `_assemble`'s visual slice is off by one (Appendix Z.1).
#
# Written as a script rather than `sbatch --wrap` because the wrapped form runs under `sh`,
# where `source` does not exist -- the first attempt died in one second with "source: not found"
# and never reached python.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python bos_test.py
