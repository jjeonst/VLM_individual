#!/bin/bash
#SBATCH --job-name=p4-figEnc
#SBATCH --partition=rtx5090,rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_fig_encode.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_fig_encode.err

# The representation figure's three encodings (Appendix H.2, Figure 8).
#
# Fifteen hundred frames, so this is minutes rather than hours: at the rates measured on the
# full run -- 3.95 frames/s with generation, 22.5 without -- the two prompted passes are about
# seven minutes each and the backbone pass under two.
#
# **bfloat16 cards only.** The same reason the image-encoder condition needed them: this figure
# compares the prompted representations against the backbone's, and a comparison between two
# precisions is not the comparison the figure claims to draw. `load_vision_backbone` refuses to
# start on a card without bfloat16 rather than quietly producing a shard at another precision.
#
#   CONDITION   cot | room | image

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python figure_encode.py --condition "${CONDITION:?CONDITION 필요}"
