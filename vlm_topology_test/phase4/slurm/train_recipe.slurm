#!/bin/bash
#SBATCH --job-name=p4-recipe
#SBATCH --partition=rtx3090,rtx4090,rtx5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_recipe.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_recipe.err

# Appendix C. Train the image-encoder policy under VC-1's behaviour-cloning recipe and see
# whether the gap to the paper survives.
#
# VC-1's config clips the gradient norm at 0.2 and runs AdamW with eps 1e-5; this reproduction
# had neither. Measured here, the gradient norm at initialisation is 0.448, so a 0.2 cap binds
# from the first update rather than sitting unused.
#
#   RUN_NAME   checkpoint name
#   CLIP       gradient norm cap, or empty for none
#   EPS        AdamW epsilon
#   SEED       initialisation seed
#
# Clipping and eps are varied one at a time. Changing both at once would say the recipe matters
# without saying which half, which is the mistake Appendix X made with width and depth and had
# to spend two more runs undoing.
#
# `--resume` when a checkpoint is already there: a requeue with no resume starts again at
# epoch 1, which happened to one of the capacity runs.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

nvidia-smi --query-gpu=name --format=csv,noheader

CHECKPOINT="/data/topovlm/checkpoints/pr2l_phase4/${RUN_NAME}.pt"
RESUME=()
[ -f "$CHECKPOINT" ] && { RESUME=(--resume "$CHECKPOINT"); echo "[slurm] 이어받기: $CHECKPOINT"; }

CLIPARG=()
[ -n "${CLIP:-}" ] && CLIPARG=(--grad-clip "$CLIP")

python train.py train --condition image_encoder \
    --run-name "${RUN_NAME:?RUN_NAME 필요}" \
    "${CLIPARG[@]}" \
    --adam-eps "${EPS:-1e-8}" \
    --seed "${SEED:-0}" \
    "${RESUME[@]}" \
    --stage-to /scratch/jonghoon/embeddings_image
