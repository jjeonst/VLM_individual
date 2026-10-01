#!/bin/bash
#SBATCH --job-name=p4-cap
#SBATCH --partition=rtx3090,rtx4090,rtx5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_capacity.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_capacity.err

# Appendix X. Train the image-encoder policy at several recurrent sizes and see whether the gap
# to the paper survives. The paper never states the size -- it says only "the same LSTM as [43]"
# -- so ours came from PIRLNav by way of VC-1, and it is the largest single thing in the policy:
# 59.8 M of 78.8 M parameters, 75.8 %.
#
#   RUN_NAME     checkpoint name
#   HIDDEN       LSTM width
#   LAYERS       LSTM depth; 0 removes the recurrence entirely (Listing 1's shape)
#   SIDE         width of each non-visual embedding
#   SEED         initialisation seed
#
# The image encoder is the condition for three reasons (Appendix X.3): its gap is the one that
# cannot be blamed on a VLM checkpoint variant, since that path never touches the language model;
# its tokens are already stored, so only the policy changes; and 80 GB stages onto a node in
# minutes where CoT's 365 GB does not.
#
# **The seed rerun at the current size is not optional.** Without it a difference between L and S
# cannot be told from an unlucky initialisation, and the experiment says nothing.
#
# `--resume` is passed when a checkpoint for this run already exists. Slurm requeued one of these
# runs onto a different node five hours in and, with no resume, it began again at epoch 1 -- the
# checkpoint it needed was on disk the whole time, carrying the optimiser moments, the transition
# count and the history. The flag takes a path rather than being a bare switch, and `train.py`
# loads whatever it is given, so the file has to be checked for first: on the very first launch
# there is nothing to resume from.
#
# All four share one staging directory. `stage_locally` copies through a private temporary name
# and moves the result into place, so a run cannot read a file another is still writing.

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
if [ -f "$CHECKPOINT" ]; then
    RESUME=(--resume "$CHECKPOINT")
    echo "[slurm] 이어받기: $CHECKPOINT"
fi

python train.py train --condition image_encoder \
    --run-name "${RUN_NAME:?RUN_NAME 필요}" \
    --lstm-hidden "${HIDDEN:?HIDDEN 필요}" \
    --lstm-layers "${LAYERS:?LAYERS 필요}" \
    --side-dim "${SIDE:?SIDE 필요}" \
    --seed "${SEED:-0}" \
    "${RESUME[@]}" \
    --stage-to /scratch/jonghoon/embeddings_image
