#!/bin/bash
#SBATCH --job-name=p4-render
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-79%8
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_render.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_render.err

# Step 1: replay every selected demonstration into RGB, actions and pose.
#
# One array task per scene, because loading a building's mesh costs far more than replaying a
# demonstration through it. Concurrency is capped at 8 so that this leaves most of the shared
# rtx2080 partition free for other people; at the measured 61 frames/s a scene takes about
# four and a half minutes, so the whole set finishes in well under an hour anyway.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

python vlm_topology_test/phase4/render.py --shard-index "${SLURM_ARRAY_TASK_ID}"
