#!/bin/bash
#SBATCH --job-name=p4-rand1b
#SBATCH --partition=rtx2080,rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_rand1b.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_rand1b.err

# Appendix W, experiment 1-B. Uniform random actions over the same 500 validation episodes,
# scored by the same success test. If this clears 3 %, the task or the scoring is loose and
# every number above it means less than it appears to.
#
# A GPU is requested only because habitat-sim renders through one; no model is loaded, and the
# weakest card that exists here is enough. Run at --parallel 1: episodes take turns inside a
# single simulator, and with no model to batch there is nothing for parallelism to overlap.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

python evaluate.py \
    --run-name stageB_random \
    --random-actions \
    --split val \
    --episodes 500 \
    --parallel 1 \
    --seed 0 \
    --out /data/topovlm/checkpoints/pr2l_phase4/stageB_random_eval_val.json
