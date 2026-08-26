#!/bin/bash
#SBATCH --job-name=p4-evalVal
#SBATCH --partition=rtx5090,rtx4090,rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_eval_val.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_eval_val.err

# The number the whole reproduction has been working towards: closed-loop success in the twenty
# validation scenes the policy has never seen. This is the setting of the paper's Table 3, so
# it is the only result here that can be put beside its 41.9% and 11.6%.
#
# Five hundred episodes rather than the paper's two thousand. The gap being measured is about
# thirty points and five hundred episodes carry a standard error near two, so the comparison
# survives the reduction while the run does not take four times as long.
#
# `--max-scenes` is deliberately absent. Concentrating the sample into a few buildings was the
# right trade for a smoke test, where the question was whether the loop worked at all; here the
# question is how the policy fares across houses, and breadth is the whole point.
#
#   CONDITION   cot | image_encoder     RUN_NAME    the checkpoint to roll out
#   EPISODES    default 500
#
# The policy is rebuilt at the width recorded in its checkpoint, so the same script serves both
# conditions without being told which is which.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python evaluate.py --run-name "${RUN_NAME:?RUN_NAME 필요}" \
    --condition "${CONDITION:?CONDITION 필요}" --split val \
    --episodes "${EPISODES:-500}" --parallel 8 \
    --out "/data/topovlm/checkpoints/pr2l_phase4/${RUN_NAME}_eval_val.json"
