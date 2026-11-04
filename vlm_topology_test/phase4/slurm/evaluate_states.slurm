#!/bin/bash
#SBATCH --job-name=p4-evalStates
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_eval_states.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_eval_states.err

# TopoPrompt step 1-a, second half: the checkpoint's own 500-episode evaluation, rerun exactly
# (bfloat16, one card, same episodes and seeds) with every step's LSTM input and output saved
# (`evaluate.py --save-states`), so `probe_lstm.py --test-on` can ask whether the LSTM holds
# revisits on the policy's own walk, where they are common, rather than on human demos.
#
#   RUN_NAME, CONDITION   checkpoint and its condition (default: the no-CoT min0 run)

set -euo pipefail
source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)
nvidia-smi --query-gpu=name --format=csv,noheader

RUN_NAME="${RUN_NAME:-stageB_nocotMin0C3}"
CONDITION="${CONDITION:-nocot_min0}"
SHARD="${SHARD:-$SLURM_ARRAY_TASK_ID}"
SHARDS="${SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
echo "[slurm] 상태 저장 평가 $RUN_NAME | $CONDITION | 샤드 $SHARD / $SHARDS"

# Stagger so two shards on one node do not read the checkpoint together (evaluate_shard.slurm).
STAGGER=$(( (SHARD % 4) * 240 ))
[ "$STAGGER" -gt 0 ] && { echo "[slurm] ${STAGGER}초 대기"; sleep "$STAGGER"; }

python evaluate.py --run-name "$RUN_NAME" --condition "$CONDITION" --split val --episodes 500 \
    --shard "$SHARD" --shards "$SHARDS" --label states \
    --save-states "/data/topovlm/probes/lstm/${RUN_NAME}_rollout"
