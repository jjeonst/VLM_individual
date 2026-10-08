#!/bin/bash
#SBATCH --job-name=p4-evalAblate
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_eval_ablate.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_eval_ablate.err

# The language ablation: does a trained PR2L policy use the tokens after the image at all?
# The checkpoint is rolled out on the same 500 episodes as its own evaluation, with the live
# visual tokens kept and the question-and-answer span swapped for a stored training frame's
# (`build_language_pool.py`, `evaluate.py --ablate-language`).
#
# **rtx2080, two cards, float16.** Turing has no bfloat16 and the 7B model does not fit one 11 GB
# card, so the language model is split across two (`probe_answer_length.load_split_vlm`). The
# embeddings the policy trained on were made in bfloat16, so this run differs from the
# checkpoint's own evaluation in precision as well as in the ablation. That matters in one
# direction only: a success rate that holds at the original level means the language span is
# unused whatever the precision did, while a drop needs an unablated float16 run beside it before
# it can be put on the ablation. Generation is skipped -- the span it would produce is replaced --
# which is also what makes the slower cards affordable.
#
#   RUN_NAME   checkpoint, e.g. stageB_cotFixC3      CONDITION   its condition, e.g. cot_fixed

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# The cluster points XDG_CACHE_HOME at a per-job scratch directory, and huggingface_hub
# resolves its hub cache through that unless told otherwise -- so every job re-downloaded the
# 3.5 GB SigLIP weights. Naming the shared cache outright makes the weights a read from /data.
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SHARD="${SHARD:-$SLURM_ARRAY_TASK_ID}"
SHARDS="${SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
echo "[slurm] 언어 절제 ${ABLATE:-all} | ${RUN_NAME:?RUN_NAME 필요} | ${CONDITION:?CONDITION 필요} | 샤드 $SHARD / $SHARDS"

# Stagger the start so two shards on one node do not read the 14 GB checkpoint together
# (evaluate_shard.slurm has the history).
STAGGER=$(( (SHARD % 4) * 240 ))
if [ "$STAGGER" -gt 0 ]; then
    echo "[slurm] ${STAGGER}초 대기"
    sleep "$STAGGER"
fi

# ABLATE: all (question and answer together) | question | answer. EPISODE_IDS: optional JSON list,
# for running a named subset whose unablated result can be read off the checkpoint's own rows.
ABLATE="${ABLATE:-all}"
SELECT=(--episodes 500)
[ -n "${EPISODE_IDS:-}" ] && SELECT=(--episode-ids "$EPISODE_IDS")
python evaluate.py --run-name "$RUN_NAME" --condition "$CONDITION" --split val "${SELECT[@]}" \
    --shard "$SHARD" --shards "$SHARDS" \
    --ablate-language "$ABLATE" --split-gpus --label "${LABEL:-langswap}"
