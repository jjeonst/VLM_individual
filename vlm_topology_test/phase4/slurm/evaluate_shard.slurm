#!/bin/bash
#SBATCH --job-name=p4-evalShard
#SBATCH --partition=rtx5090,rtx4090,rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_eval_shard.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_eval_shard.err

# The validation run, split four ways by scene. One process on one 5090 was pacing at over
# thirteen hours for twenty scenes, because the CoT condition decodes 32-48 tokens for every
# frame of every episode; four processes finish in a little over three.
#
# **The shards may land on different GPU models.** The image-encoder baseline this is compared
# against ran entirely on one 5090, so a difference in floating point between architectures is
# an uncontrolled variable here. It was raised and the split was chosen anyway; the deviation
# belongs in whatever reports the number.
#
# **One shard per node.** compute10 has 62 GB of RAM for two GPUs and slurm is not accounting
# memory there -- `AllocTRES` carries cpu and gpu but no mem -- so it will happily place two
# shards that each need to hold a 7B model in host RAM while loading. It did, and one was killed
# with SIGTERM five minutes in, mid-load. `--exclusive` is too blunt (it would take both GPUs);
# asking for most of the node's memory makes the scheduler space the shards out by itself.
#
#   CONDITION   cot | image_encoder     RUN_NAME    the checkpoint to roll out
#   SHARD/SHARDS  override the array variables, for re-running a single failed shard

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

nvidia-smi --query-gpu=name --format=csv,noheader

# A re-run of one shard is submitted as a one-element array, where SLURM_ARRAY_TASK_COUNT is 1
# and would silently widen the slice to every remaining scene. SHARDS overrides it.
SHARD="${SHARD:-$SLURM_ARRAY_TASK_ID}"
SHARDS="${SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
echo "[slurm] 샤드 $SHARD / $SHARDS"

python evaluate.py --run-name "${RUN_NAME:?RUN_NAME 필요}" \
    --condition "${CONDITION:?CONDITION 필요}" --split val --episodes 500 \
    --shard "$SHARD" --shards "$SHARDS"
