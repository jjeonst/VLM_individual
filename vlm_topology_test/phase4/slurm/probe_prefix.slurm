#!/bin/bash
#SBATCH --job-name=p4-probePrefix
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_probe_prefix.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_probe_prefix.err

# TopoPrompt T0 (step 1-c), encode half: demonstration frames encoded with no sentence (A), the
# true topology sentence (C) and a shuffled one (D), plus the room name of every node.
# `probe_prefix.py` has the method; `probe` runs afterwards as its own job on the output.
#
# **rtx2080, two cards, float16** (`probe_answer_length.load_split_vlm`). All three conditions
# are encoded the same way, so the precision is shared and the comparison is not affected by it.
#
#   LIMIT   optional: a smoke run on this many trajectories, into its own folder, one shard

set -euo pipefail
source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SHARD="${SHARD:-$SLURM_ARRAY_TASK_ID}"
SHARDS="${SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
OUT=/data/topovlm/probes/prefix
EXTRA=()
if [ -n "${LIMIT:-}" ]; then
    OUT=/data/topovlm/probes/prefix_smoke
    EXTRA=(--limit "$LIMIT")
fi
echo "[slurm] T0 인코딩 | 샤드 $SHARD / $SHARDS | 출력 $OUT ${LIMIT:+| 궤적 $LIMIT개}"

# Stagger so shards on one node do not read the checkpoint together (evaluate_shard.slurm).
STAGGER=$(( (SHARD % 4) * 240 ))
[ "$STAGGER" -gt 0 ] && { echo "[slurm] ${STAGGER}초 대기"; sleep "$STAGGER"; }

python probe_prefix.py encode --out "$OUT" --shard "$SHARD" --shards "$SHARDS" --split-gpus \
    "${EXTRA[@]}"
[ -z "${LIMIT:-}" ] || python probe_prefix.py probe --out "$OUT"
