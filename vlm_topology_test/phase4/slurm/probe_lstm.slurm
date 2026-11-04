#!/bin/bash
#SBATCH --job-name=p4-probeLSTM
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_probe_lstm.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_probe_lstm.err

# TopoPrompt step 1-a: how much of the walked path the trained policy's LSTM state holds.
# `probe_lstm.py` has the method. The policy alone, no VLM, so one rtx2080 is enough; the time
# goes to reading the stored embeddings (193 GB for nocot_min0) off /data.
#
#   RUN_NAME, CONDITION   checkpoint and the embeddings it was trained on
#   LIMIT                 optional: a smoke run on this many trajectories, into its own folder
#   PROBE_ONLY=1          skip extract and refit the probes on the vectors already on /data
#   TEST_ON               optional: score on these rollouts instead (evaluate_states.slurm)

set -euo pipefail
source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RUN_NAME="${RUN_NAME:-stageB_nocotMin0C3}"
CONDITION="${CONDITION:-nocot_min0}"
OUT=/data/topovlm/probes/lstm
EXTRA=()
if [ -n "${LIMIT:-}" ]; then
    OUT=/data/topovlm/probes/lstm_smoke
    EXTRA=(--limit "$LIMIT")
fi
echo "[slurm] $RUN_NAME | $CONDITION | 출력 $OUT ${LIMIT:+| 궤적 $LIMIT개}"

[ -n "${PROBE_ONLY:-}" ] || python probe_lstm.py extract --run-name "$RUN_NAME" --condition "$CONDITION" --out "$OUT" "${EXTRA[@]}"
[ -n "${TEST_ON:-}" ] && EXTRA+=(--test-on "$TEST_ON")
python probe_lstm.py probe --run-name "$RUN_NAME" --condition "$CONDITION" --out "$OUT" "${EXTRA[@]}"
