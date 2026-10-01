#!/bin/bash
#SBATCH --job-name=p4-migrate
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_migrate.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_migrate.err

# Pick a run up off a slower card and finish it on a faster one.
#
# A training run writes its whole state -- weights, optimiser moments, schedule position -- every
# epoch, so it can be stopped anywhere and resumed elsewhere. What it cannot survive is two
# processes holding the same run name at once: both would write the same checkpoint file and the
# reader would get whichever bytes landed last. So the first thing here is to stop the other job
# and *wait for it to be gone*, not merely ask.
#
# `torch.save` writes in place rather than through a temporary, so a cancellation landing mid-save
# can leave a short file. The checkpoint is therefore loaded and inspected before anything else
# starts; a run that reaches this point with a damaged checkpoint should stop loudly rather than
# begin again from epoch one, which is the failure that cost the capacity experiment a day.
#
#   CANCEL_JOB   the job to stop first (the one on the slower card)
#   CONDITION / RUN_NAME / STAGE_DIR / CLIP / EPS   as in train_condition.slurm

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536

nvidia-smi --query-gpu=name --format=csv,noheader
CHECKPOINT="/data/topovlm/checkpoints/pr2l_phase4/${RUN_NAME:?RUN_NAME 필요}.pt"

if [ -n "${CANCEL_JOB:-}" ]; then
    echo "[slurm] $CANCEL_JOB 를 중단하고 사라질 때까지 기다린다"
    scancel "$CANCEL_JOB" 2>/dev/null || true
    for _ in $(seq 1 60); do
        squeue -j "$CANCEL_JOB" -h 2>/dev/null | grep -q . || break
        sleep 10
    done
    if squeue -j "$CANCEL_JOB" -h 2>/dev/null | grep -q .; then
        echo "[slurm] $CANCEL_JOB 가 10분 안에 끝나지 않았다. 두 잡이 같은 체크포인트를 쓰면 안 된다."
        exit 1
    fi
    # The cancelled job may have been part-way through writing when it was killed.
    sleep 20
fi

python - "$CHECKPOINT" <<'PYEOF'
import sys, torch
path = sys.argv[1]
try:
    state = torch.load(path, map_location="cpu")
except Exception as error:
    print(f"[migrate] 체크포인트를 읽을 수 없다: {type(error).__name__}: {error}")
    raise SystemExit(1)
missing = [k for k in ("policy", "optimiser", "epoch", "seen") if k not in state]
if missing:
    print(f"[migrate] 체크포인트에 {missing} 가 없다")
    raise SystemExit(1)
print(f"[migrate] epoch {state['epoch']} 까지 학습됨, 전이 {state['seen']:,} | "
      f"클리핑 {state.get('grad_clip')} eps {state.get('adam_eps')} amp {state.get('amp')}")
PYEOF

CLIPARG=()
[ -n "${CLIP:-}" ] && CLIPARG=(--grad-clip "$CLIP")
CHUNKARG=()
[ -n "${SUMMARY_CHUNK:-}" ] && CHUNKARG=(--summary-chunk "$SUMMARY_CHUNK")

python train.py train --condition "${CONDITION:?CONDITION 필요}" --run-name "$RUN_NAME" \
    "${CLIPARG[@]}" --adam-eps "${EPS:-1e-8}" --amp "${AMP:-bf16}" --seed "${SEED:-0}" \
    "${CHUNKARG[@]}" --stage-to "${STAGE_DIR:?STAGE_DIR 필요}" \
    --workers "${SLURM_CPUS_PER_TASK:-24}" --prefetch 2 --resume "$CHECKPOINT"
