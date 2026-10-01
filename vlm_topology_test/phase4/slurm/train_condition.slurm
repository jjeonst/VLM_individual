#!/bin/bash
#SBATCH --job-name=p4-trainCond
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_train_cond.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_train_cond.err

# Train one condition under one recipe, both named by environment variables. This generalises
# train_cot_fixed.slurm (condition fixed) and train_recipe.slurm (image encoder fixed) now that
# the recipe question has moved past the image encoder: Appendix C settled that the missing
# clip+eps pair explains that condition's gap, and the next runs apply the same recipe to the
# CoT and no-CoT conditions, which the two older scripts cannot express.
#
#   CONDITION   cot_fixed | nocot | image_encoder   (picks the embedding directory)
#   RUN_NAME    checkpoint name, required
#   CLIP        gradient norm cap, or empty for none
#   EPS         AdamW epsilon (default 1e-8, the PyTorch default this reproduction used)
#   AMP         bf16 (default, what every earlier number used) | fp32 (for rtx2080: Turing has
#               no bf16 units, so bf16 jobs die on the first step there; fp32 merely disables
#               autocast). Submitting to rtx2080 without AMP=fp32 is the titanrtx mistake again.
#   SUMMARY_CHUNK  frames per checkpointed group in the frame summary. Required on an 11 GB card
#               for the 78-token conditions: eleven demonstrations run past 1,232 steps and the
#               longest, at 2,612, needs 21.2 GB in float32 without it and 4.0 GB with a group
#               of 256. Output is identical (measured 0.000e+00), compute rises about 18%.
#   STAGE_DIR   node-local staging directory, required
#   NEEDED_GB   free space the guard demands (default 420, sized for the CoT embeddings)
#
# The default partitions are the bf16-capable ones. rtx5090 is excluded from the default list
# because policy training measured 1.5-2x slower there; rtx2080 must be chosen explicitly at
# submit time (-p rtx2080) together with AMP=fp32.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536

nvidia-smi --query-gpu=name --format=csv,noheader

# The manifest is checked before anything else, for the reason train_cot_fixed.slurm records:
# a dead encoding shard leaves trajectories silently absent and training would proceed on a
# fraction of the data, producing a number that looks like a result.
EXPECTED="${EXPECTED:-7824}"
ROOT="/data/topovlm/habitat/embeddings/pr2l_habitat_web_hd_${CONDITION:?CONDITION 필요}"
COUNT=$(python - "$ROOT" <<'PYEOF'
import sys
sys.path.insert(0, "/home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4")
from pathlib import Path
from dataset import read_manifest
try:
    print(len(read_manifest(Path(sys.argv[1]))))
except Exception:
    print(0)
PYEOF
)
echo "[slurm] 매니페스트 궤적 ${COUNT} / ${EXPECTED} (${ROOT})"
if [ "$COUNT" -ne "$EXPECTED" ]; then
    echo "[slurm] 인코딩이 온전하지 않다. 부족한 채로 학습하면 결과가 조용히 틀린다."
    exit 1
fi

STAGE_DIR="${STAGE_DIR:?STAGE_DIR 필요}"
NEEDED_GB="${NEEDED_GB:-420}"
FREE_GB=$(df -BG --output=avail "$(dirname "$STAGE_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
STAGED_GB=$(du -sBG "$STAGE_DIR" 2>/dev/null | tr -dc '0-9' || echo 0)
echo "[slurm] $(hostname) 로컬 여유 ${FREE_GB} GB, 이미 스테이징 ${STAGED_GB} GB (필요 ${NEEDED_GB} GB)"
if [ "${STAGED_GB:-0}" -lt 100 ] && { [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt "$NEEDED_GB" ]; }; then
    echo "[slurm] 로컬 디스크가 부족하다. 스테이징 없이 돌리면 epoch당 읽기가 열 배 이상 늘어난다."
    exit 1
fi

CHECKPOINT="/data/topovlm/checkpoints/pr2l_phase4/${RUN_NAME:?RUN_NAME 필요}.pt"
RESUME=()
[ -f "$CHECKPOINT" ] && { RESUME=(--resume "$CHECKPOINT"); echo "[slurm] 이어받기: $CHECKPOINT"; }

CLIPARG=()
[ -n "${CLIP:-}" ] && CLIPARG=(--grad-clip "$CLIP")

CHUNKARG=()
[ -n "${SUMMARY_CHUNK:-}" ] && CHUNKARG=(--summary-chunk "$SUMMARY_CHUNK")

python train.py train --condition "$CONDITION" --run-name "$RUN_NAME" \
    "${CLIPARG[@]}" --adam-eps "${EPS:-1e-8}" --amp "${AMP:-bf16}" --seed "${SEED:-0}" \
    "${CHUNKARG[@]}" --stage-to "$STAGE_DIR" \
    --workers "${SLURM_CPUS_PER_TASK:-12}" --prefetch 2 "${RESUME[@]}"
