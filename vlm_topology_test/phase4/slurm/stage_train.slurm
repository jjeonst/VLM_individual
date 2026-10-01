#!/bin/bash
#SBATCH --job-name=p4-stageTrain
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=40:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_stage_train.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_stage_train.err

# Stage, convert and train inside one allocation, so slurm may pick any node in the partition.
#
# The two steps cannot be separate jobs unless the node is pinned: `/scratch` is local, so a
# training job scheduled onto a different node than the one that staged would find nothing there
# and copy 356 GB again. Pinning is what the separate scripts do, and it costs the queue --
# a pinned job waits for one node while seven identical ones sit free. Doing both here trades
# that away: the allocation is held during the staging hour, but any 2080 node will serve.
#
#   CONDITION      embedding directory suffix (nocot, cot_fixed, ...)
#   RUN_NAME       checkpoint name
#   STAGE_DIR      node-local target; whichever node runs this creates its own copy
#   CLIP / EPS     recipe settings (Appendix C)
#   AMP            fp32 on Turing -- these cards have no bf16 units
#   SUMMARY_CHUNK  required here: at 78 tokens a 2,612-step demonstration needs 21 GB without it
#   NEEDED_GB      free space the guard demands
#
# Whether this is faster than a 24 GB card depends on how much of the data the node's page cache
# can hold, which depends on who else is on it. The arithmetic is about 3.5x slower here
# (measured 0.992 ms/frame against a 4090's 0.27), so it wins only when reading nearly vanishes.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[slurm] $(hostname) | ${CONDITION:?CONDITION 필요} → ${STAGE_DIR:?STAGE_DIR 필요}"

ROOT="/data/topovlm/habitat/embeddings/pr2l_habitat_web_hd_${CONDITION}"
EXPECTED="${EXPECTED:-7824}"
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
echo "[slurm] 매니페스트 궤적 ${COUNT} / ${EXPECTED}"
if [ "$COUNT" -ne "$EXPECTED" ]; then
    echo "[slurm] 인코딩이 온전하지 않다. 부족한 채로 학습하면 결과가 조용히 틀린다."
    exit 1
fi

NEEDED_GB="${NEEDED_GB:-400}"
FREE_GB=$(df -BG --output=avail "$(dirname "$STAGE_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
STAGED_GB=$(du -sBG "$STAGE_DIR" 2>/dev/null | tr -dc '0-9' || echo 0)
echo "[slurm] 로컬 여유 ${FREE_GB} GB, 이미 스테이징 ${STAGED_GB} GB (필요 ${NEEDED_GB} GB)"
if [ "${STAGED_GB:-0}" -lt 100 ] && { [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt "$NEEDED_GB" ]; }; then
    echo "[slurm] 로컬 디스크가 부족하다."
    exit 1
fi

free -g | head -2

python - "$CONDITION" "$STAGE_DIR" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "/home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4")
import dataset
from train import stage_locally

condition, target = sys.argv[1], Path(sys.argv[2])
root = Path(f"/data/topovlm/habitat/embeddings/pr2l_habitat_web_hd_{condition}")
records = dataset.read_manifest(root)
print(f"[stage] {len(records):,}개 궤적", flush=True)
stage_locally(records, root, target, workers=8)
PYEOF

python convert_npy.py --root "$STAGE_DIR/embeddings/train" --workers "${SLURM_CPUS_PER_TASK:-8}"

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
    --workers "${SLURM_CPUS_PER_TASK:-8}" --prefetch 2 "${RESUME[@]}"
