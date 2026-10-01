#!/bin/bash
#SBATCH --job-name=p4-stageConv
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_stage_conv.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_stage_conv.err

# Copy one condition's embeddings to this node's /scratch, then rewrite the token files as bare
# `.npy` alongside the `.npz` originals.
#
# Both steps exist so the training job that follows starts fast instead of discovering the
# problem live: training without local staging read /data forty times (516 s/epoch vs 28), and
# training from `.npz` spent 2,974 of every 3,288 seconds in the loader (the zip container --
# 42 MB/s cold against 120 for the same bytes in a bare array). Doing this in its own job, chained
# with --dependency, means the training job finds everything in place -- the earlier pattern of
# converting after training had already started once left a run at 55 minutes an epoch.
#
#   CONDITION   embedding directory suffix (e.g. nocot)
#   STAGE_DIR   node-local target, e.g. /scratch/jonghoon/embeddings_nocot
#
# A GPU is requested only because this cluster's partitions refuse a job without one; the work
# is disk and CPU. Pin the node at submit time (-w) -- /scratch is local, and the training job
# must be pinned to the same node.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

STAGE_DIR="${STAGE_DIR:?STAGE_DIR 필요}"
echo "[slurm] $(hostname) | ${CONDITION:?CONDITION 필요} → $STAGE_DIR"
df -BG /scratch | tail -1

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
stage_locally(records, root, target, workers=12)
PYEOF

python convert_npy.py --root "$STAGE_DIR/embeddings/train" --workers "${SLURM_CPUS_PER_TASK:-12}"
