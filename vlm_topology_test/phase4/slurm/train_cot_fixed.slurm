#!/bin/bash
#SBATCH --job-name=p4-trainCoTFix
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_train_cot_fixed.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_train_cot_fixed.err

# PR2L with chain-of-thought prompting, trained on all 7,824 trajectories. This is the paper's
# own condition -- the one Table 3 reports as 41.9% -- and everything else in Stage B exists to
# give it something to be compared against.
#
# **Workers follow the allocation.** The first attempt at this condition ran 24 readers on
# compute10, which has twelve cores, and shared them with another user's job -- CPULoad was 20 on
# a 12-core node. The 24 came from the original script, written for compute08's 48 cores. Reading
# `SLURM_CPUS_PER_TASK` means the number cannot drift from what was actually granted.
#
# **rtx4090 is allowed too, and the disk guard below is what makes that safe.** The original run
# named rtx3090 alone because compute08 had the 3.4 TB needed for staging, but that node has been
# fully allocated for days and waiting on it once cost twenty hours. compute10 has 2.0 TB free,
# which is ample for 420 GB. The memory request came down from 64 GB to 48 -- 64 was sized for
# 24 workers at prefetch 8, and prefetch is 2 now, so the readers hold about 13 GB. 64 GB also
# exceeded compute10's 62 GB outright, which would have kept the job off that node no matter how
# much disk it had.
#
# **rtx3090, for the disk rather than the card.** The embeddings are about 360 GB and have to
# sit on the node's own disk, for the reason measured during the image-encoder run: reading them
# forty times over the shared volume took 516 seconds an epoch and reading them from local disk
# took 28. compute08 has 3.4 TB free; the 5090 nodes have under 600 GB and are already holding
# the image-encoder staging. The card makes little difference here -- the policy is 79M
# parameters, and the run is bounded by the summary layer's attention over 78 tokens a frame.
#
# Expect roughly 900 seconds an epoch of arithmetic, against the image encoder's 334: the same
# LSTM, but 78 tokens to attend over instead of 16, and attention is quadratic in that count.
# Forty epochs is twelve hours or so, plus the staging copy.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The DataLoader passes each tensor of each queued batch as its own file descriptor, so the
# default soft limit of 1024 is reached well before the prefetch depth this run wants. `train.py`
# raises it for itself as well; this line covers anything the process starts before that.
ulimit -n 65536


# **The manifest is checked before anything else.** This job is chained behind a twelve-shard
# encoding array, and a shard that dies leaves its trajectories simply absent -- `read_manifest`
# returns whatever is there and training proceeds happily on a fraction of the data, producing a
# number that looks like a result. `afterok` on the array covers a shard that exits non-zero but
# not one that ends up short for any other reason, so the count is verified here rather than
# assumed. 7,824 is the subsample size fixed in Plan section 2.2 and confirmed at every earlier
# stage.
EXPECTED=7824
COUNT=$(python - <<'PYEOF'
import sys
sys.path.insert(0, "/home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4")
from pathlib import Path
from dataset import read_manifest
root = Path("/data/topovlm/habitat/embeddings/pr2l_habitat_web_hd_cot_fixed")
try:
    print(len(read_manifest(root)))
except Exception:
    print(0)
PYEOF
)
echo "[slurm] 매니페스트 궤적 ${COUNT} / ${EXPECTED}"
if [ "$COUNT" -ne "$EXPECTED" ]; then
    echo "[slurm] 인코딩이 온전하지 않다. 부족한 채로 학습하면 결과가 조용히 틀린다."
    exit 1
fi

STAGE_DIR=/scratch/jonghoon/embeddings_cot_fixed
NEEDED_GB=420          # 360 GB of embeddings, with room for the run to not sit at the very edge
FREE_GB=$(df -BG --output=avail "$(dirname "$STAGE_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')

echo "[slurm] $(hostname) 로컬 여유 ${FREE_GB} GB (필요 ${NEEDED_GB} GB)"
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt "$NEEDED_GB" ]; then
    echo "[slurm] 로컬 디스크가 부족하다. 스테이징 없이 돌리면 epoch당 읽기가 열 배 이상 늘어난다."
    exit 1
fi

# 24 readers, not the default 8. Measured on this node: the raw disk saturates at four threads
# and about 650 MB/s, but the training loop only achieved 195 MB/s with eight workers, each of
# them at 26% CPU -- neither the disk nor the processor was the limit. What scales with the
# worker count is `np.load`'s trip through Python's zipfile, roughly 25 MB/s apiece, and eight of
# those is exactly the 200 MB/s observed. Twenty-four should approach the disk ceiling instead.
#
# Epoch one took 2357s, of which 1875s was waiting for data against 481s of arithmetic. Forty of
# those is 26 hours, past the 24-hour limit the first attempt was given.
# `--prefetch 2`, the default, after measuring 8 and finding it four times worse.
#
# The reasoning for raising it was that the GPU sat idle 86% of each epoch while no resource was
# saturated, which looked like the DataLoader's in-order delivery stalling behind a slow worker.
# Running each worker further ahead should have absorbed that. What actually happened: a batch
# is 270 MB rather than the 110 MB estimated, so 24 workers x 8 batches held 64 GB of shared
# memory, the page cache was evicted to make room, and reads that had been served from RAM went
# to disk instead. Throughput fell from about 130 MB/s to 22, and an epoch from 57 minutes to an
# estimated 3h45m. The prefetch depth was competing with the cache that made the reads fast.
#
# `--resume` because forty epochs do not fit in one allocation and the wall clock is not
# negotiable without an administrator. Checkpoints written from here on carry the optimiser
# state and the schedule position; the one being resumed from predates that, so AdamW's moments
# rebuild once -- a few dozen updates, recorded as a deviation.
# Resume only if a checkpoint is already there. Forty epochs do not fit in one allocation, and
# a requeue with no resume starts again at epoch one -- which is what happened to a capacity run.
CHECKPOINT=/data/topovlm/checkpoints/pr2l_phase4/stageB_cot_fixed.pt
RESUME=()
[ -f "$CHECKPOINT" ] && { RESUME=(--resume "$CHECKPOINT"); echo "[slurm] 이어받기: $CHECKPOINT"; }

python train.py train --condition cot_fixed --run-name stageB_cot_fixed --stage-to "$STAGE_DIR" \
    --workers "${SLURM_CPUS_PER_TASK:-12}" --prefetch 2 "${RESUME[@]}"
