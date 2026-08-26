#!/bin/bash
#SBATCH --job-name=p4-trainCoT
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_train_cot.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_train_cot.err

# PR2L with chain-of-thought prompting, trained on all 7,824 trajectories. This is the paper's
# own condition -- the one Table 3 reports as 41.9% -- and everything else in Stage B exists to
# give it something to be compared against.
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

STAGE_DIR=/scratch/jonghoon/embeddings_cot
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
python train.py train --condition cot --run-name stageB_cot --stage-to "$STAGE_DIR" \
    --workers 24 --prefetch 2 --resume /data/topovlm/checkpoints/pr2l_phase4/stageB_cot.pt
