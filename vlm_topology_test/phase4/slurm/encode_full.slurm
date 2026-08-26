#!/bin/bash
#SBATCH --job-name=p4-encFull
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_encode_full.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_encode_full.err

# Stage B: encode all 7,824 trajectories under the chain-of-thought prompt.
#
# Twelve shards of about 650 trajectories each, three at a time. The size is chosen so a shard
# finishes in roughly seven hours, well inside the twelve-hour limit: if one dies the loss is
# bounded, and a resubmission skips every trajectory whose file already exists. Stage A's 783
# are among those, so a tenth of the work is already done.
#
# **Four at a time, and started apart.** Jobs 10223, 10224 and 10229 were each killed with
# SIGTERM moments after the "Freezing Weights" line, which is where the 14 GB checkpoint is
# read, and every time it was a job still loading while others loaded alongside it. Ruled out
# by measurement: host memory (115 GB free when one died), the cards (2 and 3 both passed a
# 14 GB allocation and a matmul), the allocation (each job gets its own card) and account
# limits (there are none). The card count is the only real ceiling -- the node has four, and
# with three encoders running it still had 114 GB of its 125 GB free. So the throttle is four,
# and the sleep below staggers the first four by five minutes each so their loads do not
# overlap. After that the staggering is automatic, because a new task only starts when an
# earlier one finishes, and by then the others are long past loading.
#
# At 3.95 frames/s per card and 1.22M frames this is about 77 GPU-hours net: roughly 26 hours
# of wall time on three cards, 19 on four. Output is about 336 GB.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

STAGGER=$(( (SLURM_ARRAY_TASK_ID % 4) * 300 ))
echo "[slurm] shard ${SLURM_ARRAY_TASK_ID}/12, ${STAGGER}초 후 시작"
sleep "$STAGGER"

python encode.py encode --condition cot --batch-size 16 \
    --shard "$SLURM_ARRAY_TASK_ID" --shards 12
