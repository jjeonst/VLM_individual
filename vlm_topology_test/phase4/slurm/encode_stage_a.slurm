#!/bin/bash
#SBATCH --job-name=p4-encA
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_encode_stageA.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_encode_stageA.err

# Stage A: encode every tenth trajectory (783 of 7,824) under the chain-of-thought prompt.
# The point is not the number it will produce -- a tenth of the data cannot be compared with
# the paper -- but to drive the policy, training and evaluation code end to end before
# committing 250 GPU-hours to the full run.
#
# Batch 16 was measured as the throughput peak: 19.9 GiB of the card's 23.6, and 4.05 frames/s.
# Batch 24 is slower despite fitting, because the allocator thrashes near capacity.
#
# **Split across three of the node's four cards.** Stage A is shard 0 of 10, which is exactly
# the union of shards 0, 10 and 20 of 30 -- those are the indices congruent to 0, 10 and 20
# modulo 30, so between them they cover every index divisible by 10 and no index twice. Being
# disjoint, the jobs cannot collide over an output file, and any trajectory an earlier run
# finished is skipped because its file already exists. Submit one job per card:
#
#   for s in 0 10 20; do
#     sbatch --export=ALL,SHARD=$s,SHARDS=30 slurm/encode_stage_a.slurm
#   done
#
# **Start them apart, not together.** Jobs 10223, 10224 and 10229 were each killed with SIGTERM
# moments after the "Freezing Weights" line, which is where the 14 GB checkpoint is read. Every
# time, exactly the jobs still loading died and the ones already encoding lived. What was ruled
# out by measurement: host memory (the node had 115 GB free when one died), the cards (2 and 3
# each passed a 14 GB allocation and a matmul), the allocation itself (each job gets its own
# card), and account limits (there are none). What is left is contention while several
# processes read the same checkpoint at once, so give each one a head start:
#
#   for s in 0 10 20; do
#     sbatch --begin=now+$((s * 30)) --export=ALL,SHARD=$s,SHARDS=30 slurm/encode_stage_a.slurm
#   done
#
# Loading takes about eight minutes, so 5-minute gaps keep the reads from overlapping. Once a
# job is encoding it holds about 23 GB on its card and only a few GB of host memory.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

echo "[slurm] shard ${SHARD:-0} / ${SHARDS:-10}"
python encode.py encode --condition cot --batch-size 16 \
    --shard "${SHARD:-0}" --shards "${SHARDS:-10}"
