#!/bin/bash
#SBATCH --job-name=p4-env1a
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_env1a.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_env1a.err

# Appendix W, experiment 1-A. The same image-encoder checkpoint rolled out inside habitat-lab's
# own `Env`, so that the rollout loop, the episode termination and the scoring come from the
# standard implementation instead of ours. Everything this project reports was produced by
# `evaluate.py`, and this is the only check that puts a different loop around the same policy.
#
# The criterion is the aggregate success rate over the whole validation split, not a per-episode
# match. The diagnostic in Appendix V.9 showed the two loops feed the policy identical inputs and
# still part ways when a four-way argmax flips on a near tie -- one flip and a closed loop never
# comes back. Episodes cannot be paired, so the comparison needs the numbers 2,000 episodes give.
#
#   SHARD/SHARDS  which slice of the scenes this process runs

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# The cluster points XDG_CACHE_HOME at a per-job scratch directory, and huggingface_hub
# resolves its hub cache through that unless told otherwise -- so every job re-downloaded the
# 3.5 GB SigLIP weights, and shard 16382_0 sat in that download for three and a half hours with
# the GPU idle before it was requeued. Naming the shared cache outright makes the weights a
# read from /data instead of a fetch from the network.
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)
# Four shards share one node and slurm's --cpus-per-task is not enforced on torch: unset, it
# sizes its pools from the machine (16 intra-op, 32 inter-op here) and the four processes put
# 248 threads on 48 cores. Pinning them to what was actually requested stops the shards taking
# cores back off each other.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

nvidia-smi --query-gpu=name --format=csv,noheader

python validate_env.py \
    --run-name stageB_image \
    --shard "${SHARD:?SHARD 필요}" --shards "${SHARDS:?SHARDS 필요}" \
    --out "/data/topovlm/checkpoints/pr2l_phase4/validate_1a_shard0${SHARD}.json"
