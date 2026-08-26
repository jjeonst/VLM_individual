#!/bin/bash
#SBATCH --job-name=p4-encode
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_encode.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_encode.err

# Encode every rendered frame. Trajectories are dealt out round-robin across array tasks, so
# each task holds a mix of scenes and lengths and they finish at roughly the same time.
#
# Concurrency, array size and batch size are all deliberately set from the pilot measurement
# rather than guessed; see IMPLEMENTATION.md. Already-encoded trajectories are skipped, so a
# task that dies can simply be resubmitted.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

# Long generations leave the allocator holding many differently-sized blocks; expandable
# segments let it grow one region instead, which on the failing run was 5.9 GB of reserved
# but unusable memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# Prismatic builds an empty Llama-2 shell from the gated meta-llama repo before loading
# the VLM checkpoint, so the request has to be authenticated or it fails with a 401.
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python encode.py encode \
    --condition "${CONDITION:-cot}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --shards "${SLURM_ARRAY_TASK_COUNT}"
