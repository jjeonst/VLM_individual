#!/bin/bash
#SBATCH --job-name=p4-enc-fit
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_encode_fit.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_encode_fit.err

# Estimate the 4096 -> 1024 reduction from one frame per trajectory (7,824 frames) and freeze
# it. Everything encoded afterwards is reduced with this exact basis, including the frames
# encoded live during evaluation, so this runs once and its output is never regenerated.
#
# One frame per trajectory means one frame per prompt, so there is nothing to batch here; the
# cost is 7,824 single-frame generations.

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

python encode.py fit --condition "${CONDITION:-cot}"
