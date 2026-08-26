#!/bin/bash
#SBATCH --job-name=p4-encVis
#SBATCH --partition=rtx5090,rtx4090,rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_encode_vision.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_encode_vision.err

# The image-encoder control: all 7,824 trajectories through the vision backbone alone.
#
# This is the paper's comparison (a) for Habitat -- "a policy on Prismatic VLM image encoder
# embeddings" -- and the one condition that shares its pixels with PR2L while sharing none of
# the language model. Both read the same 224px frame through the same frozen DINOv2 and SigLIP;
# PR2L then spends 32 layers of Llama-2 and 40 generated tokens on it, and this stops at the
# backbone. That is why Table 3's 41.9% against 11.6% is attributable to the prompting rather
# than to the pixels.
#
# **Any bfloat16 card, whichever frees up first.** Loading only the vision half needs about
# 3 GB rather than 20, so this does not have to wait for the cards PR2L is using -- but it
# does have to be a card that does bfloat16 in hardware, because `load_vlm` runs the
# backbone in bfloat16 and the control is only a control if the arithmetic matches. That
# rules out the idle Turing cards, plentiful though they are. The weights themselves are
# the same either way: Prismatic's checkpoint holds only the projector and the language
# model, so the backbone comes from timm in both paths.
#
# Listing several partitions lets the scheduler take the first bfloat16 card that frees,
# and the throttle keeps four at most so this never crowds out the encoding it feeds.
#
# The bottleneck here is not the card. Each frame is a 900 KB read and two PIL resizes down to
# 224px, against a single forward pass through 0.7B parameters -- so the shards are sized by
# CPU count, and six of them fit in the free cores.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

# The backbone is small and comes from timm's cache, so loads do not collide the way the 14 GB
# language-model checkpoint did. A short stagger costs nothing and keeps the first reads apart.
sleep $(( SLURM_ARRAY_TASK_ID * 60 ))

# SHARDS is overridable so the same script can run a quick pilot over a fraction of the
# data before six of these are committed: sbatch --array=0-0 --export=ALL,SHARDS=800 ...
python encode.py encode --condition image_encoder --batch-size 32 \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "${SHARDS:-6}"
