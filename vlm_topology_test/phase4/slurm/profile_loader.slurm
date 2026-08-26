#!/bin/bash
#SBATCH --job-name=p4-profLoad
#SBATCH --partition=rtx3090
#SBATCH --nodelist=bml-compute08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_profile.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_profile.err

# Where the training loop's 2,867 seconds of waiting a epoch actually go.
#
# Pinned to compute08 because that is where the 365 GB of embeddings are staged; reading them
# over the shared volume instead would measure the volume, not the loader.
#
# A GPU is requested only because the partition demands one -- nothing here computes on it. The
# `pin_memory` path does depend on CUDA being present, and that path is part of what is timed.

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

ulimit -n 65536

python profile_loader.py --condition cot --workers "${WORKERS:-24}" --prefetch "${PREFETCH:-2}"
