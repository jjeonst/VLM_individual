#!/bin/bash
#SBATCH --job-name=p4-verify
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_verify_replay.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_verify_replay.err

# Does replaying the recorded actions put the agent where the person actually was?
# Compares the shipped navmesh against one rebuilt for the ObjectNav agent, and the episode's
# declared start pose against the one the recording itself begins from.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

python verify_replay.py --shard-index "${SHARD_INDEX:-0}" --episodes "${EPISODES:-40}"
