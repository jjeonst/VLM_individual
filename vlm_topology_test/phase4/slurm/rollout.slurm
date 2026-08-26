#!/bin/bash
#SBATCH --job-name=p4-rollout
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_rollout.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_rollout.err

# Thirty shortest-path-follower rollouts for the representation figure (Appendix H.2).
#
# On a 2080 because nothing here needs more: the card only renders, and no model is loaded.
# That leaves every bfloat16 card to the encoding this feeds.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

# `--spare` and the wider scene pool exist because the first run kept fewer trajectories than it
# reported: episode ids restart per building, so `{object}_{id}` collided and 28 manifest rows
# pointed at 25 files. Rollouts that walk far further than their recorded geodesic distance are
# now rejected outright as well, so the plan has to over-request to still land on thirty.
python rollout.py --split train --per-object 10 --spare 8 --per-scene 3 --scenes 10
