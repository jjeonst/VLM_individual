#!/bin/bash
#SBATCH --job-name=p4-render-pilot
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_render_pilot.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_render_pilot.err

# Pilot for step 1. Renders a couple of scenes so that the frames-per-second and the
# per-scene load time can be measured before committing to the full 1.0 TB run.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

python vlm_topology_test/phase4/render.py \
    --scenes 2 \
    --episodes-per-scene 20 \
    --manifest-dir /data/topovlm/habitat/episodes/pr2l_habitat_web_hd/train/pilot_shards
