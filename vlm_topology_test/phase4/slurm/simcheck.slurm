#!/bin/bash
#SBATCH --job-name=p4-simchk
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_simcheck.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_simcheck.err

# Does stepping several episodes through one shared simulator give the same result as running
# each on its own? It has to, because habitat-sim allows only one simulator per process and
# evaluation still needs to batch frames, so the episodes take turns and restore their agent
# state before each render and each move. This checks that claim instead of asserting it: the
# same action sequences are played both ways and the final pose, distance walked and first
# observation are compared. No model is loaded, so it costs a couple of minutes.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

python /data/topovlm/simcheck.py
