#!/bin/bash
#SBATCH --job-name=p4-attn
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_attention.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_attention.err

# Where the trained CLS query looks. No model is loaded and only a few trajectories are read,
# so this costs a couple of minutes. See attention.py for why it is the one structural question
# the parameter counts cannot answer.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

python attention.py --run-name stageA_cot --condition cot --trajectories 12
