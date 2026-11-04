#!/bin/bash
#SBATCH --job-name=p4-probeDoors
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=bml-compute02
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_probe_doors.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_probe_doors.err

# TopoPrompt step 1-d (b): open vs blocked doorway pairs, the VLM's answer, and a linear probe
# on the policy question's hidden state. `probe_doors.py` has the method.
#
# **rtx2080, two cards, float16** for the VLM (`probe_answer_length.load_split_vlm`), as T0.
# Rendering and probing run in the same job: rendering needs only the first card and takes
# about an hour, and the probe is small.
#
#   LIMIT      optional: this many scenes, for a smoke run
#   DOORS_OUT  optional: another output folder (read by `probe_doors.py`)

set -euo pipefail
source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python probe_doors.py render ${LIMIT:+--limit "$LIMIT"}
python probe_doors.py encode --split-gpus
python probe_doors.py probe
