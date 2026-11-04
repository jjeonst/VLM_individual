#!/bin/bash
#SBATCH --job-name=p4-probePrefixFit
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=bml-compute02
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_probe_prefix_fit.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_probe_prefix_fit.err

# TopoPrompt T0 (step 1-c), probe half: linear probes on what `probe_prefix.slurm` encoded.
# The result is C - D in sigma of balanced accuracy, per task, in `report.json`.
#
# A script rather than `sbatch --wrap`: a wrapped command runs under /bin/sh, which has no
# `source`, and job 47960 died on that line.
#
#   OUT     optional: a folder holding `frames/`, where the report goes as well

set -euo pipefail
source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OUT="${OUT:-/data/topovlm/probes/prefix}"
echo "[slurm] T0 탐침 | 출력 $OUT"
python probe_prefix.py probe --out "$OUT"
