#!/bin/bash
#SBATCH --job-name=p4-probeRooms
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=bml-compute02
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_probe_rooms.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_probe_rooms.err

# TopoPrompt step 1-d (a): ground-truth room of every T0 place-graph node, then the VLM's
# room-name accuracy. `probe_rooms.py` has the method. The VLM answers are T0's cache, so the
# GPU only renders a 64 x 64 semantic view per node; one rtx2080 is plenty. Needs T0's encoding
# finished (the node answers live in its output).
#
#   LIMIT   optional: a smoke run on this many trajectories, scored from its own labels file

set -euo pipefail
source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -n "${LIMIT:-}" ]; then
    python probe_rooms.py label --limit "$LIMIT"
    python probe_rooms.py score --labels labels_smoke.jsonl
else
    python probe_rooms.py label
    python probe_rooms.py score
fi
