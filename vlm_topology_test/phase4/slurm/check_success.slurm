#!/bin/bash
#SBATCH --job-name=p4-succChk
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
# Any card will do: no model is loaded, only the pathfinder.
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_succ_check.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_succ_check.err

# Is our success test the same test habitat-lab applies?
#
# The paper does not define success; Appendix C.1 defers to "the defaults provided by Habitat,
# as specified in the HM3D ObjectNav configuration file". So the question is answerable by
# comparison rather than by reading, and this runs the comparison: the same agent positions
# scored both ways, ours and the one transcribed from habitat/tasks/nav/nav.py.
#
# On a 2080 and without loading any model -- only the pathfinder is exercised.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

python evaluate.py --run-name stageB_image --condition image_encoder --split val \
    --episodes 40 --check-success
