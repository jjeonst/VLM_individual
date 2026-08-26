#!/bin/bash
#SBATCH --job-name=p4-trainA
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_train_stageA.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_train_stageA.err

# Behaviour cloning on Stage A's 782 trajectories for the paper's 40 epochs. The resulting
# success rate cannot be compared with Table 3 -- a tenth of a tenth of the data is not the
# paper's condition -- but a policy that trains here and moves sensibly in a familiar building
# is what licenses spending 150 GPU-hours on the full comparison.
#
# The eight-hour limit is generous on purpose. The arithmetic is about 18 s per epoch, so if
# this run approaches the limit the cause is reading, not computing, and the per-epoch line
# separates the two. Should reading dominate, the fix is to stage the embeddings on the node's
# /scratch -- the opposite of the rule we follow for RGB, because these are read forty times
# over rather than once.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py train --condition cot --run-name stageA_cot
