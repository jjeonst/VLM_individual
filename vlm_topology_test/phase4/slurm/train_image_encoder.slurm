#!/bin/bash
#SBATCH --job-name=p4-trainImg
#SBATCH --partition=rtx5090,rtx4090,rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_train_image.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_train_image.err

# The image-encoder control, trained on all 7,824 trajectories.
#
# Nothing here differs from the chain-of-thought run except the width of the first projection,
# and even that is read from the manifest rather than set: 2176 for the fused DINOv2+SigLIP
# patches against PR2L's 2048. Appendix C.2's general item 5 asks for exactly this -- "the exact
# same policy architecture and hyperparameters" -- so that a difference in the final numbers can
# only have come from the representation.
#
# Faster than its counterpart for two reasons, both consequences of 16 tokens a frame instead of
# 78: the embeddings are 80 GB rather than 360, and the summary layer's attention is quadratic
# in the token count. What does not shrink is the LSTM, which is 60M parameters regardless, and
# in this condition it is most of the arithmetic.
#
# **What would be suspicious.** This should train without trouble and still produce a worse
# policy -- that is the paper's claim (11.6% against 41.9%). Two outcomes would mean something
# is wrong rather than merely worse: a result close to the CoT run, which would say the two
# representations are not reaching the policy differently; or a collapse to zero stop accuracy
# like the blank-token control, which would say the visual features are not arriving at all.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Staged to the node's own disk first. Measured on compute15 while the encoding jobs were
# running: /data gave 24.6 MB/s, the copy 590, the local disk 828. Forty passes over 80 GB
# is 36 hours the slow way and 1.1 the fast one, and the page cache cannot help because
# 80 GB does not fit in the node's 62 GB.
python train.py train --condition image_encoder --run-name stageB_image \
    --stage-to /scratch/jonghoon/embeddings_image
