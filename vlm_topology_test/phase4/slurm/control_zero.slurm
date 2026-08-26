#!/bin/bash
#SBATCH --job-name=p4-ctrl0
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_control_zero.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_control_zero.err

# The control that tells us whether the VLM is contributing anything.
#
# The same policy, the same data, the same forty epochs -- but the frame representations are
# blanked, so all the network has is where it stands, which way it faces, what it did last and
# what it is looking for. Whatever this run achieves is achievable without seeing anything.
# If it matches the real run's success rate, the representation is decorative and there is no
# point spending 267 GPU-hours on the paper's comparison until we know why.
#
# Both halves run here because both are cheap. Training reads the same embeddings and blanks
# them on the GPU, so it costs what the real run cost. Evaluation never builds the model at
# all -- no encoding, no generation, no 7B forward pass -- so it finishes in minutes rather
# than the hours its counterpart takes.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py train --condition cot --run-name stageA_zero --zero-tokens || exit 1

python evaluate.py --run-name stageA_zero --condition cot --split train \
    --episodes 48 --parallel 8 --max-scenes 6 --zero-tokens \
    --out /data/topovlm/checkpoints/pr2l_phase4/stageA_zero_eval_train.json
