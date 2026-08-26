#!/bin/bash
#SBATCH --job-name=p4-evalA
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_eval_stageA.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_eval_stageA.err

# Closed-loop rollout in buildings the policy trained on. This is deliberately the easy case:
# the question is whether the agent can walk to a target and stop, not whether it generalises.
# The paper's number comes from the twenty unseen validation scenes, and Stage B will use those.
#
# Fifty episodes is chosen for time, not statistics. Every step calls the VLM, so this run is
# roughly as expensive per episode as encoding a training trajectory, and 500 steps is the cap
# for an episode that never stops. The measurement it yields -- average steps per episode --
# is what turns the Stage B evaluation estimate from a guess into arithmetic.
#
# Scene meshes go to the node's /scratch. Evaluation opens and closes a building per scene and
# that is small random reading, where NFS round-trip latency dominates; the same is not true of
# the embeddings, which are read sequentially and stay on /data.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python evaluate.py --run-name stageA_cot --condition cot --split train \
    --episodes 48 --parallel 8 --max-scenes 6 \
    --out /data/topovlm/checkpoints/pr2l_phase4/stageA_cot_eval_train.json
