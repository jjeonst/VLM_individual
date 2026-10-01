#!/bin/bash
#SBATCH --job-name=p4-evalShard
#SBATCH --partition=rtx3090,rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2

# **rtx5090 is not in the list, and the reason is measured twice.** Appendix Z.3 found generation
# 2.4-3.3x slower there, the installed torch having no sm_120 kernels. Encoding was kept off
# those cards for that reason; evaluation was not, on the guess that a four-hour shard would
# still fit the twelve-hour wall. It does not. Four shards have now died there: 14126_0 and
# 14126_1 reached one and two scenes of five in seven and a half hours before being moved, and
# 15754_0 and 15754_2 ran the full twelve and were killed at scene 2 and 4 with nothing written.
# A shard that TIMEOUTs loses everything, because results are written once at the end.
#
# **`%2` and 48G, because the node has to actually have the memory.** 15754_1 and 15754_3 asked
# for 56G on compute10, which has 64 GB total and other users on it, and both were killed during
# the checkpoint load. 15181 finished the same load inside 32 GB, so 56 was over-asking -- and
# over-asking is not free: it decides which nodes can hold the job and how many can run at once.
# 48G lets two share compute08's 128 GB, which `%2` then holds them to.
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_eval_shard.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_eval_shard.err

# The validation run, split four ways by scene. One process on one 5090 was pacing at over
# thirteen hours for twenty scenes, because the CoT condition decodes 32-48 tokens for every
# frame of every episode; four processes finish in a little over three.
#
# **The shards may land on different GPU models.** The image-encoder baseline this is compared
# against ran entirely on one 5090, so a difference in floating point between architectures is
# an uncontrolled variable here. It was raised and the split was chosen anyway; the deviation
# belongs in whatever reports the number.
#
# **One shard per node.** compute10 has 62 GB of RAM for two GPUs and slurm is not accounting
# memory there -- `AllocTRES` carries cpu and gpu but no mem -- so it will happily place two
# shards that each need to hold a 7B model in host RAM while loading. It did, and one was killed
# with SIGTERM five minutes in, mid-load. `--exclusive` is too blunt (it would take both GPUs);
# asking for most of the node's memory makes the scheduler space the shards out by itself.
#
#   CONDITION   cot | image_encoder     RUN_NAME    the checkpoint to roll out
#   SHARD/SHARDS  override the array variables, for re-running a single failed shard

set -euo pipefail

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# The cluster points XDG_CACHE_HOME at a per-job scratch directory, and huggingface_hub
# resolves its hub cache through that unless told otherwise -- so every job re-downloaded the
# 3.5 GB SigLIP weights, and shard 16382_0 sat in that download for three and a half hours with
# the GPU idle before it was requeued. Naming the shared cache outright makes the weights a
# read from /data instead of a fetch from the network.
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

nvidia-smi --query-gpu=name --format=csv,noheader

# A re-run of one shard is submitted as a one-element array, where SLURM_ARRAY_TASK_COUNT is 1
# and would silently widen the slice to every remaining scene. SHARDS overrides it.
SHARD="${SHARD:-$SLURM_ARRAY_TASK_ID}"
SHARDS="${SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
echo "[slurm] 샤드 $SHARD / $SHARDS"

# **Stagger the start, as the encoding array does.** Reading the 14 GB checkpoint is where these
# jobs die: 10223, 10224 and 10229 went that way during encoding, and so did 14925, 14926 and
# 15180 here, each killed with SIGTERM moments after "Freezing Weights" while another job on the
# same node was loading alongside it. Memory is not the cause -- 15181 finished the same load
# with the same 32 GB allowance and ran on.
#
# `--begin` does not solve this. It staggers when jobs become *eligible*, and two jobs that then
# sit in the queue for hours become runnable at the same moment and start together, which is
# exactly how 15180 died three minutes into 15181's load. The delay has to happen after the
# allocation is granted, which means here.
STAGGER=$(( (SHARD % 4) * 240 ))
if [ "$STAGGER" -gt 0 ]; then
    echo "[slurm] ${STAGGER}초 대기 — 같은 노드에서 체크포인트를 동시에 읽지 않도록"
    sleep "$STAGGER"
fi

python evaluate.py --run-name "${RUN_NAME:?RUN_NAME 필요}" \
    --condition "${CONDITION:?CONDITION 필요}" --split val --episodes 500 \
    --shard "$SHARD" --shards "$SHARDS"
