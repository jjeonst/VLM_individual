#!/bin/bash
#SBATCH --job-name=p4-evEpoch
#SBATCH --partition=rtx4090,rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_ev_epoch.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_ev_epoch.err

# Roll out an intermediate checkpoint on the same 500 episodes the epoch-40 numbers came from,
# so that success rate can be read as a curve against training epoch rather than a single point.
# The question it answers is whether our policies simply fitted further than the paper's did --
# the paper reports no training accuracy, so the only way to place its numbers on our curve is
# to have a curve.
#
# **PR2L_LEGACY_BOS is set for the cot condition and this is not optional.** Every stored
# embedding and every checkpoint predates the `_assemble` fix (Appendix Z.1), and the epoch-40
# evaluation that these will be compared against ran before the fix was synced. Encoding under
# today's code would feed the policy a representation it never trained on, and the resulting
# drop would be indistinguishable from a convergence effect -- which is the very thing being
# measured here. The image encoder never touches `_assemble`, so the switch does nothing there.
#
# **Listing a partition is the only lever; there is no soft preference here.** The nodes declare
# no features (`AvailableFeatures=(null)`), so `--prefer` has nothing to match on, and every
# partition carries the same PriorityTier and PriorityJobFactor, so the order of the names in
# `--partition=a,b,c` means nothing to the scheduler -- it takes whichever can start the job. A
# submission that names rtx5090 will land there whenever a 5090 happens to be free, even with
# 3090s idle. Naming only the partitions whose cards you actually want is the whole mechanism.
#
# **The 5090s are excluded and this is about more than speed.** Appendix Z.3 measured 423-582 s
# per CoT episode there against 137 s on a 3090 and 177 s on a 4090, because the installed torch
# has no sm_120 kernels and falls back to JIT. The stronger reason is that every epoch-40 CoT
# shard ran on a 3090 or a 4090 -- none on a 5090 -- and 3-A showed that a change in kernels
# moves the logits enough to flip a sampled token and send the generated text elsewhere. Running
# the intermediate epochs on a different architecture would put that difference inside a
# comparison whose whole subject is training epoch.
#
#   RUN_NAME      checkpoint to roll out, e.g. stageB_cot_epoch020
#   CONDITION     cot | image_encoder
#   SHARD/SHARDS  optional; leave unset to run every scene in one process
#   EPISODE_IDS   optional path to a JSON list of episode ids, for running a named subset

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

: "${RUN_NAME:?RUN_NAME 필요}"
: "${CONDITION:?CONDITION 필요}"

if [ "$CONDITION" = "cot" ]; then
    export PR2L_LEGACY_BOS=1
    echo "[slurm] PR2L_LEGACY_BOS=1 — 체크포인트가 학습한 표현으로 인코딩한다"
fi

nvidia-smi --query-gpu=name --format=csv,noheader
echo "[slurm] $RUN_NAME | $CONDITION | 샤드 ${SHARD:-전체}/${SHARDS:-1}"

SELECT=(--episodes "${EPISODES:-500}")
if [ -n "${EPISODE_IDS:-}" ]; then
    SELECT=(--episode-ids "$EPISODE_IDS")
fi

# A run that is not the 500-episode one has to be told where to write. `evaluate.py` names its
# output after the checkpoint, so a full-split run of stageB_image would land on top of the
# 500-episode file that every comparison in this project is anchored to.
WHERE=()
if [ -n "${OUT:-}" ]; then
    WHERE=(--out "$OUT")
fi

# Appendix Z.2's control. Off unless asked for, because every success rate reported so far was
# produced without it and a run that switches it on cannot be set beside those.
PRECISION=()
[ "${MATCH_PRECISION:-}" = "1" ] && { PRECISION=(--match-train-precision); \
    echo "[slurm] --match-train-precision — 학습 데이터와 같은 float16으로 반올림한다"; }

if [ -n "${SHARD:-}" ]; then
    python evaluate.py --run-name "$RUN_NAME" --condition "$CONDITION" \
        --split val "${SELECT[@]}" "${WHERE[@]}" "${PRECISION[@]}" \
        --shard "$SHARD" --shards "${SHARDS:?SHARDS 필요}"
else
    python evaluate.py --run-name "$RUN_NAME" --condition "$CONDITION" \
        --split val "${SELECT[@]}" "${WHERE[@]}" "${PRECISION[@]}"
fi
