#!/bin/bash
#SBATCH --job-name=p4-encFull
#SBATCH --partition=rtx4090,rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/data/topovlm/slurm_logs/%A_%a-p4_encode_full.out
#SBATCH --error=/data/topovlm/slurm_logs/%A_%a-p4_encode_full.err

# Stage B: encode all 7,824 trajectories under the chain-of-thought prompt.
#
#   CONDITION   cot (the original run) | cot_fixed (re-encoded after the Appendix Z.1 fix)
#
# `cot_fixed` writes to its own directory because the encoder skips any trajectory whose file
# already exists -- pointed at the old directory it would find all 7,824 in place and do nothing.
# Keeping both also leaves the pre-fix embeddings intact, which every number reported so far
# rests on. PR2L_LEGACY_BOS is deliberately unset: this run is the corrected slice.
#
# Twelve shards of about 650 trajectories each, three at a time. The size is chosen so a shard
# finishes in roughly seven hours, well inside the twelve-hour limit: if one dies the loss is
# bounded, and a resubmission skips every trajectory whose file already exists. Stage A's 783
# are among those, so a tenth of the work is already done.
#
# **Four at a time, and started apart.** Jobs 10223, 10224 and 10229 were each killed with
# SIGTERM moments after the "Freezing Weights" line, which is where the 14 GB checkpoint is
# read, and every time it was a job still loading while others loaded alongside it. Ruled out
# by measurement: host memory (115 GB free when one died), the cards (2 and 3 both passed a
# 14 GB allocation and a matmul), the allocation (each job gets its own card) and account
# limits (there are none). The card count is the only real ceiling -- the node has four, and
# with three encoders running it still had 114 GB of its 125 GB free. So the throttle is four,
# and the sleep below staggers the first four by five minutes each so their loads do not
# overlap. After that the staggering is automatic, because a new task only starts when an
# earlier one finishes, and by then the others are long past loading.
#
# At 3.95 frames/s per card and 1.22M frames this is about 77 GPU-hours net: roughly 26 hours
# of wall time on three cards, 19 on four. Output is about 336 GB.

# **Which cards this can run on.** Two things have to be true, and VRAM is only the first.
#
#   rtx2080   11 GB, Turing sm_75    out: the 7B model needs about 14 GB
#   titanrtx  24 GB, Turing sm_75    out: bfloat16 needs sm_80. Prismatic loads Llama-2 in
#                                    bf16 and generates inside a bf16 autocast, and Turing has
#                                    no bf16 units. Fitting in memory is not the same as being
#                                    able to run, which is how this list first came to include it
#   rtx5090   32 GB, Blackwell       out: Appendix Z.3 measured generation 2.4-3.3x slower here,
#                                    the installed torch having no sm_120 kernels. A shard that
#                                    takes seven hours on a 3090 would take seventeen to
#                                    twenty-three and die on the twelve-hour wall as a TIMEOUT
#   rtx3090   24 GB, Ampere sm_86    in
#   rtx4090   24 GB, Ada sm_89       in
#
# The h200 would be faster than either but is the wrong tool to reach for when a 4090 fits.
#
# `--mem=40G` is doing double duty: it is more than the job needs, and on the 62 GB rtx4090 node
# it stops slurm placing two encoders side by side, which is how an evaluation job was killed
# with SIGTERM once already.

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

STAGGER=$(( (SLURM_ARRAY_TASK_ID % 4) * 300 ))
echo "[slurm] shard ${SLURM_ARRAY_TASK_ID}/12, ${STAGGER}초 후 시작"
sleep "$STAGGER"

python encode.py encode --condition "${CONDITION:-cot}" --batch-size 16 \
    --shard "$SLURM_ARRAY_TASK_ID" --shards 12
