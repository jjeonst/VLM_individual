#!/bin/bash
#SBATCH --job-name=p4-enc-fit
#SBATCH --partition=rtx4090,rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_encode_fit.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_encode_fit.err

# Estimate the 4096 -> 1024 reduction from one frame per trajectory (7,824 frames) and freeze
# it. Everything encoded afterwards is reduced with this exact basis, including the frames
# encoded live during evaluation, so this runs once and its output is never regenerated.
#
# One frame per trajectory means one frame per prompt, so there is nothing to batch here; the
# cost is 7,824 single-frame generations.

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

# Long generations leave the allocator holding many differently-sized blocks; expandable
# segments let it grow one region instead, which on the failing run was 5.9 GB of reserved
# but unusable memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# The cluster points XDG_CACHE_HOME at a per-job scratch directory, and huggingface_hub
# resolves its hub cache through that unless told otherwise -- so every job re-downloaded the
# 3.5 GB SigLIP weights, and shard 16382_0 sat in that download for three and a half hours with
# the GPU idle before it was requeued. Naming the shared cache outright makes the weights a
# read from /data instead of a fetch from the network.
export HF_HUB_CACHE=$HF_HOME/hub
# Prismatic builds an empty Llama-2 shell from the gated meta-llama repo before loading
# the VLM checkpoint, so the request has to be authenticated or it fails with a 401.
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python encode.py fit --condition "${CONDITION:-cot}"
