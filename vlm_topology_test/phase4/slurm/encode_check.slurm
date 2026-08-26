#!/bin/bash
#SBATCH --job-name=p4-enc-check
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/topovlm/slurm_logs/%j-p4_encode_check.out
#SBATCH --error=/data/topovlm/slurm_logs/%j-p4_encode_check.err

# Two things before committing any GPU-days to encoding:
#   1. `check` confirms the visual vectors are goal-blind and the text vectors are not, which
#      is only true if the tokens are being taken from where this code believes they are.
#   2. a short `encode` measures frames per second, which decides how the full run is sized.
# The 7B model in bf16 needs about 14 GB, so this asks for a 24 GB card.

source /home/jonghoon/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

cd /home/jonghoon/Projects/VLM_individual/vlm_topology_test/phase4

# Long generations leave the allocator holding many differently-sized blocks; expandable
# segments let it grow one region instead, which on the failing run was 5.9 GB of reserved
# but unusable memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/topovlm/vlm_weights/hf_cache
# Prismatic builds an empty Llama-2 shell from the gated meta-llama repo before loading
# the VLM checkpoint, so the request has to be authenticated or it fails with a 401.
export HF_TOKEN=$(cat /data/topovlm/vlm_weights/hf_token)

python encode.py check --condition cot

# One process, one model load, several batch sizes. The earlier version spawned a process per
# batch size and paid the ~250 s model load four times over.
python - <<'PYEOF'
import time, numpy as np, torch, encode
from vlm_features import load_vlm
records = encode.read_manifest()[:1]
vlm = load_vlm()
for batch in (8, 12, 16, 24):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    frames = min(48, records[0]["steps"])
    try:
        started = time.time(); seen = 0
        for chunk in encode.encode_trajectory(vlm, records[0], batch, True,
                                              indices=list(range(frames))):
            seen += len(chunk)
        elapsed = time.time() - started
        rate = seen / elapsed
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"batch {batch:3d}: {rate:5.2f} frames/s | 최대 메모리 {peak:5.2f} GiB | "
              f"조건당 {1219318/rate/3600:5.0f} GPU-시간", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"batch {batch:3d}: 메모리 부족", flush=True)
        torch.cuda.empty_cache()
PYEOF
