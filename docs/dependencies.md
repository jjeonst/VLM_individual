# TopoVLM Dependencies

## Decision

The canonical environment is `topovlm` with Python 3.9.

Rationale:

- Habitat-Sim is required for the Habitat-first runtime path.
- On `bmlslurm`, `aihabitat` stable `habitat-sim 0.3.3` resolves to `py3.9` builds.
- PyTorch 2.2.0, CUDA 11.8, Prismatic, and the TopoVLM package can run under Python 3.9.

Python 3.10 and 3.12 are future compatibility targets only after Habitat-Sim,
Prismatic import, and any `flash-attn` requirement are verified together on `bmlslurm`.

## Create From Commands

Use these commands when creating the env manually:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n topovlm python=3.9 pip cmake=3.27 -c conda-forge -y
conda activate topovlm
conda install -y habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y "mkl<2025" "intel-openmp<2025" -c defaults
python -m pip install --upgrade pip
python -m pip install -e ".[dev,habitat]"
python -m pip install "scikit-learn" "transformers==4.38.1" "huggingface-hub<1.0" "prismatic @ git+https://github.com/TRI-ML/prismatic-vlms.git"
```

The repo also provides `environment.yml` for a single-file target spec. Prefer
the explicit commands above when debugging solver or Git dependency failures.

## Dependency Tiers

### Core

Required for config parsing, synthetic training smoke tests, validation preflight,
and W&B dry-run plumbing:

- Python 3.9
- PyTorch 2.2.0 / torchvision 0.17.0 / torchaudio 2.2.0 / CUDA 11.8 runtime
- `mkl<2025` and `intel-openmp<2025`; PyTorch 2.2.0 import fails on this host
  with `mkl 2025` because of the missing `iJIT_NotifyEvent` symbol
- NumPy
- Pillow
- PyYAML
- scikit-learn for PR2L PCA projection fitting
- W&B
- pytest and ruff for development validation

### Prismatic

Required for prompt-conditioned VLM feature extraction:

- `prismatic @ git+https://github.com/TRI-ML/prismatic-vlms.git`
- `transformers==4.38.1` and `huggingface-hub<1.0`; without this pin, pip may
  install `transformers 5.x`, which requires newer PyTorch and breaks Prismatic
  imports
- Prismatic transitive dependencies such as `transformers`, `accelerate`,
  `timm`, `sentencepiece`, `jsonlines`, `rich`, and `huggingface_hub`
- Hugging Face access for gated language backbones when the selected model needs it

`prism-dinosiglip+7b` uses the Llama2-7B-pure language backbone. Even when the
Prismatic checkpoint is stored under `/data/topovlm/vlm_weights`, the loader may
need Hugging Face authorization for `meta-llama/Llama-2-7b-hf` tokenizer/config
metadata. Do not commit tokens. Use one of these runtime credential routes:

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the runtime environment
- the standard user token file `~/.cache/huggingface/token`
- `model.vlm.hf_token_path` pointing to a private token file outside the repo

`flash-attn` is not installed by default. Prismatic documents it as required for
VLM training; TopoVLM's first target is frozen-VLM feature extraction. Add and
verify `flash-attn` only when a run path actually requires it.

### Habitat

Required only when generating/evaluating live Habitat episodes rather than
training from prebuilt graph caches:

- Habitat-Sim with headless/Bullet support on cluster nodes
- Habitat-Lab stable package or source install through `.[habitat]`
- HM3D/Habitat-Matterport scene assets and ObjectNav episode configs

Verify `import habitat_sim` and the exact scene dataset path before planning Slurm jobs.

## Verification

After install:

```bash
python - <<'PY'
import torch
import yaml
import wandb
import prismatic
import transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("yaml", yaml.__version__)
print("wandb", wandb.__version__)
print("transformers", transformers.__version__)
print("prismatic", getattr(prismatic, "__version__", "import_ok"))
PY

python -m pip check
python train.py --exp habitat/prismatic_bc_smoke --mode preflight --allow-missing-data
python validate.py --runner cache_audit --exp habitat/prismatic_bc_smoke --allow-missing-data
```

Full PR2L validation additionally needs real `/data/topovlm/habitat` payloads
and Prismatic weights.

## Sources

- PR2L project page: https://pr2l.github.io/
- PR2L paper: https://arxiv.org/abs/2402.02651
- Prismatic VLMs: https://github.com/TRI-ML/prismatic-vlms
- Habitat-Lab: https://github.com/facebookresearch/habitat-lab
- Habitat-Sim: https://github.com/facebookresearch/habitat-sim
- PyTorch install selector: https://pytorch.org/get-started/locally/
