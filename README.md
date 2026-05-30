# TopoVLM

TopoVLM is the canonical Habitat-first implementation surface for frozen-VLM
topological navigation experiments. The current target is a PR2L-style baseline:

1. collect or receive Habitat ObjectNav expert episodes,
2. cache prompt-conditioned frozen VLM hidden states,
3. compress frame features into topological graph nodes,
4. train a graph-conditioned behavior-cloning policy.

The repository intentionally uses root entrypoints plus responsibility folders,
not a `src/` or `topovlm/` package wrapper.

## Setup

Canonical development happens on `bmlslurm` in the `topovlm` conda environment.
The first environment is Python 3.10, matching the PR2L example and Prismatic's
tested stack. Python 3.12 is a future compatibility target, but it is not the
first canonical env because Habitat-Lab/Habitat-Sim and Prismatic publish
different tested-version guidance. On `bmlslurm`, `aihabitat` stable
`habitat-sim 0.3.3` currently resolves to Python 3.9 builds, so live
Habitat-Sim belongs behind a separate env/container gate.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n topovlm python=3.10 pip cmake=3.27 -c conda-forge -y
conda activate topovlm
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y "mkl<2025" "intel-openmp<2025" -c defaults
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install "transformers==4.38.1" "huggingface-hub<1.0" "prismatic @ git+https://github.com/TRI-ML/prismatic-vlms.git"
```

See `docs/dependencies.md` for the dependency tiers and Habitat-Sim notes.

## W&B

The repo-local W&B contract is `artifacts/contracts/wandb_identity_contract.json`.
Dave selected the canonical entity `topovlm`. New W&B-backed runs must resolve
entity/project/group/run names from that contract rather than command-line
overrides.

Current canonical projects:

- `TopoVLM`: ordinary training, validation, and final row logging.
- `TopoVLM-sweep`: W&B sweep search runs created through `sweep_wandb.py`.

## Canonical Entrypoints

- `train.py`: training, cache building, and train-owned preflight modes.
- `validate.py`: validation, cache audit, and offline diagnostics.
- `sweep_wandb.py`: W&B sweep creation/agent plumbing only.

Semantic experiment choices live in committed YAML under `configs/`; command-line
flags are reserved for operational selectors such as `--exp`, `--mode`, and
runtime paths.

## Storage Contract

Large shared payloads are external to the repo:

- Habitat data: `/data/topovlm/habitat`
- VLM weights/cache: `/data/topovlm/vlm_weights/<vlm_name>`
- Checkpoints: repo-local ignored `checkpoints/` unless a Slurm wrapper supplies
  `CHECKPOINT_DIR`

Repo-owned durable records, contracts, and small manifests belong under
`artifacts/`. Runtime code must not import from `artifacts/`.

## Dataset License Boundary

HM3D is licensed for non-commercial academic use. Matterport's academic-use
terms define derived information broadly enough to include models or algorithms
trained on the dataset. Treat Habitat/HM3D scene files, semantic annotations,
episode-derived caches, topological graphs, VLM feature caches, trained policy
checkpoints, and any other HM3D-derived payloads as non-redistributable unless a
separate reviewed license decision says otherwise.

This likely explains why PR2L released paper/project material but not a public
trained checkpoint or full derived cache bundle. TopoVLM code may be open, but
HM3D-derived weights, caches, manifests containing substantial scene-derived
content, and hosted interactive demos must stay private/internal by default.

## First Smoke Config

The first canonical config is:

```bash
python train.py --exp habitat/prismatic_bc_smoke --debug
python validate.py --runner data_preflight --exp habitat/prismatic_bc_smoke
python validate.py --runner cache_audit --exp habitat/prismatic_bc_smoke
```

The debug train path uses a synthetic graph dataset so entrypoint, model, loss,
checkpoint, and config plumbing can be checked before Habitat/Prismatic payloads
are installed. Non-debug training reads graph caches declared by the Habitat data
config.

## Missing Live Inputs

The repo is runnable for synthetic/debug smoke tests. Habitat-scale PR2L work
still needs these live inputs before real training or evaluation:

- `/data/topovlm/habitat` with Habitat scenes, ObjectNav episodes, and expert demonstrations.
- `/data/topovlm/vlm_weights/prismatic/<model_id>` or Hugging Face access for Prismatic weights.
- PR2L-faithful VLM token cache generation, PCA/projection metadata, and graph cache manifests.
- A generated Slurm script after data, env, checkpoint, and W&B contracts are stable.

## Reference Prototype

`individual_researcher_initial/` contains the initial individual-researcher code
verbatim enough to preserve source lineage. It is reference-only: canonical code
must not import it, run its `.slurm` files, or treat its hardcoded paths as the
TopoVLM runtime contract.

## Slurm Boundary

Slurm scripts are generated under `slurm/habitat/` through the approved Slurm MCP
after the repo is clean and committed. Do not submit scheduler jobs from the Mac
mini or from ad hoc shell scripts.
