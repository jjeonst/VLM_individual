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
The canonical environment is Python 3.9 because `aihabitat` stable
`habitat-sim 0.3.3` resolves to Python 3.9 builds on `bmlslurm`. Python 3.10
matched PR2L's public notebook setup, but Habitat-Sim is required for the
Habitat-first runtime path.

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
  `CHECKPOINT_DIR`; retained runs write `checkpoint_manifest.json` with config,
  data/cache, W&B contract, source commit, selected checkpoint, and finality
  metadata

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

The PR2L-faithful Habitat-Web path uses Matterport3D v1 scene ids such as
`mp3d/17DRP5sb8fy/17DRP5sb8fy.glb`, not HM3D scene ids. MP3D v1 has a separate
Matterport3D terms/access route from HM3D. Do not treat HM3D account access as
MP3D v1 approval; acquire or confirm MP3D v1 access before downloading or using
the MP3D habitat archive for this path.

## First Smoke Config

The first smoke config is:

```bash
python train.py --exp habitat/prismatic_bc_smoke --debug
python validate.py --runner data_preflight --exp habitat/prismatic_bc_smoke
python validate.py --runner objectnav_audit --exp habitat/prismatic_bc_smoke
python validate.py --runner cache_audit --exp habitat/prismatic_bc_smoke
python validate.py --runner data_preflight --exp habitat/prismatic_bc_smoke_staged --allow-missing-data
```

The debug train path uses a synthetic graph dataset so entrypoint, model, loss,
checkpoint, and config plumbing can be checked before Habitat/Prismatic payloads
are installed. Non-debug training reads graph caches declared by the Habitat data
config.

## PR2L-Faithful BC Path

The canonical paper-faithful reimplementation config is:

```bash
python validate.py --runner habitat_web_audit --exp habitat/pr2l_habitat_bc_faithful --allow-missing-data
python validate.py --runner habitat_web_scene_audit --exp habitat/pr2l_habitat_bc_faithful --allow-missing-data
python validate.py --runner pr2l_manifest_audit --exp habitat/pr2l_habitat_bc_faithful --allow-missing-data
python validate.py --runner vlm_auth_audit --exp habitat/pr2l_habitat_bc_faithful --allow-missing-data
python train.py --exp habitat/pr2l_habitat_bc_faithful --mode build_episodes
python train.py --exp habitat/pr2l_habitat_bc_faithful --mode build_cache
python train.py --exp habitat/pr2l_habitat_bc_faithful --mode train
```

This is a PR2L-faithful TopoVLM implementation path, not an exact PR2L
reproduction claim. Exact reproduction still depends on access to the same
Habitat-Web human demonstration distribution and matching evaluation protocol.

The implementation uses Prismatic as a frozen VLM, asks the PR2L ObjectNav
prompt, caches last-two-layer visual-token representations, pools visual tokens,
applies PCA projection, builds topology graph nodes, and trains node-level
behavior cloning with inflection and stop/turn weighting.

The scene/object-balanced stage-3 config is:

```bash
python train.py --exp habitat/pr2l_habitat_bc_balanced_subset --mode build_selection
python validate.py --runner habitat_web_selection_audit --exp habitat/pr2l_habitat_bc_balanced_subset --allow-missing-data
python train.py --exp habitat/pr2l_habitat_bc_balanced_subset --mode build_episodes
python train.py --exp habitat/pr2l_habitat_bc_balanced_subset --mode build_cache
python train.py --exp habitat/pr2l_habitat_bc_balanced_subset --mode train
```

`build_selection` writes a deterministic source-trajectory manifest under
`/data/topovlm/habitat/episode_selections/...`; `build_episodes` then renders
only those selected Habitat-Web replays.
On Slurm, use the `_staged` balanced config so staged inputs are read from
`data/topovlm/...` and output materialization bundles are written under
`OUTPUT_DIR/data/topovlm/habitat/...` for MCP stage-out.
After a staged `build_episodes` or `build_cache` job finishes, audit the staged
bundle by setting `TOPOVLM_DATA_OUTPUT_ROOT` to the staged-out data root and
running the canonical validators, for example:

```bash
TOPOVLM_DATA_OUTPUT_ROOT=<stageout>/data/topovlm/habitat python validate.py --runner pr2l_manifest_audit --exp habitat/pr2l_habitat_bc_balanced_subset_staged
```

## Missing Live Inputs

The repo is runnable for synthetic/debug smoke tests. Habitat-scale PR2L work
still needs these live inputs before real training or evaluation:

- `/data/topovlm/habitat` with Habitat scenes, ObjectNav episodes, and expert demonstrations.
- `/data/topovlm/vlm_weights/prismatic/<model_id>` or Hugging Face access for Prismatic weights.
- Habitat-Web trajectory manifests under `episodes/pr2l_habitat_web/<split>/manifest.jsonl`.
- MP3D scene assets under `/data/topovlm/habitat/scene_datasets/mp3d`.

`objectnav_audit` opens the staged ObjectNav HM3D v2 shard files, samples one
raw episode, and resolves its `scene_id` against
`/data/topovlm/habitat/scene_datasets/hm3d`. The `*_staged` config set is for
Slurm jobs that copy shared `/data/topovlm/...` inputs into the job-local
`data/topovlm/` directory before running from scratch.

`habitat_web_audit` opens the Habitat-Web Hugging Face source clone declared by
`configs/data/pr2l_habitat_web.yaml`, verifies that Git LFS payloads are
materialized, samples `reference_replay` actions, and reports missing MP3D
scenes. Habitat-Web stores action/state replays, not embedded RGB frames, so
`train.py --mode build_episodes` renders those replays against MP3D scenes
before `episodes/.../manifest.jsonl` can point to NumPy RGB/action arrays.
`habitat_web_scene_audit` scans the materialized Habitat-Web train split and
writes the current scene/object/action inventory to
`artifacts/evidence/habitat_web_scene_inventory_audit.json`; this diagnostic
artifact currently lists 56 required MP3D scene GLBs and 28 object categories.

## Reference Prototype

`individual_researcher_initial/` contains the initial individual-researcher code
verbatim enough to preserve source lineage. It is reference-only: canonical code
must not import it, run its `.slurm` files, or treat its hardcoded paths as the
TopoVLM runtime contract.

## Slurm Boundary

Slurm scripts are generated under `slurm/habitat/` through the approved Slurm MCP
after the repo is clean and committed. Do not submit scheduler jobs from the Mac
mini or from ad hoc shell scripts.
