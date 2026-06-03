# TopoVLM

TopoVLM is the canonical Habitat-first implementation surface for frozen-VLM
topological navigation experiments. The current development target is a
PR2L-style baseline on HM3D ObjectNav:

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

The current canonical development path uses HM3D ObjectNav and generates
shortest-path expert trajectories from Habitat. The deferred PR2L Habitat-Web
path uses Matterport3D v1 scene ids such as
`mp3d/17DRP5sb8fy/17DRP5sb8fy.glb`, not HM3D scene ids. MP3D v1 has a separate
Matterport3D terms/access route from HM3D. Do not treat HM3D account access as
MP3D v1 approval; acquire or confirm MP3D v1 access before downloading or using
the MP3D habitat archive for this path.

## Canonical HM3D PR2L-Style BC Path

The repo keeps one active Habitat experiment config:

```bash
python validate.py --runner objectnav_audit --exp habitat/pr2l_hm3d_bc --allow-missing-data
python validate.py --runner pr2l_manifest_audit --exp habitat/pr2l_hm3d_bc --allow-missing-data
python validate.py --runner vlm_auth_audit --exp habitat/pr2l_hm3d_bc --allow-missing-data
python train.py --exp habitat/pr2l_hm3d_bc --mode build_episodes
python train.py --exp habitat/pr2l_hm3d_bc --mode build_cache
python train.py --exp habitat/pr2l_hm3d_bc --mode train
```

`build_episodes` opens the HM3D ObjectNav Habitat config, uses Habitat
`ShortestPathFollower` to generate expert action trajectories, writes RGB/action
NumPy payloads directly under `/data/topovlm/habitat`, and records
`hm3d_objectnav_shortest_path` provenance in
`episodes/pr2l_hm3d_objectnav/<split>/manifest.jsonl`. The existing canonical
HM3D experiment YAMLs, `configs/exp/habitat/pr2l_hm3d_bc.yaml` and
`configs/exp/habitat/pr2l_hm3d_bc_val.yaml`, carry the scene/object-balanced
selection manifests. The active train path uses a deterministic
`balanced_subset_size: 6000` materialized HM3D subset after timeout-skipped
shortest-path rollouts; this is a PR2L-style HM3D baseline, not a claim that the
paper's Habitat-Web subset was exactly reproduced. This is the active TopoVLM
development path; MP3D/Habitat-Web remains a separate external-data branch.

Smoke and subset runs should be driven by `--debug`, tests, or explicit
runtime/job manifests, not by extra experiment YAML files. Slurm is reserved for
GPU-heavy `build_cache` and `train` jobs. For staged cache jobs, generated
wrappers stage shared `/data/topovlm/...` inputs into job-local `data/...`;
runtime path resolution maps canonical `/data/...` config paths to that staged
mirror while stage-out materializers write bundles under
`OUTPUT_DIR/data/topovlm/habitat`. After a staged `build_cache` job finishes,
audit the bundle by setting `TOPOVLM_DATA_OUTPUT_ROOT` to the staged-out data
root:

```bash
TOPOVLM_DATA_OUTPUT_ROOT=<stageout>/data/topovlm/habitat python validate.py --runner pr2l_manifest_audit --exp habitat/pr2l_hm3d_bc
TOPOVLM_DATA_OUTPUT_ROOT=<stageout>/data/topovlm/habitat python validate.py --runner cache_audit --exp habitat/pr2l_hm3d_bc
```

## Missing Live Inputs

The current HM3D path needs these live inputs before real training or evaluation:

- `/data/topovlm/habitat` with HM3D scenes and ObjectNav episodes.
- `/data/topovlm/vlm_weights/prismatic/<model_id>` or Hugging Face access for Prismatic weights.
- Generated shortest-path trajectory manifests under `episodes/pr2l_hm3d_objectnav/<split>/manifest.jsonl`.

`objectnav_audit` opens the ObjectNav HM3D v2 shard files, samples one raw
episode, and resolves its `scene_id` against the canonical HM3D scene layout
under `/data/topovlm/habitat/scene_datasets/hm3d_v0.2`.

## Reference Prototype

`individual_researcher_initial/` contains the initial individual-researcher code
verbatim enough to preserve source lineage. It is reference-only: canonical code
must not import it, run its `.slurm` files, or treat its hardcoded paths as the
TopoVLM runtime contract.

## Slurm Boundary

Slurm scripts are generated under `slurm/habitat/` through the approved Slurm MCP
after the repo is clean and committed. Do not submit scheduler jobs from the Mac
mini or from ad hoc shell scripts.
