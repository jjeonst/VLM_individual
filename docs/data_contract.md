# TopoVLM Data Contract

Large payloads are shared through `/data/topovlm`, not committed to this repo.
The current `bmlslurm` surface uses `/data/topovlm` as the shared Habitat and
VLM payload root.

## Root Layout

```text
/data/topovlm/
  habitat/
    episodes/
      train/manifest.jsonl
      val/manifest.jsonl
    rgb/
    actions/
    episode_selections/
    scene_datasets/
    graphs/
    embeddings/
  vlm_weights/
    prismatic/
      prism-dinosiglip+7b/
```

`configs/data/default.yaml` is the canonical config entry for these paths.
Runtime code resolves relative episode, graph, and embedding paths under
`data_root`.

## Episode Manifest

Each line in `episodes/<split>/manifest.jsonl` must contain:

```json
{
  "episode_id": "scene_target_000001",
  "split": "train",
  "scene_id": "hm3d/train/scene.glb",
  "goal_text": "toilet",
  "rgb_path": "rgb/scene_target_000001.npy",
  "actions_path": "actions/scene_target_000001.npy"
}
```

The graph-cache builder reads `rgb_path` as a NumPy RGB frame array and
`actions_path` as a NumPy action array. Optional `source_dataset`,
`source_trajectory_id`, and `object_category` fields preserve the episode source
lineage. The current HM3D path records `source_dataset` as
`hm3d_objectnav_shortest_path`.

For `trajectory_source: objectnav_shortest_path`, `train.py --mode
build_episodes` opens the configured HM3D ObjectNav Habitat environment, uses
Habitat's `ShortestPathFollower`, and writes NumPy RGB/action arrays. Action ids
are: `STOP=0`, `MOVE_FORWARD=1`, `TURN_LEFT=2`, and `TURN_RIGHT=3`.

The deferred Habitat-Web source data is kept under
`/data/topovlm/habitat/sources/habitat_web_hf_metadata`. Those source shards
store `reference_replay` action/state records, not embedded RGB frames. With
`trajectory_source: habitat_web_replay`, `build_episodes` renders those replay
states against MP3D scenes. Habitat-Web action ids additionally include
`LOOK_UP=4` and `LOOK_DOWN=5`.

For scene/object-balanced subset runs, `train.py --mode build_selection` writes
`episode_selections/.../*.jsonl` under `data_root`. For HM3D, each line names an
ObjectNav `source_trajectory_id`, `episode_id`, `scene_id`, `object_category`,
and source shard. If `data.episode_selection_manifest` is configured,
`build_episodes` renders only those selected source episodes.
When `TOPOVLM_DATA_OUTPUT_ROOT` is set, output-producing materializers write
episode arrays, graph caches, embeddings, and their manifests under that root
while reading source inputs from `data.data_root`. In Slurm jobs, the generated
wrapper sets `RUN_ROOT` and `OUTPUT_DIR`; materializers then write the same data
bundle layout under `OUTPUT_DIR/<data_root>/...` for stage-out.

Staged materialization outputs are audited by pointing the existing validators at
the output data root:

```bash
TOPOVLM_DATA_OUTPUT_ROOT=<stageout>/data/topovlm/habitat python validate.py --runner pr2l_manifest_audit --exp habitat/pr2l_hm3d_bc
TOPOVLM_DATA_OUTPUT_ROOT=<stageout>/data/topovlm/habitat python validate.py --runner cache_audit --exp habitat/pr2l_hm3d_bc
```

For `cache_format: pr2l_token_trajectory`, graph cache payloads contain
node-level action labels:

```text
nodes: [num_nodes, pooled_visual_tokens, projected_feature_dim]
node_actions: [num_nodes]
action_mask: [num_nodes]
frame_ranges: [num_nodes, 2]
```

The graph manifest records `prediction_target=nodes`, token count, feature dim,
representation id, and metadata path. The metadata JSON records the PR2L prompt
template, generated VLM text when enabled, hidden layer selection, token pooling,
and projection lineage.

## Still Missing

- HM3D scene assets and ObjectNav episode shards under `/data/topovlm/habitat`.
- Prismatic VLM weights or Hugging Face access for the gated Llama 2 metadata.

The deferred PR2L-faithful path also needs MP3D scene assets under
`/data/topovlm/habitat/scene_datasets/mp3d`.

## PR2L Reference Targets

The PR2L Habitat experiments used Habitat-Matterport 3D v1 train/validation
scenes, Habitat-Web human ObjectNav demonstrations, and a tenth-sampled
demonstration subset. They used a Prismatic VLM with Dino+SigLIP vision backbone
and Llama2-7B-pure language backbone, then trained a policy on promptable VLM
representations rather than asking the VLM for actions.

Sources: https://arxiv.org/abs/2402.02651 and https://pr2l.github.io/.
