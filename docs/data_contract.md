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
`source_trajectory_id`, and `object_category` fields preserve Habitat-Web
lineage when that source is available.

Habitat-Web source data is kept under
`/data/topovlm/habitat/sources/habitat_web_hf_metadata`. The source shards store
`reference_replay` action/state records, not embedded RGB frames. A PR2L-ready
episode manifest is produced by `train.py --mode build_episodes`, which renders
those replay states against MP3D scenes and writes NumPy RGB/action arrays.
Action ids are: `STOP=0`, `MOVE_FORWARD=1`, `TURN_LEFT=2`, `TURN_RIGHT=3`,
`LOOK_UP=4`, and `LOOK_DOWN=5`.

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

- MP3D scene assets under `/data/topovlm/habitat/scene_datasets/mp3d`.

## PR2L Reference Targets

The PR2L Habitat experiments used Habitat-Matterport 3D v1 train/validation
scenes, Habitat-Web human ObjectNav demonstrations, and a tenth-sampled
demonstration subset. They used a Prismatic VLM with Dino+SigLIP vision backbone
and Llama2-7B-pure language backbone, then trained a policy on promptable VLM
representations rather than asking the VLM for actions.

Sources: https://arxiv.org/abs/2402.02651 and https://pr2l.github.io/.
