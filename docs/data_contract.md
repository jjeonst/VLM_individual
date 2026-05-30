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
`actions_path` as a NumPy action array. A PR2L-faithful cache builder must also
record the prompt template, VLM model id, VLM layer selection, PCA/projection
metadata, and token pooling policy for each cache manifest.

## Still Missing

- Habitat-Web or replacement expert demonstrations for behavior cloning.
- A cache manifest format that records PR2L token-level representation lineage,
  not only one pooled feature per frame.
- A Slurm stage-in/stage-out contract after the data and checkpoint roots are
  fixed.

## PR2L Reference Targets

The PR2L Habitat experiments used Habitat-Matterport 3D v1 train/validation
scenes, Habitat-Web human ObjectNav demonstrations, and a tenth-sampled
demonstration subset. They used a Prismatic VLM with Dino+SigLIP vision backbone
and Llama2-7B-pure language backbone, then trained a policy on promptable VLM
representations rather than asking the VLM for actions.

Sources: https://arxiv.org/abs/2402.02651 and https://pr2l.github.io/.
