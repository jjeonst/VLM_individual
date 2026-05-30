# PR2L Implementation Notes

This note records implementation consequences only. It is not a literature note
or manuscript claim surface.

## What PR2L Does

PR2L uses a pretrained generative VLM as a prompt-conditioned representation
extractor. The VLM is queried with task-relevant questions about each visual
observation. The generated text is not the action policy. Instead, hidden-state
representations associated with the image, prompt, and generated text become the
state input for a learned policy trained with RL, BC, or offline RL.

For Habitat ObjectNav, the project page describes the prompt:

```text
Would a [target object] be found here? Why or why not?
```

The second sentence is used to elicit chain-of-thought-style semantic reasoning
about likely object locations. The policy still learns actions from data.

## Habitat-Specific Details To Match

From the paper's Habitat appendix:

- Dataset: Habitat-Web human ObjectNav demonstrations, originally 77k
  trajectories and 12M steps.
- Subset: approximately one tenth of that dataset, sampled by target object and
  scene, totaling about 7550 trajectories and 1.1M steps.
- Scenes: Habitat-Matterport 3D v1 train/validation split.
- VLM: Prismatic VLM with Dino+SigLIP vision backbone and Llama2-7B-pure
  language backbone, 224px image version.
- Representation: last two VLM layers are used as promptable representations.
- Compression: VLM token dimensions are reduced with PCA/projection before the
  learned policy consumes them.
- Policy: token representations are pooled/compressed and processed by a learned
  Transformer layer before action prediction.
- Training: behavior cloning, with inflection/action upweighting and
  trajectory-aware batching/gradient accumulation.
- Compute reported by paper: Habitat training on A100; data generation and
  evaluation parallelized on A5000 GPUs.

## Current Scaffold Scope

The current TopoVLM scaffold implements the PR2L-faithful representation and
behavior-cloning path, but only the synthetic/debug scale has been executed so
far. Live Habitat-Web training still needs rendered RGB/action trajectory arrays
derived from the Habitat-Web replays and MP3D scenes.

Current implementation pieces:

- `configs/exp/habitat/pr2l_habitat_bc_faithful.yaml` selects the canonical
  PR2L-faithful behavior-cloning path.
- `cache_format: pr2l_token_trajectory` extracts Prismatic visual-token hidden
  states from the last two configured layers, pools visual tokens, fits/applies a
  PCA projection, and writes node-level action labels.
- `GraphTransformerPolicy` supports node-level action logits for trajectory BC.
- The BC objective supports inflection weighting and stop/turn action weighting.
- Habitat-Web action ids are `STOP=0`, `MOVE_FORWARD=1`, `TURN_LEFT=2`,
  `TURN_RIGHT=3`, `LOOK_UP=4`, and `LOOK_DOWN=5`.
- `validate.py --runner habitat_web_audit` checks Habitat-Web Git LFS
  materialization, samples replay actions, and reports missing MP3D scenes.
- `validate.py --runner pr2l_manifest_audit` checks PR2L trajectory manifests
  and missing payloads before cache building.

Still missing live input:

- Habitat-Web replays are action/state records and do not embed RGB frames.
  Without MP3D scene assets and a replay renderer that writes NumPy RGB/action
  arrays under `/data/topovlm/habitat/episodes/pr2l_habitat_web`, TopoVLM can
  validate code/config/synthetic smoke paths but cannot claim PR2L reproduction
  or paper-scale training.

## Canonical TopoVLM Implication

We should keep PR2L as a frozen-VLM representation baseline first. Fine-tuning a
VLM or asking a VLM to output navigation actions is a different experiment lane,
not the PR2L replication baseline.

Sources: https://pr2l.github.io/ and https://arxiv.org/abs/2402.02651.
