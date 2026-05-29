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

## Gap Against Current Scaffold

The current TopoVLM scaffold is intentionally narrower than the PR2L Habitat
implementation. It can smoke-test config, policy, loss, and checkpoint plumbing,
but it does not yet implement the PR2L-faithful representation path.

Missing implementation pieces:

- Token-level Prismatic hidden-state extraction for the last two layers.
- Prompt generation variants, including CoT and non-CoT prompt families.
- PCA/projection fit artifact and deterministic application to all cached tokens.
- Cache manifest fields for model id, prompt, decoding settings, layer ids,
  projection id, token pooling, source image identity, and source demonstration.
- Full-trajectory BC batching and inflection/action upweighting.
- Habitat-Web/HM3D data ingestion and split-preserving subsampling.

## Canonical TopoVLM Implication

We should keep PR2L as a frozen-VLM representation baseline first. Fine-tuning a
VLM or asking a VLM to output navigation actions is a different experiment lane,
not the PR2L replication baseline.

Sources: https://pr2l.github.io/ and https://arxiv.org/abs/2402.02651.
