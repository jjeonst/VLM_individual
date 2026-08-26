"""Ask the frozen vision-language model a question about each frame and keep what it thought.

This is the part of the method the paper's claim rests on. The model is never trained; it is
shown a frame, asked whether the target object would be found there, and made to answer. The
answer text is thrown away. What is kept is the model's **internal state while producing that
answer** -- the vectors it computed at every position of its input and output -- because those
carry granular information about the scene that the sentence it finally writes does not
(Section 3.1).

**The token sequence.** Prismatic inserts the image's patch embeddings directly after the
opening token, so what the language model actually reads is

    [BOS]  [256 visual tokens]  [question tokens]  ->  [32-48 generated tokens]

That order matters. A causal language model lets each position attend only to earlier ones, so
the **visual positions cannot see the question** -- they encode the layout of the view without
knowing which object is being sought -- while the generated positions know the goal but hold
the view only in summary. The paper feeds the policy all three groups, and it is worth keeping
in mind that they carry different things rather than three copies of one thing. This module
exposes a check for exactly that property: encoding one frame under two different goals must
leave the visual vectors bit-for-bit identical.

**Why decoding is written out here rather than handed to `generate`.** Two requirements make
the library's own generation unusable. First, `PrismaticVLM.generate` returns only the decoded
string, and `generate_batch` is a loop over single items despite its name, so a batch of frames
would be processed one at a time -- at roughly a second per frame and 1.2 million frames, that
is the difference between days and weeks. Second, sampling has to be reproducible **per frame**:
the paper fixes a random seed before decoding, and a seed fixed per batch would make the answer
for a given frame depend on which other frames happened to sit beside it. Each frame therefore
draws from its own generator.

**That is not enough to make a frame's answer independent of its batch, and measurement says so
(experiment 3-A).** Re-encoding the same frames splits one per call instead of four changed the
generated text for four frames in twelve, and shifted the rest by about 1e-3. The random stream
is batch-independent, but the logits it draws against are not: batch shape selects different
matmul kernels, the half-precision result moves in the last bits, and a sampled token that sat
near a probability boundary lands on the other side, after which the continuation diverges.
Given an identical batch the pipeline is exactly reproducible -- bit-for-bit, 12 frames of 12 --
so this is the hardware's arithmetic, not loose state on our side.

The consequence is bounded. The paper samples at temperature 0.4, so a frame's representation is
a draw from a distribution rather than a fixed vector; training saw one draw and evaluation gets
another from the same distribution. What cannot be claimed is that a given frame encodes to the
same thing however the work is divided.

**Batching without padding.** Every frame of one trajectory asks about the same object, so the
question text is identical across the batch and no padding is needed at all. Batches are
therefore formed within a trajectory, which keeps the attention mask trivially correct.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

MODEL_DIR = "/data/topovlm/vlm_weights/prismatic_224px/prism-dinosiglip-224px+7b"

# Every embedding on disk and every trained checkpoint predates the fix in `_assemble` below,
# which means they were built from a visual block that started at the opening token rather than
# at patch zero (Appendix Z.1). Setting PR2L_LEGACY_BOS=1 reproduces that older cut.
#
# It exists so the two questions can be asked separately. Comparing a stored trajectory against a
# fresh encoding under the fixed code answers "is the fix correct?" and "is the encode-PCA-store
# chain sound?" at the same time, and a mismatch would not say which. Under legacy the chain is
# tested alone; the fixed-vs-legacy difference then measures the fix on its own.
#
# Anything trained on legacy embeddings must be evaluated under legacy too. Encoding one way and
# running a policy trained the other way is the half-migrated state Z.1 warns against: the policy
# sees a representation it never learned, and the drop cannot be attributed.
LEGACY_BOS = os.environ.get("PR2L_LEGACY_BOS", "") == "1"

# Appendix C.2, PR2L items 3 and 4.
LAYERS = (-2, -1)
TEMPERATURE = 0.4
MIN_NEW_TOKENS = 32
MAX_NEW_TOKENS = 48

# Appendix C.2, general item 5: the 16x16 grid of visual tokens is average-pooled with a 4x4
# kernel, leaving 16. Note this happens *after* the model has read all 256 of them.
VISUAL_GRID = 16
POOL = 4
POOLED_VISUAL_TOKENS = (VISUAL_GRID // POOL) ** 2

# Section 4.3. The chain-of-thought condition asks for the reason; the ablation drops it.
PROMPT_WITH_COT = "Would a {goal} be found here? Why or why not?"
PROMPT_WITHOUT_COT = "Would a {goal} be found here?"


@dataclass
class FrameFeatures:
    """One frame's promptable representation, before any dimensionality reduction."""

    tokens: np.ndarray          # [N, len(LAYERS), 4096] float32
    visual_count: int           # how many of the leading tokens are pooled visual ones
    prompt_count: int           # BOS plus question tokens
    generated_count: int
    generated_text: str


def load_vlm(device: str = "cuda"):
    """Load the 224px Prismatic model, frozen, in the precision it was trained in."""
    from prismatic import load

    vlm = load(MODEL_DIR)
    vlm.to(device, dtype=vlm.llm_backbone.half_precision_dtype)
    vlm.eval()
    for parameter in vlm.parameters():
        parameter.requires_grad_(False)
    return vlm


class VisionOnly(torch.nn.Module):
    """Just the frozen vision backbone, wearing the same interface the full model has.

    The image-encoder condition never reaches the language model, so building one is seven
    billion parameters of waste -- and, more usefully, avoiding it drops the memory needed from
    a 24 GB card to about 3 GB, which puts the work on the cluster's idle rtx2080s instead of
    queueing behind the cards PR2L needs.

    **The weights are the same ones.** Prismatic's checkpoint holds only the projector and the
    language model -- `from_pretrained` asserts exactly that -- so the vision backbone is
    fetched from timm in both paths and is identical either way. That is what makes the paper's
    "both approaches receive the same visual features" true of this implementation and not just
    of the paper.
    """

    def __init__(self, vision_backbone) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone


def load_vision_backbone(device: str = "cuda"):
    """Build the vision half of the model named in the checkpoint's own config."""
    import json

    from prismatic.models.materialize import get_vision_backbone_and_transform

    config = json.loads((Path(MODEL_DIR) / "config.json").read_text())["model"]
    backbone, transform = get_vision_backbone_and_transform(
        config["vision_backbone_id"], config["image_resize_strategy"])
    if not hasattr(backbone, "image_transform"):
        backbone.image_transform = transform

    # bfloat16, because that is what the backbone computes in when PR2L runs it: `load_vlm`
    # casts the whole model to `llm_backbone.half_precision_dtype`, and both backbones report
    # bfloat16. The control's whole worth rests on the paper's "both approaches receive the same
    # visual features", so the arithmetic has to match, not merely be accurate -- a float32 run
    # would be the more exact of the two and still the wrong one. The cost is that this cannot
    # run on Turing cards, which is why the job asks for a partition that has bfloat16.
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"{torch.cuda.get_device_name(0)}는 bfloat16을 지원하지 않는다. "
            "CoT 경로와 정밀도를 맞추려면 Ampere 이상 카드가 필요하다.")
    model = VisionOnly(backbone).to(device, dtype=torch.bfloat16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_prompt(vlm, goal: str, with_cot: bool = True, template: str | None = None) -> str:
    """Wrap the paper's question in the conversation format this model was trained on.

    The paper gives the question itself (Section 4.3) but not the surrounding format. Prismatic's
    "pure" Llama-2 backbone was trained to read `In: <message>\\nOut: `, so asking in any other
    shape would query the model outside the format it learned. The builder supplies that wrapper.

    `template` replaces the question itself. Only the representation figure passes it, to ask
    Appendix D's "What room is this?" -- same wrapper, different question inside it.
    """
    template = template or (PROMPT_WITH_COT if with_cot else PROMPT_WITHOUT_COT)
    builder = vlm.get_prompt_builder()
    builder.add_turn(role="human", message=template.format(goal=goal))
    return builder.get_prompt()


def frame_seed(episode_id: str, frame_index: int) -> int:
    """A seed that depends only on which frame this is.

    Python's built-in hashing is salted per process, so it cannot be used: the same frame would
    decode differently on different runs. A digest keeps the seed stable across runs, machines
    and batch sizes.
    """
    key = f"{episode_id}:{frame_index}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**63)


def _stack_pixel_values(image_transform, frames: np.ndarray, device: str):
    """Apply the model's own image preprocessing to a batch of frames."""
    from PIL import Image

    transformed = [image_transform(Image.fromarray(frame)) for frame in frames]
    if isinstance(transformed[0], dict):
        return {key: torch.stack([t[key] for t in transformed]).to(device)
                for key in transformed[0]}
    return torch.stack(transformed).to(device)


def _sample(logits: torch.Tensor, generators: list[torch.Generator],
            forbid_eos: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Draw one token per batch item, each from that item's own generator.

    `forbid_eos` marks items that have not yet produced the minimum number of tokens; the paper
    asks for at least 32, so ending early is simply not allowed for them.
    """
    logits = logits.float()
    logits[forbid_eos, eos_id] = float("-inf")
    probabilities = torch.softmax(logits / TEMPERATURE, dim=-1)
    drawn = [torch.multinomial(probabilities[i], 1, generator=generators[i])
             for i in range(len(generators))]
    return torch.cat(drawn, dim=0)


@torch.inference_mode()
def encode_batch(vlm, frames: np.ndarray, goal: str, episode_id: str | Sequence[str],
                 frame_indices: list[int], with_cot: bool = True,
                 template: str | None = None) -> list[FrameFeatures]:
    """Run one batch of frames through the model and return their representations.

    All frames must share a goal, which they do when they come from one trajectory, so the
    question is identical across the batch and the input needs no padding.

    `episode_id` may be one name, when the batch is a slice of a single trajectory, or one name
    per frame. Evaluation needs the second form: it steps several episodes at once and groups
    them by goal, so a batch there spans episodes, and passing a single name would give two
    different episodes standing at the same step the same decoding seed.
    """
    device = next(vlm.parameters()).device
    tokenizer = vlm.llm_backbone.tokenizer
    eos_id = tokenizer.eos_token_id
    batch = len(frame_indices)

    prompt = build_prompt(vlm, goal, with_cot=with_cot, template=template)
    prompt_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(device)
    prompt_length = prompt_ids.shape[1]
    input_ids = prompt_ids.expand(batch, -1).contiguous()
    pixel_values = _stack_pixel_values(vlm.vision_backbone.image_transform, frames, device)

    owners = [episode_id] * batch if isinstance(episode_id, str) else list(episode_id)
    if len(owners) != batch:
        raise ValueError(f"{len(owners)} episode ids for {batch} frames")

    generators = []
    for owner, index in zip(owners, frame_indices):
        generator = torch.Generator(device=device)
        generator.manual_seed(frame_seed(owner, index))
        generators.append(generator)

    autocast_dtype = vlm.llm_backbone.half_precision_dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        output = vlm(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     pixel_values=pixel_values, output_hidden_states=True,
                     use_cache=True, return_dict=True)

        prefix = torch.stack([output.hidden_states[layer] for layer in LAYERS], dim=2)
        visual_count = prefix.shape[1] - prompt_length
        if visual_count != VISUAL_GRID**2:
            raise RuntimeError(
                f"expected {VISUAL_GRID**2} visual tokens, found {visual_count}; the model or "
                "its resolution is not the one the paper specifies")

        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        token = _sample(output.logits[:, -1], generators, ~finished, eos_id)
        generated_hidden, generated_ids = [], []
        past = output.past_key_values

        # The prefill asked for every layer's hidden states because the library offers no way
        # to request only two, and it also produced logits at every one of the ~277 prefix
        # positions when only the last is needed. Together those are gigabytes that stay
        # reachable through `output` for as long as the name is bound, which is what puts the
        # batch size ceiling far below where the model itself would put it. Only `prefix`,
        # `token` and `past` are still needed, so the rest is released here.
        del output

        for step in range(MAX_NEW_TOKENS):
            output = vlm(input_ids=token[:, None], past_key_values=past,
                         output_hidden_states=True, use_cache=True, return_dict=True)
            past = output.past_key_values
            generated_hidden.append(
                torch.stack([output.hidden_states[layer][:, 0] for layer in LAYERS], dim=1))
            generated_ids.append(token)
            lengths += (~finished).long()
            finished |= (token == eos_id)
            if bool(finished.all()):
                break
            forbid = (lengths < MIN_NEW_TOKENS) | finished
            token = _sample(output.logits[:, -1], generators, forbid, eos_id)

    return _assemble(prefix, generated_hidden, generated_ids, lengths, prompt_length,
                     tokenizer)


def _assemble(prefix: torch.Tensor, generated_hidden: list[torch.Tensor],
              generated_ids: list[torch.Tensor], lengths: torch.Tensor,
              prompt_length: int, tokenizer) -> list[FrameFeatures]:
    """Pool the visual tokens and lay each frame's tokens out in one array."""
    batch = prefix.shape[0]
    layers = len(LAYERS)

    # The patches do not start at position 0. Prismatic builds the sequence as
    # `torch.cat([input_embeddings[:, :1], projected_patch_embeddings, input_embeddings[:, 1:]])`
    # (`prismatic/models/vlms/prismatic.py:329`), so the opening token comes first and the 256
    # patches follow it.
    #
    # Slicing from 0 -- which this did until it was caught -- put the opening token where patch
    # zero belongs and pushed every patch one place along, so the 4x4 pooling below averaged
    # squares that are not neighbours on the 16x16 grid, and patch 255 escaped pooling into the
    # text block. Nothing was lost, since all 257 vectors stayed in the representation, but the
    # spatial structure the pooling exists to preserve was scrambled.
    #
    # The length check above cannot see this: the sequence is `1 + 256 + (prompt_length - 1)`
    # long, so `prefix.shape[1] - prompt_length` is 256 wherever the cut is made.
    if LEGACY_BOS:
        visual = prefix[:, :VISUAL_GRID**2]             # opening token + patches 0..254
        text = prefix[:, VISUAL_GRID**2:]               # patch 255 + question
    else:
        visual = prefix[:, 1:1 + VISUAL_GRID**2]        # [B, 256, L, 4096]
        text = torch.cat([prefix[:, :1], prefix[:, 1 + VISUAL_GRID**2:]], dim=1)

    # Average over each 4x4 block of the 16x16 grid the patches are laid out on.
    grid = visual.reshape(batch, VISUAL_GRID, VISUAL_GRID, layers, -1)
    grid = grid.reshape(batch, VISUAL_GRID // POOL, POOL, VISUAL_GRID // POOL, POOL, layers, -1)
    pooled = grid.mean(dim=(2, 4)).reshape(batch, POOLED_VISUAL_TOKENS, layers, -1)

    stacked = torch.stack(generated_hidden, dim=1) if generated_hidden else None
    ids = torch.stack(generated_ids, dim=1) if generated_ids else None

    results = []
    for item in range(batch):
        count = int(lengths[item])
        pieces = [pooled[item], text[item]]
        if stacked is not None and count > 0:
            pieces.append(stacked[item, :count])
        tokens = torch.cat(pieces, dim=0).float().cpu().numpy()
        answer = (tokenizer.decode(ids[item, :count], skip_special_tokens=True).strip()
                  if ids is not None and count > 0 else "")
        results.append(FrameFeatures(
            tokens=tokens, visual_count=POOLED_VISUAL_TOKENS,
            prompt_count=prompt_length, generated_count=count, generated_text=answer))
    return results


@torch.inference_mode()
def encode_vision_batch(vlm, frames: np.ndarray) -> list[FrameFeatures]:
    """The image-encoder condition: the picture, and nothing the language model did with it.

    This is the paper's control (Section 4.2, "a policy on Prismatic VLM image encoder
    embeddings"), and the comparison it supports is only fair because the two conditions start
    from the same tensor. Both run the same frozen DINOv2 and SigLIP over the same 224px image;
    PR2L then hands the result to the projector and the seven-billion-parameter language model
    and reads the last two layers back, while this stops at the backbone.

    Three things that apply to PR2L do not apply here, and the paper is explicit about each:

    * **No PCA.** Appendix C.2 lists it under "For PR2L-specific design choices", and the
      sentence itself reads "To reduce the size of VLM representations for PR2L".
    * **No stacking of two layers.** Also PR2L-specific (item 3), and there is only one output
      here to begin with -- a vision backbone has no "last two layers" in that sense.
    * **The policy's input is 2176 wide, not 2048.** That is the fused width, DINOv2's 1024
      plus SigLIP's 1152, and Listing 1 takes the width as a constructor argument precisely so
      that it can differ between conditions. Everything downstream of the projection to 1024 is
      identical, which is the control Appendix C.2's general item 5 asks for.

    The 4x4 pooling *does* apply: general item 5 governs "policies that receive visual
    observations as a sequence of tokens", and 256 patches is exactly that.
    """
    device = next(vlm.parameters()).device
    pixel_values = _stack_pixel_values(vlm.vision_backbone.image_transform, frames, device)

    # The same shape as `encode_batch`: float32 pixels in, autocast doing the cast, weights in
    # bfloat16. Matching the form and not just the dtype keeps the two paths' arithmetic the
    # same, which is the property the paper's comparison rests on.
    autocast_dtype = vlm.vision_backbone.half_precision_dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        patches = vlm.vision_backbone(pixel_values)          # [B, 256, 2176]

    batch, count, width = patches.shape
    if count != VISUAL_GRID**2:
        raise ValueError(f"패치 {count}개, {VISUAL_GRID**2}개를 기대했다")

    grid = patches.reshape(batch, VISUAL_GRID, VISUAL_GRID, width)
    grid = grid.reshape(batch, VISUAL_GRID // POOL, POOL, VISUAL_GRID // POOL, POOL, width)
    pooled = grid.mean(dim=(2, 4)).reshape(batch, POOLED_VISUAL_TOKENS, width)
    pooled = pooled.float().cpu().numpy()

    return [FrameFeatures(tokens=pooled[item], visual_count=POOLED_VISUAL_TOKENS,
                          prompt_count=0, generated_count=0, generated_text="")
            for item in range(batch)]
