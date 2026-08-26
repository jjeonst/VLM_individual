"""Encode the collected images with the vision-language model, keeping several tokens.

The previous experiment reduced each frame to a **single vector** taken from the last
position of the model's input. That vector knows which object is being sought, because by
that position the model has read the question, but the spatial arrangement of the view is
compressed into one summary.

The prior work this study follows keeps **many tokens per frame** instead, drawn from the
image, the question, and the model's own generated answer, and lets the policy attend over
them. This module does the same, and the reason it matters follows from how the model reads
its input. The language backbone lets each position attend only to positions before it, and
the input is laid out as

    [start] [image patches ...] [question] [generated answer]

so the image positions carry the layout of the view but cannot see which object was asked
for, while the later text positions know the goal but hold the view only in summary. Taking
tokens from both gives the policy the layout *and* the goal-conditioned judgement. That the
image positions are goal-blind is not an assumption here: encoding the same picture under
different target objects leaves them bit-for-bit identical.

Per frame the output is therefore

- **16 image tokens**, obtained by arranging the image positions on their square grid and
  average-pooling to 4x4, which keeps coarse left/right and near/far structure, and
- **4 text tokens**, obtained by pooling the question and answer positions into four groups.

Each token comes from the last two layers joined together, again following the prior work.

**Every token is then shortened by principal component analysis (PCA).** A token leaves the
language model as 4096 numbers per layer, which is far wider than the number of
demonstrations can support: with a few hundred demonstrations a policy reading 8192 numbers
per token has enough freedom to fit the demonstrations without learning anything about the
route. Principal component analysis finds the directions along which the tokens actually
vary and keeps only those, discarding directions in which every token looks the same. The
prior work reduces each layer's 4096 numbers to **1024** this way, so that a token joined
across two layers is 2048 numbers wide, and this module does the same.

The directions are found from a sample of frames drawn across the whole set of
demonstrations before the main pass begins, and are then applied unchanged to every frame,
so that all frames land in one common coordinate system. They are saved next to the encoded
data as ``pca.npz`` because the policy is deployed on frames that are encoded live, and
those frames must be reduced by the same directions or the policy sees a different input
space than the one it was trained on.

The prior work samples **one frame per demonstration** for this fit. That work has 7550
demonstrations; this study has a few hundred, so one frame each would leave fewer samples
than the number of directions being estimated. The number of frames sampled per
demonstration is therefore a setting, defaulting to four, and the fraction of the original
variation retained is reported so that the choice can be checked rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm_topology_test.phase3_fixed_route.collect import DATA_ROOT  # noqa: E402

WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
LAYERS = (-2, -1)
IMAGE_GRID = 4      # image positions are pooled to IMAGE_GRID x IMAGE_GRID tokens
TEXT_TOKENS = 4     # question and answer positions are pooled to this many tokens
PCA_COMPONENTS = 1024   # width each layer is reduced to; 0 keeps the full width
PCA_FILE = "pca.npz"

PROMPTS = {
    "baseline": "Would a {goal} be found here? Why or why not?",
    "structure": ("Where can I go from here, and which of those directions most likely "
                  "leads to a {goal}? Why?"),
}

# The simulator names its object categories for machines, not for readers: the television
# category is spelled "tv_monitor". That string is put in front of a language model, which
# has never seen it written that way in ordinary text, so it is translated back into the
# word the prior work uses. Categories already spelled as ordinary English pass through.
GOAL_PHRASE = {"tv_monitor": "television"}


def goal_phrase(category: str) -> str:
    """The wording used inside the question for a simulator object category."""
    return GOAL_PHRASE.get(category, category)


class Projection:
    """The principal directions used to shorten every token, fitted once and reused.

    ``mean`` is subtracted first so the directions describe how tokens differ from a typical
    token rather than where they sit in general, and ``components`` holds the directions
    themselves, one per column, ordered from the one along which tokens vary most.
    """

    def __init__(self, mean: np.ndarray, components: np.ndarray,
                 explained: float | None = None):
        self.mean = mean.astype(np.float32)
        self.components = components.astype(np.float32)
        self.explained = explained

    @property
    def width(self) -> int:
        return int(self.components.shape[1])

    def apply(self, block: np.ndarray) -> np.ndarray:
        """[..., hidden] -> [..., width]."""
        return (block.astype(np.float32) - self.mean) @ self.components

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, components=self.components,
                 explained=np.asarray(self.explained if self.explained is not None else -1.0))

    @classmethod
    def load(cls, path: Path) -> "Projection":
        payload = np.load(path)
        explained = float(payload["explained"]) if "explained" in payload.files else None
        return cls(payload["mean"], payload["components"],
                   None if explained is not None and explained < 0 else explained)

    @classmethod
    def fit(cls, rows: np.ndarray, components: int) -> "Projection":
        """Find the directions of greatest variation in ``rows`` [samples, hidden]."""
        rows = rows.astype(np.float32)
        mean = rows.mean(axis=0)
        centred = rows - mean
        # The covariance is only hidden x hidden, so its eigenvectors are cheap to get even
        # when there are many samples; they are the principal directions.
        covariance = (centred.T @ centred).astype(np.float64) / max(len(rows) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1][:components]
        kept = values[order]
        total = float(np.clip(values, 0.0, None).sum())
        explained = float(np.clip(kept, 0.0, None).sum() / total) if total > 0 else None
        return cls(mean, vectors[:, order], explained)


class MultiTokenEncoder:
    """Encode one frame into a set of tokens spanning image, question and answer."""

    def __init__(self, max_new_tokens: int = 48, projection: Projection | None = None):
        import torch
        from prismatic import load

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.projection = projection
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"{torch.cuda.get_device_name(0)} does not support bfloat16; schedule this "
                "job on a GPU that does so that every condition shares one numeric format")
        self.vlm = load(WEIGHTS, hf_token=os.environ.get("HF_TOKEN"))
        self.vlm.to(self.device, dtype=self.dtype)
        self.vlm.requires_grad_(False)
        self.vlm.eval()
        self.hidden = int(self.vlm.llm_backbone.llm.config.hidden_size)
        self.tokens = IMAGE_GRID * IMAGE_GRID + TEXT_TOKENS

    @property
    def width(self) -> int:
        """Numbers per token after the layers are joined, and after PCA if it is in use."""
        per_layer = self.hidden if self.projection is None else self.projection.width
        return per_layer * len(LAYERS)

    def _prompt(self, question: str, answer: str | None = None) -> str:
        builder = self.vlm.get_prompt_builder()
        builder.add_turn(role="human", message=question)
        if answer is not None:
            builder.add_turn(role="gpt", message=answer)
        return builder.get_prompt()

    def _pool_image(self, block):
        """[positions, dim] on a square grid -> [IMAGE_GRID^2, dim]."""
        torch = self.torch
        count, dim = block.shape
        side = int(round(count ** 0.5))
        if side * side != count:                 # trim a leading class token if present
            side = int(round((count - 1) ** 0.5))
            block = block[count - side * side:]
        grid = block.reshape(side, side, dim).permute(2, 0, 1)[None]
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (IMAGE_GRID, IMAGE_GRID))
        return pooled[0].permute(1, 2, 0).reshape(IMAGE_GRID * IMAGE_GRID, dim)

    def _pool_text(self, block):
        """[positions, dim] -> [TEXT_TOKENS, dim] by averaging consecutive groups."""
        torch = self.torch
        if block.shape[0] == 0:
            return torch.zeros(TEXT_TOKENS, block.shape[-1], device=block.device)
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            block.permute(1, 0)[None], TEXT_TOKENS)
        return pooled[0].permute(1, 0)

    def encode_layers(self, image, question: str) -> tuple[np.ndarray, str]:
        """One frame -> [tokens, layers, hidden], before any shortening."""
        torch, vlm = self.torch, self.vlm
        answer = vlm.generate(image, self._prompt(question), do_sample=False,
                              max_new_tokens=self.max_new_tokens)
        prompt = self._prompt(question, answer)
        input_ids = vlm.llm_backbone.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attention = torch.ones_like(input_ids)
        pixel_values = vlm.vision_backbone.image_transform(image)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device, self.dtype)
                            for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values[None, ...].to(self.device, self.dtype)

        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            patches = vlm.vision_backbone(pixel_values)
            projected = vlm.projector(patches)
            embedded = vlm.llm_backbone.embed_input_ids(input_ids)
            fused = torch.cat([embedded[:, :1], projected, embedded[:, 1:]], dim=1)
            patch_mask = torch.full((projected.shape[0], projected.shape[1]), True,
                                    dtype=attention.dtype, device=self.device)
            fused_mask = torch.cat([attention[:, :1], patch_mask, attention[:, 1:]], dim=1)
            output = vlm.llm_backbone(input_ids=None, attention_mask=fused_mask,
                                      inputs_embeds=fused, output_hidden_states=True,
                                      return_dict=True)

        visual_count = int(projected.shape[1])
        per_layer = []
        for layer in LAYERS:
            hidden = output.hidden_states[layer][0].float()
            per_layer.append(torch.cat([self._pool_image(hidden[1:1 + visual_count]),
                                        self._pool_text(hidden[1 + visual_count:])], dim=0))
        block = torch.stack(per_layer, dim=1)          # [tokens, layers, hidden]
        return block.cpu().numpy(), str(answer)

    def encode(self, image, question: str) -> tuple[np.ndarray, str]:
        """One frame -> [tokens, width], with the layers joined and PCA applied if fitted."""
        block, answer = self.encode_layers(image, question)
        return self.flatten(block), answer

    def flatten(self, block: np.ndarray) -> np.ndarray:
        """[tokens, layers, hidden] -> [tokens, width]."""
        if self.projection is not None:
            block = self.projection.apply(block)
        return block.reshape(block.shape[0], -1)


def demo_keys(payload) -> list[str]:
    return sorted({name.split("|")[0] for name in payload.files})


def fit_projection(encoder, shards, goals, template, *, components, frames_per_demo,
                   rng) -> tuple[Projection, dict]:
    """Find the principal directions from frames sampled across every demonstration.

    The frames encoded here are kept so the main pass does not have to encode them again.
    """
    from PIL import Image

    cache, rows, started = {}, [], time.time()
    for position, shard in enumerate(shards, start=1):
        payload = np.load(shard)
        for key in demo_keys(payload):
            images = payload[f"{key}|rgb"]
            question = template.format(goal=goal_phrase(goals.get(key, "object")))
            picks = rng.choice(len(images), size=min(frames_per_demo, len(images)),
                               replace=False)
            for index in sorted(int(p) for p in picks):
                block, _ = encoder.encode_layers(
                    Image.fromarray(images[index][..., :3].astype("uint8"), mode="RGB"),
                    question)
                cache[(shard.name, key, index)] = block.astype(np.float16)
                rows.append(block.reshape(-1, encoder.hidden))
        print(f"  [pca {position}/{len(shards)}] {shard.stem}: {len(cache)} frames, "
              f"{sum(len(r) for r in rows)} token samples so far", flush=True)

    stack = np.concatenate(rows, axis=0)
    rows.clear()
    print(f"[encode] fitting {components} directions from {len(stack)} token samples "
          f"({time.time() - started:.0f} s of encoding)", flush=True)
    projection = Projection.fit(stack, components)
    report = {"components": projection.width, "fit_frames": len(cache),
              "fit_samples": int(len(stack)), "frames_per_demo": frames_per_demo,
              "explained_variance_ratio": (None if projection.explained is None
                                           else round(projection.explained, 4))}
    if len(stack) < 10 * components:
        print(f"[encode] WARNING: {len(stack)} samples for {components} directions is thin; "
              "the later directions rest on very little data", flush=True)
    return projection, cache, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", required=True, choices=["random", "matched", "junction"])
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="baseline")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS,
                        help="numbers kept per layer; 0 keeps the full 4096")
    parser.add_argument("--pca-frames-per-demo", type=int, default=4,
                        help="frames sampled from each demonstration to find the directions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image

    source = DATA_ROOT / f"demos_{args.rule}"
    target = DATA_ROOT / f"encoded_{args.rule}__{args.prompt}"
    target.mkdir(parents=True, exist_ok=True)
    goals = {r["key"]: r["object"]
             for r in json.loads((source / "index.json").read_text())["records"]}
    template = PROMPTS[args.prompt]

    shards = sorted(source.glob("*.npz"))
    pending = [s for s in shards if not (target / s.name).exists()]
    print(f"[encode] rule={args.rule} prompt={args.prompt}: {len(shards)} buildings, "
          f"{len(pending)} to do", flush=True)
    if not pending:
        print("[encode] every building already encoded")
        return

    pca_path = target / PCA_FILE
    if args.pca_components and not pca_path.exists() and len(pending) != len(shards):
        raise SystemExit(
            f"{len(shards) - len(pending)} buildings are already encoded but {pca_path} is "
            "missing, so the directions used for them cannot be recovered and a new fit "
            "would put the remaining buildings in a different coordinate system; delete "
            f"{target} and encode from the start")

    encoder = MultiTokenEncoder(args.max_new_tokens)
    cache: dict = {}
    pca_report: dict = {"applied": False}
    if args.pca_components:
        if pca_path.exists():
            encoder.projection = Projection.load(pca_path)
            pca_report = {"applied": True, "reused": True,
                          "components": encoder.projection.width,
                          "explained_variance_ratio": encoder.projection.explained}
            print(f"[encode] reusing the directions in {pca_path}", flush=True)
        else:
            projection, cache, report = fit_projection(
                encoder, shards, goals, template, components=args.pca_components,
                frames_per_demo=args.pca_frames_per_demo,
                rng=np.random.default_rng(args.seed))
            projection.save(pca_path)
            encoder.projection = projection
            pca_report = {"applied": True, "reused": False, **report}
            print(f"[encode] kept {projection.width} of {encoder.hidden} directions per "
                  f"layer, holding {projection.explained:.3f} of the variation -> "
                  f"{pca_path}", flush=True)
    print(f"[encode] ready: {encoder.tokens} tokens per frame, width {encoder.width}",
          flush=True)

    started, frames, reused, examples = time.time(), 0, 0, []
    for position, shard in enumerate(pending, start=1):
        payload = np.load(shard)
        stored = {}
        for key in demo_keys(payload):
            images = payload[f"{key}|rgb"]
            question = template.format(goal=goal_phrase(goals.get(key, "object")))
            block = np.empty((len(images), encoder.tokens, encoder.width), dtype=np.float16)
            for index, frame in enumerate(images):
                seen = cache.pop((shard.name, key, index), None)
                if seen is not None:                 # already encoded to fit the directions
                    block[index] = encoder.flatten(seen).astype(np.float16)
                    reused += 1
                    continue
                picture = Image.fromarray(frame[..., :3].astype("uint8"), mode="RGB")
                tokens, answer = encoder.encode(picture, question)
                block[index] = tokens.astype(np.float16)
                if len(examples) < 6 and index == 0:
                    examples.append({"key": key, "question": question, "answer": answer})
            stored[f"{key}|tokens"] = block
            stored[f"{key}|label"] = payload[f"{key}|label"]
            stored[f"{key}|oracle"] = payload[f"{key}|oracle"]
            frames += len(images)
        np.savez_compressed(target / shard.name, **stored)
        rate = frames / max(time.time() - started, 1e-6)
        print(f"  [{position}/{len(pending)}] {shard.stem}: {frames} frames so far, "
              f"{rate:.2f} frames/s", flush=True)

    manifest = {"rule": args.rule, "prompt": args.prompt, "prompt_template": template,
                "tokens_per_frame": encoder.tokens, "token_width": encoder.width,
                "layers": list(LAYERS), "image_grid": IMAGE_GRID, "text_tokens": TEXT_TOKENS,
                "pca": pca_report, "frames": frames, "frames_reused_from_pca_fit": reused,
                "seconds": round(time.time() - started, 1),
                "example_answers": examples}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[encode] {frames} frames -> {target}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "example_answers"}, indent=2))


if __name__ == "__main__":
    main()
