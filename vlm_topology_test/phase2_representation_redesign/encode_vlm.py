"""Turn collected camera frames into vision-language model representations.

This is step 2 of the experiment described in ``experiment_design_2x2.md``. It reads the
RGB frames written by ``nav_baseline.collect`` and, for every frame, asks the vision-language
model a question about the episode's target object, lets it generate an answer, and stores
the model's internal state as the representation the policy will later consume.

Two points about how the representation is taken:

- **The question is a factor of the experiment.** Two questions are available. The baseline
  question asks whether the target object is present, which is what the prior work used. The
  structure question asks which directions lead onward and which of them is likely to reach
  the target, which is the condition this study adds.
- **The representation is read from the last position of the input sequence**, after the
  model has read the image, the question, and its own generated answer. Reading it from the
  image positions instead—as the earlier implementation did—yields a vector that is
  identical no matter which object was requested, because the language backbone only lets a
  position attend to positions before it. Section 2.3 of the design document reports that
  measurement.

Because the model is frozen, a frame and a question always produce the same vector, so this
computation is done once and its output is reused by every training run. The script writes
one file per scene and skips scenes that already have output, so an interrupted run can be
restarted with the same command.

Run (7B model, 24 GB GPU):
  python -m vlm_topology_test.phase2_representation_redesign.encode_vlm \
      --data exp_shortest_800 --prompt structure
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/data/topovlm/nav_baseline/data")
OUT_ROOT = Path("/data/topovlm/nav_baseline/encoded")
WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"

PROMPTS = {
    "baseline": "Would a {goal} be found here? Why or why not?",
    "structure": ("Where can I go from here, and which of those directions most likely "
                  "leads to a {goal}? Why?"),
}


class LastTokenEncoder:
    """Encode one frame into the hidden state at the final position of the input sequence."""

    def __init__(self, max_new_tokens: int = 48):
        import torch
        from prismatic import load

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        # Every condition must be encoded with the same numeric format, otherwise the
        # representations differ systematically between conditions and the comparison is
        # confounded. bfloat16 needs an Ampere or newer GPU, so refuse older ones up front
        # rather than after ten minutes of model loading.
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"{torch.cuda.get_device_name(0)} does not support bfloat16; schedule this "
                "job on a GPU that does so that all conditions share one numeric format")
        self.vlm = load(WEIGHTS, hf_token=os.environ.get("HF_TOKEN"))
        self.vlm.to(self.device, dtype=self.dtype)
        self.vlm.requires_grad_(False)
        self.vlm.eval()
        self.dim = int(self.vlm.llm_backbone.llm.config.hidden_size)

    def _prompt(self, question: str, answer: str | None = None) -> str:
        builder = self.vlm.get_prompt_builder()
        builder.add_turn(role="human", message=question)
        if answer is not None:
            builder.add_turn(role="gpt", message=answer)
        return builder.get_prompt()

    def encode(self, image, question: str) -> tuple[np.ndarray, str]:
        torch, vlm = self.torch, self.vlm
        # Greedy decoding keeps the representation deterministic, so a rerun reproduces it.
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
        vector = output.hidden_states[-1][0, -1, :].float().cpu().numpy()
        return vector, str(answer)


def episode_goals(data_dir: Path) -> dict[str, str]:
    """Map each stored episode key to the object the agent was asked to find."""
    index = json.loads((data_dir / "index.json").read_text())
    return {record["key"]: record["object"] for record in index["records"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="collection directory under the data root")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="encode every Nth frame; 1 keeps the per-step rate the policy "
                             "will run at, avoiding a train/deploy mismatch")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit-scenes", type=int, default=0, help="0 = all scenes")
    args = parser.parse_args()

    from PIL import Image

    data_dir = DATA_ROOT / args.data
    out_dir = OUT_ROOT / f"{args.data}__{args.prompt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    goals = episode_goals(data_dir)
    template = PROMPTS[args.prompt]

    shards = sorted(data_dir.glob("*.npz"))
    if args.limit_scenes:
        shards = shards[: args.limit_scenes]
    pending = [s for s in shards if not (out_dir / s.name).exists()]
    print(f"[encode] {args.data} x prompt={args.prompt} | {len(shards)} scenes, "
          f"{len(pending)} still to do, stride={args.frame_stride}", flush=True)
    if not pending:
        print("[encode] nothing to do; all scenes already encoded", flush=True)
        return

    encoder = LastTokenEncoder(args.max_new_tokens)
    print(f"[encode] model loaded, representation dimension {encoder.dim}", flush=True)

    started = time.time()
    total_frames = 0
    samples_of_text = []
    for scene_index, shard in enumerate(pending, start=1):
        payload = np.load(shard)
        keys = sorted({name.split("|")[0] for name in payload.files})
        stored = {}
        for key in keys:
            frames = payload[f"{key}|rgb"][:: args.frame_stride]
            goal = goals.get(key, "object")
            question = template.format(goal=goal)
            vectors = np.empty((len(frames), encoder.dim), dtype=np.float16)
            for index, frame in enumerate(frames):
                image = Image.fromarray(frame[..., :3].astype("uint8"), mode="RGB")
                vector, answer = encoder.encode(image, question)
                vectors[index] = vector.astype(np.float16)
                if len(samples_of_text) < 8 and index == 0:
                    samples_of_text.append({"episode": key, "goal": goal, "answer": answer})
            stored[f"{key}|repr"] = vectors
            stored[f"{key}|label"] = payload[f"{key}|label"][:: args.frame_stride]
            stored[f"{key}|prev_action"] = payload[f"{key}|prev_action"][:: args.frame_stride]
            stored[f"{key}|goal_text"] = np.asarray(goal)
            total_frames += len(frames)
        np.savez_compressed(out_dir / shard.name, **stored)
        elapsed = time.time() - started
        rate = total_frames / max(elapsed, 1e-6)
        remaining = (len(pending) - scene_index) * (total_frames / scene_index)
        print(f"  [{scene_index}/{len(pending)}] {shard.stem}: {len(keys)} episodes, "
              f"{total_frames} frames so far, {rate:.2f} frames/s, "
              f"~{remaining / max(rate, 1e-6) / 60:.0f} min left", flush=True)

    manifest = {
        "data": args.data, "prompt": args.prompt, "prompt_template": template,
        "extraction": "last position of the fused sequence, final layer",
        "frame_stride": args.frame_stride, "max_new_tokens": args.max_new_tokens,
        "representation_dim": encoder.dim, "frames_encoded": total_frames,
        "seconds": round(time.time() - started, 1),
        "example_answers": samples_of_text,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[encode] {total_frames} frames -> {out_dir}", flush=True)
    print(json.dumps({k: v for k, v in manifest.items() if k != "example_answers"}, indent=2))


if __name__ == "__main__":
    main()
