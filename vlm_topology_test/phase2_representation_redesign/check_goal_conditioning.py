"""Does the extracted PR2L representation actually depend on the navigation goal?

The repository extracts the policy's state representation from the *visual token*
positions of the Prismatic VLM (``encoders/prismatic.py:encode_image_goal_tokens``).
In the fused input sequence those positions come before the prompt:

    [BOS] [visual tokens ...] [prompt text] [generated answer]

Llama-2 is a causal decoder, so a position can only attend to positions to its left.
If that is the case here, the hidden states at the visual-token positions cannot see the
prompt, which means the extracted representation is independent of which object the agent
was asked to find, and independent of the answer the VLM generated.

This script tests that directly. For each sample frame it encodes the SAME image under
several DIFFERENT goal objects and reports, for two extraction points,

  - ``visual_tokens``: the pooled last-two-layer visual tokens the repository uses,
  - ``last_token``:    the final-position hidden state, which has read image, prompt and
                       generated answer,

how much the representation changes when only the goal changes. A maximum absolute
difference of exactly zero means the representation carries no goal information.

Run (7B VLM, needs a 24 GB GPU):
  python vlm_topology_test/check_goal_conditioning.py --frames 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
RGB_DIR = Path("/data/topovlm/habitat/rgb/pr2l_hm3d_objectnav/train")
OUT = Path(__file__).resolve().parents[0] / "results" / "goal_conditioning_check.json"
PROMPT = "Would a {goal} be found here? Why or why not?"
GOALS = ["chair", "toilet", "bed", "plant"]


class Probe:
    """Runs one Prismatic forward pass and returns several candidate representations."""

    def __init__(self, generate_answer: bool):
        import torch
        from prismatic import load

        self.torch = torch
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.generate_answer = generate_answer
        self.vlm = load(WEIGHTS, hf_token=os.environ.get("HF_TOKEN"))
        self.vlm.to(self.device, dtype=self.dtype)
        self.vlm.requires_grad_(False)
        self.vlm.eval()

    def _prompt_for(self, goal: str, answer: str | None):
        builder = self.vlm.get_prompt_builder()
        builder.add_turn(role="human", message=PROMPT.format(goal=goal))
        if answer is not None:
            builder.add_turn(role="gpt", message=answer)
        return builder.get_prompt()

    def encode(self, image, goal: str) -> dict:
        torch, vlm = self.torch, self.vlm
        answer = None
        if self.generate_answer:
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            answer = vlm.generate(image, self._prompt_for(goal, None), do_sample=False,
                                  max_new_tokens=32)
        prompt = self._prompt_for(goal, answer)
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
        # what the repository extracts: last two layers at the visual-token positions
        visual = torch.cat([output.hidden_states[i][0, 1:1 + visual_count, :].float()
                            for i in (-2, -1)], dim=-1).cpu().numpy()
        # what "the stage right before the output" means: the final position
        last = output.hidden_states[-1][0, -1, :].float().cpu().numpy()
        return {"visual_tokens": visual, "last_token": last, "answer": answer,
                "visual_count": visual_count, "sequence_length": int(fused.shape[1])}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    difference = np.abs(a - b)
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    denominator = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    cosine = float(np.dot(flat_a, flat_b) / denominator) if denominator > 0 else float("nan")
    return {"max_abs_diff": round(float(difference.max()), 8),
            "mean_abs_diff": round(float(difference.mean()), 8),
            "cosine": round(cosine, 6),
            "identical": bool(difference.max() == 0.0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--generate", action="store_true",
                        help="also let the VLM generate an answer before extraction")
    args = parser.parse_args()

    from PIL import Image

    episodes = sorted(glob.glob(str(RGB_DIR / "*.npy")))
    images = []
    for path in episodes:
        frames = np.load(path, mmap_mode="r")
        if len(frames) > 5:
            images.append(np.asarray(frames[len(frames) // 2]))
        if len(images) >= args.frames:
            break
    print(f"[check] {len(images)} frames, goals={GOALS}, generate_answer={args.generate}",
          flush=True)

    probe = Probe(generate_answer=args.generate)
    print("[check] Prismatic loaded", flush=True)

    per_frame = []
    for index, frame in enumerate(images):
        image = Image.fromarray(frame[..., :3].astype("uint8"), mode="RGB")
        encodings = {}
        for goal in GOALS:
            encodings[goal] = probe.encode(image, goal)
            print(f"  frame {index} goal={goal}: seq_len={encodings[goal]['sequence_length']} "
                  f"visual_tokens={encodings[goal]['visual_count']} "
                  f"answer={str(encodings[goal]['answer'])[:60]!r}", flush=True)
        base = GOALS[0]
        record = {"frame": index, "answers": {g: encodings[g]["answer"] for g in GOALS},
                  "visual_tokens": {}, "last_token": {}}
        for goal in GOALS[1:]:
            record["visual_tokens"][f"{base}_vs_{goal}"] = compare(
                encodings[base]["visual_tokens"], encodings[goal]["visual_tokens"])
            record["last_token"][f"{base}_vs_{goal}"] = compare(
                encodings[base]["last_token"], encodings[goal]["last_token"])
        per_frame.append(record)

    visual_identical = all(c["identical"] for r in per_frame for c in r["visual_tokens"].values())
    last_identical = all(c["identical"] for r in per_frame for c in r["last_token"].values())
    verdict = {
        "visual_token_representation_is_goal_independent": visual_identical,
        "last_token_representation_is_goal_independent": last_identical,
        "interpretation": (
            "Visual-token representations are byte-identical across goals, so the extraction "
            "point the repository uses carries no goal or language information."
            if visual_identical else
            "Visual-token representations change with the goal, so the extraction point does "
            "carry goal information and the causal-ordering concern does not apply."),
    }
    result = {"config": {"frames": len(images), "goals": GOALS,
                         "generate_answer": args.generate, "prompt": PROMPT},
              "verdict": verdict, "per_frame": per_frame}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("\n=== goal-conditioning verdict ===")
    print(json.dumps(verdict, indent=2))
    example = per_frame[0]
    print("\nframe 0, chair vs toilet:")
    print("  visual_tokens:", example["visual_tokens"].get("chair_vs_toilet"))
    print("  last_token   :", example["last_token"].get("chair_vs_toilet"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
