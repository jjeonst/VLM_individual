"""Read what the VLM actually answers for each candidate navigation prompt, and time it.

Before committing to a large encoding run we need to know two things about every candidate
prompt: whether the VLM produces a substantive, navigation-relevant answer at all, and how
long one encoding takes. A prompt that elicits "I am not sure" is useless as a
representation, and the per-frame time determines how many episodes we can afford.

For each sample frame the script generates an answer under every candidate prompt, prints
it for human inspection, and records wall-clock time for the generation and for the
representation forward pass.

Run (7B VLM, 24 GB GPU):
  python vlm_topology_test/check_prompt_candidates.py --frames 5 --max-new-tokens 48
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
RGB_DIR = Path("/data/topovlm/habitat/rgb/pr2l_hm3d_objectnav/train")
MANIFEST = Path("/data/topovlm/habitat/episodes/pr2l_hm3d_objectnav/train/manifest.jsonl")
OUT = Path(__file__).resolve().parents[0] / "results" / "prompt_candidates_check.json"

CANDIDATES = {
    "baseline_presence":
        "Would a {goal} be found here? Why or why not?",
    "A_direction":
        "Which direction should I move to find a {goal}? Explain what you see.",
    "C_structure_and_goal":
        "Where can I go from here, and which of those directions most likely leads to a "
        "{goal}? Why?",
    "D_room_connectivity":
        "What room is this? Which rooms connect from here, and where would a {goal} be?",
}


class PromptProbe:
    def __init__(self, max_new_tokens: int):
        import torch
        from prismatic import load

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.vlm = load(WEIGHTS, hf_token=os.environ.get("HF_TOKEN"))
        self.vlm.to(self.device, dtype=self.dtype)
        self.vlm.requires_grad_(False)
        self.vlm.eval()

    def _prompt(self, text: str, answer: str | None = None) -> str:
        builder = self.vlm.get_prompt_builder()
        builder.add_turn(role="human", message=text)
        if answer is not None:
            builder.add_turn(role="gpt", message=answer)
        return builder.get_prompt()

    def run(self, image, template: str, goal: str) -> dict:
        torch, vlm = self.torch, self.vlm
        question = template.format(goal=goal)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
        started = time.time()
        answer = vlm.generate(image, self._prompt(question), do_sample=False,
                              max_new_tokens=self.max_new_tokens)
        torch.cuda.synchronize()
        generation_seconds = time.time() - started

        prompt = self._prompt(question, answer)
        input_ids = vlm.llm_backbone.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attention = torch.ones_like(input_ids)
        pixel_values = vlm.vision_backbone.image_transform(image)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device, self.dtype) for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values[None, ...].to(self.device, self.dtype)

        torch.cuda.synchronize()
        started = time.time()
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
        last_token = output.hidden_states[-1][0, -1, :].float().cpu().numpy()
        torch.cuda.synchronize()
        forward_seconds = time.time() - started

        return {"question": question, "answer": answer,
                "answer_words": len(str(answer).split()),
                "generation_seconds": round(generation_seconds, 3),
                "forward_seconds": round(forward_seconds, 3),
                "total_seconds": round(generation_seconds + forward_seconds, 3),
                "representation_norm": round(float(np.linalg.norm(last_token)), 3)}


def sample_frames(count: int):
    """Take frames from different episodes and different points along each trajectory."""
    goals = {}
    for line in MANIFEST.open():
        record = json.loads(line)
        goals[record["episode_id"]] = record.get("object_category")
    chosen = []
    for path in sorted(glob.glob(str(RGB_DIR / "*.npy"))):
        episode_id = Path(path).stem
        frames = np.load(path, mmap_mode="r")
        if len(frames) < 8:
            continue
        for fraction in (0.25, 0.6):
            index = int(len(frames) * fraction)
            chosen.append({"episode": episode_id, "frame_index": index,
                           "goal": goals.get(episode_id, "chair"),
                           "image": np.asarray(frames[index])})
            if len(chosen) >= count:
                return chosen
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    from PIL import Image

    frames = sample_frames(args.frames)
    print(f"[prompts] {len(frames)} frames, {len(CANDIDATES)} candidates, "
          f"max_new_tokens={args.max_new_tokens}", flush=True)

    probe = PromptProbe(args.max_new_tokens)
    print("[prompts] Prismatic loaded\n", flush=True)

    records = []
    for item in frames:
        image = Image.fromarray(item["image"][..., :3].astype("uint8"), mode="RGB")
        print(f"=== frame: {item['episode'][:45]} step {item['frame_index']} "
              f"goal={item['goal']} ===", flush=True)
        for name, template in CANDIDATES.items():
            result = probe.run(image, template, item["goal"])
            print(f"  [{name}] ({result['total_seconds']}s, {result['answer_words']} words)",
                  flush=True)
            print(f"    Q: {result['question']}", flush=True)
            print(f"    A: {result['answer']}\n", flush=True)
            records.append({"episode": item["episode"], "frame_index": item["frame_index"],
                            "goal": item["goal"], "candidate": name, **result})

    timing = {}
    for name in CANDIDATES:
        rows = [r for r in records if r["candidate"] == name]
        timing[name] = {
            "mean_total_seconds": round(float(np.mean([r["total_seconds"] for r in rows])), 3),
            "mean_generation_seconds": round(float(np.mean([r["generation_seconds"] for r in rows])), 3),
            "mean_forward_seconds": round(float(np.mean([r["forward_seconds"] for r in rows])), 3),
            "mean_answer_words": round(float(np.mean([r["answer_words"] for r in rows])), 1),
        }

    result = {"config": {"frames": len(frames), "max_new_tokens": args.max_new_tokens,
                         "candidates": CANDIDATES},
              "timing_per_frame": timing, "records": records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("=== timing per frame (seconds) ===")
    print(json.dumps(timing, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
