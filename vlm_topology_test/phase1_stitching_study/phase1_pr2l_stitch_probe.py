"""Stitching, PR2L-style: does the Prismatic VLM *embedding* (the representation
PR2L feeds to the RL policy) recognise the shared junction that makes stitching
possible?

For every stitch pair A,B (mined WITHOUT the goal filter — all cases):
  positive : embed(A @ junction)  vs  embed(B @ junction)   # same place, 2 routes
  negative : embed(A @ junction)  vs  embed(B @ start), embed(B @ goal)  # diff place
If the embedding is topology-aware, positive similarity > negative similarity, i.e.
the representation knows A and B pass through the same junction (→ stitchable).
We report mean sims, separation, and AUC, vs a raw-pixel baseline.

Uses the SAME Prismatic VLM as PR2L (encoders/prismatic.py). No Qwen, no chat MC.

Run (needs the HF cache from the smoke test):
  HF_HOME=/home/jonghoon/hf_cache HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 \
  python vlm_topology_test/phase1_pr2l_stitch_probe.py --scenes 106
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm_topology_test.phase1_stitching_study.phase1_build_instances import (  # noqa: E402
    POS_DIR, load_scene, find_pairs, obj_of, load_frame, xz)

WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
OUT = Path(__file__).resolve().parents[0] / "results" / "pr2l_stitch_probe.json"
PROMPT = "Would a {goal} be found here? Why or why not?"


# ---------------- Prismatic embedding (PR2L representation) ----------------
class PrismaticEmbedder:
    def __init__(self):
        import torch
        from prismatic import load
        self.torch = torch
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        import os
        self.vlm = load(WEIGHTS, hf_token=os.environ.get("HF_TOKEN"))
        self.vlm.to(self.device, dtype=self.dtype)
        self.vlm.requires_grad_(False)
        self.vlm.eval()

    def embed(self, image, goal_text: str) -> np.ndarray:
        torch, vlm = self.torch, self.vlm
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image.astype(np.uint8))
        pb = vlm.get_prompt_builder()
        pb.add_turn(role="human", message=PROMPT.format(goal=goal_text))
        prompt = pb.get_prompt()
        input_ids = vlm.llm_backbone.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attn = torch.ones_like(input_ids)
        px = vlm.vision_backbone.image_transform(image)
        if isinstance(px, dict):
            px = {k: v[None, ...].to(self.device, self.dtype) for k, v in px.items()}
        else:
            px = px[None, ...].to(self.device, self.dtype)
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            patch = vlm.vision_backbone(px)
            proj = vlm.projector(patch)
            emb = vlm.llm_backbone.embed_input_ids(input_ids)
            fused = torch.cat([emb[:, :1], proj, emb[:, 1:]], dim=1)
            pmask = torch.full((proj.shape[0], proj.shape[1]), True, dtype=attn.dtype, device=self.device)
            fmask = torch.cat([attn[:, :1], pmask, attn[:, 1:]], dim=1)
            out = vlm.llm_backbone(input_ids=None, attention_mask=fmask, inputs_embeds=fused,
                                   output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()  # last-layer, last token
        return h / (np.linalg.norm(h) + 1e-8)


def cos(u, v):
    return float(np.dot(u, v))


def pixel_vec(frame):
    g = frame.astype(np.float32).mean(-1)
    g = np.asarray([[g[int(i * g.shape[0] / 32)][int(j * g.shape[1] / 32)]
                     for j in range(32)] for i in range(32)]).ravel()
    g = g - g.mean()
    return g / (np.linalg.norm(g) + 1e-8)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=106)
    ap.add_argument("--max-pairs", type=int, default=60)
    args = ap.parse_args()

    scenes = sorted(p.stem for p in POS_DIR.glob("*.npz"))[: args.scenes]
    pairs = []
    for sc in scenes:
        trajs = load_scene(sc)
        if len(trajs) < 2:
            continue
        for pr in find_pairs(trajs)[:2]:
            pr["scene"] = sc
            pairs.append(pr)
    pairs.sort(key=lambda d: d["score"], reverse=True)
    pairs = pairs[: args.max_pairs]
    print(f"mined {len(pairs)} stitch pairs (no goal filter) over {len(scenes)} scenes", flush=True)

    emb = PrismaticEmbedder()
    print("[embedder] Prismatic loaded", flush=True)

    vlm_pos, vlm_neg, px_pos, px_neg = [], [], [], []
    per_pair = []
    for k, p in enumerate(pairs):
        epA, epB = p["epA"], p["epB"]
        gA, gB = obj_of(epA), obj_of(epB)
        tJA, tJB = p["tJ_A"], p["tJ_B"]
        fAj = load_frame(epA, tJA)
        fBj = load_frame(epB, tJB)
        fBs = load_frame(epB, 0)
        fBg = load_frame(epB, len(p["B"]) - 1)
        eAj = emb.embed(fAj, gA); eBj = emb.embed(fBj, gB)
        eBs = emb.embed(fBs, gB); eBg = emb.embed(fBg, gB)
        pos = cos(eAj, eBj)
        neg1, neg2 = cos(eAj, eBs), cos(eAj, eBg)
        vlm_pos.append(pos); vlm_neg += [neg1, neg2]
        # raw-pixel baseline
        pAj, pBj, pBs, pBg = map(pixel_vec, (fAj, fBj, fBs, fBg))
        px_pos.append(cos(pAj, pBj)); px_neg += [cos(pAj, pBs), cos(pAj, pBg)]
        per_pair.append({"scene": p["scene"], "epA": epA, "epB": epB,
                         "junction_sim": round(pos, 3),
                         "neg_start": round(neg1, 3), "neg_goal": round(neg2, 3),
                         "correct_ranked_top": bool(pos > max(neg1, neg2))})
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(pairs)} pairs embedded", flush=True)

    result = {
        "num_pairs": len(pairs),
        "VLM_embedding": {
            "junction_sim_mean": round(float(np.mean(vlm_pos)), 3),
            "nonjunction_sim_mean": round(float(np.mean(vlm_neg)), 3),
            "separation": round(float(np.mean(vlm_pos) - np.mean(vlm_neg)), 3),
            "AUC": round(auc(vlm_pos, vlm_neg), 3),
            "pct_junction_ranked_top": round(100 * np.mean([d["correct_ranked_top"] for d in per_pair]), 1),
        },
        "raw_pixel_baseline": {
            "junction_sim_mean": round(float(np.mean(px_pos)), 3),
            "nonjunction_sim_mean": round(float(np.mean(px_neg)), 3),
            "AUC": round(auc(px_pos, px_neg), 3),
        },
        "per_pair": per_pair,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("\n=== RESULT (does PR2L VLM embedding recognise the shared junction?) ===")
    print(json.dumps({k: v for k, v in result.items() if k != "per_pair"}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
