"""Stitching probe, EXACT PR2L representation.

Same question as phase1_pr2l_stitch_probe.py, but instead of the last-layer /
last-token hidden state, we use the *identical* representation the PR2L policy
consumes, reproduced end-to-end:

  encode_image_goal_tokens (prompt + generated answer, last-two-layer visual
  tokens, 4x4 adaptive-avg pool, bank mean-reduce  ->  16 tokens x 8192)
    -> PCA projection (8192 -> 1024, the fitted embeddings/pr2l_hm3d_bc)
    -> mean over the 16 tokens  ->  1024-d frame vector  ->  L2 normalise

This is the vector the GraphTransformerPolicy sees per node (we token-mean-pool
it to one vector for a place-similarity comparison). Config mirrors the graph
cache metadata exactly:
  layers=-2,-1  pool=4  bank=mean  projection=pca:1024  include_generated_text=True
  prompt="Would a {goal_text} be found here? Why or why not?"

For every stitch pair A,B (mined identically to the last-token probe):
  positive : rep(A @ junction)  vs  rep(B @ junction)   # same place, 2 routes
  negative : rep(A @ junction)  vs  rep(B @ start), rep(B @ goal)  # diff place
Report mean sims, separation, AUC, junction-ranked-top, vs a raw-pixel baseline.

Run (needs HF cache + PCA at /data/topovlm/habitat/embeddings/pr2l_hm3d_bc):
  python vlm_topology_test/phase1_pr2l_stitch_probe_exact.py --scenes 106
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm_topology_test.phase1_stitching_study.phase1_build_instances import (  # noqa: E402
    POS_DIR, load_scene, find_pairs, obj_of, load_frame)

WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
PCA_PATH = "/data/topovlm/habitat/embeddings/pr2l_hm3d_bc/projection_pca.npz"
OUT = Path(__file__).resolve().parents[0] / "results" / "pr2l_stitch_probe_exact.json"


def build_encoder():
    """PrismaticEncoder with the EXACT deployed PR2L representation config."""
    import os
    from configs.schema import VLMConfig
    from encoders.prismatic import PrismaticEncoder
    cfg = VLMConfig(
        backend="prismatic",
        model_id="prism-dinosiglip+7b",
        weights_path=WEIGHTS,
        hf_token_path=None,  # falls back to HF_TOKEN env
        device="cuda",
        dtype="bfloat16",
        frozen=True,
        representation="pr2l_visual_tokens_last_two_layers",
        hidden_layer_indices=[-2, -1],
        visual_pool_grid=4,
        visual_bank_reduction="mean",
        projection="pca",
        projection_path=PCA_PATH,
        projection_dim=1024,
        include_generated_text=True,
        generation_seed=0,
        generation_temperature=0.4,
        max_new_tokens=48,
        output_dim=1024,
        prompt_template="Would a {goal_text} be found here? Why or why not?",
    )
    _ = os  # (env HF_TOKEN read inside encoder)
    return PrismaticEncoder(cfg)


class PR2LExactRep:
    """Reproduce the per-node PR2L vector: pooled visual tokens -> PCA -> token-mean."""

    def __init__(self):
        from PIL import Image
        self.Image = Image
        self.enc = build_encoder()
        z = np.load(PCA_PATH)
        self.mean = z["mean"].astype("float32")             # (8192,)
        self.components = z["components"].astype("float32")  # (1024, 8192)

    def rep(self, image, goal_text: str) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = self.Image.fromarray(image.astype("uint8"), mode="RGB")
        tokens = self.enc.encode_image_goal_tokens(image, goal_text)["tokens"]  # (16, 8192)
        tokens = np.asarray(tokens, dtype="float32")
        projected = (tokens - self.mean) @ self.components.T   # (16, 1024)  identical to cache
        v = projected.mean(axis=0)                             # token-mean -> (1024,)
        return v / (np.linalg.norm(v) + 1e-8)


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

    rep = PR2LExactRep()
    print("[rep] PR2L exact representation ready (Prismatic + PCA)", flush=True)

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
        eAj = rep.rep(fAj, gA); eBj = rep.rep(fBj, gB)
        eBs = rep.rep(fBs, gB); eBg = rep.rep(fBg, gB)
        pos = cos(eAj, eBj)
        neg1, neg2 = cos(eAj, eBs), cos(eAj, eBg)
        vlm_pos.append(pos); vlm_neg += [neg1, neg2]
        pAj, pBj, pBs, pBg = map(pixel_vec, (fAj, fBj, fBs, fBg))
        px_pos.append(cos(pAj, pBj)); px_neg += [cos(pAj, pBs), cos(pAj, pBg)]
        per_pair.append({"scene": p["scene"], "epA": epA, "epB": epB,
                         "junction_sim": round(pos, 3),
                         "neg_start": round(neg1, 3), "neg_goal": round(neg2, 3),
                         "correct_ranked_top": bool(pos > max(neg1, neg2))})
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{len(pairs)} pairs done", flush=True)

    result = {
        "representation": "PR2L exact: layers=-2,-1 pool=4 bank=mean pca:1024 gen_text=True, token-mean-pooled",
        "num_pairs": len(pairs),
        "VLM_embedding": {
            "junction_sim_mean": round(float(np.mean(vlm_pos)), 3),
            "nonjunction_sim_mean": round(float(np.mean(vlm_neg)), 3),
            "separation": round(float(np.mean(vlm_pos) - np.mean(vlm_neg)), 3),
            "AUC": round(auc(vlm_pos, vlm_neg), 3),
            "pct_junction_ranked_top": round(100 * np.mean([d["correct_ranked_top"] for d in per_pair]), 1),
            "junction_sim_min": round(float(np.min(vlm_pos)), 3),
            "junction_sim_max": round(float(np.max(vlm_pos)), 3),
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
    print("\n=== RESULT (EXACT PR2L representation: shared-junction recognition) ===")
    print(json.dumps({k: v for k, v in result.items() if k != "per_pair"}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
