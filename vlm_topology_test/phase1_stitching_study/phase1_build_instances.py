"""Phase 1 — build observation-based stitching MC instances from HM3D.

Mines stitch pairs from the replayed ObjectNav positions (two trajectories A,B
that share a subpath with divergent start/goal), then constructs a multiple-choice
question with real egocentric RGB frames:
  - CORRECT : A[start_A .. junction] ⊕ B[junction .. goal_B]  (novel stitched route)
  - D1 broken-seam : A prefix ⊕ B suffix joined at a WRONG (spatially jumped) point
  - D3 wrong-goal  : route A itself (ends at goal_A, not goal_B)
  - D4 reversed    : A prefix ⊕ B toward start_B (ends at wrong node)
Each route is shown as K subsampled panoramic-forward RGB frames.

Positions: survey/week2/results/objectnav_branching/positions/train/<scene>.npz
RGB:       /data/topovlm/habitat/rgb/pr2l_hm3d_objectnav/train/<episode_id>.npy
(frame index t aligned to position index t via the replay.)

Outputs (vlm_topology_test/results/instances/):
  - <scene>_<epA>_<epB>.png   montage (rows = A, B, and the 4 options)
  - <scene>_<epA>_<epB>.json  manifest (episode ids, junction, frame indices, gold)

Usage:
  python vlm_topology_test/phase1_build_instances.py --top 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `survey` import
from survey.week2.find_r2r_overlaps import nearest_dists, longest_run_under, arc_length

REPO = Path(__file__).resolve().parents[2]
POS_DIR = REPO / "survey/week2/results/objectnav_branching/positions/train"
RGB_DIR = Path("/data/topovlm/habitat/rgb/pr2l_hm3d_objectnav/train")
MANIFEST = Path("/data/topovlm/habitat/episodes/pr2l_hm3d_objectnav/train/manifest.jsonl")
OUT = REPO / "vlm_topology_test/results/instances"

_OBJ = None
def obj_of(ep):
    """Target object category of an episode (goal identity)."""
    global _OBJ
    if _OBJ is None:
        _OBJ = {}
        for line in MANIFEST.open():
            r = json.loads(line)
            _OBJ[r["episode_id"]] = r.get("object_category")
    return _OBJ.get(ep)

# NOTE: every scene's ObjectNav episodes target a SINGLE object category, so
# goal_A and goal_B are always the same category. We therefore cannot require
# different-object goals; instead we require the goal FRAMES to be visually
# distinct (DINOv2 cosine < GOAL_SIM_MAX) so "reached goal_A" vs "reached goal_B"
# is distinguishable (addresses 'the two goals look the same').
MATCH = 0.6          # m, shared-subpath tolerance
MIN_RUN_M = 1.0      # m, minimum shared subpath (a junction crossing)
MAX_RUN_M = 15.0     # m
MIN_DIVERGE = 3.0    # m, start and goal must differ by this
MIN_TAIL = 1.0       # m, minimal distinct tail
MIN_LEN = 20         # min trajectory length (steps)
SEAM_JUMP = 2.0      # m, broken-seam minimum spatial jump
GOAL_SIM_MAX = 0.6   # DINOv2 cosine above this = goals look the same → reject
K = 5                # frames shown per route


def xz(p):
    return p[:, [0, 2]]


def rgb_path(ep):
    return RGB_DIR / f"{ep}.npy"


def load_scene(scene):
    trajs = {k: v for k, v in np.load(POS_DIR / f"{scene}.npz").items()
             if v.shape[0] >= MIN_LEN and rgb_path(k).exists()}
    return trajs


def find_pairs(trajs):
    """Yield stitch-pair dicts: shared subpath + divergent endpoints."""
    items = list(trajs.items())
    out = []
    for i in range(len(items)):
        epA, A = items[i]
        for j in range(i + 1, len(items)):
            epB, B = items[j]
            objA, objB = obj_of(epA), obj_of(epB)  # same category in this data
            dA = nearest_dists(A, B)
            dB = nearest_dists(B, A)
            _, ia0, ia1 = longest_run_under(dA, MATCH)
            _, ib0, ib1 = longest_run_under(dB, MATCH)
            if ia1 - ia0 < 2 or ib1 - ib0 < 2:
                continue
            run_m = max(arc_length(A, ia0, ia1), arc_length(B, ib0, ib1))
            if not (MIN_RUN_M <= run_m <= MAX_RUN_M):
                continue
            aXZ, bXZ = xz(A), xz(B)
            start_div = np.hypot(*(aXZ[0] - bXZ[0]))
            goal_div = np.hypot(*(aXZ[-1] - bXZ[-1]))
            if start_div < MIN_DIVERGE or goal_div < MIN_DIVERGE:
                continue
            # four divergent tails: distinct approach AND departure beyond the shared run
            preA = arc_length(A, 0, ia0); postA = arc_length(A, ia1, len(A) - 1)
            preB = arc_length(B, 0, ib0); postB = arc_length(B, ib1, len(B) - 1)
            min_tail = min(preA, postA, preB, postB)
            if min_tail < MIN_TAIL:
                continue
            # junction = middle of A's shared run; matching index on B
            tJ_A = (ia0 + ia1) // 2
            pJ = aXZ[tJ_A]
            tJ_B = int(np.argmin(np.hypot(*(bXZ - pJ).T)))
            if len(B) - tJ_B < 5:
                continue
            out.append({
                "epA": epA, "epB": epB, "A": A, "B": B, "objA": objA, "objB": objB,
                "tJ_A": tJ_A, "tJ_B": tJ_B, "junction_xz": [float(pJ[0]), float(pJ[1])],
                "run_m": round(float(run_m), 2), "min_tail_m": round(float(min_tail), 2),
                "start_div": round(float(start_div), 2), "goal_div": round(float(goal_div), 2),
                "score": min_tail + 0.3 * run_m,
            })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def route_steps(pair):
    """Return dict of option -> list of (episode_id, frame_index)."""
    epA, epB = pair["epA"], pair["epB"]
    A, B = pair["A"], pair["B"]
    tJ_A, tJ_B = pair["tJ_A"], pair["tJ_B"]
    bXZ = xz(B); pJ = np.array(pair["junction_xz"])
    lenA, lenB = len(A), len(B)

    correct = [(epA, i) for i in range(0, tJ_A + 1)] + [(epB, j) for j in range(tJ_B, lenB)]
    d3 = [(epA, i) for i in range(0, lenA)]
    d4 = [(epA, i) for i in range(0, tJ_A + 1)] + [(epB, j) for j in range(tJ_B, -1, -1)]

    # D1 broken seam: continue B from a point spatially jumped from the junction
    jump_idx = None
    for j in range(tJ_B + 1, lenB - 3):
        if np.hypot(*(bXZ[j] - pJ)) >= SEAM_JUMP:
            jump_idx = j
            break
    d1 = None
    if jump_idx is not None:
        d1 = [(epA, i) for i in range(0, tJ_A + 1)] + [(epB, j) for j in range(jump_idx, lenB)]
    return {"correct": correct, "D1_broken_seam": d1, "D3_wrong_goal": d3, "D4_reversed": d4}


def subsample(steps, k=K):
    if len(steps) <= k:
        return steps
    idx = np.linspace(0, len(steps) - 1, k).round().astype(int)
    return [steps[i] for i in idx]


def load_frame(ep, t):
    arr = np.load(rgb_path(ep), mmap_mode="r")
    return np.asarray(arr[min(max(t, 0), arr.shape[0] - 1)]).astype(np.uint8)


# --- DINOv2 visual-distinctness of goal frames (lazy-loaded) ---
_DINO = None
def _dino():
    global _DINO
    if _DINO is None:
        import torch, timm
        from PIL import Image
        m = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m",
                              pretrained=True, num_classes=0).eval().cuda()
        cfg = timm.data.resolve_data_config({}, model=m)
        tf = timm.data.create_transform(**cfg)
        def embed(ep, t):
            img = Image.fromarray(load_frame(ep, t))
            x = tf(img).unsqueeze(0).cuda()
            with torch.inference_mode():
                f = m(x)[0]
            return (f / f.norm()).cpu().numpy()
        _DINO = embed
    return _DINO


def goal_frame_sim(pair):
    embed = _dino()
    gA = embed(pair["epA"], len(pair["A"]) - 1)
    gB = embed(pair["epB"], len(pair["B"]) - 1)
    return float(np.dot(gA, gB))


def draw_instance(pair, options, gold_key, out_png):
    # context rows (A, B) + option rows
    ctx = {
        "route A (start_A→goal_A)": [(pair["epA"], i) for i in range(len(pair["A"]))],
        "route B (start_B→goal_B)": [(pair["epB"], j) for j in range(len(pair["B"]))],
    }
    rows = list(ctx.items()) + [(k, v) for k, v in options.items() if v is not None]
    fig, axes = plt.subplots(len(rows), K, figsize=(K * 2.3, len(rows) * 1.9))
    for r, (label, steps) in enumerate(rows):
        fs = subsample(steps)
        for c in range(K):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(fs):
                ep, t = fs[c]
                ax.imshow(load_frame(ep, t))
            if c == 0:
                ax.set_title(label, fontsize=8, loc="left", x=0.0, y=0.5,
                             ha="right", rotation=0)
    # row labels on the left
    for r, (label, _) in enumerate(rows):
        axes[r, 0].text(-0.05, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8, fontweight="bold")
    jx = [round(v, 1) for v in pair["junction_xz"]]
    fig.suptitle(
        f"Stitching MC — scene {pair['scene']}   Q: which route goes start_A -> goal_B "
        f"(goal_A={pair['objA']}, goal_B={pair['objB']})?   gold = {gold_key}\n"
        f"shared {pair['run_m']}m, min divergent tail {pair['min_tail_m']}m, "
        f"start_div {pair['start_div']}m, goal_div {pair['goal_div']}m, junction {jx}", fontsize=10)
    fig.tight_layout(rect=(0.08, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--scenes", type=int, default=20, help="how many scenes to scan")
    args = ap.parse_args()

    scenes = sorted(p.stem for p in POS_DIR.glob("*.npz"))[: args.scenes]
    all_pairs = []
    for sc in scenes:
        trajs = load_scene(sc)
        if len(trajs) < 2:
            continue
        for pr in find_pairs(trajs)[:2]:
            pr["scene"] = sc
            all_pairs.append(pr)
    all_pairs.sort(key=lambda d: d["score"], reverse=True)
    print(f"scanned {len(scenes)} scenes → {len(all_pairs)} stitch-pair candidates")

    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    rejected_sim = 0
    for pair in all_pairs:
        options = route_steps(pair)
        if options["D1_broken_seam"] is None:
            continue  # need a valid broken-seam distractor
        # goal frames must be visually distinct (else goal_A ≈ goal_B)
        gsim = goal_frame_sim(pair)
        pair["goal_sim"] = round(gsim, 2)
        if gsim > GOAL_SIM_MAX:
            rejected_sim += 1
            continue
        name = f"{pair['scene']}_{pair['epA'][-4:]}_{pair['epB'][-4:]}"
        draw_instance(pair, options, "correct", OUT / f"{name}.png")
        manifest = {
            "scene": pair["scene"], "epA": pair["epA"], "epB": pair["epB"],
            "goal_A_object": pair["objA"], "goal_B_object": pair["objB"],
            "goal_frame_sim": pair["goal_sim"],
            "junction_xz": pair["junction_xz"], "shared_run_m": pair["run_m"],
            "min_tail_m": pair["min_tail_m"],
            "start_div_m": pair["start_div"], "goal_div_m": pair["goal_div"],
            "gold": "correct",
            "options": {k: ([[e, int(t)] for e, t in subsample(v)] if v else None)
                        for k, v in options.items()},
            "question": "route A와 route B가 주어졌을 때, start_A에서 goal_B로 가는 올바른 경로는?",
        }
        (OUT / f"{name}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"  [{made+1}] {name}: {pair['objA']}(goals), shared {pair['run_m']}m, "
              f"goal_div {pair['goal_div']}m, goal_sim {pair['goal_sim']}")
        made += 1
        if made >= args.top:
            break
    print(f"wrote {made} instances (rejected {rejected_sim} for goal_sim>{GOAL_SIM_MAX}) → {OUT}")


if __name__ == "__main__":
    main()
