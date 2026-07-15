"""Week 3 — Generate concrete topology-probe instances from Week 2 artifacts.

Proves the benchmark in docs/week3_benchmark_proposal.md is constructible TODAY:
each probe gets real (prompt, gold-answer) instances built mechanically from the
R2R overlap pairs and the replayed ObjectNav positions produced in Week 2.

Probes:
  1. Shared-subpath detection   (R2R)      -> probe1_shared_subpath.json
  2. Trajectory stitching        (R2R)      -> probe2_stitching.json
  3. Branch validity judgment    (ObjectNav)-> probe3_branch_validity.json
  4. Bottleneck / doorway id      (ObjectNav)-> probe4_bottleneck.json

Input is serialized as bare (x, z) waypoint coordinates (no instruction text), so
a correct answer needs spatial/topology reasoning rather than language priors.

Usage:
  python -m survey.week3.build_probe_examples --split val_unseen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from survey.week2.find_r2r_overlaps import (
    load_r2r_paths, nearest_dists, longest_run_under, arc_length,
)
from survey.week2.draw_r2r_stitch import build_stitch, _find

REPO_ROOT = Path(__file__).resolve().parents[2]
W2 = REPO_ROOT / "survey" / "week2" / "results" / "objectnav_branching"
OUT = REPO_ROOT / "survey" / "week3" / "results" / "probes"

R2R_MATCH = 1.0  # metres, matches the ~1 m downsample spacing used for prompts


# --------------------------------------------------------------------------- #
def downsample(pos: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a path (T,>=3 or T,2) to ~``spacing`` m arc-length steps, (K,2) xz."""
    xz = pos[:, [0, 2]] if pos.shape[1] >= 3 else pos
    seg = np.sqrt((np.diff(xz, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.arange(0.0, cum[-1] + 1e-9, spacing)
    out = [xz[np.searchsorted(cum, t, side="right") - 1] for t in targets]
    out.append(xz[-1])
    arr = np.asarray(out, dtype=float)
    # drop consecutive duplicates
    keep = np.concatenate([[True], (np.abs(np.diff(arr, axis=0)).sum(1) > 1e-6)])
    return arr[keep]


def ser(xz: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(z), 2)] for x, z in xz]


# --------------------------------------------------------------------------- #
def probe1_and_2(split: str, pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    paths = load_r2r_paths(split)
    p1_items, p2_items = [], []
    for epa, epb in pairs:
        a = _find(paths, epa)
        b = _find(paths, epb)
        A = downsample(a.positions, 1.0)
        B = downsample(b.positions, 1.0)
        # gold shared run on the DOWNSAMPLED sequences (what the model sees)
        da = nearest_dists(np.c_[A[:, 0], np.zeros(len(A)), A[:, 1]],
                           np.c_[B[:, 0], np.zeros(len(B)), B[:, 1]])
        db = nearest_dists(np.c_[B[:, 0], np.zeros(len(B)), B[:, 1]],
                           np.c_[A[:, 0], np.zeros(len(A)), A[:, 1]])
        _, ia0, ia1 = longest_run_under(da, R2R_MATCH)
        _, ib0, ib1 = longest_run_under(db, R2R_MATCH)

        p1_items.append({
            "id": f"r2r_{a.scene_key}_{epa}_{epb}",
            "input": {"route_A": ser(A), "route_B": ser(B)},
            "question": "Do route A and route B share a contiguous common subpath? "
                        "If yes, give the index ranges A[i0:i1] and B[j0:j1] of the shared segment.",
            "gold": {"shares_subpath": True,
                     "A_range": [ia0, ia1], "B_range": [ib0, ib1]},
            "correctness_criterion": "index-range IoU >= 0.5 on both A and B, and shares_subpath matches",
            "source": f"R2R {split} {a.scene_key} ep {epa}<->{epb}",
        })

        # gold stitched route (full-resolution build_stitch, then downsample)
        info = build_stitch(a, b, match_dist=0.5)
        stitched = downsample(info["stitched"], 1.0)
        p2_items.append({
            "id": f"r2r_stitch_{a.scene_key}_{epa}_{epb}",
            "input": {"route_A": ser(A), "route_B": ser(B)},
            "question": "Can you stitch a NEW route from A's start to B's goal that reuses "
                        "their shared corridor without discontinuities? If yes, output the "
                        "waypoint sequence.",
            "gold": {"stitchable": True,
                     "junction_xz": [round(float(info["stitch_join"][0]), 2),
                                     round(float(info["stitch_join"][2]), 2)],
                     "stitched_route": ser(stitched),
                     "start_xz": ser(A[:1])[0], "goal_xz": ser(B[-1:])[0]},
            "correctness_criterion": "sequence starts at A.start, ends at B.goal, every "
                                     "consecutive gap <= 1.5 m (connectivity preserved), "
                                     "and passes within 1.0 m of junction_xz",
            "source": f"R2R {split} {a.scene_key} ep {epa}<->{epb}",
        })
    return (
        {"probe": "shared_subpath_detection", "instances": p1_items},
        {"probe": "trajectory_stitching", "instances": p2_items},
    )


# --------------------------------------------------------------------------- #
def _load_scene_npz(scene_key: str) -> dict[str, np.ndarray]:
    data = np.load(W2 / "positions" / "train" / f"{scene_key}.npz")
    return {k: data[k] for k in data.files}


def probe4_bottleneck() -> dict:
    st = json.loads((W2 / "train" / "spatial_branch.json").read_text())
    sc = st["best_bottleneck_scene"]
    gold_cell = st["best_scene_top_cells"][0]
    trajs = _load_scene_npz(sc)
    # show a manageable subset of routes
    chosen = list(trajs.items())[:8]
    routes = [ser(downsample(pos, 1.0)) for _, pos in chosen]
    return {
        "probe": "bottleneck_identification",
        "instances": [{
            "id": f"objnav_bottleneck_{sc}",
            "input": {"routes": routes},
            "question": "These are expert routes in one scene. Which single location "
                        "(x, z) do the most distinct routes pass through (the bottleneck / doorway)?",
            "gold": {"bottleneck_xz": gold_cell["xz"],
                     "traffic_fraction": gold_cell["fraction"],
                     "routes_shown": len(routes),
                     "routes_total_in_scene": st["best_scene_num_trajectories"]},
            "correctness_criterion": "answer within 0.75 m of bottleneck_xz (or in the "
                                     "scene's top-3 traffic cells)",
            "source": f"ObjectNav scene {sc} (busiest cell {gold_cell['fraction']:.0%} of "
                      f"{st['best_scene_num_trajectories']} routes)",
        }],
    }


def probe3_branch_validity() -> dict:
    st = json.loads((W2 / "train" / "spatial_branch.json").read_text())
    items = []
    # (a) a dead-end excursion -> gold = dead-end (returns near its start)
    de = st["deadend_examples"][0]
    trajs = _load_scene_npz(de["scene_key"])
    pos = trajs[de["episode_id"]]
    i, j = de["return_from"], de["return_to"]
    excursion = downsample(pos[i:j + 1], 1.0)
    items.append({
        "id": f"objnav_branch_deadend_{de['scene_key']}",
        "input": {"segment": ser(excursion), "target_object": de["object"]},
        "question": "Taking this next segment: does it make progress to a new region "
                    "toward the goal, or is it a dead-end that returns near where it started?",
        "gold": {"label": "dead-end", "detour_m": de["detour_m"], "gap_closed_m": de["gap_closed_m"]},
        "correctness_criterion": "label == dead-end (segment endpoint within ~1 m of its start "
                                 "after a long detour)",
        "source": f"ObjectNav dead-end {de['scene_key']} ep …{de['episode_id'][-6:]}",
    })
    # (b) a clean route -> gold = progress (start far from stop)
    sc = st["best_bottleneck_scene"]
    btrajs = _load_scene_npz(sc)
    dead_ids = {d["episode_id"] for d in st["deadend_examples"]}
    clean = next((p for ep, p in btrajs.items()
                  if ep not in dead_ids and float(np.hypot(*(p[-1, [0, 2]] - p[0, [0, 2]]))) > 3.0), None)
    if clean is not None:
        seg = downsample(clean, 1.0)
        items.append({
            "id": f"objnav_branch_progress_{sc}",
            "input": {"segment": ser(seg), "target_object": None},
            "question": "Taking this next segment: does it make progress to a new region "
                        "toward the goal, or is it a dead-end that returns near where it started?",
            "gold": {"label": "goal-route",
                     "start_to_end_m": round(float(np.hypot(*(seg[-1] - seg[0]))), 2)},
            "correctness_criterion": "label == goal-route (endpoint far from start)",
            "source": f"ObjectNav clean route {sc}",
        })
    return {"probe": "branch_validity_judgment", "instances": items}


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val_unseen", help="R2R split for probes 1-2")
    parser.add_argument("--pairs", nargs="*", default=["94:1114", "52:1807"],
                        help="R2R episode pairs 'epA:epB' for probes 1-2")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(p.split(":")) for p in args.pairs]

    p1, p2 = probe1_and_2(args.split, pairs)
    p3 = probe3_branch_validity()
    p4 = probe4_bottleneck()

    for fname, payload in [
        ("probe1_shared_subpath.json", p1),
        ("probe2_stitching.json", p2),
        ("probe3_branch_validity.json", p3),
        ("probe4_bottleneck.json", p4),
    ]:
        (OUT / fname).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        n = len(payload["instances"])
        print(f"  wrote {fname}: {n} instance(s)")
    print(f"probe instances -> {OUT}")


if __name__ == "__main__":
    main()
