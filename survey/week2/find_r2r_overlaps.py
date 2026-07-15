"""Week 2 · Step 1 — Mine overlapping R2R-VLN-CE trajectory pairs.

Goal (from docs/individual_researcher_topology_survey.md, Week 2):
  Find two or more navigation trajectories that share a physical subpath, so we
  can later draw a stitching figure (shared subpath + branch/merge + stitched
  route).

Method (dataset-native only, no model/training):
  - Load R2R-VLN-CE GT paths. Episode metadata lives in ``{split}.json.gz``
    (episode_id, scene_id, instruction) and the dense ground-truth path lives in
    ``{split}_gt.json.gz`` keyed by episode_id under ``locations`` = [[x,y,z],..].
    (Same source/convention as analysis/code/predictive_branching_datasets.py.)
  - Habitat world axes: x = locations[:,0], z = locations[:,2] are the floor
    plane; y is height and is dropped.
  - Overlap is measured by **actual floor-plane distance** between the two dense
    paths: a point counts as "on the shared corridor" when it is within
    ``match_dist`` metres of the other path. This is deliberately NOT a fixed
    grid: a grid can split two points only a few cm apart across a cell boundary,
    which fragments the contiguous shared run and misplaces the merge point.
  - For every trajectory pair *within the same scene* we compute:
      * shared_points        : how many of A's / B's points lie within match_dist
      * longest_shared_run_m : arc length (metres) of the longest *contiguous*
                               stretch that stays within match_dist of the other
                               path -- the "B -> C" shared subpath length
      * start/goal diverge   : whether the paired endpoints are > match_dist apart
                               (a stitch yields a NEW route only when they differ)

Outputs (survey/week2/results/r2r_overlaps/<split>/):
  - overlap_pairs.json  : ranked candidate pairs with full detail + instructions
  - overlap_pairs.csv   : same, flat table for eyeballing in a spreadsheet
  - summary.json        : run configuration + per-scene episode counts

Usage:
  python -m survey.week2.find_r2r_overlaps --split val_unseen --top 25
  python survey/week2/find_r2r_overlaps.py --split val_unseen --match-dist 0.5
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TOPOVLM_DATASET_ROOT", "/data/topovlm"))
R2R_DATASET_ROOT = (
    DATA_ROOT
    / "r2r_vlnce_v1_3_preprocessed"
    / "extracted"
    / "R2R_VLNCE_v1-3_preprocessed"
)
RESULT_ROOT = REPO_ROOT / "survey" / "week2" / "results" / "r2r_overlaps"


@dataclass(frozen=True)
class R2RPath:
    episode_id: str
    trajectory_id: int
    scene_id: str
    scene_key: str
    instruction_text: str
    positions: np.ndarray  # (T, 3) world coords [x, y, z]


# --------------------------------------------------------------------------- #
# Shared distance helpers (also imported by draw_r2r_stitch.py so Step 1 and
# Step 2 detect overlap with the exact same, grid-free definition).
# --------------------------------------------------------------------------- #
def nearest_dists(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """For each floor-plane point of ``p``, the distance to the nearest ``q`` point."""
    p_xz = p[:, [0, 2]]
    q_xz = q[:, [0, 2]]
    diff = p_xz[:, None, :] - q_xz[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1)).min(axis=1)


def longest_run_under(vals: np.ndarray, thr: float) -> tuple[int, int, int]:
    """Longest contiguous run of indices whose value is <= ``thr``. (len, start, end)."""
    best_len, best_start, best_end = 0, 0, 0
    run_start = 0
    run = 0
    for i, v in enumerate(vals):
        if v <= thr:
            if run == 0:
                run_start = i
            run += 1
            if run > best_len:
                best_len, best_start, best_end = run, run_start, i
        else:
            run = 0
    return best_len, best_start, best_end


def arc_length(positions: np.ndarray, i0: int, i1: int) -> float:
    """Floor-plane path length (metres) of the segment positions[i0..i1] inclusive."""
    if i1 <= i0:
        return 0.0
    seg = positions[i0 : i1 + 1][:, [0, 2]]
    return float(np.sqrt((np.diff(seg, axis=0) ** 2).sum(axis=1)).sum())


def _xz_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.hypot(p[0] - q[0], p[2] - q[2]))


# --------------------------------------------------------------------------- #
def _load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _scene_key(scene_id: str) -> str:
    return Path(scene_id).stem


def load_r2r_paths(split: str, dedupe_trajectories: bool = True) -> list[R2RPath]:
    """Load R2R GT paths.

    R2R annotates each physical route (``trajectory_id``) with ~3 different
    language instructions, each stored as its own episode sharing the *same* GT
    ``locations``. With ``dedupe_trajectories`` we keep one path per
    (scene, trajectory_id) so physically identical routes don't flood the ranking.
    """
    split_dir = R2R_DATASET_ROOT / split
    episodes_payload = _load_json_gz(split_dir / f"{split}.json.gz")
    gt_payload = _load_json_gz(split_dir / f"{split}_gt.json.gz")

    paths: list[R2RPath] = []
    seen: set[tuple[str, int]] = set()
    for episode in episodes_payload["episodes"]:
        episode_id = str(episode["episode_id"])
        gt = gt_payload.get(episode_id)
        if gt is None:
            continue
        positions = np.asarray(gt.get("locations", []), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] < 3:
            continue
        positions = positions[:, :3]
        scene_id = str(episode["scene_id"])
        trajectory_id = int(episode.get("trajectory_id", -1))
        if dedupe_trajectories:
            key = (scene_id, trajectory_id)
            if key in seen:
                continue
            seen.add(key)
        paths.append(
            R2RPath(
                episode_id=episode_id,
                trajectory_id=trajectory_id,
                scene_id=scene_id,
                scene_key=_scene_key(scene_id),
                instruction_text=str(
                    episode.get("instruction", {}).get("instruction_text", "")
                ),
                positions=positions,
            )
        )
    return paths


def _bbox(positions: np.ndarray) -> tuple[float, float, float, float]:
    xz = positions[:, [0, 2]]
    return float(xz[:, 0].min()), float(xz[:, 0].max()), float(xz[:, 1].min()), float(xz[:, 1].max())


def _bboxes_far(a: np.ndarray, b: np.ndarray, margin: float) -> bool:
    ax0, ax1, az0, az1 = _bbox(a)
    bx0, bx1, bz0, bz1 = _bbox(b)
    return (ax0 - bx1 > margin or bx0 - ax1 > margin
            or az0 - bz1 > margin or bz0 - az1 > margin)


def score_scene_pairs(
    paths: list[R2RPath],
    match_dist: float,
    min_shared_run_m: float,
) -> list[dict[str, Any]]:
    """All within-scene pairs whose longest contiguous shared run >= min_shared_run_m."""
    by_scene: dict[str, list[R2RPath]] = defaultdict(list)
    for path in paths:
        by_scene[path.scene_id].append(path)

    candidates: list[dict[str, Any]] = []
    for scene_id, scene_paths in by_scene.items():
        n = len(scene_paths)
        for i in range(n):
            a = scene_paths[i]
            for j in range(i + 1, n):
                b = scene_paths[j]
                if _bboxes_far(a.positions, b.positions, match_dist):
                    continue

                d_a = nearest_dists(a.positions, b.positions)
                d_b = nearest_dists(b.positions, a.positions)
                shared_a = int((d_a <= match_dist).sum())
                shared_b = int((d_b <= match_dist).sum())
                if shared_a == 0 or shared_b == 0:
                    continue

                _, ia0, ia1 = longest_run_under(d_a, match_dist)
                _, ib0, ib1 = longest_run_under(d_b, match_dist)
                run_m = max(arc_length(a.positions, ia0, ia1),
                            arc_length(b.positions, ib0, ib1))
                if run_m < min_shared_run_m:
                    continue

                # Divergent tails: how much unique path each side has before/after
                # the shared corridor. All four large => clean "X->B->C->Y" shape.
                pre_a = arc_length(a.positions, 0, ia0)
                post_a = arc_length(a.positions, ia1, len(a.positions) - 1)
                pre_b = arc_length(b.positions, 0, ib0)
                post_b = arc_length(b.positions, ib1, len(b.positions) - 1)
                min_tail = min(pre_a, post_a, pre_b, post_b)
                # Do the two paths traverse the shared corridor the same way round?
                same_dir = (_xz_dist(a.positions[ia0], b.positions[ib0])
                            <= _xz_dist(a.positions[ia0], b.positions[ib1]))

                start_diff = _xz_dist(a.positions[0], b.positions[0]) > match_dist
                goal_diff = _xz_dist(a.positions[-1], b.positions[-1]) > match_dist
                candidates.append(
                    {
                        "scene_key": a.scene_key,
                        "scene_id": scene_id,
                        "episode_id_a": a.episode_id,
                        "episode_id_b": b.episode_id,
                        "trajectory_id_a": a.trajectory_id,
                        "trajectory_id_b": b.trajectory_id,
                        "a_len_points": int(len(a.positions)),
                        "b_len_points": int(len(b.positions)),
                        "shared_points_a": shared_a,
                        "shared_points_b": shared_b,
                        "longest_shared_run_m": round(run_m, 2),
                        "min_divergent_tail_m": round(float(min_tail), 2),
                        "same_direction": bool(same_dir),
                        "shared_fraction_a": round(shared_a / len(a.positions), 3),
                        "shared_fraction_b": round(shared_b / len(b.positions), 3),
                        "start_diverges": bool(start_diff),
                        "goal_diverges": bool(goal_diff),
                        # A genuine stitch produces a NEW route only when endpoints differ.
                        "stitch_candidate": bool(start_diff or goal_diff),
                        # The textbook "A->B->C->D vs X->B->C->Y" shape: shared middle,
                        # both ends distinct => cleanest branch + merge to visualise.
                        "both_ends_diverge": bool(start_diff and goal_diff),
                        "instruction_a": a.instruction_text,
                        "instruction_b": b.instruction_text,
                    }
                )
    return candidates


def rank_candidates(candidates: list[dict[str, Any]], rank_by: str = "run") -> list[dict[str, Any]]:
    """Rank overlap candidates.

    rank_by="run"     : strongest raw overlap (longest shared corridor first).
                        Surfaces near-identical routes -- best *evidence* that a
                        shared subpath exists.
    rank_by="balance" : cleanest "X->B->C->Y" figure first -- pairs where both
                        paths keep substantial unique tails on both sides of the
                        shared corridor (largest min_divergent_tail_m).
    """
    if rank_by == "balance":
        key = lambda c: (
            c["both_ends_diverge"],
            round(c["min_divergent_tail_m"], 1),
            c["longest_shared_run_m"],
        )
    else:
        key = lambda c: (
            c["both_ends_diverge"],
            c["stitch_candidate"],
            c["longest_shared_run_m"],
            c["shared_points_a"] + c["shared_points_b"],
        )
    return sorted(candidates, key=key, reverse=True)


def _truncate(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_outputs(
    split: str,
    match_dist: float,
    min_shared_run_m: float,
    paths: list[R2RPath],
    ranked: list[dict[str, Any]],
    top: int,
) -> Path:
    out_dir = RESULT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    top_rows = ranked[:top]

    (out_dir / "overlap_pairs.json").write_text(
        json.dumps(top_rows, indent=2, ensure_ascii=False) + "\n"
    )

    csv_fields = [
        "scene_key",
        "episode_id_a",
        "episode_id_b",
        "trajectory_id_a",
        "trajectory_id_b",
        "longest_shared_run_m",
        "min_divergent_tail_m",
        "same_direction",
        "shared_points_a",
        "shared_points_b",
        "a_len_points",
        "b_len_points",
        "shared_fraction_a",
        "shared_fraction_b",
        "start_diverges",
        "goal_diverges",
        "stitch_candidate",
        "both_ends_diverge",
    ]
    with (out_dir / "overlap_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in top_rows:
            writer.writerow({k: row[k] for k in csv_fields})

    by_scene: dict[str, int] = defaultdict(int)
    for path in paths:
        by_scene[path.scene_key] += 1
    summary = {
        "split": split,
        "match_dist_m": match_dist,
        "min_shared_run_m": min_shared_run_m,
        "num_paths_loaded": len(paths),
        "num_scenes": len(by_scene),
        "episodes_per_scene": dict(sorted(by_scene.items())),
        "num_candidate_pairs": len(ranked),
        "num_stitch_candidates": sum(1 for c in ranked if c["stitch_candidate"]),
        "num_both_ends_diverge": sum(1 for c in ranked if c["both_ends_diverge"]),
        "top_written": len(top_rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return out_dir


def print_table(ranked: list[dict[str, Any]], top: int) -> None:
    header = (
        f"{'#':>2}  {'scene':<12} {'epA':>6} {'epB':>6} "
        f"{'run_m':>6} {'minTail':>7} {'dir':>4} {'lenA':>5} {'lenB':>5}  "
        f"{'stitch':<7} both_ends"
    )
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(ranked[:top], start=1):
        print(
            f"{idx:>2}  {row['scene_key']:<12} "
            f"{row['episode_id_a']:>6} {row['episode_id_b']:>6} "
            f"{row['longest_shared_run_m']:>6} {row['min_divergent_tail_m']:>7} "
            f"{('same' if row['same_direction'] else 'opp'):>4} "
            f"{row['a_len_points']:>5} {row['b_len_points']:>5}  "
            f"{('yes' if row['stitch_candidate'] else 'no'):<7} "
            f"{'yes' if row['both_ends_diverge'] else 'no'}"
        )
    if ranked:
        best = ranked[0]
        print("\nTop pair instructions:")
        print(f"  A (ep {best['episode_id_a']}): {_truncate(best['instruction_a'])}")
        print(f"  B (ep {best['episode_id_b']}): {_truncate(best['instruction_b'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="val_unseen",
        help="R2R split to mine (val_unseen has 11 dense scenes; good for Step 1).",
    )
    parser.add_argument(
        "--match-dist",
        type=float,
        default=0.5,
        help="Two paths 'share' a point where they come within this many metres "
        "(coordinate distance, not a grid). Larger = more tolerant matching.",
    )
    parser.add_argument(
        "--min-shared-run",
        type=float,
        default=2.0,
        help="Keep only pairs whose longest contiguous shared run is at least this "
        "many metres (filters incidental crossings).",
    )
    parser.add_argument(
        "--rank-by",
        choices=["run", "balance"],
        default="run",
        help="run = strongest overlap (default); balance = cleanest X->B->C->Y "
        "figure (both paths keep long unique tails on both sides).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="How many top-ranked pairs to write / print.",
    )
    args = parser.parse_args()

    print(f"Loading R2R split '{args.split}' from {R2R_DATASET_ROOT} ...")
    paths = load_r2r_paths(args.split)
    print(f"  loaded {len(paths)} trajectories")

    candidates = score_scene_pairs(paths, args.match_dist, args.min_shared_run)
    ranked = rank_candidates(candidates, args.rank_by)
    print(
        f"  found {len(ranked)} candidate pairs "
        f"(shared run >= {args.min_shared_run} m within {args.match_dist} m), "
        f"{sum(1 for c in ranked if c['both_ends_diverge'])} with both endpoints divergent\n"
    )

    print_table(ranked, args.top)
    out_dir = write_outputs(
        args.split, args.match_dist, args.min_shared_run, paths, ranked, args.top
    )
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
