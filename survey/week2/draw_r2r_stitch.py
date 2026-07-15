"""Week 2 · Step 2 — Draw the R2R stitching figure for a chosen trajectory pair.

Takes one overlapping pair found by ``find_r2r_overlaps.py`` and renders a
top-down (floor-plane) figure showing:
  - trajectory A and trajectory B as two coloured routes,
  - their shared subpath (the contiguous corridor both traverse) highlighted,
  - the merge junction (where the two routes join the shared corridor) and the
    branch junction (where they split toward different goals),
  - a stitched route = A's prefix + shared corridor + B's suffix, i.e. a NEW
    route (start_A -> goal_B) that neither original trajectory walked.

This is the "navigation stitching figure" deliverable (assignment output #2).
Coordinates: Habitat floor plane (x, z); y is height and is dropped.

Defaults target the headline val_unseen example: a clean, same-direction
"X->B->C->Y" pair (scene QUCTc6BB5sX, episode 94 <-> 1114) picked from
``find_r2r_overlaps.py --rank-by balance``. Override with CLI flags for others.

Usage:
  python -m survey.week2.draw_r2r_stitch
  python survey/week2/draw_r2r_stitch.py --split val_unseen \
      --episode-a 94 --episode-b 1114
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from survey.week2.find_r2r_overlaps import (
    R2RPath,
    load_r2r_paths,
    longest_run_under,
    nearest_dists,
)

RESULT_ROOT = Path(__file__).resolve().parents[2] / "survey" / "week2" / "results" / "r2r_overlaps"

COLOR_A = "#1f77b4"  # trajectory A
COLOR_B = "#ff7f0e"  # trajectory B
COLOR_SHARED = "#7b3fa0"  # shared subpath
COLOR_STITCH = "#2ca02c"  # stitched (new) route


def _xz(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return positions[:, 0], positions[:, 2]


def _find(paths: list[R2RPath], episode_id: str) -> R2RPath:
    for p in paths:
        if p.episode_id == episode_id:
            return p
    raise SystemExit(f"episode_id {episode_id!r} not found in this split")


def build_stitch(a: R2RPath, b: R2RPath, match_dist: float) -> dict:
    """Compute shared-run ranges, junctions, and a stitched route (start_A -> goal_B).

    Overlap is detected by actual floor-plane distance between the two dense GT
    paths (points within ``match_dist`` metres count as "on the same corridor"),
    using the same ``nearest_dists`` / ``longest_run_under`` helpers as Step 1.
    """
    d_a = nearest_dists(a.positions, b.positions)
    d_b = nearest_dists(b.positions, a.positions)
    len_a, i0, i1 = longest_run_under(d_a, match_dist)
    len_b, j0, j1 = longest_run_under(d_b, match_dist)
    if len_a == 0 or len_b == 0:
        raise SystemExit("the two episodes do not come within match_dist anywhere")

    # Orientation: does A's shared-run entry (A[i0]) line up with B[j0] or B[j1]?
    def d(p, q) -> float:
        return float(np.hypot(p[0] - q[0], p[2] - q[2]))

    same_dir = d(a.positions[i0], b.positions[j0]) <= d(a.positions[i0], b.positions[j1])

    # Build a stitched route that ALWAYS ends at goal_B (= B[-1]), independent of
    # travel direction. B's tail toward goal_B leaves the shared corridor at B[j1]
    # (goal_B lies beyond j1). We attach it to whichever A shared-run boundary
    # coincides with B[j1] so the join is geometrically continuous:
    #   - same direction : B[j1] ~ A[i1] (A's exit)  -> route reuses the corridor
    #   - opposite dir.   : B[j1] ~ A[i0] (A's entry) -> route joins at that junction
    b_goal_tail = b.positions[j1:]              # B[j1] -> ... -> goal_B
    junction = b.positions[j1]
    a_cut = i1 if d(a.positions[i1], junction) <= d(a.positions[i0], junction) else i0
    a_head = a.positions[: a_cut + 1]           # start_A -> junction
    stitched = (
        np.vstack([a_head, b_goal_tail[1:]]) if len(b_goal_tail) > 1 else a_head
    )

    return {
        "a_run": (i0, i1),
        "b_run": (j0, j1),
        "same_dir": same_dir,
        "merge_pt": a.positions[i0],   # where A joins the shared corridor
        "branch_pt": a.positions[i1],  # where A leaves the shared corridor
        "stitch_join": junction,       # junction where A's head meets B's goal-tail
        "shared_poly": b.positions[j0 : j1 + 1],  # the shared corridor polyline
        "stitched": stitched,
    }


def draw(a: R2RPath, b: R2RPath, info: dict, out_path: Path, match_dist: float) -> None:
    fig, (ax_main, ax_stitch) = plt.subplots(1, 2, figsize=(15, 7))

    for ax in (ax_main, ax_stitch):
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")

    # ---- Left: overlay with shared subpath + junctions ----
    ax_main.plot(*_xz(a.positions), "-", color=COLOR_A, lw=2, alpha=0.9,
                 label=f"trajectory A (ep {a.episode_id})")
    ax_main.plot(*_xz(b.positions), "-", color=COLOR_B, lw=2, alpha=0.9,
                 label=f"trajectory B (ep {b.episode_id})")
    ax_main.plot(*_xz(info["shared_poly"]), "-", color=COLOR_SHARED, lw=6, alpha=0.45,
                 solid_capstyle="round", label="shared subpath")

    # endpoints
    ax_main.scatter(*a.positions[0, [0, 2]], color=COLOR_A, marker="o", s=90,
                    edgecolor="black", zorder=5, label="start A")
    ax_main.scatter(*a.positions[-1, [0, 2]], color=COLOR_A, marker="*", s=240,
                    edgecolor="black", zorder=5, label="goal A")
    ax_main.scatter(*b.positions[0, [0, 2]], color=COLOR_B, marker="o", s=90,
                    edgecolor="black", zorder=5, label="start B")
    ax_main.scatter(*b.positions[-1, [0, 2]], color=COLOR_B, marker="*", s=240,
                    edgecolor="black", zorder=5, label="goal B")

    # junctions
    ax_main.scatter(info["merge_pt"][0], info["merge_pt"][2], color="white",
                    marker="P", s=200, edgecolor=COLOR_SHARED, linewidths=2, zorder=6)
    ax_main.annotate("merge\n(join corridor)", (info["merge_pt"][0], info["merge_pt"][2]),
                     textcoords="offset points", xytext=(8, 8), fontsize=9,
                     color=COLOR_SHARED, fontweight="bold")
    ax_main.scatter(info["branch_pt"][0], info["branch_pt"][2], color="white",
                    marker="X", s=200, edgecolor="crimson", linewidths=2, zorder=6)
    ax_main.annotate("branch\n(split to goals)", (info["branch_pt"][0], info["branch_pt"][2]),
                     textcoords="offset points", xytext=(8, -20), fontsize=9,
                     color="crimson", fontweight="bold")

    ax_main.set_title(
        f"R2R shared subpath — scene {a.scene_key}\n"
        f"shared run: A[{info['a_run'][0]}:{info['a_run'][1]}], "
        f"B[{info['b_run'][0]}:{info['b_run'][1]}]  (match≤{match_dist} m)"
    )
    ax_main.legend(loc="best", fontsize=8, framealpha=0.9)

    # ---- Right: the stitched NEW route ----
    ax_stitch.plot(*_xz(a.positions), "-", color=COLOR_A, lw=1, alpha=0.25)
    ax_stitch.plot(*_xz(b.positions), "-", color=COLOR_B, lw=1, alpha=0.25)
    ax_stitch.plot(*_xz(info["stitched"]), "--", color=COLOR_STITCH, lw=3,
                   label="stitched route: start A → goal B")
    ax_stitch.scatter(*a.positions[0, [0, 2]], color=COLOR_A, marker="o", s=90,
                      edgecolor="black", zorder=5, label="start A")
    ax_stitch.scatter(*b.positions[-1, [0, 2]], color=COLOR_B, marker="*", s=240,
                      edgecolor="black", zorder=5, label="goal B")
    ax_stitch.set_title("Stitched route (A prefix + shared corridor + B suffix)\n"
                        "a route neither trajectory walked")
    ax_stitch.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Trajectory stitching candidate — {a.scene_key}: ep {a.episode_id} ↔ ep {b.episode_id}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--episode-a", default="94")
    parser.add_argument("--episode-b", default="1114")
    parser.add_argument("--match-dist", type=float, default=0.5,
                        help="Two paths count as overlapping where they come within this "
                             "many metres (coordinate distance, not grid cells).")
    args = parser.parse_args()

    paths = load_r2r_paths(args.split)
    a = _find(paths, args.episode_a)
    b = _find(paths, args.episode_b)
    if a.scene_id != b.scene_id:
        raise SystemExit("the two episodes are in different scenes; cannot stitch")

    info = build_stitch(a, b, args.match_dist)

    out_dir = RESULT_ROOT / args.split / "figures"
    fig_path = out_dir / f"stitch_{a.scene_key}_{a.episode_id}_{b.episode_id}.png"
    draw(a, b, info, fig_path, args.match_dist)

    meta = {
        "split": args.split,
        "scene_key": a.scene_key,
        "episode_a": a.episode_id,
        "episode_b": b.episode_id,
        "trajectory_a": a.trajectory_id,
        "trajectory_b": b.trajectory_id,
        "match_dist_m": args.match_dist,
        "a_shared_run_idx": list(info["a_run"]),
        "b_shared_run_idx": list(info["b_run"]),
        "same_direction": info["same_dir"],
        "merge_point_xz": [float(info["merge_pt"][0]), float(info["merge_pt"][2])],
        "branch_point_xz": [float(info["branch_pt"][0]), float(info["branch_pt"][2])],
        "stitched_len_points": int(len(info["stitched"])),
        "instruction_a": a.instruction_text,
        "instruction_b": b.instruction_text,
        "figure": str(fig_path.relative_to(Path(__file__).resolve().parents[2])),
    }
    meta_path = out_dir / f"stitch_{a.scene_key}_{a.episode_id}_{b.episode_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(f"scene {a.scene_key}: ep {a.episode_id} (traj {a.trajectory_id}) ↔ "
          f"ep {b.episode_id} (traj {b.trajectory_id})")
    print(f"  shared run A[{info['a_run'][0]}:{info['a_run'][1]}] / "
          f"B[{info['b_run'][0]}:{info['b_run'][1]}], same_dir={info['same_dir']}")
    print(f"  merge {meta['merge_point_xz']} -> branch {meta['branch_point_xz']}")
    print(f"  wrote figure: {fig_path}")
    print(f"  wrote meta:   {meta_path}")


if __name__ == "__main__":
    main()
