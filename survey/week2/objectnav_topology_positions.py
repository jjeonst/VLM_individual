"""Week 2 · Step 5-7 (positions) — Spatial branch candidates from replayed
ObjectNav positions: doorway / bottleneck / wrong-turn(dead-end), plus a
top-down figure.

Consumes the per-scene position files produced by
``objectnav_replay_positions.py`` (results/objectnav_branching/positions/<split>/).
These are the candidate types that CANNOT be found from actions alone — they are
inherently spatial — so here we use the recovered (x, z) trajectories.

Detections:
  * Bottleneck / doorway : quantise each scene's floor plane into ``cell`` metre
    cells; a cell's "traffic" = number of DISTINCT trajectories that pass through
    it. Cells where a large fraction of the scene's routes funnel through are
    bottlenecks (a doorway is the narrow, high-traffic case).
  * Wrong-turn / dead-end: a trajectory that leaves a location and later returns
    near it after travelling a long detour in between = a there-and-back backtrack
    out of a dead-end. (This is exactly what actions alone could not reveal.)

Outputs (results/objectnav_branching/<split>/):
  - spatial_branch.json          : stats + top bottleneck cells + dead-end examples
  - spatial_branch_candidates.md : doorway/bottleneck/wrong-turn table (real numbers)
  - objectnav_topdown.png        : a scene top-down: trajectory overlay + bottleneck
                                   heat + one highlighted dead-end trajectory

Usage:
  python -m survey.week2.objectnav_topology_positions --split train
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "survey" / "week2" / "results" / "objectnav_branching"
POS_ROOT = RESULT_ROOT / "positions"

CELL_M = 0.5           # floor-plane grid for traffic counting
REVISIT_RADIUS_M = 0.75  # "returned near" threshold
REVISIT_MIN_GAP = 12     # steps between leaving and returning
REVISIT_MIN_DETOUR_M = 2.5  # path length travelled away in between


def load_scene_positions(split: str) -> dict[str, dict[str, np.ndarray]]:
    """scene_key -> {episode_id: (T,3) positions}."""
    pos_dir = POS_ROOT / split
    scenes: dict[str, dict[str, np.ndarray]] = {}
    for npz in sorted(pos_dir.glob("*.npz")):
        data = np.load(npz)
        scenes[npz.stem] = {k: data[k] for k in data.files}
    return scenes


def load_index(split: str) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    path = POS_ROOT / split / "index.jsonl"
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            idx[row["episode_id"]] = row
    return idx


def _cell(x: float, z: float) -> tuple[int, int]:
    return (int(math.floor(x / CELL_M)), int(math.floor(z / CELL_M)))


def scene_bottlenecks(trajs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Cell traffic = # distinct trajectories through each cell."""
    cell_trajs: dict[tuple[int, int], set[str]] = defaultdict(set)
    for ep, pos in trajs.items():
        seen = {_cell(float(p[0]), float(p[2])) for p in pos}
        for c in seen:
            cell_trajs[c].add(ep)
    n = len(trajs)
    ranked = sorted(cell_trajs.items(), key=lambda kv: len(kv[1]), reverse=True)
    top = [
        {
            "cell": list(c),
            "xz": [round((c[0] + 0.5) * CELL_M, 2), round((c[1] + 0.5) * CELL_M, 2)],
            "traffic": len(eps),
            "fraction": round(len(eps) / n, 3),
        }
        for c, eps in ranked[:10]
    ]
    return {"num_trajectories": n, "top_cells": top}


def detect_deadend(pos: np.ndarray) -> dict[str, Any] | None:
    """There-and-back backtrack: return near an earlier point after a long detour."""
    xz = pos[:, [0, 2]]
    # cumulative path length for detour measurement
    seg = np.sqrt((np.diff(xz, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    best = None
    for i in range(len(xz)):
        # look for a later point j that returns near i
        for j in range(i + REVISIT_MIN_GAP, len(xz)):
            if abs(cum[j] - cum[i]) < REVISIT_MIN_DETOUR_M:
                continue
            if float(np.hypot(*(xz[j] - xz[i]))) <= REVISIT_RADIUS_M:
                detour = float(cum[j] - cum[i])
                straight = float(np.hypot(*(xz[j] - xz[i])))
                if best is None or detour > best["detour_m"]:
                    best = {
                        "return_from": i,
                        "return_to": j,
                        "detour_m": round(detour, 2),
                        "gap_closed_m": round(straight, 2),
                    }
                break
    return best


def analyse(scenes: dict[str, dict[str, np.ndarray]], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scene_stats = {}
    deadend_examples = []
    deadend_count = total_traj = 0
    for sc, trajs in scenes.items():
        total_traj += len(trajs)
        bn = scene_bottlenecks(trajs)
        scene_stats[sc] = bn
        for ep, pos in trajs.items():
            de = detect_deadend(pos)
            if de is not None:
                deadend_count += 1
                de.update({"episode_id": ep, "scene_key": sc,
                           "object": index.get(ep, {}).get("object_category"),
                           "length": int(len(pos))})
                deadend_examples.append(de)

    # scene with the strongest bottleneck (highest top-cell fraction, enough routes)
    def scene_bottleneck_score(sc):
        st = scene_stats[sc]
        if st["num_trajectories"] < 10 or not st["top_cells"]:
            return -1.0
        return st["top_cells"][0]["fraction"]

    best_scene = max(scene_stats, key=scene_bottleneck_score)
    deadend_examples.sort(key=lambda d: d["detour_m"], reverse=True)
    return {
        "num_scenes": len(scenes),
        "total_trajectories": total_traj,
        "deadend_episodes": deadend_count,
        "deadend_pct": round(100 * deadend_count / max(total_traj, 1), 1),
        "best_bottleneck_scene": best_scene,
        "best_scene_top_cells": scene_stats[best_scene]["top_cells"][:5],
        "best_scene_num_trajectories": scene_stats[best_scene]["num_trajectories"],
        "deadend_examples": deadend_examples[:15],
        "scene_stats": scene_stats,
    }


def draw(scenes, index, stats, out_path: Path) -> None:
    sc = stats["best_bottleneck_scene"]
    trajs = scenes[sc]
    top_cells = stats["best_scene_top_cells"]

    # pick a dead-end example in this scene if any, else the global top one
    de_here = [d for d in stats["deadend_examples"] if d["scene_key"] == sc]
    de = de_here[0] if de_here else (stats["deadend_examples"][0] if stats["deadend_examples"] else None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    for ax in (ax1, ax2):
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, lw=0.3, alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")

    # Left: all trajectories + bottleneck cells
    for ep, pos in trajs.items():
        ax1.plot(pos[:, 0], pos[:, 2], "-", color="#4c78a8", lw=0.6, alpha=0.25)
    for rank, c in enumerate(top_cells):
        x, z = c["xz"]
        size = 120 + 500 * c["fraction"]
        ax1.scatter([x], [z], s=size, color="#d62728", alpha=0.5,
                    edgecolor="black", zorder=4)
        if rank == 0:
            ax1.annotate(f"bottleneck\n{c['traffic']}/{stats['best_scene_num_trajectories']} routes "
                         f"({c['fraction']:.0%})", (x, z), textcoords="offset points",
                         xytext=(8, 8), fontsize=9, color="#d62728", fontweight="bold")
    ax1.set_title(f"A. Doorway / bottleneck — scene {sc}\n"
                  f"{stats['best_scene_num_trajectories']} expert routes; red = cells most routes funnel through")

    # Right: one dead-end trajectory highlighted
    if de is not None:
        dpos = scenes[de["scene_key"]][de["episode_id"]]
        # faint context: other trajectories of that scene
        for ep, pos in scenes[de["scene_key"]].items():
            ax2.plot(pos[:, 0], pos[:, 2], "-", color="#bbb", lw=0.5, alpha=0.3)
        i, j = de["return_from"], de["return_to"]
        # highlight the there-and-back excursion segment [i..j]
        ax2.plot(dpos[:, 0], dpos[:, 2], "-", color="#f2a29f", lw=1.4, alpha=0.7)
        ax2.plot(dpos[i:j + 1, 0], dpos[i:j + 1, 2], "-", color="#e45756", lw=2.6,
                 label="dead-end excursion")
        # farthest point of the excursion from the return point
        far = i + int(np.argmax(np.hypot(*(dpos[i:j + 1, [0, 2]] - dpos[i, [0, 2]]).T)))
        ax2.scatter(*dpos[0, [0, 2]], color="#54a24b", s=110, marker="o",
                    edgecolor="black", zorder=5, label="start")
        ax2.scatter(*dpos[-1, [0, 2]], color="#54a24b", s=240, marker="*",
                    edgecolor="black", zorder=5, label="STOP")
        ax2.scatter(*dpos[i, [0, 2]], color="black", s=70, zorder=6)
        ax2.scatter(*dpos[far, [0, 2]], color="black", s=70, zorder=6)
        ax2.annotate("went into dead-end", dpos[far, [0, 2]],
                     textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold")
        ax2.annotate(f"returned near here\ndetour {de['detour_m']} m → gap {de['gap_closed_m']} m",
                     dpos[i, [0, 2]], textcoords="offset points", xytext=(-8, -30),
                     ha="right", fontsize=9, color="#e45756", fontweight="bold")
        ax2.set_title(f"B. Wrong-turn / dead-end — scene {de['scene_key']}\n"
                      f"ep …{de['episode_id'][-6:]} (target {de['object']}): "
                      f"there-and-back backtrack")
        ax2.legend(loc="best", fontsize=8)
    else:
        ax2.set_title("B. Wrong-turn / dead-end — none detected")

    fig.suptitle("HM3D ObjectNav — spatial branch candidates from replayed positions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_table(stats: dict[str, Any], path: Path) -> None:
    sc = stats["best_bottleneck_scene"]
    top = stats["best_scene_top_cells"][0] if stats["best_scene_top_cells"] else {"fraction": 0, "traffic": 0, "xz": [0, 0]}
    de_n = stats["deadend_episodes"]
    de_pct = stats["deadend_pct"]
    de_ex = stats["deadend_examples"][0] if stats["deadend_examples"] else None
    de_line = (f"top: scene `{de_ex['scene_key']}`, ep …{de_ex['episode_id'][-6:]}, "
               f"detour {de_ex['detour_m']} m returning within {de_ex['gap_closed_m']} m"
               if de_ex else "none detected")
    md = f"""# ObjectNav spatial branch candidates (deliverable #3, position-based)

Source: expert positions recovered by replaying HM3D ObjectNav actions in the
Habitat simulator (`objectnav_replay_positions.py`), {stats['total_trajectories']}
trajectories over {stats['num_scenes']} scenes. Floor plane (x, z), cell {CELL_M} m.

These are the **spatial** candidates that action sequences alone could not give.

| Candidate | Observation cue | Action choices | Why topology? | Valid / invalid | How to visualize | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| **Doorway / bottleneck** | RGB of a narrow passage many routes use | pass through vs turn away | many distinct routes funnel through one narrow cell — the connectivity chokepoint | valid = pass the connector toward goal; invalid = miss it | trajectory overlay + traffic heat (Panel A) | scene `{sc}`: busiest cell carries **{top['traffic']}/{stats['best_scene_num_trajectories']} routes ({top['fraction']:.0%})** at xz={top['xz']} |
| **Wrong-turn / dead-end** | RGB shows a dead-end / blocked region | go in vs avoid | entering forces a there-and-back detour, changing path length & cost | valid = avoid; invalid = enter, must return | highlighted route with return point (Panel B) | **{de_n} routes ({de_pct}%)** leave then return within {REVISIT_RADIUS_M} m after a ≥{REVISIT_MIN_DETOUR_M} m detour; {de_line} |
| **Shared doorway across routes** | same connector seen from different approaches | which side to enter/exit | different start rooms reuse the same doorway (shared subpath, like R2R) | valid = correct connector; invalid = wrong room exit | overlay showing convergence (Panel A) | high-traffic cells = shared connectors across {stats['best_scene_num_trajectories']} routes in `{sc}` |

## Method notes
- Bottleneck traffic = # of **distinct** trajectories whose path enters a given {CELL_M} m cell; a doorway is the narrow, high-traffic case.
- Dead-end = a route returns within {REVISIT_RADIUS_M} m of an earlier point after travelling ≥{REVISIT_MIN_DETOUR_M} m away (≥{REVISIT_MIN_GAP} steps apart) — a genuine there-and-back, which the action-only view could not detect (all ≥180° spins were at spawn).
- Figure: `results/objectnav_branching/{{split}}/objectnav_topdown.png`
"""
    path.write_text(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    scenes = load_scene_positions(args.split)
    if not scenes:
        raise SystemExit(f"no positions found under {POS_ROOT / args.split} — run "
                         f"objectnav_replay_positions.py first")
    index = load_index(args.split)
    print(f"loaded {sum(len(v) for v in scenes.values())} trajectories over {len(scenes)} scenes")

    stats = analyse(scenes, index)
    print(f"  bottleneck scene: {stats['best_bottleneck_scene']} "
          f"(busiest cell {stats['best_scene_top_cells'][0]['fraction']:.0%} of routes)")
    print(f"  dead-end routes: {stats['deadend_episodes']} ({stats['deadend_pct']}%)")

    out_dir = RESULT_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    # drop bulky per-scene stats from the saved json
    save = {k: v for k, v in stats.items() if k != "scene_stats"}
    (out_dir / "spatial_branch.json").write_text(json.dumps(save, indent=2, ensure_ascii=False) + "\n")
    write_table(stats, out_dir / "spatial_branch_candidates.md")
    draw(scenes, index, stats, out_dir / "objectnav_topdown.png")
    print(f"  wrote spatial_branch.json / spatial_branch_candidates.md / objectnav_topdown.png")


if __name__ == "__main__":
    main()
