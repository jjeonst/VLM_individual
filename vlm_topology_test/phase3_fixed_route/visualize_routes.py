"""Draw each fixed route on the building's floor plan and report how it was scored.

The tables in the results write-up say how often each route succeeded but not what the route
looked like, and several of the findings only make sense once the geometry is visible. This
module renders one picture per route showing, on the navigable floor area of the building:

- **the starting poses**, one per demonstration, spread by the small nudge applied at
  collection time,
- **the demonstrated route**, that is the path the shortest-path expert actually walked,
- **the goal viewpoints**, the set of standing positions from which the target object is
  visible, which is what both the expert and the success test aim at,
- **where the expert came to rest**, which is the detail that decides whether a route can be
  scored as a success at all.

That last point is the reason this exists. Success requires stopping within 0.1 m of a goal
viewpoint, but the agent moves in 0.25 m steps and turns in 30° increments, so the positions
it can occupy form a lattice while the viewpoints do not sit on one. On some routes the
nearest reachable position happens to fall inside the 0.1 m circle and every demonstration
succeeds; on others it falls just outside and none do. Seeing the resting points next to the
viewpoints makes that visible in a way the numbers alone do not.

Positions are read from the stored demonstrations rather than re-simulated, so the drawn path
is the one the policies were actually trained on. The simulator is opened only to obtain the
floor plan and the goal viewpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nav_baseline.env import geodesic_to_goals, open_scene  # noqa: E402
from nav_baseline.visualize_detour import (  # noqa: E402
    METERS_PER_PIXEL, path_length, to_pixels, topdown_map)
from vlm_topology_test.phase3_fixed_route.collect import episode_for  # noqa: E402

RESULT_DIR = Path(__file__).resolve().parents[0] / "results"
OUT_DIR = RESULT_DIR / "route_maps"
RULES = ("random", "matched", "junction")


def load_demo_positions(data_root: Path, rule: str, scene: str) -> list[np.ndarray]:
    """Every demonstration's walked path for one route, as arrays of world positions."""
    shard = data_root / f"demos_{rule}" / f"{scene}.npz"
    if not shard.exists():
        return []
    payload = np.load(shard)
    keys = sorted({name.split("|")[0] for name in payload.files})
    return [payload[f"{key}|position"] for key in keys if f"{key}|position" in payload.files]


def route_scores(rule: str, scene: str, rollout_dir: Path) -> dict:
    """Per-route success counts for both representations, read from the rollout results."""
    out = {}
    for representation, label in (("oracle", "oracle"), ("vlm_baseline", "vlm")):
        path = rollout_dir / f"{rule}_{representation}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        runs = [r for r in payload["runs"]["true"] if r["scene"] == scene]
        if not runs:
            continue
        for threshold in (0.1, 0.25):
            hits = sum(1 for r in runs
                       if r["stopped"] and r["final_geodesic"] is not None
                       and r["final_geodesic"] <= threshold)
            out[f"{label}@{threshold}"] = (hits, len(runs))
    return out


def draw_route(view, bounds, paths, goal_positions, info, out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 7.5))
    axis.imshow(view, cmap="gray_r", origin="upper", alpha=0.35)

    if goal_positions:
        columns, rows = to_pixels(np.stack(goal_positions), bounds)
        axis.plot(columns, rows, ".", color="#ff9f1c", markersize=2.5, alpha=0.55,
                  label=f"goal viewpoints ({len(goal_positions)})")

    for index, positions in enumerate(paths):
        columns, rows = to_pixels(positions, bounds)
        axis.plot(columns, rows, "-", color="#1f77b4", linewidth=1.0, alpha=0.35,
                  label="demonstrated route" if index == 0 else None)

    starts = np.stack([p[0] for p in paths])
    ends = np.stack([p[-1] for p in paths])
    sc, sr = to_pixels(starts, bounds)
    ec, er = to_pixels(ends, bounds)
    axis.plot(sc, sr, "o", color="#2a9d8f", markersize=6, label="start (jittered)")
    axis.plot(ec, er, "s", color="#d62728", markersize=6, label="expert stopped here")

    pad = 40
    xs = np.concatenate([sc, ec]); ys = np.concatenate([sr, er])
    axis.set_xlim(max(xs.min() - pad, 0), min(xs.max() + pad, view.shape[1]))
    axis.set_ylim(min(ys.max() + pad, view.shape[0]), max(ys.min() - pad, 0))

    axis.set_title(info["title"], fontsize=10)
    axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axis.set_xticks([]); axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/topovlm/phase3_fixed_route",
                        help="where demos_<rule> live")
    parser.add_argument("--results", default=None,
                        help="directory holding routes_<rule>.json (default: results/)")
    parser.add_argument("--rollout", default=None,
                        help="directory holding <rule>_<representation>.json rollout results")
    parser.add_argument("--out", default=None)
    parser.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    result_dir = Path(args.results) if args.results else RESULT_DIR
    rollout_dir = Path(args.rollout) if args.rollout else result_dir / "rollout"
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for rule in args.rules:
        routes = json.loads((result_dir / f"routes_{rule}.json").read_text())["routes"]
        demo_index = json.loads((data_root / f"demos_{rule}" / "index.json").read_text())
        reached = {}
        for record in demo_index["records"]:
            hit, total = reached.get(record["scene"], (0, 0))
            reached[record["scene"]] = (hit + int(record["reached"]), total + 1)

        for route in routes:
            scene = route["scene"]
            paths = load_demo_positions(data_root, rule, scene)
            if not paths:
                print(f"  [skip] {rule}/{scene}: no stored demonstrations", flush=True)
                continue
            episode = episode_for(route)
            sim = open_scene(scene, with_rgb=False)
            try:
                view, bounds = topdown_map(sim.pathfinder, float(paths[0][0][1]))
                finals = [geodesic_to_goals(sim.pathfinder, p[-1], episode.goal_positions)
                          for p in paths]
                start_distance = float(np.median(
                    [geodesic_to_goals(sim.pathfinder, p[0], episode.goal_positions)
                     for p in paths]))
            finally:
                sim.close()

            hit, total = reached.get(scene, (0, 0))
            scores = route_scores(rule, scene, rollout_dir)
            lengths = [path_length(p) for p in paths]
            final_median = float(np.median(finals))
            record = {
                "rule": rule, "scene": scene, "object": route["object"],
                "episode_id": route["episode_id"],
                "viewpoints": len(episode.goal_positions),
                "start_geodesic_median_m": round(start_distance, 2),
                "walked_median_m": round(float(np.median(lengths)), 2),
                "steps_median": int(np.median([len(p) for p in paths])),
                "expert_final_median_m": round(final_median, 3),
                "expert_final_min_m": round(float(np.min(finals)), 3),
                "expert_final_max_m": round(float(np.max(finals)), 3),
                "expert_reached": [hit, total],
                "scores": {k: list(v) for k, v in scores.items()},
                "image": f"results/route_maps/{rule}_{scene}.png",
            }
            index.append(record)

            def fmt(key):
                if key not in scores:
                    return "n/a"
                a, b = scores[key]
                return f"{a}/{b}"

            title = (f"{rule} · {scene} · goal = {route['object']}\n"
                     f"expert stopped at {final_median:.3f} m "
                     f"(reached 0.1 m in {hit}/{total} demos)\n"
                     f"policy @0.1 m  oracle {fmt('oracle@0.1')}  vlm {fmt('vlm@0.1')}"
                     f"   |   @0.25 m  oracle {fmt('oracle@0.25')}  vlm {fmt('vlm@0.25')}")
            draw_route(view, bounds, paths, episode.goal_positions,
                       {"title": title}, out_dir / f"{rule}_{scene}.png")
            print(f"  {rule}/{scene} ({route['object']}): expert final {final_median:.3f} m, "
                  f"reached {hit}/{total}", flush=True)

    (result_dir / "route_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {len(index)} route maps to {out_dir}")
    print(f"wrote {result_dir / 'route_index.json'}")


if __name__ == "__main__":
    main()
