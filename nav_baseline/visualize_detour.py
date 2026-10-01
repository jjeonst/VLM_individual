"""Draw the shortest-path route and the detour route for the same episode, side by side.

The detour data used in the experiment is only meaningful if the detour really is a
different route through the building rather than a small wobble around the optimal path.
This script makes that visible: for a handful of episodes it drives the same start and goal
twice, once with the shortest-path expert and once with the detour generator, and plots both
trajectories on top of the scene's navigable-area map.

Each figure also reports two numbers that quantify the difference:

- **path length ratio**: how much longer the walked route is than the shortest route,
- **label agreement**: the fraction of steps where the executed action equalled the stored
  expert label. Shortest-path runs sit at 1.0 by construction; detour runs must be lower,
  which is direct evidence that the recorded states are off the optimal path while their
  labels still point at the goal.

Output: one PNG per episode plus a summary JSON under ``nav_baseline/results/detour_vis/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nav_baseline.collect import (ExpertDriver, PerturbationDriver, WaypointDetourDriver,
                                  collect_episode)
from nav_baseline.env import (FORWARD_M, build_expert, geodesic_to_goals, load_balanced_episodes,
                              load_episodes, make_sim, reset_to, scene_glb_path,
                              scene_keys_with_episodes)

OUT_DIR = Path(__file__).resolve().parents[0] / "results" / "detour_vis"
METERS_PER_PIXEL = 0.05


def topdown_map(pathfinder, height: float):
    """Boolean navigable-area map of the scene at the agent's height, plus its world bounds."""
    view = np.asarray(pathfinder.get_topdown_view(METERS_PER_PIXEL, height))
    bounds = pathfinder.get_bounds()
    return view, bounds


def to_pixels(positions: np.ndarray, bounds) -> tuple[np.ndarray, np.ndarray]:
    """Convert world (x, y, z) coordinates to column/row indices of the top-down map."""
    columns = (positions[:, 0] - bounds[0][0]) / METERS_PER_PIXEL
    rows = (positions[:, 2] - bounds[0][2]) / METERS_PER_PIXEL
    return columns, rows


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return float(steps.sum())


def draw(view, bounds, runs: dict, episode, out_path: Path, goal_positions):
    figure, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6))
    if len(runs) == 1:
        axes = [axes]
    colours = {"shortest": "#1f77b4", "detour": "#d62728"}
    for axis, (name, run) in zip(axes, runs.items()):
        axis.imshow(view, cmap="gray_r", origin="upper", alpha=0.35)
        positions = run["positions"]
        columns, rows = to_pixels(positions, bounds)
        axis.plot(columns, rows, "-", color=colours.get(name, "k"), linewidth=2,
                  label=f"{name} route")
        axis.plot(columns[0], rows[0], "o", color="green", markersize=10, label="start")
        axis.plot(columns[-1], rows[-1], "s", color="black", markersize=9, label="end")
        if goal_positions:
            goal_columns, goal_rows = to_pixels(np.stack(goal_positions), bounds)
            axis.plot(goal_columns, goal_rows, ".", color="orange", markersize=2,
                      alpha=0.5, label="goal viewpoints")
        waypoint = run.get("waypoint")
        if waypoint is not None:
            wc, wr = to_pixels(waypoint[None, :], bounds)
            axis.plot(wc, wr, "*", color="purple", markersize=18, label="detour waypoint")
        axis.set_title(f"{name}\nlength {run['path_length']:.1f} m "
                       f"(ratio {run['length_ratio']:.2f}), "
                       f"label agreement {run['label_agreement']:.2f}")
        axis.legend(loc="upper right", fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(f"{episode.scene_key} · episode {episode.episode_id} · "
                    f"goal = {episode.object_category}")
    figure.tight_layout()
    figure.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--episodes-per-scene", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--perturbation-ratio", type=float, default=0.25)
    parser.add_argument("--object", default=None,
                        help="only use episodes whose target is this object "
                             "(chair, sofa, bed, toilet, tv_monitor, plant)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = []

    for scene_key in scene_keys_with_episodes()[: args.scenes]:
        episodes = (load_episodes(scene_key, limit=args.episodes_per_scene,
                                  object_category=args.object) if args.object
                    else load_balanced_episodes(scene_key, limit=args.episodes_per_scene))
        if not episodes:
            continue
        sim = make_sim(scene_glb_path(scene_key))
        expert = build_expert(sim)
        try:
            view, bounds = topdown_map(sim.pathfinder, float(episodes[0].start_position[1]))
            for episode in episodes:
                if not episode.goal_positions:
                    continue
                reset_to(sim, episode)
                shortest_geodesic = geodesic_to_goals(
                    sim.pathfinder, sim.get_agent(0).get_state().position,
                    episode.goal_positions)
                if not np.isfinite(shortest_geodesic) or shortest_geodesic < 1.0:
                    continue

                runs = {}
                for name in ("shortest", "detour"):
                    reset_to(sim, episode)
                    if name == "shortest":
                        driver = ExpertDriver()
                    elif rng.random() < args.perturbation_ratio:
                        driver = PerturbationDriver(rng)
                    else:
                        driver = WaypointDetourDriver(expert, sim, episode, rng)
                    samples, info = collect_episode(sim, expert, episode, driver,
                                                    max_steps=args.max_steps, min_steps=4,
                                                    save_rgb=False)
                    if samples is None:
                        break
                    walked = path_length(samples["position"])
                    runs[name] = {
                        "positions": samples["position"],
                        "path_length": walked,
                        "length_ratio": walked / max(shortest_geodesic, 1e-6),
                        "label_agreement": info["label_matches_executed"],
                        "driver": info["driver"],
                        "steps": info["steps"],
                        "waypoint": getattr(driver, "waypoint", None),
                    }
                if len(runs) != 2:
                    continue

                name = f"{scene_key}_{episode.object_category}_ep{episode.episode_id}.png"
                draw(view, bounds, runs, episode, OUT_DIR / name, episode.goal_positions)
                summary.append({
                    "figure": name, "scene": scene_key, "episode": episode.episode_id,
                    "object": episode.object_category,
                    "shortest_geodesic_m": round(float(shortest_geodesic), 2),
                    **{f"{k}_{field}": (round(float(v[field]), 3)
                                        if isinstance(v[field], float) else v[field])
                       for k, v in runs.items()
                       for field in ("path_length", "length_ratio", "label_agreement",
                                     "driver", "steps")},
                })
                print(f"  {name}: shortest {runs['shortest']['path_length']:.1f} m "
                      f"(agree {runs['shortest']['label_agreement']:.2f}) vs "
                      f"detour {runs['detour']['path_length']:.1f} m "
                      f"(ratio {runs['detour']['length_ratio']:.2f}, "
                      f"agree {runs['detour']['label_agreement']:.2f}, "
                      f"{runs['detour']['driver']})", flush=True)
        finally:
            sim.close()

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary:
        detour_ratio = float(np.mean([s["detour_length_ratio"] for s in summary]))
        detour_agreement = float(np.mean([s["detour_label_agreement"] for s in summary]))
        shortest_agreement = float(np.mean([s["shortest_label_agreement"] for s in summary]))
        print(f"\n[detour] {len(summary)} episodes")
        print(f"[detour] mean detour length ratio      = {detour_ratio:.2f} (1.0 = shortest)")
        print(f"[detour] mean label agreement, detour  = {detour_agreement:.2f}")
        print(f"[detour] mean label agreement, shortest= {shortest_agreement:.2f}")
    print(f"\nwrote figures and summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
