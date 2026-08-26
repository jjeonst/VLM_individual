"""Collect demonstrations for the fixed routes chosen by ``routes.py``.

Each building contributes one route, and that route is demonstrated many times with the
starting pose **slightly perturbed** each time. The perturbation matters for the same reason
the observation ablation does: with a perfectly fixed start the policy could reach the goal
by replaying a memorised action sequence without looking at anything. Small changes in where
the agent begins make that shortcut fail while leaving the route itself unchanged.

Every step records what the agent saw and what the expert would do from that state:

- the camera image, kept so the vision-language model can encode it later,
- the **oracle features**, which state the distance and direction to the nearest goal
  viewpoint and serve as the comparison representation,
- the expert's action, which is the label.

The expert stops within ``--goal-radius`` of a goal viewpoint. This experiment uses the
standard 0.1 m, which is stricter than the 1.0 m used previously, so the script reports how
often the expert actually achieves it. If the expert cannot reach 0.1 m the target is
unattainable for any policy trained on its demonstrations, and that has to be visible before
anything is trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nav_baseline.env import (  # noqa: E402
    ACTION_NAME, Episode, build_expert, expert_action, geodesic_to_goals, goal_vector,
    load_episodes, open_scene, reset_to)

RESULT_DIR = Path(__file__).resolve().parents[0] / "results"
DATA_ROOT = Path("/data/topovlm/phase3_fixed_route")


def load_routes(rule: str) -> list[dict]:
    payload = json.loads((RESULT_DIR / f"routes_{rule}.json").read_text())
    return payload["routes"]


def episode_for(route: dict) -> Episode:
    """Rebuild the exact problem a route refers to, keeping its goal viewpoints."""
    matches = [e for e in load_episodes(route["scene"], object_category=route["object"])
               if e.episode_id == route["episode_id"]]
    if not matches:
        raise ValueError(f"episode {route['episode_id']} not found in {route['scene']}")
    return matches[0]


def jitter_start(episode: Episode, rng, position_m: float, rotation_deg: float,
                 pathfinder) -> Episode:
    """A copy of the problem whose start is nudged slightly, staying on navigable ground."""
    if position_m <= 0 and rotation_deg <= 0:
        return episode
    from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs, quat_from_angle_axis

    for _ in range(20):
        offset = rng.normal(0.0, position_m / 2.0, size=3).astype(np.float32)
        offset[1] = 0.0
        candidate = episode.start_position + offset
        if pathfinder.is_navigable(candidate):
            break
    else:
        candidate = episode.start_position

    angle = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg))
    turn = quat_from_angle_axis(float(angle), np.array([0.0, 1.0, 0.0]))
    rotated = turn * quat_from_coeffs(episode.start_rotation)
    return Episode(episode_id=episode.episode_id, scene_key=episode.scene_key,
                   object_category=episode.object_category,
                   start_position=candidate.astype(np.float32),
                   start_rotation=np.asarray(quat_to_coeffs(rotated), dtype=np.float32),
                   goal_positions=episode.goal_positions)


def demonstrate(sim, expert, episode, *, max_steps: int, goal_radius: float):
    """Follow the expert once, recording observations, oracle features and labels."""
    observations = reset_to(sim, episode)
    start = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    if not np.isfinite(start):
        return None, {"skipped": "unreachable"}

    rgb, oracle, labels, positions = [], [], [], []
    forward_steps = 0
    for _ in range(max_steps):
        action = expert_action(expert, sim, episode)
        rgb.append(np.asarray(observations["rgb"])[..., :3].astype(np.uint8))
        oracle.append(goal_vector(sim, episode))
        positions.append(np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32))
        labels.append(action)
        if action == 0:
            break
        observations = sim.step(ACTION_NAME[action])
        if action == 1:
            forward_steps += 1

    final = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    samples = {"rgb": np.stack(rgb, axis=0),
               "oracle": np.stack(oracle, axis=0).astype(np.float32),
               "label": np.asarray(labels, dtype=np.int64),
               "position": np.stack(positions, axis=0)}
    info = {"steps": len(labels), "start_geodesic": round(float(start), 3),
            "final_geodesic": round(float(final), 3),
            "walked_m": round(forward_steps * 0.25, 2),
            "reached": bool(labels[-1] == 0 and final <= goal_radius)}
    return samples, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", required=True, choices=["random", "matched", "junction"])
    parser.add_argument("--repeats", type=int, default=20,
                        help="demonstrations per route, each with a different start nudge")
    parser.add_argument("--jitter-m", type=float, default=0.3)
    parser.add_argument("--jitter-deg", type=float, default=15.0)
    parser.add_argument("--goal-radius", type=float, default=0.1,
                        help="how close the expert must get before it stops")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = DATA_ROOT / f"demos_{args.rule}"
    out_dir.mkdir(parents=True, exist_ok=True)
    routes = load_routes(args.rule)
    print(f"[collect] rule={args.rule}: {len(routes)} routes x {args.repeats} demonstrations, "
          f"goal radius {args.goal_radius} m", flush=True)

    index, reached_count, total = [], 0, 0
    for route in routes:
        episode = episode_for(route)
        sim = open_scene(route["scene"], with_rgb=True)
        expert = build_expert(sim, goal_radius=args.goal_radius)
        stored = {}
        try:
            for repeat in range(args.repeats):
                nudged = jitter_start(episode, rng, args.jitter_m, args.jitter_deg,
                                      sim.pathfinder)
                samples, info = demonstrate(sim, expert, nudged, max_steps=args.max_steps,
                                            goal_radius=args.goal_radius)
                if samples is None:
                    continue
                key = f"{route['scene']}_{route['object']}_{route['episode_id']}_r{repeat}"
                for field, value in samples.items():
                    stored[f"{key}|{field}"] = value
                index.append({"key": key, "scene": route["scene"], "object": route["object"],
                              "episode_id": route["episode_id"], "repeat": repeat, **info})
                total += 1
                reached_count += int(info["reached"])
        finally:
            sim.close()
        if stored:
            np.savez_compressed(out_dir / f"{route['scene']}.npz", **stored)
        recent = [r for r in index if r["scene"] == route["scene"]]
        print(f"  {route['scene']} ({route['object']}): {len(recent)} demos, "
              f"expert reached {args.goal_radius}m in "
              f"{sum(r['reached'] for r in recent)}/{len(recent)}", flush=True)

    steps = [r["steps"] for r in index]
    summary = {"rule": args.rule, "routes": len(routes), "demonstrations": total,
               "goal_radius_m": args.goal_radius,
               "jitter_m": args.jitter_m, "jitter_deg": args.jitter_deg,
               "expert_reached_rate": round(reached_count / max(total, 1), 3),
               "median_steps": int(np.median(steps)) if steps else 0,
               "max_steps_seen": int(max(steps)) if steps else 0,
               "records": index}
    (out_dir / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[collect] {total} demonstrations -> {out_dir}")
    print(f"[collect] expert reached {args.goal_radius} m in "
          f"{summary['expert_reached_rate'] * 100:.0f}% of them; "
          f"median {summary['median_steps']} steps, longest {summary['max_steps_seen']}")
    if summary["expert_reached_rate"] < 0.9:
        print("[collect] WARNING: the expert itself often fails to reach the goal radius, "
              "so no policy trained on these demonstrations can be expected to.")


if __name__ == "__main__":
    main()
