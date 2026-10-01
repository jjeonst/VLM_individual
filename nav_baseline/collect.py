"""Trajectory collection for the nav_baseline navigator.

Every collection mode shares one recording loop, and the loop enforces a single rule:

    the action that is EXECUTED may be anything, but the action that is STORED as the label
    is always the expert's optimal action toward the FINAL goal from the visited state.

That separation is what makes detour and recovery data useful. Behavior cloning copies the
labels, so labelling a detour with the detour's own actions would teach the policy to take
detours. Labelling it with the optimal action instead teaches the policy how to get back on
route from places the shortest path never visits, which is exactly the information that
shortest-path-only data lacks.

Modes:

- ``expert``  : the shortest-path expert drives. This is the standard behavior cloning
  corpus and contains only states that lie on an optimal path.
- ``detour``  : the agent is driven along deliberately suboptimal routes while the expert
  keeps labelling. Two generators are mixed. The waypoint generator sends the agent through
  a random intermediate navigable point before heading to the goal, which produces
  genuinely different routes through the building. The perturbation generator injects short
  bursts of random actions into an otherwise optimal run, which produces the small
  off-course states a policy meets when it makes a minor mistake.
- ``dagger``  : a trained policy drives while the expert labels (Ross et al., 2011).

Each stored sample holds the depth observation (quantised to uint8), the goal-relative
vector, the previous action, and the expert label. RGB frames can be stored alongside with
``--save-rgb`` so that the same trajectories can later be encoded by a vision-language model
without collecting them again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nav_baseline.env import (
    ACTION_NAME, NUM_ACTIONS, SUCCESS_DISTANCE_M, build_expert, depth_observation,
    expert_action, expert_action_to, geodesic_to_goals, goal_vector, load_balanced_episodes,
    make_sim,
    reset_to, sample_detour_waypoint, scene_glb_path, scene_keys_with_episodes)

OUT_ROOT = Path("/data/topovlm/nav_baseline/data")


class ExpertDriver:
    """Executes the expert action, so the agent stays on an optimal path."""

    name = "expert"

    def __call__(self, step, teacher, sim, episode):
        return teacher


class WaypointDetourDriver:
    """Drives to a random intermediate waypoint first, then continues to the goal.

    While the waypoint has not been reached the executed action heads for the waypoint, so
    the agent walks a route the shortest path would never take. The label recorded by the
    caller is unaffected and still points at the final goal.
    """

    name = "waypoint"

    def __init__(self, expert, sim, episode, rng, arrive_m: float = 1.0,
                 max_leg_steps: int = 120):
        self.expert = expert
        self.arrive_m = arrive_m
        self.max_leg_steps = max_leg_steps
        self.waypoint = sample_detour_waypoint(
            sim.pathfinder, sim.get_agent(0).get_state().position, episode.goal_positions, rng)
        self.reached = self.waypoint is None
        self.leg_steps = 0

    def __call__(self, step, teacher, sim, episode):
        if self.reached:
            return teacher
        position = sim.get_agent(0).get_state().position
        distance = geodesic_to_goals(sim.pathfinder, position, [self.waypoint])
        self.leg_steps += 1
        if (not np.isfinite(distance) or distance <= self.arrive_m
                or self.leg_steps > self.max_leg_steps):
            self.reached = True
            return teacher
        action = expert_action_to(self.expert, self.waypoint)
        return teacher if action == 0 else action


class PerturbationDriver:
    """Follows the expert but injects short bursts of random actions.

    This produces the mildly off-course states that a policy reaches after a single wrong
    turn, complementing the larger route changes made by the waypoint generator.
    """

    name = "perturbation"

    def __init__(self, rng, burst_probability: float = 0.06, burst_length: int = 4):
        self.rng = rng
        self.burst_probability = burst_probability
        self.burst_length = burst_length
        self.remaining = 0

    def __call__(self, step, teacher, sim, episode):
        if self.remaining > 0:
            self.remaining -= 1
            return int(self.rng.integers(1, NUM_ACTIONS))  # never STOP during a burst
        if self.rng.random() < self.burst_probability:
            self.remaining = self.burst_length - 1
            return int(self.rng.integers(1, NUM_ACTIONS))
        return teacher


class PolicyDriver:
    """Executes the learned policy's action (DAgger); the expert still supplies the label."""

    name = "policy"

    def __init__(self, policy, device, min_steps, beta, rng):
        self.policy = policy
        self.device = device
        self.min_steps = min_steps
        self.beta = beta
        self.rng = rng
        self.hidden = None

    def __call__(self, step, teacher, sim, episode):
        if self.rng.random() < self.beta:
            return teacher
        depth = torch.from_numpy(depth_observation(sim.get_sensor_observations())[None])
        goal = torch.from_numpy(goal_vector(sim, episode)[None])
        previous = torch.tensor([0], dtype=torch.long, device=self.device)
        action, self.hidden = self.policy.act(
            depth.float().to(self.device), goal.float().to(self.device), previous,
            self.hidden, allow_stop=(step >= self.min_steps))
        return action


def collect_episode(sim, expert, episode, driver, *, max_steps, min_steps, save_rgb):
    """Record one episode. The label is always the expert action toward the final goal."""
    observations = reset_to(sim, episode)
    start_geodesic = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                                       episode.goal_positions)
    if not np.isfinite(start_geodesic) or start_geodesic < SUCCESS_DISTANCE_M:
        return None, {"skipped": "unreachable_or_already_at_goal"}

    depths, rgbs, goals, previous_actions, labels, executed = [], [], [], [], [], []
    positions = []
    previous_action = 0
    forward_steps = 0
    for step in range(max_steps):
        teacher = expert_action(expert, sim, episode)          # the label, toward the goal
        positions.append(np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32))
        depths.append((depth_observation(observations) * 255).astype(np.uint8))
        if save_rgb:
            rgbs.append(np.asarray(observations["rgb"])[..., :3].astype(np.uint8))
        goals.append(goal_vector(sim, episode))
        previous_actions.append(previous_action)
        labels.append(teacher)

        action = driver(step, teacher, sim, episode)            # what is actually executed
        if action == 0 and step < min_steps:
            action = teacher if teacher != 0 else 1
        executed.append(action)
        if action == 0:
            break
        observations = sim.step(ACTION_NAME[action])
        if action == 1:
            forward_steps += 1
        previous_action = action

    final_geodesic = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                                       episode.goal_positions)
    samples = {
        "depth": np.stack(depths, axis=0),
        "goal": np.stack(goals, axis=0).astype(np.float32),
        "prev_action": np.asarray(previous_actions, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
        "executed": np.asarray(executed, dtype=np.int64),
        "position": np.stack(positions, axis=0),
    }
    if save_rgb:
        samples["rgb"] = np.stack(rgbs, axis=0)
    agreement = float(np.mean(samples["label"] == samples["executed"]))
    info = {"steps": len(labels), "driver": driver.name,
            "start_geodesic": round(float(start_geodesic), 3),
            "final_geodesic": round(float(final_geodesic), 3),
            "forward_steps": forward_steps,
            "label_matches_executed": round(agreement, 3)}
    return samples, info


def make_driver(mode, *, expert, sim, episode, rng, args, policy, device):
    if mode == "expert":
        return ExpertDriver()
    if mode == "detour":
        if rng.random() < args.perturbation_ratio:
            return PerturbationDriver(rng, burst_probability=args.burst_probability,
                                      burst_length=args.burst_length)
        return WaypointDetourDriver(expert, sim, episode, rng)
    if mode == "dagger":
        return PolicyDriver(policy, device, args.min_steps, args.dagger_beta, rng)
    raise ValueError(f"Unsupported collection mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["expert", "detour", "dagger"], default="expert")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--episodes-per-scene", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--perturbation-ratio", type=float, default=0.25,
                        help="fraction of detour episodes made by perturbation instead of a waypoint")
    parser.add_argument("--burst-probability", type=float, default=0.06)
    parser.add_argument("--burst-length", type=int, default=4)
    parser.add_argument("--dagger-beta", type=float, default=0.0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-rgb", action="store_true",
                        help="also store RGB frames for later vision-language encoding")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = None
    if args.mode == "dagger":
        from nav_baseline.policy import NavPolicy
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for DAgger collection")
        state = torch.load(args.checkpoint, map_location=device)
        policy = NavPolicy(**state["model_args"]).to(device)
        policy.load_state_dict(state["model"])
        policy.eval()
    print(f"[collect] mode={args.mode} save_rgb={args.save_rgb} "
          f"perturbation_ratio={args.perturbation_ratio}", flush=True)

    out_dir = OUT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = scene_keys_with_episodes()[: args.scenes]
    print(f"[collect] {len(scenes)} scenes x {args.episodes_per_scene} episodes", flush=True)

    index, total_steps = [], 0
    driver_counts: dict[str, int] = {}
    for scene_index, scene_key in enumerate(scenes, start=1):
        episodes = load_balanced_episodes(scene_key, limit=args.episodes_per_scene)
        if not episodes:
            continue
        sim = make_sim(scene_glb_path(scene_key), with_rgb=args.save_rgb)
        expert = build_expert(sim)
        stored = {}
        try:
            for episode in episodes:
                if not episode.goal_positions:
                    continue
                reset_to(sim, episode)
                driver = make_driver(args.mode, expert=expert, sim=sim, episode=episode,
                                     rng=rng, args=args, policy=policy, device=device)
                samples, info = collect_episode(sim, expert, episode, driver,
                                                max_steps=args.max_steps,
                                                min_steps=args.min_steps,
                                                save_rgb=args.save_rgb)
                if samples is None:
                    continue
                # Episode ids restart at 0 for each target object, so the object must be
                # part of the key or episodes with different targets overwrite each other.
                key = f"{scene_key}_{episode.object_category}_{episode.episode_id}"
                for field, value in samples.items():
                    stored[f"{key}|{field}"] = value
                total_steps += info["steps"]
                driver_counts[info["driver"]] = driver_counts.get(info["driver"], 0) + 1
                index.append({"key": key, "scene": scene_key,
                              "object": episode.object_category, **info})
        finally:
            sim.close()
        if stored:
            np.savez_compressed(out_dir / f"{scene_key}.npz", **stored)
        print(f"  [{scene_index}/{len(scenes)}] {scene_key}: {len(index)} episodes, "
              f"{total_steps} steps, drivers={driver_counts}", flush=True)

    agreements = [r["label_matches_executed"] for r in index]
    summary = {"mode": args.mode, "episodes": len(index), "steps": total_steps,
               "driver_counts": driver_counts, "save_rgb": args.save_rgb,
               "mean_label_matches_executed": round(float(np.mean(agreements)), 3)
               if agreements else None,
               "records": index}
    (out_dir / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[collect] {len(index)} episodes / {total_steps} steps -> {out_dir}", flush=True)
    print(f"[collect] drivers={driver_counts} "
          f"label==executed rate={summary['mean_label_matches_executed']}", flush=True)


if __name__ == "__main__":
    main()
