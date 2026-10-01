"""Closed-loop evaluation for the nav_baseline navigator.

The policy is run step by step in the simulator until it emits STOP or reaches the step
limit. Two metrics are reported, following the standard embodied-navigation definitions
(Anderson et al., 2018, "On Evaluation of Embodied Navigation Agents"):

- **Success Rate (SR)**: the fraction of episodes in which the agent stops within
  ``SUCCESS_DISTANCE_M`` geodesic metres of a goal viewpoint.
- **SPL** (Success weighted by Path Length): success weighted by how close the travelled
  path was to the shortest path, so that wandering to the goal scores lower than going
  directly.

Every evaluation also runs the shortest-path expert on the same episodes. The expert score
is the achievable upper bound for this task setup, so the gap between policy and expert
isolates what the learned policy is missing rather than how hard the episodes happen to be.

Scenes are split deterministically: by default the first ``--train-scenes`` scenes are the
ones used for collection, and evaluation runs on the scenes after them, so the reported
numbers measure generalisation to unseen buildings rather than memorisation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nav_baseline.env import (
    ACTION_NAME, FORWARD_M, SUCCESS_DISTANCE_M, build_expert, depth_observation,
    expert_action, geodesic_to_goals, goal_vector, load_episodes, make_sim, reset_to,
    scene_glb_path, scene_keys_with_episodes)
from nav_baseline.policy import NavPolicy

RESULT_ROOT = Path(__file__).resolve().parents[0] / "results"


def summarise(rollouts: list[dict]) -> dict:
    """Aggregate per-episode outcomes into success rate, SPL, and diagnostic averages."""
    if not rollouts:
        return {"episodes": 0}
    finite = [r["final_geodesic"] for r in rollouts if np.isfinite(r["final_geodesic"])]
    return {
        "episodes": len(rollouts),
        "success_rate": round(float(np.mean([r["success"] for r in rollouts])), 3),
        "spl": round(float(np.mean([r["spl"] for r in rollouts])), 3),
        "stop_rate": round(float(np.mean([r["stopped"] for r in rollouts])), 3),
        "mean_final_geodesic_m": round(float(np.mean(finite)), 3) if finite else None,
        "mean_steps": round(float(np.mean([r["steps"] for r in rollouts])), 1),
    }


def score(success: bool, stopped: bool, start_geodesic: float, forward_steps: int,
          final_geodesic: float, steps: int) -> dict:
    path_length = forward_steps * FORWARD_M
    spl = (start_geodesic / max(start_geodesic, path_length, 1e-6)) if success else 0.0
    return {"success": bool(success), "stopped": bool(stopped), "spl": round(float(spl), 3),
            "start_geodesic": round(float(start_geodesic), 3),
            "final_geodesic": float(final_geodesic), "steps": int(steps)}


def rollout_policy(sim, policy, device, episode, *, max_steps, min_steps):
    """Drive the episode with the learned policy, one action at a time."""
    observations = reset_to(sim, episode)
    start_geodesic = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                                       episode.goal_positions)
    hidden = None
    prev_action = 0
    forward_steps = 0
    stopped = False
    steps = 0
    for step in range(max_steps):
        depth = torch.from_numpy(depth_observation(observations)[None]).float().to(device)
        goal = torch.from_numpy(goal_vector(sim, episode)[None]).float().to(device)
        prev = torch.tensor([prev_action], dtype=torch.long, device=device)
        action, hidden = policy.act(depth, goal, prev, hidden, allow_stop=(step >= min_steps))
        steps = step + 1
        if action == 0:
            stopped = True
            break
        observations = sim.step(ACTION_NAME[action])
        if action == 1:
            forward_steps += 1
        prev_action = action
    final_geodesic = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                                       episode.goal_positions)
    success = stopped and final_geodesic <= SUCCESS_DISTANCE_M
    return score(success, stopped, start_geodesic, forward_steps, final_geodesic, steps)


def rollout_expert(sim, expert, episode, *, max_steps):
    """Drive the same episode with the shortest-path expert to obtain the upper bound."""
    reset_to(sim, episode)
    start_geodesic = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                                       episode.goal_positions)
    forward_steps = 0
    stopped = False
    steps = 0
    for step in range(max_steps):
        action = expert_action(expert, sim, episode)
        steps = step + 1
        if action == 0:
            stopped = True
            break
        sim.step(ACTION_NAME[action])
        if action == 1:
            forward_steps += 1
    final_geodesic = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                                       episode.goal_positions)
    success = stopped and final_geodesic <= SUCCESS_DISTANCE_M
    return score(success, stopped, start_geodesic, forward_steps, final_geodesic, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-scenes", type=int, default=20,
                        help="number of leading scenes reserved for collection/training")
    parser.add_argument("--eval-scenes", type=int, default=5)
    parser.add_argument("--episodes-per-scene", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--split", choices=["heldout", "train"], default="heldout",
                        help="'heldout' evaluates on scenes after the training scenes")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    policy = NavPolicy(**state["model_args"]).to(device)
    policy.load_state_dict(state["model"])
    policy.eval()
    print(f"[eval] policy from {args.checkpoint} (epoch {state.get('epoch')}, "
          f"trained on {state.get('data')})", flush=True)

    all_scenes = scene_keys_with_episodes()
    scenes = (all_scenes[args.train_scenes: args.train_scenes + args.eval_scenes]
              if args.split == "heldout" else all_scenes[: args.eval_scenes])
    print(f"[eval] split={args.split}, scenes={scenes}", flush=True)

    policy_rollouts, expert_rollouts = [], []
    for scene_key in scenes:
        episodes = load_episodes(scene_key, limit=args.episodes_per_scene)
        if not episodes:
            continue
        sim = make_sim(scene_glb_path(scene_key))
        expert = build_expert(sim)
        try:
            for episode in episodes:
                if not episode.goal_positions:
                    continue
                start = geodesic_to_goals(sim.pathfinder, episode.start_position,
                                          episode.goal_positions)
                if not np.isfinite(start) or start < SUCCESS_DISTANCE_M:
                    continue
                policy_result = rollout_policy(sim, policy, device, episode,
                                               max_steps=args.max_steps, min_steps=args.min_steps)
                expert_result = rollout_expert(sim, expert, episode, max_steps=args.max_steps)
                policy_rollouts.append({"scene": scene_key, **policy_result})
                expert_rollouts.append({"scene": scene_key, **expert_result})
        finally:
            sim.close()
        print(f"  {scene_key}: policy SR so far {np.mean([r['success'] for r in policy_rollouts]):.3f} "
              f"({len(policy_rollouts)} episodes)", flush=True)

    result = {
        "checkpoint": args.checkpoint,
        "config": {"split": args.split, "scenes": scenes,
                   "episodes_per_scene": args.episodes_per_scene,
                   "max_steps": args.max_steps, "min_steps": args.min_steps,
                   "success_distance_m": SUCCESS_DISTANCE_M},
        "policy": summarise(policy_rollouts),
        "expert_upper_bound": summarise(expert_rollouts),
        "policy_rollouts": policy_rollouts,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    name = f"eval_{args.split}{args.tag}.json"
    (RESULT_ROOT / name).write_text(json.dumps(result, indent=2) + "\n")
    print("\n=== nav_baseline closed-loop evaluation ===")
    print(json.dumps({"policy": result["policy"],
                      "expert_upper_bound": result["expert_upper_bound"]}, indent=2))
    print(f"\nwrote {RESULT_ROOT / name}")


if __name__ == "__main__":
    main()
