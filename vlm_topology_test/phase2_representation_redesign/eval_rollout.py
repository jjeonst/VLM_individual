"""Closed-loop evaluation of a trained policy in the simulator (step 4).

The policy is driven one step at a time: the simulator renders the current view, the frozen
vision-language model turns it into a representation under the same question the policy was
trained with, the policy reads the sequence of representations so far and chooses an action,
and the simulator applies it. This continues until the policy stops or the step limit is
reached. Because the policy cannot know which view comes next, the model must run inside the
loop and the frames cannot be batched, which makes this the most expensive step.

Reported measurements
---------------------
- **success rate**: stopped within ``--success-distance`` geodesic metres of a goal viewpoint.
- **path length ratio** (successes only): distance walked divided by the shortest distance
  from the start. A value near 1.0 means the agent went more or less straight to the goal.
- **progress**: how much of the initial distance to the goal was removed, whether or not the
  episode succeeded. This is reported because a binary success rate carries no information
  when every condition fails, and the experiment must still be able to rank conditions in
  that case.
- **reached without stopping**: episodes that came within the success distance at some point
  but never stopped there, which separates "cannot navigate" from "cannot tell it arrived".
- **expert upper bound**: the same episodes solved by the shortest-path follower.

Correctness requirements enforced here
--------------------------------------
- The question used to encode observations is taken from the trained condition, so a policy
  trained on the structure question is never evaluated with the baseline question.
- The compression projection is loaded from the checkpoint rather than refitted, so the
  policy sees vectors in the space it was trained on.
- All conditions are evaluated on the same episodes from scenes that were never used for
  collection.

Run:
  python -m vlm_topology_test.phase2_representation_redesign.eval_rollout \\
      --checkpoint /data/topovlm/nav_baseline/policies/cond3/model.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from nav_baseline.env import (ACTION_NAME, FORWARD_M, build_expert, expert_action,
                              geodesic_to_goals, load_balanced_episodes, make_sim, reset_to,
                              scene_glb_path, scene_keys_with_episodes)
from vlm_topology_test.phase2_representation_redesign.encode_vlm import PROMPTS, LastTokenEncoder
from vlm_topology_test.phase2_representation_redesign.train_policy import PR2LPolicy

RESULT_DIR = Path(__file__).resolve().parents[0] / "results" / "rollout"


def prompt_for_checkpoint(state: dict, override: str | None) -> str:
    """The question the policy was trained with; evaluating with another one is invalid."""
    if override:
        return override
    sources = " ".join(state.get("data", []))
    matches = [name for name in PROMPTS if f"__{name}" in sources]
    if len(matches) != 1:
        raise ValueError(f"cannot infer the training question from {sources!r}; pass --prompt")
    return matches[0]


class PolicyRunner:
    """Keeps the growing sequence of compressed representations and picks the next action."""

    def __init__(self, state: dict, device, context_window: int = 0):
        self.device = device
        # Position embeddings past the longest training trajectory were never optimised, so
        # the sequence handed to the policy is limited to the range it actually learned.
        trained = int(state.get("max_train_length") or 0)
        limit = context_window or trained or 0
        self.context_window = min(limit, state["model_args"].get("max_positions", 512)) \
            if limit else None
        self.policy = PR2LPolicy(**state["model_args"]).to(device).eval()
        self.policy.load_state_dict(state["model"])
        projection = state["projection"]
        self.mean = torch.as_tensor(projection["mean"], device=device)
        self.components = torch.as_tensor(projection["components"], device=device)
        self.reset()

    def reset(self):
        self.history = []

    def act(self, vector: np.ndarray, allow_stop: bool = True) -> int:
        compressed = (torch.as_tensor(vector, device=self.device, dtype=torch.float32)
                      - self.mean) @ self.components.T
        self.history.append(compressed)
        span = self.context_window or self.policy.max_positions
        window = self.history[-span:]
        features = torch.stack(window, dim=0)[None]
        mask = torch.ones(1, features.shape[1], dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            logits = self.policy(features, mask)[0, -1]
        if not allow_stop:
            logits = logits.clone()
            logits[0] = float("-inf")
        return int(logits.argmax(-1).item())


def rollout_policy(sim, encoder, runner, episode, question, *, max_steps, min_steps):
    observations = reset_to(sim, episode)
    runner.reset()
    from PIL import Image

    start = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    closest = start
    forward_steps = 0
    stopped = False
    steps = 0
    for step in range(max_steps):
        image = Image.fromarray(np.asarray(observations["rgb"])[..., :3].astype("uint8"),
                                mode="RGB")
        vector, _ = encoder.encode(image, question)
        action = runner.act(vector, allow_stop=(step >= min_steps))
        steps = step + 1
        if action == 0:
            stopped = True
            break
        observations = sim.step(ACTION_NAME[action])
        if action == 1:
            forward_steps += 1
        here = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                                 episode.goal_positions)
        closest = min(closest, here)
    final = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    return summarise_run(start, final, closest, forward_steps, stopped, steps)


def rollout_expert(sim, expert, episode, *, max_steps):
    reset_to(sim, episode)
    start = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    closest = start
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
        closest = min(closest, geodesic_to_goals(sim.pathfinder,
                                                 sim.get_agent(0).get_state().position,
                                                 episode.goal_positions))
    final = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    return summarise_run(start, final, closest, forward_steps, stopped, steps)


def summarise_run(start, final, closest, forward_steps, stopped, steps) -> dict:
    walked = forward_steps * FORWARD_M
    progress = ((start - final) / start) if np.isfinite(start) and start > 0 else 0.0
    return {"start_geodesic": _round(start), "final_geodesic": _round(final),
            "closest_geodesic": _round(closest), "walked_m": round(float(walked), 2),
            "stopped": bool(stopped), "steps": int(steps),
            "progress_ratio": round(float(np.clip(progress, -1.0, 1.0)), 3)}


def _round(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 3)


def score(runs: list[dict], thresholds=(0.1, 1.0)) -> dict:
    """Aggregate outcomes.

    The headline threshold is the one used by the standard Habitat ObjectNav setting, which
    measures geodesic distance to the goal view points and calls an episode successful when
    the agent stops within 0.1 m of one of them. A looser 1.0 m threshold and a graded
    progress measure are reported alongside, because a strict binary criterion carries no
    information for ranking conditions when every condition is close to zero.
    """
    if not runs:
        return {"episodes": 0}
    summary = {"episodes": len(runs),
               "stop_rate": round(float(np.mean([r["stopped"] for r in runs])), 3),
               "progress_ratio_mean": round(float(np.mean([r["progress_ratio"] for r in runs])), 3),
               "progress_ratio_median": round(float(np.median([r["progress_ratio"] for r in runs])), 3),
               "mean_steps": round(float(np.mean([r["steps"] for r in runs])), 1),
               "mean_final_geodesic_m": _round(np.mean(
                   [r["final_geodesic"] for r in runs if r["final_geodesic"] is not None]))}
    for threshold in thresholds:
        successes, ratios, spls, reached_only = [], [], [], []
        for run in runs:
            success = bool(run["stopped"] and run["final_geodesic"] is not None
                           and run["final_geodesic"] <= threshold)
            successes.append(success)
            reached = (run["closest_geodesic"] is not None
                       and run["closest_geodesic"] <= threshold)
            reached_only.append(reached and not success)
            start = run["start_geodesic"] or 0.0
            if success and start > 0:
                ratios.append(run["walked_m"] / max(start, 1e-6))
                spls.append(start / max(start, run["walked_m"], 1e-6))
        key = f"at_{threshold}m"
        summary[key] = {
            "success_rate": round(float(np.mean(successes)), 3),
            "spl": round(float(np.sum(spls) / len(runs)), 3),
            "successes": int(np.sum(successes)),
            "path_length_ratio_mean": round(float(np.mean(ratios)), 3) if ratios else None,
            "path_length_ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
            "reached_but_did_not_stop": round(float(np.mean(reached_only)), 3),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default=None,
                        help="override the question; by default it is taken from the checkpoint")
    parser.add_argument("--train-scenes", type=int, default=40,
                        help="number of leading scenes reserved for collection")
    parser.add_argument("--eval-scenes", type=int, default=10)
    parser.add_argument("--episodes-per-scene", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--min-steps", type=int, default=0,
                        help="steps during which STOP is not allowed. The default of 0 means "
                             "no restriction: unlike the earlier data, none of the collected "
                             "episodes begin with a STOP label, so the policy has no reason "
                             "to stop immediately and suppressing it would only distort results")
    parser.add_argument("--success-distance", type=float, default=0.1,
                        help="headline threshold, matching the standard ObjectNav setting")
    parser.add_argument("--context-window", type=int, default=0,
                        help="number of most recent steps fed to the policy; 0 uses the "
                             "longest trajectory seen in training, which keeps position "
                             "indices inside the range the model was trained on")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--skip-expert", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    prompt_name = prompt_for_checkpoint(state, args.prompt)
    template = PROMPTS[prompt_name]
    run_name = Path(args.checkpoint).parent.name
    print(f"[eval] {run_name}: trained on {state.get('data')}, epoch {state.get('epoch')}",
          flush=True)
    print(f"[eval] question = {prompt_name!r}: {template}", flush=True)

    scenes = scene_keys_with_episodes()[args.train_scenes: args.train_scenes + args.eval_scenes]
    print(f"[eval] {len(scenes)} held-out scenes x {args.episodes_per_scene} episodes",
          flush=True)

    runner = PolicyRunner(state, device, context_window=args.context_window)
    print(f"[eval] context window = {runner.context_window or 'full'} steps "
          f"(longest training trajectory {state.get('max_train_length')})", flush=True)
    encoder = LastTokenEncoder(args.max_new_tokens)
    print("[eval] policy and vision-language model ready", flush=True)

    policy_runs, expert_runs = [], []
    for scene_index, scene_key in enumerate(scenes, start=1):
        episodes = load_balanced_episodes(scene_key, limit=args.episodes_per_scene)
        if not episodes:
            continue
        sim = make_sim(scene_glb_path(scene_key), with_rgb=True)
        expert = build_expert(sim)
        try:
            for episode in episodes:
                if not episode.goal_positions:
                    continue
                start = geodesic_to_goals(sim.pathfinder, episode.start_position,
                                          episode.goal_positions)
                if not np.isfinite(start) or start < args.success_distance:
                    continue
                question = template.format(goal=episode.object_category)
                policy_runs.append({"scene": scene_key, "object": episode.object_category,
                                    **rollout_policy(sim, encoder, runner, episode, question,
                                                     max_steps=args.max_steps,
                                                     min_steps=args.min_steps)})
                if not args.skip_expert:
                    expert_runs.append({"scene": scene_key,
                                        **rollout_expert(sim, expert, episode,
                                                         max_steps=args.max_steps)})
        finally:
            sim.close()
        current = score(policy_runs)
        print(f"  [{scene_index}/{len(scenes)}] {scene_key}: {current['episodes']} episodes, "
              f"success@0.1m {current['at_0.1m']['success_rate']}, "
              f"success@1.0m {current['at_1.0m']['success_rate']}, "
              f"progress {current['progress_ratio_mean']}",
              flush=True)

    result = {
        "run_name": run_name, "checkpoint": args.checkpoint,
        "question": {"name": prompt_name, "template": template},
        "trained_on": state.get("data"),
        "config": {"scenes": scenes, "episodes_per_scene": args.episodes_per_scene,
                   "max_steps": args.max_steps, "min_steps": args.min_steps,
                   "success_distance_m": args.success_distance,
                   "context_window": runner.context_window},
        "policy": score(policy_runs),
        "expert_upper_bound": score(expert_runs) if expert_runs else None,
        "policy_runs": policy_runs,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{run_name}{args.tag}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n=== {run_name} ===")
    print(json.dumps({"policy": result["policy"],
                      "expert_upper_bound": result["expert_upper_bound"]}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
