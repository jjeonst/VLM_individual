"""Drive the trained policy along its route and check that it is really looking.

Two things are measured.

**Does it reach the goal?** The policy is placed at the route's start, with the same kind of
small nudge used during collection but drawn fresh, and it drives until it stops or runs out
of steps. Success is reported at three distances from a goal viewpoint, because one number
conflates two different abilities:

- **0.1 m** is the standard used by the simulator's own configuration and by the prior work,
  and it is the number to quote when comparing against them.
- **0.25 m** is the length of one forward step. A policy that walked to the right place but
  could not land inside the tighter circle counts here and not at 0.1 m, so the gap between
  the two measures how much of the failure is precision of the final stop rather than an
  inability to find the goal. The two are worth separating because the reachable positions
  form a lattice of roughly this spacing while goal viewpoints do not sit on it, so the
  tighter number is depressed by geometry that has nothing to do with navigation.
- **1.0 m** is kept for continuity with the earlier experiment.

How much of the initial distance was closed is reported alongside, and the shortest-path
expert solves the same problems to give the achievable ceiling.

**Is it using what it sees?** On a fixed route a policy can score well by reproducing the
demonstrated action sequence without looking at anything. To rule that out, every episode is
repeated with the observation replaced:

- ``blank``: a mid-grey image, carrying no information at all,
- ``shuffled``: an image taken from a different building, so the input is a real view but the
  wrong one.

If the policy genuinely reads its input, replacing the input should change what it does and
should destroy performance. If the scores barely move, the policy is replaying a memorised
sequence and its apparent success means nothing. The agreement between the actions taken
under the true observation and under the replacement is reported as a direct measure of this.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nav_baseline.env import (  # noqa: E402
    ACTION_NAME, FORWARD_M, build_expert, expert_action, geodesic_to_goals, goal_vector,
    open_scene, reset_to)
from vlm_topology_test.phase3_fixed_route.collect import (  # noqa: E402
    DATA_ROOT, episode_for, jitter_start, load_routes)
from vlm_topology_test.phase3_fixed_route.policy import FixedRoutePolicy  # noqa: E402
from vlm_topology_test.phase3_fixed_route.train import CKPT_ROOT  # noqa: E402

RESULT_DIR = Path(__file__).resolve().parents[0] / "results" / "rollout"
OBSERVATION_MODES = ("true", "blank", "shuffled")


class Runner:
    """Holds the growing sequence of frames and picks the next action."""

    def __init__(self, state: dict, device):
        self.device = device
        self.policy = FixedRoutePolicy(**state["model_args"]).to(device).eval()
        self.policy.load_state_dict(state["model"])
        self.mean = torch.as_tensor(state["normalisation"]["mean"], device=device)
        self.scale = torch.as_tensor(state["normalisation"]["scale"], device=device)
        self.limit = min(int(state.get("max_train_length") or 0) or self.policy.max_positions,
                         self.policy.max_positions)
        self.reset()

    def reset(self):
        self.history = []

    def act(self, tokens: np.ndarray) -> int:
        block = (torch.as_tensor(tokens, device=self.device, dtype=torch.float32)
                 - self.mean) / self.scale
        self.history.append(block)
        window = self.history[-self.limit:]
        sequence = torch.stack(window, dim=0)[None]
        mask = torch.ones(1, sequence.shape[1], dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            logits = self.policy(sequence, mask)[0, -1]
        return int(logits.argmax(-1).item())


def substitute(image: np.ndarray, mode: str, decoy: np.ndarray | None) -> np.ndarray:
    """Replace the view according to the observation mode."""
    if mode == "true":
        return image
    if mode == "blank":
        return np.full_like(image, 128)
    if mode == "shuffled" and decoy is not None:
        return decoy
    return np.full_like(image, 128)


def rollout(sim, episode, runner, encoder, question, *, mode, decoy, max_steps,
            representation):
    from PIL import Image

    observations = reset_to(sim, episode)
    runner.reset()
    start = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    closest, forward_steps, stopped, steps = start, 0, False, 0
    actions = []
    for step in range(max_steps):
        if representation == "oracle":
            base = goal_vector(sim, episode)[None, :]
            tokens = base if mode == "true" else np.zeros_like(base)
        else:
            frame = substitute(np.asarray(observations["rgb"])[..., :3].astype(np.uint8),
                               mode, decoy)
            picture = Image.fromarray(frame, mode="RGB")
            tokens, _ = encoder.encode(picture, question)
        action = runner.act(tokens)
        actions.append(action)
        steps = step + 1
        if action == 0:
            stopped = True
            break
        observations = sim.step(ACTION_NAME[action])
        if action == 1:
            forward_steps += 1
        closest = min(closest, geodesic_to_goals(
            sim.pathfinder, sim.get_agent(0).get_state().position, episode.goal_positions))
    final = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    return {"start_geodesic": _round(start), "final_geodesic": _round(final),
            "closest_geodesic": _round(closest), "walked_m": round(forward_steps * FORWARD_M, 2),
            "stopped": bool(stopped), "steps": steps, "actions": actions,
            "progress_ratio": round(float(np.clip((start - final) / max(start, 1e-6), -1, 1)), 3)}


def rollout_expert(sim, expert, episode, *, max_steps):
    reset_to(sim, episode)
    start = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    closest, forward_steps, stopped, steps = start, 0, False, 0
    for step in range(max_steps):
        action = expert_action(expert, sim, episode)
        steps = step + 1
        if action == 0:
            stopped = True
            break
        sim.step(ACTION_NAME[action])
        if action == 1:
            forward_steps += 1
        closest = min(closest, geodesic_to_goals(
            sim.pathfinder, sim.get_agent(0).get_state().position, episode.goal_positions))
    final = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    return {"start_geodesic": _round(start), "final_geodesic": _round(final),
            "closest_geodesic": _round(closest), "walked_m": round(forward_steps * FORWARD_M, 2),
            "stopped": bool(stopped), "steps": steps, "actions": [],
            "progress_ratio": round(float(np.clip((start - final) / max(start, 1e-6), -1, 1)), 3)}


def _round(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 3)


def score(runs: list[dict], thresholds=(0.1, 0.25, 1.0)) -> dict:
    if not runs:
        return {"episodes": 0}
    summary = {"episodes": len(runs),
               "stop_rate": round(float(np.mean([r["stopped"] for r in runs])), 3),
               "progress_ratio_mean": round(float(np.mean([r["progress_ratio"] for r in runs])), 3),
               "mean_steps": round(float(np.mean([r["steps"] for r in runs])), 1),
               "mean_final_geodesic_m": _round(np.mean(
                   [r["final_geodesic"] for r in runs if r["final_geodesic"] is not None]))}
    for threshold in thresholds:
        wins, ratios = [], []
        for run in runs:
            ok = bool(run["stopped"] and run["final_geodesic"] is not None
                      and run["final_geodesic"] <= threshold)
            wins.append(ok)
            if ok and run["start_geodesic"]:
                ratios.append(run["walked_m"] / max(run["start_geodesic"], 1e-6))
        summary[f"at_{threshold}m"] = {
            "success_rate": round(float(np.mean(wins)), 3),
            "successes": int(np.sum(wins)),
            "path_length_ratio_median": round(float(np.median(ratios)), 3) if ratios else None}
    return summary


def action_agreement(reference: list[dict], other: list[dict]) -> float | None:
    """How often the replaced-observation run chose the same action as the true run."""
    same = total = 0
    for a, b in zip(reference, other):
        for x, y in zip(a["actions"], b["actions"]):
            same += int(x == y)
            total += 1
    return round(same / total, 3) if total else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repeats", type=int, default=5,
                        help="rollouts per route, each with a fresh start nudge")
    parser.add_argument("--jitter-m", type=float, default=0.3)
    parser.add_argument("--jitter-deg", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--goal-radius", type=float, default=0.1)
    parser.add_argument("--modes", nargs="+", default=list(OBSERVATION_MODES),
                        choices=list(OBSERVATION_MODES))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    representation, rule, prompt = state["representation"], state["rule"], state["prompt"]
    run_name = Path(args.checkpoint).parent.name
    print(f"[eval] {run_name}: representation={representation} rule={rule} "
          f"positions={state['model_args'].get('use_positions')}", flush=True)

    encoder, template = None, None
    if representation == "vlm":
        from vlm_topology_test.phase3_fixed_route.encode import (
            PCA_FILE, PROMPTS, MultiTokenEncoder, Projection, goal_phrase)
        template = PROMPTS[prompt]
        # Frames are encoded live here, so they must be shortened by the same principal
        # directions that were used to build the training set; otherwise the policy is fed
        # a different coordinate system than the one it learned in.
        pca_path = DATA_ROOT / f"encoded_{rule}__{prompt}" / PCA_FILE
        projection = Projection.load(pca_path) if pca_path.exists() else None
        encoder = MultiTokenEncoder(projection=projection)
        expected = int(state["model_args"]["input_dim"])
        if encoder.width != expected:
            raise SystemExit(
                f"the policy was trained on tokens {expected} wide but this encoder produces "
                f"{encoder.width}; the principal directions in {pca_path} do not match the "
                "checkpoint, so the two were built by different encoding runs")
        print(f"[eval] question = {prompt!r}, token width {encoder.width}"
              + (f", directions from {pca_path}" if projection else ", no PCA"), flush=True)
    modes = [m for m in args.modes if representation == "vlm" or m != "shuffled"]

    routes = load_routes(rule)
    runner = Runner(state, device)
    rng = np.random.default_rng(args.seed)
    decoy = (np.random.default_rng(0).integers(0, 255, (256, 256, 3))).astype(np.uint8)

    results = {mode: [] for mode in modes}
    expert_runs = []
    for route in routes:
        episode = episode_for(route)
        sim = open_scene(route["scene"], with_rgb=True)
        expert = build_expert(sim, goal_radius=args.goal_radius)
        question = template.format(goal=goal_phrase(route["object"])) if template else None
        try:
            for repeat in range(args.repeats):
                seed = int(rng.integers(1 << 30))
                for mode in modes:
                    nudged = jitter_start(episode, np.random.default_rng(seed),
                                          args.jitter_m, args.jitter_deg, sim.pathfinder)
                    outcome = rollout(sim, nudged, runner, encoder, question, mode=mode,
                                      decoy=decoy, max_steps=args.max_steps,
                                      representation=representation)
                    results[mode].append({"scene": route["scene"], "repeat": repeat, **outcome})
                nudged = jitter_start(episode, np.random.default_rng(seed), args.jitter_m,
                                      args.jitter_deg, sim.pathfinder)
                expert_runs.append({"scene": route["scene"],
                                    **rollout_expert(sim, expert, nudged,
                                                     max_steps=args.max_steps)})
        finally:
            sim.close()
        current = score(results[modes[0]])
        print(f"  {route['scene']}: success@0.1m {current['at_0.1m']['success_rate']} "
              f"@0.25m {current['at_0.25m']['success_rate']}, "
              f"progress {current['progress_ratio_mean']}", flush=True)

    payload = {"run": run_name, "checkpoint": args.checkpoint,
               "representation": representation, "rule": rule, "prompt": prompt,
               "use_positions": state["model_args"].get("use_positions"),
               "config": {"repeats": args.repeats, "jitter_m": args.jitter_m,
                          "jitter_deg": args.jitter_deg, "max_steps": args.max_steps,
                          "goal_radius": args.goal_radius},
               "expert_upper_bound": score(expert_runs)}
    for mode in modes:
        payload[mode] = score(results[mode])
        if mode != "true":
            payload[mode]["action_agreement_with_true"] = action_agreement(
                results["true"], results[mode])
    payload["runs"] = {m: results[m] for m in modes}

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{run_name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n=== {run_name} ===")
    print(json.dumps({k: payload[k] for k in ["expert_upper_bound", *modes]}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
