"""Choose one fixed route per building for the simplest navigation check.

Stage zero of this study fixes the starting pose and the goal so that the policy has exactly
one route to learn per building. Which route is chosen can change the answer, so three
selection rules are provided and the experiment runs all three.

- ``random``: any problem from the building. This is the baseline with no condition imposed.
- ``matched``: a problem whose starting distance falls in a chosen band, so that buildings
  are comparable to one another and no route is trivially short.
- ``junction``: a problem whose shortest path passes a junction, meaning a place where the
  agent could go more than one way. Such a route cannot be solved by walking straight, so it
  requires reading the layout.

A **junction** here is a point on the route where the agent, standing on the route, has at
least two clearly different navigable directions available. It is detected geometrically
from the route and the navigation mesh rather than from any annotation: the route is walked,
and at each point the surroundings are sampled in a ring to see how many separate openings
exist. A point with three or more openings, one of which the route does not take, counts as
a junction.

The selected routes are written to a file so that every later step of the experiment reads
exactly the same problems.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nav_baseline.env import (  # noqa: E402
    ACTION_NAME, build_expert, expert_action, geodesic_to_goals, load_balanced_episodes,
    open_scene, reset_to, scene_keys_with_episodes)

RESULT_DIR = Path(__file__).resolve().parents[0] / "results"
SELECTION_RULES = ("random", "matched", "junction")
MATCHED_BAND_M = (5.0, 8.0)
RING_RADIUS_M = 1.5
RING_SAMPLES = 24
MIN_OPENINGS_FOR_JUNCTION = 3


def walk_expert(sim, expert, episode, max_steps: int = 500):
    """Follow the shortest path and return the positions visited."""
    reset_to(sim, episode)
    positions = [np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32)]
    for _ in range(max_steps):
        action = expert_action(expert, sim, episode)
        if action == 0:
            break
        sim.step(ACTION_NAME[action])
        positions.append(np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32))
    return np.stack(positions, axis=0)


def count_openings(pathfinder, position) -> int:
    """How many separate navigable directions lead away from a point.

    A ring of points is sampled around the position. Each sample is navigable or not, and
    the number of separate runs of navigable samples around the ring is the number of
    openings. A corridor gives two openings, a dead end gives one, and a junction gives
    three or more.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, RING_SAMPLES, endpoint=False)
    navigable = []
    for angle in angles:
        probe = np.asarray(position, dtype=np.float32).copy()
        probe[0] += RING_RADIUS_M * np.cos(angle)
        probe[2] += RING_RADIUS_M * np.sin(angle)
        navigable.append(bool(pathfinder.is_navigable(probe)))
    if all(navigable):
        return 1  # open floor rather than a junction of corridors
    runs = 0
    for index, value in enumerate(navigable):
        if value and not navigable[index - 1]:
            runs += 1
    return runs


def route_junctions(pathfinder, positions, stride: int = 4) -> int:
    """Number of points along a route that look like junctions."""
    found = 0
    for index in range(0, len(positions), stride):
        if count_openings(pathfinder, positions[index]) >= MIN_OPENINGS_FOR_JUNCTION:
            found += 1
    return found


def describe(sim, expert, episode) -> dict | None:
    """Measure the one route this problem defines, or return None if it is unusable."""
    reset_to(sim, episode)
    start = geodesic_to_goals(sim.pathfinder, sim.get_agent(0).get_state().position,
                              episode.goal_positions)
    if not np.isfinite(start) or start < 1.0:
        return None
    positions = walk_expert(sim, expert, episode)
    if len(positions) < 5:
        return None
    walked = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    final = geodesic_to_goals(sim.pathfinder, positions[-1], episode.goal_positions)
    return {
        "scene": episode.scene_key,
        "episode_id": episode.episode_id,
        "object": episode.object_category,
        "start_position": [float(x) for x in episode.start_position],
        "start_rotation": [float(x) for x in episode.start_rotation],
        "start_geodesic_m": round(float(start), 3),
        "expert_steps": int(len(positions) - 1),
        "expert_walked_m": round(walked, 2),
        "expert_final_geodesic_m": round(float(final), 3),
        "junctions_on_route": int(route_junctions(sim.pathfinder, positions)),
    }


def choose(candidates: list[dict], rule: str, rng) -> dict | None:
    """Pick one route from a building's candidates according to the rule."""
    usable = [c for c in candidates if c["expert_final_geodesic_m"] <= 1.0]
    if not usable:
        return None
    if rule == "random":
        return usable[int(rng.integers(len(usable)))]
    if rule == "matched":
        low, high = MATCHED_BAND_M
        band = [c for c in usable if low <= c["start_geodesic_m"] <= high]
        if band:
            return band[int(rng.integers(len(band)))]
        # No problem lands in the band; take the one closest to its middle.
        middle = (low + high) / 2
        return min(usable, key=lambda c: abs(c["start_geodesic_m"] - middle))
    if rule == "junction":
        with_junction = [c for c in usable if c["junctions_on_route"] > 0]
        if with_junction:
            return max(with_junction, key=lambda c: c["junctions_on_route"])
        return None
    raise ValueError(f"unknown selection rule: {rule}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--candidates-per-scene", type=int, default=12,
                        help="problems examined per building before one is chosen")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    selected = {rule: [] for rule in SELECTION_RULES}
    surveyed = []

    for scene_key in scene_keys_with_episodes()[: args.scenes]:
        episodes = load_balanced_episodes(scene_key, limit=args.candidates_per_scene)
        if not episodes:
            continue
        sim = open_scene(scene_key)
        expert = build_expert(sim)
        try:
            candidates = []
            for episode in episodes:
                if not episode.goal_positions:
                    continue
                described = describe(sim, expert, episode)
                if described is not None:
                    candidates.append(described)
        finally:
            sim.close()
        if not candidates:
            continue
        surveyed.extend(candidates)
        for rule in SELECTION_RULES:
            picked = choose(candidates, rule, rng)
            if picked is not None:
                selected[rule].append(picked)
        print(f"  {scene_key}: {len(candidates)} candidates, "
              + ", ".join(f"{r}={'yes' if selected[r] and selected[r][-1]['scene'] == scene_key else 'none'}"
                          for r in SELECTION_RULES), flush=True)

    for rule in SELECTION_RULES:
        rows = selected[rule]
        path = RESULT_DIR / f"routes_{rule}.json"
        path.write_text(json.dumps({"rule": rule, "routes": rows}, indent=2) + "\n")
        if rows:
            distances = [r["start_geodesic_m"] for r in rows]
            junctions = [r["junctions_on_route"] for r in rows]
            print(f"\n[{rule}] {len(rows)} routes -> {path.name}")
            print(f"  start distance: min {min(distances):.1f} median "
                  f"{np.median(distances):.1f} max {max(distances):.1f} m")
            print(f"  expert steps: median {int(np.median([r['expert_steps'] for r in rows]))}")
            print(f"  junctions on route: median {int(np.median(junctions))} "
                  f"(range {min(junctions)}-{max(junctions)})")
        else:
            print(f"\n[{rule}] no routes selected")

    (RESULT_DIR / "route_survey.json").write_text(
        json.dumps({"scenes": args.scenes, "candidates": surveyed}, indent=2) + "\n")
    print(f"\nsurveyed {len(surveyed)} candidate routes in total")


if __name__ == "__main__":
    main()
