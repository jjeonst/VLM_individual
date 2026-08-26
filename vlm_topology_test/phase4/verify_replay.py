"""Measure how faithfully the replay reproduces the path each person actually walked.

Roughly three demonstrations in ten still carry the pose the agent held at every step. Those
poses are too sparse and too irregular to serve as the source of the data, but they are exactly
what is needed to answer the question this module exists for: **does feeding the recorded
actions back into the simulator put the agent where the person actually was?** If it does not,
every image rendered after the point of divergence shows something the person never saw, and
the demonstration is no longer a demonstration.

Two measurements are reported.

- **Single-step accuracy** starts each action from the *recorded* previous pose. It isolates
  whether one action is simulated correctly, without the error from earlier steps piling up.
- **Whole-trajectory drift** starts once at the beginning and never consults the recorded poses
  again, which is what the real replay does. This is the number that matters.

Both are reported per action type, because the failures are not spread evenly: turning and
stopping are exact, and it is forward motion near obstacles that goes wrong.

The two settings under test, and why they matter:

**Navmesh.** The mesh files shipped with HM3D describe where an agent of radius 0.10 m and
height 1.5 m may walk. ObjectNav's agent is wider and shorter -- radius 0.18 m, height 0.88 m --
so habitat-lab rebuilds the walkable surface to match whenever the agent differs from the one
the file was built for (`sims/habitat_simulator/habitat_simulator.py`, the block guarded by
`default_agent_navmesh`). Using the shipped file instead leaves the agent able to squeeze
through gaps the real one cannot, which changes exactly those steps where a person walked into
something.

**Start pose.** Each episode declares a starting position, and the recorded replay also holds
the pose the agent was in before its first action. These disagree for a minority of episodes,
always by one forward step, so which one the replay trusts is worth measuring rather than
assuming.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from render import (PITCH_ACTIONS, load_selection, make_sim, recompute_navmesh, scene_path)

MATCH_TOLERANCE_M = 0.01


def describe_navmesh(pathfinder) -> str:
    settings = pathfinder.nav_mesh_settings
    return (f"radius {settings.agent_radius:.2f} m, height {settings.agent_height:.2f} m, "
            f"navigable area {pathfinder.navigable_area:.1f} m^2")


def episodes_with_poses(shard: str, limit: int) -> tuple[dict, list[dict]]:
    """Demonstrations whose every step carries a recorded pose, which alone can be checked."""
    raw = json.loads(gzip.open(shard, "rt").read())
    usable = []
    for episode in raw["episodes"]:
        replay = episode.get("reference_replay") or []
        if len(replay) > 5 and all(isinstance(s.get("agent_state"), dict) for s in replay):
            usable.append(episode)
        if len(usable) >= limit:
            break
    return raw, usable


def apply_action(sim, action: str) -> None:
    if action == "TURN_LEFT":
        sim.step("turn_left")
    elif action == "TURN_RIGHT":
        sim.step("turn_right")
    elif action == "MOVE_FORWARD":
        sim.step("move_forward")


def set_pose(sim, position, rotation) -> None:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_coeffs(np.asarray(rotation, dtype=np.float32))
    sim.get_agent(0).set_state(state, reset_sensors=True)


def position(sim) -> np.ndarray:
    return np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32)


def single_step_errors(sim, episodes: list[dict]) -> dict[str, list[float]]:
    """Error of one action taken from the pose the recording says the agent was in."""
    errors: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        replay = episode["reference_replay"]
        for index in range(1, len(replay)):
            previous = replay[index - 1]["agent_state"]
            expected = np.asarray(replay[index]["agent_state"]["position"], dtype=np.float32)
            set_pose(sim, previous["position"], previous["rotation"])
            apply_action(sim, str(replay[index]["action"]))
            errors[str(replay[index]["action"])].append(
                float(np.linalg.norm(position(sim) - expected)))
    return errors


def trajectory_drift(sim, episodes: list[dict], start_from_replay: bool) -> list[float]:
    """Error accumulated when the replay runs unaided, as it does when producing the data."""
    drifts = []
    for episode in episodes:
        replay = episode["reference_replay"]
        first = replay[0]["agent_state"]
        if start_from_replay:
            set_pose(sim, first["position"], first["rotation"])
        else:
            set_pose(sim, episode["start_position"], episode["start_rotation"])
        for step in replay:
            action = str(step["action"])
            if action not in PITCH_ACTIONS:
                apply_action(sim, action)
            expected = np.asarray(step["agent_state"]["position"], dtype=np.float32)
            drifts.append(float(np.linalg.norm(position(sim) - expected)))
    return drifts


def report(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label:<34} (측정값 없음)")
        return
    array = np.asarray(values)
    print(f"  {label:<34} {len(array):6d}개 | 중앙 {np.median(array):.4f} m "
          f"평균 {array.mean():.4f} m 최대 {array.max():7.3f} m | "
          f"{MATCH_TOLERANCE_M} m 이내 {100 * (array <= MATCH_TOLERANCE_M).mean():5.1f}%")


def start_pose_disagreement(episodes: list[dict]) -> None:
    gaps = [float(np.linalg.norm(
        np.asarray(e["start_position"], dtype=np.float32)
        - np.asarray(e["reference_replay"][0]["agent_state"]["position"], dtype=np.float32)))
        for e in episodes]
    mismatched = [g for g in gaps if g > MATCH_TOLERANCE_M]
    print(f"  선언된 start_position 과 재생 첫 자세의 불일치: "
          f"{len(mismatched)}/{len(gaps)} ({100 * len(mismatched) / max(len(gaps), 1):.0f}%)"
          + (f", 거리 중앙 {np.median(mismatched):.3f} m 최대 {max(mismatched):.3f} m"
             if mismatched else ""))


def goal_distance_check(sim, episodes: list[dict], raw_shard: dict) -> None:
    """Can the agent still reach the goal viewpoints once the surface is rebuilt?

    The viewpoints shipped with each episode were generated for the narrow agent the shipped
    navmesh describes. Goal viewpoints sit close to furniture, which is exactly where a wider
    agent stops fitting, so rebuilding the surface for the ObjectNav agent may leave some of
    them unreachable. Measuring the distance from a pose that is *known* to be a successful
    stopping point separates that possibility from a fault in the replay.
    """
    from render import geodesic_to_viewpoints, goal_viewpoints

    distances = []
    for episode in episodes:
        viewpoints = goal_viewpoints(raw_shard, episode["scene_id"],
                                     episode["object_category"])
        final = episode["reference_replay"][-1]["agent_state"]["position"]
        distances.append(geodesic_to_viewpoints(sim.pathfinder, final, viewpoints))
    array = np.asarray(distances)
    finite = array[np.isfinite(array)]
    print(f"  사람이 실제로 멈춘 지점 → 목표 관찰 지점: {len(array)}개 | "
          f"0.1 m 이내 {100 * (array <= 0.1).mean():5.1f}% | "
          f"도달 불가 {int((~np.isfinite(array)).sum())}개"
          + (f" | 유한값 중앙 {np.median(finite):.3f} m 최대 {finite.max():.3f} m"
             if len(finite) else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()

    grouped = load_selection()
    shard = list(grouped)[args.shard_index]
    raw_shard, episodes = episodes_with_poses(shard, args.episodes)
    if not episodes:
        print(f"[verify] {Path(shard).name}: 자세가 기록된 시연이 없다")
        return 1

    glb = scene_path(episodes[0]["scene_id"])
    print(f"[verify] {glb.name}: 자세가 온전히 기록된 시연 {len(episodes)}개로 검사\n")
    start_pose_disagreement(episodes)

    sim = make_sim(glb)
    try:
        for rebuilt in (False, True):
            if rebuilt:
                recompute_navmesh(sim)
            else:
                sim.pathfinder.load_nav_mesh(str(glb).replace(".basis.glb", ".basis.navmesh"))
            print(f"\n=== 내비메시 {'재계산' if rebuilt else '배포본 그대로'}: "
                  f"{describe_navmesh(sim.pathfinder)} ===")

            print(" 한 스텝만 (기록된 직전 자세에서 출발)")
            for action, values in sorted(single_step_errors(sim, episodes).items()):
                report(action, values)

            print(" 궤적 전체 (처음 한 번만 자세를 주고 끝까지 재생)")
            report("start_position 에서 출발", trajectory_drift(sim, episodes, False))
            report("재생 첫 자세에서 출발", trajectory_drift(sim, episodes, True))

            print(" 목표 관찰 지점 도달 가능성 (재생과 무관하게, 알려진 정답 지점에서)")
            goal_distance_check(sim, episodes, raw_shard)
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
