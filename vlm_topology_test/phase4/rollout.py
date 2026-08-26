"""Expert rollouts for the representation figure (paper Appendix H.2, Figure 8).

Figure 8 asks a different question from the rest of the reproduction. Everything else measures
whether the policy acts well; this asks whether the *representations* are shaped in a way that
makes acting well possible -- whether states the VLM would describe as the same kind of room
land near each other, and whether the good states separate from the bad ones. It is the paper's
own evidence for why prompting helps, so it is worth having alongside a success rate.

The figure's caption fixes what has to be generated:

    PCA of PR2L (above) and image encoder (below) representations of observations from thirty
    episode rollouts of expert policies in all Habitat tasks. The points' colors correspond to
    their value under Habitat's built-in oracle shortest path follower (a near-optimal policy).
    More yellow is better. Boxes correspond to points the VLM has labeled as a given household
    room, in response to the task prompt of "What room is this?"

So: not the human demonstrations the policy trains on, but **rollouts of Habitat's built-in
shortest path follower**, which is what this module produces.

**Why the colour can be computed rather than learned.** The paper colours by the oracle's value,
and Appendix D defines the reward that oracle is optimal for: a terminal `+10 * SPL` plus, at
every move, "a shaping reward of the change in geodesic distance to the nearest goal object
instance". Both terms are functions of the remaining geodesic distance, so recording that
distance at every step is enough to reconstruct the colour afterwards -- no value function has
to be fitted.

It is *not* enough to assume the two agree in sign. An earlier version of this note claimed the
shaping term makes value a monotone decreasing function of remaining distance; that is wrong.
Potential-based shaping with a potential of -d gives `V_shaped = V_task + d`, which *increases*
with distance, because the dense shaping reward is collected in proportion to how far the agent
had to travel. Measured on these rollouts with the paper's own gamma of 0.99, the literal
discounted return correlates with remaining distance at +0.685 -- the opposite of what Figure
8's caption describes. `PLAN.md` section 6.3 carries the numbers and the resulting decision to
draw both colourings.

**Why the follower's goal radius is not the success radius.** Success is scored at 0.1 m, but
the agent moves in 0.25 m steps, so a follower asked to arrive within 0.1 m can circle a goal it
can never land on precisely, burning hundreds of steps at a nearly constant distance. Those
frames are near-duplicates and would flood the figure's "close to goal" colour bin. The follower
is therefore given a radius of one step, and the rollout stops as soon as the agent is actually
within the success distance or stops making progress.

**Scene choice follows Appendix D**: households holding at least one instance of each of the
three target objects, so that the three columns of the figure are drawn from the same buildings
and a difference between them cannot be a difference between houses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import (SCENE_CONFIG_NAME, SCENE_ROOT, cache_scene, load_episodes, make_sim)
from render import agent_pose, geodesic_to_viewpoints

HABITAT_ROOT = Path("/data/topovlm/habitat")
OUT_ROOT = HABITAT_ROOT / "figure_rollouts"

# Appendix D: the simplified setting keeps toilet, bed and sofa, dropping plants and televisions
# ("numerous unlabeled instances") and chairs ("significantly more common ... much shorter
# episodes"). Figure 8's three columns are these.
FIGURE_OBJECTS = ("toilet", "bed", "sofa")
MAX_STEPS = 500                      # objectnav_hm3d.yaml, as everywhere else
SUCCESS_DISTANCE = 0.1               # objectnav.yaml, the distance the rest of the pipeline scores
FOLLOWER_RADIUS = 0.25               # one forward step; see the note on goal radius above
STALL_STEPS = 30                     # give up once this many steps pass without getting closer
STALL_MARGIN = 0.05                  # ... by at least this much, in metres

# A near-optimal follower should walk about as far as the geodesic distance it started from.
# Much further means it did not go where the recorded distance was measured, and the colour on
# such a trajectory would be describing a goal the agent was not walking to.
MIN_EFFICIENCY = 0.5


def episode_fingerprint(episode: dict) -> str:
    """A short name for an episode, derived from what makes it that episode."""
    payload = (np.asarray(episode["start_position"], dtype=np.float32).tobytes()
               + np.asarray(episode["start_rotation"], dtype=np.float32).tobytes())
    return f"{episode['episode_id']}-{hashlib.md5(payload).hexdigest()[:6]}"


def scenes_with_all_objects(episodes: list[dict]) -> set[str]:
    """Buildings holding every one of the figure's target objects (Appendix D's criterion)."""
    present = defaultdict(set)
    for episode in episodes:
        present[episode["scene_id"]].add(episode["object_category"])
    return {scene for scene, objects in present.items()
            if set(FIGURE_OBJECTS).issubset(objects)}


def nearest_viewpoint(pathfinder, position: np.ndarray, viewpoints: np.ndarray,
                      candidates: int = 20):
    """The goal the follower should walk to: the reachable viewpoint that is closest.

    A scene lists hundreds of viewpoints per object, and asking for a path to each one is slow.
    The shortlist is by straight-line distance, which can be wrong on its own -- the viewpoint
    just through a wall is the farthest to walk to -- so the shortlist is only used to narrow
    the field, and the winner is chosen by an actual navigable path.
    """
    if len(viewpoints) == 0:
        return None, float("inf")
    order = np.argsort(np.linalg.norm(viewpoints - np.asarray(position), axis=1))
    best, best_distance = None, float("inf")
    for index in order[:candidates]:
        point = viewpoints[index]
        distance = geodesic_to_viewpoints(pathfinder, position, point[None, :])
        if np.isfinite(distance) and distance < best_distance:
            best, best_distance = point, float(distance)
    return best, best_distance


def run_episode(sim, episode: dict) -> dict | None:
    """Walk the shortest path follower to the goal, recording what it saw and how far it had left."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs

    state = habitat_sim.AgentState()
    state.position = np.asarray(episode["start_position"], dtype=np.float32)
    state.rotation = quat_from_coeffs(episode["start_rotation"])
    sim.get_agent(0).set_state(state, reset_sensors=True)

    goal, goal_distance = nearest_viewpoint(
        sim.pathfinder, state.position, episode["viewpoints"])
    if goal is None:
        return None

    follower = sim.make_greedy_follower(
        agent_id=0, goal_radius=FOLLOWER_RADIUS, stop_key=None,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right")

    frames, distances, poses = [], [], []
    best, stalled = float("inf"), 0
    for _ in range(MAX_STEPS):
        # Distance to the nearest of *all* view points, which is what habitat-lab's DistanceToGoal
        # reports and what the rest of this pipeline scores against. `goal_distance` above is to
        # the single point the follower walks to, and the two are recorded separately so a
        # trajectory that ends up at a different instance is visible rather than silently mixed in.
        distance = float(geodesic_to_viewpoints(sim.pathfinder, agent_pose(sim)[:3],
                                                episode["viewpoints"]))
        frames.append(np.asarray(sim.get_sensor_observations()["rgb"])[..., :3].copy())
        distances.append(distance)
        poses.append(np.asarray(agent_pose(sim), dtype=np.float32))

        if distance <= SUCCESS_DISTANCE:        # arrived by the criterion the pipeline scores
            break
        stalled = 0 if distance < best - STALL_MARGIN else stalled + 1
        best = min(best, distance)
        if stalled >= STALL_STEPS:              # circling a goal it cannot land on
            break

        try:
            action = follower.next_action_along(goal)
        except Exception:                       # the follower gives up when no path remains
            break
        if action is None:                      # the follower considers itself arrived
            break
        sim.step(action)

    if len(frames) < 5:
        return None
    walked = float(np.linalg.norm(np.diff(np.stack(poses)[:, :3], axis=0), axis=1).sum())
    start_distance = distances[0]
    return {"frames": np.stack(frames), "distances": np.array(distances, dtype=np.float32),
            "poses": np.stack(poses), "start_geodesic": start_distance,
            "goal_geodesic": goal_distance, "walked": walked,
            "efficiency": start_distance / max(start_distance, walked),
            "reached": bool(distances[-1] <= SUCCESS_DISTANCE)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--per-object", type=int, default=10,
                        help="episodes per target object; the paper uses thirty in total")
    parser.add_argument("--spare", type=int, default=8,
                        help="extra episodes attempted per object, to survive rejections")
    parser.add_argument("--per-scene", type=int, default=3,
                        help="cap on trajectories one building may contribute to one column")
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/jonghoon/hm3d"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes = load_episodes(args.split, objects=FIGURE_OBJECTS)
    usable = scenes_with_all_objects(episodes)
    print(f"[rollout] {args.split} 에피소드 {len(episodes):,} | "
          f"세 물체를 모두 가진 장면 {len(usable)}개", flush=True)
    if not usable:
        print("[rollout] 조건을 만족하는 장면이 없다")
        return 1

    rng = np.random.default_rng(args.seed)
    chosen_scenes = sorted(usable)
    picked = [chosen_scenes[i] for i in
              rng.choice(len(chosen_scenes), size=min(args.scenes, len(chosen_scenes)),
                         replace=False)]

    wanted = defaultdict(list)
    for episode in episodes:
        if episode["scene_id"] in picked:
            wanted[episode["object_category"]].append(episode)

    # Plan more than are wanted. Some episodes have no navigable path, and some rollouts are
    # discarded below for walking somewhere other than where their distance was measured; without
    # slack the figure quietly ends up with fewer than the thirty the paper drew.
    attempts = args.per_object + args.spare
    plan = defaultdict(list)
    for name in FIGURE_OBJECTS:
        group = wanted[name]
        if not group:
            print(f"[rollout] {name}: 고른 장면에 에피소드가 없다")
            return 1
        take = rng.choice(len(group), size=min(attempts, len(group)), replace=False)
        for index in sorted(take):
            plan[group[index]["scene_id"]].append(group[index])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stale = sorted(OUT_ROOT.glob("*.npz")) + sorted(OUT_ROOT.glob("manifest.jsonl"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"[rollout] 이전 산출물 {len(stale)}개 삭제 (전량 재생성한다)", flush=True)

    args.scratch.mkdir(parents=True, exist_ok=True)
    written, rejected, started = [], [], time.time()
    kept = defaultdict(int)

    per_scene: dict[tuple[str, str], int] = defaultdict(int)
    for scene, group in sorted(plan.items()):
        glb = cache_scene(scene, args.scratch)
        sim = make_sim(glb, SCENE_ROOT / SCENE_CONFIG_NAME)
        try:
            for episode in group:
                category = episode["object_category"]
                if kept[category] >= args.per_object:
                    continue
                # One building must not fill a column. In the first run seven of the ten `bed`
                # trajectories came from scene 00017, whose annotated bed view points look into a
                # bathroom -- the VLM answered "This is a bathroom" for every one of them, and it
                # was right. A column drawn mostly from one house describes that house, so the
                # figure's claim about representations could not be read off it either way.
                if per_scene[(scene, category)] >= args.per_scene:
                    continue
                result = run_episode(sim, episode)
                if result is None:
                    print(f"[rollout] {episode['episode_id']}: 경로 없음, 건너뜀", flush=True)
                    continue

                # Neither the episode id nor the scene is enough to name a trajectory. Ids
                # restart at 1 in every building, so `{object}_{id}` collides across scenes; and
                # the train split reuses ids *within* a building too -- two `toilet` episodes of
                # scene 00099 both call themselves 8, starting 17.6 m and 15.0 m from the goal.
                # A digest of the start pose settles it, because that is what actually
                # distinguishes two episodes with the same target in the same house.
                name = f"{category}_{Path(scene).parent.name}_{episode_fingerprint(episode)}"
                if result["efficiency"] < MIN_EFFICIENCY:
                    rejected.append((name, result["efficiency"]))
                    print(f"[rollout] {name}: 효율 {result['efficiency']:.2f} — 버림 "
                          f"({result['start_geodesic']:.1f}m 거리에 {result['walked']:.1f}m 이동)",
                          flush=True)
                    continue

                np.savez(OUT_ROOT / f"{name}.npz", frames=result["frames"],
                         distances=result["distances"], poses=result["poses"])
                kept[category] += 1
                per_scene[(scene, category)] += 1
                written.append({
                    "name": name, "episode_id": episode["episode_id"],
                    "object_category": category, "scene_id": scene,
                    "steps": int(len(result["distances"])),
                    "start_geodesic": round(result["start_geodesic"], 3),
                    "goal_geodesic": round(result["goal_geodesic"], 3),
                    "final_geodesic": round(float(result["distances"][-1]), 3),
                    "walked": round(result["walked"], 3),
                    "efficiency": round(result["efficiency"], 3),
                    "reached": result["reached"],
                    "rgb_path": str((OUT_ROOT / f"{name}.npz").relative_to(HABITAT_ROOT)),
                })
                print(f"[rollout] {name}: {written[-1]['steps']}스텝, "
                      f"{written[-1]['start_geodesic']:.1f}m → "
                      f"{written[-1]['final_geodesic']:.2f}m, "
                      f"효율 {written[-1]['efficiency']:.2f}, "
                      f"도달 {result['reached']}", flush=True)
        finally:
            sim.close()

    (OUT_ROOT / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in written))

    steps = [row["steps"] for row in written]
    reached = sum(row["reached"] for row in written)
    print(f"\n[rollout] 궤적 {len(written)} / 프레임 {sum(steps):,} | "
          f"평균 {np.mean(steps):.0f}스텝 | 목표 도달 {reached}/{len(written)} | "
          f"효율 중앙 {np.median([r['efficiency'] for r in written]):.2f} | "
          f"버린 궤적 {len(rejected)} | {time.time() - started:.0f}초")
    for category in FIGURE_OBJECTS:
        print(f"[rollout]   {category:8s} {kept[category]}궤적")
    print(f"[rollout] {OUT_ROOT / 'manifest.jsonl'} 저장")

    # The check that was missing last time. A manifest listing more trajectories than there are
    # files on disk is what a filename collision looks like from the outside, and nothing warned:
    # 28 rows pointed at 25 files and the run reported success.
    names = [row["name"] for row in written]
    files = {path.stem for path in OUT_ROOT.glob("*.npz")}
    if len(set(names)) != len(names) or set(names) != files:
        print(f"[rollout] 실패 — 매니페스트 {len(names)}행 / 고유 이름 {len(set(names))} / "
              f"파일 {len(files)}개가 어긋난다. 이름 충돌을 의심할 것.")
        return 1

    # The follower is near-optimal, so anything much below full arrival means the goal choice or
    # the navmesh is wrong, not that the task is hard. The test is against the radius the follower
    # was actually given, not against the 0.1 m the *policy* is scored at: a follower asked to
    # stop within one step of the goal has done its job at 0.2 m, and testing it at 0.1 m failed
    # eleven perfectly good trajectories on the first run.
    arrived = sum(row["final_geodesic"] <= FOLLOWER_RADIUS for row in written)
    print(f"[rollout] 추종기 도착 {arrived}/{len(written)} (반경 {FOLLOWER_RADIUS}m) | "
          f"그 중 {reached}개는 성공 판정 거리 {SUCCESS_DISTANCE}m 이내")
    if written and arrived / len(written) < 0.8:
        print("[rollout] 경고 — 최단경로 추종기가 목표에 못 닿는 비율이 높다. "
              "목표 지점 선택이나 내비메시를 의심할 것.")
        return 1
    if sum(kept.values()) < 3 * args.per_object:
        print(f"[rollout] 경고 — 목표 {3 * args.per_object}궤적 중 {sum(kept.values())}개만 남았다. "
              "--spare 를 늘리거나 --scenes 를 늘릴 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
