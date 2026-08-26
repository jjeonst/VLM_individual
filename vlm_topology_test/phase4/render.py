"""Replay the selected human demonstrations into images, actions and poses.

The demonstrations chosen by ``subsample.py`` contain no images. Each one is a starting pose
and a list of the actions a person took. Habitat's dynamics are deterministic -- the paper
states this outright in Appendix C.1 -- so placing the agent at that starting pose and feeding
it the same actions reproduces exactly what the person saw. This module performs that replay
and writes out the three things later steps need:

- **RGB**, 480x640x3 bytes per step, the observation the policy is trained to act on,
- **actions**, one integer per step, the label the policy is trained to predict,
- **pose**, the agent's horizontal position and heading at each step, which the paper's
  observation space includes alongside the image (Appendix C.1, items 2 and 3).

The pose is written here rather than recovered later because the replay already has to run;
recovering it afterwards would mean opening every scene a second time for no gain.

**What is recorded when.** At each step the observation is captured *before* the action is
executed, because that is the situation the policy will face: it sees a view and must choose
what to do from it. The number of frames therefore equals the number of actions, and the last
frame is the view from which the person decided to stop.

**Two edits to the recorded action list**, both established by inspecting the whole dataset
and both recorded in ``IMPLEMENTATION.md``:

1. Every episode's recorded list *begins* with a STOP that marks "no action taken yet" rather
   than a decision to stop -- the total STOP count is exactly twice the episode count, one at
   each end. Keeping it would teach the policy to stop at once, so the leading entry is
   dropped.
2. The people who produced these demonstrations could tilt the camera up and down, and 27.7%
   of episodes contain such steps. The paper removes the pitch-changing actions from the
   action space (Appendix C.1), so those steps are removed here and everything is rendered at
   a level camera. This costs nothing in path fidelity: tilting changes the camera angle but
   not the agent's position or heading, so the route the person walked is preserved exactly.

**Scenes are opened once each.** Loading a building's mesh takes far longer than replaying a
demonstration through it, so episodes are grouped by scene, and the intended way to run this
across the cluster is one array task per scene.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HABITAT_ROOT = Path("/data/topovlm/habitat")
SCENE_ROOT = HABITAT_ROOT / "scene_datasets"
SCENE_DATASET_CONFIG = SCENE_ROOT / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"
SELECTION_FILE = (HABITAT_ROOT / "episode_selections" / "pr2l_habitat_web_hd"
                  / "train_every_tenth.jsonl")

DATASET_NAME = "pr2l_habitat_web_hd"
SPLIT = "train"
OUT_ROOT = HABITAT_ROOT

# Agent, camera and dynamics. Appendix C.1 states that the spaces and agent specification are
# "largely the same as the defaults provided by Habitat, as specified in the HM3D ObjectNav
# configuration file", and gives only the turn angle (30 degrees) and the image size in the
# text. The remaining values are therefore read from that configuration file itself
# (habitat-lab 0.3.3, benchmark/nav/objectnav/objectnav_hm3d.yaml, with unset fields falling
# back to config/default_structured_configs.py) rather than from any other implementation.
RGB_HEIGHT, RGB_WIDTH = 480, 640     # objectnav_hm3d.yaml: height 480, width 640
HFOV_DEG = 79.0                      # objectnav_hm3d.yaml: hfov 79
CAMERA_HEIGHT = 0.88                 # objectnav_hm3d.yaml: rgb_sensor position [0, 0.88, 0]
AGENT_HEIGHT = 0.88                  # objectnav_hm3d.yaml: height 0.88
AGENT_RADIUS = 0.18                  # objectnav_hm3d.yaml: radius 0.18
TURN_DEG = 30.0                      # objectnav_hm3d.yaml: turn_angle 30 (also Appendix C.1)
FORWARD_M = 0.25                     # default_structured_configs.py: forward_step_size 0.25

# ObjectNav turns *off* Habitat's default wall sliding. This matters for replay: with sliding
# on, an agent that walks into a wall slides along it instead of stopping, so a demonstration
# in which the person bumped a wall would put the agent somewhere the person never stood, and
# every frame after that point would be wrong.
ALLOW_SLIDING = False                # objectnav_hm3d.yaml: allow_sliding False
MAX_CLIMB = 0.2                      # default_structured_configs.py: max_climb
MAX_SLOPE = 45.0                     # default_structured_configs.py: max_slope

# An episode succeeds when the agent stops within this navigable distance of a viewpoint from
# which the target object is visible.
SUCCESS_DISTANCE_M = 0.1             # objectnav.yaml: success_distance 0.1, distance_to VIEW_POINTS

# Action ids follow the convention used across this repository.
STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
ACTION_ID = {"STOP": STOP, "MOVE_FORWARD": MOVE_FORWARD,
             "TURN_LEFT": TURN_LEFT, "TURN_RIGHT": TURN_RIGHT}
SIM_ACTION = {MOVE_FORWARD: "move_forward", TURN_LEFT: "turn_left", TURN_RIGHT: "turn_right"}
PITCH_ACTIONS = {"LOOK_UP", "LOOK_DOWN"}


def payload_id(episode_id: str) -> str:
    """A filename-safe form of an episode id, which contains a colon."""
    return re.sub(r"[^0-9A-Za-z_.-]", "_", episode_id)


def scene_path(scene_id: str) -> Path:
    """Locate the mesh a demonstration refers to, e.g. 'hm3d/train/00744-XXXX/XXXX.basis.glb'."""
    return SCENE_ROOT / scene_id


def load_selection() -> dict[str, list[dict]]:
    """Group the selected demonstrations by the scene file they live in."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in SELECTION_FILE.open():
        record = json.loads(line)
        grouped[record["shard_path"]].append(record)
    return dict(sorted(grouped.items()))


def clean_replay(replay: list[dict]) -> list[tuple[int, np.ndarray | None]]:
    """Apply the two edits described in the module docstring, keeping any recorded pose.

    Roughly three steps in ten still carry the position the agent actually held at that moment.
    Those are useless as a source of poses -- too sparse and too irregular to replace the
    replay -- but they are exactly what is needed to check that the replay is faithful, so they
    are carried alongside each action rather than discarded.

    Dropping the tilt steps does not break the pairing: tilting moves the camera and not the
    agent, so every later recorded position still belongs to the step it is paired with.
    """
    entries = replay[1:] if len(replay) > 1 and str(replay[0]["action"]) == "STOP" else replay
    cleaned = []
    for step in entries:
        name = str(step["action"])
        if name in PITCH_ACTIONS:
            continue
        state = step.get("agent_state")
        recorded = (np.asarray(state["position"], dtype=np.float32)
                    if isinstance(state, dict) and "position" in state else None)
        cleaned.append((ACTION_ID[name], recorded))
    return cleaned


def make_sim(glb: Path):
    """Open a simulator matching the paper's agent and camera."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.scene_dataset_config_file = str(SCENE_DATASET_CONFIG)
    backend.enable_physics = False
    backend.allow_sliding = ALLOW_SLIDING

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [RGB_HEIGHT, RGB_WIDTH]
    rgb.hfov = HFOV_DEG
    rgb.position = [0.0, CAMERA_HEIGHT, 0.0]

    agent = habitat_sim.AgentConfiguration()
    agent.height = AGENT_HEIGHT
    agent.radius = AGENT_RADIUS
    agent.sensor_specifications = [rgb]
    agent.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    recompute_navmesh(sim)
    return sim


def recompute_navmesh(sim) -> None:
    """Rebuild the walkable surface for the agent this task actually uses.

    The mesh files shipped with HM3D describe where an agent of radius 0.10 m and height 1.5 m
    may walk, but ObjectNav's agent is wider and shorter, at radius 0.18 m and height 0.88 m.
    A wider agent cannot fit through gaps a narrow one can, so replaying against the shipped
    surface lets the agent squeeze past obstacles the person was stopped by, and from that
    point on the replay is walking a different route.

    habitat-lab rebuilds the surface for exactly this reason whenever the configured agent
    differs from the one the file was built for; see the block guarded by
    ``default_agent_navmesh`` in ``sims/habitat_simulator/habitat_simulator.py``. Measured on
    40 demonstrations, doing the same here moves whole-trajectory agreement with the recorded
    poses from a median error of 0.173 m to 0.0000 m.
    """
    import habitat_sim

    settings = habitat_sim.nav.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = AGENT_RADIUS
    settings.agent_height = AGENT_HEIGHT
    settings.agent_max_climb = MAX_CLIMB
    settings.agent_max_slope = MAX_SLOPE
    settings.include_static_objects = False
    if not sim.recompute_navmesh(sim.pathfinder, settings):
        raise RuntimeError("navmesh recomputation failed")


def agent_pose(sim) -> tuple[float, float, float, float]:
    """Position and heading of the agent, in world coordinates, as (x, y, z, yaw).

    The paper's observation space gives the policy only the two horizontal coordinates and the
    heading (Appendix C.1, items 2 and 3); Habitat's world is y-up, so those are x, z and the
    rotation about the vertical axis. The height is stored anyway because it costs one float
    and is needed to measure distances to goal viewpoints, which are given in three dimensions.

    Absolute values are stored rather than the episode-relative ones Habitat's own GPS and
    compass sensors report. The relative form can be derived from these together with the start
    pose but not the other way round, so storing the absolute form leaves the choice open; which
    form the policy actually receives is settled in the training step.
    """
    from habitat_sim.utils.common import quat_to_angle_axis

    state = sim.get_agent(0).get_state()
    angle, axis = quat_to_angle_axis(state.rotation)
    yaw = float(angle * (1.0 if axis[1] >= 0 else -1.0))
    return (float(state.position[0]), float(state.position[1]),
            float(state.position[2]), yaw)


def goal_viewpoints(raw_shard: dict, scene_id: str, category: str) -> np.ndarray:
    """Every pose from which the target object counts as found, for one episode.

    Habitat judges an ObjectNav episode successful when the agent stops within 0.1 m of one of
    these viewpoints (Appendix C.1 defers to the Habitat defaults for this). They are listed
    per scene and category rather than per episode, so they are read once per scene.
    """
    key = f"{Path(scene_id).name}_{category}"
    points = [np.asarray(view["agent_state"]["position"], dtype=np.float32)
              for goal in raw_shard["goals_by_category"].get(key, [])
              for view in goal.get("view_points", [])]
    return np.stack(points, axis=0) if points else np.empty((0, 3), dtype=np.float32)


def geodesic_to_viewpoints(pathfinder, position, viewpoints: np.ndarray) -> float:
    """Shortest navigable distance from a position to the closest goal viewpoint.

    A scene lists hundreds of viewpoints per target object. Habitat can search to all of them
    in one query, which is both faster and safer than shortlisting by straight-line distance:
    the viewpoint that looks nearest across a wall can be the farthest to walk to.
    """
    import habitat_sim

    if len(viewpoints) == 0:
        return float("inf")
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(position, dtype=np.float32)
    path.requested_ends = [np.asarray(point, dtype=np.float32) for point in viewpoints]
    if pathfinder.find_path(path) and np.isfinite(path.geodesic_distance):
        return float(path.geodesic_distance)
    return float("inf")


def replay_episode(sim, record: dict,
                   steps: list[tuple[int, np.ndarray | None]]) -> dict | None:
    """Walk one demonstration, capturing the view and pose before each action.

    Wherever the source data still holds the position the agent actually occupied, the
    simulated position is compared against it. Habitat is deterministic, so a faithful replay
    should agree to within floating-point noise; a systematic gap means the replay has drifted
    away from the path the person walked.
    """
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs

    if not steps:
        return None

    state = habitat_sim.AgentState()
    state.position = np.asarray(record["start_position"], dtype=np.float32)
    state.rotation = quat_from_coeffs(np.asarray(record["start_rotation"], dtype=np.float32))
    sim.get_agent(0).set_state(state, reset_sensors=True)

    frames = np.empty((len(steps), RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
    poses = np.empty((len(steps), 4), dtype=np.float32)
    drifts = []
    observations = sim.get_sensor_observations()
    for index, (action, recorded) in enumerate(steps):
        frames[index] = np.asarray(observations["rgb"])[..., :3]
        poses[index] = agent_pose(sim)
        if recorded is not None:
            drifts.append(float(np.linalg.norm(poses[index][:3] - recorded)))
        if action == STOP:
            break
        observations = sim.step(SIM_ACTION[action])

    used = index + 1
    return {"rgb": frames[:used], "pose": poses[:used],
            "actions": np.asarray([a for a, _ in steps[:used]], dtype=np.int64),
            "drifts": drifts,
            "final_position": np.asarray(sim.get_agent(0).get_state().position,
                                         dtype=np.float32)}


def write_episode(record: dict, payload: dict, final_geodesic: float) -> dict:
    """Save one demonstration's three arrays and return its manifest line."""
    name = payload_id(record["episode_id"])
    relative = {}
    for kind in ("rgb", "actions", "pose"):
        directory = OUT_ROOT / kind / DATASET_NAME / SPLIT
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{name}.npy", payload[kind])
        relative[f"{kind}_path"] = str(Path(kind) / DATASET_NAME / SPLIT / f"{name}.npy")
    return {
        "episode_id": name,
        "source_trajectory_id": record["episode_id"],
        "split": SPLIT,
        "scene_id": record["scene_id"],
        "scene_key": record["scene_key"],
        "object_category": record["object_category"],
        "goal_text": record["object_category"],
        "steps": int(len(payload["actions"])),
        "final_geodesic": round(final_geodesic, 4) if np.isfinite(final_geodesic) else None,
        "reached": bool(final_geodesic <= SUCCESS_DISTANCE_M),
        "source_dataset": "habitat_web_hd",
        **relative,
    }


def render_shard(shard: str, records: list[dict], limit: int | None = None) -> dict:
    """Replay every selected demonstration in one scene."""
    wanted = {r["episode_id"]: r for r in records}
    raw = json.loads(gzip.open(shard, "rt").read())

    episodes = []
    for episode in raw["episodes"]:
        if episode["episode_id"] not in wanted:
            continue
        steps = clean_replay(episode.get("reference_replay") or [])
        episodes.append((wanted[episode["episode_id"]], episode, steps))
    if limit is not None:
        episodes = episodes[:limit]
    if not episodes:
        return {"shard": shard, "written": 0, "records": []}

    glb = scene_path(episodes[0][1]["scene_id"])
    if not glb.exists():
        raise FileNotFoundError(glb)

    started = time.time()
    sim = make_sim(glb)
    load_seconds = time.time() - started
    print(f"[render] {glb.name}: scene loaded in {load_seconds:.1f}s, "
          f"{len(episodes)} demonstrations", flush=True)

    viewpoint_cache: dict[str, np.ndarray] = {}
    manifest, frames_written, skipped, reached = [], 0, 0, 0
    all_drifts: list[float] = []
    try:
        for selection, episode, steps in episodes:
            record = dict(selection)
            record["start_position"] = episode["start_position"]
            record["start_rotation"] = episode["start_rotation"]
            payload = replay_episode(sim, record, steps)
            if payload is None:
                skipped += 1
                continue
            category = record["object_category"]
            if category not in viewpoint_cache:
                viewpoint_cache[category] = goal_viewpoints(raw, record["scene_id"], category)
            distance = geodesic_to_viewpoints(
                sim.pathfinder, payload["final_position"], viewpoint_cache[category])
            manifest.append(write_episode(record, payload, distance))
            frames_written += len(payload["actions"])
            reached += int(distance <= SUCCESS_DISTANCE_M)
            all_drifts.extend(payload["drifts"])
    finally:
        sim.close()

    elapsed = time.time() - started
    render_seconds = elapsed - load_seconds
    rate = reached / max(len(manifest), 1)
    drift = describe_drift(all_drifts)
    print(f"[render] {glb.name}: {len(manifest)} demonstrations, {frames_written} frames, "
          f"{skipped} skipped, {render_seconds:.1f}s replay "
          f"({frames_written / max(render_seconds, 1e-6):.1f} frames/s), "
          f"목표 도달 {reached}/{len(manifest)} ({100 * rate:.0f}%), {drift}", flush=True)
    return {"shard": shard, "scene": glb.name, "written": len(manifest),
            "frames": frames_written, "skipped": skipped, "reached": reached,
            "drifts": all_drifts,
            "load_seconds": round(load_seconds, 1),
            "render_seconds": round(render_seconds, 1), "records": manifest}


def describe_drift(drifts: list[float]) -> str:
    """Summarise how far the replayed positions sit from the recorded ones."""
    if not drifts:
        return "기록된 자세 없음"
    values = np.asarray(drifts)
    return (f"자세 오차 {len(values)}개 비교: 중앙 {np.median(values):.3f} m, "
            f"평균 {values.mean():.3f} m, 최대 {values.max():.3f} m, "
            f"0.05 m 이내 {100 * (values <= 0.05).mean():.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int,
                        help="index into the sorted scene list, for slurm array tasks")
    parser.add_argument("--scenes", type=int, default=None,
                        help="render only the first N scenes (pilot runs)")
    parser.add_argument("--episodes-per-scene", type=int, default=None,
                        help="cap demonstrations per scene (pilot runs)")
    parser.add_argument("--manifest-dir", type=Path,
                        default=HABITAT_ROOT / "episodes" / DATASET_NAME / SPLIT / "shards")
    args = parser.parse_args()

    grouped = load_selection()
    shards = list(grouped)
    if args.shard_index is not None:
        if not 0 <= args.shard_index < len(shards):
            print(f"[error] shard index {args.shard_index} outside 0..{len(shards) - 1}",
                  file=sys.stderr)
            return 1
        shards = [shards[args.shard_index]]
    elif args.scenes is not None:
        shards = shards[:args.scenes]

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    totals = {"scenes": 0, "written": 0, "frames": 0, "skipped": 0, "reached": 0,
              "render_seconds": 0.0}
    drifts: list[float] = []
    for shard in shards:
        result = render_shard(shard, grouped[shard], limit=args.episodes_per_scene)
        if not result["records"]:
            continue
        target = args.manifest_dir / f"{Path(shard).name.replace('.json.gz', '')}.jsonl"
        with target.open("w") as handle:
            for record in result["records"]:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        for key in ("written", "frames", "skipped", "reached", "render_seconds"):
            totals[key] += result[key]
        totals["scenes"] += 1
        drifts.extend(result["drifts"])

    rate = totals["frames"] / max(totals["render_seconds"], 1e-6)
    reach_rate = totals["reached"] / max(totals["written"], 1)
    print(f"\n[render] {totals['scenes']} scenes, {totals['written']} demonstrations, "
          f"{totals['frames']} frames, {totals['skipped']} skipped")
    print(f"[render] {rate:.1f} frames/s, "
          f"{totals['frames'] * 900 / 1024 / 1024:.1f} GB written")
    print(f"[render] 재생 검증 1 — 기록된 자세와의 비교: {describe_drift(drifts)}")
    print(f"[render] 재생 검증 2 — 목표 관찰 지점 {SUCCESS_DISTANCE_M} m 이내에서 종료: "
          f"{totals['reached']}/{totals['written']} ({100 * reach_rate:.1f}%)")
    if drifts and float(np.median(drifts)) > 0.05:
        print("[render] WARNING: 재생한 위치가 기록된 위치와 벌어져 있다. 재생 절차가 "
              "원래 궤적을 따라가지 못하고 있다는 직접적인 증거다.")
    elif reach_rate < 0.8:
        print("[render] WARNING: 자세는 맞는데 목표 도달률이 낮다. 재생이 아니라 성공 "
              "판정 쪽(목표 관찰 지점 조회)을 의심해야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
