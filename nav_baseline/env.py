"""Habitat ObjectNav environment wrapper for the nav_baseline navigator.

This module owns everything that touches the simulator so that data collection,
training, and evaluation share one definition of the task:

- episode loading (start pose + goal viewpoints) from the raw HM3D ObjectNav shards,
- the observation the policy consumes (downsampled depth + a goal-relative vector),
- the shortest-path expert used for behavior cloning and DAgger labels,
- the success criterion (geodesic distance to a goal viewpoint at STOP time).

Action ids follow the HM3D ObjectNav convention used throughout the repo:
``STOP=0``, ``MOVE_FORWARD=1``, ``TURN_LEFT=2``, ``TURN_RIGHT=3``.
"""
from __future__ import annotations

import glob
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/data/topovlm")
HABITAT_ROOT = DATA_ROOT / "habitat"
SCENE_ROOT = HABITAT_ROOT / "scene_datasets" / "hm3d_v0.2"
SCENE_DATASET_CONFIG = SCENE_ROOT / "hm3d_annotated_basis.scene_dataset_config.json"
OBJECTNAV_ROOT = HABITAT_ROOT / "datasets" / "objectnav" / "hm3d" / "v2" / "objectnav_hm3d_v2"

FORWARD_M = 0.25
TURN_DEG = 30.0
AGENT_HEIGHT = 0.88
AGENT_RADIUS = 0.18
ACTION_NAME = {1: "move_forward", 2: "turn_left", 3: "turn_right"}
NUM_ACTIONS = 4

DEPTH_SIZE = 64          # depth is downsampled to DEPTH_SIZE x DEPTH_SIZE before use
DEPTH_MAX_M = 10.0       # depth values are clipped and normalised by this range
SUCCESS_DISTANCE_M = 1.0  # geodesic distance to a goal viewpoint that counts as success


@dataclass
class Episode:
    """One navigation problem: where the agent starts and which viewpoints count as success."""

    episode_id: str
    scene_key: str
    object_category: str
    start_position: np.ndarray
    start_rotation: np.ndarray
    goal_positions: list[np.ndarray] = field(default_factory=list)


def scene_keys_with_episodes(split: str = "train") -> list[str]:
    """Scene directories that have both a mesh and ObjectNav episodes (145 of 800 in HM3D)."""
    content = {Path(p).name.split(".")[0]
               for p in glob.glob(str(OBJECTNAV_ROOT / split / "content" / "*.json.gz"))}
    return [p.name for p in sorted((SCENE_ROOT / split).iterdir())
            if p.is_dir() and p.name.split("-", 1)[-1] in content]


def scene_glb_path(scene_key: str, split: str = "train") -> str:
    name = scene_key.split("-", 1)[-1]
    return str(SCENE_ROOT / split / scene_key / f"{name}.basis.glb")


def open_scene(scene_key: str, split: str = "train", *, with_rgb: bool = False,
               resolution: int = 256, use_cache: bool = True):
    """Open a scene, serving its files from the node's local disk when one is available.

    The shared filesystem is read over the network by every node, so the mesh is mirrored
    locally on first use. See ``nav_baseline.scene_cache`` for the measurement that decides
    whether this is worth doing.
    """
    from nav_baseline.scene_cache import scene_paths

    mesh, config = scene_paths(scene_key, split, use_cache=use_cache)
    return make_sim(mesh, with_rgb=with_rgb, resolution=resolution,
                    scene_dataset_config=config)


def load_episodes(scene_key: str, split: str = "train", limit: int | None = None,
                  object_category: str | None = None) -> list[Episode]:
    """Read episodes for one scene, resolving each episode's goal viewpoints.

    HM3D ObjectNav stores goals at the dataset level under ``goals_by_category`` keyed by
    ``"<scene>.basis.glb_<category>"``; every goal carries a list of ``view_points`` whose
    ``agent_state.position`` is a navigable stand-point that sees the object.
    """
    name = scene_key.split("-", 1)[-1]
    shard = OBJECTNAV_ROOT / split / "content" / f"{name}.json.gz"
    if not shard.exists():
        return []
    with gzip.open(shard, "rt") as handle:
        payload = json.load(handle)

    viewpoints_by_category: dict[str, list[np.ndarray]] = {}
    for key, goals in payload.get("goals_by_category", {}).items():
        category = key.split(".basis.glb_")[-1]
        points = []
        for goal in goals:
            for view_point in goal.get("view_points", []) or []:
                position = view_point.get("agent_state", {}).get("position")
                if position is not None:
                    points.append(np.asarray(position, dtype=np.float32))
        viewpoints_by_category[category] = points

    # Episodes in a shard are grouped by target object, so taking a prefix would return a
    # single category. Filter first, then truncate, so that `limit` samples the requested
    # object rather than whatever happens to come first in the file.
    episodes = []
    raw = payload["episodes"]
    if object_category is not None:
        raw = [entry for entry in raw if entry["object_category"] == object_category]
    for entry in (raw[:limit] if limit else raw):
        category = entry["object_category"]
        episodes.append(Episode(
            episode_id=str(entry["episode_id"]),
            scene_key=scene_key,
            object_category=category,
            start_position=np.asarray(entry["start_position"], dtype=np.float32),
            start_rotation=np.asarray(entry["start_rotation"], dtype=np.float32),
            goal_positions=viewpoints_by_category.get(category, []),
        ))
    return episodes


OBJECT_CATEGORIES = ("toilet", "tv_monitor", "bed", "sofa", "plant", "chair")


def load_balanced_episodes(scene_key: str, split: str = "train", limit: int | None = None,
                           categories=OBJECT_CATEGORIES) -> list[Episode]:
    """Take episodes round-robin across target objects instead of in shard order.

    Episodes are stored grouped by target object and the groups are heavily skewed, so
    reading a prefix returns almost only chairs. Chairs are also a poor goal for studying
    routes because a building contains many of them: a detour can end at a different chair
    than the shortest path did, which changes the problem rather than the route. Objects
    such as toilets occupy a single room, so the detour and the shortest path necessarily
    share a destination. Round-robin sampling keeps every object represented and puts the
    concentrated ones first.
    """
    per_category = {category: load_episodes(scene_key, split, object_category=category)
                    for category in categories}
    ordered, index = [], 0
    while limit is None or len(ordered) < limit:
        emitted = False
        for category in categories:
            episodes = per_category.get(category) or []
            if index < len(episodes):
                ordered.append(episodes[index])
                emitted = True
                if limit is not None and len(ordered) >= limit:
                    break
        if not emitted:
            break
        index += 1
    return ordered


def make_sim(scene_glb: str, *, with_rgb: bool = False, resolution: int = 256,
             scene_dataset_config: str | None = None):
    """Open a simulator with a depth sensor (and optionally RGB, for future encoders)."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.scene_dataset_config_file = str(scene_dataset_config or SCENE_DATASET_CONFIG)
    backend.enable_physics = False

    sensors = []
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [resolution, resolution]
    depth.position = [0.0, AGENT_HEIGHT, 0.0]
    sensors.append(depth)
    if with_rgb:
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [resolution, resolution]
        rgb.position = [0.0, AGENT_HEIGHT, 0.0]
        sensors.append(rgb)

    agent = habitat_sim.AgentConfiguration()
    agent.height = AGENT_HEIGHT
    agent.radius = AGENT_RADIUS
    agent.sensor_specifications = sensors
    agent.action_space = {
        "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def reset_to(sim, episode: Episode):
    """Place the agent at the episode start pose and return the first observations."""
    from habitat_sim.utils.common import quat_from_coeffs

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = episode.start_position
    state.rotation = quat_from_coeffs(episode.start_rotation)
    agent.set_state(state)
    return sim.get_sensor_observations()


def geodesic_to_goals(pathfinder, position, goal_positions) -> float:
    """Shortest navigable distance from ``position`` to the closest goal viewpoint."""
    import habitat_sim

    best = float("inf")
    for goal in goal_positions:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(position, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if pathfinder.find_path(path) and np.isfinite(path.geodesic_distance):
            best = min(best, float(path.geodesic_distance))
    return best


def nearest_goal(pathfinder, position, goal_positions):
    """The goal viewpoint with the smallest geodesic distance, and that distance."""
    import habitat_sim

    best_goal, best_distance = None, float("inf")
    for goal in goal_positions:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(position, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if pathfinder.find_path(path) and path.geodesic_distance < best_distance:
            best_distance = float(path.geodesic_distance)
            best_goal = np.asarray(goal, dtype=np.float32)
    return best_goal, best_distance


def goal_vector(sim, episode: Episode) -> np.ndarray:
    """Goal-relative observation: (normalised geodesic distance, sin bearing, cos bearing).

    The bearing is the straight-line direction to the nearest goal viewpoint expressed in the
    agent's frame (Habitat agents face -Z, with +X to the right). This is the standard
    "GPS+compass" goal signal; it says where the goal is but not how to get around walls,
    which is what the depth observation supplies.
    """
    from habitat_sim.utils.common import quat_rotate_vector

    state = sim.get_agent(0).get_state()
    goal, distance = nearest_goal(sim.pathfinder, state.position, episode.goal_positions)
    if goal is None:
        return np.zeros(3, dtype=np.float32)
    world_delta = goal - np.asarray(state.position, dtype=np.float32)
    local = quat_rotate_vector(state.rotation.inverse(), world_delta)
    bearing = float(np.arctan2(-local[0], -local[2]))
    return np.asarray([min(distance, 30.0) / 30.0, np.sin(bearing), np.cos(bearing)],
                      dtype=np.float32)


def depth_observation(observations) -> np.ndarray:
    """Downsample the depth image to DEPTH_SIZE x DEPTH_SIZE and normalise it to [0, 1]."""
    depth = np.asarray(observations["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    rows = np.linspace(0, depth.shape[0] - 1, DEPTH_SIZE).astype(np.int32)
    cols = np.linspace(0, depth.shape[1] - 1, DEPTH_SIZE).astype(np.int32)
    small = depth[np.ix_(rows, cols)]
    return np.clip(small, 0.0, DEPTH_MAX_M) / DEPTH_MAX_M


def build_expert(sim, goal_radius: float = SUCCESS_DISTANCE_M):
    """Greedy geodesic follower that produces the shortest-path action toward a goal."""
    import habitat_sim

    return habitat_sim.GreedyGeodesicFollower(
        sim.pathfinder, sim.get_agent(0), goal_radius=goal_radius,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right", stop_key="stop")


def expert_action_to(expert, point) -> int:
    """Expert action from the agent's current pose toward an arbitrary navigable point."""
    try:
        action = expert.next_action_along(np.asarray(point, dtype=np.float32))
    except Exception:
        return 0
    if action is None or action == "stop":
        return 0
    return {"move_forward": 1, "turn_left": 2, "turn_right": 3}[action]


def expert_action(expert, sim, episode: Episode) -> int:
    """Ask the expert what to do from the agent's current pose (this is always the label).

    The label is defined relative to the FINAL goal even when the agent is being driven
    somewhere else, so that detour and recovery states are annotated with the optimal way
    back rather than with the detour itself.
    """
    goal, _ = nearest_goal(sim.pathfinder, sim.get_agent(0).get_state().position,
                           episode.goal_positions)
    if goal is None:
        return 0
    return expert_action_to(expert, goal)


def sample_detour_waypoint(pathfinder, start, goal_positions, rng, *,
                           min_m: float = 3.0, max_m: float = 10.0, tries: int = 25):
    """A navigable point that turns the route into a genuine detour.

    Walking start -> waypoint -> goal should visit places the shortest path never touches,
    so the waypoint must be reachable, at least ``min_m`` away from the start, and make the
    total route meaningfully longer than going straight to the goal.
    """
    direct = geodesic_to_goals(pathfinder, start, goal_positions)
    if not np.isfinite(direct):
        return None
    best = None
    for _ in range(tries):
        point = pathfinder.get_random_navigable_point_near(start, max_m)
        if point is None or not np.all(np.isfinite(point)):
            continue
        to_waypoint = geodesic_to_goals(pathfinder, start, [point])
        if not np.isfinite(to_waypoint) or to_waypoint < min_m:
            continue
        from_waypoint = geodesic_to_goals(pathfinder, point, goal_positions)
        if not np.isfinite(from_waypoint):
            continue
        if to_waypoint + from_waypoint > direct * 1.25:   # a real detour, not a shortcut
            return np.asarray(point, dtype=np.float32)
        best = np.asarray(point, dtype=np.float32)
    return best
