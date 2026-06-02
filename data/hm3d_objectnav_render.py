"""Render HM3D ObjectNav shortest-path expert episodes into RGB/action payloads."""

from __future__ import annotations

from contextlib import contextmanager
import json
import re
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from configs.schema import TopoVLMConfig
from data.habitat_manifest import (
    resolve_data_path,
    resolve_materialization_data_root,
    resolve_runtime_data_root,
)
from data.habitat_objectnav import (
    ObjectNavSelectionRecord,
    load_objectnav_selection_records,
    objectnav_source_trajectory_id,
    resolve_objectnav_scene_path,
)


class ObjectNavExpertEnv(Protocol):
    """Expose the Habitat Env methods needed to materialize ObjectNav expert episodes."""

    episodes: list[object]
    current_episode: object
    sim: object

    def reset(self) -> dict[str, object]:
        ...

    def step(self, action: int) -> dict[str, object]:
        ...

    def close(self) -> None:
        ...


class ShortestPathFollowerLike(Protocol):
    """Expose the shortest-path follower action interface."""

    def get_next_action(self, goal_position: np.ndarray) -> object:
        ...


def build_hm3d_objectnav_episode_manifest(
    cfg: TopoVLMConfig,
    *,
    env: ObjectNavExpertEnv | None = None,
    follower: ShortestPathFollowerLike | None = None,
) -> dict[str, object]:
    data_root = Path(cfg.data.data_root)
    output_data_root = resolve_materialization_data_root(cfg.data.data_root)
    manifest_path = resolve_data_path(output_data_root, cfg.data.episodes_manifest)
    rgb_dir = output_data_root / "rgb" / cfg.data.dataset_name / cfg.data.split
    actions_dir = output_data_root / "actions" / cfg.data.dataset_name / cfg.data.split
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    selection_records = (
        load_objectnav_selection_records(cfg.data)
        if cfg.data.episode_selection_manifest is not None
        else None
    )
    selection_ids = (
        {record.source_trajectory_id for record in selection_records}
        if selection_records is not None
        else None
    )
    existing_records_by_id = (
        _load_existing_payload_records(cfg, output_data_root, selection_records)
        if selection_records is not None
        else {}
    )
    remaining_selection_ids = (
        selection_ids.difference(existing_records_by_id)
        if selection_ids is not None
        else None
    )
    if existing_records_by_id:
        print(
            json.dumps(
                {
                    "event": "hm3d_build_episodes_resume",
                    "existing_payloads": len(existing_records_by_id),
                    "remaining_selected": len(remaining_selection_ids or set()),
                    "manifest_tmp": str(tmp_manifest_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    owns_env = False
    skipped = []
    written = list(existing_records_by_id.values())
    if selection_ids is not None:
        seen_selection_ids = set(existing_records_by_id)
    else:
        seen_selection_ids = set()
    try:
        with tmp_manifest_path.open("w", encoding="utf-8") as handle:
            for record in written:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if written:
                handle.flush()

            if remaining_selection_ids is None or remaining_selection_ids:
                owns_env = env is None
                if env is None:
                    env = _open_habitat_env(cfg)
                active_selection_ids = remaining_selection_ids or selection_ids
                if active_selection_ids is not None:
                    _filter_env_episodes_to_selection(env, active_selection_ids)
                if follower is None:
                    follower = _build_shortest_path_follower(env, cfg)
                total_episodes = len(env.episodes)
                max_resets = total_episodes
                if selection_ids is None and cfg.data.max_episodes is not None:
                    max_resets = min(max_resets, int(cfg.data.max_episodes))
                for attempted in range(1, max_resets + 1):
                    observations = env.reset()
                    episode = env.current_episode
                    source_trajectory_id = objectnav_source_trajectory_id(episode)
                    if (
                        active_selection_ids is not None
                        and source_trajectory_id not in active_selection_ids
                    ):
                        continue
                    seen_selection_ids.add(source_trajectory_id)
                    try:
                        with _episode_rollout_timeout(cfg, env):
                            frames, actions = _rollout_shortest_path_episode(
                                cfg, env, follower, observations
                            )
                    except Exception as exc:
                        skipped.append(source_trajectory_id)
                        print(
                            json.dumps(
                                {
                                    "event": "hm3d_build_episodes_skip",
                                    "attempted": attempted,
                                    "source_trajectory_id": source_trajectory_id,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        continue
                    record = _episode_payload_record(cfg, source_trajectory_id, episode)
                    np.save(resolve_data_path(output_data_root, record["rgb_path"]), frames)
                    np.save(resolve_data_path(output_data_root, record["actions_path"]), actions)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    written.append(record)
                    if len(written) == 1 or len(written) % 25 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "hm3d_build_episodes_progress",
                                    "attempted": attempted,
                                    "written": len(written),
                                    "skipped": len(skipped),
                                    "manifest_tmp": str(tmp_manifest_path),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            if selection_ids is not None and seen_selection_ids != selection_ids:
                missing_ids = sorted(selection_ids.difference(seen_selection_ids))
                raise ValueError(
                    f"Selection manifest contains ObjectNav ids absent from source: {missing_ids[:5]}"
                )
        tmp_manifest_path.replace(manifest_path)
    except Exception:
        tmp_manifest_path.unlink(missing_ok=True)
        raise
    finally:
        if owns_env:
            env.close()

    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "source_data_root": str(data_root),
        "output_data_root": str(output_data_root),
        "episodes_written": len(written),
        "episodes_skipped": len(skipped),
        "rgb_dir": str(rgb_dir),
        "actions_dir": str(actions_dir),
        "selection_manifest": cfg.data.episode_selection_manifest,
        "selected_source_episodes": len(selection_ids) if selection_ids is not None else None,
    }


def _filter_env_episodes_to_selection(
    env: ObjectNavExpertEnv, selection_ids: set[str]
) -> None:
    selected_episodes = [
        episode
        for episode in env.episodes
        if objectnav_source_trajectory_id(episode) in selection_ids
    ]
    selected_ids = {objectnav_source_trajectory_id(episode) for episode in selected_episodes}
    missing_ids = selection_ids.difference(selected_ids)
    if missing_ids:
        raise ValueError(
            "Selection manifest contains ObjectNav ids absent from Habitat env: "
            f"{sorted(missing_ids)[:5]}"
        )
    env.episodes = selected_episodes


def _load_existing_payload_records(
    cfg: TopoVLMConfig,
    output_data_root: Path,
    selection_records: list[ObjectNavSelectionRecord],
) -> dict[str, dict[str, object]]:
    records = {}
    for selection_record in selection_records:
        source_trajectory_id = selection_record.source_trajectory_id
        record = _selection_payload_record(cfg, selection_record)
        rgb_path = resolve_data_path(output_data_root, record["rgb_path"])
        actions_path = resolve_data_path(output_data_root, record["actions_path"])
        if _valid_npy_payload(rgb_path, expected_ndim=4) and _valid_npy_payload(
            actions_path, expected_ndim=1
        ):
            records[source_trajectory_id] = record
    return records


def _selection_payload_record(
    cfg: TopoVLMConfig, selection_record: ObjectNavSelectionRecord
) -> dict[str, object]:
    payload_id = _safe_payload_id(selection_record.source_trajectory_id)
    rgb_rel = Path("rgb") / cfg.data.dataset_name / cfg.data.split / f"{payload_id}.npy"
    actions_rel = (
        Path("actions") / cfg.data.dataset_name / cfg.data.split / f"{payload_id}.npy"
    )
    return {
        "episode_id": payload_id,
        "split": cfg.data.split,
        "scene_id": str(resolve_objectnav_scene_path(cfg.data, selection_record.scene_id)),
        "goal_text": selection_record.object_category,
        "rgb_path": str(rgb_rel),
        "actions_path": str(actions_rel),
        "source_dataset": "hm3d_objectnav_shortest_path",
        "source_trajectory_id": selection_record.source_trajectory_id,
        "object_category": selection_record.object_category,
    }


def _episode_payload_record(
    cfg: TopoVLMConfig, source_trajectory_id: str, episode: object
) -> dict[str, object]:
    selection_record = ObjectNavSelectionRecord(
        source_trajectory_id=source_trajectory_id,
        episode_id=str(getattr(episode, "episode_id")),
        scene_id=str(getattr(episode, "scene_id")),
        object_category=str(getattr(episode, "object_category")),
        shard_path="",
    )
    return _selection_payload_record(cfg, selection_record)


def _valid_npy_payload(path: Path, *, expected_ndim: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        array = np.load(path, mmap_mode="r")
        try:
            return array.ndim == expected_ndim
        finally:
            mmap_obj = getattr(array, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            del array
    except Exception:
        return False


def _rollout_shortest_path_episode(
    cfg: TopoVLMConfig,
    env: ObjectNavExpertEnv,
    follower: ShortestPathFollowerLike,
    observations: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    actions = []
    current_observations = observations
    for _ in range(cfg.eval.max_steps):
        frames.append(_rgb_frame(cfg, current_observations))
        goal_position = _select_goal_position(env)
        action = _action_id(follower.get_next_action(goal_position))
        actions.append(action)
        if action == 0:
            return np.stack(frames, axis=0).astype("uint8"), np.asarray(actions, dtype=np.int64)
        current_observations = env.step(action)
    raise ValueError(
        f"Shortest-path rollout exceeded max_steps={cfg.eval.max_steps} "
        f"for episode {getattr(env.current_episode, 'episode_id')}"
    )


@contextmanager
def _episode_rollout_timeout(
    cfg: TopoVLMConfig, env: ObjectNavExpertEnv
) -> Iterator[None]:
    timeout_seconds = cfg.eval.episode_timeout_seconds
    if timeout_seconds <= 0:
        yield
        return

    def _raise_timeout(signum: int, frame: object) -> None:
        raise TimeoutError(
            f"Episode rollout exceeded episode_timeout_seconds={timeout_seconds} "
            f"for episode {getattr(env.current_episode, 'episode_id')}"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _open_habitat_env(cfg: TopoVLMConfig) -> ObjectNavExpertEnv:
    import habitat

    habitat_config = habitat.get_config(cfg.data.habitat_config)
    runtime_data_root = resolve_runtime_data_root(cfg.data.data_root)
    _configure_habitat_dataset(habitat_config, cfg, runtime_data_root)
    return habitat.Env(config=habitat_config)


def _configure_habitat_dataset(
    habitat_config: object, cfg: TopoVLMConfig, data_root: Path
) -> None:
    from omegaconf import OmegaConf

    OmegaConf.set_readonly(habitat_config, False)
    habitat_config.habitat.dataset.split = cfg.data.split
    if data_root != Path(cfg.data.data_root):
        habitat_config.habitat.dataset.data_path = str(
            data_root / cfg.data.objectnav_dataset_dir / "{split}" / "{split}.json.gz"
        )
        habitat_config.habitat.dataset.scenes_dir = str(data_root / "scene_datasets")
        habitat_config.habitat.simulator.scene_dataset = str(
            data_root / cfg.data.scene_dataset_config
        )
    OmegaConf.set_readonly(habitat_config, True)


def _build_shortest_path_follower(
    env: ObjectNavExpertEnv, cfg: TopoVLMConfig
) -> ShortestPathFollowerLike:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    return ShortestPathFollower(
        env.sim,
        goal_radius=cfg.eval.success_distance,
        return_one_hot=False,
        stop_on_error=True,
    )


def _select_goal_position(env: ObjectNavExpertEnv) -> np.ndarray:
    episode = env.current_episode
    candidates = _goal_position_candidates(episode)
    agent_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float32)
    best_position = None
    best_distance = None
    for candidate in candidates:
        distance = env.sim.geodesic_distance(agent_position, candidate, episode=episode)
        if not np.isfinite(distance):
            distance = float(np.linalg.norm(candidate - agent_position))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_position = candidate
    if best_position is None:
        raise ValueError(f"Episode has no ObjectNav goal positions: {getattr(episode, 'episode_id')}")
    return best_position


def _goal_position_candidates(episode: object) -> list[np.ndarray]:
    candidates = []
    for goal in getattr(episode, "goals", []):
        for view_point in getattr(goal, "view_points", []) or []:
            agent_state = getattr(view_point, "agent_state", None)
            position = getattr(agent_state, "position", None)
            if position is not None:
                candidates.append(np.asarray(position, dtype=np.float32))
        position = getattr(goal, "position", None)
        if position is not None:
            candidates.append(np.asarray(position, dtype=np.float32))
    return candidates


def _rgb_frame(cfg: TopoVLMConfig, observations: dict[str, object]) -> np.ndarray:
    if cfg.data.image_key not in observations:
        raise KeyError(f"Missing observation image key: {cfg.data.image_key}")
    return np.asarray(observations[cfg.data.image_key])[..., :3].copy()


def _action_id(action: object) -> int:
    if action is None:
        return 0
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return int(action.item())
        return int(np.argmax(action))
    return int(action)


def _safe_payload_id(episode_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("_")
