"""Render Habitat-Web replays into PR2L-ready RGB/action episode payloads."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from configs.schema import TopoVLMConfig
from data.habitat_manifest import resolve_data_path
from data.habitat_objectnav import resolve_objectnav_scene_path
from data.habitat_web import HABITAT_WEB_ACTION_TO_ID, HabitatWebReplayDataset


class ReplayRenderer(Protocol):
    """Render replay agent states for one Habitat scene."""

    def render_episode(self, scene_path: Path, replay: list[dict[str, object]]) -> np.ndarray:
        ...


def build_habitat_web_episode_manifest(
    cfg: TopoVLMConfig, *, renderer: ReplayRenderer | None = None
) -> dict[str, object]:
    """Materialize Habitat-Web replays as NumPy RGB/action arrays and a manifest."""

    data_root = Path(cfg.data.data_root)
    dataset = HabitatWebReplayDataset(cfg.data)
    manifest_path = resolve_data_path(data_root, cfg.data.episodes_manifest)
    rgb_dir = data_root / "rgb" / cfg.data.dataset_name / cfg.data.split
    actions_dir = data_root / "actions" / cfg.data.dataset_name / cfg.data.split
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if renderer is None:
        renderer = HabitatSimReplayRenderer(cfg)

    written = []
    dropped_leading_stop_count = 0
    try:
        with tmp_manifest_path.open("w", encoding="utf-8") as handle:
            for raw, shard in dataset.iter_raw_records(max_episodes=cfg.data.max_episodes):
                replay = _validated_replay(raw, shard)
                replay, dropped_leading_stop = _drop_leading_initial_stop(replay)
                dropped_leading_stop_count += int(dropped_leading_stop)
                scene_id = str(raw["scene_id"])
                scene_path = resolve_objectnav_scene_path(cfg.data, scene_id)
                if not scene_path.exists():
                    raise FileNotFoundError(scene_path)
                frames = renderer.render_episode(scene_path, replay).astype("uint8")
                actions = np.asarray(
                    [HABITAT_WEB_ACTION_TO_ID[str(step["action"])] for step in replay],
                    dtype=np.int64,
                )
                if len(frames) != len(actions):
                    raise ValueError(
                        f"Rendered frame/action length mismatch for {raw['episode_id']}: "
                        f"{len(frames)} vs {len(actions)}"
                    )
                payload_id = _safe_payload_id(str(raw["episode_id"]))
                rgb_rel = (
                    Path("rgb") / cfg.data.dataset_name / cfg.data.split / f"{payload_id}.npy"
                )
                actions_rel = (
                    Path("actions")
                    / cfg.data.dataset_name
                    / cfg.data.split
                    / f"{payload_id}.npy"
                )
                np.save(resolve_data_path(data_root, rgb_rel), frames)
                np.save(resolve_data_path(data_root, actions_rel), actions)
                record = {
                    "episode_id": payload_id,
                    "split": cfg.data.split,
                    "scene_id": scene_id,
                    "goal_text": str(raw["object_category"]),
                    "rgb_path": str(rgb_rel),
                    "actions_path": str(actions_rel),
                    "source_dataset": "habitat_web",
                    "source_trajectory_id": str(raw["episode_id"]),
                    "object_category": str(raw["object_category"]),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                written.append(record)
        tmp_manifest_path.replace(manifest_path)
    finally:
        close = getattr(renderer, "close", None)
        if callable(close):
            close()

    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "episodes_written": len(written),
        "rgb_dir": str(rgb_dir),
        "actions_dir": str(actions_dir),
        "dropped_leading_stop_count": dropped_leading_stop_count,
    }


class HabitatSimReplayRenderer:
    """Habitat-Sim renderer for replay agent states."""

    def __init__(self, cfg: TopoVLMConfig):
        self.cfg = cfg
        self._scene_path: Path | None = None
        self._sim = None
        self._agent = None

    def render_episode(self, scene_path: Path, replay: list[dict[str, object]]) -> np.ndarray:
        self._ensure_scene(scene_path)
        frames = []
        for step in replay:
            self._set_agent_state(step)
            observation = self._sim.get_sensor_observations()[self.cfg.data.image_key]
            frames.append(np.asarray(observation)[..., :3].copy())
        return np.stack(frames, axis=0)

    def _ensure_scene(self, scene_path: Path) -> None:
        if self._scene_path == scene_path:
            return
        if self._sim is not None:
            self._sim.close()
        self._scene_path = scene_path
        self._sim, self._agent = _open_habitat_sim(self.cfg, scene_path)

    def _set_agent_state(self, step: dict[str, object]) -> None:
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs

        raw_state = step["agent_state"]
        if not isinstance(raw_state, dict):
            raise ValueError("Habitat-Web replay step is missing agent_state")
        state = habitat_sim.AgentState()
        state.position = np.asarray(raw_state["position"], dtype=np.float32)
        state.rotation = quat_from_coeffs(raw_state["rotation"])
        self._agent.set_state(state, reset_sensors=True)

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            self._agent = None


def _open_habitat_sim(cfg: TopoVLMConfig, scene_path: Path):
    import habitat_sim

    data_root = Path(cfg.data.data_root)
    scene_dataset_config = resolve_data_path(data_root, cfg.data.scene_dataset_config)
    if not scene_dataset_config.exists():
        raise FileNotFoundError(scene_dataset_config)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = str(scene_path)
    backend_cfg.scene_dataset_config_file = str(scene_dataset_config)
    backend_cfg.enable_physics = False

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = cfg.data.image_key
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [cfg.data.image_height, cfg.data.image_width]
    sensor_spec.position = [0.0, 0.88, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    return sim, sim.initialize_agent(0)


def _validated_replay(raw: dict[str, object], shard: Path) -> list[dict[str, object]]:
    replay = raw.get("reference_replay")
    if not isinstance(replay, list) or not replay:
        raise ValueError(f"Missing reference_replay in {shard}")
    typed_replay = []
    for step in replay:
        if not isinstance(step, dict):
            raise ValueError(f"Malformed reference_replay step in {shard}")
        action = str(step.get("action"))
        if action not in HABITAT_WEB_ACTION_TO_ID:
            raise ValueError(f"Unsupported Habitat-Web action in {shard}: {action}")
        if not isinstance(step.get("agent_state"), dict):
            raise ValueError(f"Missing agent_state in {shard}")
        typed_replay.append(step)
    return typed_replay


def _drop_leading_initial_stop(
    replay: list[dict[str, object]]
) -> tuple[list[dict[str, object]], bool]:
    if len(replay) > 1 and replay[0].get("action") == "STOP":
        return replay[1:], True
    return replay, False


def _safe_payload_id(episode_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("_")
