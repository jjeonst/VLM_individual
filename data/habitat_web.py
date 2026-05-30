"""Habitat-Web replay source loading and audit helpers."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from configs.schema import DataConfig
from data.habitat_manifest import resolve_data_path
from data.habitat_objectnav import resolve_objectnav_scene_path


HABITAT_WEB_ACTION_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5,
}
INVENTORY_LIST_LIMIT = 500
INVENTORY_TOP_COUNT = 30


@dataclass(frozen=True)
class HabitatWebReplayEpisode:
    """Represent one Habitat-Web ObjectNav human-demonstration replay."""

    episode_id: str
    scene_id: str
    object_category: str
    shard_path: str
    replay_length: int
    actions: tuple[str, ...]
    has_embedded_rgb: bool


@dataclass(frozen=True)
class HabitatWebSelectionRecord:
    """Represent one selected Habitat-Web source replay for subset materialization."""

    source_trajectory_id: str
    scene_id: str
    object_category: str
    shard_path: str
    replay_length: int


class HabitatWebReplayDataset:
    """Iterate Habitat-Web ObjectNav replay shards without loading all shards."""

    def __init__(self, config: DataConfig, *, split: str | None = None):
        self.config = config
        self.data_root = Path(config.data_root)
        self.split = split or config.split
        self.dataset_dir = resolve_data_path(self.data_root, config.objectnav_dataset_dir)
        self.split_dir = self.dataset_dir / self.split
        self.split_index = self.split_dir / f"{self.split}.json.gz"
        self.content_dir = self.split_dir / "content"
        if not self.split_index.exists():
            raise FileNotFoundError(self.split_index)
        if not self.content_dir.exists():
            raise FileNotFoundError(self.content_dir)
        self.content_files = sorted(self.content_dir.glob("*.json.gz"))
        if not self.content_files:
            raise FileNotFoundError(f"No Habitat-Web content shards in {self.content_dir}")

    def iter_episodes(
        self, *, max_episodes: int | None = None
    ) -> Iterator[HabitatWebReplayEpisode]:
        emitted = 0
        for raw, shard in self.iter_raw_records(max_episodes=max_episodes):
            replay = raw.get("reference_replay")
            if not isinstance(replay, list):
                raise ValueError(f"Missing reference_replay in {shard}")
            actions = tuple(str(step["action"]) for step in replay)
            unknown = sorted(set(actions).difference(HABITAT_WEB_ACTION_TO_ID))
            if unknown:
                raise ValueError(f"Unsupported Habitat-Web actions in {shard}: {unknown}")
            yield HabitatWebReplayEpisode(
                episode_id=str(raw["episode_id"]),
                scene_id=str(raw["scene_id"]),
                object_category=str(raw["object_category"]),
                shard_path=str(shard),
                replay_length=len(replay),
                actions=actions,
                has_embedded_rgb=_replay_has_embedded_rgb(replay),
            )

    def iter_raw_records(
        self, *, max_episodes: int | None = None
    ) -> Iterator[tuple[dict[str, object], Path]]:
        emitted = 0
        for shard in self.content_files:
            if is_git_lfs_pointer(shard):
                continue
            for raw in _load_habitat_web_payload(shard):
                yield raw, shard
                emitted += 1
                if max_episodes is not None and emitted >= max_episodes:
                    return

    def summary(self, *, sample_episodes: int = 4) -> dict[str, object]:
        split_index_materialized = self.split_index.exists() and not is_git_lfs_pointer(
            self.split_index
        )
        pointer_shards = [str(path) for path in self.content_files if is_git_lfs_pointer(path)]
        materialized_shards = [
            str(path) for path in self.content_files if path.exists() and not is_git_lfs_pointer(path)
        ]
        samples = []
        missing_scenes = set()
        action_vocab = set()
        replay_lengths = []
        for episode in self.iter_episodes(max_episodes=sample_episodes):
            scene_path = resolve_objectnav_scene_path(self.config, episode.scene_id)
            samples.append(
                {
                    "episode_id": episode.episode_id,
                    "scene_id": episode.scene_id,
                    "object_category": episode.object_category,
                    "scene_path": str(scene_path),
                    "scene_exists": scene_path.exists(),
                    "replay_length": episode.replay_length,
                    "has_embedded_rgb": episode.has_embedded_rgb,
                    "first_actions": list(episode.actions[:8]),
                    "last_actions": list(episode.actions[-8:]),
                }
            )
            replay_lengths.append(episode.replay_length)
            action_vocab.update(episode.actions)
            if not scene_path.exists():
                missing_scenes.add(str(scene_path))
        return {
            "split": self.split,
            "dataset_dir": str(self.dataset_dir),
            "split_index": str(self.split_index),
            "split_index_materialized": split_index_materialized,
            "content_dir": str(self.content_dir),
            "content_shards": len(self.content_files),
            "materialized_content_shards": len(materialized_shards),
            "lfs_pointer_content_shards": len(pointer_shards),
            "sample_episodes": samples,
            "sampled_episodes": len(samples),
            "sample_action_vocab": sorted(action_vocab),
            "sample_replay_length_min": min(replay_lengths) if replay_lengths else None,
            "sample_replay_length_max": max(replay_lengths) if replay_lengths else None,
            "missing_sample_scenes": sorted(missing_scenes),
            "requires_scene_rendering": any(not sample["has_embedded_rgb"] for sample in samples),
        }

    def inventory(self, *, max_episodes: int | None = None) -> dict[str, object]:
        split_index_materialized = self.split_index.exists() and not is_git_lfs_pointer(
            self.split_index
        )
        pointer_shards = [str(path) for path in self.content_files if is_git_lfs_pointer(path)]
        materialized_shards = [
            str(path) for path in self.content_files if path.exists() and not is_git_lfs_pointer(path)
        ]
        scene_counts: Counter[str] = Counter()
        object_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        scene_paths: dict[str, str] = {}
        existing_scene_ids = set()
        missing_scene_ids = set()
        replay_length_min = None
        replay_length_max = None
        replay_length_sum = 0
        episodes = 0
        episodes_with_embedded_rgb = 0

        for episode in self.iter_episodes(max_episodes=max_episodes):
            episodes += 1
            scene_counts[episode.scene_id] += 1
            object_counts[episode.object_category] += 1
            action_counts.update(episode.actions)
            replay_length_sum += episode.replay_length
            replay_length_min = (
                episode.replay_length
                if replay_length_min is None
                else min(replay_length_min, episode.replay_length)
            )
            replay_length_max = (
                episode.replay_length
                if replay_length_max is None
                else max(replay_length_max, episode.replay_length)
            )
            if episode.has_embedded_rgb:
                episodes_with_embedded_rgb += 1
            scene_path = resolve_objectnav_scene_path(self.config, episode.scene_id)
            scene_paths[episode.scene_id] = str(scene_path)
            if scene_path.exists():
                existing_scene_ids.add(episode.scene_id)
            else:
                missing_scene_ids.add(episode.scene_id)

        sorted_scene_ids = sorted(scene_counts)
        sorted_missing_scene_ids = sorted(missing_scene_ids)
        sorted_existing_scene_ids = sorted(existing_scene_ids)
        return {
            "split": self.split,
            "max_episodes_applied": max_episodes,
            "dataset_dir": str(self.dataset_dir),
            "split_index": str(self.split_index),
            "split_index_materialized": split_index_materialized,
            "content_dir": str(self.content_dir),
            "content_shards": len(self.content_files),
            "materialized_content_shards": len(materialized_shards),
            "lfs_pointer_content_shards": len(pointer_shards),
            "episodes": episodes,
            "unique_scenes": len(scene_counts),
            "unique_object_categories": len(object_counts),
            "unique_actions": len(action_counts),
            "action_counts": [
                {"action": action, "count": count}
                for action, count in sorted(action_counts.items())
            ],
            "object_counts_top": [
                {"object_category": category, "count": count}
                for category, count in object_counts.most_common(INVENTORY_TOP_COUNT)
            ],
            "scene_counts_top": [
                {"scene_id": scene_id, "count": count}
                for scene_id, count in scene_counts.most_common(INVENTORY_TOP_COUNT)
            ],
            "required_scene_ids": sorted_scene_ids[:INVENTORY_LIST_LIMIT],
            "required_scene_ids_listed": min(len(sorted_scene_ids), INVENTORY_LIST_LIMIT),
            "missing_scene_ids": sorted_missing_scene_ids[:INVENTORY_LIST_LIMIT],
            "missing_scene_ids_listed": min(
                len(sorted_missing_scene_ids), INVENTORY_LIST_LIMIT
            ),
            "missing_scene_paths": [
                scene_paths[scene_id]
                for scene_id in sorted_missing_scene_ids[:INVENTORY_LIST_LIMIT]
            ],
            "missing_scene_count": len(missing_scene_ids),
            "existing_scene_ids": sorted_existing_scene_ids[:INVENTORY_LIST_LIMIT],
            "existing_scene_ids_listed": min(
                len(sorted_existing_scene_ids), INVENTORY_LIST_LIMIT
            ),
            "existing_scene_count": len(existing_scene_ids),
            "episodes_with_embedded_rgb": episodes_with_embedded_rgb,
            "requires_scene_rendering": episodes_with_embedded_rgb != episodes,
            "replay_length_min": replay_length_min,
            "replay_length_max": replay_length_max,
            "replay_length_mean": (replay_length_sum / episodes) if episodes else None,
        }


def load_habitat_web_summary(
    config: DataConfig, *, sample_episodes: int = 4, split: str | None = None
) -> dict[str, object]:
    dataset = HabitatWebReplayDataset(config, split=split)
    return dataset.summary(sample_episodes=sample_episodes)


def load_habitat_web_inventory(
    config: DataConfig, *, split: str | None = None, max_episodes: int | None = None
) -> dict[str, object]:
    dataset = HabitatWebReplayDataset(config, split=split)
    return dataset.inventory(max_episodes=max_episodes)


def build_habitat_web_balanced_selection_manifest(config: DataConfig) -> dict[str, object]:
    if config.episode_selection_manifest is None:
        raise ValueError("DataConfig.episode_selection_manifest is required")
    if config.balanced_subset_size is None:
        raise ValueError("DataConfig.balanced_subset_size is required")
    if config.balanced_subset_size <= 0:
        raise ValueError("DataConfig.balanced_subset_size must be positive")
    dataset = HabitatWebReplayDataset(config)
    buckets: dict[tuple[str, str], list[HabitatWebSelectionRecord]] = {}
    for raw, shard in dataset.iter_raw_records(max_episodes=config.max_episodes):
        record = _selection_record_from_raw(raw, shard)
        key = (record.scene_id, record.object_category)
        buckets.setdefault(key, []).append(record)

    selected = []
    scene_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    target = int(config.balanced_subset_size)
    while len(selected) < target:
        candidates = [key for key, bucket in buckets.items() if bucket]
        if not candidates:
            break
        scene_id, object_category = min(
            candidates,
            key=lambda key: (
                scene_counts[key[0]],
                object_counts[key[1]],
                pair_counts[key],
                key[0],
                key[1],
            ),
        )
        record = buckets[(scene_id, object_category)].pop(0)
        selected.append(record)
        scene_counts[scene_id] += 1
        object_counts[object_category] += 1
        pair_counts[(scene_id, object_category)] += 1

    manifest_path = _selection_manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        with tmp_manifest_path.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(
                    json.dumps(_selection_record_to_json(record), sort_keys=True) + "\n"
                )
        tmp_manifest_path.replace(manifest_path)
    except Exception:
        tmp_manifest_path.unlink(missing_ok=True)
        raise
    summary = _selection_summary(config, selected)
    summary.update(
        {
            "status": "ok",
            "selection_manifest": str(manifest_path),
            "requested_subset_size": target,
            "available_pair_count": len(buckets),
            "source_max_episodes_applied": config.max_episodes,
        }
    )
    return summary


def load_habitat_web_selection_ids(config: DataConfig) -> set[str]:
    return {record.source_trajectory_id for record in load_habitat_web_selection_records(config)}


def load_habitat_web_selection_summary(config: DataConfig) -> dict[str, object]:
    records = load_habitat_web_selection_records(config)
    summary = _selection_summary(config, records)
    summary.update(
        {
            "status": "ok",
            "selection_manifest": str(_selection_manifest_path(config)),
            "requested_subset_size": config.balanced_subset_size,
            "source_max_episodes_applied": config.max_episodes,
        }
    )
    return summary


def load_habitat_web_selection_records(
    config: DataConfig,
) -> list[HabitatWebSelectionRecord]:
    manifest_path = _selection_manifest_path(config)
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            records.append(_selection_record_from_json(raw, manifest_path, line_number))
    if not records:
        raise ValueError(f"Habitat-Web selection manifest is empty: {manifest_path}")
    return records


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _selection_manifest_path(config: DataConfig) -> Path:
    if config.episode_selection_manifest is None:
        raise ValueError("DataConfig.episode_selection_manifest is required")
    return resolve_data_path(Path(config.data_root), config.episode_selection_manifest)


def _selection_record_to_json(record: HabitatWebSelectionRecord) -> dict[str, object]:
    return {
        "source_trajectory_id": record.source_trajectory_id,
        "episode_id": record.source_trajectory_id,
        "scene_id": record.scene_id,
        "object_category": record.object_category,
        "shard_path": record.shard_path,
        "replay_length": record.replay_length,
    }


def _selection_record_from_raw(
    raw: dict[str, object], shard: Path
) -> HabitatWebSelectionRecord:
    replay = raw.get("reference_replay")
    if not isinstance(replay, list):
        raise ValueError(f"Missing reference_replay in {shard}")
    return HabitatWebSelectionRecord(
        source_trajectory_id=str(raw["episode_id"]),
        scene_id=str(raw["scene_id"]),
        object_category=str(raw["object_category"]),
        shard_path=str(shard),
        replay_length=len(replay),
    )


def _selection_record_from_json(
    raw: dict[str, object], manifest_path: Path, line_number: int
) -> HabitatWebSelectionRecord:
    required = {
        "source_trajectory_id",
        "scene_id",
        "object_category",
        "shard_path",
        "replay_length",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(
            f"Missing selection fields in {manifest_path}:{line_number}: {sorted(missing)}"
        )
    return HabitatWebSelectionRecord(
        source_trajectory_id=str(raw["source_trajectory_id"]),
        scene_id=str(raw["scene_id"]),
        object_category=str(raw["object_category"]),
        shard_path=str(raw["shard_path"]),
        replay_length=int(raw["replay_length"]),
    )


def _selection_summary(
    config: DataConfig, records: list[HabitatWebSelectionRecord]
) -> dict[str, object]:
    scene_counts: Counter[str] = Counter(record.scene_id for record in records)
    object_counts: Counter[str] = Counter(record.object_category for record in records)
    pair_counts: Counter[tuple[str, str]] = Counter(
        (record.scene_id, record.object_category) for record in records
    )
    missing_scene_paths = []
    existing_scene_ids = set()
    for scene_id in scene_counts:
        scene_path = resolve_objectnav_scene_path(config, scene_id)
        if scene_path.exists():
            existing_scene_ids.add(scene_id)
        else:
            missing_scene_paths.append(str(scene_path))
    return {
        "selected_episodes": len(records),
        "unique_scenes": len(scene_counts),
        "unique_object_categories": len(object_counts),
        "unique_scene_object_pairs": len(pair_counts),
        "scene_count_min": min(scene_counts.values()) if scene_counts else None,
        "scene_count_max": max(scene_counts.values()) if scene_counts else None,
        "object_count_min": min(object_counts.values()) if object_counts else None,
        "object_count_max": max(object_counts.values()) if object_counts else None,
        "pair_count_min": min(pair_counts.values()) if pair_counts else None,
        "pair_count_max": max(pair_counts.values()) if pair_counts else None,
        "existing_scene_count": len(existing_scene_ids),
        "missing_scene_count": len(missing_scene_paths),
        "missing_scene_paths": sorted(missing_scene_paths),
        "object_counts": [
            {"object_category": category, "count": count}
            for category, count in sorted(object_counts.items())
        ],
        "scene_counts_top": [
            {"scene_id": scene_id, "count": count}
            for scene_id, count in scene_counts.most_common(INVENTORY_TOP_COUNT)
        ],
    }


def _load_habitat_web_payload(path: Path) -> list[dict[str, object]]:
    if is_git_lfs_pointer(path):
        raise ValueError(f"Habitat-Web shard is a Git LFS pointer, not payload: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Habitat-Web shard must contain a mapping: {path}")
    episodes = loaded["episodes"]
    if not isinstance(episodes, list):
        raise ValueError(f"Habitat-Web shard episodes must be a list: {path}")
    return episodes


def _replay_has_embedded_rgb(replay: list[object]) -> bool:
    for raw_step in replay:
        if not isinstance(raw_step, dict):
            continue
        agent_state = raw_step.get("agent_state")
        if not isinstance(agent_state, dict):
            continue
        sensor_data = agent_state.get("sensor_data")
        if not isinstance(sensor_data, dict):
            continue
        rgb = sensor_data.get("rgb")
        if isinstance(rgb, dict) and {"data", "array", "image", "bytes", "pixels"}.intersection(rgb):
            return True
    return False
