"""Loader and audit helpers for Habitat ObjectNav episode shards."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from configs.schema import DataConfig
from data.habitat_manifest import resolve_data_path


@dataclass(frozen=True)
class ObjectNavEpisode:
    """Represent one raw Habitat ObjectNav episode record."""

    episode_id: str
    scene_id: str
    object_category: str
    shard_path: str


@dataclass(frozen=True)
class ObjectNavSelectionRecord:
    """Represent one selected HM3D ObjectNav episode for subset materialization."""

    source_trajectory_id: str
    episode_id: str
    scene_id: str
    object_category: str
    shard_path: str


class HabitatObjectNavDataset:
    """Iterate raw Habitat ObjectNav episode shards without loading every shard at once."""

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
            raise FileNotFoundError(f"No ObjectNav content shards in {self.content_dir}")

    def iter_episodes(self, *, max_episodes: int | None = None) -> Iterator[ObjectNavEpisode]:
        emitted = 0
        for shard in self.content_files:
            for raw in _load_episode_payload(shard):
                yield ObjectNavEpisode(
                    episode_id=str(raw["episode_id"]),
                    scene_id=str(raw["scene_id"]),
                    object_category=str(raw["object_category"]),
                    shard_path=str(shard),
                )
                emitted += 1
                if max_episodes is not None and emitted >= max_episodes:
                    return

    def first_episode(self) -> ObjectNavEpisode:
        for episode in self.iter_episodes(max_episodes=1):
            return episode
        raise ValueError(f"No ObjectNav episodes in {self.content_dir}")

    def resolve_scene_path(self, episode: ObjectNavEpisode) -> Path:
        return resolve_objectnav_scene_path(self.config, episode.scene_id)

    def summary(self, *, sample_episodes: int = 1) -> dict[str, object]:
        samples = []
        missing_scenes = []
        for episode in self.iter_episodes(max_episodes=sample_episodes):
            scene_path = self.resolve_scene_path(episode)
            sample = {
                "episode_id": episode.episode_id,
                "scene_id": episode.scene_id,
                "object_category": episode.object_category,
                "scene_path": str(scene_path),
                "scene_exists": scene_path.exists(),
            }
            samples.append(sample)
            if not scene_path.exists():
                missing_scenes.append(str(scene_path))
        return {
            "split": self.split,
            "dataset_dir": str(self.dataset_dir),
            "split_index": str(self.split_index),
            "content_dir": str(self.content_dir),
            "content_shards": len(self.content_files),
            "sample_episodes": samples,
            "missing_sample_scenes": missing_scenes,
        }


def resolve_objectnav_scene_path(config: DataConfig, scene_id: str) -> Path:
    data_root = Path(config.data_root)
    scene_root = resolve_data_path(data_root, config.scene_dataset_dir)
    target = Path(scene_id)
    if target.is_absolute():
        return target
    parts = target.parts
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "scene_datasets":
        return data_root / Path(*parts[1:])
    if len(parts) >= 1 and parts[0] == "scene_datasets":
        return data_root / target
    return scene_root / target


def load_objectnav_summary(config: DataConfig, *, sample_episodes: int = 1) -> dict[str, object]:
    dataset = HabitatObjectNavDataset(config)
    return dataset.summary(sample_episodes=sample_episodes)


def build_objectnav_balanced_selection_manifest(config: DataConfig) -> dict[str, object]:
    if config.episode_selection_manifest is None:
        raise ValueError("DataConfig.episode_selection_manifest is required")
    if config.balanced_subset_size is None:
        raise ValueError("DataConfig.balanced_subset_size is required")
    dataset = HabitatObjectNavDataset(config)
    records_by_bucket: dict[tuple[str, str], list[ObjectNavEpisode]] = defaultdict(list)
    total_episodes = 0
    for episode in dataset.iter_episodes():
        records_by_bucket[(episode.scene_id, episode.object_category)].append(episode)
        total_episodes += 1

    selected = []
    bucket_offsets = {bucket: 0 for bucket in records_by_bucket}
    target_count = int(config.balanced_subset_size)
    while len(selected) < target_count:
        emitted_this_round = 0
        for bucket in sorted(records_by_bucket):
            offset = bucket_offsets[bucket]
            bucket_records = records_by_bucket[bucket]
            if offset >= len(bucket_records):
                continue
            selected.append(bucket_records[offset])
            bucket_offsets[bucket] = offset + 1
            emitted_this_round += 1
            if len(selected) >= target_count:
                break
        if emitted_this_round == 0:
            break

    data_root = Path(config.data_root)
    manifest_path = resolve_data_path(data_root, config.episode_selection_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for episode in selected:
            record = {
                "source_trajectory_id": objectnav_source_trajectory_id(episode),
                "episode_id": episode.episode_id,
                "scene_id": episode.scene_id,
                "object_category": episode.object_category,
                "shard_path": episode.shard_path,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "status": "ok",
        "selection_manifest": str(manifest_path),
        "selected_episodes": len(selected),
        "requested_episodes": target_count,
        "source_episodes": total_episodes,
        "source_buckets": len(records_by_bucket),
        "split": config.split,
    }


def load_objectnav_selection_ids(config: DataConfig) -> set[str]:
    return {
        record.source_trajectory_id for record in load_objectnav_selection_records(config)
    }


def load_objectnav_selection_summary(config: DataConfig) -> dict[str, object]:
    records = load_objectnav_selection_records(config)
    missing_scene_paths = []
    scenes = set()
    objects = set()
    for record in records:
        scenes.add(record.scene_id)
        objects.add(record.object_category)
        scene_path = resolve_objectnav_scene_path(config, record.scene_id)
        if not scene_path.exists():
            missing_scene_paths.append(str(scene_path))
    return {
        "selection_manifest": str(
            resolve_data_path(Path(config.data_root), _require_selection_manifest(config))
        ),
        "selected_episodes": len(records),
        "unique_scenes": len(scenes),
        "unique_object_categories": len(objects),
        "missing_scene_paths": sorted(set(missing_scene_paths))[:20],
        "missing_scene_count": len(set(missing_scene_paths)),
    }


def load_objectnav_selection_records(config: DataConfig) -> list[ObjectNavSelectionRecord]:
    manifest_path = resolve_data_path(
        Path(config.data_root), _require_selection_manifest(config)
    )
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            records.append(
                ObjectNavSelectionRecord(
                    source_trajectory_id=str(raw["source_trajectory_id"]),
                    episode_id=str(raw["episode_id"]),
                    scene_id=str(raw["scene_id"]),
                    object_category=str(raw["object_category"]),
                    shard_path=str(raw["shard_path"]),
                )
            )
    return records


def objectnav_source_trajectory_id(episode: ObjectNavEpisode | object) -> str:
    return f"{getattr(episode, 'scene_id')}:{getattr(episode, 'episode_id')}"


def _load_episode_payload(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"ObjectNav shard must contain a mapping: {path}")
    episodes = loaded["episodes"]
    if not isinstance(episodes, list):
        raise ValueError(f"ObjectNav shard episodes must be a list: {path}")
    return episodes


def _require_selection_manifest(config: DataConfig) -> str:
    if config.episode_selection_manifest is None:
        raise ValueError("DataConfig.episode_selection_manifest is required")
    return config.episode_selection_manifest
