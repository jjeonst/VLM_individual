"""Loader and audit helpers for Habitat ObjectNav episode shards."""

from __future__ import annotations

import gzip
import json
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
    return scene_root / scene_id


def load_objectnav_summary(config: DataConfig, *, sample_episodes: int = 1) -> dict[str, object]:
    dataset = HabitatObjectNavDataset(config)
    return dataset.summary(sample_episodes=sample_episodes)


def _load_episode_payload(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"ObjectNav shard must contain a mapping: {path}")
    episodes = loaded["episodes"]
    if not isinstance(episodes, list):
        raise ValueError(f"ObjectNav shard episodes must be a list: {path}")
    return episodes
