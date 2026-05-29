"""Manifest records for Habitat episodes and cached topology graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HabitatEpisodeRecord:
    """Point to one recorded Habitat expert episode."""

    episode_id: str
    split: str
    scene_id: str
    goal_text: str
    rgb_path: str
    actions_path: str

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "HabitatEpisodeRecord":
        required = {"episode_id", "split", "scene_id", "goal_text", "rgb_path", "actions_path"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Missing episode manifest keys: {sorted(missing)}")
        return cls(
            episode_id=str(raw["episode_id"]),
            split=str(raw["split"]),
            scene_id=str(raw["scene_id"]),
            goal_text=str(raw["goal_text"]),
            rgb_path=str(raw["rgb_path"]),
            actions_path=str(raw["actions_path"]),
        )


@dataclass(frozen=True)
class GraphRecord:
    """Point to one cached topology graph and action target."""

    episode_id: str
    split: str
    scene_id: str
    goal_text: str
    graph_path: str
    embedding_path: str
    target_action: int
    num_nodes: int

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "GraphRecord":
        required = {
            "episode_id",
            "split",
            "scene_id",
            "goal_text",
            "graph_path",
            "embedding_path",
            "target_action",
            "num_nodes",
        }
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Missing graph manifest keys: {sorted(missing)}")
        return cls(
            episode_id=str(raw["episode_id"]),
            split=str(raw["split"]),
            scene_id=str(raw["scene_id"]),
            goal_text=str(raw["goal_text"]),
            graph_path=str(raw["graph_path"]),
            embedding_path=str(raw["embedding_path"]),
            target_action=int(raw["target_action"]),
            num_nodes=int(raw["num_nodes"]),
        )


def load_episode_records(path: Path) -> list[HabitatEpisodeRecord]:
    return [HabitatEpisodeRecord.from_dict(raw) for raw in _load_json_records(path)]


def load_graph_records(path: Path) -> list[GraphRecord]:
    return [GraphRecord.from_dict(raw) for raw in _load_json_records(path)]


def resolve_data_path(root: Path, path: str | Path) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    return root / target


def _load_json_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, dict) and "records" in loaded:
        loaded = loaded["records"]
    if not isinstance(loaded, list):
        raise ValueError(f"Manifest must contain a list of records: {path}")
    return loaded
