"""Path and cache preflight checks for Habitat TopoVLM configs."""

from __future__ import annotations

from pathlib import Path

from configs.schema import TopoVLMConfig
from data.habitat_manifest import load_episode_records, load_graph_records, resolve_data_path
from data.habitat_objectnav import load_objectnav_summary
from data.habitat_web import (
    load_habitat_web_selection_summary,
    load_habitat_web_inventory,
    load_habitat_web_summary,
    is_git_lfs_pointer,
)


def run_data_preflight(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    repo_root = Path.cwd()
    data_root = Path(cfg.data.data_root)
    habitat_config = repo_root / cfg.data.habitat_config
    objectnav_dataset_dir = resolve_data_path(data_root, cfg.data.objectnav_dataset_dir)
    objectnav_split_index = objectnav_dataset_dir / cfg.data.split / f"{cfg.data.split}.json.gz"
    objectnav_content_dir = objectnav_dataset_dir / cfg.data.split / "content"
    scene_dataset_dir = resolve_data_path(data_root, cfg.data.scene_dataset_dir)
    scene_dataset_config = resolve_data_path(data_root, cfg.data.scene_dataset_config)
    episode_manifest = resolve_data_path(data_root, cfg.data.episodes_manifest)
    graph_manifest = resolve_data_path(data_root, cfg.data.graph_manifest)
    vlm_weights = Path(cfg.model.vlm.weights_path)
    content_shards = (
        sorted(objectnav_content_dir.glob("*.json.gz")) if objectnav_content_dir.exists() else []
    )
    materialized_content_shards = [
        path for path in content_shards if not is_git_lfs_pointer(path)
    ]
    checks = {
        "data_root": data_root.exists(),
        "habitat_config": habitat_config.exists(),
        "objectnav_dataset_dir": objectnav_dataset_dir.exists(),
        "objectnav_split_index": objectnav_split_index.exists(),
        "objectnav_content_dir": objectnav_content_dir.exists(),
        "objectnav_content_shards": bool(content_shards),
        "objectnav_content_shards_materialized": bool(materialized_content_shards),
        "scene_dataset_dir": scene_dataset_dir.exists(),
        "scene_dataset_config": scene_dataset_config.exists(),
        "episode_manifest": episode_manifest.exists(),
        "graph_manifest": graph_manifest.exists(),
        "vlm_weights_path": vlm_weights.exists(),
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing and not allow_missing_data:
        raise FileNotFoundError(f"Missing TopoVLM inputs: {', '.join(missing)}")
    return {
        "status": "ok" if not missing else "missing_allowed",
        "config_name": cfg.config_name,
        "checks": checks,
        "paths": {
            "data_root": str(data_root),
            "habitat_config": str(habitat_config),
            "objectnav_dataset_dir": str(objectnav_dataset_dir),
            "objectnav_split_index": str(objectnav_split_index),
            "objectnav_content_dir": str(objectnav_content_dir),
            "scene_dataset_dir": str(scene_dataset_dir),
            "scene_dataset_config": str(scene_dataset_config),
            "episode_manifest": str(episode_manifest),
            "graph_manifest": str(graph_manifest),
            "vlm_weights_path": str(vlm_weights),
        },
        "objectnav_content_shards": len(content_shards),
        "objectnav_materialized_content_shards": len(materialized_content_shards),
    }


def run_objectnav_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    scene_dataset_config = resolve_data_path(
        Path(cfg.data.data_root), cfg.data.scene_dataset_config
    )
    summary = load_objectnav_summary(cfg.data, sample_episodes=1)
    missing = []
    if not scene_dataset_config.exists():
        missing.append(str(scene_dataset_config))
    missing.extend(summary["missing_sample_scenes"])
    missing = sorted(set(missing))
    if missing and not allow_missing_data:
        raise FileNotFoundError(f"Missing ObjectNav/HM3D inputs: {missing[:5]}")
    return {
        "status": "ok" if not missing else "missing_allowed",
        "config_name": cfg.config_name,
        "scene_dataset_config": str(scene_dataset_config),
        "objectnav": summary,
    }


def run_pr2l_manifest_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    data_root = Path(cfg.data.data_root)
    episode_manifest = resolve_data_path(data_root, cfg.data.episodes_manifest)
    if not episode_manifest.exists():
        if allow_missing_data:
            return {
                "status": "missing_allowed",
                "episode_manifest": str(episode_manifest),
                "records": 0,
                "missing_payloads": [str(episode_manifest)],
            }
        raise FileNotFoundError(episode_manifest)
    records = load_episode_records(episode_manifest)
    if cfg.data.max_episodes is not None:
        records = records[: cfg.data.max_episodes]
    missing_payloads = []
    scenes = set()
    objects = set()
    for record in records:
        scenes.add(record.scene_id)
        if record.object_category is not None:
            objects.add(record.object_category)
        for relative_path in (record.rgb_path, record.actions_path):
            payload_path = resolve_data_path(data_root, relative_path)
            if not payload_path.exists():
                missing_payloads.append(str(payload_path))
    if missing_payloads and not allow_missing_data:
        raise FileNotFoundError(f"Missing PR2L trajectory payloads: {missing_payloads[:5]}")
    return {
        "status": "ok" if not missing_payloads else "missing_allowed",
        "config_name": cfg.config_name,
        "episode_manifest": str(episode_manifest),
        "records": len(records),
        "unique_scenes": len(scenes),
        "unique_object_categories": len(objects),
        "missing_payloads": missing_payloads[:20],
        "missing_payload_count": len(missing_payloads),
    }


def run_habitat_web_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    summary = load_habitat_web_summary(cfg.data, sample_episodes=4)
    schema_sample = None
    if cfg.data.split != "train_sample":
        try:
            schema_sample = load_habitat_web_summary(
                cfg.data, sample_episodes=4, split="train_sample"
            )
        except FileNotFoundError:
            schema_sample = None
    missing = []
    if not summary["split_index_materialized"]:
        missing.append(str(summary["split_index"]))
    if summary["materialized_content_shards"] == 0:
        missing.append(str(summary["content_dir"]))
    missing.extend(summary["missing_sample_scenes"])
    if missing and not allow_missing_data:
        raise FileNotFoundError(f"Missing Habitat-Web inputs: {missing[:5]}")
    return {
        "status": "ok" if not missing else "missing_allowed",
        "config_name": cfg.config_name,
        "habitat_web": summary,
        "schema_sample": schema_sample,
        "missing_inputs": missing[:20],
        "missing_input_count": len(missing),
    }


def run_habitat_web_scene_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    inventory = load_habitat_web_inventory(
        cfg.data, max_episodes=cfg.data.max_episodes
    )
    missing = []
    metadata_missing_count = 0
    if not inventory["split_index_materialized"]:
        missing.append(str(inventory["split_index"]))
        metadata_missing_count += 1
    if inventory["materialized_content_shards"] == 0:
        missing.append(str(inventory["content_dir"]))
        metadata_missing_count += 1
    missing.extend(inventory["missing_scene_paths"])
    if missing and not allow_missing_data:
        raise FileNotFoundError(f"Missing Habitat-Web scene inputs: {missing[:5]}")
    return {
        "status": "ok" if not missing else "missing_allowed",
        "config_name": cfg.config_name,
        "habitat_web_scene_inventory": inventory,
        "missing_inputs": missing[:20],
        "missing_input_count": metadata_missing_count + inventory["missing_scene_count"],
    }


def run_habitat_web_selection_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    missing = []
    if cfg.data.episode_selection_manifest is None:
        raise ValueError("DataConfig.episode_selection_manifest is required")
    selection_manifest = resolve_data_path(
        Path(cfg.data.data_root), cfg.data.episode_selection_manifest
    )
    if not selection_manifest.exists():
        missing.append(str(selection_manifest))
        summary = {
            "selection_manifest": str(selection_manifest),
            "selected_episodes": 0,
        }
    else:
        summary = load_habitat_web_selection_summary(cfg.data)
        missing.extend(summary["missing_scene_paths"])
    if missing and not allow_missing_data:
        raise FileNotFoundError(f"Missing Habitat-Web selection inputs: {missing[:5]}")
    missing_input_count = 0 if not missing else len(missing)
    if "missing_scene_count" in summary:
        missing_input_count = summary["missing_scene_count"]
        if not selection_manifest.exists():
            missing_input_count += 1
    return {
        "status": "ok" if not missing else "missing_allowed",
        "config_name": cfg.config_name,
        "habitat_web_selection": summary,
        "missing_inputs": missing[:20],
        "missing_input_count": missing_input_count,
    }


def run_cache_audit(cfg: TopoVLMConfig, *, allow_missing_data: bool = False) -> dict[str, object]:
    data_root = Path(cfg.data.data_root)
    graph_manifest = resolve_data_path(data_root, cfg.data.graph_manifest)
    if not graph_manifest.exists():
        if allow_missing_data:
            return {
                "status": "missing_allowed",
                "graph_manifest": str(graph_manifest),
                "records": 0,
                "missing_graphs": [],
            }
        raise FileNotFoundError(graph_manifest)
    records = load_graph_records(graph_manifest)
    missing_graphs = []
    for record in records:
        graph_path = resolve_data_path(data_root, record.graph_path)
        if not graph_path.exists():
            missing_graphs.append(str(graph_path))
    if missing_graphs and not allow_missing_data:
        raise FileNotFoundError(f"Missing graph payloads: {missing_graphs[:5]}")
    return {
        "status": "ok" if not missing_graphs else "missing_allowed",
        "graph_manifest": str(graph_manifest),
        "records": len(records),
        "missing_graphs": missing_graphs,
    }
