"""Path and cache preflight checks for Habitat TopoVLM configs."""

from __future__ import annotations

from pathlib import Path

from configs.schema import TopoVLMConfig
from data.habitat_manifest import (
    load_episode_records,
    load_graph_records,
    resolve_data_path,
    resolve_materialization_data_root,
)
from data.habitat_objectnav import load_objectnav_summary
from data.habitat_objectnav import load_objectnav_selection_records
from data.habitat_objectnav import load_objectnav_selection_summary
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


def run_objectnav_selection_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    if cfg.data.episode_selection_manifest is None:
        raise ValueError("DataConfig.episode_selection_manifest is required")
    selection_manifest = resolve_data_path(
        Path(cfg.data.data_root), cfg.data.episode_selection_manifest
    )
    missing = []
    if not selection_manifest.exists():
        missing.append(str(selection_manifest))
        summary = {
            "selection_manifest": str(selection_manifest),
            "selected_episodes": 0,
        }
    else:
        summary = load_objectnav_selection_summary(cfg.data)
        missing.extend(summary["missing_scene_paths"])
    if missing and not allow_missing_data:
        raise FileNotFoundError(f"Missing ObjectNav selection inputs: {missing[:5]}")
    return {
        "status": "ok" if not missing else "missing_allowed",
        "config_name": cfg.config_name,
        "objectnav_selection": summary,
        "missing_inputs": missing[:20],
        "missing_input_count": len(missing),
    }


def run_pr2l_manifest_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    source_data_root = Path(cfg.data.data_root)
    audit_data_root = resolve_materialization_data_root(cfg.data.data_root)
    episode_manifest = resolve_data_path(audit_data_root, cfg.data.episodes_manifest)
    if not episode_manifest.exists():
        if allow_missing_data:
            return {
                "status": "missing_allowed",
                "config_name": cfg.config_name,
                "source_data_root": str(source_data_root),
                "audit_data_root": str(audit_data_root),
                "episode_manifest": str(episode_manifest),
                "records": 0,
                "missing_payloads": [str(episode_manifest)],
                "missing_payload_count": 1,
            }
        raise FileNotFoundError(episode_manifest)
    all_records = load_episode_records(episode_manifest)
    records = all_records
    if cfg.data.max_episodes is not None:
        records = records[: cfg.data.max_episodes]
    selected_source_ids = set()
    missing_selected_source_ids = []
    if cfg.data.episode_selection_manifest is not None:
        selection_records = load_objectnav_selection_records(cfg.data)
        selected_source_ids = {record.source_trajectory_id for record in selection_records}
        materialized_source_ids = {
            record.source_trajectory_id
            for record in all_records
            if record.source_trajectory_id is not None
        }
        missing_selected_source_ids = sorted(
            selected_source_ids.difference(materialized_source_ids)
        )
        if missing_selected_source_ids and not allow_missing_data:
            raise FileNotFoundError(
                "Missing selected source episodes in PR2L manifest: "
                f"{missing_selected_source_ids[:5]}"
            )
    missing_payloads = []
    scenes = set()
    objects = set()
    for record in records:
        scenes.add(record.scene_id)
        if record.object_category is not None:
            objects.add(record.object_category)
        for relative_path in (record.rgb_path, record.actions_path):
            payload_path = resolve_data_path(audit_data_root, relative_path)
            if not payload_path.exists():
                missing_payloads.append(str(payload_path))
    if missing_payloads and not allow_missing_data:
        raise FileNotFoundError(f"Missing PR2L trajectory payloads: {missing_payloads[:5]}")
    missing_count = len(missing_payloads) + len(missing_selected_source_ids)
    return {
        "status": "ok" if missing_count == 0 else "missing_allowed",
        "config_name": cfg.config_name,
        "source_data_root": str(source_data_root),
        "audit_data_root": str(audit_data_root),
        "episode_manifest": str(episode_manifest),
        "records": len(records),
        "unique_scenes": len(scenes),
        "unique_object_categories": len(objects),
        "selected_source_episodes": len(selected_source_ids),
        "missing_selected_source_ids": missing_selected_source_ids[:20],
        "missing_selected_source_count": len(missing_selected_source_ids),
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


def run_vlm_auth_audit(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    if cfg.model.vlm.backend != "prismatic":
        raise ValueError(f"Unsupported VLM backend: {cfg.model.vlm.backend}")
    from encoders.prismatic import inspect_prismatic_hf_auth

    audit = inspect_prismatic_hf_auth(cfg.model.vlm)
    missing = []
    if audit["requires_private_hf_auth"] and not audit["token_available"]:
        missing.append("hf_token")
    if missing and not allow_missing_data:
        raise FileNotFoundError("Missing Prismatic Hugging Face token for gated LLM metadata")
    return {
        "status": "ok" if not missing else "missing_allowed",
        "config_name": cfg.config_name,
        "vlm_auth": audit,
        "missing_inputs": missing,
        "missing_input_count": len(missing),
    }


def run_cache_audit(cfg: TopoVLMConfig, *, allow_missing_data: bool = False) -> dict[str, object]:
    source_data_root = Path(cfg.data.data_root)
    audit_data_root = resolve_materialization_data_root(cfg.data.data_root)
    graph_manifest = resolve_data_path(audit_data_root, cfg.data.graph_manifest)
    if not graph_manifest.exists():
        if allow_missing_data:
            return {
                "status": "missing_allowed",
                "config_name": cfg.config_name,
                "source_data_root": str(source_data_root),
                "audit_data_root": str(audit_data_root),
                "graph_manifest": str(graph_manifest),
                "records": 0,
                "missing_graphs": [str(graph_manifest)],
                "missing_graph_count": 1,
            }
        raise FileNotFoundError(graph_manifest)
    records = load_graph_records(graph_manifest)
    expected_records = _expected_cache_records(cfg, audit_data_root)
    missing_graphs = []
    for record in records:
        graph_path = resolve_data_path(audit_data_root, record.graph_path)
        if not graph_path.exists():
            missing_graphs.append(str(graph_path))
    incomplete_count = max(expected_records - len(records), 0) if expected_records is not None else 0
    if (missing_graphs or incomplete_count) and not allow_missing_data:
        if incomplete_count:
            raise FileNotFoundError(
                f"Graph manifest has {len(records)} records but expected {expected_records}"
            )
        raise FileNotFoundError(f"Missing graph payloads: {missing_graphs[:5]}")
    missing_count = len(missing_graphs) + incomplete_count
    return {
        "status": "ok" if missing_count == 0 else "missing_allowed",
        "config_name": cfg.config_name,
        "source_data_root": str(source_data_root),
        "audit_data_root": str(audit_data_root),
        "graph_manifest": str(graph_manifest),
        "records": len(records),
        "expected_records": expected_records,
        "incomplete_record_count": incomplete_count,
        "missing_graphs": missing_graphs,
        "missing_graph_count": len(missing_graphs),
    }


def _expected_cache_records(cfg: TopoVLMConfig, audit_data_root: Path) -> int | None:
    episode_manifest = resolve_data_path(audit_data_root, cfg.data.episodes_manifest)
    if not episode_manifest.exists():
        return None
    records = load_episode_records(episode_manifest)
    if cfg.data.max_episodes is not None:
        records = records[: cfg.data.max_episodes]
    return len(records)
