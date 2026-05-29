"""Path and cache preflight checks for Habitat TopoVLM configs."""

from __future__ import annotations

from pathlib import Path

from configs.schema import TopoVLMConfig
from data.habitat_manifest import load_graph_records, resolve_data_path


def run_data_preflight(
    cfg: TopoVLMConfig, *, allow_missing_data: bool = False
) -> dict[str, object]:
    repo_root = Path.cwd()
    data_root = Path(cfg.data.data_root)
    habitat_config = repo_root / cfg.data.habitat_config
    episode_manifest = resolve_data_path(data_root, cfg.data.episodes_manifest)
    graph_manifest = resolve_data_path(data_root, cfg.data.graph_manifest)
    vlm_weights = Path(cfg.model.vlm.weights_path)
    checks = {
        "data_root": data_root.exists(),
        "habitat_config": habitat_config.exists(),
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
            "episode_manifest": str(episode_manifest),
            "graph_manifest": str(graph_manifest),
            "vlm_weights_path": str(vlm_weights),
        },
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
