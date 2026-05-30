"""Checkpoint and checkpoint manifest writing."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from configs.schema import TopoVLMConfig


def save_checkpoint(
    *,
    output_root: Path,
    cfg: TopoVLMConfig,
    model: object,
    optimizer: object,
    epoch: int,
    metrics: dict[str, float],
    wandb_run_id: Optional[str] = None,
    wandb_run_url: Optional[str] = None,
) -> Path:
    import torch

    run_dir = output_root / cfg.run_name / f"seed_{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": dataclasses.asdict(cfg),
        },
        checkpoint_path,
    )
    manifest_path = run_dir / "checkpoint_manifest.json"
    manifest = {
        "artifact_type": "topovlm_checkpoint_manifest",
        "schema_version": 1,
        "status": "smoke" if cfg.debug or cfg.data.synthetic_debug else "needs_audit",
        "finality_class": "smoke" if cfg.debug or cfg.data.synthetic_debug else "candidate",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_name": cfg.config_name,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "epoch": epoch,
        "checkpoint_file": checkpoint_path.name,
        "selected_checkpoint_file": checkpoint_path.name,
        "checkpoint_files": [
            {
                "path": checkpoint_path.name,
                "role": "latest",
                "epoch": epoch,
                "metrics": metrics,
            }
        ],
        "metrics": metrics,
        "source_commit": _resolve_source_commit(),
        "data": {
            "dataset_name": cfg.data.dataset_name,
            "cache_format": cfg.data.cache_format,
            "episodes_manifest": cfg.data.episodes_manifest,
            "graph_manifest": cfg.data.graph_manifest,
            "graph_cache_dir": cfg.data.graph_cache_dir,
            "embeddings_dir": cfg.data.embeddings_dir,
        },
        "training": {
            "max_epochs": cfg.max_epochs,
            "save_every_epochs": cfg.save_every_epochs,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
        "model_family": cfg.model.policy.type,
        "policy_prediction_target": cfg.model.policy.prediction_target,
        "vlm_backend": cfg.model.vlm.backend,
        "vlm_model_id": cfg.model.vlm.model_id,
        "vlm_representation": cfg.model.vlm.representation,
        "wandb": {
            "enabled": cfg.wandb,
            "entity": cfg.wandb_entity,
            "project": cfg.wandb_project,
            "group": cfg.wandb_group,
            "run_name": cfg.wandb_run_name,
            "run_id": wandb_run_id,
            "run_url": wandb_run_url,
            "contract_path": cfg.wandb_contract_path,
            "contract_role_id": cfg.wandb_contract_role_id,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint_path


def _resolve_source_commit() -> Optional[str]:
    if os.environ.get("SOURCE_COMMIT"):
        return os.environ["SOURCE_COMMIT"]
    stage_manifest = os.environ.get("RUN_ROOT")
    if stage_manifest is not None:
        manifest_path = Path(stage_manifest) / "stage_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return str(manifest["code"]["source_commit"])
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
