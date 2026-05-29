"""Checkpoint and checkpoint manifest writing."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from configs.schema import TopoVLMConfig


def save_checkpoint(
    *,
    output_root: Path,
    cfg: TopoVLMConfig,
    model: object,
    optimizer: object,
    epoch: int,
    metrics: dict[str, float],
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
        "status": "smoke" if cfg.debug else "needs_audit",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_name": cfg.config_name,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "epoch": epoch,
        "checkpoint_file": checkpoint_path.name,
        "metrics": metrics,
        "model_family": cfg.model.policy.type,
        "vlm_backend": cfg.model.vlm.backend,
        "vlm_model_id": cfg.model.vlm.model_id,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint_path
