"""Offline policy evaluation over cached Habitat graph records."""

from __future__ import annotations

import json
import os
from pathlib import Path

from configs.schema import TopoVLMConfig
from topovlm_data.habitat_dataset import HabitatGraphDataset, collate_graph_batch
from policies import build_policy
from utils.checkpoint_io import resolve_source_commit


def run_offline_policy_eval(cfg: TopoVLMConfig, checkpoint_dir: str) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    checkpoint_path = Path(checkpoint_dir) / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    dataset = HabitatGraphDataset(cfg.data)
    majority_baseline = _compute_majority_action_baseline(
        dataset, cfg.model.policy.prediction_target
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_graph_batch,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg.model.policy).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state["model"])
    policy.eval()

    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            logits = policy(batch["graph_nodes"].to(device), batch["graph_mask"].to(device))
            pred = logits.argmax(dim=-1)
            if logits.ndim == 3:
                target = batch["node_actions"].to(device)
                mask = batch["action_mask"].to(device)
                correct += int(((pred == target) & mask).sum().item())
                total += int(mask.sum().item())
            else:
                target = batch["target_action"].to(device)
                correct += int((pred == target).sum().item())
                total += int(target.numel())
    result = {
        "status": "ok",
        "checkpoint": str(checkpoint_path),
        "examples": total,
        "action_accuracy": correct / total if total else 0.0,
        "majority_class_baseline": majority_baseline,
    }
    result_manifest = _write_artifact_result_manifest(cfg, result)
    if result_manifest is not None:
        result["result_manifest"] = str(result_manifest)
    return result


def _write_artifact_result_manifest(
    cfg: TopoVLMConfig, result: dict[str, object]
) -> Path | None:
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if artifact_dir is None:
        return None
    manifest_path = Path(artifact_dir) / "result_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_type": "topovlm_validation_result_manifest",
        "schema_version": 1,
        "status": result["status"],
        "config_name": cfg.config_name,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "source_commit": resolve_source_commit(),
        "checkpoint": result["checkpoint"],
        "metrics": {
            "examples": result["examples"],
            "action_accuracy": result["action_accuracy"],
            "majority_class_baseline": result["majority_class_baseline"],
        },
        "durable_lane_roots": {
            "artifact_dir": str(Path(artifact_dir)),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _compute_majority_action_baseline(
    dataset: HabitatGraphDataset, prediction_target: str
) -> dict[str, object]:
    label_counts: dict[int, int] = {}
    total = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if prediction_target == "nodes" and "node_actions" in item and "action_mask" in item:
            labels = item["node_actions"][item["action_mask"]]
            for action in labels.tolist():
                action_id = int(action)
                label_counts[action_id] = label_counts.get(action_id, 0) + 1
                total += 1
        else:
            action_id = int(item["target_action"])
            label_counts[action_id] = label_counts.get(action_id, 0) + 1
            total += 1
    if total == 0:
        return {
            "name": "majority_class",
            "action": None,
            "examples": 0,
            "correct": 0,
            "action_accuracy": 0.0,
            "label_counts": {},
        }
    majority_action, majority_correct = sorted(
        label_counts.items(), key=lambda item: (-item[1], item[0])
    )[0]
    return {
        "name": "majority_class",
        "action": majority_action,
        "examples": total,
        "correct": majority_correct,
        "action_accuracy": majority_correct / total,
        "label_counts": {str(action): count for action, count in sorted(label_counts.items())},
    }
