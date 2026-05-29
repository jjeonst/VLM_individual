"""Offline policy evaluation over cached Habitat graph records."""

from __future__ import annotations

from pathlib import Path

from configs.schema import TopoVLMConfig
from data.habitat_dataset import HabitatGraphDataset, collate_graph_batch
from policies import build_policy


def run_offline_policy_eval(cfg: TopoVLMConfig, checkpoint_dir: str) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    checkpoint_path = Path(checkpoint_dir) / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    dataset = HabitatGraphDataset(cfg.data)
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
            target = batch["target_action"].to(device)
            correct += int((pred == target).sum().item())
            total += int(target.numel())
    return {
        "status": "ok",
        "checkpoint": str(checkpoint_path),
        "examples": total,
        "action_accuracy": correct / total if total else 0.0,
    }
