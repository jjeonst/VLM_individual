"""Behavior-cloning training loop for cached Habitat topology graphs."""

from __future__ import annotations

from pathlib import Path

from configs.schema import TopoVLMConfig
from data.habitat_dataset import HabitatGraphDataset, collate_graph_batch
from data.synthetic import SyntheticGraphDataset, collate_synthetic_batch
from objectives import build_objective
from policies import build_policy
from utils.checkpoint_io import save_checkpoint
from utils.randomness import seed_everything


def run_training(cfg: TopoVLMConfig) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset, collate_fn = _build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
    )
    policy = build_policy(cfg.model.policy).to(device)
    objective = build_objective(cfg.objectives).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        policy.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            nodes = batch["graph_nodes"].to(device)
            mask = batch["graph_mask"].to(device)
            target = batch["target_action"].to(device)
            logits = policy(nodes, mask)
            loss = objective(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(target.numel())
            total_examples += int(target.numel())
        mean_loss = total_loss / max(total_examples, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss, "examples": total_examples})
        if epoch % cfg.save_every_epochs == 0:
            save_checkpoint(
                output_root=Path(cfg.output_root),
                cfg=cfg,
                model=policy,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": mean_loss},
            )
    return {
        "status": "ok",
        "config_name": cfg.config_name,
        "run_name": cfg.run_name,
        "output_root": str(Path(cfg.output_root)),
        "epochs": cfg.max_epochs,
        "history": history,
        "synthetic_debug": cfg.data.synthetic_debug,
    }


def _build_dataset(cfg: TopoVLMConfig):
    if cfg.data.synthetic_debug:
        return SyntheticGraphDataset(cfg.data, cfg.model.policy, cfg.seed), collate_synthetic_batch
    return HabitatGraphDataset(cfg.data), collate_graph_batch
