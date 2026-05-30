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
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    history = []
    last_checkpoint = None
    for epoch in range(1, cfg.max_epochs + 1):
        policy.train()
        total_loss = 0.0
        total_examples = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            nodes = batch["graph_nodes"].to(device)
            mask = batch["graph_mask"].to(device)
            logits = policy(nodes, mask)
            if logits.ndim == 3:
                target = batch["node_actions"].to(device)
                action_mask = batch["action_mask"].to(device)
                sample_weight = _build_node_action_weights(
                    target, action_mask, cfg.objectives.behavior_cloning
                )
                loss = objective(logits, target, sample_weight, action_mask)
                examples = int(action_mask.sum().item())
            else:
                target = batch["target_action"].to(device)
                loss = objective(logits, target)
                examples = int(target.numel())
            (loss / cfg.gradient_accumulation_steps).backward()
            if step % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * examples
            total_examples += examples
        if len(loader) % cfg.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        mean_loss = total_loss / max(total_examples, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss, "examples": total_examples})
        if epoch % cfg.save_every_epochs == 0:
            last_checkpoint = save_checkpoint(
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
        "checkpoint_path": str(last_checkpoint) if last_checkpoint is not None else None,
        "checkpoint_manifest": (
            str(last_checkpoint.parent / "checkpoint_manifest.json")
            if last_checkpoint is not None
            else None
        ),
    }


def _build_dataset(cfg: TopoVLMConfig):
    if cfg.data.synthetic_debug:
        return SyntheticGraphDataset(cfg.data, cfg.model.policy, cfg.seed), collate_synthetic_batch
    return HabitatGraphDataset(cfg.data), collate_graph_batch


def _build_node_action_weights(target, action_mask, config):
    import torch

    weights = torch.ones_like(target, dtype=torch.float32)
    if config.stop_turn_weight != 1.0:
        stop_turn = target.new_zeros(target.shape, dtype=action_mask.dtype)
        for action_id in config.stop_turn_action_ids:
            stop_turn = stop_turn | (target == int(action_id))
        weights = torch.where(stop_turn, weights * float(config.stop_turn_weight), weights)
    if config.inflection_weight != 1.0 and target.shape[1] > 1:
        inflection = target[:, 1:] != target[:, :-1]
        weights[:, 1:] = torch.where(
            inflection, weights[:, 1:] * float(config.inflection_weight), weights[:, 1:]
        )
    return weights * action_mask.float()
