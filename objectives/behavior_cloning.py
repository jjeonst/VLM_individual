"""Behavior-cloning objective for discrete Habitat actions."""

from __future__ import annotations

from configs.schema import BehaviorCloningConfig


class BehaviorCloningObjective:
    """Compute cross-entropy loss for expert action labels."""

    def __init__(self, config: BehaviorCloningConfig):
        import torch

        self.config = config
        weight = None
        if config.class_weights is not None:
            weight = torch.as_tensor(config.class_weights, dtype=torch.float32)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=weight, label_smoothing=config.label_smoothing, reduction="none"
        )

    def to(self, device: object) -> "BehaviorCloningObjective":
        if self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(device)
        return self

    def __call__(self, logits, target, sample_weight=None, target_mask=None):
        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.reshape(-1)
            if sample_weight is not None:
                sample_weight = sample_weight.reshape(-1)
            if target_mask is not None:
                target_mask = target_mask.reshape(-1)
        losses = self.loss_fn(logits, target)
        weights = sample_weight if sample_weight is not None else losses.new_ones(losses.shape)
        if target_mask is not None:
            weights = weights * target_mask.to(weights.dtype)
        weighted = losses * weights
        return weighted.sum() / weights.sum().clamp_min(1.0)
