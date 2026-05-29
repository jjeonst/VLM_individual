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
            weight=weight, label_smoothing=config.label_smoothing
        )

    def to(self, device: object) -> "BehaviorCloningObjective":
        if self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(device)
        return self

    def __call__(self, logits, target):
        return self.loss_fn(logits, target)
