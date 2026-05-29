"""Training objectives."""

from configs.schema import ObjectivesConfig
from objectives.behavior_cloning import BehaviorCloningObjective


def build_objective(config: ObjectivesConfig) -> BehaviorCloningObjective:
    if config.names != ["behavior_cloning"]:
        raise ValueError(f"Unsupported objectives: {config.names}")
    return BehaviorCloningObjective(config.behavior_cloning)


__all__ = ["BehaviorCloningObjective", "build_objective"]
