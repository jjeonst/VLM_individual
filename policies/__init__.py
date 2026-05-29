"""Policy modules."""

from configs.schema import PolicyConfig


def build_policy(config: PolicyConfig):
    if config.type == "graph_transformer_bc":
        from policies.graph_policy import GraphTransformerPolicy

        return GraphTransformerPolicy(config)
    raise ValueError(f"Unsupported policy type: {config.type}")


__all__ = ["build_policy"]
