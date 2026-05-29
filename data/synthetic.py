"""Synthetic graph dataset for dependency-light training smoke tests."""

from __future__ import annotations

from configs.schema import DataConfig, PolicyConfig


class SyntheticGraphDataset:
    """Generate deterministic graph/action examples for smoke training."""

    def __init__(self, data_config: DataConfig, policy_config: PolicyConfig, seed: int):
        import torch

        self.size = data_config.max_episodes or 4
        self.input_dim = policy_config.input_dim
        self.num_actions = policy_config.num_actions
        generator = torch.Generator().manual_seed(seed)
        self.examples = []
        for idx in range(self.size):
            n_nodes = 2 + idx % 4
            nodes = torch.randn(n_nodes, self.input_dim, generator=generator)
            target = int(idx % self.num_actions)
            self.examples.append((nodes, target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        nodes, target = self.examples[index]
        return {
            "episode_id": f"synthetic_{index}",
            "goal_text": "synthetic goal",
            "graph_nodes": nodes,
            "target_action": target,
        }


def collate_synthetic_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    import torch

    nodes = [item["graph_nodes"].float() for item in batch]
    max_nodes = max(node.shape[0] for node in nodes)
    feature_dim = nodes[0].shape[1]
    padded = torch.zeros(len(nodes), max_nodes, feature_dim)
    mask = torch.zeros(len(nodes), max_nodes, dtype=torch.bool)
    for idx, node in enumerate(nodes):
        padded[idx, : node.shape[0]] = node
        mask[idx, : node.shape[0]] = True
    return {
        "graph_nodes": padded,
        "graph_mask": mask,
        "target_action": torch.as_tensor(
            [int(item["target_action"]) for item in batch], dtype=torch.long
        ),
    }
