"""PyTorch dataset for cached Habitat topology graphs."""

from __future__ import annotations

from pathlib import Path

from configs.schema import DataConfig
from data.habitat_manifest import load_graph_records, resolve_data_path


class HabitatGraphDataset:
    """Load cached topology graphs and target expert actions."""

    def __init__(self, config: DataConfig):
        import numpy as np

        self._np = np
        self.config = config
        self.data_root = Path(config.data_root)
        manifest = resolve_data_path(self.data_root, config.graph_manifest)
        self.records = load_graph_records(manifest)
        if config.max_episodes is not None:
            self.records = self.records[: config.max_episodes]
        if not self.records:
            raise ValueError(f"No graph records in {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        graph_path = resolve_data_path(self.data_root, record.graph_path)
        payload = self._np.load(graph_path)
        target_action = int(payload["target_action"])
        return {
            "episode_id": record.episode_id,
            "goal_text": record.goal_text,
            "graph_nodes": payload["nodes"].astype("float32"),
            "target_action": target_action,
        }


def collate_graph_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    import torch

    nodes = [torch.as_tensor(item["graph_nodes"], dtype=torch.float32) for item in batch]
    actions = torch.as_tensor([int(item["target_action"]) for item in batch], dtype=torch.long)
    max_nodes = max(node.shape[0] for node in nodes)
    feature_dim = nodes[0].shape[1]
    padded = torch.zeros(len(nodes), max_nodes, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(nodes), max_nodes, dtype=torch.bool)
    for idx, node in enumerate(nodes):
        n_nodes = node.shape[0]
        padded[idx, :n_nodes] = node
        mask[idx, :n_nodes] = True
    return {
        "episode_id": [item["episode_id"] for item in batch],
        "goal_text": [item["goal_text"] for item in batch],
        "graph_nodes": padded,
        "graph_mask": mask,
        "target_action": actions,
    }
