import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from configs.schema import DataConfig, ModelConfig, PolicyConfig, TopoVLMConfig
from data.habitat_dataset import HabitatGraphDataset, collate_graph_batch
from policies import build_policy
from training.runner import run_training


class PR2LTrajectoryTest(unittest.TestCase):
    def test_dataset_collates_token_nodes_and_node_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_dir = root / "graphs/pr2l"
            graph_dir.mkdir(parents=True)
            np.savez_compressed(
                graph_dir / "episode_0.npz",
                nodes=np.zeros((3, 4, 8), dtype="float32"),
                target_action=np.asarray(2, dtype="int64"),
                node_actions=np.asarray([0, 1, 2], dtype="int64"),
                action_mask=np.ones(3, dtype=bool),
            )
            manifest = root / "graphs/pr2l/manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "episode_id": "episode_0",
                        "split": "train",
                        "scene_id": "scene",
                        "goal_text": "chair",
                        "graph_path": "graphs/pr2l/episode_0.npz",
                        "embedding_path": "embeddings/episode_0.npy",
                        "target_action": 2,
                        "num_nodes": 3,
                        "prediction_target": "nodes",
                        "num_tokens": 4,
                        "feature_dim": 8,
                        "num_actions": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = HabitatGraphDataset(
                DataConfig(data_root=str(root), graph_manifest="graphs/pr2l/manifest.jsonl")
            )
            batch = collate_graph_batch([dataset[0]])

            self.assertEqual(tuple(batch["graph_nodes"].shape), (1, 3, 4, 8))
            self.assertEqual(batch["node_actions"].tolist(), [[0, 1, 2]])
            self.assertEqual(batch["action_mask"].tolist(), [[True, True, True]])

    def test_node_policy_predicts_per_node_logits(self):
        policy = build_policy(
            PolicyConfig(
                input_dim=8,
                hidden_dim=16,
                transformer_heads=4,
                transformer_layers=1,
                num_actions=4,
                prediction_target="nodes",
            )
        )
        logits = policy(torch.zeros(2, 3, 4, 8), torch.ones(2, 3, dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (2, 3, 4))

    def test_synthetic_pr2l_training_smoke_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TopoVLMConfig(
                config_name="habitat/pr2l_habitat_bc_faithful",
                run_name="test_pr2l_synthetic",
                output_root=tmpdir,
                max_epochs=1,
                data=DataConfig(
                    synthetic_debug=True,
                    max_episodes=2,
                    batch_size=2,
                    num_workers=0,
                ),
                model=ModelConfig(
                    policy=PolicyConfig(
                        input_dim=8,
                        hidden_dim=16,
                        transformer_heads=4,
                        transformer_layers=1,
                        num_actions=4,
                        prediction_target="nodes",
                    )
                ),
            )
            result = run_training(cfg)

            self.assertEqual(result["status"], "ok")
            self.assertTrue((Path(tmpdir) / "test_pr2l_synthetic/seed_42/model.pt").exists())


if __name__ == "__main__":
    unittest.main()
