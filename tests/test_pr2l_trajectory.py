import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from configs.schema import DataConfig, ModelConfig, PolicyConfig, TopoVLMConfig, VLMConfig
from data.habitat_cache import build_habitat_graph_cache
from data.habitat_dataset import HabitatGraphDataset, collate_graph_batch
from evaluation.preflight import run_cache_audit, run_pr2l_manifest_audit
from policies import build_policy
from training.runner import run_training


class PR2LTrajectoryTest(unittest.TestCase):
    def test_pr2l_cache_builder_writes_node_action_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episodes/pr2l_habitat_web/train").mkdir(parents=True)
            (root / "rgb").mkdir()
            (root / "actions").mkdir()
            np.save(root / "rgb/episode_0.npy", np.zeros((3, 2, 2, 3), dtype="uint8"))
            np.save(root / "actions/episode_0.npy", np.asarray([1, 2, 0], dtype="int64"))
            (root / "episodes/pr2l_habitat_web/train/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": "episode_0",
                        "split": "train",
                        "scene_id": "scene",
                        "goal_text": "chair",
                        "rgb_path": "rgb/episode_0.npy",
                        "actions_path": "actions/episode_0.npy",
                        "source_dataset": "habitat_web",
                        "source_trajectory_id": "demo_0",
                        "object_category": "chair",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    cache_format="pr2l_token_trajectory",
                    episodes_manifest="episodes/pr2l_habitat_web/train/manifest.jsonl",
                    graph_manifest="graphs/pr2l/manifest.jsonl",
                    graph_cache_dir="graphs/pr2l",
                    embeddings_dir="embeddings/pr2l",
                    max_episodes=1,
                ),
                model=ModelConfig(
                    vlm=VLMConfig(
                        representation="pr2l_visual_tokens_last_two_layers",
                        projection="none",
                        output_dim=8,
                        weights_path=str(root / "fake_prismatic"),
                    ),
                    policy=PolicyConfig(input_dim=8, prediction_target="nodes"),
                ),
            )

            with patch("data.habitat_cache.build_vlm_encoder", return_value=_FakePR2LEncoder()):
                result = build_habitat_graph_cache(cfg)

            graph_payload = np.load(root / "graphs/pr2l/episode_0.npz")
            self.assertEqual(result["graphs_written"], 1)
            self.assertEqual(graph_payload["nodes"].shape[-2:], (4, 8))
            self.assertEqual(graph_payload["node_actions"].tolist(), [0])

    def test_pr2l_cache_builder_can_write_to_output_data_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "outputs/data/topovlm/habitat"
            (source_root / "episodes/pr2l_habitat_web/train").mkdir(parents=True)
            (source_root / "rgb").mkdir()
            (source_root / "actions").mkdir()
            np.save(source_root / "rgb/episode_0.npy", np.zeros((2, 2, 2, 3), dtype="uint8"))
            np.save(source_root / "actions/episode_0.npy", np.asarray([1, 0], dtype="int64"))
            (source_root / "episodes/pr2l_habitat_web/train/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": "episode_0",
                        "split": "train",
                        "scene_id": "scene",
                        "goal_text": "chair",
                        "rgb_path": "rgb/episode_0.npy",
                        "actions_path": "actions/episode_0.npy",
                        "source_dataset": "habitat_web",
                        "source_trajectory_id": "demo_0",
                        "object_category": "chair",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(source_root),
                    cache_format="pr2l_token_trajectory",
                    episodes_manifest="episodes/pr2l_habitat_web/train/manifest.jsonl",
                    graph_manifest="graphs/pr2l/manifest.jsonl",
                    graph_cache_dir="graphs/pr2l",
                    embeddings_dir="embeddings/pr2l",
                    max_episodes=1,
                ),
                model=ModelConfig(
                    vlm=VLMConfig(
                        representation="pr2l_visual_tokens_last_two_layers",
                        projection="none",
                        output_dim=8,
                        weights_path=str(source_root / "fake_prismatic"),
                    ),
                    policy=PolicyConfig(input_dim=8, prediction_target="nodes"),
                ),
            )

            with patch("data.habitat_cache.build_vlm_encoder", return_value=_FakePR2LEncoder()):
                with patch.dict(os.environ, {"TOPOVLM_DATA_OUTPUT_ROOT": str(output_root)}):
                    result = build_habitat_graph_cache(cfg)

            self.assertEqual(result["source_data_root"], str(source_root))
            self.assertEqual(result["output_data_root"], str(output_root))
            self.assertTrue((output_root / "graphs/pr2l/manifest.jsonl").exists())
            self.assertFalse((source_root / "graphs/pr2l/manifest.jsonl").exists())

    def test_pr2l_cache_builder_fails_before_encoder_load_without_hf_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "prism-dinosiglip+7b"
            weights.mkdir()
            (weights / "config.json").write_text(
                '{"model": {"llm_backbone_id": "llama2-7b-pure"}}\n',
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    cache_format="pr2l_token_trajectory",
                    episodes_manifest="episodes/pr2l_habitat_web/train/manifest.jsonl",
                ),
                model=ModelConfig(
                    vlm=VLMConfig(
                        representation="pr2l_visual_tokens_last_two_layers",
                        weights_path=str(weights),
                    )
                ),
            )

            with patch.dict(os.environ, {"HOME": tmpdir}, clear=True):
                with patch("data.habitat_cache.build_vlm_encoder") as build_encoder:
                    with self.assertRaisesRegex(FileNotFoundError, "meta-llama/Llama-2-7b-hf"):
                        build_habitat_graph_cache(cfg)

            build_encoder.assert_not_called()

    def test_pr2l_audits_can_read_materialization_output_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "outputs/data/topovlm/habitat"
            (output_root / "episodes/pr2l_habitat_web/train").mkdir(parents=True)
            (output_root / "rgb").mkdir()
            (output_root / "actions").mkdir()
            (output_root / "graphs/pr2l").mkdir(parents=True)
            (output_root / "embeddings/pr2l").mkdir(parents=True)
            np.save(output_root / "rgb/episode_0.npy", np.zeros((2, 2, 2, 3), dtype="uint8"))
            np.save(output_root / "actions/episode_0.npy", np.asarray([1, 0], dtype="int64"))
            np.savez_compressed(
                output_root / "graphs/pr2l/episode_0.npz",
                nodes=np.zeros((1, 4, 8), dtype="float32"),
                target_action=np.asarray(0, dtype="int64"),
            )
            np.save(output_root / "embeddings/pr2l/episode_0.npy", np.zeros((1, 8), dtype="float32"))
            (output_root / "episodes/pr2l_habitat_web/train/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": "episode_0",
                        "split": "train",
                        "scene_id": "scene",
                        "goal_text": "chair",
                        "rgb_path": "rgb/episode_0.npy",
                        "actions_path": "actions/episode_0.npy",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_root / "graphs/pr2l/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": "episode_0",
                        "split": "train",
                        "scene_id": "scene",
                        "goal_text": "chair",
                        "graph_path": "graphs/pr2l/episode_0.npz",
                        "embedding_path": "embeddings/pr2l/episode_0.npy",
                        "target_action": 0,
                        "num_nodes": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(source_root),
                    episodes_manifest="episodes/pr2l_habitat_web/train/manifest.jsonl",
                    graph_manifest="graphs/pr2l/manifest.jsonl",
                )
            )

            with patch.dict(os.environ, {"TOPOVLM_DATA_OUTPUT_ROOT": str(output_root)}):
                episode_result = run_pr2l_manifest_audit(cfg)
                cache_result = run_cache_audit(cfg)

            self.assertEqual(episode_result["audit_data_root"], str(output_root))
            self.assertEqual(episode_result["records"], 1)
            self.assertEqual(episode_result["missing_payload_count"], 0)
            self.assertEqual(cache_result["audit_data_root"], str(output_root))
            self.assertEqual(cache_result["records"], 1)
            self.assertEqual(cache_result["expected_records"], 1)
            self.assertEqual(cache_result["incomplete_record_count"], 0)
            self.assertEqual(cache_result["missing_graph_count"], 0)

    def test_cache_audit_detects_incomplete_graph_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episodes/pr2l_habitat_web/train").mkdir(parents=True)
            (root / "graphs/pr2l").mkdir(parents=True)
            for episode_id in ("episode_0", "episode_1"):
                np.save(root / f"rgb_{episode_id}.npy", np.zeros((1, 2, 2, 3), dtype="uint8"))
                np.save(root / f"actions_{episode_id}.npy", np.asarray([0], dtype="int64"))
            (root / "episodes/pr2l_habitat_web/train/manifest.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "episode_id": episode_id,
                            "split": "train",
                            "scene_id": "scene",
                            "goal_text": "chair",
                            "rgb_path": f"rgb_{episode_id}.npy",
                            "actions_path": f"actions_{episode_id}.npy",
                        }
                    )
                    + "\n"
                    for episode_id in ("episode_0", "episode_1")
                ),
                encoding="utf-8",
            )
            np.savez_compressed(root / "graphs/pr2l/episode_0.npz", nodes=np.zeros((1, 1, 2)))
            (root / "graphs/pr2l/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": "episode_0",
                        "split": "train",
                        "scene_id": "scene",
                        "goal_text": "chair",
                        "graph_path": "graphs/pr2l/episode_0.npz",
                        "embedding_path": "embeddings/pr2l/episode_0.npy",
                        "target_action": 0,
                        "num_nodes": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    episodes_manifest="episodes/pr2l_habitat_web/train/manifest.jsonl",
                    graph_manifest="graphs/pr2l/manifest.jsonl",
                )
            )

            result = run_cache_audit(cfg, allow_missing_data=True)

            self.assertEqual(result["status"], "missing_allowed")
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["expected_records"], 2)
            self.assertEqual(result["incomplete_record_count"], 1)

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
                config_name="habitat/pr2l_hm3d_bc",
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
            manifest_path = Path(result["checkpoint_manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "ok")
            self.assertTrue((Path(tmpdir) / "test_pr2l_synthetic/seed_42/model.pt").exists())
            self.assertEqual(manifest["status"], "smoke")
            self.assertEqual(manifest["finality_class"], "smoke")
            self.assertEqual(manifest["data"]["cache_format"], "single_action_graph")
            self.assertEqual(manifest["wandb"]["entity"], "topovlm")
            self.assertEqual(manifest["wandb"]["contract_role_id"], "habitat_bc")
            self.assertEqual(manifest["selected_checkpoint_file"], "model.pt")

    def test_wandb_training_logs_and_records_run_identity(self):
        fake_wandb = _FakeWandbModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TopoVLMConfig(
                config_name="habitat/pr2l_hm3d_bc",
                run_name="test_pr2l_wandb",
                output_root=tmpdir,
                max_epochs=1,
                wandb=True,
                wandb_group="pr2l_prismatic_policy",
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
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                result = run_training(cfg)
            manifest = json.loads(
                Path(result["checkpoint_manifest"]).read_text(encoding="utf-8")
            )

            self.assertEqual(fake_wandb.init_kwargs["entity"], "topovlm")
            self.assertEqual(fake_wandb.init_kwargs["project"], "TopoVLM")
            self.assertEqual(fake_wandb.init_kwargs["group"], "pr2l_prismatic_policy")
            self.assertTrue(
                fake_wandb.init_kwargs["name"].startswith("pr2l_prismatic_policy_seed42_")
            )
            self.assertEqual(
                fake_wandb.run.logs[0]["payload"]["train_examples"],
                result["history"][0]["examples"],
            )
            self.assertEqual(fake_wandb.run.logs[0]["step"], 1)
            self.assertTrue(fake_wandb.run.finished)
            self.assertEqual(manifest["wandb"]["enabled"], True)
            self.assertEqual(manifest["wandb"]["run_id"], "fake-run-id")
            self.assertEqual(manifest["wandb"]["run_url"], "https://wandb.local/fake-run-id")


class _FakePR2LEncoder:
    def encode_image_goal_tokens(self, image, goal_text):
        return {
            "tokens": np.ones((4, 8), dtype="float32"),
            "generated_text": "synthetic",
        }


class _FakeWandbRun:
    id = "fake-run-id"
    url = "https://wandb.local/fake-run-id"

    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, payload, step=None):
        self.logs.append({"payload": dict(payload), "step": step})

    def finish(self):
        self.finished = True


class _FakeWandbModule:
    def __init__(self):
        self.run = _FakeWandbRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


if __name__ == "__main__":
    unittest.main()
