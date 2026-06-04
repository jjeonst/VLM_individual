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
from data.habitat_cache import (
    _fit_pca_projection,
    _load_or_fit_projection,
    build_habitat_graph_cache,
)
from data.habitat_dataset import HabitatGraphDataset, collate_graph_batch
from evaluation.preflight import run_cache_audit, run_pr2l_manifest_audit
from policies import build_policy
from training.runner import run_training


class PR2LTrajectoryTest(unittest.TestCase):
    def _build_pr2l_selection_gap_config(self, root: Path) -> TopoVLMConfig:
        manifest_rel = "episodes/pr2l_hm3d_objectnav/train/manifest.jsonl"
        selection_rel = "episode_selections/pr2l_hm3d_objectnav/train_subset.jsonl"
        (root / "episodes/pr2l_hm3d_objectnav/train").mkdir(parents=True)
        (root / "episode_selections/pr2l_hm3d_objectnav").mkdir(parents=True)
        np.save(root / "rgb_episode_0.npy", np.zeros((1, 2, 2, 3), dtype="uint8"))
        np.save(root / "actions_episode_0.npy", np.asarray([0], dtype="int64"))
        (root / manifest_rel).write_text(
            json.dumps(
                {
                    "episode_id": "episode_0",
                    "split": "train",
                    "scene_id": "scene_a/scene.glb",
                    "goal_text": "chair",
                    "rgb_path": "rgb_episode_0.npy",
                    "actions_path": "actions_episode_0.npy",
                    "source_dataset": "hm3d_objectnav_shortest_path",
                    "source_trajectory_id": "scene_a/scene.glb:0",
                    "object_category": "chair",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (root / selection_rel).write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in (
                    {
                        "source_trajectory_id": "scene_a/scene.glb:0",
                        "episode_id": "0",
                        "scene_id": "scene_a/scene.glb",
                        "object_category": "chair",
                        "shard_path": "content/scene_a.json.gz",
                    },
                    {
                        "source_trajectory_id": "scene_b/scene.glb:1",
                        "episode_id": "1",
                        "scene_id": "scene_b/scene.glb",
                        "object_category": "table",
                        "shard_path": "content/scene_b.json.gz",
                    },
                )
            ),
            encoding="utf-8",
        )
        return TopoVLMConfig(
            data=DataConfig(
                data_root=str(root),
                episodes_manifest=manifest_rel,
                episode_selection_manifest=selection_rel,
            )
        )

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

    def test_pr2l_cache_builder_respects_episode_selection_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episodes/pr2l_hm3d_objectnav/train").mkdir(parents=True)
            (root / "episode_selections/pr2l_hm3d_objectnav").mkdir(parents=True)
            (root / "rgb").mkdir()
            (root / "actions").mkdir()
            np.save(root / "rgb/keep.npy", np.zeros((2, 2, 2, 3), dtype="uint8"))
            np.save(root / "actions/keep.npy", np.asarray([1, 0], dtype="int64"))
            np.save(root / "rgb/skip.npy", np.zeros((2, 2, 2, 3), dtype="uint8"))
            np.save(root / "actions/skip.npy", np.asarray([2, 0], dtype="int64"))
            (root / "episodes/pr2l_hm3d_objectnav/train/manifest.jsonl").write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in (
                        {
                            "episode_id": "episode_skip",
                            "split": "train",
                            "scene_id": "scene_skip",
                            "goal_text": "sofa",
                            "rgb_path": "rgb/skip.npy",
                            "actions_path": "actions/skip.npy",
                            "source_dataset": "hm3d_objectnav_shortest_path",
                            "source_trajectory_id": "scene_skip:0",
                            "object_category": "sofa",
                        },
                        {
                            "episode_id": "episode_keep",
                            "split": "train",
                            "scene_id": "scene_keep",
                            "goal_text": "chair",
                            "rgb_path": "rgb/keep.npy",
                            "actions_path": "actions/keep.npy",
                            "source_dataset": "hm3d_objectnav_shortest_path",
                            "source_trajectory_id": "scene_keep:0",
                            "object_category": "chair",
                        },
                    )
                ),
                encoding="utf-8",
            )
            (root / "episode_selections/pr2l_hm3d_objectnav/train_subset.jsonl").write_text(
                json.dumps({"source_trajectory_id": "scene_keep:0"}) + "\n",
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    cache_format="pr2l_token_trajectory",
                    episodes_manifest="episodes/pr2l_hm3d_objectnav/train/manifest.jsonl",
                    episode_selection_manifest="episode_selections/pr2l_hm3d_objectnav/train_subset.jsonl",
                    graph_manifest="graphs/pr2l/manifest.jsonl",
                    graph_cache_dir="graphs/pr2l",
                    embeddings_dir="embeddings/pr2l",
                ),
                model=ModelConfig(
                    vlm=VLMConfig(
                        representation="pr2l_visual_tokens_last_two_layers",
                        projection="none",
                        output_dim=8,
                    ),
                    policy=PolicyConfig(input_dim=8, prediction_target="nodes"),
                ),
            )

            with patch("data.habitat_cache.build_vlm_encoder", return_value=_FakePR2LEncoder()):
                result = build_habitat_graph_cache(cfg)

            graph_records = [
                json.loads(line)
                for line in (root / "graphs/pr2l/manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(result["graphs_written"], 1)
            self.assertEqual([record["episode_id"] for record in graph_records], ["episode_keep"])

    def test_pr2l_cache_builder_allows_missing_selected_episode_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = self._build_pr2l_selection_gap_config(root)
            cfg.data.cache_format = "pr2l_token_trajectory"
            cfg.data.allow_missing_selected_episode_records = True
            cfg.data.graph_manifest = "graphs/pr2l/manifest.jsonl"
            cfg.data.graph_cache_dir = "graphs/pr2l"
            cfg.data.embeddings_dir = "embeddings/pr2l"
            cfg.model = ModelConfig(
                vlm=VLMConfig(
                    representation="pr2l_visual_tokens_last_two_layers",
                    projection="none",
                    output_dim=8,
                    weights_path=str(root / "fake_prismatic"),
                ),
                policy=PolicyConfig(input_dim=8, prediction_target="nodes"),
            )

            with patch("data.habitat_cache.build_vlm_encoder", return_value=_FakePR2LEncoder()):
                result = build_habitat_graph_cache(cfg)

            graph_records = [
                json.loads(line)
                for line in (root / "graphs/pr2l/manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(result["graphs_written"], 1)
            self.assertEqual([record["episode_id"] for record in graph_records], ["episode_0"])

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

    def test_pr2l_missing_non_train_projection_fails_before_fit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = TopoVLMConfig(
                data=DataConfig(data_root=str(root), split="val"),
                model=ModelConfig(
                    vlm=VLMConfig(
                        projection="pca",
                        projection_path="embeddings/pr2l_hm3d_bc/projection_pca.npz",
                        projection_dim=4,
                    )
                ),
            )

            with patch("data.habitat_cache._fit_projection") as fit_projection:
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "split=val.*embeddings/pr2l_hm3d_bc/projection_pca.npz",
                ):
                    _load_or_fit_projection(cfg, None, [], root, root)

            fit_projection.assert_not_called()

    def test_pr2l_missing_train_projection_still_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "output"
            root.mkdir()
            output_root.mkdir()
            cfg = TopoVLMConfig(
                data=DataConfig(data_root=str(root), split="train"),
                model=ModelConfig(
                    vlm=VLMConfig(
                        projection="pca",
                        projection_path="embeddings/pr2l_hm3d_bc/projection_pca.npz",
                        projection_dim=4,
                    )
                ),
            )
            projection = {"mean": np.zeros(8), "components": np.zeros((4, 8))}

            with patch(
                "data.habitat_cache._fit_projection", return_value=projection
            ) as fit_projection:
                result = _load_or_fit_projection(cfg, None, [], root, output_root)

            self.assertIs(result, projection)
            fit_projection.assert_called_once()
            self.assertEqual(
                fit_projection.call_args.args[4],
                output_root / "embeddings/pr2l_hm3d_bc/projection_pca.npz",
            )

    def test_pr2l_pca_projection_uses_numpy_only(self):
        sample_matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype="float32",
        )

        mean, components, explained_variance = _fit_pca_projection(sample_matrix, 2, np)

        self.assertEqual(mean.dtype, np.float32)
        self.assertEqual(components.dtype, np.float32)
        self.assertEqual(explained_variance.dtype, np.float32)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(components.shape, (2, 3))
        self.assertEqual(explained_variance.shape, (2,))
        self.assertTrue(np.all(explained_variance >= 0))
        self.assertTrue(np.allclose(components @ components.T, np.eye(2), atol=1e-5))

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

    def test_pr2l_manifest_audit_fails_when_selected_source_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._build_pr2l_selection_gap_config(Path(tmpdir))

            with self.assertRaisesRegex(
                FileNotFoundError, "Missing selected source episodes in PR2L manifest"
            ):
                run_pr2l_manifest_audit(cfg)

    def test_pr2l_manifest_audit_allows_missing_selected_source_in_allow_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._build_pr2l_selection_gap_config(Path(tmpdir))

            result = run_pr2l_manifest_audit(cfg, allow_missing_data=True)

            self.assertEqual(result["status"], "missing_allowed")
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["selected_source_episodes"], 2)
            self.assertEqual(result["missing_selected_source_count"], 1)
            self.assertEqual(result["missing_selected_source_ids"], ["scene_b/scene.glb:1"])
            self.assertEqual(result["missing_payload_count"], 0)

    def test_pr2l_manifest_audit_still_fails_on_missing_payload_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episodes/pr2l_hm3d_objectnav/train").mkdir(parents=True)
            (root / "actions").mkdir()
            np.save(root / "actions/episode_0.npy", np.asarray([0], dtype="int64"))
            (root / "episodes/pr2l_hm3d_objectnav/train/manifest.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": "episode_0",
                        "split": "train",
                        "scene_id": "scene",
                        "goal_text": "chair",
                        "rgb_path": "rgb/missing_episode_0.npy",
                        "actions_path": "actions/episode_0.npy",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    episodes_manifest="episodes/pr2l_hm3d_objectnav/train/manifest.jsonl",
                )
            )

            with self.assertRaisesRegex(
                FileNotFoundError, "Missing PR2L trajectory payloads"
            ):
                run_pr2l_manifest_audit(cfg)

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
