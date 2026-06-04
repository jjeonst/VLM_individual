import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

import validate
from configs import build_config_from_exp
from policies import build_policy


class OfflineEvalTest(unittest.TestCase):
    def test_validate_offline_policy_eval_emits_majority_baseline(self):
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "graphs" / "hm3d_val" / "manifest.jsonl"
            checkpoint_dir = root / "checkpoint"
            exp_path = root / "hm3d_val_exp.yaml"
            graph_dir = root / "graphs" / "hm3d_val"
            graph_dir.mkdir(parents=True)
            checkpoint_dir.mkdir()
            _write_graph_record(
                manifest_path,
                graph_dir / "episode_0.npz",
                "episode_0",
                target_action=1,
                node_actions=np.asarray([1, 1, 2], dtype=np.int64),
                action_mask=np.asarray([True, True, False], dtype=bool),
            )
            _write_graph_record(
                manifest_path,
                graph_dir / "episode_1.npz",
                "episode_1",
                target_action=2,
                node_actions=np.asarray([2, 1, 0], dtype=np.int64),
                action_mask=np.asarray([True, False, True], dtype=bool),
            )
            _write_exp_config(exp_path, root)
            cfg = build_config_from_exp(str(exp_path))
            policy = build_policy(cfg.model.policy)
            torch.save({"model": policy.state_dict()}, checkpoint_dir / "model.pt")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                validate.main(
                    [
                        "--runner",
                        "offline_policy_eval",
                        "--exp",
                        str(exp_path),
                        "--checkpoint-dir",
                        str(checkpoint_dir),
                    ]
                )
            result = json.loads(stdout.getvalue())

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["examples"], 4)
            self.assertIn("action_accuracy", result)
            self.assertEqual(
                result["majority_class_baseline"],
                {
                    "name": "majority_class",
                    "action": 1,
                    "examples": 4,
                    "correct": 2,
                    "action_accuracy": 0.5,
                    "label_counts": {"0": 1, "1": 2, "2": 1},
                },
            )

    def test_validate_offline_policy_eval_uses_explicit_output_data_root(self):
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "stageout_data"
            configured_root = root / "configured_data_root_without_cache"
            manifest_path = output_root / "graphs" / "hm3d_val" / "manifest.jsonl"
            checkpoint_dir = root / "checkpoint"
            exp_path = root / "hm3d_val_exp.yaml"
            graph_dir = output_root / "graphs" / "hm3d_val"
            graph_dir.mkdir(parents=True)
            checkpoint_dir.mkdir()
            _write_graph_record(
                manifest_path,
                graph_dir / "episode_0.npz",
                "episode_0",
                target_action=1,
                node_actions=np.asarray([1, 1, 2], dtype=np.int64),
                action_mask=np.asarray([True, True, False], dtype=bool),
            )
            _write_exp_config(exp_path, configured_root)
            cfg = build_config_from_exp(str(exp_path))
            policy = build_policy(cfg.model.policy)
            torch.save({"model": policy.state_dict()}, checkpoint_dir / "model.pt")

            stdout = io.StringIO()
            with patch.dict(
                "os.environ",
                {"TOPOVLM_DATA_OUTPUT_ROOT": str(output_root)},
                clear=False,
            ), redirect_stdout(stdout):
                validate.main(
                    [
                        "--runner",
                        "offline_policy_eval",
                        "--exp",
                        str(exp_path),
                        "--checkpoint-dir",
                        str(checkpoint_dir),
                    ]
                )
            result = json.loads(stdout.getvalue())

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["examples"], 2)


def _write_graph_record(
    manifest_path: Path,
    graph_path: Path,
    episode_id: str,
    *,
    target_action: int,
    node_actions: np.ndarray,
    action_mask: np.ndarray,
) -> None:
    nodes = np.zeros((3, 2), dtype=np.float32)
    np.savez(
        graph_path,
        nodes=nodes,
        target_action=np.asarray(target_action, dtype=np.int64),
        node_actions=node_actions,
        action_mask=action_mask,
    )
    record = {
        "episode_id": episode_id,
        "split": "val",
        "scene_id": "scene.glb",
        "goal_text": "chair",
        "graph_path": str(graph_path.relative_to(manifest_path.parents[2])),
        "embedding_path": "embeddings/hm3d_val/unused.npz",
        "target_action": target_action,
        "num_nodes": 3,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_exp_config(path: Path, data_root: Path) -> None:
    raw = {
        "defaults": [
            {"train": "default"},
            {"data": "default"},
            {"model": "default"},
            {"objectives": "default"},
            {"eval": "default"},
        ],
        "config_name": "test_hm3d_val_offline_eval",
        "device": "cpu",
        "data": {
            "data_root": str(data_root),
            "split": "val",
            "graph_manifest": "graphs/hm3d_val/manifest.jsonl",
            "batch_size": 2,
            "num_workers": 0,
            "max_episodes": None,
        },
        "model": {
            "policy": {
                "input_dim": 2,
                "hidden_dim": 8,
                "transformer_heads": 2,
                "transformer_layers": 1,
                "dropout": 0.0,
                "num_actions": 4,
                "prediction_target": "nodes",
            }
        },
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
