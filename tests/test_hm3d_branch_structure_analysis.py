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


class HM3DBranchStructureAnalysisTest(unittest.TestCase):
    def test_validate_runner_writes_analysis_result_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_episode(
                root,
                "scene_a/scene.glb:0",
                "scene_a/scene.glb",
                "chair",
                np.asarray([1, 1, 2, 1, 0], dtype=np.int64),
            )
            _write_episode(
                root,
                "scene_a/scene.glb:1",
                "scene_a/scene.glb",
                "chair",
                np.asarray([1, 3, 1, 0], dtype=np.int64),
            )
            exp_path = root / "hm3d_branch_exp.yaml"
            _write_exp_config(exp_path, root)
            artifact_dir = root / "artifacts" / "analysis" / "hm3d_branch_structure"

            stdout = io.StringIO()
            with patch.dict(
                "os.environ", {"ARTIFACT_DIR": str(artifact_dir)}, clear=False
            ), redirect_stdout(stdout):
                validate.main(
                    [
                        "--runner",
                        "hm3d_branch_structure",
                        "--exp",
                        str(exp_path),
                    ]
                )
            result = json.loads(stdout.getvalue())

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["analysis_name"], "hm3d_branch_structure")
            self.assertEqual(result["metrics"]["episodes"], 2)
            self.assertEqual(result["metrics"]["steps"], 9)
            self.assertEqual(result["metrics"]["action_counts"]["MOVE_FORWARD"], 5)
            self.assertEqual(result["metrics"]["action_counts"]["TURN_LEFT"], 1)
            self.assertEqual(result["metrics"]["action_counts"]["TURN_RIGHT"], 1)
            self.assertEqual(result["metrics"]["turn_structure"]["turn_run_starts"], 2)
            self.assertEqual(
                result["metrics"]["length_bin_turn_pressure"][0]["length_bin"], "2-32"
            )
            self.assertEqual(
                result["metrics"]["scene_object_trajectory_diversity"][
                    "groups_with_first_turn_diversity"
                ],
                1,
            )
            self.assertEqual(
                [axis["axis"] for axis in result["evidence_axes"]],
                [
                    "turn-run pressure",
                    "first-turn split",
                    "same scene-object trajectory diversity",
                    "object-category branch diversity",
                    "latent or policy representation separability",
                ],
            )
            self.assertEqual(result["evidence_axes"][0]["evidence_status"], "supports")
            self.assertEqual(result["evidence_axes"][-1]["evidence_status"], "insufficient")
            result_manifest = artifact_dir / "result_manifest.json"
            self.assertEqual(result["result_manifest"], str(result_manifest))
            figure_path = artifact_dir / "branch_structure_summary.png"
            self.assertEqual(
                result["figures"],
                [{"path": str(figure_path), "role": "branch_structure_summary"}],
            )
            self.assertTrue(figure_path.exists())
            manifest = json.loads(result_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_type"], "topovlm_analysis_result_manifest")
            self.assertEqual(manifest["analysis_name"], "hm3d_branch_structure")
            self.assertEqual(manifest["durable_lane_roots"]["artifact_dir"], str(artifact_dir))
            self.assertEqual(manifest["evidence_axes"], result["evidence_axes"])
            self.assertEqual(manifest["figures"], result["figures"])
            self.assertIn("not_claimed", manifest["operational_definition"])


def _write_episode(
    root: Path,
    source_trajectory_id: str,
    scene_id: str,
    object_category: str,
    actions: np.ndarray,
) -> None:
    manifest_path = root / "episodes" / "pr2l_hm3d_objectnav" / "train" / "manifest.jsonl"
    actions_dir = root / "actions" / "pr2l_hm3d_objectnav" / "train"
    rgb_dir = root / "rgb" / "pr2l_hm3d_objectnav" / "train"
    actions_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record_index = (
        sum(1 for _ in manifest_path.open("r", encoding="utf-8"))
        if manifest_path.exists()
        else 0
    )
    actions_path = actions_dir / f"episode_{record_index}.npy"
    rgb_path = rgb_dir / f"episode_{record_index}.npy"
    np.save(actions_path, actions)
    np.save(rgb_path, np.zeros((len(actions), 1, 1, 3), dtype=np.uint8))
    record = {
        "episode_id": str(record_index),
        "split": "train",
        "scene_id": scene_id,
        "goal_text": object_category,
        "rgb_path": str(rgb_path.relative_to(root)),
        "actions_path": str(actions_path.relative_to(root)),
        "source_dataset": "hm3d_objectnav_shortest_path",
        "source_trajectory_id": source_trajectory_id,
        "object_category": object_category,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_exp_config(path: Path, data_root: Path) -> None:
    payload = {
        "defaults": [
            {"train": "pr2l_hm3d_bc"},
            {"data": "pr2l_hm3d_objectnav"},
            {"model": "pr2l_prismatic_hm3d"},
            {"objectives": "pr2l_bc_hm3d"},
            {"eval": "default"},
        ],
        "data": {
            "data_root": str(data_root),
            "episodes_manifest": "episodes/pr2l_hm3d_objectnav/train/manifest.jsonl",
            "episode_selection_manifest": None,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
