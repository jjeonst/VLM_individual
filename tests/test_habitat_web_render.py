import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from configs.schema import DataConfig, TopoVLMConfig
from data.habitat_web_render import build_habitat_web_episode_manifest


class HabitatWebRenderTest(unittest.TestCase):
    def test_build_episode_manifest_with_fake_renderer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "sources/habitat_web/train_sample"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            scene_dir = root / "scene_datasets/mp3d/scene"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene.glb").write_text("fake", encoding="utf-8")
            _write_json_gz(split_dir / "train_sample.json.gz", {"episodes": []})
            _write_json_gz(
                content_dir / "scene.json.gz",
                {
                    "episodes": [
                        {
                            "episode_id": "episode:0",
                            "scene_id": "mp3d/scene/scene.glb",
                            "object_category": "chair",
                            "reference_replay": [
                                _step("STOP", [0.0, 0.0, 0.0]),
                                _step("MOVE_FORWARD", [0.0, 0.0, 1.0]),
                                _step("STOP", [0.0, 0.0, 1.0]),
                            ],
                        }
                    ]
                },
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    dataset_name="pr2l_habitat_web",
                    objectnav_dataset_dir="sources/habitat_web",
                    scene_dataset_dir="scene_datasets",
                    split="train_sample",
                    episodes_manifest="episodes/pr2l_habitat_web/train_sample/manifest.jsonl",
                    max_episodes=1,
                )
            )

            result = build_habitat_web_episode_manifest(cfg, renderer=_FakeRenderer())

            manifest_path = Path(result["manifest"])
            record = json.loads(manifest_path.read_text(encoding="utf-8").strip())
            frames = np.load(root / record["rgb_path"])
            actions = np.load(root / record["actions_path"])
            self.assertEqual(result["episodes_written"], 1)
            self.assertEqual(result["dropped_leading_stop_count"], 1)
            self.assertEqual(result["source_data_root"], str(root))
            self.assertEqual(result["output_data_root"], str(root))
            self.assertEqual(record["episode_id"], "episode_0")
            self.assertEqual(record["source_trajectory_id"], "episode:0")
            self.assertEqual(record["goal_text"], "chair")
            self.assertEqual(tuple(frames.shape), (2, 2, 2, 3))
            self.assertEqual(actions.tolist(), [1, 0])

    def test_missing_scene_does_not_leave_tmp_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "sources/habitat_web/train_sample"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            _write_json_gz(split_dir / "train_sample.json.gz", {"episodes": []})
            _write_json_gz(
                content_dir / "scene.json.gz",
                {
                    "episodes": [
                        {
                            "episode_id": "episode:0",
                            "scene_id": "mp3d/missing/missing.glb",
                            "object_category": "chair",
                            "reference_replay": [
                                _step("MOVE_FORWARD", [0.0, 0.0, 1.0]),
                            ],
                        }
                    ]
                },
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    dataset_name="pr2l_habitat_web",
                    objectnav_dataset_dir="sources/habitat_web",
                    scene_dataset_dir="scene_datasets",
                    split="train_sample",
                    episodes_manifest="episodes/pr2l_habitat_web/train_sample/manifest.jsonl",
                    max_episodes=1,
                )
            )

            with self.assertRaises(FileNotFoundError):
                build_habitat_web_episode_manifest(cfg, renderer=_FakeRenderer())

            manifest = root / "episodes/pr2l_habitat_web/train_sample/manifest.jsonl"
            self.assertFalse(manifest.exists())
            self.assertFalse(manifest.with_suffix(".jsonl.tmp").exists())

    def test_build_episode_manifest_uses_selection_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "sources/habitat_web/train_sample"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            scene_dir = root / "scene_datasets/mp3d/scene"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene.glb").write_text("fake", encoding="utf-8")
            selection_dir = root / "episode_selections"
            selection_dir.mkdir()
            (selection_dir / "subset.jsonl").write_text(
                json.dumps(
                    {
                        "source_trajectory_id": "episode:1",
                        "scene_id": "mp3d/scene/scene.glb",
                        "object_category": "table",
                        "shard_path": "source",
                        "replay_length": 1,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json_gz(split_dir / "train_sample.json.gz", {"episodes": []})
            _write_json_gz(
                content_dir / "scene.json.gz",
                {
                    "episodes": [
                        {
                            "episode_id": "episode:0",
                            "scene_id": "mp3d/scene/scene.glb",
                            "object_category": "chair",
                            "reference_replay": [_step("MOVE_FORWARD", [0.0, 0.0, 1.0])],
                        },
                        {
                            "episode_id": "episode:1",
                            "scene_id": "mp3d/scene/scene.glb",
                            "object_category": "table",
                            "reference_replay": [_step("STOP", [0.0, 0.0, 0.0])],
                        },
                    ]
                },
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    dataset_name="pr2l_habitat_web_subset",
                    objectnav_dataset_dir="sources/habitat_web",
                    scene_dataset_dir="scene_datasets",
                    split="train_sample",
                    episodes_manifest="episodes/pr2l_habitat_web_subset/train_sample/manifest.jsonl",
                    episode_selection_manifest="episode_selections/subset.jsonl",
                )
            )

            result = build_habitat_web_episode_manifest(cfg, renderer=_FakeRenderer())

            manifest_path = Path(result["manifest"])
            records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(result["episodes_written"], 1)
            self.assertEqual(result["selected_source_episodes"], 1)
            self.assertEqual(records[0]["source_trajectory_id"], "episode:1")
            self.assertEqual(records[0]["goal_text"], "table")

    def test_build_episode_manifest_can_write_to_output_data_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            output_root = Path(tmpdir) / "outputs/data/topovlm/habitat"
            split_dir = root / "sources/habitat_web/train_sample"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            scene_dir = root / "scene_datasets/mp3d/scene"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene.glb").write_text("fake", encoding="utf-8")
            _write_json_gz(split_dir / "train_sample.json.gz", {"episodes": []})
            _write_json_gz(
                content_dir / "scene.json.gz",
                {
                    "episodes": [
                        {
                            "episode_id": "episode:0",
                            "scene_id": "mp3d/scene/scene.glb",
                            "object_category": "chair",
                            "reference_replay": [_step("MOVE_FORWARD", [0.0, 0.0, 1.0])],
                        }
                    ]
                },
            )
            cfg = TopoVLMConfig(
                data=DataConfig(
                    data_root=str(root),
                    dataset_name="pr2l_habitat_web",
                    objectnav_dataset_dir="sources/habitat_web",
                    scene_dataset_dir="scene_datasets",
                    split="train_sample",
                    episodes_manifest="episodes/pr2l_habitat_web/train_sample/manifest.jsonl",
                    max_episodes=1,
                )
            )

            with patch.dict(os.environ, {"TOPOVLM_DATA_OUTPUT_ROOT": str(output_root)}):
                result = build_habitat_web_episode_manifest(cfg, renderer=_FakeRenderer())

            manifest_path = output_root / "episodes/pr2l_habitat_web/train_sample/manifest.jsonl"
            record = json.loads(manifest_path.read_text(encoding="utf-8").strip())
            self.assertEqual(result["source_data_root"], str(root))
            self.assertEqual(result["output_data_root"], str(output_root))
            self.assertTrue((output_root / record["rgb_path"]).exists())
            self.assertFalse((root / "episodes/pr2l_habitat_web/train_sample/manifest.jsonl").exists())


class _FakeRenderer:
    def render_episode(self, scene_path, replay):
        return np.zeros((len(replay), 2, 2, 3), dtype="uint8")


def _step(action: str, position: list[float]) -> dict[str, object]:
    return {
        "action": action,
        "agent_state": {
            "position": position,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "sensor_data": None,
        },
    }


def _write_json_gz(path: Path, payload: dict[str, object]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    unittest.main()
