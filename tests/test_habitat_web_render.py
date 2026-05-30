import gzip
import json
import tempfile
import unittest
from pathlib import Path

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
