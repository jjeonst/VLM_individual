import gzip
import json
import tempfile
import unittest
from pathlib import Path

from configs.schema import DataConfig
from data.habitat_web import HABITAT_WEB_ACTION_TO_ID, is_git_lfs_pointer
from data.habitat_web import load_habitat_web_summary


class HabitatWebReplayTest(unittest.TestCase):
    def test_summary_reads_reference_replay_and_flags_missing_scene(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "sources/habitat_web/train_sample"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            _write_json_gz(
                split_dir / "train_sample.json.gz",
                {"episodes": [], "category_to_task_category_id": {}},
            )
            _write_json_gz(
                content_dir / "scene.json.gz",
                {
                    "episodes": [
                        {
                            "episode_id": "episode_0",
                            "scene_id": "mp3d/scene/scene.glb",
                            "object_category": "chair",
                            "reference_replay": [
                                {
                                    "action": "STOP",
                                    "agent_state": {
                                        "position": [0.0, 0.0, 0.0],
                                        "rotation": [0.0, 0.0, 0.0, 1.0],
                                        "sensor_data": {
                                            "rgb": {
                                                "position": [0.0, 0.88, 0.0],
                                                "rotation": [0.0, 0.0, 0.0, 1.0],
                                            }
                                        },
                                    },
                                },
                                {
                                    "action": "MOVE_FORWARD",
                                    "agent_state": {
                                        "position": [0.0, 0.0, 1.0],
                                        "rotation": [0.0, 0.0, 0.0, 1.0],
                                        "sensor_data": None,
                                    },
                                },
                                {
                                    "action": "LOOK_DOWN",
                                    "agent_state": {
                                        "position": [0.0, 0.0, 1.0],
                                        "rotation": [0.0, 0.0, 0.0, 1.0],
                                        "sensor_data": None,
                                    },
                                },
                            ],
                        }
                    ]
                },
            )
            cfg = DataConfig(
                data_root=str(root),
                objectnav_dataset_dir="sources/habitat_web",
                scene_dataset_dir="scene_datasets",
                split="train_sample",
            )

            summary = load_habitat_web_summary(cfg, sample_episodes=1)

            self.assertTrue(summary["split_index_materialized"])
            self.assertEqual(summary["materialized_content_shards"], 1)
            self.assertEqual(summary["sample_action_vocab"], ["LOOK_DOWN", "MOVE_FORWARD", "STOP"])
            self.assertEqual(summary["sample_replay_length_min"], 3)
            self.assertTrue(summary["requires_scene_rendering"])
            self.assertEqual(len(summary["missing_sample_scenes"]), 1)
            self.assertEqual(HABITAT_WEB_ACTION_TO_ID["LOOK_DOWN"], 5)

    def test_lfs_pointer_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pointer = Path(tmpdir) / "shard.json.gz"
            pointer.write_text(
                "\n".join(
                    [
                        "version https://git-lfs.github.com/spec/v1",
                        "oid sha256:abc",
                        "size 123",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(is_git_lfs_pointer(pointer))


def _write_json_gz(path: Path, payload: dict[str, object]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    unittest.main()
