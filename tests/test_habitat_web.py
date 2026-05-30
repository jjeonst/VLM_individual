import gzip
import json
import tempfile
import unittest
from pathlib import Path

from configs.schema import DataConfig
from data.habitat_web import HABITAT_WEB_ACTION_TO_ID, is_git_lfs_pointer
from data.habitat_web import load_habitat_web_inventory, load_habitat_web_summary


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

    def test_inventory_counts_scenes_objects_and_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "sources/habitat_web/train"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            existing_scene = root / "scene_datasets/mp3d/existing/existing.glb"
            existing_scene.parent.mkdir(parents=True)
            existing_scene.write_text("placeholder", encoding="utf-8")
            _write_json_gz(
                split_dir / "train.json.gz",
                {"episodes": [], "category_to_task_category_id": {}},
            )
            _write_json_gz(
                content_dir / "batch.json.gz",
                {
                    "episodes": [
                        {
                            "episode_id": "episode_0",
                            "scene_id": "mp3d/existing/existing.glb",
                            "object_category": "chair",
                            "reference_replay": [
                                {"action": "MOVE_FORWARD", "agent_state": {}},
                                {"action": "STOP", "agent_state": {}},
                            ],
                        },
                        {
                            "episode_id": "episode_1",
                            "scene_id": "mp3d/missing/missing.glb",
                            "object_category": "table",
                            "reference_replay": [
                                {"action": "TURN_LEFT", "agent_state": {}},
                                {"action": "LOOK_UP", "agent_state": {}},
                                {"action": "STOP", "agent_state": {}},
                            ],
                        },
                    ]
                },
            )
            cfg = DataConfig(
                data_root=str(root),
                objectnav_dataset_dir="sources/habitat_web",
                scene_dataset_dir="scene_datasets",
                split="train",
            )

            inventory = load_habitat_web_inventory(cfg)

            self.assertEqual(inventory["episodes"], 2)
            self.assertEqual(inventory["unique_scenes"], 2)
            self.assertEqual(inventory["unique_object_categories"], 2)
            self.assertEqual(inventory["existing_scene_count"], 1)
            self.assertEqual(inventory["missing_scene_count"], 1)
            self.assertEqual(inventory["missing_scene_ids"], ["mp3d/missing/missing.glb"])
            self.assertEqual(inventory["replay_length_min"], 2)
            self.assertEqual(inventory["replay_length_max"], 3)
            self.assertEqual(inventory["action_counts"][0], {"action": "LOOK_UP", "count": 1})
            self.assertEqual(
                inventory["object_counts_top"],
                [
                    {"object_category": "chair", "count": 1},
                    {"object_category": "table", "count": 1},
                ],
            )


def _write_json_gz(path: Path, payload: dict[str, object]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    unittest.main()
