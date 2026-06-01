import gzip
import json
import tempfile
import unittest
from pathlib import Path

from configs.schema import DataConfig
from data.habitat_objectnav import (
    HabitatObjectNavDataset,
    build_objectnav_balanced_selection_manifest,
    objectnav_source_trajectory_id,
)


class HabitatObjectNavDatasetTest(unittest.TestCase):
    def test_loads_sharded_episode_and_resolves_scene_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/train"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            with gzip.open(split_dir / "train.json.gz", "wt", encoding="utf-8") as handle:
                json.dump({"episodes": []}, handle)
            shard = content_dir / "scene.json.gz"
            payload = {
                "episodes": [
                    {
                        "episode_id": "0",
                        "scene_id": "hm3d_v0.2/train/00001-scene/scene.basis.glb",
                        "object_category": "chair",
                    }
                ]
            }
            with gzip.open(shard, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            scene_path = root / "scene_datasets/hm3d/hm3d_v0.2/train/00001-scene"
            scene_path.mkdir(parents=True)
            (scene_path / "scene.basis.glb").write_text("glb", encoding="utf-8")

            config = DataConfig(data_root=str(root))
            dataset = HabitatObjectNavDataset(config)
            episode = dataset.first_episode()

            self.assertEqual(episode.object_category, "chair")
            self.assertTrue(dataset.resolve_scene_path(episode).exists())

    def test_source_trajectory_id_normalizes_habitat_absolute_scene_path(self):
        episode = type(
            "Episode",
            (),
            {
                "scene_id": (
                    "/data/topovlm/habitat/scene_datasets/hm3d_v0.2/"
                    "train/00001-scene/scene.basis.glb"
                ),
                "episode_id": "7",
                "object_category": "chair",
            },
        )()

        self.assertEqual(
            objectnav_source_trajectory_id(episode),
            "hm3d_v0.2/train/00001-scene/scene.basis.glb:7:chair",
        )

    def test_resolves_scene_path_with_data_scene_prefix(self):
        config = DataConfig(data_root="/data/topovlm/habitat")
        dataset_scene_id = "data/scene_datasets/hm3d/train/scene/scene.basis.glb"

        dataset = HabitatObjectNavDataset.__new__(HabitatObjectNavDataset)
        dataset.config = config

        self.assertEqual(
            str(dataset.resolve_scene_path(type("Episode", (), {"scene_id": dataset_scene_id})())),
            "/data/topovlm/habitat/scene_datasets/hm3d/train/scene/scene.basis.glb",
        )

    def test_builds_scene_object_balanced_selection_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/train"
            content_dir = split_dir / "content"
            content_dir.mkdir(parents=True)
            with gzip.open(split_dir / "train.json.gz", "wt", encoding="utf-8") as handle:
                json.dump({"episodes": []}, handle)
            payload = {
                "episodes": [
                    _episode("a0", "scene_a/scene.basis.glb", "chair"),
                    _episode("a1", "scene_a/scene.basis.glb", "chair"),
                    _episode("a2", "scene_a/scene.basis.glb", "table"),
                    _episode("b0", "scene_b/scene.basis.glb", "chair"),
                ]
            }
            with gzip.open(content_dir / "episodes.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)

            config = DataConfig(
                data_root=str(root),
                episode_selection_manifest=(
                    "episode_selections/pr2l_hm3d_objectnav/train_scene_object_balanced.jsonl"
                ),
                balanced_subset_size=3,
            )
            result = build_objectnav_balanced_selection_manifest(config)
            manifest = root / config.episode_selection_manifest
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(result["selected_episodes"], 3)
            self.assertEqual(
                [record["source_trajectory_id"] for record in records],
                [
                    "scene_a/scene.basis.glb:a0:chair",
                    "scene_a/scene.basis.glb:a2:table",
                    "scene_b/scene.basis.glb:b0:chair",
                ],
            )


def _episode(episode_id: str, scene_id: str, object_category: str) -> dict[str, str]:
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "object_category": object_category,
    }


if __name__ == "__main__":
    unittest.main()
