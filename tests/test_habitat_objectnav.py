import gzip
import json
import tempfile
import unittest
from pathlib import Path

from configs.schema import DataConfig
from data.habitat_objectnav import HabitatObjectNavDataset


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


if __name__ == "__main__":
    unittest.main()
