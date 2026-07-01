import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.code.hm3d_trajectory_notebook import select_scene_trajectory_records


class HM3DTrajectoryNotebookTest(unittest.TestCase):
    def test_none_max_trajectories_selects_all_records_from_ranked_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_episode(root, "0", "scene_a/scene.glb", "bed", [1, 2, 0])
            _write_episode(root, "1", "scene_a/scene.glb", "bed", [1, 3, 0])
            _write_episode(root, "2", "scene_a/scene.glb", "bed", [1, 1, 0])
            _write_episode(root, "3", "scene_b/scene.glb", "chair", [1, 2, 0])

            records = select_scene_trajectory_records(
                root,
                max_trajectories=None,
                min_steps=1,
                min_turns=0,
                require_graph_cache=False,
            )

            self.assertEqual([record["episode_id"] for record in records], ["0", "1", "2"])
            self.assertTrue(all(record["scene_id"] == "scene_a/scene.glb" for record in records))


def _write_episode(
    root: Path,
    episode_id: str,
    scene_id: str,
    object_category: str,
    actions: list[int],
) -> None:
    manifest_path = root / "episodes" / "pr2l_hm3d_objectnav" / "train" / "manifest.jsonl"
    actions_dir = root / "actions" / "pr2l_hm3d_objectnav" / "train"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    actions_path = actions_dir / f"episode_{episode_id}.npy"
    np.save(actions_path, np.asarray(actions, dtype=np.int64))
    record = {
        "episode_id": episode_id,
        "split": "train",
        "scene_id": scene_id,
        "goal_text": object_category,
        "actions_path": str(actions_path.relative_to(root)),
        "source_dataset": "hm3d_objectnav_shortest_path",
        "source_trajectory_id": f"{scene_id}:{episode_id}",
        "object_category": object_category,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
