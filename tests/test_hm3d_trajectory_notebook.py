import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.code import export_hm3d_scene_topdown_pngs as scene_export
from analysis.code.hm3d_trajectory_notebook import (
    _episode_selection_key,
    _selection_record_key,
    marker_legend_handles,
    select_scene_trajectory_records,
    trajectory_record_key,
)


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

    def test_user_facing_notebooks_keep_inline_image_outputs_only(self):
        repo_root = Path(__file__).resolve().parents[1]
        notebook_paths = [
            repo_root / "analysis/code/hm3d_01_environment_topdown_trajectories.ipynb",
            repo_root / "analysis/code/hm3d_02_observation_latent_trajectories.ipynb",
            repo_root / "analysis/code/hm3d_03_vlm_cached_latent_trajectories.ipynb",
        ]
        minimum_image_outputs = {
            "hm3d_01_environment_topdown_trajectories.ipynb": 2,
            "hm3d_02_observation_latent_trajectories.ipynb": 1,
            "hm3d_03_vlm_cached_latent_trajectories.ipynb": 1,
        }

        for notebook_path in notebook_paths:
            with self.subTest(notebook=notebook_path.name):
                notebook = json.loads(notebook_path.read_text())
                code_cells = [
                    cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
                ]
                source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)

                self.assertIn("plt.show()", source)
                if notebook_path.name == "hm3d_01_environment_topdown_trajectories.ipynb":
                    self.assertIn("replay_selected_objectnav_scene_topdowns", source)
                    self.assertIn("for replay in scene_replays", source)
                    self.assertEqual(source.count("plt.show()"), 1)
                    self.assertNotIn("plot_dataset_action_overview", source)
                else:
                    self.assertIn("max_trajectories=None", source)
                self.assertNotIn("save_figure", source)
                self.assertNotIn("DEFAULT_RESULT_DIR", source)
                self.assertNotIn("fig.savefig", source)

                outputs = [
                    output
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                ]
                image_outputs = [
                    output
                    for output in outputs
                    if "image/png" in output.get("data", {})
                ]
                stream_outputs = [
                    output for output in outputs if output.get("output_type") == "stream"
                ]
                error_outputs = [
                    output for output in outputs if output.get("output_type") == "error"
                ]

                self.assertGreaterEqual(
                    len(image_outputs), minimum_image_outputs[notebook_path.name]
                )
                self.assertEqual(stream_outputs, [])
                self.assertEqual(error_outputs, [])

    def test_marker_legend_names_start_and_endpoint(self):
        labels = [handle.get_label() for handle in marker_legend_handles()]

        self.assertEqual(
            labels,
            [
                "start pose (circle)",
                "agent STOP / last pose (x)",
                "object goal position (star)",
            ],
        )

    def test_objectnav_selection_key_includes_object_category(self):
        scene_id = "hm3d_v0.2/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb"
        chair_episode = SimpleNamespace(
            scene_id=scene_id,
            episode_id="0",
            object_category="chair",
        )
        toilet_episode = SimpleNamespace(
            scene_id=scene_id,
            episode_id="0",
            object_category="toilet",
        )

        self.assertNotEqual(
            _episode_selection_key(chair_episode),
            _episode_selection_key(toilet_episode),
        )
        self.assertEqual(
            _selection_record_key(
                SimpleNamespace(
                    source_trajectory_id=_episode_selection_key(chair_episode)[0],
                    object_category="chair",
                )
            ),
            _episode_selection_key(chair_episode),
        )

    def test_trajectory_record_key_separates_repeated_episode_ids(self):
        chair_record = {
            "episode_id": "0",
            "scene_id": "hm3d_v0.2/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb",
            "goal_text": "chair",
            "source_trajectory_id": "scene_a:0",
            "object_category": "chair",
        }
        toilet_record = {
            "episode_id": "0",
            "scene_id": "hm3d_v0.2/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb",
            "goal_text": "toilet",
            "source_trajectory_id": "scene_a:0",
            "object_category": "toilet",
        }
        other_scene_record = {
            "episode_id": "0",
            "scene_id": "hm3d_v0.2/train/00024-3XYAD64HpDr/3XYAD64HpDr.basis.glb",
            "goal_text": "chair",
            "source_trajectory_id": "scene_b:0",
            "object_category": "chair",
        }

        keys = {
            trajectory_record_key(chair_record),
            trajectory_record_key(toilet_record),
            trajectory_record_key(other_scene_record),
        }

        self.assertEqual(len(keys), 3)

    def test_scene_topdown_export_writes_manifest_and_pngs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_replays = [
                {
                    "scene": "00006-HkseAnWCgqk",
                    "topdown_map": np.zeros((24, 24), dtype=np.uint8),
                    "records": [
                        _plot_record("0", "chair", "scene_a:0"),
                        _plot_record("1", "chair", "scene_a:1"),
                    ],
                    "trajectories": [
                        _trajectory("0", "chair", "scene_a:0", [[3, 3], [5, 5]], [8, 8]),
                        _trajectory("1", "chair", "scene_a:1", [[4, 3], [6, 5]], [8, 8]),
                    ],
                },
                {
                    "scene": "00024-3XYAD64HpDr",
                    "topdown_map": np.zeros((24, 24), dtype=np.uint8),
                    "records": [_plot_record("2", "bed", "scene_b:2")],
                    "trajectories": [
                        _trajectory("2", "bed", "scene_b:2", [[10, 10], [12, 11]], [14, 15])
                    ],
                },
            ]

            manifest = scene_export.write_scene_topdown_pngs(
                scene_replays,
                root,
                data_root="data/habitat",
                exp="habitat/pr2l_hm3d_bc",
                dpi=80,
                command=["python", "analysis/code/export_hm3d_scene_topdown_pngs.py"],
            )

            self.assertEqual(manifest["scene_count"], 2)
            self.assertEqual(manifest["selected_episode_count"], 3)
            self.assertEqual(manifest["category_count_distribution"], {"1": 2})
            self.assertEqual(manifest["goal_count_distribution"], {"1": 2})
            self.assertEqual(manifest["stop_count_distribution"], {"1": 1, "2": 1})
            self.assertEqual(len(manifest["figures"]), 2)
            self.assertTrue((root / "result_manifest.json").exists())
            for figure in manifest["figures"]:
                self.assertTrue((root / figure["png_path"]).exists())
                self.assertIn("unique_start_position_count", figure)
                self.assertIn("unique_stop_position_count", figure)

    def test_scene_topdown_export_writes_scene_outputs_during_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_replays = [
                {
                    "scene": "00006-HkseAnWCgqk",
                    "topdown_map": np.zeros((24, 24), dtype=np.uint8),
                    "records": [_plot_record("0", "chair", "scene_a:0")],
                    "trajectories": [
                        _trajectory("0", "chair", "scene_a:0", [[3, 3], [5, 5]], [8, 8])
                    ],
                },
                {
                    "scene": "00024-3XYAD64HpDr",
                    "topdown_map": np.zeros((24, 24), dtype=np.uint8),
                    "records": [_plot_record("1", "bed", "scene_b:1")],
                    "trajectories": [
                        _trajectory("1", "bed", "scene_b:1", [[10, 10], [12, 11]], [14, 15])
                    ],
                },
            ]
            observed_png_counts = []
            observed_manifest_status = []

            def fake_replay(data_root, *, exp, on_scene_replay=None):
                self.assertEqual(str(data_root), "data/habitat")
                self.assertEqual(exp, "habitat/pr2l_hm3d_bc")
                self.assertIsNotNone(on_scene_replay)
                for scene_index, replay in enumerate(scene_replays):
                    on_scene_replay(scene_index, replay)
                    observed_png_counts.append(len(list((root / "scenes").glob("*.png"))))
                    manifest_path = root / "result_manifest.json"
                    self.assertTrue(manifest_path.exists())
                    observed_manifest_status.append(json.loads(manifest_path.read_text())["status"])
                return scene_replays

            original_replay = scene_export.replay_selected_objectnav_scene_topdowns
            scene_export.replay_selected_objectnav_scene_topdowns = fake_replay
            try:
                manifest = scene_export.export_scene_topdown_pngs(
                    "data/habitat",
                    exp="habitat/pr2l_hm3d_bc",
                    output_dir=root,
                    dpi=80,
                    command=["python", "analysis/code/export_hm3d_scene_topdown_pngs.py"],
                )
            finally:
                scene_export.replay_selected_objectnav_scene_topdowns = original_replay

            self.assertEqual(observed_png_counts, [1, 2])
            self.assertEqual(observed_manifest_status, ["running", "running"])
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["scene_count"], 2)


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


def _plot_record(
    episode_id: str,
    object_category: str,
    source_trajectory_id: str,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "scene_id": "hm3d_v0.2/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb",
        "goal_text": object_category,
        "object_category": object_category,
        "source_trajectory_id": source_trajectory_id,
    }


def _trajectory(
    episode_id: str,
    object_category: str,
    source_trajectory_id: str,
    grid_positions: list[list[int]],
    goal_grid_position: list[int],
) -> dict[str, object]:
    record = _plot_record(episode_id, object_category, source_trajectory_id)
    return {
        "record": record,
        "grid_positions": np.asarray(grid_positions, dtype=np.int64),
        "goal_grid_position": np.asarray(goal_grid_position, dtype=np.int64),
    }
