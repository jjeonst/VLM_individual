import contextlib
import io
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np

from configs.schema import TopoVLMConfig
from data.hm3d_objectnav_render import (
    _configure_habitat_dataset,
    _filter_env_episodes_to_selection,
    build_hm3d_objectnav_episode_manifest,
)


class HM3DObjectNavRenderTest(unittest.TestCase):
    def test_builds_manifest_from_shortest_path_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TopoVLMConfig()
            cfg.data.data_root = tmpdir
            cfg.data.dataset_name = "hm3d_objectnav_render_test"
            cfg.data.episodes_manifest = (
                "episodes/hm3d_objectnav_render_test/train/manifest.jsonl"
            )
            cfg.eval.max_steps = 4
            env = _FakeEnv([_FakeEpisode("0", "scene/scene.glb", "chair")])
            follower = _FakeFollower([1, 2, 0])

            result = build_hm3d_objectnav_episode_manifest(cfg, env=env, follower=follower)
            manifest = Path(result["manifest"])
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rgb = np.load(Path(tmpdir) / records[0]["rgb_path"])
            actions = np.load(Path(tmpdir) / records[0]["actions_path"])

            self.assertEqual(result["episodes_written"], 1)
            self.assertEqual(records[0]["source_dataset"], "hm3d_objectnav_shortest_path")
            self.assertEqual(records[0]["goal_text"], "chair")
            self.assertEqual(rgb.shape, (3, 2, 2, 3))
            self.assertEqual(actions.tolist(), [1, 2, 0])

    def test_episode_timeout_skips_episode_and_logs_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TopoVLMConfig()
            cfg.data.data_root = tmpdir
            cfg.data.dataset_name = "hm3d_objectnav_render_test"
            cfg.data.episodes_manifest = (
                "episodes/hm3d_objectnav_render_test/train/manifest.jsonl"
            )
            cfg.eval.episode_timeout_seconds = 1
            env = _FakeEnv([_FakeEpisode("0", "scene/scene.glb", "chair")])
            follower = _SlowFollower()
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                result = build_hm3d_objectnav_episode_manifest(cfg, env=env, follower=follower)

            manifest = Path(result["manifest"])
            events = [
                json.loads(line)
                for line in stdout.getvalue().splitlines()
                if line.startswith("{")
            ]

            self.assertEqual(result["episodes_written"], 0)
            self.assertEqual(result["episodes_skipped"], 1)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "")
            self.assertEqual(events[0]["event"], "hm3d_build_episodes_skip")
            self.assertEqual(events[0]["error_type"], "TimeoutError")
            self.assertIn("episode_timeout_seconds=1", events[0]["error"])

    def test_filters_selection_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TopoVLMConfig()
            cfg.data.data_root = tmpdir
            cfg.data.dataset_name = "pr2l_hm3d_objectnav_balanced_subset"
            cfg.data.episodes_manifest = (
                "episodes/pr2l_hm3d_objectnav_balanced_subset/train/manifest.jsonl"
            )
            cfg.data.episode_selection_manifest = (
                "episode_selections/pr2l_hm3d_objectnav/train_scene_object_balanced.jsonl"
            )
            selection = Path(tmpdir) / cfg.data.episode_selection_manifest
            selection.parent.mkdir(parents=True)
            selection.write_text(
                json.dumps(
                    {
                        "source_trajectory_id": "scene_b/scene.glb:1",
                        "episode_id": "1",
                        "scene_id": "scene_b/scene.glb",
                        "object_category": "table",
                        "shard_path": "content/scene_b.json.gz",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            env = _FakeEnv(
                [
                    _FakeEpisode("0", "scene_a/scene.glb", "chair"),
                    _FakeEpisode("1", "scene_b/scene.glb", "table"),
                ]
            )
            follower = _FakeFollower([0])

            result = build_hm3d_objectnav_episode_manifest(cfg, env=env, follower=follower)
            records = [
                json.loads(line)
                for line in Path(result["manifest"]).read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(result["episodes_written"], 1)
            self.assertEqual(records[0]["source_trajectory_id"], "scene_b/scene.glb:1")
            self.assertEqual(env.reset_count, 1)

    def test_filters_env_episode_list_before_resetting(self):
        env = _FakeEnv(
            [
                _FakeEpisode("0", "scene_a/scene.glb", "chair"),
                _FakeEpisode("1", "scene_b/scene.glb", "table"),
                _FakeEpisode("2", "scene_c/scene.glb", "sofa"),
            ]
        )

        _filter_env_episodes_to_selection(
            env,
            {
                "scene_b/scene.glb:1",
                "scene_c/scene.glb:2",
            },
        )

        self.assertEqual(
            [episode.episode_id for episode in env.episodes],
            ["1", "2"],
        )

    def test_habitat_dataset_split_follows_data_config(self):
        cfg = TopoVLMConfig()
        cfg.data.split = "val"
        cfg.data.data_root = "/data/topovlm/habitat"
        habitat_config = types.SimpleNamespace(
            habitat=types.SimpleNamespace(
                dataset=types.SimpleNamespace(
                    split="train",
                    data_path=(
                        "/data/topovlm/habitat/datasets/objectnav/hm3d/v2/"
                        "objectnav_hm3d_v2/{split}/{split}.json.gz"
                    ),
                    scenes_dir="/data/topovlm/habitat/scene_datasets",
                ),
                simulator=types.SimpleNamespace(
                    scene_dataset=(
                        "/data/topovlm/habitat/scene_datasets/hm3d_v0.2/"
                        "hm3d_annotated_basis.scene_dataset_config.json"
                    )
                ),
            )
        )
        fake_omegaconf = types.SimpleNamespace(OmegaConf=_FakeOmegaConf)
        original_omegaconf = sys.modules.get("omegaconf")
        sys.modules["omegaconf"] = fake_omegaconf
        try:
            _configure_habitat_dataset(habitat_config, cfg, Path(cfg.data.data_root))
        finally:
            if original_omegaconf is None:
                del sys.modules["omegaconf"]
            else:
                sys.modules["omegaconf"] = original_omegaconf
        self.assertEqual(habitat_config.habitat.dataset.split, "val")


class _FakeEnv:
    def __init__(self, episodes):
        self.episodes = episodes
        self.current_episode = None
        self.sim = _FakeSim()
        self._next_index = 0
        self._frame_id = 0
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.current_episode = self.episodes[self._next_index]
        self._next_index += 1
        self._frame_id = 0
        return self._observation()

    def step(self, action):
        self._frame_id += 1
        return self._observation()

    def close(self):
        pass

    def _observation(self):
        return {
            "rgb": np.full((2, 2, 3), self._frame_id, dtype=np.uint8),
        }


class _FakeFollower:
    def __init__(self, actions):
        self.actions = list(actions)

    def get_next_action(self, goal_position):
        return self.actions.pop(0)


class _SlowFollower:
    def get_next_action(self, goal_position):
        time.sleep(2)
        return 1


class _FakeSim:
    def get_agent_state(self):
        return type("AgentState", (), {"position": [0.0, 0.0, 0.0]})()

    def geodesic_distance(self, position_a, position_b, episode=None):
        return float(np.linalg.norm(np.asarray(position_b) - np.asarray(position_a)))


class _FakeEpisode:
    def __init__(self, episode_id, scene_id, object_category):
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.object_category = object_category
        self.goals = [_FakeGoal()]


class _FakeGoal:
    position = [1.0, 0.0, 0.0]
    view_points = []


class _FakeOmegaConf:
    @staticmethod
    def set_readonly(config, readonly):
        pass


if __name__ == "__main__":
    unittest.main()
