from pathlib import Path
import unittest

import yaml

from configs import build_config_from_exp


class ConfigBuilderTest(unittest.TestCase):
    def test_pr2l_hm3d_config_loads(self):
        cfg = build_config_from_exp("habitat/pr2l_hm3d_bc")
        self.assertEqual(cfg.config_name, "habitat/pr2l_hm3d_bc")
        self.assertEqual(cfg.run_name, "pr2l_hm3d_bc")
        self.assertEqual(cfg.data.dataset_name, "pr2l_hm3d_objectnav")
        self.assertEqual(cfg.data.trajectory_source, "objectnav_shortest_path")
        self.assertEqual(cfg.data.cache_format, "pr2l_token_trajectory")
        self.assertEqual(cfg.data.data_root, "/data/topovlm/habitat")
        self.assertEqual(
            cfg.data.objectnav_dataset_dir,
            "datasets/objectnav/hm3d/v2/objectnav_hm3d_v2",
        )
        self.assertEqual(cfg.data.scene_dataset_dir, "scene_datasets")
        self.assertEqual(
            cfg.data.scene_dataset_config,
            "scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json",
        )
        self.assertEqual(
            cfg.data.episodes_manifest,
            "episodes/pr2l_hm3d_objectnav/train/manifest.jsonl",
        )
        self.assertEqual(
            cfg.data.graph_manifest,
            "graphs/pr2l_hm3d_bc/train/manifest.jsonl",
        )
        self.assertEqual(cfg.model.vlm.representation, "pr2l_visual_tokens_last_two_layers")
        self.assertEqual(cfg.model.vlm.hidden_layer_indices, [-2, -1])
        self.assertEqual(cfg.model.vlm.visual_pool_grid, 4)
        self.assertEqual(cfg.model.vlm.projection, "pca")
        self.assertEqual(cfg.model.vlm.projection_dim, 1024)
        self.assertEqual(cfg.model.vlm.output_dim, 1024)
        self.assertEqual(cfg.model.policy.input_dim, 1024)
        self.assertEqual(cfg.model.policy.num_actions, 4)
        self.assertEqual(cfg.model.policy.prediction_target, "nodes")
        self.assertEqual(cfg.objectives.behavior_cloning.stop_turn_action_ids, [0, 2, 3])
        self.assertEqual(cfg.gradient_accumulation_steps, 8)
        self.assertTrue(cfg.wandb)
        self.assertEqual(cfg.wandb_entity, "topovlm")
        self.assertEqual(cfg.wandb_project, "TopoVLM")
        self.assertEqual(cfg.wandb_group, "pr2l_prismatic_policy")
        self.assertEqual(cfg.wandb_contract_role_id, "habitat_bc")

    def test_debug_overrides_are_operational(self):
        cfg = build_config_from_exp("habitat/pr2l_hm3d_bc", debug=True)
        self.assertTrue(cfg.debug)
        self.assertFalse(cfg.wandb)
        self.assertTrue(cfg.data.synthetic_debug)
        self.assertEqual(cfg.max_epochs, 1)
        self.assertEqual(cfg.data.max_episodes, 4)
        self.assertEqual(cfg.data.batch_size, 2)
        self.assertEqual(cfg.data.num_workers, 0)

    def test_minimal_canonical_habitat_exp_configs_are_tracked(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "exp" / "habitat"
        self.assertEqual(
            sorted(path.name for path in config_root.glob("*.yaml")),
            [
                "pr2l_habitat_bc_faithful.yaml",
                "pr2l_habitat_bc_tiny_smoke.yaml",
                "pr2l_hm3d_bc.yaml",
            ],
        )

    def test_pr2l_habitat_web_faithful_config_loads(self):
        cfg = build_config_from_exp("habitat/pr2l_habitat_bc_faithful")
        self.assertEqual(cfg.config_name, "habitat/pr2l_habitat_bc_faithful")
        self.assertEqual(cfg.run_name, "pr2l_habitat_bc_faithful")
        self.assertEqual(cfg.data.dataset_name, "pr2l_habitat_web")
        self.assertEqual(cfg.data.trajectory_source, "habitat_web_replay")
        self.assertEqual(cfg.data.cache_format, "pr2l_token_trajectory")
        self.assertEqual(cfg.data.data_root, "/data/topovlm/habitat")
        self.assertEqual(
            cfg.data.objectnav_dataset_dir,
            "sources/habitat_web_hf_metadata/objectnav/objectnav_mp3d_thda_70k",
        )
        self.assertEqual(cfg.data.scene_dataset_dir, "scene_datasets")
        self.assertEqual(
            cfg.data.scene_dataset_config,
            "scene_datasets/mp3d/mp3d.scene_dataset_config.json",
        )
        self.assertEqual(
            cfg.data.episodes_manifest,
            "episodes/pr2l_habitat_web/train/manifest.jsonl",
        )
        self.assertEqual(
            cfg.data.graph_manifest,
            "graphs/pr2l_habitat_bc_faithful/train/manifest.jsonl",
        )
        self.assertEqual(
            cfg.data.episode_selection_manifest,
            "episode_selections/pr2l_habitat_web/train_scene_object_balanced_7550.jsonl",
        )
        self.assertEqual(cfg.data.balanced_subset_size, 7550)
        self.assertEqual(cfg.model.vlm.representation, "pr2l_visual_tokens_last_two_layers")
        self.assertEqual(cfg.model.vlm.hidden_layer_indices, [-2, -1])
        self.assertEqual(cfg.model.vlm.visual_pool_grid, 4)
        self.assertEqual(cfg.model.vlm.projection, "pca")
        self.assertEqual(cfg.model.vlm.projection_dim, 1024)
        self.assertTrue(cfg.model.vlm.include_generated_text)
        self.assertEqual(cfg.model.vlm.output_dim, 2048)
        self.assertEqual(cfg.model.policy.input_dim, 2048)
        self.assertEqual(cfg.model.policy.num_actions, 6)
        self.assertEqual(cfg.model.policy.prediction_target, "nodes")
        self.assertEqual(cfg.objectives.behavior_cloning.stop_turn_action_ids, [0, 2, 3, 4, 5])
        self.assertEqual(cfg.gradient_accumulation_steps, 8)
        self.assertTrue(cfg.wandb)
        self.assertEqual(cfg.wandb_entity, "topovlm")
        self.assertEqual(cfg.wandb_project, "TopoVLM")
        self.assertEqual(cfg.wandb_group, "pr2l_prismatic_policy")
        self.assertEqual(cfg.wandb_contract_role_id, "habitat_bc")

    def test_pr2l_habitat_web_tiny_smoke_config_loads(self):
        cfg = build_config_from_exp("habitat/pr2l_habitat_bc_tiny_smoke")
        self.assertEqual(cfg.config_name, "habitat/pr2l_habitat_bc_tiny_smoke")
        self.assertEqual(cfg.run_name, "pr2l_habitat_bc_tiny_smoke")
        self.assertEqual(cfg.data.dataset_name, "pr2l_habitat_web_tiny_smoke")
        self.assertEqual(cfg.data.trajectory_source, "habitat_web_replay")
        self.assertEqual(cfg.data.max_episodes, 4)
        self.assertEqual(cfg.data.batch_size, 2)
        self.assertEqual(cfg.data.num_workers, 0)
        self.assertEqual(
            cfg.data.episodes_manifest,
            "episodes/pr2l_habitat_web_tiny_smoke/train/manifest.jsonl",
        )
        self.assertEqual(
            cfg.data.graph_manifest,
            "graphs/pr2l_habitat_bc_tiny_smoke/train/manifest.jsonl",
        )
        self.assertEqual(
            cfg.model.vlm.projection_path,
            "embeddings/pr2l_habitat_bc_tiny_smoke/projection_pca.npz",
        )
        self.assertIsNone(cfg.data.episode_selection_manifest)
        self.assertIsNone(cfg.data.balanced_subset_size)
        self.assertEqual(cfg.model.vlm.output_dim, 2048)
        self.assertEqual(cfg.model.policy.input_dim, 2048)
        self.assertEqual(cfg.model.policy.num_actions, 6)
        self.assertEqual(cfg.max_epochs, 1)
        self.assertEqual(cfg.gradient_accumulation_steps, 1)
        self.assertFalse(cfg.wandb)

    def test_domain_configs_only_declare_default_overrides(self):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        for domain in ("data", "eval", "model", "objectives", "train"):
            default = _load_yaml(config_root / domain / "default.yaml")
            for path in sorted((config_root / domain).glob("*.yaml")):
                if path.name == "default.yaml":
                    continue
                duplicate_paths = _duplicate_default_leaf_paths(default, _load_yaml(path))
                self.assertEqual(
                    duplicate_paths,
                    [],
                    f"{path} repeats default values: {duplicate_paths}",
                )
                self.assertNotEqual(
                    _load_yaml(path),
                    {},
                    f"{path} is an empty placeholder config",
                )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise AssertionError(f"Config file must be a mapping: {path}")
    return loaded


def _duplicate_default_leaf_paths(
    default: dict, override: dict, prefix: tuple[str, ...] = ()
) -> list[str]:
    duplicate_paths = []
    for key, value in override.items():
        if key not in default:
            continue
        default_value = default[key]
        current_path = prefix + (str(key),)
        if isinstance(value, dict) and isinstance(default_value, dict):
            duplicate_paths.extend(
                _duplicate_default_leaf_paths(default_value, value, current_path)
            )
        elif value == default_value:
            duplicate_paths.append(".".join(current_path))
    return duplicate_paths
