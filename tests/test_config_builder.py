from pathlib import Path
import unittest

import yaml

from configs import build_config_from_exp


class ConfigBuilderTest(unittest.TestCase):
    def test_habitat_smoke_config_loads(self):
        cfg = build_config_from_exp("habitat/prismatic_bc_smoke")
        self.assertEqual(cfg.config_name, "habitat/prismatic_bc_smoke")
        self.assertEqual(cfg.model.vlm.backend, "prismatic")
        self.assertEqual(cfg.model.policy.type, "graph_transformer_bc")
        self.assertEqual(cfg.objectives.names, ["behavior_cloning"])
        self.assertEqual(cfg.data.dataset_name, "habitat_objectnav_hm3d")
        self.assertEqual(
            cfg.data.objectnav_dataset_dir,
            "datasets/objectnav/hm3d/v2/objectnav_hm3d_v2",
        )
        self.assertEqual(
            cfg.data.scene_dataset_config,
            "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        )

    def test_debug_overrides_are_operational(self):
        cfg = build_config_from_exp("habitat/prismatic_bc_smoke", debug=True)
        self.assertTrue(cfg.debug)
        self.assertFalse(cfg.wandb)
        self.assertTrue(cfg.data.synthetic_debug)
        self.assertEqual(cfg.max_epochs, 1)
        self.assertEqual(cfg.data.batch_size, 2)

    def test_staged_habitat_smoke_config_loads(self):
        cfg = build_config_from_exp("habitat/prismatic_bc_smoke_staged")
        self.assertEqual(cfg.config_name, "habitat/prismatic_bc_smoke_staged")
        self.assertEqual(cfg.data.data_root, "data/topovlm/habitat")
        self.assertEqual(
            cfg.data.habitat_config,
            "configs/habitat/pr2l_objectnav_staged.yaml",
        )
        self.assertEqual(
            cfg.model.vlm.weights_path,
            "data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b",
        )

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


def _duplicate_default_leaf_paths(default: dict, override: dict, prefix: tuple[str, ...] = ()) -> list[str]:
    duplicate_paths = []
    for key, value in override.items():
        if key not in default:
            continue
        default_value = default[key]
        current_path = prefix + (str(key),)
        if isinstance(value, dict) and isinstance(default_value, dict):
            duplicate_paths.extend(_duplicate_default_leaf_paths(default_value, value, current_path))
        elif value == default_value:
            duplicate_paths.append(".".join(current_path))
    return duplicate_paths


if __name__ == "__main__":
    unittest.main()
