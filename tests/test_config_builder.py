import unittest

from configs import build_config_from_exp


class ConfigBuilderTest(unittest.TestCase):
    def test_habitat_smoke_config_loads(self):
        cfg = build_config_from_exp("habitat/prismatic_bc_smoke")
        self.assertEqual(cfg.config_name, "habitat/prismatic_bc_smoke")
        self.assertEqual(cfg.model.vlm.backend, "prismatic")
        self.assertEqual(cfg.model.policy.type, "graph_transformer_bc")
        self.assertEqual(cfg.objectives.names, ["behavior_cloning"])
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
        self.assertEqual(cfg.data.data_root, "data/habitat")
        self.assertEqual(
            cfg.data.habitat_config,
            "configs/habitat/pr2l_objectnav_staged.yaml",
        )
        self.assertEqual(
            cfg.model.vlm.weights_path,
            "data/vlm_weights/prismatic/prism-dinosiglip+7b",
        )


if __name__ == "__main__":
    unittest.main()
