import unittest

from configs import build_config_from_exp


class ConfigBuilderTest(unittest.TestCase):
    def test_habitat_smoke_config_loads(self):
        cfg = build_config_from_exp("habitat/prismatic_bc_smoke")
        self.assertEqual(cfg.config_name, "habitat/prismatic_bc_smoke")
        self.assertEqual(cfg.model.vlm.backend, "prismatic")
        self.assertEqual(cfg.model.policy.type, "graph_transformer_bc")
        self.assertEqual(cfg.objectives.names, ["behavior_cloning"])

    def test_debug_overrides_are_operational(self):
        cfg = build_config_from_exp("habitat/prismatic_bc_smoke", debug=True)
        self.assertTrue(cfg.debug)
        self.assertFalse(cfg.wandb)
        self.assertTrue(cfg.data.synthetic_debug)
        self.assertEqual(cfg.max_epochs, 1)
        self.assertEqual(cfg.data.batch_size, 2)


if __name__ == "__main__":
    unittest.main()
