"""
训练脚本配置加载测试
"""
import importlib.util
import unittest
from pathlib import Path


def _load_training_config_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "training" / "_config.py"
    spec = importlib.util.spec_from_file_location("training_config_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载配置模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainingScriptConfig(unittest.TestCase):
    def test_all_training_yaml_configs_can_be_loaded(self):
        module = _load_training_config_module()

        for config_name in ("tree_2025", "linear_2025", "icir_2025"):
            config = module.load_config(config_name)
            self.assertIn("model_type", config)
            self.assertIn("label_names", config)

    def test_linear_yaml_loads_required_preprocess_keys(self):
        module = _load_training_config_module()

        config = module.load_config("linear_2025")

        self.assertEqual(config["model_type"], "linear")
        self.assertTrue(config["normalize_features"])
        self.assertTrue(config["neutralize_features"])
        self.assertTrue(config["renormalize_features_after_neutralize"])
        self.assertTrue(config["normalize_labels"])
        self.assertFalse(config["neutralize_labels"])
        self.assertTrue(config["symmetric_orthogonalization"])
        self.assertEqual(config["factor_cache_names"], [])

    def test_resolve_runtime_config_defaults_to_local_config(self):
        module = _load_training_config_module()
        local_config = {"model_type": "tree", "top_k_factors": 5}

        config = module.resolve_runtime_config(
            local_config=local_config,
            default_yaml_name="tree_2025",
        )

        self.assertEqual(config["model_type"], "tree")
        self.assertEqual(config["top_k_factors"], 5)
        self.assertIsNot(config, local_config)

    def test_resolve_runtime_config_uses_default_yaml_name_when_requested(self):
        module = _load_training_config_module()
        local_config = {"model_type": "linear", "top_k_factors": 20}

        config = module.resolve_runtime_config(
            local_config=local_config,
            default_yaml_name="linear_2025",
            config_source="yaml",
        )

        self.assertEqual(config["model_type"], "linear")
        self.assertEqual(config["top_k_factors"], 50)
        self.assertTrue(config["normalize_features"])

    def test_resolve_runtime_config_rejects_unknown_source(self):
        module = _load_training_config_module()

        with self.assertRaises(ValueError):
            module.resolve_runtime_config(
                local_config={"model_type": "tree"},
                default_yaml_name="tree_2025",
                config_source="unknown",
            )


if __name__ == "__main__":
    unittest.main()
