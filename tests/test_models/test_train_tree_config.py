"""
train_tree 时间配置测试
"""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "training" / "train_tree.py"
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPEC = importlib.util.spec_from_file_location("train_tree", SCRIPT_PATH)
train_tree = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(train_tree)


class TestTrainTreeConfig(unittest.TestCase):
    def test_build_effective_local_config_extends_to_latest_date(self):
        base_config = copy.deepcopy(train_tree.LOCAL_CONFIG)
        base_config["end_time"] = "2025-12-31"
        base_config["rolling_windows"] = [
            {
                "name": "Test_2025",
                "train": ("2022-01-01", "2023-12-20"),
                "valid": ("2024-01-01", "2024-12-20"),
                "test": ("2025-01-01", "2025-12-31"),
            }
        ]

        result = train_tree.build_effective_local_config(
            base_config=base_config,
            latest_date="2026-07-14",
        )

        self.assertEqual(result["end_time"], "2026-07-14")
        self.assertEqual(result["rolling_windows"][-1]["name"], "Test_2026")
        self.assertEqual(result["rolling_windows"][-1]["train"], ("2023-01-01", "2024-12-20"))
        self.assertEqual(result["rolling_windows"][-1]["valid"], ("2025-01-01", "2025-12-20"))
        self.assertEqual(result["rolling_windows"][-1]["test"], ("2026-01-01", "2026-07-14"))

    def test_build_effective_local_config_keeps_existing_range_when_not_needed(self):
        base_config = copy.deepcopy(train_tree.LOCAL_CONFIG)
        result = train_tree.build_effective_local_config(
            base_config=base_config,
            latest_date="2025-12-31",
        )

        self.assertEqual(result["end_time"], "2025-12-31")
        self.assertEqual(result["rolling_windows"][-1]["name"], "Test_2025")


if __name__ == "__main__":
    unittest.main()
