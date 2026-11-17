"""
train_from_selected 配置与因子清单解析测试
"""
import tempfile
import unittest
from pathlib import Path

from scripts.training.train_from_selected import (
    LOCAL_CONFIG,
    build_effective_local_config,
    load_selected_factors,
)


class TestTrainFromSelectedConfig(unittest.TestCase):
    def test_local_config_defaults_to_main_board_universe(self):
        self.assertEqual(LOCAL_CONFIG["instruments"], "main_board")

    def test_load_selected_factors_supports_txt_factor_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            factor_list_path = Path(tmpdir) / "selected_factors_tree.txt"
            factor_list_path.write_text(
                "1. VOL_REV\n2. ROC30\n3. BETA20\n",
                encoding="utf-8",
            )

            source_files, factor_names = load_selected_factors(str(factor_list_path))

        self.assertEqual(factor_names, ["VOL_REV", "ROC30", "BETA20"])
        self.assertTrue(source_files)
        self.assertTrue(all(isinstance(item, str) and item for item in source_files))

    def test_build_effective_local_config_extends_to_latest_calendar_date(self):
        config = dict(LOCAL_CONFIG)
        config["end_time"] = "2025-12-31"
        config["rolling_windows"] = [
            {
                "name": "Test_2025",
                "train": ("2022-01-01", "2023-12-20"),
                "valid": ("2024-01-01", "2024-12-20"),
                "test": ("2025-01-01", "2025-12-31"),
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            calendar_path = Path(tmpdir) / "day.txt"
            calendar_path.write_text("2025-12-31\n2026-07-14\n", encoding="utf-8")
            effective = build_effective_local_config(config, calendar_path=calendar_path)

        self.assertEqual(effective["end_time"], "2026-07-14")
        self.assertEqual(effective["rolling_windows"][-1]["name"], "Test_2026")
        self.assertEqual(effective["rolling_windows"][-1]["test"], ("2026-01-01", "2026-07-14"))

    def test_build_effective_local_config_disables_label_neutralize_by_default(self):
        config = dict(LOCAL_CONFIG)
        config["neutralize_labels"] = True

        effective = build_effective_local_config(config, calendar_path=None, latest_calendar_date="2026-07-14")

        self.assertFalse(effective["neutralize_labels"])

    def test_local_config_defaults_to_selected_live_profile_for_tdx(self):
        self.assertEqual(LOCAL_CONFIG["live_strategy_name"], "selected")
        self.assertEqual(LOCAL_CONFIG["live_runtime_model_name"], "tree")


if __name__ == "__main__":
    unittest.main()
