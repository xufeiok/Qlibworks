"""
generate_tree_targets 目标持仓导出测试
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qlworks.live.tree_strategy import get_live_strategy_config
from scripts.live.generate_tree_targets import generate_targets


class TestGenerateTreeTargets(unittest.TestCase):
    def test_selected_strategy_uses_selected_score_file_and_tree_runtime(self):
        config = get_live_strategy_config("selected")

        self.assertEqual(config["model_name"], "tree_selected")
        self.assertEqual(config["score_file"], "score_tree_selected.csv")
        self.assertEqual(config["runtime_model_name"], "tree")

    def test_generate_targets_supports_selected_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            score_path = project_root / "scripts" / "training" / "score_tree_selected.csv"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2025-01-08"] * 3),
                    "instrument": ["000001.sz", "600519.sh", "000002.sz"],
                    "score": [0.91, 0.96, 0.72],
                    "raw_score": [0.11, 0.22, 0.33],
                }
            ).to_csv(score_path, index=False)

            with patch("scripts.live.generate_tree_targets.PROJECT_ROOT", project_root):
                target_df, output_path = generate_targets(strategy_name="selected")

            self.assertEqual(output_path, project_root / "runtime" / "live" / "tree" / "signals" / "daily" / "target_positions_20250108.csv")
            self.assertEqual(target_df["instrument"].tolist(), ["600519.sh", "000001.sz", "000002.sz"])

            state_path = project_root / "runtime" / "live" / "tree" / "state" / "latest_target.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["model_name"], "tree_selected")
            self.assertEqual(state["score_file"], "score_tree_selected.csv")


if __name__ == "__main__":
    unittest.main()
