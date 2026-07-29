"""
实盘联动公共逻辑测试
"""
import unittest

import pandas as pd

from qlworks.live.symbol_mapper import normalize_symbol_to_tdx, normalize_symbol_to_qlib
from qlworks.live.targets import build_daily_target_positions


class TestSymbolMapper(unittest.TestCase):
    def test_normalize_symbol_to_tdx_from_qlib_style(self):
        self.assertEqual(normalize_symbol_to_tdx("600000.sh"), "600000.SH")
        self.assertEqual(normalize_symbol_to_tdx("000001.sz"), "000001.SZ")

    def test_normalize_symbol_to_qlib_from_tdx_style(self):
        self.assertEqual(normalize_symbol_to_qlib("600000.SH"), "600000.sh")
        self.assertEqual(normalize_symbol_to_qlib("000001.SZ"), "000001.sz")

    def test_normalize_symbol_rejects_invalid_code(self):
        with self.assertRaises(ValueError):
            normalize_symbol_to_tdx("SH600000")


class TestBuildDailyTargetPositions(unittest.TestCase):
    def test_build_daily_target_positions_keeps_top_k_and_equal_weights(self):
        score_df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2025-01-08"] * 4),
                "instrument": ["000001.sz", "000002.sz", "600000.sh", "600519.sh"],
                "score": [0.91, 0.82, 0.61, 0.95],
                "raw_score": [0.10, 0.20, 0.30, 0.40],
            }
        )

        result = build_daily_target_positions(
            score_df=score_df,
            trade_date="2025-01-08",
            top_k=2,
            score_threshold=0.8,
            buy_pct=0.95,
        )

        self.assertEqual(result["instrument"].tolist(), ["600519.sh", "000001.sz"])
        self.assertAlmostEqual(result["target_weight"].sum(), 0.95, places=8)
        self.assertTrue((result["target_weight"] == 0.475).all())

    def test_build_daily_target_positions_returns_empty_when_no_symbol_passes_threshold(self):
        score_df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2025-01-08"] * 2),
                "instrument": ["000001.sz", "000002.sz"],
                "score": [0.11, 0.12],
                "raw_score": [0.10, 0.20],
            }
        )

        result = build_daily_target_positions(
            score_df=score_df,
            trade_date="2025-01-08",
            top_k=5,
            score_threshold=0.7,
            buy_pct=0.95,
        )

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
