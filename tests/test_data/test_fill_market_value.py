"""
fill_market_value 合并逻辑测试
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "data" / "fill_market_value.py"
SPEC = importlib.util.spec_from_file_location("fill_market_value", MODULE_PATH)
fill_market_value = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fill_market_value)


class TestMergeMarketValueSeries(unittest.TestCase):
    def test_merge_extends_existing_series_when_new_values_are_longer(self):
        existing = np.array([1.0, 2.0, 3.0], dtype="<f4")
        new_vals = np.array([np.nan, 20.0, 30.0, 40.0, 50.0], dtype="<f4")

        start_idx, merged, overwrite_count = fill_market_value.merge_market_value_series(
            existing_start_idx=0,
            existing_vals=existing,
            new_start_idx=0,
            new_vals=new_vals,
        )

        self.assertEqual(start_idx, 0)
        self.assertEqual(overwrite_count, 4)
        np.testing.assert_allclose(merged[:5], np.array([1.0, 20.0, 30.0, 40.0, 50.0], dtype="<f4"))

    def test_merge_preserves_earlier_existing_range_when_new_start_is_later(self):
        existing = np.array([10.0, 11.0, 12.0], dtype="<f4")
        new_vals = np.array([20.0, np.nan], dtype="<f4")

        start_idx, merged, overwrite_count = fill_market_value.merge_market_value_series(
            existing_start_idx=5,
            existing_vals=existing,
            new_start_idx=6,
            new_vals=new_vals,
        )

        self.assertEqual(start_idx, 5)
        self.assertEqual(overwrite_count, 1)
        np.testing.assert_allclose(merged, np.array([10.0, 20.0, 12.0], dtype="<f4"))


if __name__ == "__main__":
    unittest.main()
