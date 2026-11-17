"""
2024 标签体检脚本测试
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "training" / "inspect_2024_labels.py"
    spec = importlib.util.spec_from_file_location("inspect_2024_labels_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class TestInspect2024Labels(unittest.TestCase):
    def test_flag_label_anomalies_marks_extreme_and_bad_price_rows(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
                "instrument": ["000001.sz", "000002.sz", "000003.sz"],
                "open": [10.0, 0.0, 12.0],
                "close": [10.5, 11.0, 12.2],
                "next_open": [10.2, 11.2, 0.01],
                "future_close_5d": [10.6, 120.0, 0.01],
                "label_5d": [0.039216, 9.714286, -0.999],
            }
        )

        result = MODULE.flag_label_anomalies(df, abs_label_threshold=5.0, min_price_threshold=0.05)

        self.assertEqual(result["is_anomaly"].tolist(), [False, True, True])
        self.assertEqual(result["price_issue"].tolist(), [False, True, True])
        self.assertEqual(result["label_issue"].tolist(), [False, True, False])

    def test_summarize_anomalies_by_stock_and_date(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
                "instrument": ["000001.sz", "000001.sz", "000002.sz", "000003.sz"],
                "label_5d": [8.0, -7.0, 6.0, 0.1],
                "is_anomaly": [True, True, True, False],
                "label_issue": [True, True, True, False],
                "price_issue": [False, False, True, False],
            }
        )

        by_stock = MODULE.summarize_anomalies_by_stock(df)
        by_date = MODULE.summarize_anomalies_by_date(df)

        self.assertEqual(by_stock.iloc[0]["instrument"], "000001.sz")
        self.assertEqual(int(by_stock.iloc[0]["anomaly_count"]), 2)
        self.assertEqual(str(by_date.iloc[0]["datetime"].date()), "2024-01-02")
        self.assertEqual(int(by_date.iloc[0]["anomaly_count"]), 2)

    def test_build_overview_contains_key_counts(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
                "instrument": ["000001.sz", "000002.sz", "000003.sz"],
                "label_5d": [0.1, 8.0, -9.0],
                "is_anomaly": [False, True, True],
                "label_issue": [False, True, True],
                "price_issue": [False, False, True],
            }
        )

        overview = MODULE.build_overview(df)

        self.assertEqual(overview["total_rows"], 3)
        self.assertEqual(overview["anomaly_rows"], 2)
        self.assertEqual(overview["label_issue_rows"], 2)
        self.assertEqual(overview["price_issue_rows"], 1)


if __name__ == "__main__":
    unittest.main()
