"""
train_from_selected 运行期质量控制测试
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _load_train_from_selected_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "training" / "train_from_selected.py"
    spec = importlib.util.spec_from_file_location("train_from_selected_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_train_from_selected_module()


class _FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, dataset, segment="test"):
        return dataset.predictions[(self.model_name, segment)]


class TestTrainFromSelectedRuntime(unittest.TestCase):
    def test_extract_label_series_supports_multiindex_columns(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-02"), "000001.sz"),
                (pd.Timestamp("2025-01-02"), "000002.sz"),
            ],
            names=["datetime", "instrument"],
        )
        columns = pd.MultiIndex.from_tuples(
            [
                ("feature", "f1"),
                ("label", "LABEL_5D"),
            ]
        )
        frame = pd.DataFrame([[1.0, 0.1], [2.0, 0.2]], index=index, columns=columns)

        result = MODULE.extract_label_series(frame)

        expected = pd.Series([0.1, 0.2], index=index, name="LABEL_5D")
        pd.testing.assert_series_equal(result, expected)

    def test_collect_model_diagnostics_builds_observable_metrics(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-02"), "000001.sz"),
                (pd.Timestamp("2025-01-02"), "000002.sz"),
                (pd.Timestamp("2025-01-02"), "000003.sz"),
                (pd.Timestamp("2025-01-03"), "000001.sz"),
                (pd.Timestamp("2025-01-03"), "000002.sz"),
                (pd.Timestamp("2025-01-03"), "000003.sz"),
                (pd.Timestamp("2025-01-06"), "000001.sz"),
                (pd.Timestamp("2025-01-06"), "000002.sz"),
                (pd.Timestamp("2025-01-06"), "000003.sz"),
            ],
            names=["datetime", "instrument"],
        )
        train_label = pd.Series([0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5], index=index, name="LABEL_5D")
        valid_label = pd.Series([0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.4, 0.6, 0.8], index=index, name="LABEL_5D")
        dataset = SimpleNamespace(
            predictions={
                ("lgb", "train"): pd.Series([0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5], index=index),
                ("lgb", "valid"): pd.Series([0.19, 0.41, 0.58, 0.31, 0.49, 0.71, 0.42, 0.61, 0.79], index=index),
                ("xgb", "train"): pd.Series([0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3], index=index),
                ("xgb", "valid"): pd.Series([0.6, 0.4, 0.2, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4], index=index),
            }
        )

        diagnostics = MODULE.collect_model_diagnostics(
            models=[_FakeModel("lgb"), _FakeModel("xgb")],
            model_names=["lgb", "xgb"],
            dataset=dataset,
            train_label=train_label,
            valid_label=valid_label,
            model_ic_history={},
            min_ic_samples=2,
        )
        self.assertEqual(len(diagnostics), 2)
        self.assertIsNotNone(diagnostics[0]["valid_rmse"])
        self.assertIsNotNone(diagnostics[1]["valid_rmse"])
        self.assertGreaterEqual(diagnostics[0]["raw_weight"], 0.0)
        self.assertGreaterEqual(diagnostics[1]["raw_weight"], 0.0)

    def test_resolve_model_weights_uses_non_equal_weight_when_available(self):
        diagnostics = [
            {"model_name": "lgb", "raw_weight": 0.6},
            {"model_name": "xgb", "raw_weight": 0.3},
            {"model_name": "cat", "raw_weight": 0.1},
        ]

        weights, used_equal_weight = MODULE.resolve_model_weights(diagnostics)

        self.assertFalse(used_equal_weight)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_assess_window_quality_rejects_abnormal_window(self):
        diagnostics = [
            {
                "model_name": "lgb",
                "train_n": 100,
                "train_rmse": 8.0,
                "valid_n": 100,
                "valid_rmse": 12.0,
            },
            {
                "model_name": "xgb",
                "train_n": 100,
                "train_rmse": 9.0,
                "valid_n": 100,
                "valid_rmse": 15.0,
            },
            {
                "model_name": "cat",
                "train_n": 100,
                "train_rmse": 10.0,
                "valid_n": 100,
                "valid_rmse": 11.0,
            },
        ]

        is_qualified, reasons = MODULE.assess_window_quality("Test_2025", diagnostics, MODULE.LOCAL_CONFIG)

        self.assertFalse(is_qualified)
        self.assertTrue(any("健康模型数不足" in reason for reason in reasons))

    def test_assess_window_quality_accepts_window_with_enough_healthy_models(self):
        diagnostics = [
            {
                "model_name": "lgb",
                "train_n": 100,
                "train_rmse": 0.08,
                "valid_n": 100,
                "valid_rmse": 0.09,
            },
            {
                "model_name": "xgb",
                "train_n": 100,
                "train_rmse": 0.10,
                "valid_n": 100,
                "valid_rmse": 0.12,
            },
            {
                "model_name": "cat",
                "train_n": 100,
                "train_rmse": 7.5,
                "valid_n": 100,
                "valid_rmse": 9.5,
            },
        ]

        is_qualified, reasons = MODULE.assess_window_quality("Test_2024", diagnostics, MODULE.LOCAL_CONFIG)

        self.assertTrue(is_qualified)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
