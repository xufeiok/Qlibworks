"""
quantile_norm 处理器测试
"""
import unittest
from unittest.mock import patch

import pandas as pd

import qlworks.processors.quantile_norm as quantile_norm
from qlworks.processors.quantile_norm import CSQuantileNorm, QuantileNormProcessor


class TestCSQuantileNorm(unittest.TestCase):
    def test_cs_quantile_norm_ranks_each_datetime_cross_section(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-02"), "000001.sz"),
                (pd.Timestamp("2025-01-02"), "000002.sz"),
                (pd.Timestamp("2025-01-02"), "600000.sh"),
                (pd.Timestamp("2025-01-03"), "000001.sz"),
                (pd.Timestamp("2025-01-03"), "000002.sz"),
                (pd.Timestamp("2025-01-03"), "600000.sh"),
            ],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame(
            {
                "f1": [3.0, 1.0, 2.0, 1.0, 1.0, None],
                "f2": [30.0, 10.0, 20.0, None, 5.0, 15.0],
            },
            index=index,
        )

        result = CSQuantileNorm(eps=1e-6).transform(df)
        expected = (
            df.groupby(level="datetime", group_keys=False)
            .rank(pct=True, method="average", na_option="keep")
            .clip(1e-6, 1 - 1e-6)
        )

        pd.testing.assert_frame_equal(result, expected.astype("float32"))

    def test_cs_quantile_norm_only_updates_requested_fields_group(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-02"), "000001.sz"),
                (pd.Timestamp("2025-01-02"), "000002.sz"),
                (pd.Timestamp("2025-01-03"), "000001.sz"),
                (pd.Timestamp("2025-01-03"), "000002.sz"),
            ],
            names=["datetime", "instrument"],
        )
        columns = pd.MultiIndex.from_tuples(
            [
                ("feature", "f1"),
                ("feature", "f2"),
                ("label", "LABEL_5D"),
            ]
        )
        df = pd.DataFrame(
            [
                [2.0, 10.0, 0.1],
                [1.0, 20.0, 0.2],
                [5.0, 50.0, 0.3],
                [4.0, 40.0, 0.4],
            ],
            index=index,
            columns=columns,
        )

        result = CSQuantileNorm(fields_group="feature").transform(df)
        expected_feature = (
            df["feature"]
            .groupby(level="datetime", group_keys=False)
            .rank(pct=True, method="average", na_option="keep")
            .clip(1e-6, 1 - 1e-6)
        )

        pd.testing.assert_frame_equal(result["feature"], expected_feature.astype("float32"))
        pd.testing.assert_frame_equal(result["label"], df["label"])

    def test_cs_quantile_norm_requires_datetime_multi_index(self):
        df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})

        with self.assertRaises(ValueError):
            CSQuantileNorm().transform(df)

    def test_cs_quantile_norm_chunked_path_keeps_same_values(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-02"), "000001.sz"),
                (pd.Timestamp("2025-01-02"), "000002.sz"),
                (pd.Timestamp("2025-01-03"), "000001.sz"),
                (pd.Timestamp("2025-01-03"), "000002.sz"),
                (pd.Timestamp("2025-01-04"), "000001.sz"),
                (pd.Timestamp("2025-01-04"), "000002.sz"),
            ],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame(
            {
                "f1": [2.0, 1.0, 5.0, 4.0, 7.0, 6.0],
                "f2": [1.0, 2.0, 4.0, 3.0, 6.0, 7.0],
            },
            index=index,
        )

        expected = (
            df.groupby(level="datetime", group_keys=False)
            .rank(pct=True, method="average", na_option="keep")
            .clip(1e-6, 1 - 1e-6)
            .astype("float32")
        )

        with patch.object(quantile_norm, "CS_QUANTILE_DATE_CHUNK_SIZE", 1):
            result = CSQuantileNorm(eps=1e-6).transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestQuantileNormProcessor(unittest.TestCase):
    def test_quantile_norm_processor_preserves_axis_zero_semantics(self):
        df = pd.DataFrame(
            [[3.0, 1.0, 2.0], [30.0, 10.0, 20.0]],
            columns=["f1", "f2", "f3"],
        )

        result = QuantileNormProcessor(axis=0, eps=1e-6).transform(df)
        expected = df.rank(axis=1, pct=True, method="average", na_option="keep").clip(1e-6, 1 - 1e-6)

        pd.testing.assert_frame_equal(result, expected.astype("float32"))

    def test_quantile_norm_processor_preserves_axis_one_semantics(self):
        df = pd.DataFrame(
            [[3.0, 1.0, 2.0], [30.0, 10.0, 20.0]],
            columns=["f1", "f2", "f3"],
        )

        result = QuantileNormProcessor(axis=1, eps=1e-6).transform(df)
        expected = df.rank(axis=0, pct=True, method="average", na_option="keep").clip(1e-6, 1 - 1e-6)

        pd.testing.assert_frame_equal(result, expected.astype("float32"))


if __name__ == "__main__":
    unittest.main()
