"""
dataset 模块测试
"""
import unittest
import pandas as pd

from qlworks.features.dataset import (
    CustomFeatureCache,
    PreparedDatasetView,
    _build_processors,
    _build_static_warehouse_frame,
    _slice_feature_cache,
    wrap_dataset_with_cached_train_frame,
)


class TestDataset(unittest.TestCase):
    def test_build_static_warehouse_frame_filters_date_and_instruments(self):
        idx = pd.MultiIndex.from_tuples(
            [
                ("000001.SZ", pd.Timestamp("2020-01-01")),
                ("000001.SZ", pd.Timestamp("2020-01-02")),
                ("000002.SZ", pd.Timestamp("2020-01-02")),
                ("000003.SZ", pd.Timestamp("2020-01-03")),
            ],
            names=["instrument", "datetime"],
        )
        loaded_factors = {
            "f1": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx),
            "f2": pd.Series([10.0, 20.0, 30.0, 40.0], index=idx),
        }

        result = _build_static_warehouse_frame(
            loaded_factors,
            start_time="2020-01-02",
            end_time="2020-01-02",
            instruments=["000001.SZ", "000002.SZ"],
        )

        expected_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "000001.SZ"),
                (pd.Timestamp("2020-01-02"), "000002.SZ"),
            ],
            names=["datetime", "instrument"],
        )
        expected_columns = pd.MultiIndex.from_tuples(
            [("feature", "f1"), ("feature", "f2")]
        )

        self.assertTrue(result.index.equals(expected_index))
        self.assertTrue(result.columns.equals(expected_columns))
        self.assertEqual(result.loc[(pd.Timestamp("2020-01-02"), "000001.SZ"), ("feature", "f1")], 2.0)
        self.assertEqual(result.loc[(pd.Timestamp("2020-01-02"), "000002.SZ"), ("feature", "f2")], 30.0)

    def test_slice_feature_cache_filters_columns_and_date_range(self):
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-01"), "000001.SZ"),
                (pd.Timestamp("2020-01-02"), "000001.SZ"),
                (pd.Timestamp("2020-01-02"), "000002.SZ"),
                (pd.Timestamp("2020-01-03"), "000002.SZ"),
            ],
            names=["datetime", "instrument"],
        )
        warehouse_df = pd.DataFrame(
            {
                ("feature", "f1"): [1.0, 2.0, 3.0, 4.0],
                ("feature", "f2"): [10.0, 20.0, 30.0, 40.0],
            },
            index=idx,
        )
        cache = CustomFeatureCache(
            warehouse_df=warehouse_df,
            qlib_feature_expr_map={"f3": "Ref($close, 1)"},
            label_exprs=["LABEL_EXPR"],
            label_names=["LABEL0"],
            freq="day",
            feature_order=["f1", "f2", "f3"],
        )

        sliced_df, exprs, names = _slice_feature_cache(
            cache,
            selected_feature_names=["f2", "f3"],
            start_time="2020-01-02",
            end_time="2020-01-02",
        )

        expected_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "000001.SZ"),
                (pd.Timestamp("2020-01-02"), "000002.SZ"),
            ],
            names=["datetime", "instrument"],
        )
        self.assertTrue(sliced_df.index.equals(expected_index))
        self.assertEqual(sliced_df.columns.tolist(), [("feature", "f2")])
        self.assertEqual(exprs, ["Ref($close, 1)"])
        self.assertEqual(names, ["f3"])

    def test_wrap_dataset_with_cached_train_frame_reuses_train_prepare(self):
        class FakeDataset:
            def __init__(self):
                self.calls = []

            def prepare(self, segment, col_set=None, data_key=None, **kwargs):
                self.calls.append((segment, col_set, data_key, kwargs))
                return f"base:{segment}:{col_set}:{data_key}"

        base_dataset = FakeDataset()
        train_frame = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "f2": [10.0, 20.0],
                "LABEL_5D": [0.1, 0.2],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2020-01-02"), "000001.SZ"),
                    (pd.Timestamp("2020-01-02"), "000002.SZ"),
                ],
                names=["datetime", "instrument"],
            ),
        )

        wrapped = wrap_dataset_with_cached_train_frame(
            base_dataset,
            train_frame=train_frame,
            selected_feature_names=["f2"],
            label_names=["LABEL_5D"],
            learn_data_key="L",
            infer_data_key="I",
        )

        cached_train = wrapped.prepare("train")
        self.assertEqual(cached_train.columns.tolist(), ["f2", "LABEL_5D"])

        train_valid = wrapped.prepare(["train", "valid"], col_set=["feature", "label"], data_key="L")
        self.assertEqual(train_valid[0].columns.tolist(), [("feature", "f2"), ("label", "LABEL_5D")])
        self.assertEqual(train_valid[1], "base:valid:['feature', 'label']:L")
        self.assertEqual(len(base_dataset.calls), 1)

    def test_wrap_dataset_with_cached_train_frame_can_reuse_valid_prepare(self):
        class FakeDataset:
            def __init__(self):
                self.calls = []

            def prepare(self, segment, col_set=None, data_key=None, **kwargs):
                self.calls.append((segment, col_set, data_key, kwargs))
                return f"base:{segment}:{col_set}:{data_key}"

        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "000001.SZ"),
                (pd.Timestamp("2020-01-02"), "000002.SZ"),
            ],
            names=["datetime", "instrument"],
        )
        train_frame = pd.DataFrame({"f1": [1.0, 2.0], "LABEL_5D": [0.1, 0.2]}, index=idx)
        valid_frame = pd.DataFrame({"f1": [3.0, 4.0], "LABEL_5D": [0.3, 0.4]}, index=idx)

        wrapped = wrap_dataset_with_cached_train_frame(
            FakeDataset(),
            train_frame=train_frame,
            selected_feature_names=["f1"],
            label_names=["LABEL_5D"],
            learn_data_key="L",
            infer_data_key="I",
            valid_frame=valid_frame,
        )

        train_valid = wrapped.prepare(["train", "valid"], col_set=["feature", "label"], data_key="L")
        self.assertEqual(train_valid[0].columns.tolist(), [("feature", "f1"), ("label", "LABEL_5D")])
        self.assertEqual(train_valid[1].columns.tolist(), [("feature", "f1"), ("label", "LABEL_5D")])
        self.assertEqual(wrapped._base_dataset.calls, [])

    def test_prepared_dataset_view_list_prepare_uses_cache_and_delegates_rest(self):
        class FakeDataset:
            def __init__(self):
                self.calls = []

            def prepare(self, segment, col_set=None, data_key=None, **kwargs):
                self.calls.append((segment, col_set, data_key, kwargs))
                return f"base:{segment}:{col_set}:{data_key}"

        wrapped = PreparedDatasetView(
            FakeDataset(),
            cached_prepare_results={
                ("train", None, None): "cached-train",
                ("train", ("feature", "label"), "L"): "cached-train-fl",
            },
        )

        self.assertEqual(wrapped.prepare("train"), "cached-train")
        result = wrapped.prepare(["train", "valid"], col_set=["feature", "label"], data_key="L")
        self.assertEqual(result, ["cached-train-fl", "base:valid:['feature', 'label']:L"])
        self.assertEqual(wrapped._base_dataset.calls, [("valid", ["feature", "label"], "L", {})])

    def test_build_processors_separates_label_normalize_and_neutralize(self):
        infer_processors, learn_processors = _build_processors(
            model_type="tree",
            normalize_labels=True,
            neutralize_labels=False,
        )
        self.assertEqual(
            [p["class"] for p in infer_processors],
            ["CSQuantileNorm", "Fillna"],
        )
        self.assertEqual(
            [p["class"] for p in learn_processors],
            ["DropnaLabel", "CSQuantileNorm", "CSQuantileNorm", "Fillna"],
        )
        self.assertEqual(learn_processors[1]["kwargs"]["fields_group"], "label")
        self.assertEqual(learn_processors[2]["kwargs"]["fields_group"], "feature")

    def test_build_processors_can_enable_label_neutralize_without_label_quantile(self):
        _, learn_processors = _build_processors(
            model_type="tree",
            normalize_features=True,
            normalize_labels=False,
            neutralize_labels=True,
        )
        self.assertEqual(
            [p["class"] for p in learn_processors],
            ["DropnaLabel", "CSNeutralize", "CSQuantileNorm", "Fillna"],
        )
        self.assertEqual(learn_processors[1]["kwargs"]["fields_group"], "label")

    def test_build_processors_requires_feature_normalize_for_tree_model(self):
        with self.assertRaisesRegex(ValueError, "tree.*normalize_features=True"):
            _build_processors(
                model_type="tree",
                normalize_features=False,
            )

    def test_build_processors_uses_model_specific_feature_normalizer(self):
        infer_tree, learn_tree = _build_processors(
            model_type="tree",
            normalize_features=True,
        )
        self.assertEqual(infer_tree[0]["class"], "CSQuantileNorm")
        self.assertEqual(learn_tree[1]["class"], "CSQuantileNorm")

        infer_linear, learn_linear = _build_processors(
            model_type="linear",
            normalize_features=True,
        )
        self.assertEqual(infer_linear[0]["class"], "RobustZScoreNorm")
        self.assertEqual(learn_linear[1]["class"], "RobustZScoreNorm")

    def test_build_processors_linear_renormalizes_after_feature_neutralize(self):
        infer_processors, learn_processors = _build_processors(
            model_type="linear",
            normalize_features=True,
            neutralize_features=True,
            renormalize_features_after_neutralize=True,
        )
        self.assertEqual(
            [p["class"] for p in infer_processors],
            ["RobustZScoreNorm", "Fillna", "CSNeutralize", "RobustZScoreNorm", "Fillna"],
        )
        self.assertEqual(
            [p["class"] for p in learn_processors],
            ["DropnaLabel", "RobustZScoreNorm", "Fillna", "CSNeutralize", "RobustZScoreNorm", "Fillna"],
        )

    def test_build_processors_tree_does_not_renormalize_after_feature_neutralize_by_default(self):
        infer_processors, learn_processors = _build_processors(
            model_type="tree",
            normalize_features=True,
            neutralize_features=True,
        )
        self.assertEqual(
            [p["class"] for p in infer_processors],
            ["CSQuantileNorm", "Fillna", "CSNeutralize"],
        )
        self.assertEqual(
            [p["class"] for p in learn_processors],
            ["DropnaLabel", "CSQuantileNorm", "Fillna", "CSNeutralize"],
        )

    def test_build_processors_tree_can_renormalize_after_feature_neutralize_when_enabled(self):
        infer_processors, learn_processors = _build_processors(
            model_type="tree",
            normalize_features=True,
            neutralize_features=True,
            renormalize_features_after_neutralize=True,
        )
        self.assertEqual(
            [p["class"] for p in infer_processors],
            ["CSQuantileNorm", "Fillna", "CSNeutralize", "CSQuantileNorm", "Fillna"],
        )
        self.assertEqual(
            [p["class"] for p in learn_processors],
            ["DropnaLabel", "CSQuantileNorm", "Fillna", "CSNeutralize", "CSQuantileNorm", "Fillna"],
        )


if __name__ == "__main__":
    unittest.main()
