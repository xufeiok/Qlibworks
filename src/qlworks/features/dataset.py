from __future__ import annotations

import os
import sys
from dataclasses import dataclass
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, List, Optional, Type


from qlworks.features.builder import FeatureBundle


# DuckDB + Parquet 预计算因子到 Qlib 表达式的映射
# 训练时直接用 Qlib 引擎从 .bin 数据求值，和 Parquet 结果一致
FACTOR_CACHE_EXPRESSIONS = {
    "ret_1d": "$close / Ref($close, 1) - 1",
    "ma_5": "Mean($close, 5)",
    "price_position_20": "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))",
}


@dataclass
class CustomFeatureCache:
    """缓存底层特征表，供同一窗口内重复切列/切时间范围复用。"""

    warehouse_df: pd.DataFrame
    qlib_feature_expr_map: Dict[str, str]
    label_exprs: List[str]
    label_names: List[str]
    freq: str
    feature_order: List[str]
    resolved_instruments: Optional[List[str]] = None


class PreparedDatasetView:
    """包装 Qlib Dataset，对已知 prepare 结果优先走缓存。"""

    def __init__(self, base_dataset, cached_prepare_results: Optional[Dict[tuple, object]] = None):
        self._base_dataset = base_dataset
        self._cached_prepare_results = cached_prepare_results or {}

    @staticmethod
    def _normalize_col_set(col_set):
        if isinstance(col_set, list):
            return tuple(col_set)
        return col_set

    def _make_key(self, segment, col_set=None, data_key=None):
        return (segment, self._normalize_col_set(col_set), data_key)

    def prepare(self, segment, col_set=None, data_key=None, **kwargs):
        if isinstance(segment, (list, tuple)):
            return [self.prepare(seg, col_set=col_set, data_key=data_key, **kwargs) for seg in segment]

        if not kwargs:
            cache_key = self._make_key(segment, col_set=col_set, data_key=data_key)
            if cache_key in self._cached_prepare_results:
                return self._cached_prepare_results[cache_key]
        return self._base_dataset.prepare(segment, col_set=col_set, data_key=data_key, **kwargs)

    def __getattr__(self, item):
        return getattr(self._base_dataset, item)


def wrap_dataset_with_cached_train_frame(
    dataset,
    train_frame: pd.DataFrame,
    selected_feature_names: List[str],
    label_names: List[str],
    learn_data_key=None,
    infer_data_key=None,
    valid_frame: Optional[pd.DataFrame] = None,
):
    """复用已准备好的训练段结果，避免同窗口二次训练 prepare 再跑一遍 processor。"""
    def _build_cached_results(frame: pd.DataFrame):
        feature_cols = [name for name in selected_feature_names if name in frame.columns]
        label_cols = [name for name in label_names if name in frame.columns]
        selected_cols = feature_cols + label_cols
        cached_frame = frame.loc[:, selected_cols].copy()
        grouped_parts = {}
        if feature_cols:
            grouped_parts["feature"] = cached_frame.loc[:, feature_cols].copy()
        if label_cols:
            grouped_parts["label"] = cached_frame.loc[:, label_cols].copy()
        grouped_frame = pd.concat(grouped_parts, axis=1) if grouped_parts else cached_frame
        return cached_frame, grouped_frame, feature_cols

    cached_train, grouped_train, train_feature_cols = _build_cached_results(train_frame)
    cached_results = {
        ("train", None, None): cached_train,
    }

    if train_feature_cols:
        if learn_data_key is not None:
            cached_results[("train", ("feature", "label"), learn_data_key)] = grouped_train
            cached_results[("train", "feature", learn_data_key)] = grouped_train["feature"]
        if infer_data_key is not None:
            cached_results[("train", "feature", infer_data_key)] = grouped_train["feature"]

    if valid_frame is not None:
        cached_valid, grouped_valid, valid_feature_cols = _build_cached_results(valid_frame)
        cached_results[("valid", None, None)] = cached_valid
        if valid_feature_cols and learn_data_key is not None:
            cached_results[("valid", ("feature", "label"), learn_data_key)] = grouped_valid
            cached_results[("valid", "feature", learn_data_key)] = grouped_valid["feature"]
        if valid_feature_cols and infer_data_key is not None:
            cached_results[("valid", "feature", infer_data_key)] = grouped_valid["feature"]

    return PreparedDatasetView(dataset, cached_prepare_results=cached_results)


def _load_factors_from_warehouse(feature_bundle, start_time, end_time):
    """
    尝试从 warehouse 加载因子数据。

    返回：
    - loaded_factors: {因子名: MultiIndex Series}
    - remaining_factors: warehouse 缺失的因子名列表
    """
    from pathlib import Path
    from qlworks.config import WAREHOUSE_DIR
    
    warehouse_base = Path(WAREHOUSE_DIR)
    loaded_factors = {}
    remaining_factors = []
    
    # 遍历所有需要加载的因子（从 names 获取因子名）
    for field_name in feature_bundle.names:
        factor_dir = warehouse_base / field_name
        
        # 检查 warehouse 中是否有该因子的 parquet 文件
        if factor_dir.exists():
            parquet_files = sorted(factor_dir.glob("*.parquet"))
            if parquet_files:
                # 有 parquet 文件，尝试加载
                dfs = []
                for file in parquet_files:
                    # 根据文件名判断年份是否在时间范围内
                    try:
                        file_year = int(file.stem)
                        if start_time and file_year < int(start_time[:4]):
                            continue
                        if end_time and file_year > int(end_time[:4]):
                            continue
                    except ValueError:
                        pass  # 非年份文件名，直接加载
                    
                    try:
                        df = pd.read_parquet(file)
                        if {"instrument", "datetime"}.issubset(df.index.names):
                            df = df.reset_index()
                        dfs.append(df)
                    except Exception as e:
                        print(f"    [警告] 读取 {file} 失败：{e}")
                
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    if "datetime" not in df.columns and {"instrument", "datetime"}.issubset(df.index.names):
                        df = df.reset_index()
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    else:
                        print(f"    [警告] {field_name} 的 warehouse 数据缺少 datetime 字段，回退到 Qlib 表达式")
                        remaining_factors.append(field_name)
                        continue
                    if start_time:
                        df = df[df['datetime'] >= pd.Timestamp(start_time)]
                    if end_time:
                        df = df[df['datetime'] <= pd.Timestamp(end_time)]
                    value_col = None
                    if 'value' in df.columns:
                        value_col = 'value'
                    elif field_name in df.columns:
                        value_col = field_name
                    elif {'instrument', 'datetime'}.issubset(df.columns):
                        candidate_cols = [col for col in df.columns if col not in {'instrument', 'datetime'}]
                        if len(candidate_cols) == 1:
                            value_col = candidate_cols[0]
                    if not df.empty and value_col is not None and {'instrument', 'datetime'}.issubset(df.columns):
                        dup_mask = df.duplicated(subset=['instrument', 'datetime'], keep='last')
                        dup_count = int(dup_mask.sum())
                        if dup_count:
                            dup_keys = int(df.loc[dup_mask, ['instrument', 'datetime']].drop_duplicates().shape[0])
                            print(f"    [warehouse 去重] {field_name}: 删除 {dup_count:,} 行重复记录，涉及 {dup_keys:,} 个键")
                            df = df.loc[~dup_mask].copy()
                        series = df.set_index(['instrument', 'datetime'])[value_col].sort_index()
                        series.index.names = ['instrument', 'datetime']
                        loaded_factors[field_name] = series
                        print(f"    [warehouse 直载] {field_name}: {len(series):,} 条记录")
                        continue
        
        # warehouse 中没有，需要后续用表达式构建
        remaining_factors.append(field_name)
    
    return loaded_factors, remaining_factors


def _resolve_static_instruments(instruments, start_time=None, end_time=None, verbose=True):
    from qlib.data import D

    if instruments is None:
        return None
    if isinstance(instruments, str):
        inst_conf = D.instruments(instruments)
        resolved = D.list_instruments(inst_conf, start_time=start_time, end_time=end_time, as_list=True)
        if verbose:
            print(f"    [静态池过滤] {instruments}: {len(resolved):,} 只股票")
        return resolved
    if isinstance(instruments, dict) and "market" in instruments:
        resolved = D.list_instruments(instruments, start_time=start_time, end_time=end_time, as_list=True)
        market = instruments.get("market", "custom")
        if verbose:
            print(f"    [静态池过滤] {market}: {len(resolved):,} 只股票")
        return resolved
    return instruments


def _build_static_warehouse_frame(loaded_factors, start_time=None, end_time=None, instruments=None):
    """在拼接前先裁剪 warehouse 静态表，减少不必要的索引对齐与内存占用。"""
    if not loaded_factors:
        return pd.DataFrame()

    filtered_series = []
    resolved_instruments = instruments
    if resolved_instruments is not None:
        resolved_instruments = set(resolved_instruments)

    start_ts = pd.Timestamp(start_time) if start_time else None
    end_ts = pd.Timestamp(end_time) if end_time else None

    for name, series in loaded_factors.items():
        frame = series.rename(name).reset_index()
        if start_ts is not None:
            frame = frame[frame["datetime"] >= start_ts]
        if end_ts is not None:
            frame = frame[frame["datetime"] <= end_ts]
        if resolved_instruments is not None:
            frame = frame[frame["instrument"].isin(resolved_instruments)]
        if frame.empty:
            continue
        filtered_series.append(frame.set_index(["instrument", "datetime"])[name].rename(name))

    if not filtered_series:
        return pd.DataFrame()

    warehouse_df = pd.concat(filtered_series, axis=1, sort=True)
    warehouse_df.index.names = ["instrument", "datetime"]
    warehouse_df = warehouse_df.swaplevel("instrument", "datetime").sort_index()
    warehouse_df.index.names = ["datetime", "instrument"]
    warehouse_df.columns = pd.MultiIndex.from_product([["feature"], warehouse_df.columns])
    return warehouse_df


def _slice_feature_cache(
    feature_cache: CustomFeatureCache,
    selected_feature_names: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """从缓存中按因子和时间范围裁剪特征。"""
    selected_names = list(selected_feature_names or feature_cache.feature_order)
    missing_names = [name for name in selected_names if name not in feature_cache.feature_order]
    if missing_names:
        raise ValueError(f"缓存中不存在这些因子: {missing_names}")

    warehouse_df = feature_cache.warehouse_df
    if not warehouse_df.empty:
        if start_time or end_time:
            start_ts = pd.Timestamp(start_time) if start_time else None
            end_ts = pd.Timestamp(end_time) if end_time else None
            warehouse_df = warehouse_df.loc[start_ts:end_ts]

        warehouse_names = [name for name in selected_names if ("feature", name) in warehouse_df.columns]
        if warehouse_names:
            warehouse_df = warehouse_df.loc[:, pd.IndexSlice["feature", warehouse_names]]
        else:
            warehouse_df = warehouse_df.iloc[0:0, 0:0]

    qlib_feature_names = [name for name in selected_names if name in feature_cache.qlib_feature_expr_map]
    qlib_feature_exprs = [feature_cache.qlib_feature_expr_map[name] for name in qlib_feature_names]
    return warehouse_df, qlib_feature_exprs, qlib_feature_names


def build_custom_feature_cache(
    instruments: str | list = "csi300",
    feature_bundle: Optional[FeatureBundle] = None,
    custom_features: Optional[Dict[str, str]] = None,
    custom_labels: Optional[Dict[str, str]] = None,
    factor_cache_names: Optional[List[str]] = None,
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    freq: str = "day",
) -> CustomFeatureCache:
    """构建窗口级底层特征缓存，供多次重组数据集复用。"""
    if feature_bundle is not None:
        all_feature_exprs = list(feature_bundle.fields)
        all_feature_names = list(feature_bundle.names)
        label_exprs = list(feature_bundle.label_fields)
        label_names = list(feature_bundle.label_names)
    elif custom_features is not None and custom_labels is not None:
        all_feature_exprs = list(custom_features.values())
        all_feature_names = list(custom_features.keys())
        label_exprs = list(custom_labels.values())
        label_names = list(custom_labels.keys())
    else:
        raise ValueError("必须提供 feature_bundle，或者同时提供 custom_features 和 custom_labels。")

    loaded_factors = {}
    qlib_feature_expr_map = {}
    feature_order = []

    if feature_bundle is not None:
        loaded_factors, _ = _load_factors_from_warehouse(feature_bundle, start_time, end_time)
        expr_map = dict(zip(feature_bundle.names, feature_bundle.fields))
        for field_name in feature_bundle.names:
            feature_order.append(field_name)
            if field_name in loaded_factors:
                print(f"    [使用 warehouse] {field_name}")
                continue
            expr = expr_map.get(field_name)
            if expr is None:
                print(f"    [警告] 因子 {field_name} 缺少表达式定义")
                continue
            qlib_feature_expr_map[field_name] = expr
            print(f"    [Qlib 回退] {field_name}: {expr}")
    else:
        feature_order = list(all_feature_names)
        qlib_feature_expr_map = dict(zip(all_feature_names, all_feature_exprs))

    if factor_cache_names:
        for name in factor_cache_names:
            if name in feature_order:
                print(f"    [跳过] {name} 已在 feature_bundle 中")
                continue
            expr = FACTOR_CACHE_EXPRESSIONS.get(name)
            if expr is None:
                raise ValueError(f"未知因子 `{name}`，可用：{list(FACTOR_CACHE_EXPRESSIONS.keys())}")
            feature_order.append(name)
            qlib_feature_expr_map[name] = expr
            print(f"    [因子注入] {name}: {expr}")

    resolved_instruments = _resolve_static_instruments(
        instruments,
        start_time=start_time,
        end_time=end_time,
        verbose=True,
    )
    warehouse_df = _build_static_warehouse_frame(
        loaded_factors,
        start_time=start_time,
        end_time=end_time,
        instruments=resolved_instruments,
    )

    return CustomFeatureCache(
        warehouse_df=warehouse_df,
        qlib_feature_expr_map=qlib_feature_expr_map,
        label_exprs=label_exprs,
        label_names=label_names,
        freq=freq,
        feature_order=feature_order,
        resolved_instruments=resolved_instruments,
    )


class InstrumentAwareStaticDataLoader:
    """兼容 Qlib 股票池别名的静态加载器，避免 warehouse 分支退回全量扫描。"""

    def __init__(self, config, join="outer", default_instruments=None):
        from qlib.data.dataset.loader import StaticDataLoader

        self._loader = StaticDataLoader(config=config, join=join)
        self._default_instruments = default_instruments

    def load(self, instruments=None, start_time=None, end_time=None):
        if self._default_instruments is not None and (
            isinstance(instruments, str) or (isinstance(instruments, dict) and "market" in instruments)
        ):
            resolved_instruments = self._default_instruments
        else:
            resolved_instruments = _resolve_static_instruments(instruments, start_time=start_time, end_time=end_time)
        if resolved_instruments is not None:
            self._loader._maybe_load_raw_data()
            available_instruments = set(self._loader._data.index.get_level_values("instrument").unique())
            resolved_instruments = [inst for inst in resolved_instruments if inst in available_instruments]
        return self._loader.load(resolved_instruments, start_time=start_time, end_time=end_time)


def _build_mixed_loader_config(
    feature_cache: CustomFeatureCache,
    selected_feature_names,
    feature_exprs,
    feature_names,
    label_exprs,
    label_names,
    freq,
    start_time=None,
    end_time=None,
):
    """构造 warehouse 静态特征 + Qlib 回退特征/标签 的混合加载器配置。"""
    warehouse_df, cached_feature_exprs, cached_feature_names = _slice_feature_cache(
        feature_cache,
        selected_feature_names=selected_feature_names,
        start_time=start_time,
        end_time=end_time,
    )
    if not feature_exprs and not feature_names:
        feature_exprs = cached_feature_exprs
        feature_names = cached_feature_names

    if warehouse_df.empty:
        return {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": (feature_exprs, feature_names),
                    "label": (label_exprs, label_names),
                },
                "freq": freq,
            },
        }

    loaders = [{
        "class": "InstrumentAwareStaticDataLoader",
        "module_path": "qlworks.features.dataset",
        "kwargs": {"config": warehouse_df, "default_instruments": feature_cache.resolved_instruments},
    }]
    qlib_config = {}
    if feature_exprs:
        qlib_config["feature"] = (feature_exprs, feature_names)
    qlib_config["label"] = (label_exprs, label_names)
    loaders.append({
        "class": "QlibDataLoader",
        "module_path": "qlib.data.dataset.loader",
        "kwargs": {"config": qlib_config, "freq": freq},
    })
    return {
        "class": "NestedDataLoader",
        "module_path": "qlib.data.dataset.loader",
        "kwargs": {"dataloader_l": loaders},
    }


def _build_processors(
    model_type: str = "tree",
    normalize_features: bool = True,
    neutralize_features: bool = False,
    renormalize_features_after_neutralize: Optional[bool] = None,
    normalize_labels: bool = False,
    neutralize_labels: bool = False,
    symmetric_orthogonalization: bool = False,
) -> tuple[list, list]:
    """
    工厂函数：根据模型流派和中性化需求构建 Processor 流水线。

    统一了 create_alpha158_dataset 和 create_custom_dataset 中重复的 Processor 装配逻辑。

    Args:
        model_type: 模型类型 ("tree" / "linear" / "nn")
        normalize_features: 是否对特征进行标准化
        neutralize_features: 是否对特征进行截面中性化
        renormalize_features_after_neutralize: 特征中性化后是否再标准化；
            默认按模型分流：tree=False，linear/nn=True
        normalize_labels: 是否对标签进行截面分位数化（即标准化）
        neutralize_labels: 是否对标签进行截面中性化
        symmetric_orthogonalization: 是否开启对称正交化（仅 linear/nn）

    Returns:
        (infer_processors, learn_processors) 二元组
    """
    model_norm_map = {
        "tree": "CSQuantileNorm(特征横截面分位数化)",
        "linear": "RobustZScoreNorm(特征稳健ZScore标准化)",
        "nn": "RobustZScoreNorm(特征稳健ZScore标准化)",
    }
    if not normalize_features:
        recommended = model_norm_map.get(model_type, "按模型选择合适的特征标准化方式")
        raise ValueError(
            f"当前 model_type='{model_type}' 必须设置 normalize_features=True。"
            f"推荐标准化方式: {recommended}。"
        )
    if renormalize_features_after_neutralize is None:
        renormalize_features_after_neutralize = model_type != "tree"

    base_infer: list = []
    base_learn: list = [{"class": "DropnaLabel"}]

    # 1. 标签处理：先中性化，再做截面分位数化
    if neutralize_labels:
        label_neutralize = {
            "class": "CSNeutralize",
            "module_path": "qlworks.processors.neutralize",
            "kwargs": {"fields_group": "label"},
        }
        base_learn.append(label_neutralize)

    if normalize_labels:
        label_rank = {
            "class": "CSQuantileNorm",
            "module_path": "qlworks.processors.quantile_norm",
            "kwargs": {"fields_group": "label"},
        }
        base_learn.append(label_rank)

    # 2. 特征标准化
    if model_type == "tree":
        norm_cfg = {
            "class": "CSQuantileNorm",
            "module_path": "qlworks.processors.quantile_norm",
            "kwargs": {"fields_group": "feature"},
        }
        base_infer.append(norm_cfg)
        base_learn.append(norm_cfg)
    else:
        norm_cfg = {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}}
        base_infer.append(norm_cfg)
        base_learn.append(norm_cfg)

    # 3. 特征中性化（线性模型推荐开启，树模型不推荐）
    if neutralize_features:
        feat_neutralize = {
            "class": "CSNeutralize",
            "module_path": "qlworks.processors.neutralize",
            "kwargs": {"fields_group": "feature"},
        }
        # 中性化前先补齐缺失，避免回归残差化阶段因 NaN 失稳
        pre_fill_value = 0.5 if model_type == "tree" else 0
        pre_fill = {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": pre_fill_value}}
        base_infer.append(pre_fill)
        base_learn.append(pre_fill)
        base_infer.append(feat_neutralize)
        base_learn.append(feat_neutralize)
        if renormalize_features_after_neutralize:
            base_infer.append(norm_cfg)
            base_learn.append(norm_cfg)
            post_fill_kwargs = {"fields_group": "feature"}
            if model_type == "tree":
                post_fill_kwargs["fill_value"] = 0.5
            elif pre_fill_value is not None:
                post_fill_kwargs["fill_value"] = pre_fill_value
            base_infer.append({"class": "Fillna", "kwargs": post_fill_kwargs})
            base_learn.append({"class": "Fillna", "kwargs": post_fill_kwargs})
    else:
        # 非中性化流派：先 Fillna 收尾
        fill_value = 0.5 if model_type == "tree" else None
        fill_kwargs = {"fields_group": "feature"}
        if fill_value is not None:
            fill_kwargs["fill_value"] = fill_value
        base_infer.append({"class": "Fillna", "kwargs": fill_kwargs})
        base_learn.append({"class": "Fillna", "kwargs": fill_kwargs})

    # 4. 对称正交化（仅 linear/nn，消除共线性）
    if symmetric_orthogonalization and model_type != "tree":
        feat_ortho = {
            "class": "CSSymmetricOrthogonalize",
            "module_path": "qlworks.processors.orthogonalize",
            "kwargs": {"fields_group": "feature"},
        }
        base_infer.append(feat_ortho)
        base_learn.append(feat_ortho)

    return base_infer, base_learn


def create_dataset_from_handler(handler, segments):
    """
    功能概述：
    - 从 Handler 构建 DatasetH，并支持动态切换 segment。
    输入：
    - handler: Qlib DataHandler 对象。
    - segments: 数据切分配置，如 {"train": ("2020-01-01", "2020-06-30"), ...}。
    输出：
    - DatasetH 对象。
    """
    from qlib.data.dataset import DatasetH

    dataset = DatasetH(handler, segments)
    return dataset


def create_alpha158_dataset(
    instruments: str | list = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2020-06-30",
    freq: str = "day",
    infer_processors: Optional[list] = None,
    learn_processors: Optional[list] = None,
    segments: Optional[Dict[str, tuple]] = None,
    model_type: str = "tree",
    normalize_features: bool = True,
    normalize_labels: bool = False,
    neutralize_labels: bool = False,
    neutralize_features: bool = False,
    renormalize_features_after_neutralize: Optional[bool] = None,
    **kwargs,
):
    """
    功能概述：
    - 构建 Alpha158 标准数据集，适合树模型（如 LightGBM/XGBoost）。
    输入：
    - 股票池、时间区间、切分配置。
    输出：
    - `(handler, dataset)` 二元组。
    边界条件：
    - 如果处理器未指定，自动根据模型类型匹配最佳流水线。
    性能/安全注意事项：
    - 使用 Qlib 内置的 Alpha158 处理器，无需手动配置。
    """
    from qlib.contrib.data.handler import Alpha158

    if infer_processors is None and learn_processors is None:
        infer_processors, learn_processors = _build_processors(
            model_type=model_type,
            normalize_features=normalize_features,
            neutralize_features=neutralize_features,
            renormalize_features_after_neutralize=renormalize_features_after_neutralize,
            normalize_labels=normalize_labels,
            neutralize_labels=neutralize_labels,
            symmetric_orthogonalization=False,
        )
    
    segments = segments or {
        "train": (fit_start_time, fit_end_time),
        "valid": ("2020-07-01", "2020-09-30"),
        "test": ("2020-10-01", end_time),
    }
    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        freq=freq,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )
    dataset = create_dataset_from_handler(handler, segments)
    return handler, dataset


def create_custom_dataset(
    instruments: str | list = "csi300",
    feature_bundle: Optional[FeatureBundle] = None,
    custom_features: Optional[Dict[str, str]] = None,
    custom_labels: Optional[Dict[str, str]] = None,
    feature_cache: Optional[CustomFeatureCache] = None,
    selected_feature_names: Optional[List[str]] = None,
    factor_cache_names: Optional[List[str]] = None,
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2020-06-30",
    freq: str = "day",
    model_type: str = "tree", # ["tree", "linear", "nn"]
    normalize_features: bool = True,  # 是否对特征 X 进行标准化
    neutralize_features: bool = False, # 是否对特征 X 进行中性化
    renormalize_features_after_neutralize: Optional[bool] = None, # 特征中性化后是否再次标准化
    normalize_labels: bool = False,    # 是否对标签 Y 进行分位数化
    neutralize_labels: bool = False,   # 是否对标签 Y 进行中性化
    infer_processors: Optional[list] = None,
    learn_processors: Optional[list] = None,
    segments: Optional[Dict[str, tuple]] = None,
    symmetric_orthogonalization: bool = False,
):
    """
    功能概述：
    - 构建完全自定义的数据集，支持传入 FeatureBundle 或 任意因子表达式字典。
    - 优先从 factor_data/warehouse 加载已有的单因子 parquet 文件
    - warehouse 中没有的因子才使用表达式构建
    
    加载顺序：
    1. warehouse/{因子名}/{年份}.parquet (已有单因子文件)
    2. factor_cache_names 中定义的表达式
    3. feature_bundle 中定义的表达式
    
    输入：
    - feature_bundle: 来自 builder.py 的 FeatureBundle 对象 (推荐)。
    - custom_features: 特征字典，格式如 `{"MA5": "Mean($close, 5)"}` (作为备选)。
    - custom_labels: 标签字典，格式如 `{"LABEL": "Ref($close, -2)/$close - 1"}`。
    输出：
    - `(handler, dataset)` 二元组。
    边界条件：
    - feature_bundle 优先级高于 custom_features/custom_labels。
    性能/安全注意事项：
    - 使用底层的 DataHandlerLP 灵活组装，完全兼容 Qlib 的 Processor 流水线。
    """
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.data.handler import check_transform_proc

    if feature_cache is None:
        feature_cache = build_custom_feature_cache(
            instruments=instruments,
            feature_bundle=feature_bundle,
            custom_features=custom_features,
            custom_labels=custom_labels,
            factor_cache_names=factor_cache_names,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
        )

    data_loader_config = _build_mixed_loader_config(
        feature_cache=feature_cache,
        selected_feature_names=selected_feature_names,
        feature_exprs=[],
        feature_names=[],
        label_exprs=feature_cache.label_exprs,
        label_names=feature_cache.label_names,
        freq=feature_cache.freq,
        start_time=start_time,
        end_time=end_time,
    )

    if infer_processors is None and learn_processors is None:
        infer_processors, learn_processors = _build_processors(
            model_type=model_type,
            normalize_features=normalize_features,
            neutralize_features=neutralize_features,
            renormalize_features_after_neutralize=renormalize_features_after_neutralize,
            normalize_labels=normalize_labels,
            neutralize_labels=neutralize_labels,
            symmetric_orthogonalization=symmetric_orthogonalization,
        )
    
    segments = segments or {
        "train": (fit_start_time, fit_end_time),
        "valid": ("2020-07-01", "2020-09-30"),
        "test": ("2020-10-01", end_time),
    }

    # 核心：将 fit_start_time 和 fit_end_time 注入到需要它们的 Processor 中（如 RobustZScoreNorm）
    infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
    learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

    handler = DataHandlerLP(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        data_loader=data_loader_config,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )
    dataset = create_dataset_from_handler(handler, segments)
    return handler, dataset


def create_alpha360_dataset(
    instruments="csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2020-06-30",
    freq: str = "day",
    infer_processors: Optional[list] = None,
    learn_processors: Optional[list] = None,
    segments: Optional[Dict[str, tuple]] = None,
):
    """
    功能概述：
    - 构建 Alpha360 标准数据集，适合深度学习或高维表达能力更强的模型。
    输入：
    - 股票池、时间区间、切分配置。
    输出：
    - `(handler, dataset)` 二元组。
    边界条件：
    - 需要更高内存与计算资源。
    性能/安全注意事项：
    - 推荐在资源允许时使用，并优先配合批量缓存。
    - 注意：对于 Alpha360 的量价序列特征，通常不做截面中性化，
      而是保持时序原始比例，由深度学习模型（如时序网络）自行提取截面和时序模式。
    """
    from qlib.contrib.data.handler import Alpha360

    infer_processors = infer_processors or [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    learn_processors = learn_processors or [
        {"class": "DropnaLabel"},
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    
    segments = segments or {
        "train": (fit_start_time, fit_end_time),
        "valid": ("2020-07-01", "2020-09-30"),
        "test": ("2020-10-01", end_time),
    }
    handler = Alpha360(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        freq=freq,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )
    dataset = create_dataset_from_handler(handler, segments)
    return handler, dataset
 
class DebuggablePipeline: 
    def __init__(self, processors, name='pipeline'): 
        self.processors = processors 
        self.name = name 
    def __call__(self, df): 
        import sys 
        for i, proc in enumerate(self.processors): 
            before = df.shape 
            try: 
                df = proc(df) 
            except Exception as e: 
                raise RuntimeError(f'[{self.name}] step {i} failed: {e}') 
            after = df.shape 
            if before[1] != after[1]: 
                print(f'[DBG] [{self.name}] step {i}: cols {before[1]} to ', file=sys.stderr) 
        return df 


if __name__ == "__main__":
    print("=== features/dataset.py 独立调用示例 ===")

    try:
        import qlib
        from qlworks.config import QLIB_DATA_DIR
        qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")

        print("\n[1] 开始构建 Alpha158 数据集 (Handler + DatasetH)...")
        handler, dataset = create_alpha158_dataset(
            instruments=["600000.SH"],
            start_time="2020-01-02",
            end_time="2020-01-31",
            fit_start_time="2020-01-02",
            fit_end_time="2020-01-15"
        )

        print("\n[2] 数据集构建成功！")
        print("提取训练集 (Train Frame) 的前两行：")
        df_train = dataset.prepare("train")
        print(df_train.head(2))

    except Exception as e:
        print(f"\n[!] 演示跳过：{e} (需确保 pyqlib 安装且有本地数据)")
