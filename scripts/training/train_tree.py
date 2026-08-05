import os
import sys
import warnings
import argparse
import copy

# MLflow 在某些环境下会尝试写 Roaming 目录，Qlib 导入前先限制
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

# Conda site-packages 和 Roaming 路径优先级调整，避免 Roaming 版本覆盖
sp = list(sys.path)
conda_sp = [p for p in sp if 'Anaconda' in p and 'site-packages' in p]
roaming_sp = [p for p in sp if 'Roaming' in p]
other_sp = [p for p in sp if p not in conda_sp and p not in roaming_sp]
sys.path = conda_sp + other_sp + roaming_sp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import gc
from pathlib import Path
import pandas as pd
import numpy as np
from qlib.data.dataset.handler import DataHandlerLP

# 引入 src 路径到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import (
    create_custom_dataset,
    build_custom_feature_cache,
    wrap_dataset_with_cached_train_frame,
)
from qlworks.models.training import (
    train_lgb_model, train_xgb_model, train_catboost_model,
    predict_ensemble_models, compute_ic, compute_ic_ewma
)
from qlworks.evaluation.selector import (
    compute_factor_ics, check_redundancy, check_icir_stability,
    select_top_by_abs,
)
from qlworks.models import prepare_feature_selection_data, cached_select_features
from qlworks.config import QLIB_DATA_DIR
from qlworks.factors.filter_utils import filter_codes_post, filter_untradeable_labels, apply_label_filter
import qlib
from _config import resolve_runtime_config

# ==============================================================================
# [全局配置]
# 可通过 `--config-source yaml` 切换使用 scripts/training/configs/ 下的 YAML 配置
# ==============================================================================
DEFAULT_YAML_CONFIG_NAME = "tree_2025"
LOCAL_CONFIG = {
    # 股票池：使用本地 instruments/main_board.txt
    # main_board 包含 600/601/603/000 开头的主板股票，支持 PIT 格式
    # 如需全市场测试，改为 all.txt 即可
    "instruments": "csi500", 
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",
    
    # --- 预处理配置 ---
    # 不同模型流派的最佳实践：
    # 1) normalize_features: 特征标准化
    #    - tree: True → 使用 CSQuantileNorm（截面分位数化，适合树模型）
    #    - linear/nn: True → 使用 RobustZScoreNorm（适合线性模型）
    # 2) neutralize_features: 特征中性化
    #    - 使用 CSNeutralize 剥离行业/市值效应
    #    - tree 模型不推荐开启（可通过特征交互学习行业效应）
    #    - linear 模型推荐开启
    # 3) renormalize_features_after_neutralize: 中性化后是否再标准化
    #    - tree 模型不推荐（中性化后再分位数化会引入额外失真）
    # 4) normalize_labels: 标签分位数化
    #    - 将标签转为截面分位数，消除极端值影响
    # 5) neutralize_labels: 标签中性化
    #    - 剥离行业/市值效应后的纯 alpha 标签
    "model_type": "tree",  # 模型流派 (tree / linear / nn)
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"],  # T+1开盘买入, T+5收盘卖出
    "label_names": ["LABEL_5D"],  # 标签名称
    "factor_files": ["reversal_momentum_factors","quality_factors","style_factors","risk_factors","sentiment_factors","other_factors",],  # 因子配置文件
    "factor_cache_names": [],  # DuckDB + Parquet 预计算因子回退
    "normalize_features": True,  # 特征截面分位数化（树模型推荐）
    "neutralize_features": False,  # [AQR修正] 树模型不推荐特征中性化
    "renormalize_features_after_neutralize": False,  # 树模型不需要再标准化
    "normalize_labels": True,  # 标签截面 rank 归一化 → [0,1]（通过 Processor 管线在每个 fold 内独立执行，消除前视偏差）
    "neutralize_labels": False,  # 标签中性化（树模型不推荐）
    "use_dynamic_filter": True,  # 启用流动性过滤（成交量>0 + 近20日均成交额>500万），涨跌停过滤已移至 filter_untradeable_labels
    "filter_new_stocks": True,   # 过滤上市不满 250 日次新股
    "filter_st": True,           # 过滤 ST 股票
    
    # 标签可交易性过滤（剔除涨跌停无法买入的样本）
    "filter_untradeable_labels": True,
    
    # 预测置信度阈值（弱信号日降低无效交易）
    # 每日预测中，绝对偏离中位最小的 bottom_X% 设为 NaN（不交易）
    "prediction_confidence_threshold": 0.2,  # 设为 None 或 0 则关闭
    
    # 窗口级质量闸门
    "window_quality_gate": {
        "enabled": True,
        "min_valid_samples": 100,  # [修正] 30→100，至少覆盖约3个交易日
        "min_healthy_models": 2,
    },
    
    # [Renaissance 标准] 滚动窗口 (Walk-Forward Optimization)
    # 训练→验证→测试 间隔 10 天 Embargo，消除前视偏差
    "rolling_windows": [
        {
            "name": "Test_2023",
            "train": ("2020-01-01", "2021-12-20"),
            "valid": ("2022-01-01", "2022-12-20"),
            "test":  ("2023-01-01", "2023-12-31"),
        },
        {
            "name": "Test_2024",
            "train": ("2021-01-01", "2022-12-20"),
            "valid": ("2023-01-01", "2023-12-20"),
            "test":  ("2024-01-01", "2024-12-31"),
        },
        {
            "name": "Test_2025",
            "train": ("2022-01-01", "2023-12-20"),
            "valid": ("2024-01-01", "2024-12-20"),
            "test":  ("2025-01-01", "2025-12-31"),
        }
    ],
    "top_k_factors": 20,  # 每窗口保留的因子数量
    "feature_selection_date_stride": 2,  # IC 采样跨步（2=隔日采样）
    "train_models": ["lgb", "xgb", "cat"],
    "model_params": {
        "lgb": {
            "num_boost_round": 500, "early_stopping_rounds": 50,
            "learning_rate": 0.02, "max_depth": 5, "num_leaves": 15,
            "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "device_type": "gpu", "gpu_device_id": 0,
        },
        "xgb": {
            "num_boost_round": 500, "early_stopping_rounds": 50,
            "learning_rate": 0.02, "max_depth": 5,
            "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "device": "cuda",
        },
        "cat": {
            "num_boost_round": 500, "early_stopping_rounds": 50,
            "learning_rate": 0.02, "depth": 5,
            "l2_leaf_reg": 3.0, "subsample": 0.7,
            "task_type": "CPU",  # CatBoost GPU 与 LGB/XGB 共存时 CUDA 资源冲突，强制 CPU
        },
    },
    "purged_kfold": {
        "enabled": True,
        "n_splits": 3,
        "purge_days": 11,
    },
    "factor_redundancy_check": {
        "enabled": True,                   # [AQR 标准] 因子冗余剔除
        "correlation_threshold": 0.90,     # 相关系数阈值
    },
    "icir_stability_check": {
        "enabled": True,                   # [Citadel Alpha Lab 标准] ICIR 稳定性
        "rolling_window": 60,              # 滚动窗口
        "keep_ratio": 0.8,                 # ICIR 正向占比要求
    },
    "hyperparam_search": {
        "enabled": False,  # 默认关闭以加速（需要 Optuna，耗时较长）
        "method": "coarse_grid",
        "n_trials_per_model": 12,
    },
    "feature_selection": {
        "method": "embedded",   
        "algo": "lightgbm",     
        "label_col": "LABEL_5D",  
        "remove_collinearity": False,
    },
    # P2: 增强评估配置
    "evaluation": {
        "ic_decay_horizons": [1, 3, 5, 10, 20],  # IC衰减分析的时间跨度
        "industry_exposure_check": True,           # 行业暴露监控
        "marketcap_group_ic": True,                # 市值分组 IC
    },
    # [Renaissance 标准] 多空非对称配置
    # A股融券困难，空头端仅取极强信号（bottom 10%），多头端取 top 30%
    "long_short_ratio": {"long_pct": 0.30, "short_pct": 0.10},
    # 标签窗口级缓存（消除 5 次重复 D.features() 调用）
    "cache_window_labels": True,
}


# ==============================================================================
# [辅助函数] 日历与配置
# ==============================================================================

def get_latest_qlib_calendar_date(calendar_path: str | Path | None = None) -> str:
    """
    读取 Qlib 交易日历中最新的日期。

    参数:
    - calendar_path: Qlib 日历文件路径，默认为 qlib_data/calendars/day.txt

    返回:
    - 最新日期字符串 YYYY-MM-DD

    异常:
    - 日历文件不存在或为空时抛出异常
    """
    path = Path(calendar_path) if calendar_path else Path(QLIB_DATA_DIR) / "calendars" / "day.txt"
    if not path.exists():
        raise FileNotFoundError(f"Qlib 日历文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        dates = [line.strip() for line in f if line.strip()]

    if not dates:
        raise ValueError(f"Qlib 日历文件为空: {path}")

    return dates[-1]


def build_effective_local_config(base_config: dict | None = None, latest_date: str | None = None) -> dict:
    """
    根据 Qlib 数据最新日期动态扩展配置。

    参数:
    - base_config: 基础配置字典（默认 LOCAL_CONFIG）
    - latest_date: 覆盖的最新日期（默认从 Qlib 日历读取）

    返回:
    - 更新了 end_time/rolling_windows 的配置字典

    策略:
    - 自动扩展到数据覆盖的最大日期
    - 若数据覆盖到 2026 年，自动追加 Test_2026 窗口
    """
    config = copy.deepcopy(base_config or LOCAL_CONFIG)
    resolved_latest_date = latest_date or get_latest_qlib_calendar_date()
    latest_ts = pd.Timestamp(resolved_latest_date)

    if latest_ts > pd.Timestamp(config["end_time"]):
        config["end_time"] = resolved_latest_date

    if latest_ts >= pd.Timestamp("2026-01-01"):
        test_2026_window = {
            "name": "Test_2026",
            "train": ("2023-01-01", "2024-12-20"),
            "valid": ("2025-01-01", "2025-12-20"),
            "test": ("2026-01-01", resolved_latest_date),
        }

        windows = []
        replaced = False
        for window in config.get("rolling_windows", []):
            if window.get("name") == "Test_2026":
                windows.append(test_2026_window)
                replaced = True
            else:
                windows.append(window)
        if not replaced:
            windows.append(test_2026_window)
        config["rolling_windows"] = windows

    return config


# ==============================================================================
# [辅助函数] 因子 IC 批量筛选
# ==============================================================================

def _batch_factor_ic_selection(feature_cache, label_expr, label_name, train_start, train_end,
                                out_dir=None, batch_size=20, top_k=60, stride=2):
    """按 batch_size 分批计算因子 IC，避免 OOM。

    参数:
    - feature_cache: 含 factor_series_list 的特征缓存
    - label_expr: Qlib 标签表达式，如 "Ref($close, -1) / $close - 1"
    - label_name: 标签名
    - train_start, train_end: 训练时间区间
    - batch_size: 每批因子数
    - top_k: 保留因子数
    - stride: 采样步长（2=隔日采 IC）

    返回:
    - selected_names: 按 |IC| 降序排列的 top_k 因子名列表
    """
    from qlib.data import D
    import gc

    factor_names = feature_cache.factor_names
    if not factor_names:
        print("    [skip] factor_names is empty, cannot select")
        return []

    print(f"    [batch] {len(factor_names)} factors, batch_size={batch_size}")

    all_instruments = feature_cache.resolved_instruments
    if not all_instruments:
        raise RuntimeError("No instruments available")
    batch_size_instr = 500
    label_frames = []
    for i in range(0, len(all_instruments), batch_size_instr):
        batch_inst = all_instruments[i:i+batch_size_instr]
        try:
            _df = D.features(
                batch_inst,
                [label_expr],
                start_time=train_start,
                end_time=train_end,
                freq="day",
            )
            if _df is not None and not _df.empty:
                label_frames.append(_df)
        except Exception:
            continue
    if not label_frames:
        raise RuntimeError("标签加载失败")

    label_df = pd.concat(label_frames)
    label_df.index.names = ["instrument", "datetime"]
    label_series = label_df[label_df.columns[0]].sort_index()
    label_series = label_series.rename(label_name)
    label_dates = label_series.index.get_level_values("datetime").unique()
    if stride > 1:
        label_dates = label_dates[::stride]
        label_series = label_series[label_series.index.get_level_values("datetime").isin(label_dates)]
    print(f"    [标签] 加载完成: {len(label_series):,} 条, {len(label_dates)} 个交易日")

    # 分批计算 IC
    all_ic_results = {}
    for batch_start in range(0, len(factor_names), batch_size):
        batch_names = factor_names[batch_start:batch_start + batch_size]
        end_idx = min(batch_start+batch_size, len(factor_names))
        print(f"    [分批] 处理 [{batch_start+1}-{end_idx}]/{len(factor_names)}: "
              f"{', '.join(batch_names[:3])}{'...' if len(batch_names)>3 else ''}")

        batch_df = feature_cache.get_warehouse_df(batch_names, start_time=train_start, end_time=train_end)
        if batch_df.empty:
            continue

        # 规范化列名
        if isinstance(batch_df.columns, pd.MultiIndex):
            batch_df.columns = batch_df.columns.get_level_values(1)
        batch_df = batch_df.swaplevel().sort_index() if batch_df.index.names[0] == "datetime" else batch_df.sort_index()

        common_index = batch_df.index.intersection(label_series.index)
        if len(common_index) < 100:
            continue

        batch_df = batch_df.loc[common_index]
        labels = label_series.loc[common_index]

        if stride > 1:
            _dates = batch_df.index.get_level_values("datetime").unique()[::stride]
            _mask = batch_df.index.get_level_values("datetime").isin(_dates)
            batch_df = batch_df.loc[_mask]
            labels = labels.loc[_mask]

        # 逐列计算 Spearman IC（逻辑收敛于 selector.compute_factor_ics）
        batch_ics = compute_factor_ics(batch_df, labels, min_samples=50)
        for col, ic_val in batch_ics.items():
            all_ic_results[col] = ic_val

        del batch_df
        gc.collect()

    # 按 |IC| 降序取 top_k（逻辑收敛于 selector.select_top_by_abs）
    ic_df = pd.Series(all_ic_results)
    selected = select_top_by_abs(ic_df, top_k)
    if selected:
        print(f"    [筛选完成] Top {len(selected)} 因子 (|IC| 范围: "
              f"{abs(ic_df[selected[-1]]):.4f} ~ {abs(ic_df[selected[0]]):.4f})")
    else:
        print("    [筛选完成] 无因子通过 IC 筛选，返回空列表")

    if out_dir is not None:
        ic_df.to_csv(Path(out_dir) / "batch_factor_ic.csv")
        pd.Series(selected).to_csv(Path(out_dir) / "batch_selected_factors.csv", index=False, header=["factor"])

    return selected


# ==============================================================================
# [P1 辅助函数] Purged K-Fold 交叉验证
# ==============================================================================

def _infer_model_type(model) -> str:
    """从模型对象推断模型类型名（lgb/xgb/cat）。

    参数:
    - model: 训练好的 Qlib 模型对象

    返回:
    - 模型类型字符串: 'lgb', 'xgb', 'cat', 或其他可识别名称
    """
    cls_name = str(type(model).__name__).lower()
    if "lgb" in cls_name or "lightgbm" in cls_name:
        return "lgb"
    if "xgb" in cls_name:
        return "xgb"
    if "cat" in cls_name or "catboost" in cls_name:
        return "cat"
    return cls_name


def _purged_kfold_train(
    dataset,
    train_models_config,
    model_params,
    selected_models,
    purged_kfold_config,
    bundle_all,
    CONFIG,
    window_name,
    feature_cache,
    segments,
    window_selected_factors,  # [P0修复] 使用窗口筛选因子，而非全量
) -> list:
    """Purged K-Fold 交叉验证训练。

    将训练期切分为 n_splits 个时序折叠，每折叠间有 purge_days 间隔。
    各折叠独立训练模型，等权集成预测。

    [P0修复] 各折叠使用 window_selected_factors 训练，确保 CV 验证的因子集
    与最终预测一致。

    返回:
    - models: 所有折叠训练出的模型列表（扁平化）
    """
    if not purged_kfold_config.get("enabled", False):
        return None  # 调用方回退到标准训练

    n_splits = purged_kfold_config["n_splits"]
    purge_days = purged_kfold_config["purge_days"]

    train_start = pd.Timestamp(segments["train"][0])
    train_end = pd.Timestamp(segments["train"][1])

    print(f"\n    [Purged K-Fold] n_splits={n_splits}, purge={purge_days}d")
    print(f"    训练期: {train_start.date()} ~ {train_end.date()}")

    try:
        train_frame_full, valid_frame_full = dataset.prepare(["train", "valid"])
    except Exception:
        print(f"    [CV] 无法获取全量数据，回退标准训练")
        return None

    # 提取日期索引
    train_dates = sorted(train_frame_full.index.get_level_values("datetime").unique())

    # 计算折叠边界
    n_dates = len(train_dates)
    if n_dates < n_splits * 2:
        print(f"    [警告] 训练期仅 {n_dates} 天，不足 Purged K-Fold 最低要求，回退标准训练")
        return None

    fold_size = n_dates // (n_splits + 1)

    all_fold_models: list[list] = []
    fold_ic_results = []
    # [P0修复] 追踪每折成功训练的模型类型，用于容错合并
    model_type_order = list(selected_models)

    for fold in range(n_splits):
        fold_valid_start_idx = fold_size * (fold + 1)
        fold_valid_end_idx = min(fold_size * (fold + 2), n_dates)
        fold_train_end_idx = max(0, fold_valid_start_idx - purge_days)

        if fold_train_end_idx <= 0 or fold_valid_start_idx >= n_dates:
            print(f"    [CV fold {fold+1}] 边界不足，跳过")
            continue

        fold_train_start = str(train_dates[0].date())
        fold_train_end = str(train_dates[fold_train_end_idx].date())
        fold_valid_start = str(train_dates[fold_valid_start_idx].date())
        fold_valid_end = str(train_dates[fold_valid_end_idx - 1].date())

        print(f"    [CV fold {fold+1}/{n_splits}] train=[{fold_train_start}, {fold_train_end}], "
              f"valid=[{fold_valid_start}, {fold_valid_end}]")

        fold_segments = {
            "train": (fold_train_start, fold_train_end),
            "valid": (fold_valid_start, fold_valid_end),
            "test": segments["test"],
        }

        try:
            # [P0修复] 每折使用 window_selected_factors 而非全量 feature_cache.factor_names
            _, dataset_fold = create_custom_dataset(
                instruments=CONFIG["instruments"],
                feature_cache=feature_cache,
                selected_feature_names=window_selected_factors,
                start_time=fold_segments["train"][0],
                end_time=fold_segments["test"][1],
                fit_start_time=fold_segments["train"][0],
                fit_end_time=fold_segments["train"][1],
                segments=fold_segments,
                model_type=CONFIG["model_type"],
                normalize_features=CONFIG["normalize_features"],
                neutralize_features=CONFIG["neutralize_features"],
                renormalize_features_after_neutralize=CONFIG["renormalize_features_after_neutralize"],
                normalize_labels=CONFIG["normalize_labels"],
                neutralize_labels=CONFIG["neutralize_labels"],
                use_dynamic_filter=CONFIG.get("use_dynamic_filter", False),
            )

            fold_train_frame = dataset_fold.prepare("train")
            # [Virtu-Renaissance 修复] 每折独立过滤不可交易标签
            if CONFIG.get("filter_untradeable_labels", False):
                _fold_instruments = fold_train_frame.index.get_level_values("instrument").unique().tolist()
                fold_train_frame = apply_label_filter(
                    fold_train_frame, _fold_instruments,
                    fold_train_start, fold_train_end, bundle_all.label_names
                )
            dataset_fold = wrap_dataset_with_cached_train_frame(
                dataset_fold,
                train_frame=fold_train_frame,
                selected_feature_names=window_selected_factors,
                label_names=bundle_all.label_names,
                learn_data_key=DataHandlerLP.DK_L,
                infer_data_key=DataHandlerLP.DK_I,
            )

            # 训练该折叠的模型（按模型类型有序）
            fold_models = []
            if "lgb" in selected_models:
                fold_models.append(train_lgb_model(dataset_fold, params=model_params.get("lgb")))
            if "xgb" in selected_models:
                fold_models.append(train_xgb_model(dataset_fold, params=model_params.get("xgb")))
            if "cat" in selected_models:
                fold_models.append(train_catboost_model(dataset_fold, params=model_params.get("cat")))

            if fold_models:
                all_fold_models.append(fold_models)

                # 快速验证集 IC
                try:
                    valid_frame = dataset_fold.prepare("valid")
                    if "label" in valid_frame.columns.get_level_values(0):
                        actual_label = valid_frame["label"].squeeze()
                        if isinstance(actual_label, pd.DataFrame):
                            actual_label = actual_label.iloc[:, 0]
                        for m in fold_models:
                            val_pred = m.predict(dataset_fold, segment="valid")
                            if isinstance(val_pred, pd.DataFrame):
                                val_pred = val_pred.iloc[:, 0]
                            common_idx = actual_label.index.intersection(val_pred.index)
                            if len(common_idx) >= 30:
                                ic_val = compute_ic(val_pred.loc[common_idx], actual_label.loc[common_idx])
                                fold_ic_results.append(ic_val)
                except Exception:
                    pass

            del dataset_fold, fold_train_frame
            gc.collect()

        except Exception as e:
            print(f"    [CV fold {fold+1}] 训练失败: {e}")
            continue

    if fold_ic_results:
        print(f"    [CV 汇总] {len(fold_ic_results)} 个折叠验证, IC 均值: {np.mean(fold_ic_results):.4f}, "
              f"std: {np.std(fold_ic_results):.4f}")

    # [P0修复] 容错合并：按模型类型分组，允许不同折叠的模型类型不完全一致
    if all_fold_models:
        # 统计每类模型在各折叠中的出现次数
        type_models: dict[str, list] = {mt: [] for mt in model_type_order}
        for fold_models in all_fold_models:
            for m_idx, m_type in enumerate(model_type_order):
                if m_idx < len(fold_models):
                    type_models[m_type].append(fold_models[m_idx])

        merged_models = []
        for m_type in model_type_order:
            type_list = type_models.get(m_type, [])
            if type_list:
                merged_models.extend(type_list)
                print(f"    [CV 合并] {m_type}: {len(type_list)}/{len(all_fold_models)} 折叠成功")
            else:
                print(f"    [警告] {m_type}: 所有折叠训练失败，跳过")

        if merged_models:
            print(f"    [CV 完成] {len(merged_models)} 个模型 (来自 {len(all_fold_models)} 折叠)")
            return merged_models

    return None


"""
# [已迁移] 标签可交易性过滤 → qlworks.factors.filter_utils
#   filter_untradeable_labels / apply_label_filter
"""


# ==============================================================================
# [P2 辅助函数] IC 衰减分析
# ==============================================================================

def _compute_ic_decay(feature_cache, selected_factors, instruments, train_start, train_end, horizons):
    """计算因子在不同前瞻周期下的 IC 衰减曲线。

    参数:
    - feature_cache: 特征缓存
    - selected_factors: 选出的因子名列表
    - instruments: 股票列表
    - train_start, train_end: 训练期
    - horizons: IC 衰减分析的时间跨度列表，如 [1, 3, 5, 10, 20]

    返回:
    - decay_df: DataFrame, 行=horizon, 列=factor, 值=IC
    """
    from qlib.data import D

    print(f"\n    [IC 衰减分析] horizons={horizons}")

    # 获取因子数据
    factor_df = feature_cache.get_warehouse_df(selected_factors, start_time=train_start, end_time=train_end)
    if factor_df.empty:
        return pd.DataFrame()

    if isinstance(factor_df.columns, pd.MultiIndex):
        factor_df.columns = factor_df.columns.get_level_values(1)

    decay_results = {}
    for horizon in horizons:
        label_expr = f"Ref($close, -{horizon}) / Ref($open, -1) - 1"
        try:
            label_frames = []
            for i in range(0, len(instruments), 500):
                batch_inst = instruments[i:i+500]
                _df = D.features(batch_inst, [label_expr], start_time=train_start, end_time=train_end, freq="day")
                if _df is not None and not _df.empty:
                    label_frames.append(_df)

            if not label_frames:
                continue

            label_df = pd.concat(label_frames)
            label_series = label_df[label_df.columns[0]].sort_index()

            horizon_ic = {}
            for col in selected_factors:
                if col not in factor_df.columns:
                    continue
                feat = factor_df[col].dropna()
                lab = label_series.reindex(feat.index).dropna()
                common = feat.index.intersection(lab.index)
                if len(common) >= 50:
                    ic_val = compute_ic(feat.loc[common], lab.loc[common])
                    horizon_ic[col] = ic_val
                else:
                    horizon_ic[col] = np.nan

            decay_results[f"H{horizon}"] = horizon_ic
            print(f"      H{horizon}: mean|IC|={np.nanmean(np.abs(list(horizon_ic.values()))):.4f}")

        except Exception as e:
            print(f"      H{horizon}: 计算失败 ({e})")

    if decay_results:
        decay_df = pd.DataFrame(decay_results).T
        return decay_df
    return pd.DataFrame()


# ==============================================================================
# [P2 辅助函数] 行业/市值分组分析
# ==============================================================================

def _compute_group_ic(predictions_df, actual_label_series, analysis_type="industry"):
    """计算行业或市值分组 IC。

    参数:
    - predictions_df: 预测结果 DataFrame，含 instrument/datetime MultiIndex
    - actual_label_series: 实际标签 Series
    - analysis_type: "industry" 或 "marketcap"

    返回:
    - group_ic: {group_name: IC_value} 或空 dict
    """
    from qlib.data import D

    if predictions_df.empty or actual_label_series.empty:
        return {}

    # 获取所有涉及日期的唯一股票列表
    all_instruments = list(set(
        list(predictions_df.index.get_level_values("instrument").unique()) +
        list(actual_label_series.index.get_level_values("instrument").unique())
    ))

    # 样本最多 10 个交易日
    sample_dates = sorted(predictions_df.index.get_level_values("datetime").unique())
    if len(sample_dates) > 10:
        sample_dates = sample_dates[::max(1, len(sample_dates) // 10)]

    group_ics = {}

    if analysis_type == "industry":
        try:
            # 使用最近日期的行业分类
            ref_date = str(sample_dates[-1].date()) if len(sample_dates) > 0 else "2024-12-31"
            print(f"    [行业分析] 参考日期={ref_date}")

            batch_size = 500
            ind_frames = []
            for i in range(0, len(all_instruments), batch_size):
                batch_inst = all_instruments[i:i+batch_size]
                try:
                    _df = D.features(batch_inst, ['$sw_l1'], start_time=ref_date, end_time=ref_date)
                    if _df is not None and not _df.empty:
                        ind_frames.append(_df)
                except Exception:
                    continue

            if ind_frames:
                ind_df = pd.concat(ind_frames)
                if isinstance(ind_df.columns, pd.MultiIndex):
                    ind_df.columns = ind_df.columns.droplevel(1)
                ind_map = ind_df[ind_df.columns[0]].to_dict()

                # 为每个样本添加行业标签
                aligned = predictions_df.copy()
                inst_col = aligned.index.get_level_values("instrument")
                aligned["industry"] = inst_col.map(ind_map)
                aligned["actual"] = actual_label_series.reindex(aligned.index)

                for industry_name, group in aligned.groupby("industry"):
                    if len(group) < 30:
                        continue
                    valid = group.dropna(subset=["score", "actual"])
                    if len(valid) < 20:
                        continue
                    ic_val = compute_ic(valid["score"], valid["actual"])
                    group_ics[str(industry_name)] = ic_val

        except Exception as e:
            print(f"    [行业分析] 失败: {e}")

    elif analysis_type == "marketcap":
        try:
            # 使用市值数据分组
            mkt_frames = []
            for i in range(0, len(all_instruments), 500):
                batch_inst = all_instruments[i:i+500]
                try:
                    ref_d = str(sample_dates[0].date()) if sample_dates else "2024-01-01"
                    _df = D.features(
                        batch_inst, ['$market_cap'],
                        start_time=ref_d,
                        end_time=str(sample_dates[-1].date()) if sample_dates else "2024-12-31"
                    )
                    if _df is not None and not _df.empty:
                        mkt_frames.append(_df)
                except Exception:
                    continue

            if mkt_frames:
                mkt_df = pd.concat(mkt_frames)
                if isinstance(mkt_df.columns, pd.MultiIndex):
                    mkt_df.columns = mkt_df.columns.droplevel(1)
                mkt_col = mkt_df.columns[0]
                mkt_cap = mkt_df[mkt_col].groupby(level="datetime").transform(
                    lambda x: pd.qcut(x.rank(method='first'), 3, labels=['小市值', '中市值', '大市值'])
                )

                aligned = predictions_df.copy()
                aligned["mkt_group"] = mkt_cap.reindex(aligned.index)
                aligned["actual"] = actual_label_series.reindex(aligned.index)

                for mkt_group, group in aligned.groupby("mkt_group"):
                    if len(group) < 30:
                        continue
                    valid = group.dropna(subset=["score", "actual"])
                    if len(valid) < 20:
                        continue
                    ic_val = compute_ic(valid["score"], valid["actual"])
                    group_ics[str(mkt_group)] = ic_val

        except Exception as e:
            print(f"    [市值分析] 失败: {e}")

    return group_ics


# ==============================================================================
# [P1 辅助函数] 窗口级标签缓存（消除 5 次重复 D.features() 调用）
# ==============================================================================

def _load_window_labels(instruments, label_expr, train_start, train_end,
                         label_cache: dict | None = None) -> tuple[pd.Series, dict]:
    """加载窗口训练期标签（带缓存）。

    [Bloomberg 数据管道标准] 每个窗口的标签只加载一次，
    后续 IC 计算、ICIR 检测、衰减分析等全部复用缓存。

    参数:
    - instruments: 股票列表
    - label_expr: Qlib 标签表达式
    - train_start, train_end: 训练时间区间
    - label_cache: 缓存字典，键为 (train_start, train_end)

    返回:
    - (label_series, updated_cache)
    """
    from qlib.data import D

    cache_key = f"{train_start}_{train_end}"
    if label_cache is not None and cache_key in label_cache:
        return label_cache[cache_key], label_cache

    print(f"    [标签缓存] 加载 {train_start}~{train_end} 标签...")
    label_frames = []
    for i in range(0, len(instruments), 500):
        batch_inst = instruments[i:i+500]
        try:
            _df = D.features(
                batch_inst, [label_expr],
                start_time=train_start, end_time=train_end, freq="day",
            )
            if _df is not None and not _df.empty:
                label_frames.append(_df)
        except Exception:
            continue

    if not label_frames:
        raise RuntimeError(f"标签加载失败: {train_start}~{train_end}")

    label_df = pd.concat(label_frames)
    label_series = label_df[label_df.columns[0]].sort_index()

    if label_cache is not None:
        label_cache[cache_key] = label_series

    return label_series, label_cache

def run_ml_pipeline(config_source: str = "local", config_name: str | None = None):
    CONFIG = resolve_runtime_config(
        local_config=build_effective_local_config(),
        default_yaml_name=DEFAULT_YAML_CONFIG_NAME,
        config_source=config_source,
        config_name=config_name,
    )
    print("="*60)
    print("=== 多因子树模型集成训练流水线（Pro 版） ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 引擎...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading", maxtasksperchild=None)

    # [1b] 加载 main_board 股票列表（用于后置过滤）
    print("\n[1b] 加载 main_board 股票池...")
    _main_board_stocks = set()
    try:
        _inst_dir = Path(QLIB_DATA_DIR) / "instruments" / "main_board.txt"
        if not _inst_dir.exists():
            print("  [警告] main_board.txt 不存在，后置过滤将跳过")
        else:
            with open(_inst_dir) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l: continue
                    _parts = _l.split()
                    if _parts:
                        _main_board_stocks.add(_parts[0].lower())
            print(f"  main_board 股票池: {_inst_dir} ({len(_main_board_stocks)} 只)")
            print(f"  示例: {sorted(_main_board_stocks)[:5]}")
    except Exception as e:
        print(f"  [警告] main_board 加载失败: {e}")

    # [1c] 退市信息预览
    print("\n[1c] 退市信息预览...")
    try:
        _all_txt = Path(QLIB_DATA_DIR) / "instruments" / "all.txt"
        if _all_txt.exists():
            with open(_all_txt) as _f:
                _delisted = sum(1 for _l in _f if _l.strip() and not _l.strip().endswith('9999-12-31'))
            print(f"  all.txt 中 {_delisted} 只已退市股票 (退市日期 ≠ 9999-12-31)")
    except Exception:
        pass

    # 2. 构建因子库
    print("\n[2] 构建因子库 (Factor Library)...")
    factor_files = CONFIG["factor_files"]
    bundle_all = build_factor_library_bundle(factor_files)
    
    # 注入标签配置
    bundle_all.label_fields = CONFIG["label_fields"]
    bundle_all.label_names = CONFIG["label_names"]
    
    print(f">>> 加载完成: {len(bundle_all.fields)} 个因子表达式, "
          f"标签: {bundle_all.label_names[0]} (5日跨夜动量)")

    # 3. 构建全周期特征缓存（数据底座，一次性加载）
    # =========================================================================
    # [Bloomberg 数据管道] 全周期特征缓存作为数据底座，
    # 后续各窗口通过 start_time/end_time 参数按需切片，避免重复加载。
    # =========================================================================
    print(f"\n[3] 构建全周期特征缓存 ({CONFIG['start_time']} ~ {CONFIG['end_time']})...")
    global_feature_cache = build_custom_feature_cache(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle_all,
        factor_cache_names=CONFIG["factor_cache_names"],
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        freq="day",
        use_dynamic_filter=CONFIG["use_dynamic_filter"],
    )
    print(f"    >>> 动态过滤: {'开启' if CONFIG['use_dynamic_filter'] else '关闭'}")

    # 获取全量股票列表
    all_instruments = global_feature_cache.resolved_instruments
    if not all_instruments:
        raise RuntimeError("未能解析任何股票，请检查 instruments 配置和数据")

    # =========================================================================
    # [P1 核心改进] Walk-Forward 滚动窗口独立因子筛选 + Purged K-Fold
    # =========================================================================
    # 每个窗口独立进行：
    #   1. 因子 IC 粗筛（仅用该窗口训练期数据）
    #   2. 因子冗余检测（高相关因子去重）
    #   3. ICIR 稳定性检查
    #   4. Purged K-Fold 交叉验证训练
    #   5. 多模型集成预测
    # =========================================================================

    all_predictions = []
    label_expr = CONFIG["label_fields"][0]
    label_name = CONFIG["label_names"][0]
    top_k = CONFIG["top_k_factors"]
    stride = max(int(CONFIG.get("feature_selection_date_stride", 2)), 1)
    selected_models = list(CONFIG.get("train_models", ["lgb", "xgb", "cat"]))
    model_params = CONFIG.get("model_params", {})
    model_ic_history: dict[str, list[float]] = {}

    # 跨窗口因子追踪
    all_window_selected_factors: dict[str, list[str]] = {}
    all_window_performance: dict[str, dict] = {}
    # [P1] 窗口级标签缓存——消除重复 D.features() 调用
    window_label_cache: dict[str, pd.Series] = {}
    test_label_cache: dict[str, pd.Series] = {}

    for window_idx, window in enumerate(CONFIG["rolling_windows"]):
        window_name = window["name"]
        print(f"\n{'='*60}")
        print(f"=== 滚动窗口 [{window_idx+1}/{len(CONFIG['rolling_windows'])}]: {window_name} ===")
        print(f"    训练: {window['train'][0]} ~ {window['train'][1]}")
        print(f"    验证: {window['valid'][0]} ~ {window['valid'][1]}")
        print(f"    测试: {window['test'][0]} ~ {window['test'][1]}")
        print(f"{'='*60}")

        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }

        # ---- [P1 Step 1] 窗口独立因子 IC 粗筛 ----
        print(f"\n  [Step 1] {window_name} 独立因子 IC 粗筛 (训练期数据)...")
        window_selected_factors = _batch_factor_ic_selection(
            feature_cache=global_feature_cache,
            label_expr=label_expr,
            label_name=label_name,
            train_start=segments["train"][0],
            train_end=segments["train"][1],
            out_dir=CONFIG.get("output_dir", "."),
            batch_size=20,
            top_k=top_k,
            stride=stride,
        )

        if not window_selected_factors:
            print(f"  [警告] {window_name} 无可选因子，跳过该窗口")
            continue

        print(f"\n  >>> {window_name} 初筛因子 ({len(window_selected_factors)} 个):")
        for i, fname in enumerate(window_selected_factors, 1):
            print(f"      {i}. {fname}")

        # ---- [P1 Step 2] 因子冗余检测 ----
        factor_redun_conf = CONFIG.get("factor_redundancy_check", {})
        if factor_redun_conf.get("enabled", False) and len(window_selected_factors) > 5:
            corr_threshold = factor_redun_conf.get("correlation_threshold", 0.90)
            print(f"\n  [Step 2] 因子冗余检测 (相关性 > {corr_threshold})...")

            # 从特征缓存获取窗口训练期的因子数据
            feat_data = global_feature_cache.get_warehouse_df(
                window_selected_factors,
                start_time=segments["train"][0],
                end_time=segments["train"][1],
            )

            if not feat_data.empty:
                if isinstance(feat_data.columns, pd.MultiIndex):
                    feat_data.columns = feat_data.columns.get_level_values(1)

                # 降采样以加速计算（大样本时）
                feat_data_clean = feat_data.dropna()
                if len(feat_data_clean) > 5000:
                    if isinstance(feat_data_clean.index, pd.MultiIndex):
                        feat_data_clean = feat_data_clean.groupby(
                            level='datetime', group_keys=False
                        ).apply(lambda x: x.sample(max(1, int(len(x) * 0.2)), random_state=42))
                    if len(feat_data_clean) > 5000:
                        feat_data_clean = feat_data_clean.sample(5000, random_state=42)

                # 冗余检测（逻辑收敛于 selector.check_redundancy；沿用 pearson 以保持原行为）
                # tie-break: 按 window_selected_factors 顺序（越靠前 |IC| 越大），保留靠前者
                kept = check_redundancy(
                    feat_data_clean, window_selected_factors,
                    threshold=corr_threshold,
                    method="pearson",
                )
                to_drop = set(window_selected_factors) - set(kept)

                if to_drop:
                    print(f"      剔除 {len(to_drop)} 个冗余因子: {sorted(to_drop)}")
                    print(f"      保留 {len(kept)} 个")
                    window_selected_factors = kept
                else:
                    print(f"      无高相关因子对")
            else:
                print(f"      因子数据为空，跳过冗余检测")

        # ---- [P1 Step 3] ICIR 稳定性检测 ----
        icir_conf = CONFIG.get("icir_stability_check", {})
        if icir_conf.get("enabled", False) and len(window_selected_factors) > max(top_k // 2, 3):
            rolling_w = icir_conf["rolling_window"]
            keep_ratio = icir_conf["keep_ratio"]
            min_keep = max(int(len(window_selected_factors) * keep_ratio), 3)
            print(f"\n  [Step 3] ICIR 稳定性检测 (rolling={rolling_w}d, 保留率>{keep_ratio}, 至少{min_keep}个)...")

            # 获取窗口训练期因子数据
            icir_feat_data = global_feature_cache.get_warehouse_df(
                window_selected_factors,
                start_time=segments["train"][0],
                end_time=segments["train"][1],
            )

            if not icir_feat_data.empty:
                if isinstance(icir_feat_data.columns, pd.MultiIndex):
                    icir_feat_data.columns = icir_feat_data.columns.get_level_values(1)

                # [P1] 从缓存加载标签（避免重复 D.features() 调用）
                try:
                    # 训练期标签缓存
                    train_cache_key = f"{segments['train'][0]}_{segments['train'][1]}"
                    icir_label_series, window_label_cache = _load_window_labels(
                        all_instruments, label_expr,
                        segments["train"][0], segments["train"][1],
                        window_label_cache,
                    )
                    icir_label_series = icir_label_series.rename(label_name)

                    # ICIR 稳定性检测（逻辑收敛于 selector.check_icir_stability；
                    # 向量化逐日截面 IC，等价于原逐日循环）
                    stable_factors = check_icir_stability(
                        icir_feat_data, icir_label_series, window_selected_factors,
                        rolling_window=rolling_w,
                        keep_ratio=keep_ratio,
                        min_keep=min_keep,
                    )
                    removed_ic = [f for f in window_selected_factors if f not in stable_factors]
                    if removed_ic:
                        print(f"      剔除 {len(removed_ic)} 个 ICIR 不稳定因子: {removed_ic}")
                        print(f"      保留 {len(stable_factors)} 个 ICIR 稳定因子")
                        window_selected_factors = stable_factors
                    else:
                        print(f"      所有因子 ICIR 稳定，无需剔除")
                except Exception as e:
                    import traceback
                    _tb_lines = traceback.format_exc().strip().split("\n")
                    _last_frames = [l for l in _tb_lines if "File " in l][-3:]
                    print(f"      ICIR 检测异常: {e}，跳过")
                    print(f"      [诊断] 最后 3 帧:")
                    for _f in _last_frames:
                        print(f"        {_f.strip()}")

        # 记录本窗口选中的因子
        all_window_selected_factors[window_name] = window_selected_factors

        print(f"\n  >>> {window_name} 最终因子 ({len(window_selected_factors)} 个):")
        for i, fname in enumerate(window_selected_factors, 1):
            print(f"      {i}. {fname}")

        # ---- [Step 4] 创建数据集 ----
        print(f"\n  [Step 4] 创建 {window_name} 数据集...")
        _, dataset_sub = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_cache=global_feature_cache,
            selected_feature_names=window_selected_factors,
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type=CONFIG["model_type"],
            normalize_features=CONFIG["normalize_features"],
            neutralize_features=CONFIG["neutralize_features"],
            renormalize_features_after_neutralize=CONFIG["renormalize_features_after_neutralize"],
            normalize_labels=CONFIG["normalize_labels"],
            neutralize_labels=CONFIG["neutralize_labels"],
            use_dynamic_filter=CONFIG.get("use_dynamic_filter", False),
        )

        train_frame_window = dataset_sub.prepare("train")
        print(f"    >>> 训练集: {train_frame_window.shape[0]} 行 x {train_frame_window.shape[1]} 列")

        valid_frame = None
        try:
            valid_frame = dataset_sub.prepare("valid")
            print(f"    >>> 验证集: {valid_frame.shape}")
        except Exception as e:
            print(f"      [警告] 验证集准备失败: {e}")

        # [Virtu-Renaissance 修复] 激活标签可交易性过滤：
        # 检查 T+1 开盘跳空 / T+1 一字板，标记不可交易样本。
        # 涨跌停股的特征保留（参与 CSQuantileNorm 截面分布），仅标签标 NaN。
        if CONFIG.get("filter_untradeable_labels", False):
            _train_instruments = train_frame_window.index.get_level_values("instrument").unique().tolist()
            train_frame_window = apply_label_filter(
                train_frame_window, _train_instruments,
                segments["train"][0], segments["train"][1], bundle_all.label_names
            )
            if valid_frame is not None:
                _valid_instruments = valid_frame.index.get_level_values("instrument").unique().tolist()
                valid_frame = apply_label_filter(
                    valid_frame, _valid_instruments,
                    segments["valid"][0], segments["valid"][1], bundle_all.label_names
                )

        # 缓存训练帧避免重复 prepare
        dataset_sub = wrap_dataset_with_cached_train_frame(
            dataset_sub,
            train_frame=train_frame_window,
            selected_feature_names=window_selected_factors,
            label_names=bundle_all.label_names,
            learn_data_key=DataHandlerLP.DK_L,
            infer_data_key=DataHandlerLP.DK_I,
            valid_frame=valid_frame,
        )

        del train_frame_window
        gc.collect()

        # ---- [Step 5] Purged K-Fold 训练 ----
        print(f"\n  [Step 5] {window_name} 模型训练...")
        purged_cfg = CONFIG.get("purged_kfold", {})

        if purged_cfg.get("enabled", False):
            cv_models = _purged_kfold_train(
                dataset=dataset_sub,
                train_models_config=CONFIG,
                model_params=model_params,
                selected_models=selected_models,
                purged_kfold_config=purged_cfg,
                bundle_all=bundle_all,
                CONFIG=CONFIG,
                window_name=window_name,
                feature_cache=global_feature_cache,
                segments=segments,
                window_selected_factors=window_selected_factors,  # [P0修复]
            )
        else:
            cv_models = None

        if cv_models:
            models = cv_models
            print(f"    >>> Purged K-Fold 训练完成: {len(models)} 个模型")
        else:
            # 标准训练（无 Purged K-Fold）
            models = []
            if "lgb" in selected_models:
                print("    - LightGBM...")
                models.append(train_lgb_model(dataset_sub, params=model_params.get("lgb")))
            if "xgb" in selected_models:
                print("    - XGBoost...")
                models.append(train_xgb_model(dataset_sub, params=model_params.get("xgb")))
            if "cat" in selected_models:
                print("    - CatBoost (GPU)...")
                models.append(train_catboost_model(dataset_sub, params=model_params.get("cat")))

        if not models:
            print(f"    [警告] {window_name} 无可用模型，跳过预测")
            del dataset_sub, global_feature_cache
            gc.collect()
            continue

        # ---- [Step 6] 模型 IC 加权（按模型类型分组追踪，兼容 CV/非CV） ----
        # [P0修复] 使用模型类型名（lgb/xgb/cat）作为 EWMA-IC 键，
        # 而非索引（model_0/model_1），消除 CV/非CV 模式间的不一致
        model_ic_weights = []
        # [P0修复2] 健壮检测 valid_frame 是否包含标签列。
        # Qlib DatasetH.prepare() 在不同 handler 配置下可能返回：
        # - MultiIndex 列: ("feature", col) / ("label", col)
        # - flat 列名: 直接使用字段名
        # 需要兼容两种格式。
        _has_label = False
        if valid_frame is not None:
            if isinstance(valid_frame.columns, pd.MultiIndex):
                _has_label = "label" in valid_frame.columns.get_level_values(0)
            else:
                _has_label = any("LABEL" in str(c).upper() for c in valid_frame.columns)
        if valid_frame is not None and _has_label:
            if isinstance(valid_frame.columns, pd.MultiIndex):
                actual_label = valid_frame["label"].squeeze()
            else:
                _label_cols = [c for c in valid_frame.columns if "LABEL" in str(c).upper()]
                actual_label = valid_frame[_label_cols[0]] if _label_cols else valid_frame.iloc[:, -1]
            if isinstance(actual_label, pd.DataFrame):
                actual_label = actual_label.iloc[:, 0]

            # 按模型类型聚合验证 IC
            type_ic_map: dict[str, list[float]] = {}
            for m_idx, model in enumerate(models):
                try:
                    val_pred = model.predict(dataset_sub, segment="valid")
                    if isinstance(val_pred, pd.DataFrame):
                        val_pred = val_pred.iloc[:, 0]
                    aligned_actual = actual_label.reindex(val_pred.index).dropna()
                    valid_pred_aligned = val_pred.reindex(aligned_actual.index).dropna()
                    common_idx = aligned_actual.index.intersection(valid_pred_aligned.index)

                    if len(common_idx) >= 30:
                        ic_val = compute_ic(valid_pred_aligned.loc[common_idx], aligned_actual.loc[common_idx])
                        m_type = _infer_model_type(model)
                        type_ic_map.setdefault(m_type, []).append(ic_val)
                except Exception:
                    pass

            # 各类型取平均 IC，再计算 EWMA
            for m_type in selected_models:
                ic_list = type_ic_map.get(m_type, [])
                if ic_list:
                    mean_ic = np.mean(ic_list)
                    ewma_ic = compute_ic_ewma(model_ic_history, m_type, mean_ic, half_life=4)
                    model_ic_weights.append(max(ewma_ic, 0.0))
                    print(f"      [{m_type}] 验证 IC={mean_ic:.4f} (n={len(ic_list)}), EWMA-IC={ewma_ic:.4f}")
                else:
                    # 该类型在 CV 中全部失败，用历史平均或等权
                    fallback_ic = compute_ic_ewma(model_ic_history, m_type, 0.0, half_life=4)
                    model_ic_weights.append(max(fallback_ic, 1.0 / len(selected_models) * 0.5))
                    print(f"      [{m_type}] 无有效验证IC, fallback={model_ic_weights[-1]:.4f}")
        else:
            model_ic_weights = [1.0 / len(selected_models)] * len(selected_models)
            # [诊断] 输出 valid_frame 列结构，便于追踪为何 EWMA-IC 未计算
            _col_info = "无 MultiIndex" if not isinstance(valid_frame.columns, pd.MultiIndex) else f"level0={list(valid_frame.columns.get_level_values(0).unique())}" if valid_frame is not None else "valid_frame=None"
            print(f"      [权重诊断] valid_frame 列结构: {_col_info}, 使用等权 fallback")

        # 归一化权重
        weights_arr = np.array(model_ic_weights, dtype=float)
        if weights_arr.sum() > 0:
            model_ic_weights = (weights_arr / weights_arr.sum()).tolist()
        else:
            model_ic_weights = [1.0 / len(selected_models)] * len(selected_models)

        # [P0修复] CV模式下 models 数量可能 > selected_models 数量（每类多折）。
        # 将按类型计算的权重扩展到每个模型：同类型所有折叠共享同一权重。
        if len(models) > len(model_ic_weights):
            # 按模型类型统计每类模型数量
            type_count: dict[str, int] = {}
            for m in models:
                m_type = _infer_model_type(m)
                type_count[m_type] = type_count.get(m_type, 0) + 1
            expanded_weights = []
            for m_type in selected_models:
                count = type_count.get(m_type, 0)
                w = model_ic_weights[selected_models.index(m_type)] if m_type in selected_models else 0.0
                expanded_weights.extend([w] * count)
            model_ic_weights = expanded_weights
            print(f"      集成权重 (CV展开): {[f'{w:.3f}' for w in model_ic_weights]}")

        print(f"      集成权重: {[f'{w:.3f}' for w in model_ic_weights]}")

        # ---- 质量闸门 ----
        gate_cfg = CONFIG.get("window_quality_gate", {})
        if gate_cfg.get("enabled", True):
            healthy_count = sum(1 for w in model_ic_weights if w > 1e-6)
            min_healthy = gate_cfg.get("min_healthy_models", 2)
            min_samples = gate_cfg.get("min_valid_samples", 100)

            valid_sample_count = valid_frame.shape[0] if valid_frame is not None else 0

            if valid_sample_count < min_samples:
                print(f"    [闸门] {window_name} 验证样本不足 ({valid_sample_count} < {min_samples})，跳过")
                del dataset_sub, models
                gc.collect()
                continue
            if healthy_count < min_healthy:
                print(f"    [闸门] {window_name} 健康模型不足 ({healthy_count} < {min_healthy})，跳过")
                del dataset_sub, models
                gc.collect()
                continue
            print(f"    [闸门] {window_name} 通过: {valid_sample_count} 样本, {healthy_count}/{len(models)} 健康模型")

        # ---- [Step 7] 预测 + 后处理 ----
        print(f"\n  [Step 7] {window_name} 测试集预测...")
        predictions = predict_ensemble_models(models, dataset_sub, segment="test", model_weights=model_ic_weights)

        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
        predictions = predictions.dropna(subset=["score"])

        if predictions.empty:
            print(f"    [警告] {window_name} 测试集预测为空")
            del dataset_sub, models
            gc.collect()
            continue

        # 保存原始预测（用于后续评估）
        predictions["raw_score"] = predictions["score"]

        # 截面分位数排名（标准做法——先排名再处理置信度）
        predictions["score"] = predictions.groupby(
            level="datetime"
        )["score"].rank(pct=True, na_option="keep")

        # [P1修复] 置信度阈值在排名之后应用
        # 不再直接剔除弱信号，而是用排名衰减降低其影响力
        # 排名后的弱信号（接近 0.5 的）被推向 0.5（中性）
        conf_threshold = CONFIG.get("prediction_confidence_threshold")
        if conf_threshold is not None and conf_threshold > 0:
            raw_ranked = predictions["score"].copy()
            # 信号强度 = |rank - 0.5|（距离中位越近，信号越弱）
            signal_strength = (raw_ranked - 0.5).abs()
            daily_median_strength = signal_strength.groupby(level="datetime").transform("median")
            # 弱信号当日 signal_strength < median_strength * conf_threshold
            weak_signal = signal_strength < daily_median_strength * conf_threshold
            n_weak = weak_signal.sum()
            if n_weak > 0:
                # [P1修复] 衰减而非直接剔除：弱信号排名 → 0.5（中性，不交易）
                predictions.loc[weak_signal, "score"] = 0.5
                print(f"    [置信度衰减] {n_weak} 个弱信号衰减至中性 "
                      f"({100*n_weak/max(len(predictions),1):.1f}%)")

        # [P1修复] 空头端非对称处理：A股融券困难，仅保留强空头信号
        ls_ratio = CONFIG.get("long_short_ratio", {"long_pct": 0.30, "short_pct": 0.10})
        long_cutoff = 1.0 - ls_ratio["long_pct"]
        short_cutoff = ls_ratio["short_pct"]
        # 中位附近（long_cutoff < score < 1-short_cutoff）→ 中性 0.5
        middle_mask = (predictions["score"] > short_cutoff) & (predictions["score"] < long_cutoff)
        n_middle = middle_mask.sum()
        if n_middle > 0:
            predictions.loc[middle_mask, "score"] = 0.5
            print(f"    [多空非对称] {n_middle} 个中性信号衰减 (long_top={ls_ratio['long_pct']:.0%}, "
                  f"short_bottom={ls_ratio['short_pct']:.0%})")

        # ---- [Step 8] 窗口性能记录 ----
        window_perf = {
            "n_factors": len(window_selected_factors),
            "n_models": len(models),
            "n_predictions": len(predictions),
            "model_weights": model_ic_weights,
        }

        # 如果有实际标签，计算测试集 IC
        try:
            # [P1] 从缓存加载测试期标签
            test_cache_key = f"{segments['test'][0]}_{segments['test'][1]}"
            test_label_series, test_label_cache = _load_window_labels(
                all_instruments, label_expr,
                segments["test"][0], segments["test"][1],
                test_label_cache,
            )
            test_label_series = test_label_series.rename(label_name)

            # 对齐预测和标签
            common_idx = predictions.index.intersection(test_label_series.index)
            if len(common_idx) >= 100:
                test_ic = compute_ic(
                    predictions.loc[common_idx, "score"],
                    test_label_series.loc[common_idx]
                )
                window_perf["test_ic"] = test_ic
                print(f"    [测试 IC] {window_name} rank IC = {test_ic:.4f}")
        except Exception as e:
            print(f"    [测试 IC] 计算失败: {e}")

        all_window_performance[window_name] = window_perf
        all_predictions.append(predictions)

        print(f"    >>> {window_name} 完成: {len(predictions)} 条预测")

        del dataset_sub, models
        gc.collect()

    # =========================================================================
    # [4] 后置过滤与输出
    # =========================================================================
    if not all_predictions:
        raise RuntimeError("所有窗口均未产生有效预测，无法继续")

    print("\n[4] 合并预测结果并执行后置过滤...")
    final_predictions = pd.concat(all_predictions)
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)
    
    # PIT 后置过滤: main_board + 退市 + ST + 次新股
    try:
        before = len(final_predictions)

        # 1. 加载 main_board 股票池
        _inst_path = Path(QLIB_DATA_DIR) / "instruments" / "main_board.txt"
        _board_stocks = set()
        if _inst_path.exists():
            with open(_inst_path) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l:
                        continue
                    _parts = _l.split()
                    if _parts:
                        _board_stocks.add(_parts[0].lower())

        # 2. 加载退市日期映射
        _all_path = Path(QLIB_DATA_DIR) / "instruments" / "all.txt"
        _delist_pit = {}  # {stock: delist_date}
        if _all_path.exists():
            with open(_all_path) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l:
                        continue
                    _parts = _l.split('\t')
                    if len(_parts) >= 3:
                        _code, _list_d, _delist_d = _parts[0].lower(), _parts[1], _parts[2]
                        if _delist_d != '9999-12-31':
                            _delist_pit[_code] = _delist_d

        # 3. 逐日过滤
        _filter_new = CONFIG.get("filter_new_stocks", True)
        _filter_st = CONFIG.get("filter_st", True)
        _filtered_parts = []
        _total_st_removed = 0

        for _date, _day_df in final_predictions.groupby(level="datetime"):
            _dt_str = str(_date)[:10]

            # 3a. 主板 + 未退市
            _day_insts = _day_df.index.get_level_values("instrument").str.lower()
            # [P0修复] pd.Index.isin() 返回 numpy array，需统一处理避免 .values 调用失败
            if _board_stocks:
                _in_board = np.asarray(_day_insts.isin(_board_stocks))
            else:
                _in_board = np.ones(len(_day_insts), dtype=bool)
            _not_delisted = _day_insts.map(
                lambda x: _delist_pit.get(x, "9999-12-31") >= _dt_str
            )
            # _not_delisted 是 pd.Index，.values 返回 numpy array
            _day_df = _day_df[_in_board & np.asarray(_not_delisted)]

            if _day_df.empty:
                continue

            # 3b. ST/次新股过滤
            _codes = _day_df.index.get_level_values("instrument").unique().tolist()
            _filtered_codes = filter_codes_post(
                _codes, _dt_str,
                filter_new_stocks=_filter_new,
                filter_st=_filter_st,
            )
            _total_st_removed += len(_codes) - len(_filtered_codes)

            _filtered_set = set(_filtered_codes)
            _keep = _day_df.index.get_level_values("instrument").str.lower().isin(_filtered_set)
            _day_df = _day_df[_keep]
            if not _day_df.empty:
                _filtered_parts.append(_day_df)

        if _filtered_parts:
            final_predictions = pd.concat(_filtered_parts)
            final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)
        else:
            final_predictions = final_predictions.iloc[0:0]

        after = len(final_predictions)
        print(f"\n  [PIT 过滤] {before} → {after} (剔除 {before-after} 条: 非主板/已退市/ST/次新)")
    except Exception as e:
        print(f"  [警告] 后置过滤异常: {e}，跳过")
    
    if final_predictions.empty:
        raise RuntimeError("后置过滤后无有效预测，请检查数据质量")

    print(f">>> 最终预测范围: "
          f"{final_predictions.index.get_level_values('datetime').min().date()} ~ "
          f"{final_predictions.index.get_level_values('datetime').max().date()}")
    print("    预测样例:")
    print(final_predictions.head(10))

    # =========================================================================
    # [5] 综合评估报告 (P2 增强)
    # =========================================================================
    print(f"\n{'='*60}")
    print("=== 综合评估报告 ===")
    print(f"{'='*60}")

    # 5a. 跨窗口因子稳定性
    print("\n[5a] 跨窗口因子稳定性分析:")
    all_factors_across_windows = []
    for wn, factors in all_window_selected_factors.items():
        all_factors_across_windows.extend(factors)
    factor_freq = pd.Series(all_factors_across_windows).value_counts()
    n_windows = len(all_window_selected_factors)
    stable_factors = factor_freq[factor_freq >= max(2, n_windows // 2)]  # 出现在过半窗口
    print(f"  共 {len(factor_freq)} 个不同因子被选中")
    print(f"  {len(stable_factors)} 个因子在 ≥{max(2, n_windows // 2)} 个窗口中被选中（稳定因子）:")
    for f, cnt in stable_factors.items():
        print(f"    - {f}: {cnt}/{n_windows} 窗口")

    # 5b. 跨窗口性能汇总
    print("\n[5b] 跨窗口性能汇总:")
    for wn, perf in all_window_performance.items():
        ic_str = f", Test IC={perf.get('test_ic', 'N/A'):.4f}" if 'test_ic' in perf else ""
        print(f"  {wn}: {perf['n_factors']} 因子, {perf['n_models']} 模型, "
              f"{perf['n_predictions']} 预测{ic_str}")

    # 5c. IC 衰减分析（仅对最后一个窗口的稳定因子做分析）
    eval_conf = CONFIG.get("evaluation", {})
    if eval_conf.get("ic_decay_horizons") and len(stable_factors) > 0:
        print(f"\n[5c] IC 衰减分析 (>=2 窗口稳定因子)...")
        stable_list = list(stable_factors.index[:min(10, len(stable_factors))])
        last_window_name = list(all_window_selected_factors.keys())[-1]
        last_window = CONFIG["rolling_windows"][-1]
        try:
            decay_df = _compute_ic_decay(
                global_feature_cache, stable_list, all_instruments,
                last_window["train"][0], last_window["train"][1],
                eval_conf["ic_decay_horizons"]
            )
            if not decay_df.empty:
                # 计算每个 horizon 的平均 |IC|
                print(f"  平均 |IC| 衰减曲线:")
                for horizon, row in decay_df.iterrows():
                    valid_vals = row.dropna()
                    if len(valid_vals) > 0:
                        print(f"    {horizon}: mean|IC|={np.mean(np.abs(valid_vals)):.4f} "
                              f"(基于 {len(valid_vals)} 个因子)")
        except Exception as e:
            print(f"  IC 衰减分析失败: {e}")

    # 5d. 行业/市值分组 IC
    if eval_conf.get("industry_exposure_check"):
        print(f"\n[5d] 行业分组 IC 分析...")
        try:
            last_preds = all_predictions[-1]
            if not last_preds.empty:
                # [P1] 尝试从缓存取，取不到则加载
                try:
                    last_window_test = CONFIG["rolling_windows"][-1]
                    test_cache_key = f"{last_window_test['test'][0]}_{last_window_test['test'][1]}"
                    if test_cache_key in test_label_cache:
                        test_label_s = test_label_cache[test_cache_key].rename(label_name)
                    else:
                        test_label_s, _ = _load_window_labels(
                            all_instruments, label_expr,
                            last_window_test["test"][0], last_window_test["test"][1],
                        )
                except Exception:
                    from qlib.data import D
                    label_frames = []
                    for i in range(0, len(all_instruments), 500):
                        _df = D.features(all_instruments[i:i+500], [label_expr],
                                         start_time=segments["test"][0],
                                         end_time=segments["test"][1], freq="day")
                        if _df is not None and not _df.empty:
                            label_frames.append(_df)
                    if label_frames:
                        test_labels = pd.concat(label_frames)
                        test_label_s = test_labels[test_labels.columns[0]].sort_index()
                    else:
                        test_label_s = None

                if test_label_s is not None:
                    ind_ics = _compute_group_ic(last_preds, test_label_s, analysis_type="industry")
                    if ind_ics:
                        print(f"  行业 IC 分布:")
                        for ind_name, ic_val in sorted(ind_ics.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                            print(f"    {ind_name}: IC={ic_val:.4f}")
                    if eval_conf.get("marketcap_group_ic"):
                        mkt_ics = _compute_group_ic(last_preds, test_label_s, analysis_type="marketcap")
                        if mkt_ics:
                            print(f"  市值分组 IC:")
                            for mkt_name, ic_val in sorted(mkt_ics.items()):
                                print(f"    {mkt_name}: IC={ic_val:.4f}")
        except Exception as e:
            print(f"  分组 IC 分析失败: {e}")

    # =========================================================================
    # [P2 增强] 综合量化评估报告
    # =========================================================================
    print(f"\n{'='*60}")
    print("=== [P2 增强评估] 综合量化分析 ===")
    print(f"{'='*60}")

    # P2a. 因子换手率分析
    all_factor_names = []
    for wn, factors in all_window_selected_factors.items():
        all_factor_names.extend(factors)
    unique_factors = set(all_factor_names)
    jaccard_pairs = []
    window_names = sorted(all_window_selected_factors.keys())
    for i in range(len(window_names) - 1):
        set_i = set(all_window_selected_factors[window_names[i]])
        set_j = set(all_window_selected_factors[window_names[i+1]])
        if set_i or set_j:
            jaccard = len(set_i & set_j) / max(len(set_i | set_j), 1)
            jaccard_pairs.append((window_names[i], window_names[i+1], jaccard))

    print(f"\n[P2a] 因子换手率 (Jaccard 相似度):")
    for w1, w2, jac in jaccard_pairs:
        print(f"  {w1} → {w2}: Jaccard={jac:.3f}")

    # P2b. 预测自相关检查
    try:
        if all_predictions and len(all_predictions) >= 2:
            last_preds = all_predictions[-1]
            if not last_preds.empty and 'score' in last_preds.columns:
                scores = last_preds["score"]
                if isinstance(scores.index, pd.MultiIndex):
                    daily_means = scores.groupby(level='datetime').mean()
                    if len(daily_means) >= 5:
                        autocorr = daily_means.autocorr(lag=1)
                        print(f"\n[P2b] 预测自相关 (lag=1): {autocorr:.4f} "
                              f"({'粘性信号⚠' if abs(autocorr) > 0.5 else '健康' if abs(autocorr) < 0.3 else '适中'})")
    except Exception:
        pass

    # P2c. IC 多维度评估（训练集 + 测试集对比）
    print(f"\n[P2c] 跨窗口性能一致性:")
    for wn, perf in all_window_performance.items():
        ic_val = perf.get('test_ic')
        factors = all_window_selected_factors.get(wn, [])
        if ic_val is not None:
            quality = "优秀" if abs(ic_val) >= 0.04 else "良好" if abs(ic_val) >= 0.02 else "一般"
            print(f"  {wn}: IC={ic_val:.4f} ({quality}), {len(factors)} 因子")

    # P2d. 等权基准对比
    print(f"\n[P2d] 模型集成方式对比:")
    has_ewma = any(
        perf.get('model_weights')
        and max(perf['model_weights']) - min(perf['model_weights']) > 0.05
        for perf in all_window_performance.values()
    )
    has_equal = any(
        perf.get('model_weights')
        and max(perf['model_weights']) - min(perf['model_weights']) <= 0.05
        for perf in all_window_performance.values()
    )
    for wn, perf in all_window_performance.items():
        weights = perf.get('model_weights', [])
        if weights:
            print(f"  {wn}: 权重={[f'{w:.3f}' for w in weights]} "
                  f"({'EWMA加权' if max(weights)-min(weights) > 0.05 else '等权'})")

    # =========================================================================
    # [6] 保存输出
    # =========================================================================
    # 保存选中因子清单（取最后一个窗口的因子）
    _selected_dir = Path(__file__).parent
    _selected_path = _selected_dir / "selected_factors_tree.txt"
    _selected_archive_path = _selected_dir / f"selected_factors_tree_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"

    # 保存所有窗口的因子
    with open(_selected_path, "w", encoding="utf-8") as _f:
        for wn, factors in all_window_selected_factors.items():
            _f.write(f"# {wn}\n")
            for _i, _fn in enumerate(factors, 1):
                _f.write(f"{_i}. {_fn}\n")
            _f.write("\n")

    # CSV 格式存档
    _archive_rows = []
    for wn, factors in all_window_selected_factors.items():
        for _i, _fn in enumerate(factors, 1):
            _archive_rows.append({"window": wn, "rank": _i, "factor_name": _fn})
    pd.DataFrame(_archive_rows).to_csv(_selected_archive_path, index=False, encoding="utf-8-sig")
    print(f"\n>>> 因子清单已保存: {_selected_path}")
    print(f">>> 因子存档: {_selected_archive_path}")

    # 保存预测结果
    output_path = os.path.join(os.path.dirname(__file__), "score_tree.csv")
    final_predictions.to_csv(output_path)
    print(f">>> 预测结果已保存: {output_path}")
    print("="*60)
    print("流水线完成")
    print("预测结果已保存到 score_tree.csv (Score 列)")
    print("可将 Score 列导入 Backtrader (src/qlworks/backtest/bt_runner.py) 进行回测")
    print("Backtrader 会根据 Score 排名选取前 N 只股票做多/做空")
    print("="*60)


def _parse_args():
    parser = argparse.ArgumentParser(description="树模型多因子集成训练流水线 (Pro 版)")
    parser.add_argument(
        "--config-source",
        choices=["local", "yaml"],
        default="local",
        help="配置来源: local=使用内置 LOCAL_CONFIG, yaml=读取 scripts/training/configs/ 下的 YAML",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"YAML 配置文件名（不含扩展名），默认 {DEFAULT_YAML_CONFIG_NAME}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ml_pipeline(config_source=args.config_source, config_name=args.config)
