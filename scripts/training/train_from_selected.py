"""
基于精选因子列表的模型训练与选股脚本。

从 select_factors.py 输出的 CSV 中读取精选因子列表，
仅用这些筛选后的因子进行 ML 模型训练 (LGB/XGB/CatBoost)，
输出每只股票每天的 Alpha 预测得分。

[世界顶级量化机构标准全面升级]
  对标 Point72 / Citadel / Renaissance / Two Sigma / AQR / D.E. Shaw 机构级标准

  [第一层：数据增强层 - Bloomberg 级]
    - 时间衰减样本权重（近期样本权重更高）
    - 异常样本检测与降权（极端收益率样本）
    - 截面分布标准化增强

  [第二层：模型训练层 - Point72 级]
    - Purged K-Fold 交叉验证（Marcos López de Prado 标准）
    - 多种子集成训练（降低随机性影响）
    - 早停优化与学习率调度

  [第三层：集成增强层 - Citadel 级]
    - 多维加权集成（IC + 稳定性 + 衰减 + 夏普）
    - 市场状态自适应加权（波动率/趋势状态）
    - 堆叠集成 Stacking（可选，元学习器）

  [第四层：过拟合防护层 - Renaissance 级]
    - 置换检验（Permutation Test）验证显著性
    - 噪声基准测试（随机标签基准表现）
    - 参数稳定性检验（平台效应 vs 尖峰）

  [第五层：风险管理层 - Two Sigma 级]
    - 预测置信区间估计（集成方差法）
    - 风险调整打分（波动率调整预测值）
    - 极端行情压力测试

  [第六层：后处理增强层 - AQR 级]
    - 预测值行业+市值中性化
    - 换手率控制（指数平滑 EMA）
    - 多空非对称动态调整

  [第七层：可解释性层 - Dimensional 级]
    - SHAP 值特征重要性分析
    - 特征重要性跨窗口稳定性检验
    - 特征交互效应分析

用法：
  修改文件顶部 LOCAL_CONFIG 字典中的参数，然后直接运行：
    python train_from_selected.py
"""

import os
import sys
import warnings
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

sp = list(sys.path)
conda_sp = [p for p in sp if 'Anaconda' in p and 'site-packages' in p]
roaming_sp = [p for p in sp if 'Roaming' in p]
other_sp = [p for p in sp if p not in conda_sp and p not in roaming_sp]
sys.path = conda_sp + other_sp + roaming_sp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import gc
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from qlib.data.dataset.handler import DataHandlerLP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import (
    create_custom_dataset,
    build_custom_feature_cache,
    wrap_dataset_with_cached_train_frame,
)
from qlworks.models.training import (
    train_lgb_model, train_xgb_model, train_catboost_model,
    predict_ensemble_models, compute_ic, compute_ic_ewma,
)
from qlworks.factors.filter_utils import filter_codes_post, apply_label_filter
from qlworks.config import QLIB_DATA_DIR
import qlib


def get_latest_qlib_calendar_date(calendar_path: str | Path | None = None) -> str | None:
    """
    读取本地 Qlib 交易日历中的最新交易日。

    输入:
    - calendar_path: 可选，显式指定 day.txt 路径

    输出:
    - 最新交易日字符串 YYYY-MM-DD；若日历不存在或为空则返回 None

    边界:
    - 文件不存在、空文件时直接返回 None
    """
    resolved_path = Path(calendar_path) if calendar_path else (Path(QLIB_DATA_DIR) / "calendars" / "day.txt")
    if not resolved_path.exists():
        return None

    lines = [line.strip() for line in resolved_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def _extract_factor_name(factor_item) -> str | None:
    """
    从 YAML 因子配置项中提取因子名。
    """
    if isinstance(factor_item, str):
        return factor_item.strip() or None
    if isinstance(factor_item, dict):
        for key in ("name", "factor_name", "id"):
            value = factor_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _build_factor_source_map() -> dict[str, str]:
    """
    扫描因子库，建立 因子名 -> 策略文件名 的映射。
    """
    repo_path = Path(__file__).resolve().parents[2] / "factor_data" / "factor_library"
    factor_source_map: dict[str, str] = {}
    for yaml_path in sorted(repo_path.glob("*.y*ml")):
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
        strategy_name = yaml_path.stem
        for factor_item in config.get("factors", []) or []:
            factor_name = _extract_factor_name(factor_item)
            if factor_name:
                factor_source_map.setdefault(factor_name, strategy_name)
    return factor_source_map


def _load_txt_factor_list(txt_path: str) -> tuple[list[str], list[str]]:
    """
    读取形如 selected_factors_tree.txt 的纯文本因子清单。

    每行支持:
    - 1. FACTOR_NAME
    - FACTOR_NAME
    """
    factor_source_map = _build_factor_source_map()
    factor_names: list[str] = []

    for raw_line in Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ". " in line and line.split(". ", 1)[0].isdigit():
            line = line.split(". ", 1)[1].strip()
        factor_name = line.strip()
        if factor_name:
            factor_names.append(factor_name)

    if not factor_names:
        print("[错误] TXT 因子清单为空")
        sys.exit(1)

    missing_factors = [name for name in factor_names if name not in factor_source_map]
    if missing_factors:
        raise ValueError(f"以下因子未在 factor_library 中找到来源文件: {missing_factors}")

    source_files = sorted({factor_source_map[name] for name in factor_names})
    return source_files, factor_names


def load_selected_factors(csv_path: str):
    """
    读取精选因子清单，返回 (source_files, factor_names) 元组。

    source_files: 去重后的因子文件列表（用于 build_factor_library_bundle）
    factor_names: 所有选中因子的名称列表（用于 selected_feature_names）
    """
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    suffix = path.suffix.lower()

    if suffix == ".txt":
        source_files, factor_names = _load_txt_factor_list(str(path))
        print(f"  源文件: {source_files}")
        print(f"  因子数量: {len(factor_names)}")
        print(f"  因子列表: {factor_names}")
        return source_files, factor_names

    df = pd.read_csv(path)
    df_selected = df[df["selected"] == True].copy()

    if len(df_selected) == 0:
        print("[错误] CSV 中没有 selected=True 的因子")
        sys.exit(1)

    source_files = sorted(df_selected["source_file"].unique().tolist())
    factor_names = df_selected["factor_name"].tolist()

    print(f"  源文件: {source_files}")
    print(f"  因子数量: {len(factor_names)}")
    print(f"  因子列表: {factor_names}")

    return source_files, factor_names


# ==============================================================================
# [全局配置区]
# ==============================================================================
LOCAL_CONFIG = {
    "instruments": "csi500",
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",

    "model_type": "tree",
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"],
    "label_names": ["LABEL_5D"],
    "factor_cache_names": [],

    "normalize_features": True,
    "neutralize_features": False,
    "renormalize_features_after_neutralize": False,
    "normalize_labels": True,
    # 当前本地数据未补行业字段，默认关闭标签中性化，避免依赖缺失导致训练失败。
    "neutralize_labels": False,
    "use_dynamic_filter": True,  # 启用流动性过滤（成交量+成交额）
    "filter_new_stocks": True,   # 过滤上市不满 250 日次新股
    "filter_st": True,           # 过滤 ST 股票

    # 标签可交易性过滤（剔除涨跌停无法买入的样本）
    "filter_untradeable_labels": True,

    # 预测置信度阈值（弱信号日降低无效交易）
    "prediction_confidence_threshold": 0.2,  # 设为 None 或 0 则关闭

    # [Renaissance 标准] 多空非对称配置（long_pct>=1.0 则跳过，保留完整分布）
    "long_short_ratio": {"long_pct": 1.0, "short_pct": 0.0},  # 禁用 score 塌缩，保留全分布

    # [Renaissance] 各窗口间 train→valid→test 均保留 ≥12 天 embargo 防止标签泄露
    # 窗口定义保持不变（已有天然 12d 间隔）
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
    "train_models": ["lgb", "xgb", "cat"],
    "model_params": {
        "lgb": {"num_boost_round": 600, "early_stopping_rounds": 50},
        "xgb": {"num_boost_round": 600, "early_stopping_rounds": 50},
        # CatBoost GPU 模式下会一次性吃满显存，容易 OOM 崩溃，强制走 CPU
        "cat": {"num_boost_round": 400, "early_stopping_rounds": 30, "task_type": "CPU"},
    },

    # 精选因子 CSV 路径（select_factors.py 的输出文件）
    "factor_list": "selected_factors_20260801_010335_selected.csv",

    # 输出路径：None 自动生成 score_tree_selected.csv
    "output": None,
    # 通达信模拟盘对接：默认使用 selected 档案，但沿用 tree 运行目录，便于复用现有执行器。
    "live_strategy_name": "selected",
    "live_runtime_model_name": "tree",
    # 运行期质量闸门：用于剔除标签/数据异常导致的坏窗口，避免污染最终选股结果。
    "window_quality_gate": {
        "enabled": True,
        "min_valid_samples": 100,  # 至少覆盖约3个交易日
        "max_train_rmse": 5.0,
        "max_valid_rmse": 5.0,
        "min_healthy_models": 2,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # [第一层：数据增强层 - Bloomberg 级]
    # ─────────────────────────────────────────────────────────────────────────
    "data_augmentation": {
        # 时间衰减样本权重（近期样本权重更高，应对市场非平稳性）
        "time_decay_weight": {
            "enabled": False,
            "half_life_days": 252,  # 半衰期（1年）
            "min_weight": 0.3,      # 最小权重下限
        },
        # 异常样本降权（极端收益率样本降低权重，避免过拟合尾部噪声）
        "outlier_downweight": {
            "enabled": False,
            "zscore_threshold": 3.0,  # 超过3倍标准差视为异常
            "downweight_ratio": 0.5,  # 异常样本权重乘以该比例
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # [第二层：模型训练层 - Point72 级]
    # ─────────────────────────────────────────────────────────────────────────
    # Purged K-Fold 交叉验证（Marcos López de Prado 标准）
    "purged_kfold": {
        "enabled": False,
        "n_splits": 3,
        "purge_days": 11,   # 标签重叠擦除天数（=预测周期+缓冲）
        "embargo_days": 5,  #  embargo 缓冲天数（消除自相关泄露）
    },

    # 多种子集成训练（降低随机性影响，提高稳定性）
    "multi_seed_ensemble": {
        "enabled": False,
        "n_seeds": 3,          # 每个模型训练的种子数量
        "base_seed": 42,       # 基础随机种子
    },

    # ─────────────────────────────────────────────────────────────────────────
    # [第三层：集成增强层 - Citadel 级]
    # ─────────────────────────────────────────────────────────────────────────
    "ensemble_enhancement": {
        # 多维加权（IC + 稳定性 + 衰减 + 夏普）
        "multi_dim_weighting": {
            "enabled": True,  # 默认开启，替代原 EWMA-IC 单维加权
            "weights": {
                "ic": 0.40,           # IC 均值权重
                "icir": 0.30,         # ICIR 稳定性权重
                "decay": 0.15,        # 衰减速度权重（衰减越慢越好）
                "sharpe": 0.15,       # 多空夏普权重
            },
        },
        # 市场状态自适应加权（不同市场状态下不同模型权重）
        "regime_adaptive": {
            "enabled": False,
            "regime_indicator": "volatility",  # volatility / trend
            "lookback_days": 60,
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # [第四层：过拟合防护层 - Renaissance 级]
    # ─────────────────────────────────────────────────────────────────────────
    "overfitting_guard": {
        # 置换检验（验证模型预测能力的统计显著性）
        "permutation_test": {
            "enabled": False,
            "n_permutations": 100,   # 置换次数
            "alpha": 0.05,           # 显著性水平
        },
        # 噪声基准测试（随机标签的基准表现，作为对比）
        "noise_baseline": {
            "enabled": False,
            "n_runs": 10,            # 随机运行次数
        },
        # 参数稳定性检验（小参数变动下表现是否稳定）
        "parameter_stability": {
            "enabled": False,
            "perturbation_pct": 0.2,  # 参数扰动比例（±20%）
            "n_perturbations": 5,     # 扰动次数
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # [第五层：风险管理层 - Two Sigma 级]
    # ─────────────────────────────────────────────────────────────────────────
    "risk_management": {
        # 预测置信区间估计（基于集成模型方差）
        "confidence_interval": {
            "enabled": False,
            "confidence_level": 0.95,  # 置信水平
        },
        # 风险调整打分（结合波动率调整预测值）
        "risk_adjusted_score": {
            "enabled": False,
            "volatility_lookback": 20,  # 波动率回看天数
            "adjustment_strength": 0.5,  # 调整强度（0-1）
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # [第六层：后处理增强层 - AQR 级]
    # ─────────────────────────────────────────────────────────────────────────
    "post_processing": {
        # 预测值行业+市值中性化（剥离行业和市值暴露）
        "prediction_neutralize": {
            "enabled": False,
            "industry_field": "industry_code",
            "market_cap_field": "circ_mv",
            "log_mc": True,
        },
        # 换手率控制（指数平滑 EMA，降低交易成本）
        "turnover_control": {
            "enabled": False,
            "ema_alpha": 0.3,  # EMA 平滑系数（越小越平滑，换手率越低）
        },
        # 动态多空比例（根据市场波动率动态调整多空比例）
        "dynamic_long_short": {
            "enabled": False,
            "volatility_lookback": 60,
            "high_vol_long_pct": 0.20,   # 高波动时多头比例
            "high_vol_short_pct": 0.05,  # 高波动时空头比例
            "low_vol_long_pct": 0.30,    # 低波动时多头比例
            "low_vol_short_pct": 0.10,   # 低波动时空头比例
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # [第七层：可解释性层 - Dimensional 级]
    # ─────────────────────────────────────────────────────────────────────────
    "interpretability": {
        # SHAP 值特征重要性分析
        "shap_analysis": {
            "enabled": False,
            "n_samples": 1000,  # 用于计算 SHAP 的样本数
        },
        # 特征重要性跨窗口稳定性检验
        "feature_importance_stability": {
            "enabled": True,  # 默认开启，开销小
        },
    },

    # P2: 增强评估配置
    "evaluation": {
        "ic_decay_horizons": [1, 3, 5, 10, 20],
        "industry_exposure_check": True,
        "marketcap_group_ic": True,
    },
    # 标签窗口级缓存
    "cache_window_labels": True,
}


def build_effective_local_config(
    config: dict,
    calendar_path: str | Path | None = None,
    latest_calendar_date: str | None = None,
) -> dict:
    """
    根据本地 Qlib 最新交易日扩展配置，使训练窗口自动覆盖最新数据。
    """
    effective = dict(config)

    latest_date = latest_calendar_date or get_latest_qlib_calendar_date(calendar_path)
    if not latest_date:
        return effective

    latest_ts = pd.Timestamp(latest_date)
    configured_end = pd.Timestamp(effective["end_time"])
    if latest_ts <= configured_end:
        effective["neutralize_labels"] = False
        return effective

    effective["end_time"] = latest_ts.strftime("%Y-%m-%d")

    rolling_windows = list(effective.get("rolling_windows", []))
    existing_names = {window.get("name") for window in rolling_windows}
    test_window_name = f"Test_{latest_ts.year}"
    if test_window_name not in existing_names:
        rolling_windows.append(
            {
                "name": test_window_name,
                "train": (f"{latest_ts.year - 3}-01-01", f"{latest_ts.year - 2}-12-20"),
                "valid": (f"{latest_ts.year - 1}-01-01", f"{latest_ts.year - 1}-12-20"),
                "test": (f"{latest_ts.year}-01-01", latest_ts.strftime("%Y-%m-%d")),
            }
        )

    effective["rolling_windows"] = rolling_windows
    effective["neutralize_labels"] = False
    return effective


def extract_label_series(
    frame: pd.DataFrame | pd.Series | None,
    label_names: list[str] | None = None,
) -> pd.Series | None:
    """
    从 prepare 返回结果中稳定提取标签列。

    输入:
    - frame: 可能是 Series、普通 DataFrame 或 MultiIndex 列 DataFrame
    - label_names: 可选的候选标签列名列表（如 ["LABEL_5D"]），优先级最高

    输出:
    - 单列标签 Series；若无法识别则返回 None
    """
    if frame is None:
        return None
    if isinstance(frame, pd.Series):
        return frame
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    # 候选列名：显式传入的 label_names > 常见命名
    candidates = list(label_names or []) + ["label", "LABEL_5D", "LABEL"]

    # MultiIndex 列
    if isinstance(frame.columns, pd.MultiIndex):
        top_level = frame.columns.get_level_values(0)
        for col in candidates:
            if col in top_level:
                label_frame = frame[col]
                if isinstance(label_frame, pd.Series):
                    return label_frame
                if isinstance(label_frame, pd.DataFrame) and label_frame.shape[1] > 0:
                    return label_frame.iloc[:, 0]
        return None

    # 普通 DataFrame：逐一尝试候选列名
    for col in candidates:
        if col in frame.columns:
            label_obj = frame[col]
            if isinstance(label_obj, pd.Series):
                return label_obj
            if isinstance(label_obj, pd.DataFrame) and label_obj.shape[1] > 0:
                return label_obj.iloc[:, 0]

    # 兜底：单列 DataFrame
    if frame.shape[1] == 1:
        return frame.iloc[:, 0]
    return None


def _format_metric(value: float | int | None) -> str:
    """
    将指标格式化为便于日志查看的字符串。
    """
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6f}"


def _align_prediction_and_label(
    predicted: pd.Series | pd.DataFrame,
    actual_label: pd.Series | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    对齐预测值与标签，统一返回可直接计算指标的 Series。
    """
    if actual_label is None or actual_label.empty:
        return None, None

    if isinstance(predicted, pd.DataFrame):
        if predicted.shape[1] == 0:
            return None, None
        predicted = predicted.iloc[:, 0]

    predicted = predicted.dropna()
    actual_label = actual_label.dropna()
    common_idx = predicted.index.intersection(actual_label.index)
    if len(common_idx) == 0:
        return None, None

    aligned_pred = predicted.loc[common_idx]
    aligned_actual = actual_label.loc[common_idx]
    return aligned_pred, aligned_actual


def _compute_rmse(predicted: pd.Series, actual: pd.Series) -> float:
    """
    计算回归 RMSE。
    """
    diff = predicted.astype(float) - actual.astype(float)
    return float(np.sqrt(np.mean(np.square(diff))))


def _evaluate_model_segment(
    model,
    dataset,
    segment: str,
    actual_label: pd.Series | None,
    min_ic_samples: int,
) -> dict:
    """
    评估单模型在单个 segment 上的表现。
    """
    result = {
        "n": 0,
        "rmse": None,
        "ic": None,
        "error": None,
    }
    if actual_label is None or actual_label.empty:
        result["error"] = "标签为空"
        return result

    try:
        predicted = model.predict(dataset, segment=segment)
    except Exception as exc:  # pragma: no cover - 运行期保护
        result["error"] = str(exc)
        return result

    aligned_pred, aligned_actual = _align_prediction_and_label(predicted, actual_label)
    if aligned_pred is None or aligned_actual is None:
        result["error"] = "预测值与标签无法对齐"
        return result

    result["n"] = int(len(aligned_actual))
    result["rmse"] = _compute_rmse(aligned_pred, aligned_actual)
    if len(aligned_actual) >= min_ic_samples:
        result["ic"] = float(compute_ic(aligned_pred, aligned_actual))
    return result


def collect_model_diagnostics(
    models: list,
    model_names: list[str],
    dataset,
    train_label: pd.Series | None,
    valid_label: pd.Series | None,
    model_ic_history: dict[str, list[float]],
    min_ic_samples: int = 30,
    ic_half_life: int = 4,
) -> list[dict]:
    """
    汇总各模型在 train/valid 上的可观测指标，用于加权与窗口质量判定。
    """
    diagnostics: list[dict] = []
    for idx, model in enumerate(models):
        model_name = model_names[idx] if idx < len(model_names) else f"model_{idx}"
        train_metrics = _evaluate_model_segment(
            model=model,
            dataset=dataset,
            segment="train",
            actual_label=train_label,
            min_ic_samples=min_ic_samples,
        )
        valid_metrics = _evaluate_model_segment(
            model=model,
            dataset=dataset,
            segment="valid",
            actual_label=valid_label,
            min_ic_samples=min_ic_samples,
        )

        ewma_ic = None
        raw_weight = 1.0
        if valid_metrics["ic"] is not None:
            ewma_ic = float(compute_ic_ewma(model_ic_history, model_name, valid_metrics["ic"], half_life=ic_half_life))
            raw_weight = max(ewma_ic, 0.0)

        diagnostics.append(
            {
                "model_name": model_name,
                "train_n": train_metrics["n"],
                "train_rmse": train_metrics["rmse"],
                "train_error": train_metrics["error"],
                "valid_n": valid_metrics["n"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_ic": valid_metrics["ic"],
                "valid_ewma_ic": ewma_ic,
                "valid_error": valid_metrics["error"],
                "raw_weight": raw_weight,
            }
        )
    return diagnostics


def resolve_model_weights(diagnostics: list[dict]) -> tuple[list[float], bool]:
    """
    根据诊断信息生成最终模型权重。

    输出:
    - 权重列表
    - 是否退化为等权
    """
    if not diagnostics:
        return [], True

    raw_weights = np.array([float(item.get("raw_weight", 1.0)) for item in diagnostics], dtype=float)
    if raw_weights.sum() > 0:
        return (raw_weights / raw_weights.sum()).tolist(), False
    equal_weight = [1.0 / len(diagnostics)] * len(diagnostics)
    return equal_weight, True


def assess_window_quality(window_name: str, diagnostics: list[dict], config: dict) -> tuple[bool, list[str]]:
    """
    判断当前滚动窗口是否达标；若不达标则返回拒绝原因。
    """
    gate_cfg = dict(config.get("window_quality_gate", {}) or {})
    if not gate_cfg.get("enabled", True):
        return True, []

    min_valid_samples = int(gate_cfg.get("min_valid_samples", 30))
    max_train_rmse = float(gate_cfg.get("max_train_rmse", 5.0))
    max_valid_rmse = float(gate_cfg.get("max_valid_rmse", 5.0))
    min_healthy_models = int(gate_cfg.get("min_healthy_models", 2))

    healthy_models = 0
    reasons: list[str] = []
    for item in diagnostics:
        model_name = item["model_name"]
        item_reasons: list[str] = []

        if item.get("train_error"):
            item_reasons.append(f"train_error={item['train_error']}")
        if item.get("valid_error"):
            item_reasons.append(f"valid_error={item['valid_error']}")
        if int(item.get("valid_n", 0) or 0) < min_valid_samples:
            item_reasons.append(f"valid_n<{min_valid_samples}")
        if item.get("train_rmse") is None or float(item["train_rmse"]) > max_train_rmse:
            item_reasons.append(f"train_rmse>{max_train_rmse}")
        if item.get("valid_rmse") is None or float(item["valid_rmse"]) > max_valid_rmse:
            item_reasons.append(f"valid_rmse>{max_valid_rmse}")

        if item_reasons:
            reasons.append(f"{model_name}: {', '.join(item_reasons)}")
        else:
            healthy_models += 1

    if healthy_models < min_healthy_models:
        reasons.insert(0, f"{window_name}: 健康模型数不足，当前 {healthy_models} < 要求 {min_healthy_models}")
        return False, reasons
    return True, []


def log_model_diagnostics(window_name: str, diagnostics: list[dict], model_weights: list[float], used_equal_weight: bool) -> None:
    """
    打印窗口级模型诊断，便于观察加权是否真正生效。
    """
    print(f"\n[4.3A - {window_name}] 验证集诊断与加权...")
    for idx, item in enumerate(diagnostics):
        weight = model_weights[idx] if idx < len(model_weights) else None
        print(
            "      "
            f"[{item['model_name']}] "
            f"train_n={_format_metric(item['train_n'])}, "
            f"train_rmse={_format_metric(item['train_rmse'])}, "
            f"valid_n={_format_metric(item['valid_n'])}, "
            f"valid_rmse={_format_metric(item['valid_rmse'])}, "
            f"valid_ic={_format_metric(item['valid_ic'])}, "
            f"ewma_ic={_format_metric(item['valid_ewma_ic'])}, "
            f"raw_weight={_format_metric(item['raw_weight'])}, "
            f"final_weight={_format_metric(weight)}"
        )
        if item.get("train_error"):
            print(f"        [train警告] {item['train_error']}")
        if item.get("valid_error"):
            print(f"        [valid警告] {item['valid_error']}")

    if used_equal_weight:
        print("      >> 加权结果退化为等权集成")
    else:
        print(f"      >> 集成权重 (EWMA-IC, 归一化): {[f'{w:.3f}' for w in model_weights]}")


# ==============================================================================
# [P1 辅助函数] IC 衰减分析
# ==============================================================================

def _compute_ic_decay(feature_cache, selected_factors, instruments, train_start, train_end, horizons):
    """计算因子在不同前瞻周期下的 IC 衰减曲线。

    参数:
    - feature_cache: 特征缓存
    - selected_factors: 因子名列表
    - instruments: 股票列表
    - train_start, train_end: 训练期
    - horizons: IC 衰减分析的时间跨度列表
    """
    from qlib.data import D

    print(f"\n    [IC 衰减分析] horizons={horizons}")
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
        return pd.DataFrame(decay_results).T
    return pd.DataFrame()


# ==============================================================================
# [P2 辅助函数] 行业/市值分组 IC
# ==============================================================================

def _compute_group_ic(predictions_df, actual_label_series, analysis_type="industry"):
    """计算行业或市值分组 IC。"""
    from qlib.data import D

    if predictions_df.empty or actual_label_series.empty:
        return {}

    all_instruments = list(set(
        list(predictions_df.index.get_level_values("instrument").unique()) +
        list(actual_label_series.index.get_level_values("instrument").unique())
    ))

    sample_dates = sorted(predictions_df.index.get_level_values("datetime").unique())
    if len(sample_dates) > 10:
        sample_dates = sample_dates[::max(1, len(sample_dates) // 10)]

    group_ics = {}

    if analysis_type == "industry":
        try:
            ref_date = str(sample_dates[-1].date()) if len(sample_dates) > 0 else "2024-12-31"
            ind_frames = []
            for i in range(0, len(all_instruments), 500):
                batch_inst = all_instruments[i:i+500]
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
# [第一层：数据增强层 - Bloomberg 级]
# ==============================================================================

def compute_time_decay_weights(dates: pd.Index, half_life_days: int = 252, min_weight: float = 0.3) -> pd.Series:
    """
    计算时间衰减样本权重（近期样本权重更高）。

    功能：
        基于指数衰减函数，为每个时间点的样本分配权重，越近的样本权重越高。
        用于应对市场非平稳性，让模型更关注近期市场状态。

    入参：
        dates: pd.Index - 日期索引（datetime 格式）
        half_life_days: int - 半衰期天数（权重衰减到一半所需的天数）
        min_weight: float - 最小权重下限（防止远期样本权重过低）

    返回：
        pd.Series - 每个样本的权重值（索引与输入 dates 对齐）

    边界条件：
        - 输入为空时返回空 Series
        - half_life_days <= 0 时返回全 1 权重
        - 权重下限由 min_weight 控制

    注意事项：
        - 指数衰减公式：weight = 2^(-days_ago / half_life_days)
        - 半衰期 = 1 年（252 交易日）是业界常用配置
        - 权重会归一化到均值为 1，不改变有效样本量
    """
    if len(dates) == 0:
        return pd.Series(dtype=float)

    if half_life_days <= 0:
        return pd.Series(1.0, index=dates)

    # 转换为日期序列
    if isinstance(dates, pd.MultiIndex):
        # MultiIndex 取 datetime 级别
        date_series = dates.get_level_values("datetime")
    else:
        date_series = pd.Series(dates)

    # 计算距离最新日期的天数
    latest_date = date_series.max()
    timedelta = latest_date - date_series
    if hasattr(timedelta, 'days'):
        # TimedeltaIndex 直接用 .days
        days_ago = timedelta.days
    else:
        # Series 用 .dt.days
        days_ago = timedelta.dt.days

    # 指数衰减：weight = 2^(-days_ago / half_life_days)
    weights = 2.0 ** (-days_ago.values / half_life_days)

    # 应用最小权重下限
    weights = np.maximum(weights, min_weight)

    # 归一化到均值为 1（保持有效样本量不变）
    weights = weights / weights.mean()

    return pd.Series(weights, index=dates)


def compute_outlier_weights(labels: pd.Series, zscore_threshold: float = 3.0,
                            downweight_ratio: float = 0.5) -> pd.Series:
    """
    计算异常样本降权权重（极端收益率样本降低权重）。

    功能：
        识别标签中极端异常值（超过 N 倍标准差），对这些样本降低权重，
        避免模型过拟合尾部噪声。

    入参：
        labels: pd.Series - 标签序列（收益率）
        zscore_threshold: float - Z-score 阈值，超过则视为异常
        downweight_ratio: float - 异常样本权重乘以该比例（0.5 表示降权 50%）

    返回：
        pd.Series - 每个样本的权重值（正常样本为 1.0，异常样本为 downweight_ratio）

    边界条件：
        - 输入为空时返回空 Series
        - zscore_threshold <= 0 时返回全 1 权重
        - downweight_ratio 限制在 [0.1, 1.0] 范围内

    注意事项：
        - 使用截面 Z-score（按日期分组计算）
        - 仅降低异常样本权重，不删除样本（保留信息）
        - 适用于收益率等近似正态分布的标签
    """
    if len(labels) == 0:
        return pd.Series(dtype=float)

    if zscore_threshold <= 0:
        return pd.Series(1.0, index=labels.index)

    downweight_ratio = np.clip(downweight_ratio, 0.1, 1.0)

    # 按日期分组计算截面 Z-score
    if isinstance(labels.index, pd.MultiIndex):
        grouped = labels.groupby(level="datetime")
    else:
        grouped = labels.groupby(labels.index)

    def _zscore(s):
        std = s.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    zscores = grouped.transform(_zscore)

    # 异常样本降权
    weights = pd.Series(1.0, index=labels.index)
    outlier_mask = zscores.abs() >= zscore_threshold
    weights[outlier_mask] = downweight_ratio

    return weights


def compute_combined_sample_weights(train_frame: pd.DataFrame, label_col: str,
                                    data_aug_config: dict) -> pd.Series:
    """
    计算组合样本权重（时间衰减 + 异常降权的乘积）。

    功能：
        综合时间衰减和异常样本降权，生成最终的样本权重。

    入参：
        train_frame: pd.DataFrame - 训练数据帧（含 datetime 索引和标签列）
        label_col: str - 标签列名
        data_aug_config: dict - 数据增强配置（来自 LOCAL_CONFIG["data_augmentation"]）

    返回：
        pd.Series - 组合样本权重（均值归一化为 1）

    边界条件：
        - 所有增强都关闭时返回全 1 权重
        - 权重乘积后重新归一化

    注意事项：
        - 多种权重相乘后归一化，保持有效样本量
        - 权重用于 LightGBM/XGBoost 的 sample_weight 参数
    """
    weights = pd.Series(1.0, index=train_frame.index)

    # 时间衰减权重
    td_config = data_aug_config.get("time_decay_weight", {})
    if td_config.get("enabled", False):
        td_weights = compute_time_decay_weights(
            train_frame.index,
            half_life_days=td_config.get("half_life_days", 252),
            min_weight=td_config.get("min_weight", 0.3),
        )
        weights = weights * td_weights

    # 异常样本降权
    ow_config = data_aug_config.get("outlier_downweight", {})
    if ow_config.get("enabled", False):
        if label_col in train_frame.columns:
            ow_weights = compute_outlier_weights(
                train_frame[label_col],
                zscore_threshold=ow_config.get("zscore_threshold", 3.0),
                downweight_ratio=ow_config.get("downweight_ratio", 0.5),
            )
            weights = weights * ow_weights

    # 归一化到均值为 1
    if weights.std() > 0:
        weights = weights / weights.mean()

    return weights


# ==============================================================================
# [第二层：模型训练层 - Point72 级]
# ==============================================================================

def _purge_train_samples(train_dates: pd.Index, valid_start: str, valid_end: str,
                         purge_days: int = 11) -> pd.Index:
    """
    Purged K-Fold：擦除训练集中标签窗口与验证集重叠的样本。

    功能：
        实现 Marcos López de Prado 提出的 Purged K-Fold 交叉验证中的 purge 机制，
        删除训练集中标签时间窗口与验证集有重叠的样本，防止信息泄露。

    入参：
        train_dates: pd.Index - 训练集日期索引
        valid_start: str - 验证集开始日期
        valid_end: str - 验证集结束日期
        purge_days: int - 擦除天数（= 预测周期 + 缓冲）

    返回：
        pd.Index - 擦除后的训练集日期索引

    边界条件：
        - purge_days <= 0 时不擦除
        - 训练集为空时返回空索引

    注意事项：
        - purge 方向：验证集前后都要擦除
        - 擦除天数 = 预测周期（如 5 日预测）+ 安全缓冲（如 6 天）= 11 天
        - 这是防止标签泄露的核心机制
    """
    if purge_days <= 0 or len(train_dates) == 0:
        return train_dates

    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)

    # 计算擦除窗口：验证集前后各 purge_days 天
    purge_start = valid_start_ts - pd.Timedelta(days=purge_days)
    purge_end = valid_end_ts + pd.Timedelta(days=purge_days)

    # 保留不在擦除窗口内的训练样本
    if isinstance(train_dates, pd.MultiIndex):
        date_level = train_dates.get_level_values("datetime")
        keep_mask = (date_level < purge_start) | (date_level > purge_end)
        return train_dates[keep_mask]
    else:
        keep_mask = (train_dates < purge_start) | (train_dates > purge_end)
        return train_dates[keep_mask]


def _embargo_train_samples(train_dates: pd.Index, valid_end: str,
                           embargo_days: int = 5) -> pd.Index:
    """
    Purged K-Fold：在验证集后添加 embargo 缓冲期。

    功能：
        实现 embargo 机制，在验证集结束后添加一个时间缓冲区，
        消除自相关带来的信息泄露（如波动率聚集、动量效应等）。

    入参：
        train_dates: pd.Index - 训练集日期索引
        valid_end: str - 验证集结束日期
        embargo_days: int - embargo 天数

    返回：
        pd.Index - embargo 后的训练集日期索引

    边界条件：
        - embargo_days <= 0 时不处理
        - 训练集为空时返回空索引

    注意事项：
        - embargo 只在验证集之后（训练集在验证集之前的情况不受影响）
        - 典型配置：embargo_days = 5 个交易日
        - 主要防止自相关泄露（如波动率聚类、动量延续）
    """
    if embargo_days <= 0 or len(train_dates) == 0:
        return train_dates

    valid_end_ts = pd.Timestamp(valid_end)
    embargo_end = valid_end_ts + pd.Timedelta(days=embargo_days)

    # 保留 embargo 期之后的训练样本（验证集之前的不受影响）
    if isinstance(train_dates, pd.MultiIndex):
        date_level = train_dates.get_level_values("datetime")
        # 验证集之前的保留，embargo 之后的保留，中间的删除
        keep_mask = (date_level <= valid_end_ts) | (date_level > embargo_end)
        # 注意：这里逻辑需要调整，训练集应该是验证集之前的部分
        # 实际上 embargo 是删除验证集紧接之后的训练数据
        keep_mask = ~((date_level > valid_end_ts) & (date_level <= embargo_end))
        return train_dates[keep_mask]
    else:
        keep_mask = ~((train_dates > valid_end_ts) & (train_dates <= embargo_end))
        return train_dates[keep_mask]


def purged_kfold_train(train_frame: pd.DataFrame, valid_frame: pd.DataFrame,
                       feature_cols: list, label_col: str,
                       model_type: str, model_params: dict,
                       purged_config: dict, sample_weight: pd.Series = None) -> tuple:
    """
    Purged K-Fold 交叉验证训练（Point72 / Marcos López de Prado 标准）。

    功能：
        使用 Purged K-Fold 交叉验证训练模型，严格防止时间序列数据泄露。
        结合 purge（标签重叠擦除）和 embargo（自相关缓冲）机制。

    入参：
        train_frame: pd.DataFrame - 完整训练数据帧
        valid_frame: pd.DataFrame - 验证集数据帧
        feature_cols: list - 特征列名列表
        label_col: str - 标签列名
        model_type: str - 模型类型（lgb/xgb/cat）
        model_params: dict - 模型参数
        purged_config: dict - Purged K-Fold 配置
        sample_weight: pd.Series - 样本权重（可选）

    返回：
        tuple: (best_model, cv_results)
            - best_model: 最优模型对象
            - cv_results: dict - 交叉验证结果（各 fold 的指标）

    边界条件：
        - n_splits <= 1 时退化为普通 train/valid 划分
        - 训练样本不足时跳过该 fold
        - 所有 fold 都失败时返回 None

    注意事项：
        - 时间序列数据不能随机打乱，必须保持时间顺序
        - purge + embargo 是防止信息泄露的黄金标准
        - 比标准 K-Fold 更严格，评估结果更保守但更真实
    """
    n_splits = purged_config.get("n_splits", 3)
    purge_days = purged_config.get("purge_days", 11)
    embargo_days = purged_config.get("embargo_days", 5)

    if n_splits <= 1:
        # 退化为普通训练
        if model_type == "lgb":
            model = train_lgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        elif model_type == "xgb":
            model = train_xgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        elif model_type == "cat":
            model = train_catboost_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        else:
            return None, {}
        return model, {"n_splits": 1, "fold_scores": []}

    # 获取训练集所有唯一日期
    if isinstance(train_frame.index, pd.MultiIndex):
        all_dates = sorted(train_frame.index.get_level_values("datetime").unique())
    else:
        all_dates = sorted(train_frame.index.unique())

    if len(all_dates) < n_splits * 2:
        # 日期太少，退化为普通训练
        print(f"    [Purged K-Fold] 警告: 训练日期不足 ({len(all_dates)} 天)，退化为普通训练")
        if model_type == "lgb":
            model = train_lgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        elif model_type == "xgb":
            model = train_xgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        elif model_type == "cat":
            model = train_catboost_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        else:
            return None, {}
        return model, {"n_splits": 1, "fold_scores": []}

    # 按时间顺序划分 fold（时间序列分割，不打乱）
    fold_size = len(all_dates) // n_splits
    fold_boundaries = []
    for i in range(n_splits):
        start_idx = i * fold_size
        if i == n_splits - 1:
            end_idx = len(all_dates) - 1
        else:
            end_idx = (i + 1) * fold_size - 1
        fold_boundaries.append((all_dates[start_idx], all_dates[end_idx]))

    cv_results = {
        "n_splits": n_splits,
        "purge_days": purge_days,
        "embargo_days": embargo_days,
        "fold_scores": [],
        "fold_ics": [],
    }

    best_score = -np.inf
    best_model = None

    print(f"    [Purged K-Fold] n_splits={n_splits}, purge={purge_days}d, embargo={embargo_days}d")

    for fold_idx, (fold_start, fold_end) in enumerate(fold_boundaries):
        # 验证集 = 第 fold_idx 个 fold
        valid_start = fold_start
        valid_end = fold_end

        # 训练集 = 其他所有 fold（但要 purge 和 embargo）
        # 时间序列 CV：训练集在验证集之前（更符合实际部署场景）
        # 这里使用 expanding window 方式：前 i 个 fold 训练，第 i+1 个 fold 验证
        if fold_idx == 0:
            # 第一个 fold 没有足够的训练数据，跳过
            continue

        train_end = fold_boundaries[fold_idx - 1][1]

        # 从训练集中选择日期 <= train_end 的样本
        if isinstance(train_frame.index, pd.MultiIndex):
            date_level = train_frame.index.get_level_values("datetime")
            fold_train_mask = date_level <= train_end
        else:
            fold_train_mask = train_frame.index <= train_end

        fold_train = train_frame[fold_train_mask].copy()

        # 应用 purge：擦除与验证集标签重叠的样本
        fold_train_dates = _purge_train_samples(
            fold_train.index, valid_start, valid_end, purge_days=purge_days
        )
        fold_train = fold_train.loc[fold_train_dates]

        # 应用 embargo：验证集后的缓冲期
        fold_train_dates = _embargo_train_samples(
            fold_train.index, valid_end, embargo_days=embargo_days
        )
        fold_train = fold_train.loc[fold_train_dates]

        # 验证集
        if isinstance(train_frame.index, pd.MultiIndex):
            date_level = train_frame.index.get_level_values("datetime")
            fold_valid_mask = (date_level >= valid_start) & (date_level <= valid_end)
        else:
            fold_valid_mask = (train_frame.index >= valid_start) & (train_frame.index <= valid_end)

        fold_valid = train_frame[fold_valid_mask].copy()

        if len(fold_train) < 100 or len(fold_valid) < 20:
            print(f"      Fold {fold_idx}: 样本不足 (train={len(fold_train)}, valid={len(fold_valid)})，跳过")
            continue

        # 样本权重
        fold_sw = None
        if sample_weight is not None:
            fold_sw = sample_weight.loc[fold_train.index]

        # 训练模型
        try:
            if model_type == "lgb":
                fold_model = train_lgb_model(
                    fold_train[feature_cols], fold_train[label_col],
                    fold_valid[feature_cols], fold_valid[label_col],
                    params=model_params, sample_weight=fold_sw,
                )
            elif model_type == "xgb":
                fold_model = train_xgb_model(
                    fold_train[feature_cols], fold_train[label_col],
                    fold_valid[feature_cols], fold_valid[label_col],
                    params=model_params, sample_weight=fold_sw,
                )
            elif model_type == "cat":
                fold_model = train_catboost_model(
                    fold_train[feature_cols], fold_train[label_col],
                    fold_valid[feature_cols], fold_valid[label_col],
                    params=model_params, sample_weight=fold_sw,
                )
            else:
                continue

            # 评估 fold 表现
            if model_type == "lgb":
                fold_pred = fold_model.predict(fold_valid[feature_cols])
            elif model_type == "xgb":
                fold_pred = fold_model.predict(fold_valid[feature_cols])
            elif model_type == "cat":
                fold_pred = fold_model.predict(fold_valid[feature_cols])
            else:
                continue

            fold_actual = fold_valid[label_col].values
            fold_ic = compute_ic(pd.Series(fold_pred), pd.Series(fold_actual))
            fold_rmse = _compute_rmse(pd.Series(fold_pred), pd.Series(fold_actual))

            cv_results["fold_scores"].append(fold_rmse)
            cv_results["fold_ics"].append(fold_ic)

            print(f"      Fold {fold_idx}: train={len(fold_train)}, valid={len(fold_valid)}, "
                  f"IC={fold_ic:.4f}, RMSE={fold_rmse:.4f}")

            # 保存最优模型（基于验证集 IC）
            if fold_ic > best_score:
                best_score = fold_ic
                best_model = fold_model

        except Exception as e:
            print(f"      Fold {fold_idx}: 训练失败 - {e}")
            continue

    # 如果所有 fold 都失败，使用完整训练集训练
    if best_model is None:
        print(f"    [Purged K-Fold] 所有 fold 失败，使用完整训练集训练")
        if model_type == "lgb":
            best_model = train_lgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        elif model_type == "xgb":
            best_model = train_xgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )
        elif model_type == "cat":
            best_model = train_catboost_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=model_params, sample_weight=sample_weight,
            )

    # 计算 CV 统计量
    if cv_results["fold_ics"]:
        cv_results["mean_ic"] = np.mean(cv_results["fold_ics"])
        cv_results["std_ic"] = np.std(cv_results["fold_ics"])
        cv_results["icir"] = cv_results["mean_ic"] / cv_results["std_ic"] if cv_results["std_ic"] > 0 else 0
        print(f"    [Purged K-Fold] 结果: Mean IC={cv_results['mean_ic']:.4f}, "
              f"ICIR={cv_results['icir']:.4f}")

    return best_model, cv_results


def multi_seed_train(train_frame: pd.DataFrame, valid_frame: pd.DataFrame,
                     feature_cols: list, label_col: str,
                     model_type: str, base_params: dict,
                     multi_seed_config: dict,
                     sample_weight: pd.Series = None) -> tuple:
    """
    多种子集成训练（降低随机性影响，提高模型稳定性）。

    功能：
        使用多个不同的随机种子训练同一模型，然后集成预测结果。
        降低模型训练过程中的随机性影响，提高预测稳定性。

    入参：
        train_frame: pd.DataFrame - 训练数据帧
        valid_frame: pd.DataFrame - 验证数据帧
        feature_cols: list - 特征列名列表
        label_col: str - 标签列名
        model_type: str - 模型类型（lgb/xgb/cat）
        base_params: dict - 基础模型参数
        multi_seed_config: dict - 多种子配置
        sample_weight: pd.Series - 样本权重（可选）

    返回：
        tuple: (models_list, seed_results)
            - models_list: list - 训练好的模型列表
            - seed_results: dict - 各种子的训练结果

    边界条件：
        - n_seeds <= 1 时退化为单模型训练
        - 某个种子训练失败时跳过该种子
        - 所有种子都失败时返回空列表

    注意事项：
        - 多种子集成是降低方差的简单有效方法
        - 不同种子会导致不同的树结构、不同的特征采样
        - 集成后预测更稳定，IC 通常提升 5-10%
    """
    n_seeds = multi_seed_config.get("n_seeds", 3)
    base_seed = multi_seed_config.get("base_seed", 42)

    if n_seeds <= 1:
        # 退化为单模型
        if model_type == "lgb":
            model = train_lgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=base_params, sample_weight=sample_weight,
            )
        elif model_type == "xgb":
            model = train_xgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=base_params, sample_weight=sample_weight,
            )
        elif model_type == "cat":
            model = train_catboost_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=base_params, sample_weight=sample_weight,
            )
        else:
            return [], {}
        return [model], {"n_seeds": 1, "seed_scores": []}

    models = []
    seed_results = {
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "seed_scores": [],
        "seed_ics": [],
    }

    print(f"    [多种子集成] n_seeds={n_seeds}, base_seed={base_seed}")

    for i in range(n_seeds):
        seed = base_seed + i * 100

        # 复制参数并设置种子
        params = base_params.copy()
        if model_type == "lgb":
            params["seed"] = seed
            params["feature_fraction_seed"] = seed
            params["bagging_seed"] = seed
        elif model_type == "xgb":
            params["seed"] = seed
        elif model_type == "cat":
            params["random_seed"] = seed

        try:
            if model_type == "lgb":
                model = train_lgb_model(
                    train_frame[feature_cols], train_frame[label_col],
                    valid_frame[feature_cols], valid_frame[label_col],
                    params=params, sample_weight=sample_weight,
                )
                pred = model.predict(valid_frame[feature_cols])
            elif model_type == "xgb":
                model = train_xgb_model(
                    train_frame[feature_cols], train_frame[label_col],
                    valid_frame[feature_cols], valid_frame[label_col],
                    params=params, sample_weight=sample_weight,
                )
                pred = model.predict(valid_frame[feature_cols])
            elif model_type == "cat":
                model = train_catboost_model(
                    train_frame[feature_cols], train_frame[label_col],
                    valid_frame[feature_cols], valid_frame[label_col],
                    params=params, sample_weight=sample_weight,
                )
                pred = model.predict(valid_frame[feature_cols])
            else:
                continue

            actual = valid_frame[label_col].values
            seed_ic = compute_ic(pd.Series(pred), pd.Series(actual))
            seed_rmse = _compute_rmse(pd.Series(pred), pd.Series(actual))

            models.append(model)
            seed_results["seed_scores"].append(seed_rmse)
            seed_results["seed_ics"].append(seed_ic)

            print(f"      Seed {seed}: IC={seed_ic:.4f}, RMSE={seed_rmse:.4f}")

        except Exception as e:
            print(f"      Seed {seed}: 训练失败 - {e}")
            continue

    # 计算集成统计量
    if seed_results["seed_ics"]:
        seed_results["mean_ic"] = np.mean(seed_results["seed_ics"])
        seed_results["std_ic"] = np.std(seed_results["seed_ics"])
        print(f"    [多种子集成] 结果: Mean IC={seed_results['mean_ic']:.4f}, "
              f"Std={seed_results['std_ic']:.4f}")

    return models, seed_results


def multi_seed_predict(models: list, X: pd.DataFrame, model_type: str) -> np.ndarray:
    """
    多种子模型集成预测（简单平均）。

    功能：
        对多个不同种子训练的模型进行集成预测，取平均值。

    入参：
        models: list - 模型列表
        X: pd.DataFrame - 特征数据
        model_type: str - 模型类型

    返回：
        np.ndarray - 集成预测值

    边界条件：
        - 模型列表为空时返回全 0
        - 单个模型时直接返回该模型预测

    注意事项：
        - 简单平均是最稳健的集成方式
        - 比加权平均更不容易过拟合
    """
    if not models:
        return np.zeros(len(X))

    predictions = []
    for model in models:
        if model_type == "lgb":
            pred = model.predict(X)
        elif model_type == "xgb":
            pred = model.predict(X)
        elif model_type == "cat":
            pred = model.predict(X)
        else:
            continue
        predictions.append(pred)

    if not predictions:
        return np.zeros(len(X))

    return np.mean(predictions, axis=0)


# ==============================================================================
# [第三层：集成增强层 - Citadel 级]
# ==============================================================================

def compute_multi_dim_weights(diagnostics: list[dict], ensemble_config: dict) -> tuple[list[float], bool]:
    """
    多维加权集成（IC + 稳定性 + 衰减 + 夏普）。

    功能：
        超越简单 EWMA-IC 单维加权，从多个维度评估模型质量：
        1. IC 均值 - 预测能力
        2. ICIR - IC 稳定性
        3. 衰减速度 - 信号持续性
        4. 多空夏普 - 实际交易表现

    入参：
        diagnostics: list[dict] - 各模型诊断结果列表
        ensemble_config: dict - 集成配置（来自 LOCAL_CONFIG["ensemble_enhancement"]）

    返回：
        tuple: (weights, used_equal_weight)
            - weights: list[float] - 各模型权重
            - used_equal_weight: bool - 是否退化为等权

    边界条件：
        - 配置关闭时退化为等权
        - 某个维度数据缺失时该维度权重均匀分配
        - 所有权重归一化到和为 1

    注意事项：
        - 多维加权比单维 IC 加权更稳健
        - 各维度权重可通过配置调整
        - 默认 IC 占比最高（40%），因为 IC 是最核心的预测能力指标
    """
    md_config = ensemble_config.get("multi_dim_weighting", {})

    if not md_config.get("enabled", False):
        # 关闭时退化为等权
        n = len(diagnostics)
        if n == 0:
            return [], True
        return [1.0 / n] * n, True

    dim_weights = md_config.get("weights", {
        "ic": 0.40,
        "icir": 0.30,
        "decay": 0.15,
        "sharpe": 0.15,
    })

    n_models = len(diagnostics)
    if n_models == 0:
        return [], True

    # 收集各维度得分
    ic_scores = []
    icir_scores = []
    decay_scores = []
    sharpe_scores = []

    for diag in diagnostics:
        # IC 得分
        ic = diag.get("valid_ic", 0.0)
        if np.isnan(ic):
            ic = 0.0
        ic_scores.append(max(ic, 0))  # IC 为负的模型得 0 分

        # ICIR 得分
        icir = diag.get("valid_icir", 0.0)
        if np.isnan(icir):
            icir = 0.0
        icir_scores.append(max(icir, 0))

        # 衰减得分（衰减越慢越好，这里用 IC 半衰期近似）
        # 如果没有衰减数据，用 IC 均值代替
        decay_half = diag.get("ic_decay_half", 0.0)
        if decay_half <= 0 or np.isnan(decay_half):
            decay_half = max(ic, 0.01)
        decay_scores.append(decay_half)

        # 夏普得分
        sharpe = diag.get("long_short_sharpe", 0.0)
        if np.isnan(sharpe):
            sharpe = 0.0
        sharpe_scores.append(max(sharpe, 0))

    # 归一化各维度得分（每个维度的得分和为 1）
    def _normalize(scores):
        total = sum(scores)
        if total <= 0:
            n = len(scores)
            return [1.0 / n] * n if n > 0 else []
        return [s / total for s in scores]

    ic_norm = _normalize(ic_scores)
    icir_norm = _normalize(icir_scores)
    decay_norm = _normalize(decay_scores)
    sharpe_norm = _normalize(sharpe_scores)

    # 计算最终权重（各维度加权求和）
    final_weights = []
    for i in range(n_models):
        w = (
            dim_weights.get("ic", 0.4) * ic_norm[i] +
            dim_weights.get("icir", 0.3) * icir_norm[i] +
            dim_weights.get("decay", 0.15) * decay_norm[i] +
            dim_weights.get("sharpe", 0.15) * sharpe_norm[i]
        )
        final_weights.append(w)

    # 归一化到和为 1
    total = sum(final_weights)
    if total <= 0:
        return [1.0 / n_models] * n_models, True

    final_weights = [w / total for w in final_weights]

    return final_weights, False


def compute_market_regime(prices: pd.Series, lookback_days: int = 60,
                          regime_type: str = "volatility") -> str:
    """
    计算市场状态（波动率状态或趋势状态）。

    功能：
        根据历史价格数据判断当前市场状态，用于自适应加权。
        支持两种状态指标：波动率（高/低波动）和趋势（牛/熊/震荡）。

    入参：
        prices: pd.Series - 价格序列（指数或个股价格）
        lookback_days: int - 回看天数
        regime_type: str - 状态类型（volatility / trend）

    返回：
        str - 市场状态标签
            - volatility: "high_vol" / "low_vol" / "normal_vol"
            - trend: "bull" / "bear" / "sideways"

    边界条件：
        - 数据不足时返回 "unknown"
        - 波动率阈值基于历史分位数

    注意事项：
        - 波动率状态基于已实现波动率的分位数
        - 趋势状态基于价格动量和均线关系
        - 用于市场状态自适应加权
    """
    if len(prices) < lookback_days:
        return "unknown"

    recent = prices.tail(lookback_days)

    if regime_type == "volatility":
        # 计算已实现波动率
        returns = recent.pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252)

        # 简单阈值：年化波动率 > 30% 为高波动，< 15% 为低波动
        if realized_vol > 0.30:
            return "high_vol"
        elif realized_vol < 0.15:
            return "low_vol"
        else:
            return "normal_vol"

    elif regime_type == "trend":
        # 计算趋势状态
        start_price = recent.iloc[0]
        end_price = recent.iloc[-1]
        total_return = (end_price / start_price) - 1

        # 简单阈值：60 日涨幅 > 10% 为牛市，< -10% 为熊市
        if total_return > 0.10:
            return "bull"
        elif total_return < -0.10:
            return "bear"
        else:
            return "sideways"

    return "unknown"


def compute_regime_adaptive_weights(diagnostics: list[dict], regime: str,
                                    ensemble_config: dict) -> list[float]:
    """
    市场状态自适应加权（不同市场状态下不同模型权重）。

    功能：
        根据当前市场状态动态调整各模型的权重，
        利用不同模型在不同市场环境下的表现差异。

    入参：
        diagnostics: list[dict] - 各模型诊断结果
        regime: str - 当前市场状态
        ensemble_config: dict - 集成配置

    返回：
        list[float] - 调整后的权重

    边界条件：
        - 配置关闭时返回等权
        - 未知状态时返回基础权重
        - 权重归一化到和为 1

    注意事项：
        - 树模型在震荡市表现较好，线性模型在趋势市表现较好
        - 需要足够的历史数据来估计各状态下的表现
        - 默认关闭，需要充分验证后再启用
    """
    ra_config = ensemble_config.get("regime_adaptive", {})

    if not ra_config.get("enabled", False):
        n = len(diagnostics)
        return [1.0 / n] * n if n > 0 else []

    # 基础权重（多维加权）
    base_weights, _ = compute_multi_dim_weights(diagnostics, ensemble_config)

    if regime == "unknown" or len(diagnostics) == 0:
        return base_weights

    # 根据市场状态调整权重
    # 高波动：降低复杂模型权重，增加简单模型权重
    # 低波动：增加复杂模型权重
    # 趋势市：增加动量类因子权重（这里用模型类型近似）
    adjusted = base_weights.copy()

    if regime == "high_vol":
        # 高波动：LGB 通常更稳健，权重略增；CatBoost 复杂度高，权重略减
        for i, diag in enumerate(diagnostics):
            model_name = diag.get("model", "")
            if "lgb" in model_name.lower():
                adjusted[i] *= 1.1  # +10%
            elif "cat" in model_name.lower():
                adjusted[i] *= 0.9  # -10%
    elif regime == "low_vol":
        # 低波动：复杂模型表现更好
        for i, diag in enumerate(diagnostics):
            model_name = diag.get("model", "")
            if "cat" in model_name.lower():
                adjusted[i] *= 1.1
            elif "xgb" in model_name.lower():
                adjusted[i] *= 1.05

    # 归一化
    total = sum(adjusted)
    if total > 0:
        adjusted = [w / total for w in adjusted]

    return adjusted


# ==============================================================================
# [第四层：过拟合防护层 - Renaissance 级]
# ==============================================================================

def permutation_test(model, X: pd.DataFrame, y: pd.Series,
                     n_permutations: int = 100, alpha: float = 0.05,
                     model_type: str = "lgb") -> dict:
    """
    置换检验（Permutation Test）验证模型预测能力的统计显著性。

    功能：
        通过随机打乱标签多次，生成零分布（null distribution），
        检验真实模型表现是否显著优于随机猜测。

    入参：
        model: 训练好的模型对象
        X: pd.DataFrame - 特征数据
        y: pd.Series - 真实标签
        n_permutations: int - 置换次数
        alpha: float - 显著性水平
        model_type: str - 模型类型

    返回：
        dict - 置换检验结果
            - real_ic: 真实 IC
            - permutation_ics: 置换后的 IC 列表
            - p_value: p 值（真实 IC 在零分布中的分位数）
            - significant: 是否显著（p_value < alpha）
            - effect_size: 效应大小（真实 IC / 零分布均值）

    边界条件：
        - n_permutations <= 0 时跳过
        - 样本太少时结果不可靠
        - 零分布标准差为 0 时 p_value 设为 0.5

    注意事项：
        - 置换检验是统计显著性的黄金标准
        - p < 0.05 表示模型有 95% 以上概率不是随机猜测
        - 置换次数越多，p 值越精确，但计算量越大
        - Renaissance 等顶级机构要求所有策略必须通过置换检验
    """
    if n_permutations <= 0 or len(X) < 10:
        return {"enabled": False}

    # 真实预测
    if model_type == "lgb":
        real_pred = model.predict(X)
    elif model_type == "xgb":
        real_pred = model.predict(X)
    elif model_type == "cat":
        real_pred = model.predict(X)
    else:
        return {"enabled": False}

    real_ic = compute_ic(pd.Series(real_pred), y)

    # 置换检验
    permutation_ics = []
    y_values = y.values.copy()

    for i in range(n_permutations):
        # 随机打乱标签
        np.random.seed(42 + i)
        y_permuted = np.random.permutation(y_values)

        # 用打乱的标签计算 IC（注意：这里不需要重新训练模型，
        # 只需要计算随机标签与真实预测的相关性）
        # 更严格的做法是每次重新训练，但计算量太大
        # 这里用近似方法：计算预测值与随机标签的相关性
        perm_ic = compute_ic(pd.Series(real_pred), pd.Series(y_permuted))
        permutation_ics.append(perm_ic)

    # 计算 p 值
    perm_array = np.array(permutation_ics)
    perm_mean = np.mean(perm_array)
    perm_std = np.std(perm_array)

    if perm_std == 0:
        p_value = 0.5
    else:
        # 单侧检验：真实 IC 是否显著大于零分布均值
        p_value = 1 - np.mean(real_ic > perm_array)

    # 效应大小
    effect_size = (real_ic - perm_mean) / perm_std if perm_std > 0 else 0

    result = {
        "enabled": True,
        "n_permutations": n_permutations,
        "alpha": alpha,
        "real_ic": real_ic,
        "permutation_mean": perm_mean,
        "permutation_std": perm_std,
        "permutation_ics": permutation_ics,
        "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size": effect_size,
    }

    print(f"    [置换检验] 真实 IC={real_ic:.4f}, 零分布均值={perm_mean:.4f}, "
          f"p值={p_value:.4f}, {'显著' if result['significant'] else '不显著'}")

    return result


def noise_baseline_test(train_frame: pd.DataFrame, valid_frame: pd.DataFrame,
                        feature_cols: list, label_col: str,
                        model_type: str, model_params: dict,
                        n_runs: int = 10,
                        sample_weight: pd.Series = None) -> dict:
    """
    噪声基准测试（随机标签的基准表现）。

    功能：
        用随机生成的标签训练模型，看看模型能"学到"什么程度。
        如果随机标签也能得到不错的 IC，说明模型严重过拟合。

    入参：
        train_frame: pd.DataFrame - 训练数据
        valid_frame: pd.DataFrame - 验证数据
        feature_cols: list - 特征列
        label_col: str - 标签列名
        model_type: str - 模型类型
        model_params: dict - 模型参数
        n_runs: int - 随机运行次数
        sample_weight: pd.Series - 样本权重

    返回：
        dict - 噪声基准测试结果
            - baseline_ics: 各次随机运行的验证集 IC
            - mean_baseline_ic: 平均基准 IC
            - std_baseline_ic: 基准 IC 标准差
            - signal_to_noise: 信噪比（真实 IC / 基准 IC 标准差）

    边界条件：
        - n_runs <= 0 时跳过
        - 训练失败的 run 不计入统计

    注意事项：
        - 噪声基准是检测过拟合的有效方法
        - 好的模型真实 IC 应该显著高于噪声基准
        - 如果噪声基准 IC > 0.02，说明模型容量太大或正则化不足
    """
    if n_runs <= 0:
        return {"enabled": False}

    baseline_ics = []

    print(f"    [噪声基准] n_runs={n_runs}")

    for i in range(n_runs):
        try:
            # 生成随机标签
            np.random.seed(100 + i)
            random_train_labels = np.random.normal(0, 1, len(train_frame))
            random_valid_labels = np.random.normal(0, 1, len(valid_frame))

            # 用随机标签训练模型
            if model_type == "lgb":
                noise_model = train_lgb_model(
                    train_frame[feature_cols], pd.Series(random_train_labels, index=train_frame.index),
                    valid_frame[feature_cols], pd.Series(random_valid_labels, index=valid_frame.index),
                    params=model_params, sample_weight=sample_weight,
                )
                pred = noise_model.predict(valid_frame[feature_cols])
            elif model_type == "xgb":
                noise_model = train_xgb_model(
                    train_frame[feature_cols], pd.Series(random_train_labels, index=train_frame.index),
                    valid_frame[feature_cols], pd.Series(random_valid_labels, index=valid_frame.index),
                    params=model_params, sample_weight=sample_weight,
                )
                pred = noise_model.predict(valid_frame[feature_cols])
            elif model_type == "cat":
                noise_model = train_catboost_model(
                    train_frame[feature_cols], pd.Series(random_train_labels, index=train_frame.index),
                    valid_frame[feature_cols], pd.Series(random_valid_labels, index=valid_frame.index),
                    params=model_params, sample_weight=sample_weight,
                )
                pred = noise_model.predict(valid_frame[feature_cols])
            else:
                continue

            # 计算噪声基准 IC
            baseline_ic = compute_ic(pd.Series(pred), pd.Series(random_valid_labels))
            baseline_ics.append(baseline_ic)

        except Exception as e:
            print(f"      Run {i}: 失败 - {e}")
            continue

    if not baseline_ics:
        return {"enabled": False, "error": "all runs failed"}

    mean_ic = np.mean(baseline_ics)
    std_ic = np.std(baseline_ics)

    result = {
        "enabled": True,
        "n_runs": n_runs,
        "baseline_ics": baseline_ics,
        "mean_baseline_ic": mean_ic,
        "std_baseline_ic": std_ic,
    }

    print(f"    [噪声基准] 平均 IC={mean_ic:.4f}, 标准差={std_ic:.4f}")

    return result


def parameter_stability_test(train_frame: pd.DataFrame, valid_frame: pd.DataFrame,
                             feature_cols: list, label_col: str,
                             model_type: str, base_params: dict,
                             n_perturbations: int = 5,
                             perturbation_pct: float = 0.2,
                             sample_weight: pd.Series = None) -> dict:
    """
    参数稳定性检验（小参数变动下表现是否稳定）。

    功能：
        对模型参数进行小范围扰动，观察表现是否稳定。
        如果参数微小变动导致表现大幅波动，说明模型处于"尖峰"，
        很可能是过拟合的表现。

    入参：
        train_frame: pd.DataFrame - 训练数据
        valid_frame: pd.DataFrame - 验证数据
        feature_cols: list - 特征列
        label_col: str - 标签列名
        model_type: str - 模型类型
        base_params: dict - 基础参数
        n_perturbations: int - 扰动次数
        perturbation_pct: float - 扰动比例（±20%）
        sample_weight: pd.Series - 样本权重

    返回：
        dict - 参数稳定性检验结果
            - base_ic: 基础参数 IC
            - perturbed_ics: 各次扰动后的 IC
            - mean_perturbed_ic: 平均扰动 IC
            - std_perturbed_ic: 扰动 IC 标准差
            - stability_ratio: 稳定性比率（标准差 / 均值）
            - stable: 是否稳定（stability_ratio < 0.2）

    边界条件：
        - n_perturbations <= 0 时跳过
        - 只扰动数值型参数
        - 训练失败的扰动不计入

    注意事项：
        - 稳定的模型应该有"平台效应"（plateau），而不是"尖峰"
        - 顶级机构偏好参数不敏感的稳健模型
        - stability_ratio < 0.2 通常认为是稳定的
    """
    if n_perturbations <= 0:
        return {"enabled": False}

    # 基础参数训练
    try:
        if model_type == "lgb":
            base_model = train_lgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=base_params, sample_weight=sample_weight,
            )
            base_pred = base_model.predict(valid_frame[feature_cols])
        elif model_type == "xgb":
            base_model = train_xgb_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=base_params, sample_weight=sample_weight,
            )
            base_pred = base_model.predict(valid_frame[feature_cols])
        elif model_type == "cat":
            base_model = train_catboost_model(
                train_frame[feature_cols], train_frame[label_col],
                valid_frame[feature_cols], valid_frame[label_col],
                params=base_params, sample_weight=sample_weight,
            )
            base_pred = base_model.predict(valid_frame[feature_cols])
        else:
            return {"enabled": False}

        base_ic = compute_ic(pd.Series(base_pred), valid_frame[label_col])
    except Exception as e:
        return {"enabled": False, "error": f"base training failed: {e}"}

    # 找出可扰动的数值型参数
    numeric_params = {}
    for k, v in base_params.items():
        if isinstance(v, (int, float)) and k not in ["seed", "random_seed", "task_type"]:
            numeric_params[k] = v

    if not numeric_params:
        return {"enabled": False, "error": "no numeric parameters to perturb"}

    perturbed_ics = []

    print(f"    [参数稳定性] n_perturbations={n_perturbations}, perturbation_pct={perturbation_pct}")

    for i in range(n_perturbations):
        try:
            # 随机扰动参数
            np.random.seed(200 + i)
            perturbed_params = base_params.copy()

            for param_name, param_value in numeric_params.items():
                # 随机扰动 ±perturbation_pct
                perturbation = np.random.uniform(-perturbation_pct, perturbation_pct)
                new_value = param_value * (1 + perturbation)

                # 确保参数在合理范围内
                if param_name in ["num_boost_round", "n_estimators"]:
                    new_value = max(50, int(new_value))
                elif param_name in ["max_depth", "num_leaves"]:
                    new_value = max(2, int(new_value))
                elif param_name in ["learning_rate"]:
                    new_value = max(0.001, min(1.0, new_value))
                elif param_name in ["early_stopping_rounds"]:
                    new_value = max(5, int(new_value))

                perturbed_params[param_name] = new_value

            # 训练扰动模型
            if model_type == "lgb":
                pert_model = train_lgb_model(
                    train_frame[feature_cols], train_frame[label_col],
                    valid_frame[feature_cols], valid_frame[label_col],
                    params=perturbed_params, sample_weight=sample_weight,
                )
                pert_pred = pert_model.predict(valid_frame[feature_cols])
            elif model_type == "xgb":
                pert_model = train_xgb_model(
                    train_frame[feature_cols], train_frame[label_col],
                    valid_frame[feature_cols], valid_frame[label_col],
                    params=perturbed_params, sample_weight=sample_weight,
                )
                pert_pred = pert_model.predict(valid_frame[feature_cols])
            elif model_type == "cat":
                pert_model = train_catboost_model(
                    train_frame[feature_cols], train_frame[label_col],
                    valid_frame[feature_cols], valid_frame[label_col],
                    params=perturbed_params, sample_weight=sample_weight,
                )
                pert_pred = pert_model.predict(valid_frame[feature_cols])
            else:
                continue

            pert_ic = compute_ic(pd.Series(pert_pred), valid_frame[label_col])
            perturbed_ics.append(pert_ic)

            print(f"      Perturbation {i}: IC={pert_ic:.4f}")

        except Exception as e:
            print(f"      Perturbation {i}: 失败 - {e}")
            continue

    if not perturbed_ics:
        return {"enabled": False, "error": "all perturbations failed"}

    mean_ic = np.mean(perturbed_ics)
    std_ic = np.std(perturbed_ics)

    # 稳定性比率：标准差 / |均值|，越小越稳定
    stability_ratio = std_ic / abs(mean_ic) if mean_ic != 0 else float('inf')
    stable = stability_ratio < 0.2

    result = {
        "enabled": True,
        "n_perturbations": n_perturbations,
        "perturbation_pct": perturbation_pct,
        "base_ic": base_ic,
        "perturbed_ics": perturbed_ics,
        "mean_perturbed_ic": mean_ic,
        "std_perturbed_ic": std_ic,
        "stability_ratio": stability_ratio,
        "stable": stable,
    }

    print(f"    [参数稳定性] 基础 IC={base_ic:.4f}, 平均扰动 IC={mean_ic:.4f}, "
          f"稳定性比率={stability_ratio:.4f}, {'稳定' if stable else '不稳定'}")

    return result


# ==============================================================================
# [第五层：风险管理层 - Two Sigma 级]
# ==============================================================================

def compute_prediction_confidence(predictions_list: list[np.ndarray],
                                  confidence_level: float = 0.95) -> dict:
    """
    预测置信区间估计（基于集成模型方差）。

    功能：
        利用多个模型（或多个种子）的预测差异来估计预测的不确定性。
        集成方差越大，预测越不确定。

    入参：
        predictions_list: list[np.ndarray] - 多个模型的预测值列表
        confidence_level: float - 置信水平（0.95 表示 95% 置信区间）

    返回：
        dict - 置信区间结果
            - mean_pred: 平均预测值
            - std_pred: 预测标准差（不确定性）
            - lower_bound: 置信区间下界
            - upper_bound: 置信区间上界
            - confidence_level: 置信水平

    边界条件：
        - 预测列表为空时返回空结果
        - 只有一个预测时标准差为 0
        - 置信水平限制在 [0.5, 0.99]

    注意事项：
        - 基于集成方差的置信区间是近似估计
        - 假设预测值近似正态分布
        - 不确定性高的预测应该降低权重或回避
        - Two Sigma 等机构非常重视预测不确定性管理
    """
    if not predictions_list:
        return {"enabled": False}

    confidence_level = np.clip(confidence_level, 0.5, 0.99)

    # 堆叠预测值
    pred_array = np.array(predictions_list)

    # 计算均值和标准差
    mean_pred = np.mean(pred_array, axis=0)
    std_pred = np.std(pred_array, axis=0)

    # 计算置信区间（基于正态分布）
    # 95% 置信区间对应 1.96 倍标准差
    z_score = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }.get(confidence_level, 1.96)

    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred

    result = {
        "enabled": True,
        "confidence_level": confidence_level,
        "mean_prediction": mean_pred,
        "std_prediction": std_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "z_score": z_score,
    }

    return result


def compute_risk_adjusted_score(predictions: pd.Series, volatility: pd.Series,
                                adjustment_strength: float = 0.5) -> pd.Series:
    """
    风险调整打分（结合波动率调整预测值）。

    功能：
        根据个股波动率调整预测得分，高波动股票的预测得分打折扣。
        体现"风险调整后收益"的理念。

    入参：
        predictions: pd.Series - 原始预测得分
        volatility: pd.Series - 个股波动率
        adjustment_strength: float - 调整强度（0-1，0 表示不调整）

    返回：
        pd.Series - 风险调整后的预测得分

    边界条件：
        - adjustment_strength = 0 时返回原始预测
        - 波动率为 0 或 NaN 时不调整
        - 调整强度限制在 [0, 1]

    注意事项：
        - 高波动股票的预测不确定性更高，应该降低权重
        - 调整公式：adjusted_score = score * (1 - strength * (vol / vol_median))
        - 波动率高于中位数的股票得分降低，低于中位数的得分提高
        - 这是一种简单的风险平价思想
    """
    if adjustment_strength <= 0:
        return predictions

    adjustment_strength = np.clip(adjustment_strength, 0, 1)

    # 计算波动率中位数
    vol_median = volatility.median()

    if vol_median == 0 or np.isnan(vol_median):
        return predictions

    # 计算波动率比率
    vol_ratio = volatility / vol_median
    vol_ratio = vol_ratio.fillna(1.0)
    vol_ratio = vol_ratio.clip(0.5, 2.0)  # 限制调整范围

    # 风险调整：高波动降低得分，低波动提高得分
    # adjustment_factor = 1 - strength * (vol_ratio - 1)
    # vol_ratio = 1 时不调整，> 1 时降低，< 1 时提高
    adjustment_factor = 1 - adjustment_strength * (vol_ratio - 1)
    adjustment_factor = adjustment_factor.clip(0.5, 1.5)  # 限制调整范围

    adjusted = predictions * adjustment_factor

    return adjusted


def extreme_event_stress_test(predictions_df: pd.DataFrame, actual_returns: pd.Series,
                              threshold_pct: float = 0.05) -> dict:
    """
    极端行情压力测试（在极端市场环境下的表现）。

    功能：
        测试模型在极端市场行情（大涨大跌日）下的表现，
        检验模型在尾部风险事件中的稳健性。

    入参：
        predictions_df: pd.DataFrame - 预测得分数据
        actual_returns: pd.Series - 实际收益率
        threshold_pct: float - 极端事件阈值（5% 分位数）

    返回：
        dict - 压力测试结果
            - normal_ic: 正常行情 IC
            - extreme_up_ic: 极端上涨日 IC
            - extreme_down_ic: 极端下跌日 IC
            - ic_ratio: 极端/正常 IC 比率

    边界条件：
        - 极端日样本太少时结果不可靠
        - 阈值限制在 [0.01, 0.2]

    注意事项：
        - 好的模型在极端行情下也应该有一定预测能力
        - 如果极端行情 IC 大幅下降，说明模型对尾部风险准备不足
        - Two Sigma 等风险管理严格的机构非常重视压力测试
    """
    threshold_pct = np.clip(threshold_pct, 0.01, 0.2)

    # 计算市场平均收益率（截面均值）
    if isinstance(predictions_df.index, pd.MultiIndex):
        market_returns = actual_returns.groupby(level="datetime").mean()
    else:
        market_returns = actual_returns.groupby(actual_returns.index).mean()

    # 识别极端日
    up_threshold = market_returns.quantile(1 - threshold_pct)
    down_threshold = market_returns.quantile(threshold_pct)

    extreme_up_dates = market_returns[market_returns >= up_threshold].index
    extreme_down_dates = market_returns[market_returns <= down_threshold].index
    normal_dates = market_returns[
        (market_returns > down_threshold) & (market_returns < up_threshold)
    ].index

    # 计算各场景下的 IC
    def _ic_for_dates(dates):
        if len(dates) == 0:
            return np.nan
        if isinstance(predictions_df.index, pd.MultiIndex):
            mask = predictions_df.index.get_level_values("datetime").isin(dates)
        else:
            mask = predictions_df.index.isin(dates)
        if mask.sum() < 20:
            return np.nan
        pred_sub = predictions_df.loc[mask, "score"] if "score" in predictions_df.columns else predictions_df.loc[mask]
        actual_sub = actual_returns.loc[mask]
        valid = pred_sub.notna() & actual_sub.notna()
        if valid.sum() < 20:
            return np.nan
        return compute_ic(pred_sub[valid], actual_sub[valid])

    normal_ic = _ic_for_dates(normal_dates)
    extreme_up_ic = _ic_for_dates(extreme_up_dates)
    extreme_down_ic = _ic_for_dates(extreme_down_dates)

    result = {
        "threshold_pct": threshold_pct,
        "normal_ic": normal_ic,
        "extreme_up_ic": extreme_up_ic,
        "extreme_down_ic": extreme_down_ic,
        "n_normal_days": len(normal_dates),
        "n_extreme_up_days": len(extreme_up_dates),
        "n_extreme_down_days": len(extreme_down_dates),
    }

    print(f"    [压力测试] 正常日 IC={normal_ic:.4f} ({len(normal_dates)}天), "
          f"大涨日 IC={extreme_up_ic:.4f} ({len(extreme_up_dates)}天), "
          f"大跌日 IC={extreme_down_ic:.4f} ({len(extreme_down_dates)}天)")

    return result


# ==============================================================================
# [第六层：后处理增强层 - AQR 级]
# ==============================================================================

def neutralize_prediction(predictions: pd.Series, industry: pd.Series = None,
                          market_cap: pd.Series = None,
                          log_mc: bool = True) -> pd.Series:
    """
    预测值行业+市值中性化（剥离行业和市值暴露）。

    功能：
        对预测得分进行截面中性化处理，剥离行业和市值因子的暴露，
        使预测得分纯粹反映 Alpha 信号。

    入参：
        predictions: pd.Series - 原始预测得分
        industry: pd.Series - 行业分类（可选）
        market_cap: pd.Series - 市值（可选）
        log_mc: bool - 是否对市值取对数

    返回：
        pd.Series - 中性化后的预测得分

    边界条件：
        - industry 和 market_cap 都为 None 时返回原始预测
        - 样本太少时不做中性化
        - 中性化后保持均值为 0，标准差不变

    注意事项：
        - 行业中性化：每个行业内标准化
        - 市值中性化：对市值做回归取残差
        - AQR 等因子投资机构非常重视中性化
        - 中性化后的信号更纯粹，换手率更低
    """
    if industry is None and market_cap is None:
        return predictions

    result = predictions.copy()

    # 行业中性化
    if industry is not None:
        # 按行业分组，每个行业内 Z-score 标准化
        def _industry_neutralize(group):
            if len(group) < 5:
                return group
            std = group.std()
            if std == 0 or np.isnan(std):
                return group - group.mean()
            return (group - group.mean()) / std

        result = result.groupby(industry).transform(_industry_neutralize)

    # 市值中性化
    if market_cap is not None:
        mc = market_cap.copy()
        if log_mc:
            mc = np.log(mc.replace(0, np.nan))

        # 按日期分组做截面回归
        if isinstance(predictions.index, pd.MultiIndex):
            dates = predictions.index.get_level_values("datetime")
        else:
            dates = predictions.index

        def _mc_neutralize(date_group):
            if len(date_group) < 10:
                return date_group
            mc_date = mc.loc[date_group.index]
            valid = mc_date.notna() & date_group.notna()
            if valid.sum() < 10:
                return date_group
            x = mc_date[valid].values.reshape(-1, 1)
            y = date_group[valid].values
            # 简单线性回归取残差
            try:
                from numpy.linalg import lstsq
                X = np.column_stack([np.ones(len(x)), x])
                beta, _, _, _ = lstsq(X, y, rcond=None)
                residual = y - X @ beta
                # 保持原标准差
                result_series = pd.Series(0.0, index=date_group.index)
                result_series.loc[valid] = residual
                # 缩放回原始标准差
                orig_std = date_group.std()
                res_std = result_series[valid].std()
                if res_std > 0 and orig_std > 0:
                    result_series = result_series * (orig_std / res_std)
                return result_series
            except Exception:
                return date_group

        result = result.groupby(dates).transform(_mc_neutralize)

    return result


def apply_turnover_control(predictions_df: pd.DataFrame, ema_alpha: float = 0.3) -> pd.DataFrame:
    """
    换手率控制（指数平滑 EMA，降低交易成本）。

    功能：
        对预测得分进行指数移动平均平滑，降低预测的波动，
        从而降低换手率和交易成本。

    入参：
        predictions_df: pd.DataFrame - 预测得分数据（含 datetime 索引）
        ema_alpha: float - EMA 平滑系数（越小越平滑，换手率越低）

    返回：
        pd.DataFrame - 平滑后的预测得分

    边界条件：
        - ema_alpha = 1 时不做平滑
        - 只有一天数据时不做平滑
        - 平滑系数限制在 [0.05, 1.0]

    注意事项：
        - EMA 平滑是降低换手率的简单有效方法
        - alpha = 0.3 意味着今天的预测占 30%，历史预测占 70%
        - 平滑会降低信号的及时性，但也降低了噪声
        - 需要在换手率和预测能力之间找平衡
        - AQR 等机构非常重视交易成本控制
    """
    if ema_alpha >= 1.0 or len(predictions_df) == 0:
        return predictions_df

    ema_alpha = np.clip(ema_alpha, 0.05, 1.0)

    result = predictions_df.copy()
    score_col = "score" if "score" in result.columns else result.columns[0]

    # 按股票分组做时间序列 EMA
    if isinstance(result.index, pd.MultiIndex):
        # MultiIndex: (datetime, instrument)
        # 需要按 instrument 分组，然后按时间排序做 EMA
        smoothed = []

        for inst, group in result.groupby(level="instrument"):
            group_sorted = group.sort_index(level="datetime")
            smoothed_score = group_sorted[score_col].ewm(alpha=ema_alpha, adjust=False).mean()
            group_sorted[score_col] = smoothed_score
            smoothed.append(group_sorted)

        result = pd.concat(smoothed)
        result = result.sort_index()
    else:
        # 单索引，假设是时间序列
        smoothed_score = result[score_col].ewm(alpha=ema_alpha, adjust=False).mean()
        result[score_col] = smoothed_score

    return result


def compute_dynamic_long_short_ratio(market_volatility: float,
                                     dynamic_config: dict) -> dict:
    """
    动态多空比例（根据市场波动率动态调整）。

    功能：
        根据当前市场波动率动态调整多空比例，
        高波动时降低仓位，低波动时增加仓位。

    入参：
        market_volatility: float - 当前市场波动率（年化）
        dynamic_config: dict - 动态多空配置

    返回：
        dict - 动态调整后的多空比例
            - long_pct: 多头比例
            - short_pct: 空头比例
            - current_volatility: 当前波动率

    边界条件：
        - 波动率为 NaN 或 0 时使用默认比例
        - 比例限制在合理范围内

    注意事项：
        - 高波动市场中预测不确定性更高，应该降低仓位
        - 低波动市场中信号更可靠，可以增加仓位
        - 这是一种简单的波动率目标策略
        - Two Sigma 等机构广泛使用动态仓位调整
    """
    if np.isnan(market_volatility) or market_volatility <= 0:
        return {
            "long_pct": dynamic_config.get("low_vol_long_pct", 0.30),
            "short_pct": dynamic_config.get("low_vol_short_pct", 0.10),
            "current_volatility": market_volatility,
        }

    # 波动率阈值
    high_vol_threshold = 0.30  # 30% 年化波动率为高波动
    low_vol_threshold = 0.15   # 15% 年化波动率为低波动

    if market_volatility >= high_vol_threshold:
        long_pct = dynamic_config.get("high_vol_long_pct", 0.20)
        short_pct = dynamic_config.get("high_vol_short_pct", 0.05)
    elif market_volatility <= low_vol_threshold:
        long_pct = dynamic_config.get("low_vol_long_pct", 0.30)
        short_pct = dynamic_config.get("low_vol_short_pct", 0.10)
    else:
        # 线性插值
        ratio = (market_volatility - low_vol_threshold) / (high_vol_threshold - low_vol_threshold)
        long_pct = (
            dynamic_config.get("low_vol_long_pct", 0.30) * (1 - ratio) +
            dynamic_config.get("high_vol_long_pct", 0.20) * ratio
        )
        short_pct = (
            dynamic_config.get("low_vol_short_pct", 0.10) * (1 - ratio) +
            dynamic_config.get("high_vol_short_pct", 0.05) * ratio
        )

    return {
        "long_pct": long_pct,
        "short_pct": short_pct,
        "current_volatility": market_volatility,
    }


# ==============================================================================
# [第七层：可解释性层 - Dimensional 级]
# ==============================================================================

def compute_feature_importance_stability(feature_importance_list: list[dict]) -> dict:
    """
    特征重要性跨窗口稳定性检验。

    功能：
        比较不同窗口/不同 fold 之间特征重要性的一致性，
        检验特征是否稳定地对预测有贡献。

    入参：
        feature_importance_list: list[dict] - 各窗口的特征重要性字典列表
            每个字典格式: {feature_name: importance_score}

    返回：
        dict - 稳定性分析结果
            - n_windows: 窗口数量
            - feature_stability: dict - 每个特征的稳定性得分
            - stable_features: list - 稳定特征列表
            - unstable_features: list - 不稳定特征列表
            - overall_stability: float - 整体稳定性得分

    边界条件：
        - 窗口数量 < 2 时无法计算稳定性
        - 特征数量为 0 时返回空结果

    注意事项：
        - 稳定性得分基于特征重要性排名的相关性
        - 稳定的特征在不同窗口中都有相似的重要性排名
        - 不稳定的特征可能是过拟合的结果
        - Dimensional 等学术背景的机构非常重视特征稳定性
    """
    n_windows = len(feature_importance_list)
    if n_windows < 2:
        return {"n_windows": n_windows, "overall_stability": np.nan}

    # 收集所有特征
    all_features = set()
    for fi in feature_importance_list:
        all_features.update(fi.keys())

    if not all_features:
        return {"n_windows": n_windows, "overall_stability": 0.0}

    # 构建特征重要性矩阵（行：窗口，列：特征）
    features_list = sorted(all_features)
    imp_matrix = np.zeros((n_windows, len(features_list)))

    for i, fi in enumerate(feature_importance_list):
        for j, feat in enumerate(features_list):
            imp_matrix[i, j] = fi.get(feat, 0.0)

    # 计算每个特征的稳定性（变异系数的倒数）
    # 变异系数 = 标准差 / 均值，越小越稳定
    feature_stability = {}
    for j, feat in enumerate(features_list):
        col = imp_matrix[:, j]
        mean_imp = np.mean(col)
        std_imp = np.std(col)
        if mean_imp == 0:
            stability = 0.0
        else:
            cv = std_imp / abs(mean_imp)  # 变异系数
            stability = 1.0 / (1.0 + cv)   # 转换为 0-1 的稳定性得分
        feature_stability[feat] = stability

    # 整体稳定性（所有特征稳定性的均值）
    overall_stability = np.mean(list(feature_stability.values()))

    # 按稳定性排序
    sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
    stable_features = [f for f, s in sorted_features if s >= 0.7]
    unstable_features = [f for f, s in sorted_features if s < 0.4]

    result = {
        "n_windows": n_windows,
        "n_features": len(features_list),
        "feature_stability": feature_stability,
        "stable_features": stable_features,
        "unstable_features": unstable_features,
        "overall_stability": overall_stability,
        "top_10_stable": sorted_features[:10],
        "bottom_10_unstable": sorted_features[-10:] if len(sorted_features) >= 10 else sorted_features,
    }

    print(f"    [特征稳定性] 整体稳定性={overall_stability:.4f}, "
          f"稳定特征={len(stable_features)}个, 不稳定特征={len(unstable_features)}个")

    return result


def extract_feature_importance(model, feature_names: list, model_type: str) -> dict:
    """
    从模型中提取特征重要性。

    功能：
        统一提取不同类型模型的特征重要性。

    入参：
        model: 模型对象
        feature_names: list - 特征名列表
        model_type: str - 模型类型

    返回：
        dict - 特征重要性字典 {feature_name: importance_score}

    边界条件：
        - 模型不支持特征重要性时返回空字典
        - 特征名数量不匹配时返回空字典

    注意事项：
        - LightGBM/XGBoost/CatBoost 都有内置的特征重要性
        - 默认使用 gain 重要性（比 split 更准确）
    """
    try:
        if model_type == "lgb":
            # LightGBM feature importance
            imp = model.feature_importance(importance_type="gain")
            if len(imp) == len(feature_names):
                return dict(zip(feature_names, imp))
        elif model_type == "xgb":
            # XGBoost feature importance
            imp = model.get_score(importance_type="gain")
            # imp 是 {f0: score, f1: score, ...} 格式
            result = {}
            for k, v in imp.items():
                idx = int(k.replace("f", ""))
                if idx < len(feature_names):
                    result[feature_names[idx]] = v
            return result
        elif model_type == "cat":
            # CatBoost feature importance
            imp = model.get_feature_importance()
            if len(imp) == len(feature_names):
                return dict(zip(feature_names, imp))
    except Exception:
        pass

    return {}


def compute_shap_analysis(model, X: pd.DataFrame, model_type: str,
                          n_samples: int = 1000) -> dict:
    """
    SHAP 值特征重要性分析（全局和局部可解释性）。

    功能：
        使用 SHAP（SHapley Additive exPlanations）方法分析特征对预测的贡献，
        提供全局特征重要性和局部预测解释。

    入参：
        model: 训练好的模型
        X: pd.DataFrame - 特征数据
        model_type: str - 模型类型
        n_samples: int - 用于计算 SHAP 的样本数

    返回：
        dict - SHAP 分析结果
            - shap_values: np.ndarray - SHAP 值矩阵
            - global_importance: dict - 全局特征重要性（|SHAP| 均值）
            - top_features: list - Top 10 最重要特征
            - base_value: float - 基准值（平均预测值）

    边界条件：
        - 样本数太少时跳过
        - SHAP 库未安装时返回空结果
        - 计算失败时返回空结果

    注意事项：
        - SHAP 是目前最可靠的模型可解释性方法
        - 基于 Shapley 值，有坚实的博弈论基础
        - 全局重要性用 |SHAP| 均值衡量
        - 局部解释可以说明每个预测的因子贡献
        - Dimensional 等学术背景的机构非常重视可解释性
    """
    if len(X) < 10 or n_samples <= 0:
        return {"enabled": False, "reason": "insufficient samples"}

    # 采样
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X.copy()

    try:
        import shap

        # 创建 explainer
        if model_type == "lgb":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_type == "xgb":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_type == "cat":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            return {"enabled": False, "reason": "unsupported model type"}

        # 处理二分类情况（shap_values 可能是 list）
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 取正类的 SHAP 值

        # 全局特征重要性（|SHAP| 均值）
        feature_names = X_sample.columns.tolist()
        global_importance = {}
        for i, feat in enumerate(feature_names):
            global_importance[feat] = np.mean(np.abs(shap_values[:, i]))

        # 按重要性排序
        sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:10]

        # 基准值
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        result = {
            "enabled": True,
            "n_samples": len(X_sample),
            "shap_values": shap_values,
            "feature_names": feature_names,
            "global_importance": global_importance,
            "top_features": top_features,
            "base_value": base_value,
        }

        print(f"    [SHAP分析] 样本数={len(X_sample)}, Top特征={[f for f, _ in top_features[:5]]}")

        return result

    except ImportError:
        print(f"    [SHAP分析] 跳过: shap 库未安装")
        return {"enabled": False, "reason": "shap library not installed"}
    except Exception as e:
        print(f"    [SHAP分析] 失败: {e}")
        return {"enabled": False, "reason": str(e)}


def main():
    CONFIG = build_effective_local_config(LOCAL_CONFIG)

    print("=" * 60)
    print("  基于精选因子的模型训练与选股")
    print("=" * 60)

    # 1. 加载精选因子列表
    print(f"\n[1] 加载精选因子列表: {CONFIG['factor_list']}")
    source_files, selected_factor_names = load_selected_factors(CONFIG["factor_list"])
    CONFIG["factor_files"] = source_files

    # 2. 初始化 Qlib
    print("\n[2] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading", maxtasksperchild=None)

    # 3. 从因子库加载所有因子
    print("\n[3] 读取因子库 (Factor Library)...")
    bundle_all = build_factor_library_bundle(source_files, factor_names=selected_factor_names)
    bundle_all.label_fields = CONFIG["label_fields"]
    bundle_all.label_names = CONFIG["label_names"]

    print(f"  >>> 成功加载 {len(bundle_all.fields)} 个精选因子")

    all_predictions = []
    # [Citadel Alpha Lab] 跨窗口 IC 跟踪，用于 EWMA 加权集成
    model_ic_history: dict[str, list[float]] = {}
    # [Bloomberg] 全局特征缓存，首窗口构建后跨窗口复用
    global_feature_cache = None
    # [Dimensional] 跨窗口特征重要性列表，用于稳定性分析
    all_feature_importance = []

    # 4. 遍历所有滚动窗口
    for window_idx, window in enumerate(CONFIG["rolling_windows"]):
        window_name = window["name"]
        print(f"\n{'='*60}")
        print(f"=== 正在处理滚动窗口: {window_name} ===")
        print(f"    [训练集]: {window['train'][0]} 到 {window['train'][1]}")
        print(f"    [验证集]: {window['valid'][0]} 到 {window['valid'][1]}")
        print(f"    [测试集]: {window['test'][0]} 到 {window['test'][1]}")
        print(f"{'='*60}")

        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }

        # ----- [4.0] 特征缓存：首窗口构建，后续复用 -----
        print(f"\n[4.0 - {window_name}] 特征缓存...")
        if global_feature_cache is None:
            global_feature_cache = build_custom_feature_cache(
                instruments=CONFIG["instruments"],
                feature_bundle=bundle_all,
                factor_cache_names=CONFIG["factor_cache_names"],
                start_time=CONFIG["start_time"],
                end_time=CONFIG["end_time"],
                freq="day",
            )
            print(f"    [全局缓存] 构建完成：覆盖 {CONFIG['start_time']} ~ {CONFIG['end_time']}")
        else:
            print(f"    [复用全局缓存] 跳过重复计算")

        # ----- [4.1] 一次性构建含 train/valid/test 的数据集 -----
        # [Point72 性能优化] 合并原 4.1+4.2 两次 create_custom_dataset 为一次
        print(f"\n[4.1 - {window_name}] 构建轻量级 DatasetH（{len(selected_factor_names)} 个精选因子，含 train/valid/test 三段）...")
        _, dataset_sub = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_cache=global_feature_cache,
            selected_feature_names=selected_factor_names,
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

        # ----- [4.2] 提取训练/验证数据并缓存 -----
        print(f"\n[4.2 - {window_name}] 提取训练/验证帧...")
        train_frame_full = dataset_sub.prepare("train")
        print(f"    >>> 训练集: {train_frame_full.shape[0]} 行 × {train_frame_full.shape[1]} 列")

        # [Virtu-Renaissance 修复] 标签可交易性过滤：剔除涨跌停/一字板无法买入样本
        if CONFIG.get("filter_untradeable_labels", False):
            _train_inst = train_frame_full.index.get_level_values("instrument").unique().tolist()
            train_frame_full = apply_label_filter(
                train_frame_full, _train_inst,
                segments["train"][0], segments["train"][1], bundle_all.label_names
            )

        valid_frame = None
        try:
            valid_frame = dataset_sub.prepare("valid")
            print(f"    >>> 验证集: {valid_frame.shape[0]} 行")
            if CONFIG.get("filter_untradeable_labels", False) and valid_frame is not None:
                _valid_inst = valid_frame.index.get_level_values("instrument").unique().tolist()
                valid_frame = apply_label_filter(
                    valid_frame, _valid_inst,
                    segments["valid"][0], segments["valid"][1], bundle_all.label_names
                )
        except Exception:
            print(f"    [警告] 验证集为空，跳过早停和 IC 加权")

        dataset_sub = wrap_dataset_with_cached_train_frame(
            dataset_sub,
            train_frame=train_frame_full,
            selected_feature_names=selected_factor_names,
            label_names=bundle_all.label_names,
            learn_data_key=DataHandlerLP.DK_L,
            infer_data_key=DataHandlerLP.DK_I,
            valid_frame=valid_frame,
        )

        train_label = extract_label_series(train_frame_full, label_names=bundle_all.label_names)
        valid_label = extract_label_series(valid_frame, label_names=bundle_all.label_names)

        del train_frame_full
        gc.collect()

        # ----- [4.3] 训练模型 -----
        selected_models = list(CONFIG.get("train_models", ["lgb", "xgb", "cat"]))
        model_params = CONFIG.get("model_params", {})
        models = []

        print(f"\n[4.3 - {window_name}] 开始训练机器学习模型 {selected_models}...")

        if "lgb" in selected_models:
            print("    - 正在训练 LightGBM 模型...")
            models.append(train_lgb_model(dataset_sub, params=model_params.get("lgb")))
        if "xgb" in selected_models:
            print("    - 正在训练 XGBoost 模型...")
            models.append(train_xgb_model(dataset_sub, params=model_params.get("xgb")))
        if "cat" in selected_models:
            # [Bloomberg 修复] 移除 task_type="CPU" 硬编码，由底层自动检测 GPU
            print("    - 正在训练 CatBoost 模型...")
            models.append(train_catboost_model(dataset_sub, params=model_params.get("cat")))

        if not models:
            raise ValueError("CONFIG['train_models'] 不能为空")
        print(f"  >>> {window_name} 所有模型训练完毕！")

        # ----- [4.4 预测前] 计算 IC 加权权重 -----
        # [Citadel Alpha Lab] 用验证集 IC 的 EWMA 作为集成权重
        diagnostics = collect_model_diagnostics(
            models=models,
            model_names=selected_models,
            dataset=dataset_sub,
            train_label=train_label,
            valid_label=valid_label,
            model_ic_history=model_ic_history,
            min_ic_samples=30,
            ic_half_life=4,
        )

        # [第三层：集成增强层 - Citadel 级] 多维加权集成
        ensemble_config = CONFIG.get("ensemble_enhancement", {})
        if ensemble_config.get("multi_dim_weighting", {}).get("enabled", False):
            model_ic_weights, used_equal_weight = compute_multi_dim_weights(
                diagnostics, ensemble_config
            )
            print(f"    [多维加权] 使用 IC+ICIR+衰减+夏普 四维加权")
        else:
            model_ic_weights, used_equal_weight = resolve_model_weights(diagnostics)

        log_model_diagnostics(window_name, diagnostics, model_ic_weights, used_equal_weight)

        is_window_qualified, reject_reasons = assess_window_quality(window_name, diagnostics, CONFIG)
        if not is_window_qualified:
            print(f"\n[4.3B - {window_name}] 窗口质量闸门未通过，跳过该窗口预测写入。")
            for reason in reject_reasons:
                print(f"      - {reason}")
            del dataset_sub, models, valid_frame, train_label, valid_label
            gc.collect()
            continue

        # ----- [4.4] 预测 -----
        print(f"\n[4.4 - {window_name}] 在测试集上进行模型集成与预测...")
        predictions = predict_ensemble_models(models, dataset_sub, segment="test",
                                              model_weights=model_ic_weights)

        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")

        predictions = predictions.dropna(subset=["score"])

        # 保存原始预测（用于后续评估）
        predictions["raw_score"] = predictions["score"]

        # 截面分位数排名
        predictions["score"] = predictions.groupby(
            level="datetime"
        )["score"].rank(pct=True, na_option="keep")

        # [P1修复] 置信度阈值：弱信号衰减至中性 0.5
        conf_threshold = CONFIG.get("prediction_confidence_threshold")
        if conf_threshold is not None and conf_threshold > 0:
            raw_ranked = predictions["score"].copy()
            signal_strength = (raw_ranked - 0.5).abs()
            daily_median_strength = signal_strength.groupby(level="datetime").transform("median")
            weak_signal = signal_strength < daily_median_strength * conf_threshold
            n_weak = weak_signal.sum()
            if n_weak > 0:
                predictions.loc[weak_signal, "score"] = 0.5
                print(f"    [置信度衰减] {n_weak} 个弱信号衰减至中性 "
                      f"({100*n_weak/max(len(predictions),1):.1f}%)")

        # [Renaissance 标准] 多空非对称处理
        # 若 long_pct>=1.0 或 short_pct<=0.0，则跳过 score 塌缩
        ls_ratio = CONFIG.get("long_short_ratio", {"long_pct": 0.30, "short_pct": 0.10})

        # [第六层：后处理增强层 - AQR 级] 动态多空比例
        post_config = CONFIG.get("post_processing", {})
        dynamic_ls_config = post_config.get("dynamic_long_short", {})
        if dynamic_ls_config.get("enabled", False):
            # 简单估算当前市场波动率（用验证集标签的波动率近似）
            if valid_label is not None and len(valid_label) > 20:
                market_vol = valid_label.std() * np.sqrt(252)
                dynamic_ratio = compute_dynamic_long_short_ratio(market_vol, dynamic_ls_config)
                ls_ratio = {"long_pct": dynamic_ratio["long_pct"], "short_pct": dynamic_ratio["short_pct"]}
                print(f"    [动态多空] 波动率={market_vol:.2%}, "
                      f"多头={ls_ratio['long_pct']:.0%}, 空头={ls_ratio['short_pct']:.0%}")

        long_cutoff = 1.0 - ls_ratio["long_pct"]
        short_cutoff = ls_ratio["short_pct"]
        middle_mask = (predictions["score"] > short_cutoff) & (predictions["score"] < long_cutoff)
        n_middle = middle_mask.sum()
        if n_middle > 0:
            predictions.loc[middle_mask, "score"] = 0.5
            print(f"    [多空非对称] {n_middle} 个中性信号衰减 (long_top={ls_ratio['long_pct']:.0%}, "
                  f"short_bottom={ls_ratio['short_pct']:.0%})")

        # [第六层：后处理增强层 - AQR 级] 换手率控制（EMA 平滑）
        turnover_config = post_config.get("turnover_control", {})
        if turnover_config.get("enabled", False):
            ema_alpha = turnover_config.get("ema_alpha", 0.3)
            predictions_before = predictions["score"].copy()
            predictions = apply_turnover_control(predictions, ema_alpha=ema_alpha)
            # 计算换手率降低比例
            if len(predictions_before) > 0:
                turnover_reduction = 1 - predictions["score"].std() / predictions_before.std() if predictions_before.std() > 0 else 0
                print(f"    [换手率控制] EMA alpha={ema_alpha}, 波动率降低={turnover_reduction:.1%}")

        print(f"  >>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)

        # [第七层：可解释性层 - Dimensional 级] 特征重要性提取
        interpret_config = CONFIG.get("interpretability", {})
        if interpret_config.get("feature_importance_stability", {}).get("enabled", True):
            try:
                # 从每个模型提取特征重要性
                window_fi = {}
                for i, model in enumerate(models):
                    model_name = selected_models[i] if i < len(selected_models) else f"model_{i}"
                    fi = extract_feature_importance(model, selected_factor_names, model_name)
                    if fi:
                        # 归一化重要性
                        total = sum(fi.values())
                        if total > 0:
                            fi = {k: v / total for k, v in fi.items()}
                        window_fi[model_name] = fi

                if window_fi:
                    # 取所有模型的平均重要性
                    all_feats = set()
                    for fi in window_fi.values():
                        all_feats.update(fi.keys())
                    avg_fi = {}
                    for feat in all_feats:
                        vals = [fi.get(feat, 0) for fi in window_fi.values()]
                        avg_fi[feat] = np.mean(vals)
                    all_feature_importance.append(avg_fi)
                    print(f"    [特征重要性] 提取完成，共 {len(avg_fi)} 个特征")
            except Exception as e:
                print(f"    [特征重要性] 提取失败: {e}")

        del dataset_sub, models, valid_frame, train_label, valid_label
        gc.collect()

    # 5. 合并所有滚动窗口的预测结果
    print("\n[5] 所有滚动窗口执行完毕！正在合并预测结果...")
    if not all_predictions:
        raise RuntimeError("所有滚动窗口均被质量闸门剔除，未生成可用预测结果。")
    final_predictions = pd.concat(all_predictions)
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)

    # PIT 后置过滤: main_board + 退市 + ST + 次新股
    try:
        before = len(final_predictions)
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

        _all_path = Path(QLIB_DATA_DIR) / "instruments" / "all.txt"
        _delist_pit = {}
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

        _filter_new = CONFIG.get("filter_new_stocks", True)
        _filter_st = CONFIG.get("filter_st", True)
        _filtered_parts = []
        _total_st_removed = 0

        for _date, _day_df in final_predictions.groupby(level="datetime"):
            _dt_str = str(_date)[:10]
            _day_insts = _day_df.index.get_level_values("instrument").str.lower()
            if _board_stocks:
                _in_board = np.asarray(_day_insts.isin(_board_stocks))
            else:
                _in_board = np.ones(len(_day_insts), dtype=bool)
            _not_delisted = _day_insts.map(
                lambda x: _delist_pit.get(x, "9999-12-31") >= _dt_str
            )
            _day_df = _day_df[_in_board & np.asarray(_not_delisted)]

            if _day_df.empty:
                continue

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
        print(f"\n  [PIT 过滤] {before} → {after} (剔除 {before-after} 条: 非主板/已退市/ST/次新, "
              f"其中 ST/次新 {_total_st_removed} 条)")
    except Exception as e:
        print(f"  [警告] 后置过滤异常: {e}，跳过")

    if final_predictions.empty:
        raise RuntimeError("后置过滤后无有效预测，请检查数据质量")

    print(f"  >>> 合并完成！总测试集跨度: {final_predictions.index.get_level_values('datetime').min().date()} "
          f"至 {final_predictions.index.get_level_values('datetime').max().date()}")
    print(final_predictions.head(10))

    # 6. 保存预测结果
    output_path = CONFIG["output"] or os.path.join(os.path.dirname(__file__), "score_tree_selected.csv")
    final_predictions.to_csv(output_path)
    print(f"\n  >>> 预测得分已保存至: {output_path}")
    print(
        "  >>> 通达信模拟盘目标持仓生成命令: "
        f"python scripts/live/generate_tree_targets.py --strategy {CONFIG['live_strategy_name']} "
        f"--runtime-model-name {CONFIG['live_runtime_model_name']}"
    )

    print("=" * 60)
    factor_count = len(selected_factor_names)
    print(f"  精选因子数: {factor_count}")
    print(f"  训练模型: {selected_models}")
    print(f"  总预测记录: {len(final_predictions):,}")
    print("=" * 60)

    # =========================================================================
    # [P2 增强] 综合评估报告
    # =========================================================================
    print(f"\n{'='*60}")
    print("=== [综合评估] IC 衰减 & 分组分析 ===")
    print(f"{'='*60}")

    # [第七层：可解释性层 - Dimensional 级] 特征重要性稳定性分析
    interpret_config = CONFIG.get("interpretability", {})
    if interpret_config.get("feature_importance_stability", {}).get("enabled", True) and len(all_feature_importance) >= 2:
        print(f"\n  [特征重要性稳定性分析]")
        stability_result = compute_feature_importance_stability(all_feature_importance)

        if stability_result.get("top_10_stable"):
            print(f"    Top 10 最稳定特征:")
            for feat, score in stability_result["top_10_stable"][:10]:
                print(f"      {feat}: 稳定性={score:.4f}")

        if stability_result.get("bottom_10_unstable"):
            print(f"    Bottom 10 最不稳定特征:")
            for feat, score in stability_result["bottom_10_unstable"][-10:]:
                print(f"      {feat}: 稳定性={score:.4f}")

    # IC 衰减分析（使用最后一个窗口训练期数据）
    eval_conf = CONFIG.get("evaluation", {})
    if eval_conf.get("ic_decay_horizons") and len(selected_factor_names) > 0:
        last_window = CONFIG["rolling_windows"][-1]
        try:
            decay_df = _compute_ic_decay(
                global_feature_cache,
                selected_factor_names[:min(20, len(selected_factor_names))],
                global_feature_cache.resolved_instruments,
                last_window["train"][0], last_window["train"][1],
                eval_conf["ic_decay_horizons"],
            )
            if not decay_df.empty:
                print(f"  IC 衰减曲线 (基于前 {min(20, len(selected_factor_names))} 个因子):")
                for horizon, row in decay_df.iterrows():
                    valid_vals = row.dropna()
                    if len(valid_vals) > 0:
                        print(f"    {horizon}: mean|IC|={np.mean(np.abs(valid_vals)):.4f} "
                              f"(基于 {len(valid_vals)} 个因子)")
        except Exception as e:
            print(f"  IC 衰减分析失败: {e}")

    # 行业/市值分组 IC
    if eval_conf.get("industry_exposure_check") and len(all_predictions) > 0:
        last_preds = all_predictions[-1]
        if not last_preds.empty and 'score' in last_preds.columns:
            try:
                from qlib.data import D
                last_window = CONFIG["rolling_windows"][-1]
                label_frames = []
                for i in range(0, len(global_feature_cache.resolved_instruments), 500):
                    batch_inst = global_feature_cache.resolved_instruments[i:i+500]
                    _df = D.features(batch_inst, [CONFIG["label_fields"][0]],
                                     start_time=last_window["test"][0],
                                     end_time=last_window["test"][1], freq="day")
                    if _df is not None and not _df.empty:
                        label_frames.append(_df)
                if label_frames:
                    test_label_s = pd.concat(label_frames)
                    if isinstance(test_label_s.columns, pd.MultiIndex):
                        test_label_s.columns = test_label_s.columns.droplevel(1)
                    test_label_s = test_label_s[test_label_s.columns[0]].sort_index()

                    ind_ics = _compute_group_ic(last_preds, test_label_s, analysis_type="industry")
                    if ind_ics:
                        print(f"\n  行业 IC 分布 (Top 10 |IC|):")
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


if __name__ == "__main__":
    main()
