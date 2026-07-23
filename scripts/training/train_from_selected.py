"""
基于精选因子列表的模型训练与选股脚本。

从 select_factors.py 输出的 CSV 中读取精选因子列表，
仅用这些筛选后的因子进行 ML 模型训练 (LGB/XGB/CatBoost)，
输出每只股票每天的 Alpha 预测得分。

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
    "instruments": "main_board",
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
    "use_dynamic_filter": True,

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
    "factor_list": "selected_factors_tree.txt",

    # 输出路径：None 自动生成 score_tree_selected.csv
    "output": None,
    # 通达信模拟盘对接：默认使用 selected 档案，但沿用 tree 运行目录，便于复用现有执行器。
    "live_strategy_name": "selected",
    "live_runtime_model_name": "tree",
    # 运行期质量闸门：用于剔除标签/数据异常导致的坏窗口，避免污染最终选股结果。
    "window_quality_gate": {
        "enabled": True,
        "min_valid_samples": 30,
        "max_train_rmse": 5.0,
        "max_valid_rmse": 5.0,
        "min_healthy_models": 2,
    },
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

        valid_frame = None
        try:
            valid_frame = dataset_sub.prepare("valid")
            print(f"    >>> 验证集: {valid_frame.shape[0]} 行")
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
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")

        print(f"  >>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)

        del dataset_sub, models, valid_frame, train_label, valid_label
        gc.collect()

    # 5. 合并所有滚动窗口的预测结果
    print("\n[5] 所有滚动窗口执行完毕！正在合并预测结果...")
    if not all_predictions:
        raise RuntimeError("所有滚动窗口均被质量闸门剔除，未生成可用预测结果。")
    final_predictions = pd.concat(all_predictions)
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)

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


if __name__ == "__main__":
    main()
