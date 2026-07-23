"""
精选因子筛选脚本：按因子类别分组，在每个类别内独立运行特征选择，
筛选出各类别中预测能力最强的因子。

[AQR/Citadel/Renaissance 改进]
  - 滚动窗口因子筛选，消除前瞻偏差
  - 与 train_from_selected.py 统一的股票池、动态过滤、ST/次新过滤
  - 截面排名标准化 (CSRankNorm)，与训练阶段数据处理一致

用法：
  修改文件顶部 CONFIG 字典中的参数，然后直接运行：
    python select_factors.py

输出：
  - 控制台打印每个类别的因子筛选结果（含重要性得分）
  - 保存精选因子列表至 selected_factors_{时间戳}.csv
"""

import os
import sys
import warnings
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import FeatureBundle
from qlworks.features.dataset import build_custom_feature_cache
from qlworks.factors.filter_utils import filter_codes_post
from qlworks.models import cached_select_features
from qlworks.config import QLIB_DATA_DIR
import qlib
from qlib.data import D

# ==============================================================================
# 路径与默认配置
# ==============================================================================
FACTOR_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "factor_data" / "factor_library"
ARCHIVE_DIR = FACTOR_LIBRARY_DIR / "archive"

ACTIVE_FACTOR_FILES = [
    "reversal_momentum_factors",
    "quality_factors",
    "style_factors",
    "price_volume_factors",
    "risk_factors",
    "sentiment_factors",
    "other_factors",
]

# ==============================================================================
# [全局配置区] - 在此修改运行参数
# ==============================================================================
CONFIG = {
    # 因子文件列表（从 ACTIVE_FACTOR_FILES 派生，排除 price_volume_factors）
    "factor_files": [f for f in ACTIVE_FACTOR_FILES if f != "price_volume_factors"],

    # 特征选择参数
    "top_k": 2,
    "method": "embedded",
    "algo": "lightgbm",
    "min_factors": 3,

    # --- 股票池与过滤（与 train_from_selected.py 对齐）---
    "instruments": "main_board",
    "use_dynamic_filter": True,
    "filter_new_stocks": True,
    "filter_st": True,

    # --- 滚动窗口因子筛选（训练期窗口与 train_from_selected.py 完全对齐）---
    # 每个窗口仅使用其 train 期数据做因子筛选，消除前瞻偏差。
    "rolling_windows": [
        {
            "name": "Window_2023",
            "train": ("2020-01-01", "2021-12-20"),
        },
        {
            "name": "Window_2024",
            "train": ("2021-01-01", "2022-12-20"),
        },
        {
            "name": "Window_2025",
            "train": ("2022-01-01", "2023-12-20"),
        }
    ],

    # 标签（与 train_from_selected.py 对齐）
    "label_expr": "Ref($close, -5) / Ref($open, -1) - 1",
    "label_name": "LABEL_5D",

    # 冗余检测
    "redundancy_check": True,
    "redundancy_threshold": 0.95,

    # ICIR 稳定性校验
    "icir_stability": True,
    "icir_window": 60,
    "icir_keep_ratio": 0.9,

    # 跨窗口聚合：因子在 >= min_window_ratio 比例的窗口中入选才最终选中
    "min_window_ratio": 0.5,

    # 输出：None 自动生成时间戳文件名
    "output": None,

    # 缓存
    "clean_start": False,
}


def resolve_factor_files(factor_files: List[str]) -> List[str]:
    """解析因子文件列表。支持 'all' 表示加载所有活跃因子文件。"""
    if len(factor_files) == 1 and factor_files[0] == "all":
        return ACTIVE_FACTOR_FILES
    return factor_files


def load_factor_yaml(file_path: Path) -> Optional[dict]:
    """安全加载单个因子 YAML 文件。"""
    if not file_path.exists():
        print(f"  [警告] 因子文件不存在: {file_path}")
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  [错误] 加载因子文件失败 {file_path}: {e}")
        return None


def load_factors_by_category(factor_files: List[str]) -> Dict[str, List[Dict]]:
    """
    加载指定因子文件，按 category 字段分组。

    返回: {category_name: [{name, expression, meaning, source_file}, ...], ...}
    """
    categories: Dict[str, List[Dict]] = {}
    total_factors = 0

    for fname in factor_files:
        path = FACTOR_LIBRARY_DIR / f"{fname}.yaml"
        data = load_factor_yaml(path)
        if data is None:
            continue

        file_factors = data.get("factors") or []
        total_factors += len(file_factors)
        print(f"    [加载] {fname}.yaml: {len(file_factors)} 个因子")

        for factor in file_factors:
            name = factor.get("name")
            if not name:
                continue
            expr_raw = factor.get("expression", {})
            qlib_expr = expr_raw.get("qlib", "") if isinstance(expr_raw, dict) else str(expr_raw)
            if not qlib_expr:
                continue

            cat = factor.get("category", "未分类")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "name": name,
                "expression": qlib_expr,
                "source_file": fname,
                "meaning": factor.get("meaning", ""),
                "usage": factor.get("usage_scenario", ""),
            })

    print(f"  >>> 共加载 {total_factors} 个因子，按类别分为 {len(categories)} 组")
    return categories


def build_global_bundle(factors_by_cat: Dict[str, List[Dict]], label_expr: str, label_name: str) -> FeatureBundle:
    """将所有类别的因子合并为全局 FeatureBundle。"""
    all_fields = []
    all_names = []
    for cat_name, factors in sorted(factors_by_cat.items()):
        for f in factors:
            all_fields.append(f["expression"])
            all_names.append(f["name"])
    return FeatureBundle(
        fields=all_fields,
        names=all_names,
        label_fields=[label_expr],
        label_names=[label_name],
    )


def _vectorized_daily_ic(train_frame: pd.DataFrame, factor_cols: List[str], label_col: str) -> pd.DataFrame:
    """向量化逐日 IC 计算 - groupby().corr() 替代逐日 apply(corrwith)。"""
    all_cols = factor_cols + [label_col]
    corr_matrices = train_frame.groupby(level='datetime')[all_cols].corr(method='spearman')
    daily_ic = corr_matrices.xs(label_col, level=1, axis=0)[factor_cols]
    return daily_ic


def _apply_cs_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    截面排名标准化 (CSRankNorm)：对每个交易日截面，特征值转为 [0,1] 分位数，
    缺失值填充为 0.5（中位数）。等价于树模型场景下的 CSQuantileNorm。

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列

    输出:
    - 标准化后的 DataFrame，同 shape
    """
    result = df.groupby(level="datetime").rank(pct=True, na_option="keep")
    result = result.fillna(0.5)
    return result


def _filter_stocks_post(
    df: pd.DataFrame,
    filter_new_stocks: bool = True,
    filter_st: bool = True,
) -> pd.DataFrame:
    """
    对已加载的特征+标签 DataFrame 执行后置 ST/次新股过滤。

    逐日遍历，调用 filter_codes_post 过滤每只股票，
    移除不满足条件的行。

    输入:
    - df: MultiIndex (datetime, instrument) × 列
    - filter_new_stocks: 过滤上市不足 250 日次新股
    - filter_st: 过滤 ST 股票

    输出:
    - 过滤后的 DataFrame
    """
    if df.empty:
        return df
    if not filter_new_stocks and not filter_st:
        return df

    all_dates = sorted(df.index.get_level_values("datetime").unique())
    kept_parts = []
    total_removed = 0

    for date in all_dates:
        date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
        day_slice = df.xs(date, level="datetime", drop_level=False)
        if day_slice.empty:
            continue
        codes = day_slice.index.get_level_values("instrument").unique().tolist()
        filtered_codes = filter_codes_post(
            codes, date_str,
            filter_new_stocks=filter_new_stocks,
            filter_st=filter_st,
        )
        removed = len(codes) - len(filtered_codes)
        total_removed += removed
        if filtered_codes:
            kept = day_slice[day_slice.index.get_level_values("instrument").isin(filtered_codes)]
            kept_parts.append(kept)

    if kept_parts:
        result = pd.concat(kept_parts)
        result = result.sort_index()
    else:
        result = df.iloc[0:0]

    if total_removed > 0:
        print(f"    [后置过滤] ST/次新股过滤累计移除 {total_removed:,} 行 (stock×day)")

    return result


def _compute_window_full_period(rolling_windows: List[Dict]) -> Tuple[str, str]:
    """根据滚动窗口列表计算全局缓存所需的完整时间范围。"""
    all_starts = []
    all_ends = []
    for w in rolling_windows:
        all_starts.append(w["train"][0])
        all_ends.append(w["train"][1])
    return min(all_starts), max(all_ends)


def run_single_category_selection(
    cat_name: str,
    factors: List[Dict],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fs_method: str,
    fs_algo: str,
    top_k: int,
    label_name: str,
    redundancy_check: bool = True,
    redundancy_threshold: float = 0.95,
    icir_stability: bool = True,
    icir_rolling_window: int = 60,
    icir_keep_ratio: float = 0.9,
) -> Optional[pd.DataFrame]:
    """
    在单个类别上运行特征选择。

    [性能优化] 不再创建 Qlib dataset，直接从预准备的 x_train/y_train 中按因子名切片。
    CSRankNorm 已在主流程中运行一次，此处直接复用结果。

    输入:
    - cat_name: 类别名称
    - factors: 该类别因子列表 [{name, expression, meaning, source_file}, ...]
    - x_train: 已标准化 (CSRankNorm) 的特征矩阵
    - y_train: 标签 Series
    - fs_method, fs_algo, top_k: 特征选择参数
    - label_name: 标签列名（用于 ICIR 校验）
    - redundancy_check: 是否做冗余检测
    - redundancy_threshold: Spearman 相关系数阈值
    - icir_stability: 是否做 ICIR 稳定性校验
    - icir_rolling_window: ICIR 滚动窗口天数
    - icir_keep_ratio: ICIR 保留比例
    """
    cat_factor_names = [f["name"] for f in factors]
    print(f"    [筛选] {len(factors)} 个因子, top_k={top_k}...")

    # 从全局特征矩阵中切片
    available = [c for c in cat_factor_names if c in x_train.columns]
    if len(available) == 0:
        print(f"    [错误] 该类别的因子在训练数据中不存在")
        return None
    if len(available) < 2:
        print(f"    [跳过] 只有 {len(available)} 个可用因子，直接保留")
        rows = []
        for f in factors:
            if f["name"] in available:
                rows.append({
                    "category": cat_name, "factor_name": f["name"],
                    "selected": True, "importance": 1.0, "rank": 1,
                    "meaning": f["meaning"], "source_file": f["source_file"],
                })
        return pd.DataFrame(rows)

    x_cat = x_train[available]
    y_cat = y_train

    print(f"      >>> {x_cat.shape[0]} 行, {x_cat.shape[1]} 个特征")

    # 特征选择
    try:
        if fs_method == "embedded":
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo, threshold=0.0,
                model_kwargs={"max_features": min(top_k, len(available)), "importance_type": "gain"},
                remove_collinearity=False,
            )
        elif fs_method == "filter":
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                k=min(top_k, len(available)),
                remove_collinearity=False,
            )
        else:
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                model_kwargs={"max_features": min(top_k, len(available))},
                remove_collinearity=False,
            )
    except Exception as e:
        print(f"    [错误] 特征选择失败: {e}")
        return None

    selected_set = set(fs_result.selected_features)
    scores = fs_result.feature_scores
    selected_factor_names = list(selected_set)

    # 冗余检测：来自 x_train 的因子间相关
    if redundancy_check and len(selected_factor_names) > 5:
        print(f"    [冗余检测] 阈值={redundancy_threshold}，检测 {len(selected_factor_names)} 个入选因子...")
        feat_in_data = [c for c in selected_factor_names if c in x_train.columns]
        if len(feat_in_data) > 5:
            try:
                corr_mat = x_train[feat_in_data].corr(method='spearman').abs()
                redundant_pairs = []
                for i in range(len(corr_mat.columns)):
                    for j in range(i + 1, len(corr_mat.columns)):
                        c1, c2 = corr_mat.columns[i], corr_mat.columns[j]
                        if corr_mat.iloc[i, j] > redundancy_threshold:
                            redundant_pairs.append((c1, c2, corr_mat.iloc[i, j]))
                if redundant_pairs:
                    importance_map = dict(zip(scores.index, scores.values))
                    to_drop = set()
                    for c1, c2, corr_val in redundant_pairs:
                        if c1 in to_drop or c2 in to_drop:
                            continue
                        imp1 = abs(importance_map.get(c1, 0))
                        imp2 = abs(importance_map.get(c2, 0))
                        drop_f = c2 if imp2 < imp1 else c1
                        keep_f = c1 if drop_f == c2 else c2
                        to_drop.add(drop_f)
                        print(f"      冗余对: {c1}(imp={imp1:.4f}) vs {c2}(imp={imp2:.4f}) → 保留 {keep_f}，剔除 {drop_f}")
                    selected_factor_names = [f for f in selected_factor_names if f not in to_drop]
                    selected_set = set(selected_factor_names)
                    print(f"      冗余检测完成: 剔除 {len(to_drop)} 个冗余因子，保留 {len(selected_factor_names)} 个")
            except Exception as e:
                print(f"      [跳过] 冗余检测异常: {e}")

    # [Citadel] ICIR 稳定性校验（使用 x_train + y_train 构建面板）
    if icir_stability and len(selected_factor_names) > 5:
        print(f"    [ICIR 稳定校验] 窗口={icir_rolling_window}d, keep_ratio={icir_keep_ratio}...")
        try:
            icir_feat = [c for c in selected_factor_names if c in x_train.columns]
            if len(icir_feat) > 5:
                # 构建含标签的综合面板
                combined_frame = x_train[icir_feat].copy()
                combined_frame[label_name] = y_train
                daily_ic = _vectorized_daily_ic(combined_frame, icir_feat, label_name)
                if not daily_ic.empty and len(daily_ic) > icir_rolling_window // 2:
                    rolling_mean = daily_ic.rolling(window=icir_rolling_window, min_periods=icir_rolling_window // 2).mean()
                    rolling_std = daily_ic.rolling(window=icir_rolling_window, min_periods=icir_rolling_window // 2).std()
                    rolling_icir = rolling_mean / rolling_std.replace(0, np.nan)
                    pos_ratio = (rolling_icir > 0).sum() / rolling_icir.notna().sum()
                    pos_ratio = pos_ratio.fillna(0).sort_values(ascending=False)
                    keep_count = max(int(len(pos_ratio) * icir_keep_ratio), min(top_k, len(pos_ratio)))
                    stable_factors = pos_ratio.head(keep_count).index.tolist()
                    dropped = len(selected_factor_names) - len(stable_factors)
                    if dropped > 0:
                        print(f"      ICIR 检测: {dropped} 个不稳定因子被剔除")
                        selected_factor_names = stable_factors
                        selected_set = set(selected_factor_names)
                else:
                    print(f"      daily_ic 仅 {len(daily_ic)} 行，不足 {icir_rolling_window // 2}，跳过")
        except Exception as e:
            print(f"      [跳过] ICIR 稳定校验异常: {e}")

    # 构建重要性映射
    if len(scores) > 0 and scores.max() > 0:
        scores_norm = scores / scores.max()
    else:
        scores_norm = pd.Series(1.0, index=scores.index) if len(scores) > 0 else pd.Series(dtype=float)

    rows = []
    for rank, (factor_name, importance) in enumerate(scores_norm.items(), 1):
        factor_info = next((f for f in factors if f["name"] == factor_name), None)
        rows.append({
            "category": cat_name,
            "factor_name": factor_name,
            "selected": factor_name in selected_set,
            "importance": round(float(importance), 4),
            "rank": rank,
            "meaning": (factor_info or {}).get("meaning", ""),
            "source_file": (factor_info or {}).get("source_file", ""),
        })

    return pd.DataFrame(rows)


def print_category_results(cat_name: str, results_df: pd.DataFrame):
    """打印单个类别的筛选结果。"""
    if results_df is None or len(results_df) == 0:
        return

    selected = results_df[results_df["selected"]]
    not_selected = results_df[~results_df["selected"]]

    print(f"\n  │ 选中: {len(selected)}/{len(results_df)} 个因子")
    if len(selected) > 0:
        print(f"  │ 入选因子:")
        for _, row in selected.iterrows():
            bar = "#" * int(row["importance"] * 20) + "." * (20 - int(row["importance"] * 20))
            print(f"  │   [{row['rank']:2d}] {row['factor_name']:<25s} [{bar}] {row['importance']:.3f}")
    if len(not_selected) > 0:
        print(f"  │ 淘汰因子:")
        for _, row in not_selected.iterrows():
            print(f"  │   [{row['rank']:2d}] {row['factor_name']:<25s} 得分={row['importance']:.3f}")


def _prepare_window_data(
    global_feature_cache,
    all_factor_names: List[str],
    train_start: str,
    train_end: str,
    label_expr: str,
    label_name: str,
    filter_new_stocks: bool = True,
    filter_st: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    为单个滚动窗口准备训练数据：切片时间范围 → 合并标签 → 标准化 → 过滤。

    返回: (x_train, y_train)
    """
    # 1. 从缓存切片特征数据
    warehouse_df = global_feature_cache.warehouse_df.copy()
    warehouse_df.columns = warehouse_df.columns.droplevel(0)
    if isinstance(warehouse_df.index, pd.MultiIndex):
        warehouse_df.index = warehouse_df.index.set_levels(
            warehouse_df.index.levels[1].str.lower(), level=1
        )

    # 时间切片
    start_ts = pd.Timestamp(train_start)
    end_ts = pd.Timestamp(train_end)
    warehouse_df = warehouse_df.loc[start_ts:end_ts]

    # 2. 获取标签数据
    label_raw = D.features(
        global_feature_cache.resolved_instruments,
        [label_expr],
        train_start, train_end,
    )
    if label_raw.empty:
        raise ValueError(f"标签 {label_expr} 在 {train_start}~{train_end} 为空")
    if isinstance(label_raw.columns, pd.MultiIndex):
        label_raw.columns = label_raw.columns.droplevel(1)
    label_raw = label_raw.rename(columns={label_raw.columns[0]: label_name})
    label_flat = label_raw.reset_index()
    label_flat['instrument'] = label_flat['instrument'].str.lower()
    label_flat = label_flat.set_index(['datetime', 'instrument']).sort_index()

    # 3. 合并特征与标签
    full_train_frame = warehouse_df.join(label_flat, how='inner')
    before = len(full_train_frame)
    full_train_frame = full_train_frame.dropna(subset=[label_name])

    # 4. 后置过滤：ST / 次新股
    if filter_new_stocks or filter_st:
        full_train_frame = _filter_stocks_post(
            full_train_frame,
            filter_new_stocks=filter_new_stocks,
            filter_st=filter_st,
        )

    # 5. 截面排名标准化 (CSRankNorm，等价于树模型的 CSQuantileNorm)
    feature_cols = [c for c in full_train_frame.columns if c in all_factor_names]
    x_raw = full_train_frame[feature_cols].copy()
    x_norm = _apply_cs_rank_norm(x_raw)
    y = full_train_frame[label_name].copy()

    del warehouse_df, label_raw, label_flat, full_train_frame, x_raw
    gc.collect()

    return x_norm, y


def main():
    factor_files = resolve_factor_files(CONFIG["factor_files"])
    rolling_windows = CONFIG["rolling_windows"]

    print("=" * 60)
    print("  精选因子筛选脚本 (滚动窗口版)")
    print("  功能：按因子类别分组 → 滚动窗口独立特征选择 → 跨窗口聚合")
    print("  改进：股票池与训练脚本对齐、动态过滤、ST/次新过滤、CSRankNorm")
    print("=" * 60)
    print(f"  股票池: {CONFIG['instruments']}")
    print(f"  动态过滤: {CONFIG['use_dynamic_filter']}")
    print(f"  ST 过滤: {CONFIG['filter_st']}, 次新过滤: {CONFIG['filter_new_stocks']}")
    print(f"  窗口数: {len(rolling_windows)}, 聚合阈值: {CONFIG['min_window_ratio']}")

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading")

    # 2. 加载因子并按类别分组
    print(f"\n[2] 加载因子文件: {factor_files}")
    categories = load_factors_by_category(factor_files)
    if not categories:
        print("[错误] 未加载到任何因子")
        sys.exit(1)

    # 打印类别概览
    print(f"\n[3] 类别概览:")
    for cat_name, factors in sorted(categories.items()):
        print(f"    {cat_name:<20s}: {len(factors):3d} 个因子")
    print(f"    {'总计':-<20s}: {sum(len(v) for v in categories.values()):3d} 个因子")

    all_factor_names = [f["name"] for factors in categories.values() for f in factors]

    # ─────────────────────────────────────────────────────────────────────────
    # 一次性构建全局因子包和特征缓存（全时段覆盖所有窗口）
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4a] 构建全局因子包 (Global FeatureBundle)...")
    global_bundle = build_global_bundle(categories, CONFIG["label_expr"], CONFIG["label_name"])
    print(f"    >>> 合并后共 {len(global_bundle.fields)} 个因子表达式")

    full_start, full_end = _compute_window_full_period(rolling_windows)
    print(f"\n[4b] 构建全局特征缓存 (覆盖 {full_start} ~ {full_end})...")
    global_feature_cache = build_custom_feature_cache(
        instruments=CONFIG["instruments"],
        feature_bundle=global_bundle,
        factor_cache_names=[],
        start_time=full_start,
        end_time=full_end,
        use_dynamic_filter=CONFIG["use_dynamic_filter"],
    )
    print(f"    >>> 全局缓存构建完成（动态过滤={'是' if CONFIG['use_dynamic_filter'] else '否'}）")

    # ─────────────────────────────────────────────────────────────────────────
    # 逐窗口运行因子筛选
    # ─────────────────────────────────────────────────────────────────────────
    # window_selections[category][factor_name] = [selected_in_w0, selected_in_w1, ...]
    window_selections: Dict[str, Dict[str, List[bool]]] = {}
    window_importances: Dict[str, Dict[str, List[float]]] = {}
    window_details: List[pd.DataFrame] = []

    for win_idx, window in enumerate(rolling_windows):
        win_name = window["name"]
        train_start, train_end = window["train"]

        print(f"\n{'=' * 60}")
        print(f"=== 窗口 {win_idx+1}/{len(rolling_windows)}: {win_name} ===")
        print(f"    训练期: {train_start} ~ {train_end}")
        print(f"{'=' * 60}")

        # 准备该窗口的训练数据
        print(f"\n[5.{win_idx+1}a - {win_name}] 准备训练数据...")
        try:
            x_train, y_train = _prepare_window_data(
                global_feature_cache=global_feature_cache,
                all_factor_names=all_factor_names,
                train_start=train_start,
                train_end=train_end,
                label_expr=CONFIG["label_expr"],
                label_name=CONFIG["label_name"],
                filter_new_stocks=CONFIG["filter_new_stocks"],
                filter_st=CONFIG["filter_st"],
            )
        except ValueError as e:
            print(f"    [跳过] 窗口数据准备失败: {e}")
            continue

        print(f"    >>> 准备就绪: {x_train.shape[0]} 行, {x_train.shape[1]} 个特征")
        n_stocks = len(x_train.index.get_level_values("instrument").unique())
        n_days = len(x_train.index.get_level_values("datetime").unique())
        print(f"    >>> 股票数: {n_stocks}, 交易日: {n_days}")

        # 逐类别进行特征选择
        print(f"\n[5.{win_idx+1}b - {win_name}] 逐类别特征选择 (方法={CONFIG['method']}, 算法={CONFIG['algo']}, top_k={CONFIG['top_k']})...")
        cat_index = 0
        window_results = []

        for cat_name, factors in sorted(categories.items()):
            cat_index += 1
            if len(factors) < CONFIG["min_factors"]:
                print(f"\n  [{cat_index}/{len(categories)}] 类别 '{cat_name}' ({len(factors)} 个因子) — 跳过（因子数 < {CONFIG['min_factors']})")
                rows = []
                for f in factors:
                    rows.append({
                        "category": cat_name, "factor_name": f["name"],
                        "selected": True, "importance": 1.0, "rank": 1,
                        "meaning": f["meaning"], "source_file": f["source_file"],
                        "window": win_name,
                    })
                window_results.append(pd.DataFrame(rows))
                continue

            print(f"\n  >>> [{cat_index}/{len(categories)}] 类别: '{cat_name}' ({len(factors)} 个因子) <<<")
            df = run_single_category_selection(
                cat_name=cat_name,
                factors=factors,
                x_train=x_train,
                y_train=y_train,
                fs_method=CONFIG["method"],
                fs_algo=CONFIG["algo"],
                top_k=CONFIG["top_k"],
                label_name=CONFIG["label_name"],
                redundancy_check=CONFIG["redundancy_check"],
                redundancy_threshold=CONFIG["redundancy_threshold"],
                icir_stability=CONFIG["icir_stability"],
                icir_rolling_window=CONFIG["icir_window"],
                icir_keep_ratio=CONFIG["icir_keep_ratio"],
            )

            if df is not None:
                df["window"] = win_name
                print_category_results(cat_name, df)
                window_results.append(df)
            else:
                print(f"    [失败] 类别 '{cat_name}' 特征选择未完成")

        if window_results:
            win_df = pd.concat(window_results, ignore_index=True)
            window_details.append(win_df)

            # 记录窗口选择结果
            for _, row in win_df.iterrows():
                cat = row["category"]
                fname = row["factor_name"]
                sel = row["selected"]
                imp = row["importance"]

                if cat not in window_selections:
                    window_selections[cat] = {}
                    window_importances[cat] = {}
                if fname not in window_selections[cat]:
                    window_selections[cat][fname] = []
                    window_importances[cat][fname] = []
                window_selections[cat][fname].append(sel)
                window_importances[cat][fname].append(imp)

        # 释放当前窗口内存
        del x_train, y_train
        gc.collect()

    # 释放全局缓存
    del global_feature_cache
    gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    # 跨窗口聚合：因子在 >= min_window_ratio 的窗口中入选则最终选中
    # ─────────────────────────────────────────────────────────────────────────
    n_windows = len(window_details)
    if n_windows == 0:
        print("\n[错误] 没有任何窗口完成因子筛选")
        sys.exit(1)

    min_wins = max(1, int(n_windows * CONFIG["min_window_ratio"]))
    print(f"\n{'=' * 60}")
    print(f"  跨窗口聚合（需要 >= {min_wins}/{n_windows} 窗口入选）")
    print(f"{'=' * 60}")

    final_rows = []
    for cat_name in sorted(categories.keys()):
        cat_factors = categories[cat_name]
        for f in cat_factors:
            fname = f["name"]
            sel_history = window_selections.get(cat_name, {}).get(fname, [])
            imp_history = window_importances.get(cat_name, {}).get(fname, [])

            n_selected = sum(sel_history) if sel_history else 0
            avg_importance = float(np.mean(imp_history)) if imp_history else 0.0
            final_selected = n_selected >= min_wins

            final_rows.append({
                "category": cat_name,
                "factor_name": fname,
                "selected": final_selected,
                "importance": round(avg_importance, 4),
                "n_windows_selected": n_selected,
                "total_windows": n_windows,
                "meaning": f.get("meaning", ""),
                "source_file": f.get("source_file", ""),
            })

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(["category", "selected", "importance"],
                                     ascending=[True, False, False]).reset_index(drop=True)

    selected_total = final_df["selected"].sum()
    total = len(final_df)
    print(f"\n  总览: 共 {total} 个因子，选中 {selected_total} 个 ({selected_total/total*100:.1f}%)")
    print(f"  聚合标准: 在 {min_wins}/{n_windows} 个窗口中入选")

    print(f"\n  {'类别':<20s} {'选中/总计':<12s} {'选中率':<8s}")
    print(f"  {'-'*40}")
    for cat_name in sorted(final_df["category"].unique()):
        sub = final_df[final_df["category"] == cat_name]
        s = sub["selected"].sum()
        t = len(sub)
        print(f"  {cat_name:<20s} {int(s)}/{t:<8d} {s/t*100:>6.1f}%")

    print(f"\n  各类别精选因子列表:")
    for cat_name in sorted(final_df["category"].unique()):
        sub = final_df[(final_df["category"] == cat_name) & (final_df["selected"])]
        if len(sub) == 0:
            continue
        print(f"\n  【{cat_name}】({len(sub)} 个)")
        for _, row in sub.iterrows():
            print(f"    [x] {row['factor_name']:<30s} "
                  f"(平均重要性: {row['importance']:.3f}, "
                  f"入选窗口: {int(row['n_windows_selected'])}/{int(row['total_windows'])}, "
                  f"来源: {row['source_file']})")

    # 保存 CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = CONFIG["output"] or os.path.join(os.path.dirname(__file__), f"selected_factors_{timestamp}.csv")
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  完整结果已保存至: {output_path}")

    selected_df = final_df[final_df["selected"]].copy()
    selected_output = output_path.replace(".csv", "_selected.csv")
    selected_df.to_csv(selected_output, index=False, encoding="utf-8-sig")
    print(f"  精选因子列表已保存至: {selected_output}")

    # 保存逐窗口明细
    if window_details:
        detail_df = pd.concat(window_details, ignore_index=True)
        detail_output = output_path.replace(".csv", "_by_window.csv")
        detail_df.to_csv(detail_output, index=False, encoding="utf-8-sig")
        print(f"  逐窗口明细已保存至: {detail_output}")

    print("=" * 60)
    print(f"  精选因子数: {selected_total} (跨 {n_windows} 窗口聚合)")
    print(f"  训练模型: {CONFIG['algo']}")
    print(f"  股票池: {CONFIG['instruments']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
