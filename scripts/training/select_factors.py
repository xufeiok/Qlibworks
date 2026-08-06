"""
精选因子筛选脚本：按因子类别分组，在每个类别内独立运行特征选择，
筛选出各类别中预测能力最强的因子。

[AQR/Citadel/Renaissance 改进]
  - 滚动窗口因子筛选，消除前瞻偏差
  - 与 train_from_selected.py 统一的股票池、动态过滤、ST/次新过滤
  - 截面排名标准化 (CSQuantileNorm)，与训练阶段数据处理一致

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
import json
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
from qlworks.factors.filter_utils import filter_codes_post, filter_untradeable_labels
from qlworks.models import cached_select_features
from qlworks.models.training import compute_ic
from qlworks.processors.quantile_norm import CSQuantileNorm
from qlworks.processors.neutralize import CSNeutralize
from qlworks.config import QLIB_DATA_DIR
from qlworks.evaluation.selector import (
    check_redundancy, check_icir_stability, aggregate_across_windows,
)
from qlworks.pipeline_config import (
    LABEL_EXPR, LABEL_NAME, INSTRUMENTS,
    REDUNDANCY_THRESHOLD, ICIR_WINDOW, ICIR_KEEP_RATIO,
    FILTER_ST, FILTER_NEW_STOCKS, FILTER_LIMIT_UPDOWN,
)
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
    "factor_files": [f for f in ACTIVE_FACTOR_FILES ],

    # 特征选择参数
    "top_k": 2,
    "method": "embedded",
    "algo": "lightgbm",
    "min_factors": 3,

    # --- 股票池与过滤（引用 pipeline_config 单一事实源，与粗筛/评测/训练/实盘一致）---
    "instruments": INSTRUMENTS,
    "use_dynamic_filter": True,
    "filter_new_stocks": FILTER_NEW_STOCKS,
    "filter_st": FILTER_ST,
    "filter_untradeable_labels": FILTER_LIMIT_UPDOWN,  # 涨跌停/一字板/持仓期停牌标签置 NaN

    # --- 标签 DK_L 管线（与训练端 train_tree-doubao.py 的 neutralize_labels 逐位对齐）---
    # CSNeutralize(industry+mv) → CSQuantileNorm(label)，剥离风格暴露得到纯 alpha 标签。
    # 树模型路线下因子侧不做中性化（neutralize_features=False），"纯 alpha"口径由标签侧保证。
    "industry_field": "sw_l1",        # 行业分类字段（CSNeutralize 从 Qlib 拉取 $sw_l1）
    "market_cap_field": "circ_mv",    # 市值字段（CSNeutralize 从 Qlib 拉取 $circ_mv）
    "log_mc": True,                   # 市值对数化（中性化解释变量用 log 市值）

    # --- [P2-整合] 输入源：候选池白名单（Alpha Book，评测+三关准入后的因子）---
    # 精选端只评估"单因子评测→三关准入"通过的因子，杜绝漏斗绕过；
    # 白名单仅过滤 factor_library yaml 的加载结果（保留 8 MECE 分类学与表达式）。
    "pool_whitelist": True,           # 仅对 candidate_pool.json 中 status=admitted 因子做精选
    "pool_path": None,                # 候选池路径（None=默认 factor_data/registry/candidate_pool.json）

    # --- [P2-整合] IC 粗筛（来自 train_tree-doubao 跨窗口稳定 IC + stride）---
    # 嵌入法前先按 Spearman IC 粗筛类别内因子，消除低 IC 噪声因子的干扰；
    # 跨窗口稳定 IC：要求方向与历史窗口一致（同号占比>=0.5）并按均值|IC|打分。
    "ic_coarse": {
        "enabled": True,
        "stride": 2,                  # IC 采样隔日（降低序列自相关对 IC 的污染）
        "top_k_ratio": 0.6,           # 类别内 IC 粗筛保留比例（作为嵌入法候选池）
        "min_keep": 3,                # 至少保留因子数（IC 粗筛保底）
        "stable_ic": True,            # 跨窗口稳定 IC（首窗口退化为单窗口 |IC|）
    },

    # --- [P2-整合] 置换检验（来自 train_tree-doubao，向量化批量 Spearman）---
    # 对嵌入法选中的因子做显著性检验，剔除纯噪声因子（p >= 阈值）；
    # 显著因子过少时保留原结果（min_keep 保护，防因子池崩空）。
    "permutation_test": {
        "enabled": True,
        "n_permutations": 200,        # 置换次数（200 次足够 p 值精度）
        "pvalue_threshold": 0.05,     # 双尾 p 值阈值
        "min_keep": 3,                # 至少保留显著因子数
    },

    # --- [P2-整合] 自适应配额（来自 train_tree 自适应 top_k）---
    # 嵌入法 max_features = min(max, max(min, ceil(len(候选) * ratio)))，
    # 类别规模小时自动放宽，规模大时收紧，避免固定 top_k 造成类别间失衡。
    "adaptive_top_k": {
        "enabled": True,
        "min": 3,
        "max": 10,
        "ratio": 0.6,
    },

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

    # 标签（共享配置，与 screen_factors / train_tree-doubao / 评测端一致）
    "label_expr": LABEL_EXPR,
    "label_name": LABEL_NAME,

    # 冗余检测（阈值引用 pipeline_config 单一事实源，与粗筛/训练端统一）
    "redundancy_check": True,
    "redundancy_threshold": REDUNDANCY_THRESHOLD,

    # ICIR 稳定性校验（引用共享配置，与粗筛/训练端统一）
    "icir_stability": True,
    "icir_window": ICIR_WINDOW,
    "icir_keep_ratio": ICIR_KEEP_RATIO,

    # 跨窗口聚合：因子在 >= min_window_ratio 比例的窗口中入选才最终选中
    "min_window_ratio": 0.5,

    # 输出：None 自动生成时间戳文件名
    "output": None,

    # [P1-6] 精选结果回写候选池（Alpha Book）作为准入建议（不直接 admitted）
    # 候选池的最终准入仍由 admit_to_multifactor.py 三关检验唯一决定。
    "write_pool": False,

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


def _apply_cs_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    截面排名标准化 (CSQuantileNorm)：对每个交易日截面，特征值转为 [0,1] 分位数，
    缺失值填充为 0.5（中位数）。

    基于 qlworks.processors.quantile_norm.CSQuantileNorm 实现，支持大截面
    分块处理 (CS_QUANTILE_DATE_CHUNK_SIZE=256)，降低峰值内存。

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列

    输出:
    - 标准化后的 DataFrame，同 shape
    """
    result = CSQuantileNorm().transform(df)
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


def _load_pool_whitelist(pool_path: Optional[str] = None) -> Optional[set]:
    """
    从候选池（Alpha Book）读取 status=admitted 的因子名集合，作为精选白名单。

    输入:
    - pool_path: 候选池 JSON 路径（None=默认 factor_data/registry/candidate_pool.json）

    输出:
    - 返回 admitted 因子名 set；候选池缺失/解析失败/无 admitted 因子时返回 None
      （调用方回退到全库加载，保证精选流程可运行）。
    """
    default_pool = (Path(__file__).resolve().parents[2]
                    / "factor_data" / "registry" / "candidate_pool.json")
    path = Path(pool_path) if pool_path else default_pool
    if not path.exists():
        print(f"  [白名单] 候选池不存在: {path}，跳过白名单过滤（回退全库因子）")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            pool = json.load(f)
        names = [e["name"] for e in pool.get("factors", [])
                 if e.get("status") == "admitted"]
        if not names:
            print("  [白名单] 候选池无 admitted 因子，跳过白名单过滤（回退全库因子）")
            return None
        print(f"  [白名单] 候选池 admitted 因子 {len(names)} 个（来自 {path.name}）")
        return set(names)
    except Exception as e:
        print(f"  [白名单] 候选池解析失败: {e}，跳过白名单过滤（回退全库因子）")
        return None


def _ic_coarse_select(
    x_sub: pd.DataFrame,
    y_sub: pd.Series,
    factor_names: List[str],
    ic_history: Optional[List[Dict]] = None,
    stride: int = 2,
    top_k_ratio: float = 0.6,
    min_keep: int = 3,
    stable_ic: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    """
    类别内 IC 粗筛：逐因子计算与 DK_L 标签的 Spearman IC，
    支持跨窗口稳定 IC（方向一致性 + 均值 |IC|）与隔日 stride 采样。

    输入:
    - x_sub: 类别因子矩阵（MultiIndex (datetime, instrument) × 因子列，已 CSRankNorm）
    - y_sub: DK_L 标签 Series（与 x_sub 行对齐）
    - factor_names: 类别内因子名列表（待粗筛）
    - ic_history: 跨窗口 IC 历史（list of dict，每元素 = {因子名: 该窗口IC}，时间升序）
    - stride: 隔日采样步长（降低序列自相关对 IC 的污染）
    - top_k_ratio: 保留比例（候选数 = max(ceil(len * ratio), min_keep)）
    - min_keep: 至少保留因子数
    - stable_ic: 是否启用跨窗口稳定 IC 评分

    输出:
    - (候选因子名列表[按 IC 降序], 本窗口 {因子名: IC})
    """
    avail = [c for c in factor_names if c in x_sub.columns]
    if not avail:
        return [], {}
    x_ic = x_sub[avail]
    y_ic = y_sub

    # stride 降采样：仅对 IC 计算的数据按日隔步采样（嵌入法仍用全量）
    if stride > 1:
        _dates = x_ic.index.get_level_values("datetime").unique()[::stride]
        _mask = x_ic.index.get_level_values("datetime").isin(_dates)
        x_ic = x_ic.loc[_mask]
        y_ic = y_ic.loc[_mask]

    # 逐因子 Spearman IC（列级独立 dropna，与粗筛/训练端 compute_ic 口径一致）
    ics = {}
    for col in avail:
        feat = x_ic[col].dropna()
        lab = y_ic.reindex(feat.index).dropna()
        common = feat.index.intersection(lab.index)
        if len(common) < 50:
            continue
        try:
            ics[col] = compute_ic(feat.loc[common], lab.loc[common])
        except Exception:
            ics[col] = 0.0

    # 跨窗口稳定 IC 评分：当前窗口方向与历史均值一致，且历史同号占比 >= 0.5
    stable_score = {}
    for factor, ic_val in ics.items():
        score = abs(float(ic_val))
        if stable_ic and ic_history:
            hist_ics = [h.get(factor) for h in ic_history]
            hist_ics = [x for x in hist_ics
                        if x is not None and not (isinstance(x, float) and np.isnan(x))]
            if len(hist_ics) > 0:
                hist_mean = float(np.mean(hist_ics))
                if np.sign(ic_val) != np.sign(hist_mean):
                    continue  # 方向反转 → 不稳定，剔除
                _all = hist_ics + [ic_val]
                same_ratio = np.mean([1.0 if np.sign(x) == np.sign(ic_val) else 0.0 for x in _all])
                if same_ratio < 0.5:
                    continue
                score = abs(float(np.mean(_all))) * same_ratio
        stable_score[factor] = score

    ranked = sorted(stable_score.keys(), key=lambda k: stable_score[k], reverse=True)
    keep_n = max(int(np.ceil(len(ranked) * top_k_ratio)), min_keep) if ranked else 0
    return ranked[:keep_n], ics


def _permutation_significance(
    x_sub: pd.DataFrame,
    y_sub: pd.Series,
    selected_names: List[str],
    n_perms: int = 200,
    pvalue_threshold: float = 0.05,
    min_keep: int = 3,
) -> List[str]:
    """
    置换检验：对选中的因子做双尾显著性检验（向量化批量 Spearman）。

    输入:
    - x_sub: 类别因子矩阵（含 selected_names 列）
    - y_sub: DK_L 标签 Series（与 x_sub 行对齐）
    - selected_names: 待检验因子名列表
    - n_perms: 置换次数
    - pvalue_threshold: 双尾 p 值阈值
    - min_keep: 显著因子少于该值时保留原列表（防因子池崩空）

    输出:
    - 显著因子名列表（保序；显著不足时返回原列表）
    """
    from scipy.stats import rankdata

    avail = [c for c in selected_names if c in x_sub.columns]
    if len(avail) < 2:
        return selected_names
    comb = x_sub[avail].join(y_sub.rename("_y"), how="inner").dropna()
    if len(comb) < 50:
        return selected_names

    feat_cols = [c for c in comb.columns if c != "_y"]
    X = comb[feat_cols].values.astype(np.float64)
    y = comb["_y"].values.astype(np.float64)
    n_samples, n_factors = X.shape

    # 预排名 X（不变，只排一次）+ 去中心化/标准化，后续每次置换只需一次矩阵乘
    X_ranked = np.apply_along_axis(rankdata, 0, X)
    X_centered = X_ranked - np.mean(X_ranked, axis=0)
    X_std = np.std(X_ranked, axis=0, ddof=1)
    X_std[X_std == 0] = 1.0

    def _spearman_batch(y_vec: np.ndarray) -> np.ndarray:
        y_r = rankdata(y_vec).astype(np.float64)
        y_c = y_r - np.mean(y_r)
        y_s = np.std(y_r, ddof=1)
        if y_s == 0:
            return np.zeros(n_factors)
        return (X_centered.T @ y_c) / ((n_samples - 1) * X_std * y_s)

    real_ic = _spearman_batch(y)
    rng = np.random.RandomState(42)
    perm_matrix = np.zeros((n_perms, n_factors))
    for p in range(n_perms):
        y_shuffled = y[rng.permutation(n_samples)]
        perm_matrix[p, :] = _spearman_batch(y_shuffled)

    p_values = np.mean(np.abs(perm_matrix) >= np.abs(real_ic), axis=0)
    significant = [feat_cols[i] for i in range(n_factors) if p_values[i] < pvalue_threshold]

    n_sig = len(significant)
    print(f"    [置换检验] 显著 {n_sig}/{n_factors} 个 (p<{pvalue_threshold})")
    if n_sig < min_keep:
        print(f"    [置换检验] 显著因子过少({n_sig}<{min_keep})，保留原筛选结果")
        return selected_names
    return significant


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
    ic_history: Optional[List[Dict]] = None,   # [P2] 跨窗口 IC 历史（函数内 append 本窗口 IC）
    ic_coarse_conf: Optional[Dict] = None,     # [P2] IC 粗筛配置（stride/稳定 IC/比例）
    perm_conf: Optional[Dict] = None,          # [P2] 置换检验配置
    adaptive_conf: Optional[Dict] = None,      # [P2] 自适应配额配置
) -> Optional[pd.DataFrame]:
    """
    在单个类别上运行特征选择。

    [性能优化] 不再创建 Qlib dataset，直接从预准备的 x_train/y_train 中按因子名切片。
    CSRankNorm 已在主流程中运行一次，此处直接复用结果。

    [P2-整合] 完整筛选链：IC 粗筛(跨窗口稳定 IC+stride) → 嵌入法(自适应 top_k)
    → 置换检验(显著性) → 冗余检测 → ICIR 稳定校验。

    输入:
    - cat_name: 类别名称
    - factors: 该类别因子列表 [{name, expression, meaning, source_file}, ...]
    - x_train: 已标准化 (CSRankNorm) 的特征矩阵
    - y_train: DK_L 标签 Series
    - fs_method, fs_algo, top_k: 特征选择参数
    - label_name: 标签列名（用于 ICIR 校验）
    - redundancy_check: 是否做冗余检测
    - redundancy_threshold: Spearman 相关系数阈值
    - icir_stability: 是否做 ICIR 稳定性校验
    - icir_rolling_window: ICIR 滚动窗口天数
    - icir_keep_ratio: ICIR 保留比例
    - ic_history: 跨窗口 IC 历史（list of dict，时间升序；本窗口 IC 会追加到列表尾）
    - ic_coarse_conf / perm_conf / adaptive_conf: [P2-整合] 新增筛选阶段配置
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

    # [P2-整合] IC 粗筛：跨窗口稳定 IC + stride（候选池收缩，剔除低/不稳定 IC 因子）
    ic_cfg = ic_coarse_conf or {}
    perm_cfg = perm_conf or {}
    adap_cfg = adaptive_conf or {}
    coarse_pool = available
    window_ic = {}
    if ic_cfg.get("enabled", True) and len(available) > ic_cfg.get("min_keep", 3):
        coarse_pool, window_ic = _ic_coarse_select(
            x_cat, y_cat, available,
            ic_history=ic_history,
            stride=ic_cfg.get("stride", 2),
            top_k_ratio=ic_cfg.get("top_k_ratio", 0.6),
            min_keep=ic_cfg.get("min_keep", 3),
            stable_ic=ic_cfg.get("stable_ic", True),
        )
        removed_coarse = len(available) - len(coarse_pool)
        if removed_coarse > 0:
            print(f"    [IC 粗筛] 剔除 {removed_coarse} 个低/不稳定 IC 因子，候选 {len(coarse_pool)} 个")
        if len(coarse_pool) < 2:
            coarse_pool = available  # 保护：粗筛后过少则回退全类别
    # 记录本窗口 IC（供后续窗口跨窗口稳定 IC 判断）
    if ic_history is not None and window_ic:
        ic_history.append(window_ic)

    x_cat = x_train[coarse_pool]
    y_cat = y_train

    print(f"      >>> {x_cat.shape[0]} 行, {x_cat.shape[1]} 个特征")

    # [P2-整合] 自适应 top_k：随候选规模动态调整嵌入法 max_features
    eff_top_k = top_k
    if adap_cfg.get("enabled", True):
        _a_min = adap_cfg.get("min", 3)
        _a_max = adap_cfg.get("max", 10)
        _a_ratio = adap_cfg.get("ratio", 0.6)
        eff_top_k = min(_a_max, max(_a_min, int(np.ceil(len(coarse_pool) * _a_ratio))))
        eff_top_k = min(eff_top_k, len(coarse_pool))
    print(f"    [自适应] 候选 {len(coarse_pool)} 个, top_k={eff_top_k}")

    # 特征选择
    try:
        if fs_method == "embedded":
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo, threshold=0.0,
                model_kwargs={"max_features": min(eff_top_k, len(coarse_pool)), "importance_type": "gain"},
                remove_collinearity=False,
            )
        elif fs_method == "filter":
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                k=min(eff_top_k, len(coarse_pool)),
                remove_collinearity=False,
            )
        else:
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                model_kwargs={"max_features": min(eff_top_k, len(coarse_pool))},
                remove_collinearity=False,
            )
    except Exception as e:
        print(f"    [错误] 特征选择失败: {e}")
        return None

    selected_set = set(fs_result.selected_features)
    scores = fs_result.feature_scores
    selected_factor_names = list(selected_set)

    # [P2-整合] 置换检验：剔除纯噪声因子（p >= 阈值，双尾）
    if perm_cfg.get("enabled", True) and len(selected_factor_names) >= 3:
        try:
            selected_factor_names = _permutation_significance(
                x_cat, y_cat, selected_factor_names,
                n_perms=perm_cfg.get("n_permutations", 200),
                pvalue_threshold=perm_cfg.get("pvalue_threshold", 0.05),
                min_keep=perm_cfg.get("min_keep", 3),
            )
            selected_set = set(selected_factor_names)
        except Exception as e:
            print(f"    [置换检验] 异常: {e}，保留原筛选结果")

    # 冗余检测：来自 x_train 的因子间相关（逻辑收敛于 selector.check_redundancy）
    if redundancy_check and len(selected_factor_names) > 5:
        print(f"    [冗余检测] 阈值={redundancy_threshold}，检测 {len(selected_factor_names)} 个入选因子...")
        feat_in_data = [c for c in selected_factor_names if c in x_train.columns]
        if len(feat_in_data) > 5:
            try:
                importance_map = {k: abs(v) for k, v in zip(scores.index, scores.values)}
                kept = check_redundancy(
                    x_train[feat_in_data], feat_in_data,
                    threshold=redundancy_threshold,
                    rank=importance_map,
                )
                to_drop = set(selected_factor_names) - set(kept)
                selected_factor_names = kept
                selected_set = set(selected_factor_names)
                print(f"      冗余检测完成: 剔除 {len(to_drop)} 个冗余因子，保留 {len(selected_factor_names)} 个")
                if to_drop:
                    for f in sorted(to_drop):
                        imp = abs(importance_map.get(f, 0))
                        print(f"      冗余剔除: {f}(imp={imp:.4f})")
            except Exception as e:
                print(f"      [跳过] 冗余检测异常: {e}")

    # [Citadel] ICIR 稳定性校验（使用 x_train + y_train 构建面板，逻辑收敛于 selector.check_icir_stability）
    if icir_stability and len(selected_factor_names) > 5:
        print(f"    [ICIR 稳定校验] 窗口={icir_rolling_window}d, keep_ratio={icir_keep_ratio}...")
        try:
            icir_feat = [c for c in selected_factor_names if c in x_train.columns]
            if len(icir_feat) > 5:
                stable_factors = check_icir_stability(
                    x_train[icir_feat], y_train, icir_feat,
                    rolling_window=icir_rolling_window,
                    keep_ratio=icir_keep_ratio,
                    min_keep=min(top_k, len(icir_feat)),
                )
                dropped = len(selected_factor_names) - len(stable_factors)
                if dropped > 0:
                    print(f"      ICIR 检测: {dropped} 个不稳定因子被剔除")
                    selected_factor_names = stable_factors
                    selected_set = set(selected_factor_names)
                else:
                    print(f"      所有因子 ICIR 稳定，无需剔除")
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
    filter_untradeable_labels: bool = True,
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
    log_mc: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    为单个滚动窗口准备训练数据：切片时间范围 → 合并标签 → 标准化 → 过滤。

    返回: (x_train, y_train)
    """
    # 1. 从缓存切片特征数据（按需惰性合并）
    warehouse_df = global_feature_cache.get_warehouse_df(
        selected_names=all_factor_names,
        start_time=train_start,
        end_time=train_end,
    )
    if warehouse_df.empty:
        return pd.DataFrame(), pd.Series(dtype="float64")
    # 合并后 warehouse_df 列有 MultiIndex ("feature", factor_name)，降为单层
    if isinstance(warehouse_df.columns, pd.MultiIndex):
        warehouse_df.columns = warehouse_df.columns.droplevel(0)
    # get_warehouse_df 返回的索引是 ["instrument", "datetime"] 顺序，
    # 调整为 ["datetime", "instrument"] 以匹配后续 join 的 label 数据
    if isinstance(warehouse_df.index, pd.MultiIndex) and warehouse_df.index.names == ["instrument", "datetime"]:
        warehouse_df = warehouse_df.swaplevel().sort_index()
        warehouse_df.index.names = ["datetime", "instrument"]

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

    # [数据质量] 标签可交易性过滤（与粗筛/训练端一致）：
    # 涨跌停/一字板/持仓期停牌样本标签置 NaN，避免精选结论来自不可交易样本
    if filter_untradeable_labels:
        label_raw = filter_untradeable_labels(
            label_raw, global_feature_cache.resolved_instruments, train_start, train_end
        )
        if label_raw is None or label_raw.empty:
            raise ValueError(f"标签 {label_expr} 经可交易性过滤后为空")

    label_flat = label_raw.reset_index()
    label_flat['instrument'] = label_flat['instrument'].str.lower()
    label_flat = label_flat.set_index(['datetime', 'instrument']).sort_index()

    # 3. 合并特征与标签
    full_train_frame = warehouse_df.join(label_flat, how='inner')
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

    # [P0] 标签 DK_L 管线：与训练端 train_tree-doubao.py 的 neutralize_labels 逐位对齐。
    #   ① CSNeutralize(industry+mv)：Ridge 回归剥离行业/市值风格暴露（纯 alpha 口径）
    #   ② CSQuantileNorm(label)：截面分位化，消除收益分布偏度与极端值影响
    #   若标签侧不剥离风格，模型/筛选 IC 会混入市值/行业 beta（与训练端标签错位）。
    y_mi = full_train_frame[[label_name]].copy()
    y_mi.columns = pd.MultiIndex.from_tuples([("label", label_name)])
    y_mi = CSNeutralize(
        fields_group="label",
        industry_field=industry_field,
        market_cap_field=market_cap_field,
        log_mc=log_mc,
    ).__call__(y_mi)
    y_mi = CSQuantileNorm(fields_group="label").__call__(y_mi)
    y = y_mi[("label", label_name)].rename(label_name)

    # 中性化后无行业/市值暴露样本的标签为 NaN，与特征同步剔除保持行对齐
    valid_mask = y.notna()
    y = y[valid_mask]
    x_norm = x_norm.loc[valid_mask]

    del warehouse_df, label_raw, label_flat, full_train_frame, x_raw, y_mi
    gc.collect()

    return x_norm, y


def _write_pool_suggestions(final_df: pd.DataFrame) -> None:
    """[P1-6] 精选结果回写候选池（Alpha Book）建议区。

    精选因子作为"准入建议"写入 candidate_pool.json 的 suggestions 区，
    状态为 suggested，**不直接 admitted**——最终准入由
    admit_to_multifactor.py 三关检验（相关性/边际贡献/方向一致）唯一决定。

    输入：
    - final_df: 跨窗口聚合结果（含 factor_name / importance / n_windows_selected 等列）
    """
    from datetime import datetime
    pool_path = (Path(__file__).resolve().parents[2]
                 / "factor_data" / "registry" / "candidate_pool.json")
    if not pool_path.exists():
        print(f"[write_pool] 候选池不存在: {pool_path}，跳过回写（先运行 admit_to_multifactor.py --build-all）")
        return

    selected = final_df[final_df["selected"]]
    if selected.empty:
        print("[write_pool] 无精选因子，跳过回写")
        return

    try:
        with open(pool_path, "r", encoding="utf-8") as f:
            pool = json.load(f)
    except Exception as e:
        print(f"[write_pool] 读取候选池失败({e})，跳过回写")
        return

    now = datetime.now().isoformat(timespec="seconds")
    suggestions = pool.setdefault("suggestions", [])
    existing_names = {s.get("name") for s in suggestions}
    added = 0
    for _, row in selected.iterrows():
        name = row["factor_name"]
        if name in existing_names:
            continue
        suggestions.append({
            "name": name,
            "category": row.get("category", ""),
            "sub_category": row.get("sub_category", ""),
            "importance": round(float(row.get("importance", 0)), 4),
            "n_windows_selected": int(row.get("n_windows_selected", 0)),
            "total_windows": int(row.get("total_windows", 0)),
            "status": "suggested",          # 仅建议，待 admit 三关准入
            "suggested_at": now,
        })
        existing_names.add(name)
        added += 1

    if added > 0:
        pool["_meta"]["updated_at"] = now
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, ensure_ascii=False, indent=2)
        print(f"[write_pool] 已回写 {added} 个精选因子建议至候选池: {pool_path}")
        print("           建议因子需经 admit_to_multifactor.py 三关检验后方可 admitted（训练因子池）")
    else:
        print("[write_pool] 无新增建议因子（均已存在）")


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

    # [P2-整合] 候选池白名单：仅保留"单因子评测→三关准入"通过的因子（Alpha Book）
    if CONFIG.get("pool_whitelist", True):
        whitelist = _load_pool_whitelist(CONFIG.get("pool_path"))
        if whitelist is not None:
            _before = sum(len(v) for v in categories.values())
            categories = {
                cat: [f for f in factors if f["name"] in whitelist]
                for cat, factors in categories.items()
            }
            categories = {cat: fs for cat, fs in categories.items() if fs}
            _after = sum(len(v) for v in categories.values())
            print(f"    [白名单] 全库 {_before} → 准入 {_after} 个因子"
                  f"（剔除 {_before - _after} 个未过评测/准入因子）")
            if not categories:
                print("[错误] 白名单过滤后无可用因子")
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
    # [P2-整合] 跨窗口 IC 历史：{category: [{factor: ic}, ...窗口]}，供稳定 IC 粗筛使用
    cat_ic_history: Dict[str, List[Dict]] = {}

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
                filter_untradeable_labels=CONFIG.get("filter_untradeable_labels", True),
                industry_field=CONFIG.get("industry_field", "sw_l1"),
                market_cap_field=CONFIG.get("market_cap_field", "circ_mv"),
                log_mc=CONFIG.get("log_mc", True),
            )
        except ValueError as e:
            print(f"    [跳过] 窗口数据准备失败: {e}")
            continue

        if x_train.empty or y_train.empty:
            print(f"    [跳过] 窗口数据为空（无有效样本）")
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
                ic_history=cat_ic_history.setdefault(cat_name, []),
                ic_coarse_conf=CONFIG.get("ic_coarse", {}),
                perm_conf=CONFIG.get("permutation_test", {}),
                adaptive_conf=CONFIG.get("adaptive_top_k", {}),
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

    # 向上取整保证"过半窗口"语义，与 selector.aggregate_across_windows 口径一致
    min_wins = max(1, int(np.ceil(n_windows * CONFIG["min_window_ratio"])))
    print(f"\n{'=' * 60}")
    print(f"  跨窗口聚合（需要 >= {min_wins}/{n_windows} 窗口入选）")
    print(f"{'=' * 60}")

    final_rows = []
    for cat_name in sorted(categories.keys()):
        cat_factors = categories[cat_name]
        # 跨窗口入选判定收敛于 selector.aggregate_across_windows
        # （{fname: [bool,...]} 输入与 window_selections[cat] 结构一致）
        agg = aggregate_across_windows(
            window_selections.get(cat_name, {}),
            min_window_ratio=CONFIG["min_window_ratio"],
        )
        for f in cat_factors:
            fname = f["name"]
            imp_history = window_importances.get(cat_name, {}).get(fname, [])
            avg_importance = float(np.mean(imp_history)) if imp_history else 0.0
            info = agg.get(fname, {"selected": False, "n_selected": 0})

            final_rows.append({
                "category": cat_name,
                "factor_name": fname,
                "selected": bool(info["selected"]),
                "importance": round(avg_importance, 4),
                "n_windows_selected": info["n_selected"],
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

    # [P1-6] 精选结果回写候选池（作为准入建议，不直接 admitted）
    if CONFIG.get("write_pool"):
        _write_pool_suggestions(final_df)

    print("=" * 60)
    print(f"  精选因子数: {selected_total} (跨 {n_windows} 窗口聚合)")
    print(f"  训练模型: {CONFIG['algo']}")
    print(f"  股票池: {CONFIG['instruments']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
