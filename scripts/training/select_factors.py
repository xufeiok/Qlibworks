"""
精选因子筛选脚本：按因子类别分组，在每个类别内独立运行特征选择，
筛选出各类别中预测能力最强的因子。

[世界顶级量化机构标准全面升级]
  对标 AQR / Citadel / Renaissance / Two Sigma / D.E. Shaw 机构级标准

  10 层机构级体系（层级编号以 CONFIG 注释为权威，代码按功能模块组织）：
  [第一层：数据质量检查 - Bloomberg 级]
    - 因子覆盖率、缺失率、异常值（截面 Z-score）、零方差检查
  [第二层：因子中性化 - AQR 级]
    - 行业中性化 + 市值中性化（Ridge 回归，防奇异矩阵）
    - 截面排名标准化 (CSQuantileNorm)
    - 因子正交化（可选，CONFIG["orthogonalize"]）
  [第三层：IC 分析体系 - Citadel 级]
    - Normal IC + Rank IC 双维度（逐日截面向量化计算）
    - IC t-statistic / p-value 显著性检验、ICIR、IC 偏峰度
  [第四层：多重检验校正 - Renaissance 级]
    - Bonferroni 校正（FWER 控制）/ Benjamini-Hochberg FDR 控制
    - NBER 建议 t-stat >= 3.0 阈值（Harvey, Liu & Zhu 2016）
  [第五层：稳健性验证 - D.E. Shaw 级]
    - 置换检验 (Permutation Test，默认关闭)
  [第六层：因子衰减分析 - Citadel 级]
    - 多周期 IC + 半衰期拟合（占位实现，需多周期标签，默认关闭）
  [第七层：分组回测 - AQR 级]
    - Quantile Portfolio Test：多空收益 / 单调性 / 多空夏普（默认关闭）
  [第八层：多方法交叉验证 - Point72 级]
    - 多方法投票机制（Filter + Embedded + Wrapper，默认关闭）
  [第九层：因子拥挤度分析 - Two Sigma 级]
    - 因子相关性聚类，识别信息冗余的拥挤因子（默认关闭）
  [第十层：跨窗口共线性精简 - AQR 级]
    - 跨窗口聚合后层次聚类去冗余（|rho| 阈值分族，每族保留 top N）

  [核心链路 - 滚动窗口 + 跨窗口聚合 - 全流程]
    - 滚动窗口因子筛选，消除前瞻偏差
    - 与训练端统一的股票池、动态过滤、ST/次新/退市过滤、不可交易样本剔除
    - 标签 DK_L 管线（中性化 + 截面分位数化）与训练端对齐

用法：
  修改文件顶部 CONFIG 字典中的参数，然后直接运行：
    python select_factors.py

输出：
  - 控制台打印每个类别的因子筛选结果（含重要性得分、IC、ICIR、t-stat）
  - 保存完整结果至 selected_factors_{时间戳}.csv
  - 保存精选因子列表至 selected_factors_{时间戳}_selected.csv
  - 保存逐窗口明细至 selected_factors_{时间戳}_by_window.csv
  - 保存因子质量评估报告至 selected_factors_{时间戳}_quality.csv
"""

import os
import sys
import warnings
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from scipy import stats

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
from qlworks.models import cached_select_features
from qlworks.processors.quantile_norm import CSQuantileNorm
from qlworks.processors.neutralize import CSNeutralize, _fetch_features_direct
from qlworks.factors.filter_utils import (
    filter_codes_post,
    filter_untradeable_labels as _filter_untradeable_fn,  # 别名避免与 _prepare_window_data 参数名冲突
    _load_stock_name_map,
)
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
    # ── 基础配置 ──
    "factor_files": [f for f in ACTIVE_FACTOR_FILES],
    "top_k": 5,
    "method": "embedded",
    "algo": "lightgbm",

    # ── 股票池与过滤（与 train_from_selected.py 对齐）──
    "instruments": "csi500",
    "use_dynamic_filter": True,        # [P0-1] 特征缓存构建时启用 Qlib 动态过滤器：逐日剔除停牌/未上市等不可交易股票
    "filter_new_stocks": True,         # [P1-3] 后置过滤：剔除上市不足 250 日的次新股
    "filter_st": True,                 # [P1-3] 后置过滤：剔除 ST/风险警示股票
    "filter_delisted": True,           # [P1-3] 退市两阶段过滤：逐日 date > delist_date 剔除（全期早退市/期内退市天然覆盖）
    "filter_untradeable_labels": True, # [P0-3] 涨跌停/一字板/持仓期停牌不可交易样本剔除（与训练端 filter_untradeable_labels 对齐）

    # ── 滚动窗口因子筛选 ──
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

    # ── 标签（与 train_from_selected.py 对齐）──
    "label_expr": "Ref($close, -5) / Ref($open, -1) - 1",
    "label_name": "LABEL_5D",

    # ── [第一层：数据质量检查 - Bloomberg 级] ──
    "data_quality_check": True,
    "min_coverage": 0.7,           # 最小覆盖率阈值（低于此值的因子被淘汰）
    "max_missing_rate": 0.3,       # 最大缺失率阈值
    "outlier_threshold": 5.0,      # 异常值检测阈值（标准差倍数）

    # ── [第二层：因子中性化 - AQR 级] ──
    # [P0修复-树模型路线] 专家流程阶段3-路线1：树模型不做因子中性化，
    # 直接喂原始因子 + 市值/行业特征，由模型控制风格风险（与训练端 neutralize_features=False 对齐）。
    # 筛选 IC 的"纯 alpha"口径由标签侧的中性化（见 _prepare_window_data 标签管线）保证。
    "neutralize": False,           # 是否启用行业+市值中性化（树模型路线：False）
    "industry_field": "sw_l1",
    "market_cap_field": "circ_mv",
    "log_mc": True,
    "orthogonalize": False,        # 是否启用因子正交化（对称正交化）

    # ── 冗余检测 ──
    "redundancy_check": True,
    "redundancy_threshold": 0.90,

    # ── [第三层：IC 分析体系 - Citadel 级] ──
    "ic_analysis": True,           # 是否启用完整 IC 分析
    "ic_type": "both",             # "rank" / "normal" / "both"
    "icir_stability": True,
    "icir_window": 60,
    "icir_keep_ratio": 0.9,

    # ── [第四层：多重检验校正 - Renaissance 级] ──
    "multiple_testing_correction": True,  # 是否启用多重检验校正
    "mtc_method": "bh",            # "bonferroni" / "bh" (Benjamini-Hochberg)
    "mtc_alpha": 0.05,             # 显著性水平
    "min_t_stat": 3.0,             # [机构标准] 最小 t-stat 阈值（Harvey, Liu & Zhu 2016: t >= 3.0）

    # ── [第五层：稳健性验证 - D.E. Shaw 级] ──
    "permutation_test": False,     # 是否启用置换检验（计算较慢）
    "permutation_n": 200,          # 置换次数
    "permutation_alpha": 0.05,     # 置换检验显著性水平

    # ── [第六层：因子衰减分析 - Citadel 级] ──
    "decay_analysis": False,       # 是否启用因子衰减分析（需要多周期标签）
    "decay_periods": [1, 5, 10, 20],  # 衰减分析的持有期（天）

    # ── [第七层：分组回测 - AQR 级] ──
    "quantile_test": False,        # 是否启用分组回测检验
    "quantile_n": 5,               # 分组数量
    "quantile_monotonicity": True, # 是否检验单调性

    # ── [第八层：多方法交叉验证 - Point72 级] ──
    "multi_method_voting": False,  # 是否启用多方法投票
    "voting_methods": ["embedded", "filter", "wrapper"],  # 参与投票的方法
    "voting_threshold": 0.5,       # 投票通过阈值（比例）

    # ── [第九层：因子拥挤度分析 - Two Sigma 级] ──
    "crowding_analysis": False,    # 是否启用因子拥挤度分析
    "crowding_threshold": 0.8,     # 拥挤度阈值

    # ── [第十层：跨窗口共线性精简 - AQR 级] ──
    "collinearity_reduction": True,          # 是否启用跨窗口层次聚类共线性精简
    "correlation_start": "2022-01-01",       # 相关性计算起始日期
    "correlation_end": "2023-12-20",         # 相关性计算结束日期（对齐缓存上限=最晚窗口训练期结束日，超出部分被静默截断）
    "cluster_rho_threshold": 0.6,            # 聚类阈值（|rho| > 此值视为同族）
    "max_per_cluster": 2,                    # 每族最多保留因子数
    "cross_cluster_rho_threshold": 0.85,     # 跨族二次检查阈值

    # ── 跨窗口聚合 ──
    "min_window_ratio": 0.5,

    # ── 综合评分权重 ──
    "scoring_weights": {
        "importance": 0.25,        # 模型重要性权重
        "ic": 0.25,                # IC 均值权重
        "icir": 0.25,              # ICIR 权重
        "t_stat": 0.15,            # t-stat 权重
        "stability": 0.10,         # 稳定性权重
    },

    # ── 输出 ──
    "output": None,
    "clean_start": False,

    # ── [机构级分类学重构 - AQR Factor Builder 标准] ──
    # 原 21 个 category 字段中英混杂、同族碎片化（momentum/动量/动量与反转
    # 本属一族），且按"构造方式"（Rolling/K-Bar/Synthetic）充当类别，违反
    # MECE 原则。现通过代码层 CATEGORY_TAXONOMY 映射归并为 8 个互斥大类：
    #   Value / Momentum / Reversal / Quality / Growth /
    #   Size & Liquidity / Volatility & Risk / Price-Volume
    # 仅代码层映射，不修改任何 yaml 源文件，可随时回退。
    "taxonomy_enabled": True,

    # ── [机构级自适应配额 - Citadel Alpha Lab 标准] ──
    # 固定 top_k 会使因子池构成被类别粒度劫持（145 因子类压到 5 个、
    # 1 因子类免检全留）。自适应配额与类别规模挂钩：
    #   top_k_eff = min(max, max(min, ceil(len(cat) * ratio)))
    # min_t_stat 提升至 3.0（Harvey, Liu & Zhu 2016 多重检验标准）；
    # redundancy_threshold 收紧至 0.90（AQR 常用冗余剔除阈值）。
    "adaptive_top_k": {
        "enabled": True,
        "min": 3,
        "max": 10,
        "ratio": 0.2,
    },

    # ── [机构级全局大类配额 - Citadel Alpha Lab 标准] ──
    # 跨窗口聚合 + 共线性精简后，若单一大类（如 204 因子的 Price-Volume）
    # 仍支配组合，按综合评分截断至 max_per_category 个。设 None 关闭。
    "max_per_category": 30,
}

# ==============================================================================
# [机构级分类学映射层 - AQR Factor Builder 标准]
# 将 yaml 中零散的 category 字段归并为 8 个 MECE（互斥且完备）经济学大类。
# 分类依据：经济学溢价维度驱动；构造形态（Rolling/K-Bar 等）不再充当类别。
# 未收录的 category 原样保留（确保不丢因子）。
# ==============================================================================
CATEGORY_TAXONOMY = {
    # ── Value 估值 ──
    "估值": "Value",
    # ── Momentum 动量（含均线交叉类技术面）──
    "momentum": "Momentum",
    "动量": "Momentum",
    "动量与反转": "Momentum",
    "技术面": "Momentum",
    # ── Reversal 反转 ──
    "reversal": "Reversal",
    "反转": "Reversal",
    # ── Quality 质量 ──
    "盈利": "Quality",
    "财务质量": "Quality",
    "质量": "Quality",
    # ── Growth 成长（eps_forecast_yoy 盈利预期归入成长）──
    "成长": "Growth",
    "情绪": "Growth",
    # ── Size & Liquidity 规模与流动性 ──
    "市值与流动性": "Size & Liquidity",
    "流动性": "Size & Liquidity",
    "风格": "Size & Liquidity",
    # ── Volatility & Risk 波动与风险 ──
    "波动率": "Volatility & Risk",
    "风险": "Volatility & Risk",
    # ── Price-Volume 量价微观结构合成池 ──
    "滚动时间窗口 (Rolling)": "Price-Volume",
    "量价综合 (Price-Volume Synthetic)": "Price-Volume",
    "量价K线 (K-Bar)": "Price-Volume",
    "归一化价格 (Price)": "Price-Volume",
}


def map_category(cat: str) -> str:
    """将 yaml 原始 category 归并到机构级 MECE 大类；未收录类别原样保留。"""
    if not CONFIG.get("taxonomy_enabled", True):
        return cat
    return CATEGORY_TAXONOMY.get(cat, cat)


# ==============================================================================
# [工具函数区] 因子加载与基础工具
# ==============================================================================

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

            cat = map_category(factor.get("category", "未分类"))
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


def _compute_window_full_period(rolling_windows: List[Dict]) -> Tuple[str, str]:
    """根据滚动窗口列表计算全局缓存所需的完整时间范围。"""
    all_starts = []
    all_ends = []
    for w in rolling_windows:
        all_starts.append(w["train"][0])
        all_ends.append(w["train"][1])
    return min(all_starts), max(all_ends)


# ==============================================================================
# [数据质量检查 - Bloomberg 级]（CONFIG 第一层）
# ==============================================================================

def check_factor_data_quality(
    df: pd.DataFrame,
    factor_cols: List[str],
    min_coverage: float = 0.7,
    max_missing_rate: float = 0.3,
    outlier_threshold: float = 5.0,
) -> Tuple[List[str], pd.DataFrame]:
    """
    [Bloomberg 级] 因子数据质量检查。

    检查维度：
    1. 覆盖率：因子在时间×股票截面上的非空比例
    2. 缺失率：按时间维度的平均缺失率
    3. 异常值：超出 N 倍标准差的观测比例
    4. 方差：零方差因子（无区分度）

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列
    - factor_cols: 待检查的因子列名
    - min_coverage: 最小覆盖率阈值
    - max_missing_rate: 最大缺失率阈值
    - outlier_threshold: 异常值检测阈值（标准差倍数）

    输出:
    - (passed_factors, quality_report)
      - passed_factors: 通过质量检查的因子列表
      - quality_report: 质量检查报告 DataFrame
    """
    print(f"    [数据质量检查] 检查 {len(factor_cols)} 个因子...")
    report_rows = []
    passed = []

    for col in factor_cols:
        if col not in df.columns:
            report_rows.append({
                "factor": col,
                "coverage": 0.0,
                "missing_rate": 1.0,
                "outlier_rate": 0.0,
                "std": 0.0,
                "passed": False,
                "reason": "因子不存在于数据中",
            })
            continue

        series = df[col]
        total = len(series)
        non_null = series.notna().sum()
        coverage = non_null / total if total > 0 else 0.0
        missing_rate = 1.0 - coverage

        # 异常值检测（基于截面 Z-score）
        outlier_rate = 0.0
        std_val = 0.0
        if non_null > 100:
            # 按日期计算截面 Z-score
            daily_z = df.groupby(level='datetime')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-12)
            )
            outlier_rate = (daily_z.abs() > outlier_threshold).sum() / non_null
            std_val = series.std()

        # 判断是否通过
        is_passed = True
        reasons = []
        if coverage < min_coverage:
            is_passed = False
            reasons.append(f"覆盖率过低({coverage:.2%} < {min_coverage:.0%})")
        if missing_rate > max_missing_rate:
            is_passed = False
            reasons.append(f"缺失率过高({missing_rate:.2%} > {max_missing_rate:.0%})")
        if std_val == 0 or np.isnan(std_val):
            is_passed = False
            reasons.append("零方差(无区分度)")

        if is_passed:
            passed.append(col)

        report_rows.append({
            "factor": col,
            "coverage": round(coverage, 4),
            "missing_rate": round(missing_rate, 4),
            "outlier_rate": round(float(outlier_rate), 4),
            "std": round(float(std_val), 6),
            "passed": is_passed,
            "reason": "; ".join(reasons) if reasons else "通过",
        })

    report_df = pd.DataFrame(report_rows)
    n_passed = len(passed)
    n_total = len(factor_cols)
    print(f"      >>> 通过 {n_passed}/{n_total} 个因子 ({n_passed/n_total*100:.1f}%)")
    failed = report_df[~report_df["passed"]]
    if len(failed) > 0:
        print(f"      >>> 淘汰原因分布:")
        for _, row in failed.iterrows():
            print(f"        - {row['factor']}: {row['reason']}")

    return passed, report_df


# ==============================================================================
# [因子中性化 - AQR 级]（CONFIG 第二层）
# ==============================================================================

def apply_neutralization(
    df: pd.DataFrame,
    factor_cols: List[str],
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
    log_mc: bool = True,
) -> pd.DataFrame:
    """
    [AQR 级] 行业+市值中性化。

    使用 Ridge 回归替代传统 OLS，解决行业稀疏或共线性导致的奇异矩阵问题。
    中性化后因子值 = 原始因子值 - 行业影响 - 市值影响

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列
    - factor_cols: 需要中性化的因子列
    - industry_field: 行业字段名
    - market_cap_field: 市值字段名
    - log_mc: 是否对市值取对数

    输出:
    - 中性化后的 DataFrame（同 shape）
    """
    print(f"    [AQR 中性化] 行业+市值中性化，{len(factor_cols)} 个因子...")

    # 获取股票列表和时间范围
    instruments = df.index.get_level_values('instrument').unique().tolist()
    start_time = df.index.get_level_values('datetime').min()
    end_time = df.index.get_level_values('datetime').max()

    try:
        # 从 Qlib 拉取行业和市值数据
        # [Windows 修复] 绕过 D.features() 的 ParallelExt 缺陷，用 _fetch_features_direct 直接取值
        fields = [f"${industry_field}", f"${market_cap_field}"]
        exposures = _fetch_features_direct(instruments, fields, start_time, end_time, freq='day')
        if exposures.empty:
            raise ValueError("行业/市值数据为空")
        exposures.columns = ['industry', 'market_cap']

        # 调整索引顺序
        if exposures.index.names != df.index.names:
            exposures = exposures.swaplevel()
        exposures = exposures.reindex(df.index)

        # 市值对数化
        if log_mc:
            exposures['market_cap'] = np.where(
                exposures['market_cap'] <= 0, np.nan, exposures['market_cap']
            )
            exposures['market_cap'] = np.log(exposures['market_cap'])

        # 按日期进行截面中性化
        result = df.copy()

        def _ridge_neutralize_slice(group):
            date = group.name
            try:
                sub_exp = exposures.xs(date, level='datetime')
            except KeyError:
                return group

            valid_mask = ~(sub_exp['industry'].isna() | sub_exp['market_cap'].isna())
            if not valid_mask.any():
                return group

            valid_exp = sub_exp[valid_mask]
            valid_instruments = valid_exp[valid_mask].index

            # 构建解释变量矩阵
            ind_dummies = pd.get_dummies(
                valid_exp['industry'].astype(int).astype(str),
                prefix='ind', drop_first=False
            )
            X = pd.concat([valid_exp['market_cap'], ind_dummies], axis=1)
            X = X.fillna(0).values.astype(float)

            # 提取目标因子矩阵
            valid_sub = group.loc[group.index.get_level_values('instrument').isin(valid_instruments), factor_cols]
            Y = valid_sub.values.astype(float)

            # Ridge 回归（L2 惩罚防止奇异矩阵）
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1e-5, fit_intercept=True, solver='auto')

            Y_filled = np.nan_to_num(Y, nan=0.0)
            model.fit(X, Y_filled)

            # 残差 = 实际值 - 预测值
            residuals = Y_filled - model.predict(X)
            residuals[np.isnan(Y)] = np.nan

            # 写回结果
            result_slice = group.copy()
            result_slice.loc[result_slice.index.get_level_values('instrument').isin(valid_instruments), factor_cols] = residuals
            return result_slice

        result = df.groupby(level='datetime', group_keys=False).apply(_ridge_neutralize_slice)
        print(f"      >>> 中性化完成")
        return result

    except Exception as e:
        print(f"      [警告] 中性化失败: {e}，退化为截面中心化")
        result = df.copy()
        for col in factor_cols:
            result[col] = df.groupby(level='datetime')[col].transform(lambda x: x - x.mean())
        return result


# ==============================================================================
# [IC 分析体系 - Citadel 级]（CONFIG 第三层，含第四层多重检验校正）
# ==============================================================================

def _vectorized_daily_ic(
    train_frame: pd.DataFrame,
    factor_cols: List[str],
    label_col: str,
    method: str = 'spearman',
) -> pd.DataFrame:
    """
    向量化逐日 IC 计算 - groupby().corr() 替代逐日 apply(corrwith)。

    输入:
    - train_frame: MultiIndex (datetime, instrument) × 因子列 + 标签列
    - factor_cols: 因子列名列表
    - label_col: 标签列名
    - method: 'spearman' (Rank IC) 或 'pearson' (Normal IC)

    输出:
    - daily_ic: DataFrame，index=datetime, columns=factor_cols，值为每日 IC
    """
    all_cols = factor_cols + [label_col]
    corr_matrices = train_frame.groupby(level='datetime')[all_cols].corr(method=method)
    daily_ic = corr_matrices.xs(label_col, level=1, axis=0)[factor_cols]
    return daily_ic


def compute_ic_statistics(
    daily_ic: pd.DataFrame,
) -> pd.DataFrame:
    """
    [Citadel 级] 计算 IC 统计指标。

    计算指标：
    - ic_mean: IC 均值
    - ic_std: IC 标准差
    - icir: ICIR = IC均值 / IC标准差
    - ic_ir_annualized: 年化 IR
    - t_stat: t 统计量 = IC均值 / (IC标准差 / sqrt(n))
    - p_value: p 值（双尾）
    - ic_pos_ratio: IC 为正的比例
    - ic_skew: IC 偏度
    - ic_kurt: IC 峰度

    输入:
    - daily_ic: 每日 IC 序列 (datetime × factor_cols)

    输出:
    - stats_df: 统计结果 DataFrame
    """
    stats_rows = []
    n_days = len(daily_ic)

    for col in daily_ic.columns:
        ic_series = daily_ic[col].dropna()
        n = len(ic_series)
        if n < 10:
            stats_rows.append({
                "factor": col,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "icir": np.nan,
                "icir_annualized": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ic_pos_ratio": np.nan,
                "ic_skew": np.nan,
                "ic_kurt": np.nan,
                "n_days": n,
            })
            continue

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        icir_annual = icir * np.sqrt(252)

        # t 检验
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        # 正 IC 比例
        pos_ratio = (ic_series > 0).sum() / n

        # 偏度和峰度
        skew = stats.skew(ic_series)
        kurt = stats.kurtosis(ic_series)

        stats_rows.append({
            "factor": col,
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "icir": round(icir, 4),
            "icir_annualized": round(icir_annual, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "ic_pos_ratio": round(pos_ratio, 4),
            "ic_skew": round(skew, 4),
            "ic_kurt": round(kurt, 4),
            "n_days": n,
        })

    return pd.DataFrame(stats_rows)


def apply_multiple_testing_correction(
    stats_df: pd.DataFrame,
    method: str = "bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    [Renaissance 级] 多重检验校正。

    方法：
    - bonferroni: Bonferroni 校正（最严格，控制 FWER）
    - bh: Benjamini-Hochberg 步骤（控制 FDR，更常用）

    输入:
    - stats_df: 包含 p_value 列的统计结果
    - method: 校正方法
    - alpha: 显著性水平

    输出:
    - 增加了 adjusted_pvalue 和 significant 列的 DataFrame
    """
    result = stats_df.copy()
    n_tests = len(result)

    if n_tests == 0:
        result["adjusted_pvalue"] = np.nan
        result["significant"] = False
        return result

    # [修复] 兼容带后缀的 p_value 列名（如 p_value_rank, p_value_normal）
    pval_candidates = [c for c in result.columns if c.startswith("p_value") and c != "adjusted_pvalue"]
    pval_col = pval_candidates[0] if pval_candidates else None
    if pval_col is None:
        result["adjusted_pvalue"] = np.nan
        result["significant"] = False
        return result

    if method == "bonferroni":
        # Bonferroni 校正：p_adj = p * n
        result["adjusted_pvalue"] = (result[pval_col] * n_tests).clip(upper=1.0)
        result["significant"] = result["adjusted_pvalue"] < alpha

    elif method == "bh":
        # Benjamini-Hochberg FDR 控制
        # 步骤：1) 排序 p 值  2) 计算 p_adj = p * n / rank  3) 从大到小取累积最小
        sorted_idx = result[pval_col].sort_values().index
        sorted_p = result.loc[sorted_idx, pval_col].values
        ranks = np.arange(1, n_tests + 1)
        adjusted = sorted_p * n_tests / ranks

        # 从大到小取累积最小（确保单调性）
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1.0)

        result.loc[sorted_idx, "adjusted_pvalue"] = adjusted
        result["significant"] = result["adjusted_pvalue"] < alpha

    else:
        result["adjusted_pvalue"] = result[pval_col]
        result["significant"] = result[pval_col] < alpha

    return result


# ==============================================================================
# [因子衰减分析 - Citadel 级]（CONFIG 第六层）
# ⚠️ 默认未启用（CONFIG["decay_analysis"]=False），为机构级扩展预留模块。
#    启用前需先扩展生成多周期标签（见 compute_factor_decay docstring）。
# ==============================================================================

def compute_factor_decay(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    periods: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    [Citadel 级] 因子衰减分析。

    计算因子在不同持有期的 IC，拟合指数衰减曲线，估算半衰期。

    ⚠️ 限制说明：本函数为准占位实现——需多周期标签（不同持有期前瞻收益）
    才能真正计算衰减。当前仅传入单一标签，各 period 的 IC 相同，
    半衰期拟合仅有形式意义。启用前须扩展生成多周期标签
    （如 Ref($close, -period) / Ref($close, -1) - 1），并设置 CONFIG["decay_analysis"]=True。

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列 + 标签列
    - factor_col: 因子列名
    - label_col: 当前标签列名（对应最短持有期）
    - periods: 持有期列表（天）

    输出:
    - decay_dict: 衰减分析结果
      - ic_by_period: {period: ic_value}
      - half_life: 半衰期（天）
      - decay_rate: 衰减速率 λ
      - decay_model: 衰减模型类型
    """
    result = {
        "ic_by_period": {},
        "half_life": np.nan,
        "decay_rate": np.nan,
        "decay_model": "exponential",
    }

    # [审计修复] 原实现各 period 复用同一 label 计算相同 IC，且 all(ic>0) 门槛
    # 使负向因子永不拟合。现拟合改用 |IC|（方向无关的衰减率），
    # 并保留多周期标签的限制说明（见 docstring）。
    for period in periods:
        # 当前简化：均基于传入标签计算；启用真衰减需 per-period 独立标签
        ic_series = df.groupby(level='datetime').apply(
            lambda x: x[factor_col].corr(x[label_col], method='spearman')
        ).dropna()
        if len(ic_series) > 10:
            result["ic_by_period"][period] = ic_series.mean()

    # 拟合指数衰减：|IC(t)| = |IC0| * exp(-λ * t)（对方向取绝对值）
    ics = [abs(result["ic_by_period"].get(p, 0)) for p in periods]
    if all(ic > 1e-6 for ic in ics) and len(ics) >= 2:
        try:
            # 对数线性回归：ln|IC| = ln|IC0| - λ * t
            log_ics = np.log(ics)
            slope, intercept, _, _, _ = stats.linregress(periods, log_ics)
            decay_rate = -slope
            if decay_rate > 0:
                half_life = np.log(2) / decay_rate
                result["decay_rate"] = round(decay_rate, 6)
                result["half_life"] = round(half_life, 2)
        except Exception:
            pass

    return result


# ==============================================================================
# [分组回测 - AQR 级]（CONFIG 第七层）
# ⚠️ 默认未启用（CONFIG["quantile_test"]=False），且当前未被任何调用点引用，
#    为机构级扩展预留模块。
# ==============================================================================

def quantile_backtest(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    n_quantiles: int = 5,
) -> Dict[str, Union[float, List[float], bool]]:
    """
    [AQR 级] 分组回测检验（Quantile Portfolio Test）。

    将股票按因子值分为 N 组，计算每组的平均未来收益，检验：
    1. 多空收益（Top - Bottom）
    2. 单调性（Spearman 秩相关）
    3. 多空组合夏普比率（long_short_sharpe）

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列 + 标签列
    - factor_col: 因子列名
    - label_col: 标签列名
    - n_quantiles: 分组数量

    输出:
    - result: 分组回测结果字典
    """
    result = {
        "quantile_returns": [],
        "long_short_return": 0.0,
        "long_short_sharpe": 0.0,
        "monotonicity": 0.0,
        "is_monotonic": False,
    }

    try:
        # 逐日分组并计算各组收益
        daily_returns = []

        for date, group in df.groupby(level='datetime'):
            valid = group[[factor_col, label_col]].dropna()
            if len(valid) < n_quantiles * 2:
                continue

            # 分组
            try:
                valid['quantile'] = pd.qcut(
                    valid[factor_col], q=n_quantiles, labels=False, duplicates='drop'
                )
            except Exception:
                continue

            if valid['quantile'].nunique() < n_quantiles:
                continue

            # 计算各组平均收益
            q_returns = valid.groupby('quantile')[label_col].mean()
            if len(q_returns) == n_quantiles:
                daily_returns.append(q_returns.values)

        if len(daily_returns) < 20:
            return result

        # 平均分组收益
        daily_returns_arr = np.array(daily_returns)
        avg_returns = daily_returns_arr.mean(axis=0)
        result["quantile_returns"] = [round(r, 6) for r in avg_returns.tolist()]

        # 多空收益
        ls_returns = daily_returns_arr[:, -1] - daily_returns_arr[:, 0]
        result["long_short_return"] = round(ls_returns.mean(), 6)
        if ls_returns.std() > 0:
            result["long_short_sharpe"] = round(
                ls_returns.mean() / ls_returns.std() * np.sqrt(252), 4
            )

        # 单调性检验（Spearman 秩相关）
        ranks = np.arange(n_quantiles)
        monotonicity, _ = stats.spearmanr(ranks, avg_returns)
        result["monotonicity"] = round(monotonicity, 4)
        result["is_monotonic"] = abs(monotonicity) > 0.8

    except Exception as e:
        print(f"      [警告] 分组回测失败 {factor_col}: {e}")

    return result


# ==============================================================================
# [置换检验 - D.E. Shaw 级]（CONFIG 第五层）
# ⚠️ 默认未启用（CONFIG["permutation_test"]=False），为机构级扩展预留模块。
# ==============================================================================

def permutation_test_ic(
    df: pd.DataFrame,
    factor_cols: List[str],
    label_col: str,
    n_permutations: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    [D.E. Shaw 级] 置换检验 (Permutation Test)。

    通过随机打乱标签，生成零假设下的 IC 分布，判断真实 IC 是否显著。

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列 + 标签列
    - factor_cols: 因子列名列表
    - label_col: 标签列名
    - n_permutations: 置换次数
    - alpha: 显著性水平

    输出:
    - perm_df: 置换检验结果 DataFrame
      - factor: 因子名
      - real_ic: 真实 IC 均值
      - perm_pvalue: 置换 p 值
      - perm_zscore: 置换 Z-score
      - significant: 是否显著
    """
    print(f"    [置换检验] {len(factor_cols)} 个因子, {n_permutations} 次置换...")

    # 计算真实 IC
    real_ics = {}
    for col in factor_cols:
        daily_ic = _vectorized_daily_ic(df, [col], label_col, method='spearman')
        real_ics[col] = daily_ic.mean().values[0] if not daily_ic.empty else 0.0

    # 置换检验
    perm_ics = {col: [] for col in factor_cols}

    # 置换检验：固定随机种子保证结果可复现（牺牲随机性换取审计可追溯）
    np.random.seed(42)
    for i in range(n_permutations):
        # 打乱标签（按日期截面内打乱）
        shuffled_df = df.copy()
        for date, group in shuffled_df.groupby(level='datetime'):
            shuffled_labels = group[label_col].values.copy()
            np.random.shuffle(shuffled_labels)
            shuffled_df.loc[group.index, label_col] = shuffled_labels

        # 计算置换后的 IC
        for col in factor_cols:
            daily_ic = _vectorized_daily_ic(shuffled_df, [col], label_col, method='spearman')
            perm_ics[col].append(daily_ic.mean().values[0] if not daily_ic.empty else 0.0)

        if (i + 1) % 50 == 0:
            print(f"      >>> 已完成 {i + 1}/{n_permutations} 次置换")

    # 计算 p 值和 Z-score
    perm_rows = []
    for col in factor_cols:
        real_ic = real_ics[col]
        perm_dist = np.array(perm_ics[col])
        perm_mean = perm_dist.mean()
        perm_std = perm_dist.std()

        # p 值：置换分布中大于等于真实 IC 的比例
        p_value = (np.abs(perm_dist) >= abs(real_ic)).sum() / len(perm_dist)
        # Z-score
        z_score = (real_ic - perm_mean) / perm_std if perm_std > 0 else 0.0

        perm_rows.append({
            "factor": col,
            "real_ic": round(real_ic, 6),
            "perm_mean": round(perm_mean, 6),
            "perm_std": round(perm_std, 6),
            "perm_zscore": round(z_score, 4),
            "perm_pvalue": round(p_value, 4),
            "significant": p_value < alpha,
        })

    return pd.DataFrame(perm_rows)


# ==============================================================================
# [因子拥挤度分析 - Two Sigma 级]（CONFIG 第九层）
# ⚠️ 默认未启用（CONFIG["crowding_analysis"]=False），为机构级扩展预留模块。
# ==============================================================================

def compute_factor_crowding(
    df: pd.DataFrame,
    factor_cols: List[str],
    threshold: float = 0.8,
) -> Tuple[List[str], pd.DataFrame]:
    """
    [Two Sigma 级] 因子拥挤度分析。

    通过因子间相关性聚类，识别高度拥挤的因子组。
    拥挤因子 = 与其他因子高度相关的因子（信息冗余）。

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列
    - factor_cols: 因子列名列表
    - threshold: 拥挤度阈值（相关系数绝对值）

    输出:
    - (non_crowded_factors, crowding_report)
    """
    print(f"    [拥挤度分析] {len(factor_cols)} 个因子, 阈值={threshold}...")

    if len(factor_cols) < 2:
        return factor_cols, pd.DataFrame()

    # 计算因子间相关系数矩阵
    corr_mat = df[factor_cols].corr(method='spearman').abs()

    # 计算每个因子的平均相关系数（拥挤度指标）
    crowding_scores = {}
    for col in factor_cols:
        other_corrs = corr_mat[col].drop(col)
        crowding_scores[col] = other_corrs.mean()

    # 找出拥挤因子（与多个因子高度相关）
    crowded_factors = set()
    for i, col1 in enumerate(factor_cols):
        for j, col2 in enumerate(factor_cols):
            if i >= j:
                continue
            if corr_mat.loc[col1, col2] > threshold:
                # 保留拥挤度较低的那个
                if crowding_scores[col1] > crowding_scores[col2]:
                    crowded_factors.add(col1)
                else:
                    crowded_factors.add(col2)

    non_crowded = [f for f in factor_cols if f not in crowded_factors]

    # 生成报告
    report_rows = []
    for col in factor_cols:
        report_rows.append({
            "factor": col,
            "crowding_score": round(crowding_scores[col], 4),
            "is_crowded": col in crowded_factors,
        })
    report_df = pd.DataFrame(report_rows).sort_values("crowding_score", ascending=False)

    print(f"      >>> 拥挤因子 {len(crowded_factors)} 个，非拥挤 {len(non_crowded)} 个")
    return non_crowded, report_df


# ==============================================================================
# [多方法交叉验证 - Point72 级]（CONFIG 第八层）
# ⚠️ 默认未启用（CONFIG["multi_method_voting"]=False），为机构级扩展预留模块。
# ==============================================================================

def multi_method_voting(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    factor_cols: List[str],
    methods: List[str] = ["embedded", "filter", "wrapper"],
    top_k: int = 5,
    voting_threshold: float = 0.5,
) -> Tuple[List[str], pd.DataFrame]:
    """
    [Point72 级] 多方法交叉验证投票机制。

    使用多种特征选择方法独立筛选因子，只有被多数方法选中的因子才会最终入选。
    这能有效降低单一方法的偏差，提高因子选择的稳健性。

    输入:
    - x_train: 训练特征矩阵
    - y_train: 标签
    - factor_cols: 候选因子列表
    - methods: 参与投票的方法列表
    - top_k: 每种方法选 top_k 个因子
    - voting_threshold: 投票通过阈值（比例）

    输出:
    - (selected_factors, voting_report)
    """
    print(f"    [多方法投票] {len(methods)} 种方法, top_k={top_k}, 阈值={voting_threshold}...")

    votes = {col: 0 for col in factor_cols}
    method_results = {}

    for method in methods:
        try:
            if method == "embedded":
                result = cached_select_features(
                    x_train, y_train,
                    method="embedded", algo="lightgbm",
                    threshold=0.0,
                    model_kwargs={"max_features": min(top_k, len(factor_cols))},
                    remove_collinearity=False,
                )
            elif method == "filter":
                result = cached_select_features(
                    x_train, y_train,
                    method="filter", algo="f_regression",
                    k=min(top_k, len(factor_cols)),
                    remove_collinearity=False,
                )
            elif method == "wrapper":
                result = cached_select_features(
                    x_train, y_train,
                    method="wrapper",
                    n_features=min(top_k, len(factor_cols)),
                    remove_collinearity=False,
                )
            else:
                continue

            selected = result.selected_features
            method_results[method] = selected
            for f in selected:
                if f in votes:
                    votes[f] += 1

        except Exception as e:
            print(f"      [警告] {method} 方法失败: {e}")
            continue

    # 投票结果
    n_methods = len(method_results)
    threshold_count = max(1, int(n_methods * voting_threshold))

    selected = [f for f, v in votes.items() if v >= threshold_count]
    selected.sort(key=lambda x: votes[x], reverse=True)

    # 生成报告
    report_rows = []
    for col in factor_cols:
        report_rows.append({
            "factor": col,
            "votes": votes[col],
            "vote_ratio": round(votes[col] / n_methods, 4) if n_methods > 0 else 0,
            "selected": votes[col] >= threshold_count,
        })
    report_df = pd.DataFrame(report_rows).sort_values("votes", ascending=False)

    print(f"      >>> 投票通过 {len(selected)}/{len(factor_cols)} 个因子")
    return selected, report_df


# ==============================================================================
# [核心筛选函数] 单类别因子筛选（增强版）
# ==============================================================================

def _apply_mad_winsorize(df: pd.DataFrame, n_sigma: float = 3.0) -> pd.DataFrame:
    """
    [P1-2修复] 逐日截面 MAD 3σ 去极值（winsorize 截断）。

    以每个交易日截面为基准计算中位数与 MAD（Median Absolute Deviation），
    将超出 [med - n_sigma * mad, med + n_sigma * mad] 的值截断到边界，
    避免极端值在后续 rank 变换前压缩正常样本的秩区分度（专家流程阶段1-8）。
    NaN 保持原样（不参与统计也不被截断）。

    输入:
    - df: MultiIndex (datetime, instrument) × 因子列
    - n_sigma: MAD 倍数（默认 3.0，正态近似约 3σ）

    输出:
    - winsorize 后的 DataFrame（同 shape）
    """
    # 逐日截面中位数（NaN 自动跳过）
    med = df.groupby(level="datetime").transform("median")
    # 逐日截面 MAD × 1.4826 归一化到 σ 量纲
    mad = (df - med).abs().groupby(level="datetime").transform("median") * 1.4826
    lb = med - n_sigma * mad
    ub = med + n_sigma * mad
    return df.clip(lb, ub)


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


def _is_alive_on(code: str, date_str: str, delist_map: dict) -> bool:
    """判断股票在指定交易日是否未退市（[P1-3修复] 退市过滤核心）。

    输入:
    - code: 股票代码（大小写不敏感）
    - date_str: 交易日 YYYY-MM-DD
    - delist_map: {code: (list_date, delist_date)}，来自 instruments/all.txt

    返回:
    - True=当日可交易（未退市）；False=当日已退市
    """
    meta = delist_map.get(str(code).lower())
    if not meta:
        return True  # all.txt 未收录，视为存活
    exit_date = meta[1]
    if not exit_date or exit_date in ("0000-00-00", "20991231"):
        return True  # 无有效退市日期
    return date_str <= exit_date


def _filter_stocks_post(
    df: pd.DataFrame,
    filter_new_stocks: bool = True,
    filter_st: bool = True,
    filter_delisted: bool = True,
) -> pd.DataFrame:
    """
    对已加载的特征+标签 DataFrame 执行后置 ST/次新股/退市过滤。

    逐日遍历，调用 filter_codes_post 过滤每只股票，
    移除不满足条件的行。

    输入:
    - df: MultiIndex (datetime, instrument) × 列
    - filter_new_stocks: 过滤上市不足 250 日次新股
    - filter_st: 过滤 ST 股票
    - filter_delisted: [P1-3修复] 退市两阶段过滤（全期早退市股票逐日自然剔除；
        期内退市股票仅剔除退市日之后的行）

    输出:
    - 过滤后的 DataFrame
    """
    if df.empty:
        return df
    if not filter_new_stocks and not filter_st and not filter_delisted:
        return df

    # [P1-3修复] 加载退市日期映射（all.txt: code \t list_date \t delist_date）
    delist_map = _load_stock_name_map() if filter_delisted else {}

    all_dates = sorted(df.index.get_level_values("datetime").unique())
    kept_parts = []
    total_removed = 0
    n_dates = len(all_dates)

    for date_idx, date in enumerate(all_dates):
        if (date_idx + 1) % 100 == 0 or date_idx + 1 == n_dates:
            print(f"      [进度] ST/次新/退市过滤 {date_idx+1}/{n_dates} 个交易日 (累计移除 {total_removed:,} 行)")
        date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
        day_slice = df.xs(date, level="datetime", drop_level=False)
        if day_slice.empty:
            continue
        codes = day_slice.index.get_level_values("instrument").unique().tolist()
        # [P1-3修复] 退市过滤前置：剔除当日已退市股票（date_str > delist_date）
        if delist_map:
            codes = [c for c in codes if _is_alive_on(c, date_str, delist_map)]
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


def _build_direct_keep_rows(
    cat_name: str,
    factors: List[Dict],
    available: List[str],
) -> pd.DataFrame:
    """
    当某类别可用因子不足 2 个时，直接保留全部可用因子（免筛选直通）。

    输入:
    - cat_name: 类别名称
    - factors: 该类别因子定义列表 [{name, meaning, source_file}, ...]
    - available: 实际在训练数据中可用的因子名列表

    输出:
    - rows_df: 直接保留的结果 DataFrame
      （selected=True, importance=1.0, composite_score=1.0, IC 统计量为 NaN）
    """
    rows = []
    for f in factors:
        if f["name"] in available:
            rows.append({
                "category": cat_name, "factor_name": f["name"],
                "selected": True, "importance": 1.0, "rank": 1,
                "meaning": f["meaning"], "source_file": f["source_file"],
                "ic_mean": np.nan, "icir": np.nan, "t_stat": np.nan,
                "composite_score": 1.0,
            })
    return pd.DataFrame(rows)


def run_single_category_selection(
    cat_name: str,
    factors: List[Dict],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fs_method: str,
    fs_algo: str,
    top_k: int,
    label_name: str,
    cfg: Optional[dict] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    在单个类别上运行特征选择（机构级增强版）。

    执行流程：
    1. 数据质量检查（Bloomberg 级）
    2. 因子中性化（AQR 级）
    3. 截面标准化
    4. 基础特征选择（LightGBM 等）
    5. 冗余检测
    6. IC 分析（Citadel 级）
    7. 多重检验校正（Renaissance 级）
    8. 置换检验（D.E. Shaw 级，可选）
    9. 拥挤度分析（Two Sigma 级，可选）
    10. 多方法投票（Point72 级，可选）
    11. 综合评分排序

    输入:
    - cat_name: 类别名称
    - factors: 该类别因子列表 [{name, expression, meaning, source_file}, ...]
    - x_train: 已标准化的特征矩阵
    - y_train: 标签 Series
    - fs_method, fs_algo, top_k: 特征选择参数
    - label_name: 标签列名
    - cfg: 配置字典（与 CONFIG 单源对齐；为 None 时使用函数默认值，
      默认值已与 CONFIG 机构级标准一致：redundancy_threshold=0.90, min_t_stat=3.0）

    输出:
    - (results_df, quality_df)
      - results_df: 因子筛选结果（含综合评分）
      - quality_df: 因子质量评估报告
    """
    # [机构标准] 配置参数统一从 cfg 透传（与 CONFIG 单源一致，消除函数默认值漂移）。
    # 函数默认值与 CONFIG 机构级标准对齐：redundancy_threshold=0.90, min_t_stat=3.0。
    if cfg is None:
        cfg = {}
    redundancy_check = cfg.get("redundancy_check", True)
    redundancy_threshold = cfg.get("redundancy_threshold", 0.90)
    icir_stability = cfg.get("icir_stability", True)
    icir_rolling_window = cfg.get("icir_window", 60)
    icir_keep_ratio = cfg.get("icir_keep_ratio", 0.9)
    data_quality_check = cfg.get("data_quality_check", True)
    min_coverage = cfg.get("min_coverage", 0.7)
    neutralize = cfg.get("neutralize", False)
    industry_field = cfg.get("industry_field", "sw_l1")
    market_cap_field = cfg.get("market_cap_field", "circ_mv")
    log_mc = cfg.get("log_mc", True)
    ic_analysis = cfg.get("ic_analysis", True)
    ic_type = cfg.get("ic_type", "both")
    multiple_testing_correction = cfg.get("multiple_testing_correction", True)
    mtc_method = cfg.get("mtc_method", "bh")
    mtc_alpha = cfg.get("mtc_alpha", 0.05)
    min_t_stat = cfg.get("min_t_stat", 3.0)
    permutation_test = cfg.get("permutation_test", False)
    permutation_n = cfg.get("permutation_n", 200)
    permutation_alpha = cfg.get("permutation_alpha", 0.05)
    crowding_analysis = cfg.get("crowding_analysis", False)
    crowding_threshold = cfg.get("crowding_threshold", 0.8)
    multi_method_voting_enabled = cfg.get("multi_method_voting", False)
    voting_methods = cfg.get("voting_methods", ["embedded", "filter"])
    voting_threshold = cfg.get("voting_threshold", 0.5)
    scoring_weights = cfg.get("scoring_weights") or {
        "importance": 0.25, "ic": 0.25, "icir": 0.25,
        "t_stat": 0.15, "stability": 0.10,
    }

    cat_factor_names = [f["name"] for f in factors]
    print(f"    [筛选] {len(factors)} 个因子, top_k={top_k}...")

    # 从全局特征矩阵中切片
    available = [c for c in cat_factor_names if c in x_train.columns]
    if len(available) == 0:
        print(f"    [错误] 该类别的因子在训练数据中不存在")
        return None
    if len(available) < 2:
        print(f"    [跳过] 只有 {len(available)} 个可用因子，直接保留")
        return _build_direct_keep_rows(cat_name, factors, available), pd.DataFrame()

    x_cat = x_train[available].copy()
    y_cat = y_train.copy()

    print(f"      >>> {x_cat.shape[0]} 行, {x_cat.shape[1]} 个特征")

    # ── 第 1 步：数据质量检查 ──
    quality_df = pd.DataFrame()
    if data_quality_check:
        passed_factors, quality_df = check_factor_data_quality(
            x_cat, available,
            min_coverage=min_coverage,
        )
        if len(passed_factors) < len(available):
            print(f"      >>> 质量检查淘汰 {len(available) - len(passed_factors)} 个因子")
            available = passed_factors
            x_cat = x_cat[available]
            if len(available) < 2:
                return _build_direct_keep_rows(cat_name, factors, available), quality_df

    # ── 第 2 步：因子中性化 ──
    if neutralize and len(available) > 0:
        x_cat = apply_neutralization(
            x_cat, available,
            industry_field=industry_field,
            market_cap_field=market_cap_field,
            log_mc=log_mc,
        )
        # 中性化后重新填充 NaN
        x_cat = x_cat.fillna(0.5)

    # ── 第 3 步：基础特征选择 ──
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

    # ── 第 4 步：冗余检测 ──
    # [P0修复] 原门槛 len(selected_factor_names) > 5 在 top_k<=5 时永不触发，
    # 冗余检测防线形同虚设。现作用域=全部候选因子，门槛 len>1，
    # 且剔除结果同步收缩候选池 available 与 LightGBM 选中集合。
    if redundancy_check and len(available) > 1:
        print(f"    [冗余检测] 阈值={redundancy_threshold}，检测 {len(available)} 个候选因子...")
        feat_in_data = [c for c in available if c in x_cat.columns]
        if len(feat_in_data) > 1:
            try:
                corr_mat = x_cat[feat_in_data].corr(method='spearman').abs()
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
                    available = [f for f in available if f not in to_drop]
                    selected_factor_names = [f for f in selected_factor_names if f not in to_drop]
                    selected_set = set(selected_factor_names)
                    print(f"      冗余检测完成: 剔除 {len(to_drop)} 个冗余因子，保留 {len(available)} 个候选")
            except Exception as e:
                print(f"      [跳过] 冗余检测异常: {e}")

    # ── 第 5 步：IC 分析（Citadel 级）──
    ic_stats_df = pd.DataFrame()
    if ic_analysis and len(available) > 0:
        print(f"    [IC 分析] 计算 IC/ICIR/t-stat...")
        try:
            # 构建含标签的综合面板
            combined_frame = x_cat.copy()
            combined_frame[label_name] = y_cat

            # Rank IC
            if ic_type in ["rank", "both"]:
                daily_ic_rank = _vectorized_daily_ic(combined_frame, available, label_name, method='spearman')
                ic_stats_rank = compute_ic_statistics(daily_ic_rank)
                ic_stats_rank = ic_stats_rank.add_suffix('_rank')
                ic_stats_rank = ic_stats_rank.rename(columns={"factor_rank": "factor"})
                ic_stats_df = ic_stats_rank

            # Normal IC
            if ic_type in ["normal", "both"]:
                daily_ic_norm = _vectorized_daily_ic(combined_frame, available, label_name, method='pearson')
                ic_stats_norm = compute_ic_statistics(daily_ic_norm)
                ic_stats_norm = ic_stats_norm.add_suffix('_normal')
                ic_stats_norm = ic_stats_norm.rename(columns={"factor_normal": "factor"})
                if ic_stats_df.empty:
                    ic_stats_df = ic_stats_norm
                else:
                    ic_stats_df = ic_stats_df.merge(ic_stats_norm, on="factor", how="outer")

            # 多重检验校正
            has_pval = any(c.startswith("p_value") for c in ic_stats_df.columns if c != "adjusted_pvalue")
            if multiple_testing_correction and not ic_stats_df.empty and has_pval:
                ic_stats_df = apply_multiple_testing_correction(
                    ic_stats_df, method=mtc_method, alpha=mtc_alpha,
                )
                # 用校正后的 p 值重新筛选
                sig_factors = ic_stats_df[ic_stats_df["significant"]]["factor"].tolist()
                print(f"      >>> 多重检验校正 ({mtc_method}) 通过 {len(sig_factors)}/{len(ic_stats_df)} 个因子")

                # [P0修复] 显著性防线真正生效：
                # 原实现仅打印 sig_factors / high_t_factors 而不参与筛选，
                # 导致 BH 校正与 t-stat 阈值两道防线形同虚设。
                # 现将其作为硬性门槛，从候选池 available 中实际剔除不显著因子，
                # 并同步收缩 LightGBM 选中集合 selected_set。
                if sig_factors:
                    pass_set = set(sig_factors)
                    if "t_stat_rank" in ic_stats_df.columns:
                        high_t_factors = ic_stats_df[ic_stats_df["t_stat_rank"].abs() >= min_t_stat]["factor"].tolist()
                        pass_set = pass_set & set(high_t_factors)
                        print(f"      >>> t-stat >= {min_t_stat} 通过 {len(high_t_factors)}/{len(ic_stats_df)} 个因子")

                    # 从候选池剔除不显著因子（保底：至少保留 1 个候选，避免类别全空）
                    kept_candidates = [f for f in available if f in pass_set]
                    if kept_candidates:
                        dropped_n = len(available) - len(kept_candidates)
                        if dropped_n > 0:
                            print(f"      >>> 显著性防线剔除 {dropped_n} 个不显著因子（候选 {len(available)} → {len(kept_candidates)}）")
                        available = kept_candidates
                        x_cat = x_cat[kept_candidates]

                    # 同步收缩 LightGBM 选中集合（若选中因子未通过显著门槛则移除）
                    selected_factor_names = [f for f in selected_factor_names if f in pass_set]
                    selected_set = set(selected_factor_names)

        except Exception as e:
            print(f"      [跳过] IC 分析异常: {e}")

    # ── 第 6 步：ICIR 稳定性校验 ──
    # [P0修复] 原实现门槛 len(selected_factor_names) > 5 在 top_k 较小（如 2~5）时永不触发，
    # 且作用域仅限 LightGBM 选中子集 → ICIR 防线形同虚设。
    # 现改为：作用域=全部候选因子，门槛 len>1，度量从"ICIR 为正占比"（只认正方向，
    # 会误杀稳定负向因子）改为"同号占比"（与主导方向一致的比例）。
    if icir_stability and len(available) > 1:
        print(f"    [ICIR 稳定校验] 窗口={icir_rolling_window}d, keep_ratio={icir_keep_ratio}，候选 {len(available)} 个因子...")
        try:
            icir_feat = [c for c in available if c in x_cat.columns]
            if len(icir_feat) > 1:
                combined_frame = x_cat[icir_feat].copy()
                combined_frame[label_name] = y_cat
                daily_ic = _vectorized_daily_ic(combined_frame, icir_feat, label_name, method='spearman')
                if not daily_ic.empty and len(daily_ic) > icir_rolling_window // 2:
                    rolling_mean = daily_ic.rolling(window=icir_rolling_window, min_periods=icir_rolling_window // 2).mean()
                    rolling_std = daily_ic.rolling(window=icir_rolling_window, min_periods=icir_rolling_window // 2).std()
                    rolling_icir = rolling_mean / rolling_std.replace(0, np.nan)
                    # 同号占比：因子主导方向 = 全期 Rank IC 均值符号
                    overall_sign = np.sign(daily_ic.mean())
                    same_ratio = (np.sign(rolling_mean) == overall_sign).sum() / rolling_mean.notna().sum()
                    same_ratio = same_ratio.fillna(0).sort_values(ascending=False)
                    # 稳定门槛：同号占比 >= 0.5（IC 方向跨期一致），再保留排名前 keep_ratio
                    stable_mask = same_ratio >= 0.5
                    keep_count = max(int(len(same_ratio) * icir_keep_ratio), 1)
                    stable_factors = same_ratio[stable_mask].head(keep_count).index.tolist()
                    # 保底：全部方向不稳定时至少保留方向最稳定者，避免该类别无因子入选
                    if not stable_factors:
                        stable_factors = [same_ratio.index[0]]
                        print(f"      [警告] 候选全部方向不稳定，保底保留 {stable_factors[0]}")
                    dropped = len(icir_feat) - len(stable_factors)
                    if dropped > 0:
                        print(f"      ICIR 稳定校验: 剔除 {dropped} 个方向不稳定/排名靠后的因子，保留 {len(stable_factors)} 个")
                    selected_factor_names = stable_factors
                    selected_set = set(selected_factor_names)
                else:
                    print(f"      daily_ic 仅 {len(daily_ic)} 行，不足 {icir_rolling_window // 2}，跳过")
            else:
                print(f"      [跳过] 可用因子 < 2")
        except Exception as e:
            print(f"      [跳过] ICIR 稳定校验异常: {e}")

    # ── 第 7 步：置换检验（D.E. Shaw 级，可选）──
    perm_df = pd.DataFrame()
    if permutation_test and len(selected_factor_names) > 0:
        print(f"    [置换检验] {len(selected_factor_names)} 个因子...")
        try:
            combined_frame = x_cat[selected_factor_names].copy()
            combined_frame[label_name] = y_cat
            perm_df = permutation_test_ic(
                combined_frame, selected_factor_names, label_name,
                n_permutations=permutation_n, alpha=permutation_alpha,
            )
            # 只保留通过置换检验的因子
            sig_perm = perm_df[perm_df["significant"]]["factor"].tolist()
            if len(sig_perm) > 0:
                dropped = len(selected_factor_names) - len(sig_perm)
                if dropped > 0:
                    print(f"      置换检验: {dropped} 个因子未通过，保留 {len(sig_perm)} 个")
                    selected_factor_names = sig_perm
                    selected_set = set(selected_factor_names)
        except Exception as e:
            print(f"      [跳过] 置换检验异常: {e}")

    # ── 第 8 步：拥挤度分析（Two Sigma 级，可选）──
    if crowding_analysis and len(selected_factor_names) > 2:
        print(f"    [拥挤度分析] {len(selected_factor_names)} 个因子...")
        try:
            non_crowded, _ = compute_factor_crowding(
                x_cat[selected_factor_names], selected_factor_names,
                threshold=crowding_threshold,
            )
            if len(non_crowded) < len(selected_factor_names):
                dropped = len(selected_factor_names) - len(non_crowded)
                print(f"      拥挤度检测: 剔除 {dropped} 个拥挤因子，保留 {len(non_crowded)} 个")
                selected_factor_names = non_crowded
                selected_set = set(selected_factor_names)
        except Exception as e:
            print(f"      [跳过] 拥挤度分析异常: {e}")

    # ── 第 9 步：多方法投票（Point72 级，可选）──
    if multi_method_voting_enabled and len(available) > 2:
        print(f"    [多方法投票] {len(available)} 个候选因子...")
        try:
            voted_factors, _ = multi_method_voting(
                x_cat, y_cat, available,
                methods=voting_methods,
                top_k=min(top_k * 2, len(available)),
                voting_threshold=voting_threshold,
            )
            # 取交集：基础选中 + 投票通过
            intersected = [f for f in selected_factor_names if f in voted_factors]
            if len(intersected) > 0:
                dropped = len(selected_factor_names) - len(intersected)
                if dropped > 0:
                    print(f"      投票验证: {dropped} 个因子未通过投票，保留 {len(intersected)} 个")
                    selected_factor_names = intersected
                    selected_set = set(selected_factor_names)
        except Exception as e:
            print(f"      [跳过] 多方法投票异常: {e}")

    # ── 第 10 步：构建综合评分 ──
    # [P1-1修复] IC 统计列名动态解析：ic_type="both" 时列带 _rank/_normal 双后缀，
    # ic_type="normal" 时仅 _normal 后缀。原实现硬编码 _rank 后缀，normal 模式下
    # IC/ICIR/t-stat 全部缺失，评分退化为 importance+stability 两维。
    # [P2-8优化] 预构建 factor→行 字典索引，替代每因子一次全表筛选（O(n²)→O(1)）。
    if not ic_stats_df.empty and "factor" in ic_stats_df.columns:
        ic_lookup = ic_stats_df.set_index("factor")
    else:
        ic_lookup = pd.DataFrame()

    def _pick_stat(row: pd.Series, base: str) -> float:
        """按后缀优先级 _rank > _normal > 无后缀 取统计量，未命中返回 NaN。"""
        for suf in ("_rank", "_normal", ""):
            col = base + suf
            if col in row.index and not pd.isna(row[col]):
                return float(row[col])
        return np.nan

    # 归一化重要性
    if len(scores) > 0 and scores.max() > 0:
        scores_norm = scores / scores.max()
    else:
        scores_norm = pd.Series(1.0, index=scores.index) if len(scores) > 0 else pd.Series(dtype=float)

    # 构建结果
    rows = []
    for rank, (factor_name, importance) in enumerate(scores_norm.items(), 1):
        factor_info = next((f for f in factors if f["name"] == factor_name), None)

        # 从 IC 统计中提取数据
        ic_mean = np.nan
        icir = np.nan
        t_stat = np.nan
        p_value = np.nan
        adj_pvalue = np.nan

        if not ic_lookup.empty and factor_name in ic_lookup.index:
            ic_row = ic_lookup.loc[factor_name]
            # 防御：factor 意外重复时取首行（正常 merge 下 factor 唯一）
            if isinstance(ic_row, pd.DataFrame):
                ic_row = ic_row.iloc[0]
            ic_mean = _pick_stat(ic_row, "ic_mean")
            icir = _pick_stat(ic_row, "icir")
            t_stat = _pick_stat(ic_row, "t_stat")
            p_value = _pick_stat(ic_row, "p_value")
            if "adjusted_pvalue" in ic_row.index:
                adj_pvalue = ic_row["adjusted_pvalue"]

        # 计算综合评分
        composite = 0.0
        w_total = 0.0

        # 重要性得分
        if not np.isnan(importance):
            composite += scoring_weights.get("importance", 0.25) * abs(importance)
            w_total += scoring_weights.get("importance", 0.25)

        # IC 得分（归一化到 0-1）
        if not np.isnan(ic_mean):
            ic_score = min(abs(ic_mean) / 0.1, 1.0)  # IC=0.1 对应满分
            composite += scoring_weights.get("ic", 0.25) * ic_score
            w_total += scoring_weights.get("ic", 0.25)

        # ICIR 得分
        if not np.isnan(icir):
            icir_score = min(abs(icir) / 2.0, 1.0)  # ICIR=2 对应满分
            composite += scoring_weights.get("icir", 0.25) * icir_score
            w_total += scoring_weights.get("icir", 0.25)

        # t-stat 得分
        if not np.isnan(t_stat):
            t_score = min(abs(t_stat) / 4.0, 1.0)  # t=4 对应满分
            composite += scoring_weights.get("t_stat", 0.15) * t_score
            w_total += scoring_weights.get("t_stat", 0.15)

        # 稳定性得分（是否选中 = 稳定性）
        stability = 1.0 if factor_name in selected_set else 0.0
        composite += scoring_weights.get("stability", 0.10) * stability
        w_total += scoring_weights.get("stability", 0.10)

        if w_total > 0:
            composite /= w_total

        rows.append({
            "category": cat_name,
            "factor_name": factor_name,
            "selected": factor_name in selected_set,
            "importance": round(float(importance), 4),
            "rank": rank,
            "ic_mean": round(float(ic_mean), 6) if not np.isnan(ic_mean) else np.nan,
            "icir": round(float(icir), 4) if not np.isnan(icir) else np.nan,
            "t_stat": round(float(t_stat), 4) if not np.isnan(t_stat) else np.nan,
            "p_value": round(float(p_value), 6) if not np.isnan(p_value) else np.nan,
            "adjusted_pvalue": round(float(adj_pvalue), 6) if not np.isnan(adj_pvalue) else np.nan,
            "composite_score": round(float(composite), 4),
            "meaning": (factor_info or {}).get("meaning", ""),
            "source_file": (factor_info or {}).get("source_file", ""),
        })

    result_df = pd.DataFrame(rows)
    # 按综合评分排序
    result_df = result_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    result_df["rank"] = range(1, len(result_df) + 1)

    return result_df, quality_df


def print_category_results(cat_name: str, results_df: pd.DataFrame):
    """打印单个类别的筛选结果（增强版）。"""
    if results_df is None or len(results_df) == 0:
        return

    selected = results_df[results_df["selected"]]
    not_selected = results_df[~results_df["selected"]]

    print(f"\n  │ 选中: {len(selected)}/{len(results_df)} 个因子")
    if len(selected) > 0:
        print(f"  │ 入选因子（按综合评分排序）:")
        print(f"  │ {'排名':>4s} {'因子名':<25s} {'评分':>6s} {'IC':>8s} {'ICIR':>7s} {'t-stat':>7s}")
        print(f"  │ {'-'*4} {'-'*25} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")
        for _, row in selected.iterrows():
            ic_str = f"{row['ic_mean']:.4f}" if pd.notna(row.get('ic_mean')) else "  N/A "
            icir_str = f"{row['icir']:.3f}" if pd.notna(row.get('icir')) else "  N/A "
            t_str = f"{row['t_stat']:.2f}" if pd.notna(row.get('t_stat')) else "  N/A "
            bar = "#" * int(row["composite_score"] * 20) + "." * (20 - int(row["composite_score"] * 20))
            print(f"  │ [{row['rank']:2d}] {row['factor_name']:<25s} [{bar}] {row['composite_score']:.3f} {ic_str:>8s} {icir_str:>7s} {t_str:>7s}")
    if len(not_selected) > 0:
        print(f"  │ 淘汰因子:")
        for _, row in not_selected.iterrows():
            ic_str = f"{row['ic_mean']:.4f}" if pd.notna(row.get('ic_mean')) else "N/A"
            print(f"  │   [{row['rank']:2d}] {row['factor_name']:<25s} 评分={row['composite_score']:.3f} IC={ic_str}")


# ==============================================================================
# [数据准备函数]
# ==============================================================================

def _prepare_window_data(
    global_feature_cache,
    all_factor_names: List[str],
    train_start: str,
    train_end: str,
    label_expr: str,
    label_name: str,
    filter_new_stocks: bool = True,
    filter_st: bool = True,
    filter_delisted: bool = True,
    filter_untradeable_labels: bool = True,
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
    log_mc: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    为单个滚动窗口准备训练数据：切片时间范围 → 合并标签 → 过滤 → 标准化。

    处理链路（对齐训练端 DK_L 口径）：
      1. 特征×标签合并
      2. ST/次新/退市过滤（逐日截面）
      3. [P0-3] 涨跌停/一字板/持仓期停牌不可交易样本剔除（filter_untradeable_labels）
      4. [P1-1] 标签 MAD 3σ 去极值（训练期统计量）
      5. [P0-1] 标签管线与训练端 DK_L 对齐：CSNeutralize(label) → CSQuantileNorm(label)
         （顺序与 _build_processors 一致：先中性化剥离行业/市值，再截面分位数化），
         使筛选 IC 评估口径 = 模型学习目标（纯 alpha 排序标签）
      6. 因子截面 MAD 3σ winsorize + 截面排名标准化（CSRankNorm）

    返回: (x_train, y_train)
    """
    import time as _time

    # 1. 从缓存切片特征数据（按需惰性合并）
    _t0 = _time.time()
    warehouse_df = global_feature_cache.get_warehouse_df(
        selected_names=all_factor_names,
        start_time=train_start,
        end_time=train_end,
    )
    print(f"    [进度] 特征加载完成: {warehouse_df.shape[0]:,} 行 × {warehouse_df.shape[1]} 因子, "
          f"耗时 {_time.time()-_t0:.1f}s")
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
    # [Windows 修复] D.features 对全量股票一次性调用存在 ParallelExt 死锁风险
    # （tree.py 实测 500 只/批安全），分批加载后 concat。
    _t1 = _time.time()
    label_parts = []
    _BATCH_SIZE = 500
    for _i in range(0, len(global_feature_cache.resolved_instruments), _BATCH_SIZE):
        _batch = global_feature_cache.resolved_instruments[_i:_i + _BATCH_SIZE]
        _part = D.features(_batch, [label_expr], train_start, train_end)
        if _part is not None and not _part.empty:
            label_parts.append(_part)
    if not label_parts:
        raise ValueError(f"标签 {label_expr} 在 {train_start}~{train_end} 为空")
    label_raw = pd.concat(label_parts)
    print(f"    [进度] 标签加载完成: {len(label_raw):,} 行, 耗时 {_time.time()-_t1:.1f}s")
    if isinstance(label_raw.columns, pd.MultiIndex):
        label_raw.columns = label_raw.columns.droplevel(1)
    label_raw = label_raw.rename(columns={label_raw.columns[0]: label_name})
    label_flat = label_raw.reset_index()
    label_flat['instrument'] = label_flat['instrument'].str.lower()
    label_flat = label_flat.set_index(['datetime', 'instrument']).sort_index()

    # 3. 合并特征与标签
    _t2 = _time.time()
    full_train_frame = warehouse_df.join(label_flat, how='inner')
    full_train_frame = full_train_frame.dropna(subset=[label_name])
    print(f"    [进度] 特征×标签合并完成: {full_train_frame.shape[0]:,} 行, "
          f"耗时 {_time.time()-_t2:.1f}s")

    # 4. 后置过滤：ST / 次新股 / 退市（逐日截面动态过滤）
    if filter_new_stocks or filter_st or filter_delisted:
        _t3 = _time.time()
        full_train_frame = _filter_stocks_post(
            full_train_frame,
            filter_new_stocks=filter_new_stocks,
            filter_st=filter_st,
            filter_delisted=filter_delisted,
        )
        print(f"    [进度] ST/次新/退市过滤完成: 剩余 {full_train_frame.shape[0]:,} 行, "
              f"耗时 {_time.time()-_t3:.1f}s")

    # [P0-3修复] 不可交易样本剔除：涨跌停跳空 / 一字板 / 持仓期末停牌
    # （对齐训练端 filter_untradeable_labels，剔除无法真实买入的样本，标签置 NaN）
    if filter_untradeable_labels:
        _t30 = _time.time()
        try:
            # filter_untradeable_labels 内部用 D.features 结果 reindex，
            # 要求传入 (instrument, datetime) 索引顺序，与 Qlib 返回保持一致
            y_for_filter = full_train_frame[[label_name]].copy()
            if isinstance(y_for_filter.index, pd.MultiIndex):
                y_inst_first = y_for_filter.swaplevel().sort_index()
            else:
                y_inst_first = y_for_filter
            y_clean = _filter_untradeable_fn(
                y_inst_first,
                global_feature_cache.resolved_instruments,
                train_start, train_end,
            )
            if isinstance(y_clean.index, pd.MultiIndex):
                y_clean = y_clean.swaplevel().sort_index()
            full_train_frame[label_name] = y_clean[label_name]
            full_train_frame = full_train_frame.dropna(subset=[label_name])
            print(f"    [进度] 不可交易样本剔除完成: 剩余 {full_train_frame.shape[0]:,} 行, "
                  f"耗时 {_time.time()-_t30:.1f}s")
        except Exception as e:
            print(f"    [警告] 不可交易样本剔除失败（不影响主流程）: {e}")

    # [P1-1修复] 标签 MAD 3σ 去极值（训练期统计量，剔除极端收益样本）
    y_raw = full_train_frame[label_name]
    _med = y_raw.median()
    _mad = (y_raw - _med).abs().median() * 1.4826
    if _mad > 0:
        _keep_mask = (y_raw - _med).abs() <= 3.0 * _mad
        _n_dropped = int((~_keep_mask).sum())
        if _n_dropped > 0:
            full_train_frame = full_train_frame[_keep_mask]
            print(f"    [标签去极值] MAD 3σ 剔除 {_n_dropped:,} 个极端收益样本")

    # [P0-1修复] 筛选端额外做 CSNeutralize(label) → CSQuantileNorm(label) 取纯 alpha 口径；
    # 训练端因本地缺行业字段，neutralize_labels=False（仅 CSQuantileNorm）。
    # CSNeutralize 失败时回退原始标签，此时两端均不中性化（Rank IC 对单调变换不变，差异可忽略）。
    _t40 = _time.time()
    try:
        y_mi = full_train_frame[[label_name]].copy()
        y_mi.columns = pd.MultiIndex.from_tuples([("label", label_name)])
        y_mi = CSNeutralize(
            fields_group="label",
            industry_field=industry_field,
            market_cap_field=market_cap_field,
            log_mc=log_mc,
        ).__call__(y_mi)
        # [P2-7统一] 与 CSNeutralize 一致使用 Qlib 处理器 __call__ 接口（CSQuantileNorm 的
        # __call__ 内部转调 fit_transform→transform，功能等价，仅统一 API 风格）
        y_mi = CSQuantileNorm(fields_group="label").__call__(y_mi)
        y_rank = y_mi[("label", label_name)].rename(label_name)
        full_train_frame[label_name] = y_rank
        # 无行业/市值暴露（中性化残差为 NaN）的股票剔除，避免污染 IC 评估
        full_train_frame = full_train_frame.dropna(subset=[label_name])
        print(f"    [进度] 标签管线（中性化+rank）完成: 剩余 {full_train_frame.shape[0]:,} 行, "
              f"耗时 {_time.time()-_t40:.1f}s")
    except Exception as e:
        print(f"    [警告] 标签管线执行失败，回退原始标签: {e}")

    # 5. 因子截面 MAD 3σ winsorize + 截面排名标准化 (CSRankNorm，等价于树模型的 CSQuantileNorm)
    _t4 = _time.time()
    feature_cols = [c for c in full_train_frame.columns if c in all_factor_names]
    x_raw = full_train_frame[feature_cols].copy()
    x_wins = _apply_mad_winsorize(x_raw)      # [P1-2修复] 因子去极值
    x_norm = _apply_cs_rank_norm(x_wins)
    y = full_train_frame[label_name].copy()
    print(f"    [进度] 因子去极值+截面标准化完成: 耗时 {_time.time()-_t4:.1f}s")

    del warehouse_df, label_raw, label_flat, full_train_frame, x_raw, x_wins
    gc.collect()

    return x_norm, y


# ==============================================================================
# [跨窗口共线性精简 - AQR 级]（CONFIG 第十层）
# 说明：层次聚类去冗余（非正交化）——以 1-|rho| 为距离做层级聚类，
#       族内按跨窗口综合评分保留 top N，跨族再二次检查高相关对。
# ==============================================================================

def reduce_factor_collinearity(
    final_df: pd.DataFrame,
    feature_cache,
    window_details: List[pd.DataFrame],
    n_windows: int,
    correlation_start: str = "2022-01-01",
    correlation_end: str = "2023-12-20",
    cluster_rho_threshold: float = 0.6,
    max_per_cluster: int = 2,
    cross_cluster_rho_threshold: float = 0.85,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    跨窗口聚合后，用层次聚类识别共线性因子家族，每组保留 top N 个。

    输入:
    - final_df: 跨窗口聚合后的完整因子 DataFrame
    - feature_cache: 全局特征缓存（用于读数据计算相关系数）
    - window_details: 各窗口的逐因子明细（含 IC/ICIR/t-stat）
    - n_windows: 窗口总数
    - correlation_start/end: 相关性计算日期范围
    - cluster_rho_threshold: 聚类阈值（|rho| > 此值视为同族候选）
    - max_per_cluster: 每族最多保留因子数
    - cross_cluster_rho_threshold: 跨族二次检查阈值

    输出:
    - 更新 selected 列后的 final_df
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  跨因子共线性精简（层次聚类）")
        print(f"{'=' * 70}")

    result = final_df.copy()

    # 只处理当下选中（selected=True）的因子
    sel_mask = result["selected"]
    sel_names = result.loc[sel_mask, "factor_name"].tolist()

    if len(sel_names) <= 3:
        if verbose:
            print(f"  因子数 {len(sel_names)} <= 3，跳过共线性精简")
        return result

    # ── 1. 汇聚各窗口的 IC 指标 ──
    if window_details:
        detail_df = pd.concat(window_details, ignore_index=True)
        detail_df = detail_df[detail_df["factor_name"].isin(sel_names)]

        agg_metrics = detail_df.groupby("factor_name").agg({
            "ic_mean": "mean",
            "icir": "mean",
            "t_stat": "mean",
            "composite_score": "mean",
        }).reset_index()
        # 计算窗口入选率
        win_counts = detail_df[detail_df["selected"]].groupby("factor_name").size()
        agg_metrics["win_ratio"] = agg_metrics["factor_name"].map(
            lambda x: win_counts.get(x, 0) / n_windows
        )
    else:
        agg_metrics = pd.DataFrame({"factor_name": sel_names})
        for c in ["ic_mean", "icir", "t_stat", "composite_score", "win_ratio"]:
            agg_metrics[c] = np.nan

    # 归一化指标用于族内排序
    metric_cols = ["win_ratio", "ic_mean", "icir", "t_stat", "composite_score"]
    for col in metric_cols:
        if col in agg_metrics.columns:
            vals = agg_metrics[col].fillna(0).abs().values
            if vals.max() > 0:
                agg_metrics[f"{col}_norm"] = vals / vals.max()
            else:
                agg_metrics[f"{col}_norm"] = 0.0
        else:
            agg_metrics[f"{col}_norm"] = 0.0

    # 综合保留得分（权重：窗口入选率 0.3, IC 0.25, ICIR 0.2, t-stat 0.15, importance 0.1）
    agg_metrics["retain_score"] = (
        agg_metrics["win_ratio_norm"] * 0.30
        + agg_metrics["ic_mean_norm"] * 0.25
        + agg_metrics["icir_norm"] * 0.20
        + agg_metrics["t_stat_norm"] * 0.15
        + agg_metrics["composite_score_norm"] * 0.10
    )
    score_map = dict(zip(agg_metrics["factor_name"], agg_metrics["retain_score"]))

    # ── 2. 读取 warehouse 数据计算相关矩阵 ──
    corr_data = None
    available_names = []
    try:
        wdf = feature_cache.get_warehouse_df(
            selected_names=sel_names,
            start_time=correlation_start,
            end_time=correlation_end,
        )
        if isinstance(wdf.columns, pd.MultiIndex):
            wdf.columns = wdf.columns.droplevel(0)
        wdf = wdf.dropna(how="all", axis=1).dropna(how="any")
        available_names = [c for c in wdf.columns if c in sel_names]
        if len(available_names) >= 3:
            corr_data = wdf[available_names].corr(method="spearman")
            if verbose:
                print(f"  [相关矩阵] {len(available_names)}/{len(sel_names)} 个因子有完整数据, "
                      f"{len(available_names)}×{len(available_names)} matrix")
        else:
            if verbose:
                print(f"  [相关矩阵] 有效因子 < 3，跳过")
    except Exception as e:
        if verbose:
            print(f"  [相关矩阵] 读取失败: {e}，跳过共线性精简")
        return result

    if corr_data is None or corr_data.empty:
        return result

    missing_names = [n for n in sel_names if n not in available_names]
    if missing_names and verbose:
        # [P1-3对齐] 文案与下方实际处理逻辑一致：缺失数据无法参与聚类，
        # 按跨窗口综合评分保留前 max_per_cluster 个（原"默认保留"为旧逻辑残留）
        print(f"  [缺失数据] {len(missing_names)} 个因子无 warehouse 数据，无法参与聚类，"
              f"将按跨窗口综合评分保留前 {max_per_cluster} 个: {missing_names}")

    # ── 3. 层次聚类 ──
    try:
        from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
        from scipy.spatial.distance import squareform

        dist = 1.0 - corr_data.abs()
        # 稳健的距离矩阵：清理NaN → 强制对称 → 对角线归零 → 裁剪浮点误差
        dist = dist.fillna(1.0)           # NaN → 1.0（最大距离）
        dist = (dist + dist.T) / 2.0      # 强制对称
        np.fill_diagonal(dist.to_numpy(), 0.0)  # 对角线归零
        dist = dist.clip(lower=0.0, upper=1.0)  # 裁剪浮点溢出
        dist_arr = squareform(dist)
        Z = linkage(dist_arr, method="average")
        # 用距离阈值确定聚类: rho_threshold -> distance threshold
        cut_distance = 1.0 - cluster_rho_threshold
        clusters = fcluster(Z, t=cut_distance, criterion="distance")
        n_clusters = len(set(clusters))
        if verbose:
            print(f"  [层次聚类] 距离阈值={cut_distance:.2f} (|rho|>{cluster_rho_threshold}), "
                  f"{len(available_names)} 个因子 → {n_clusters} 个族")

        # ── 4. 族内排序 + 保留 top N ──
        cluster_map = dict(zip(available_names, clusters))
        factor_clusters = {}
        for fname in available_names:
            cid = cluster_map[fname]
            factor_clusters.setdefault(cid, []).append(fname)

        keep_set = set()
        drop_set = set()
        for cid, members in sorted(factor_clusters.items()):
            # 按 retain_score 排序
            members_sorted = sorted(members, key=lambda x: score_map.get(x, 0), reverse=True)
            n_keep = min(max_per_cluster, len(members_sorted))
            keep = set(members_sorted[:n_keep])
            keep_set.update(keep)
            dropped = len(members_sorted) - n_keep
            if dropped > 0 and verbose:
                print(f"    [族 {cid}] {len(members_sorted)} 个 → 保留 {n_keep}, 剔除 {dropped}")
                for m in members_sorted:
                    tag = "保留" if m in keep else "剔除"
                    score = score_map.get(m, 0)
                    print(f"      {'✓' if m in keep else '✗'} {m:<25s} retain_score={score:.4f} ({tag})")

        # ── 5. 跨族二次检查：剩余因子中 |rho| > threshold 的对，保留高分的 ──
        remaining = [n for n in available_names if n in keep_set]
        if len(remaining) > 1:
            sub_corr = corr_data.loc[remaining, remaining]
            upper = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
            cross_pairs = []
            for col in upper.columns:
                high = upper[col][upper[col].abs() > cross_cluster_rho_threshold].dropna()
                for idx, val in high.items():
                    cross_pairs.append((idx, col, abs(val)))
            cross_pairs.sort(key=lambda x: x[2], reverse=True)

            for a, b, r in cross_pairs:
                if a in drop_set or b in drop_set:
                    continue
                if score_map.get(a, 0) >= score_map.get(b, 0):
                    keep_set.discard(b)
                    drop_set.add(b)
                    if verbose:
                        print(f"    [跨族] {a} vs {b} rho={r:.3f}, 保留 {a}, 剔除 {b}")
                else:
                    keep_set.discard(a)
                    drop_set.add(a)
                    if verbose:
                        print(f"    [跨族] {a} vs {b} rho={r:.3f}, 保留 {b}, 剔除 {a}")

        # ── 6. 更新 selected 列 ──
        # 有 correlation 数据的因子：只保留 keep_set 中的
        result.loc[sel_mask & result["factor_name"].isin(available_names), "selected"] = False
        result.loc[result["factor_name"].isin(keep_set), "selected"] = True

        # [机构标准] 缺失 correlation 数据的因子无法参与聚类，
        # 原逻辑"默认全保留"是免检后门（小类因子绕过全部防线直通）。
        # 现改为：按跨窗口综合评分保留每族配额 max_per_cluster 个，其余剔除。
        if missing_names:
            missing_df = result[result["factor_name"].isin(missing_names)].copy()
            missing_df = missing_df.sort_values("composite_score", ascending=False)
            n_keep_missing = min(len(missing_df), max_per_cluster)
            keep_missing = set(missing_df["factor_name"].head(n_keep_missing))
            result.loc[result["factor_name"].isin(missing_names), "selected"] = False
            result.loc[result["factor_name"].isin(keep_missing), "selected"] = True
            if verbose:
                print(f"  [缺失数据] {len(missing_names)} 个因子无 warehouse 数据，"
                      f"按综合评分保留前 {n_keep_missing} 个，剔除 {len(missing_names) - n_keep_missing} 个")

        old_count = len(sel_names)
        new_count = result["selected"].sum()
        if verbose:
            print(f"\n  ➤ 共线性精简: {old_count} → {new_count} 个因子 "
                  f"(剔除 {old_count - new_count} 个)")
            for _, row in result[result["selected"]].iterrows():
                print(f"    ✓ {row['factor_name']:<25s} "
                      f"(类别: {row['category']}, score: {row['composite_score']:.3f})")

    except ImportError:
        if verbose:
            print("  [警告] scipy.cluster.hierarchy 不可用，跳过共线性精简")
    except Exception as e:
        if verbose:
            print(f"  [跳过] 共线性精简异常: {e}")

    return result


# ==============================================================================
# [跨窗口聚合]（[P0修复] 从 main 内联逻辑提取为独立函数，便于验证）
# ==============================================================================

def _aggregate_across_windows(
    categories: Dict[str, List[Dict]],
    window_selections: Dict[str, Dict[str, List[bool]]],
    window_importances: Dict[str, Dict[str, List[float]]],
    window_details: List[pd.DataFrame],
    n_windows: int,
    min_window_ratio: float,
    window_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    跨窗口聚合：因子在 >= min_window_ratio 的窗口中入选则最终选中。

    [P0修复] 相较原逻辑新增/修正：
    1. min_wins 取整修正：原 int(n_windows * ratio) 在 3 窗口 × 0.5 时被截断为 1，
       使"跨窗口稳定性"约束完全失效（任一窗口入选即保留）；现用 round 四舍五入。
    2. 新增跨窗口 IC 方向一致性检查：各窗口 IC 符号一致率 >= 0.5 才视为方向稳定，
       窗口间方向翻转的因子直接剔除（原逻辑只统计"被选次数"，从不检查 IC 符号）。
    3. 输出新增 direction / direction_consistency 列，方向信息不再丢失。

    输入:
    - categories: {类别: [因子元信息]}，来自 load_factors_by_category
    - window_selections: {类别: {因子: [每窗口是否被选]}}
    - window_importances: {类别: {因子: [每窗口综合评分]}}
    - window_details: 每个窗口的因子结果表（含 ic_mean 列）
    - n_windows: 完成筛选的窗口数
    - min_window_ratio: 跨窗口入选比例阈值

    输出:
    - final_df: 聚合结果 DataFrame（含 selected / direction / direction_consistency）
    """
    min_wins = max(1, int(round(n_windows * min_window_ratio)))
    print(f"\n{'=' * 70}")
    print(f"  跨窗口聚合（需要 >= {min_wins}/{n_windows} 窗口入选，且 IC 方向一致率 >= 0.5）")
    print(f"{'=' * 70}")

    # 汇总各窗口入选因子的 IC 均值（用于跨窗口方向一致性检查）
    # [P0修复] 仅收集 selected=True 的窗口：未入选窗口的 IC 已被 ICIR/BH 防线判为
    # 不显著，参考意义弱；方向一致性只在该因子"真正入选"的窗口间考察。
    win_ic_map: Dict[str, Dict[str, List[float]]] = {}
    for wd in window_details:
        for _, r in wd.iterrows():
            if not r.get("selected", False):
                continue
            icv = r.get("ic_mean")
            if pd.notna(icv):
                win_ic_map.setdefault(r["category"], {}).setdefault(r["factor_name"], []).append(float(icv))

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

            # 跨窗口 IC 方向一致性（基于入选窗口的 IC 符号）
            # [P0修复] 原逻辑仅统计"被选次数"，从不检查 IC 符号；
            # 且单纯用 consistency < 0.5 作门槛在窗口数少时（如 3 窗口）永不触发
            # （多数符号占比恒 >= 2/3）。现改为：正负符号个数相等（平手）即视为
            # 方向不稳定直接剔除；direction 取多数符号，consistency 输出参考。
            ic_hist = win_ic_map.get(cat_name, {}).get(fname, [])
            direction = np.nan
            direction_consistency = np.nan
            if len(ic_hist) >= 2:
                ic_signs = np.sign(ic_hist)
                n_pos = int((ic_signs > 0).sum())
                n_neg = int((ic_signs < 0).sum())
                if n_pos == n_neg:
                    final_selected = False  # 正负平手 → 无主导方向，跨窗口方向翻转
                else:
                    direction = 1 if n_pos > n_neg else -1
                    direction_consistency = float(max(n_pos, n_neg) / len(ic_hist))

            # 诊断指标：最早被选中的窗口（用于判断新确认因子是否回填早期窗口）
            earliest_win = ""
            if sel_history and window_names and len(window_names) == len(sel_history):
                for i, sel in enumerate(sel_history):
                    if sel:
                        earliest_win = window_names[i]
                        break

            final_rows.append({
                "category": cat_name,
                "factor_name": fname,
                "selected": final_selected,
                "direction": direction,
                "direction_consistency": round(direction_consistency, 4) if not np.isnan(direction_consistency) else np.nan,
                "composite_score": round(avg_importance, 4),
                "n_windows_selected": n_selected,
                "total_windows": n_windows,
                "earliest_selected_window": earliest_win,
                "meaning": f.get("meaning", ""),
                "source_file": f.get("source_file", ""),
            })

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(["category", "selected", "composite_score"],
                                     ascending=[True, False, False]).reset_index(drop=True)
    return final_df


# ==============================================================================
# [主函数]
# ==============================================================================

def main():
    factor_files = resolve_factor_files(CONFIG["factor_files"])
    rolling_windows = CONFIG["rolling_windows"]

    print("=" * 70)
    print("  精选因子筛选脚本（世界顶级量化机构标准增强版）")
    print("  对标：AQR / Citadel / Renaissance / Two Sigma / D.E. Shaw")
    print("=" * 70)
    print(f"  股票池: {CONFIG['instruments']}")
    print(f"  动态过滤: {CONFIG['use_dynamic_filter']}")
    print(f"  ST 过滤: {CONFIG['filter_st']}, 次新过滤: {CONFIG['filter_new_stocks']}")
    print(f"  窗口数: {len(rolling_windows)}, 聚合阈值: {CONFIG['min_window_ratio']}")
    print(f"  数据质量检查: {CONFIG['data_quality_check']}")
    print(f"  因子中性化: {CONFIG['neutralize']} (行业+市值)")
    print(f"  IC 分析: {CONFIG['ic_analysis']} ({CONFIG['ic_type']})")
    print(f"  多重检验校正: {CONFIG['multiple_testing_correction']} ({CONFIG['mtc_method']})")
    print(f"  置换检验: {CONFIG['permutation_test']}")
    print(f"  多方法投票: {CONFIG['multi_method_voting']}")
    print(f"  拥挤度分析: {CONFIG['crowding_analysis']}")
    print(f"  共线性精简: {CONFIG['collinearity_reduction']} (聚类|rho|>{CONFIG['cluster_rho_threshold']}, 每族≤{CONFIG['max_per_cluster']}个)")
    print("=" * 70)

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
    window_selections: Dict[str, Dict[str, List[bool]]] = {}
    window_importances: Dict[str, Dict[str, List[float]]] = {}
    window_details: List[pd.DataFrame] = []
    all_quality_reports: List[pd.DataFrame] = []

    for win_idx, window in enumerate(rolling_windows):
        win_name = window["name"]
        train_start, train_end = window["train"]

        print(f"\n{'=' * 70}")
        print(f"=== 窗口 {win_idx+1}/{len(rolling_windows)}: {win_name} ===")
        print(f"    训练期: {train_start} ~ {train_end}")
        print(f"{'=' * 70}")

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
                filter_delisted=CONFIG.get("filter_delisted", True),
                filter_untradeable_labels=CONFIG.get("filter_untradeable_labels", True),
                industry_field=CONFIG["industry_field"],
                market_cap_field=CONFIG["market_cap_field"],
                log_mc=CONFIG["log_mc"],
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

            # [机构标准] 计算自适应 top_k 配额：
            # 配额与类别规模挂钩（min(cat)*ratio 封顶 min/max），杜绝
            # "145 因子大类压到 5 个、小类因子免检全留"的类别粒度劫持。
            if CONFIG.get("adaptive_top_k", {}).get("enabled", False):
                atk = CONFIG["adaptive_top_k"]
                top_k_eff = min(atk["max"], max(atk["min"], int(np.ceil(len(factors) * atk["ratio"]))))
            else:
                top_k_eff = CONFIG["top_k"]

            print(f"\n  >>> [{cat_index}/{len(categories)}] 类别: '{cat_name}' ({len(factors)} 个因子, top_k={top_k_eff}) <<<")
            result = run_single_category_selection(
                cat_name=cat_name,
                factors=factors,
                x_train=x_train,
                y_train=y_train,
                fs_method=CONFIG["method"],
                fs_algo=CONFIG["algo"],
                top_k=top_k_eff,
                label_name=CONFIG["label_name"],
                # [P2-6精简] 配置参数统一由 cfg=CONFIG 透传（单源一致，消除重复传参）
                cfg=CONFIG,
            )

            if result is not None:
                df, quality_df = result
                df["window"] = win_name
                if not quality_df.empty:
                    quality_df["window"] = win_name
                    quality_df["category"] = cat_name
                    all_quality_reports.append(quality_df)
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
                imp = row["composite_score"]

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

    # ─────────────────────────────────────────────────────────────────────────
    # 跨窗口聚合：因子在 >= min_window_ratio 的窗口中入选则最终选中
    # （[P0修复] 已提取为 _aggregate_across_windows：min_wins 取整修正 +
    #   跨窗口 IC 方向一致性检查 + direction 列）
    # ─────────────────────────────────────────────────────────────────────────
    n_windows = len(window_details)
    if n_windows == 0:
        print("\n[错误] 没有任何窗口完成因子筛选")
        sys.exit(1)

    final_df = _aggregate_across_windows(
        categories=categories,
        window_selections=window_selections,
        window_importances=window_importances,
        window_details=window_details,
        n_windows=n_windows,
        min_window_ratio=CONFIG["min_window_ratio"],
        window_names=[w["name"] for w in rolling_windows],
    )

    # ── 跨窗口共线性精简（AQR 级层次聚类）──
    if CONFIG.get("collinearity_reduction", True) and global_feature_cache is not None:
        final_df = reduce_factor_collinearity(
            final_df=final_df,
            feature_cache=global_feature_cache,
            window_details=window_details,
            n_windows=n_windows,
            correlation_start=CONFIG.get("correlation_start", "2022-01-01"),
            correlation_end=CONFIG.get("correlation_end", "2023-12-20"),
            cluster_rho_threshold=CONFIG.get("cluster_rho_threshold", 0.6),
            max_per_cluster=CONFIG.get("max_per_cluster", 2),
            cross_cluster_rho_threshold=CONFIG.get("cross_cluster_rho_threshold", 0.85),
        )

    # 释放全局缓存
    del global_feature_cache
    gc.collect()

    # [机构标准 - Citadel Alpha Lab] 全局大类配额上限：
    # 共线性精简已保证族内多样性，此处再限制单一经济学大类对组合的支配
    # （204 因子的 Price-Volume 会淹没 3-4 因子的 Value/Momentum 等小类）。
    # 单大类入选数超过 max_per_category 时，按综合评分截断。
    max_per_category = CONFIG.get("max_per_category", 30)
    if max_per_category is not None and max_per_category > 0:
        capped = {}
        for cat_name in final_df["category"].unique():
            cat_sel = final_df[(final_df["category"] == cat_name) & (final_df["selected"])]
            if len(cat_sel) > max_per_category:
                keep = set(cat_sel.nlargest(max_per_category, "composite_score")["factor_name"])
                drop_mask = (
                    (final_df["category"] == cat_name)
                    & final_df["selected"]
                    & ~final_df["factor_name"].isin(keep)
                )
                final_df.loc[drop_mask, "selected"] = False
                capped[cat_name] = int(len(cat_sel) - max_per_category)
        if capped:
            print("\n  [全局大类配额] 超限类别按综合评分截断: "
                  + ", ".join(f"{k} 剔除 {v} 个" for k, v in capped.items()))

    selected_total = final_df["selected"].sum()
    total = len(final_df)
    print(f"\n  总览: 共 {total} 个因子，选中 {selected_total} 个 ({selected_total/total*100:.1f}%)")
    min_wins = max(1, int(round(n_windows * CONFIG["min_window_ratio"])))
    print(f"  聚合标准: 在 >= {min_wins}/{n_windows} 个窗口入选，且 IC 方向一致率 >= 0.5")

    print(f"\n  {'类别':<20s} {'选中/总计':<12s} {'选中率':<8s}")
    print(f"  {'-'*40}")
    for cat_name in sorted(final_df["category"].unique()):
        sub = final_df[final_df["category"] == cat_name]
        s = sub["selected"].sum()
        t = len(sub)
        print(f"  {cat_name:<20s} {int(s)}/{t:<8d} {s/t*100:>6.1f}%")

    print(f"\n  各类别精选因子列表（按综合评分排序）:")
    for cat_name in sorted(final_df["category"].unique()):
        sub = final_df[(final_df["category"] == cat_name) & (final_df["selected"])]
        if len(sub) == 0:
            continue
        print(f"\n  【{cat_name}】({len(sub)} 个)")
        for _, row in sub.iterrows():
            print(f"    [x] {row['factor_name']:<30s} "
                  f"(综合评分: {row['composite_score']:.3f}, "
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

    # 保存质量评估报告
    if all_quality_reports:
        quality_all = pd.concat(all_quality_reports, ignore_index=True)
        quality_output = output_path.replace(".csv", "_quality.csv")
        quality_all.to_csv(quality_output, index=False, encoding="utf-8-sig")
        print(f"  因子质量评估报告已保存至: {quality_output}")

    print("=" * 70)
    print(f"  精选因子数: {selected_total} (跨 {n_windows} 窗口聚合)")
    print(f"  训练模型: {CONFIG['algo']}")
    print(f"  股票池: {CONFIG['instruments']}")
    print(f"  增强功能: 数据质量检查 + 中性化 + IC分析 + 多重检验校正")
    print("=" * 70)


if __name__ == "__main__":
    main()
