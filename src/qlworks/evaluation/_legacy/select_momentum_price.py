"""
Momentum&Price 量价趋势家族专用因子筛选脚本。

由 select_factors.py 改造而来，仅针对 factor_library/momentum_price_factors.yaml
（动量与反转 13 + 量价形态 57 + 滚动统计 145 = 215 因子）做差异化筛选。

[与通用版 select_factors.py 的差异化改造]（对应世界一流机构对"大因子族"的操作标准）
  1. 因子库硬编码为 momentum_price 家族，不参与其它家族。
  2. 多重检验校正：置换检验由"单因子名义阈值 p<0.05"升级为 Benjamini-Hochberg
     FDR 控制（族大、近亲多 → 有效独立检验数远小于 215，Bonferroni 过严）。
  3. IC 衰减由"诊断"升级为"硬门槛"：量价因子必须在持有期（周频=H5）内保持预测力，
     持有期崩塌或方向翻转的因子直接淘汰。
  4. 类内冗余阈值收紧 0.90→0.85（BETA5/10/20/30/60 等近亲需更严去重），
     全局跨类冗余 0.70→0.65。
  5. 方向一致性门控：量价因子 IC 方向在 H1/H3/H5 须同号，动量-反转混叠因子淘汰。

[运行频率与窗口前滚约定]（研究用途：产候选池 → 组因子组合 → 进实盘池；
 不对接下游选股/训练脚本的日期）
  - 本脚本筛选属"慢决策"：候选池刷新按【月度】一次（更稳可【季度】），**不建议每周**
    —— ~2 年窗口每周仅增约 1% 信息，且会在硬门槛附近引发候选池抖动，拖累下游特征稳定性。
  - "候选池 → 实盘因子组合"同属慢决策，按【月度】一次。
  - "模型训练/选股打分"与调仓对齐（周频），属快决策，不在本脚本范围。
  - 窗口前滚：每次重跑把三个**互不重叠**窗口整体前滚一个重跑步长
    （月度重跑前滚 1 个月、季度重跑前滚 3 个月），始终保持"互不重叠 + 近端数据新鲜"。

用法：
  修改文件顶部 CONFIG 字典中的参数，然后直接运行：
    python select_momentum_price.py

输出：
  - 控制台打印各子类的因子筛选结果（含重要性得分、IC 衰减/方向门控结论）
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
    compute_daily_ic_frame,
)
from qlworks.pipeline_config import (
    LABEL_EXPR, LABEL_NAME, INSTRUMENTS,
    REDUNDANCY_THRESHOLD, ICIR_WINDOW, ICIR_KEEP_RATIO,
    FILTER_ST, FILTER_NEW_STOCKS, FILTER_LIMIT_UPDOWN,
)
import qlib
from qlib.data import D

# ==============================================================================
# 路径与因子库发现
# ==============================================================================
FACTOR_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "factor_data" / "factor_library"
ARCHIVE_DIR = FACTOR_LIBRARY_DIR / "archive"

def discover_factor_files() -> List[str]:
    """自动扫描因子库目录，返回活跃家族文件名（不含 .yaml 后缀）。

    因子库按 5+1 家族文件组织（quality/value/momentum_price/risk/sentiment/other），
    家族增删无需改动脚本；archive/ 为历史备份目录，Path.glob("*.yaml") 仅匹配
    顶层文件，天然排除 archive 子目录。
    """
    files = sorted(p.stem for p in FACTOR_LIBRARY_DIR.glob("*.yaml"))
    if not files:
        print("[警告] 因子库目录下无 YAML 因子文件")
    return files

# ==============================================================================
# [全局配置区] - 在此修改运行参数
# ==============================================================================
CONFIG = {
    # 因子文件列表：本脚本固定为 momentum_price 量价趋势家族（差异化改造点 1）
    "factor_files": ["momentum_price_factors"],

    # 特征选择参数
    "top_k": 2,              # [量价差异化] filter/wrapper 保留数 + ICIR 校验保底；embedded 走阈值驱动
    "method": "embedded",
    "algo": "lightgbm",
    "min_factors": 3,

    # [量价差异化] 重要性阈值筛选：替代嵌入法的固定 top_k 数量截断，让"数量成为输出"。
    # 嵌入法只给出重要性排序，此处按"importance >= min_relative × 最大 importance"保留，
    # 最终规模由后续 FDR 置换检验 + 冗余去重 + IC 衰减硬门槛自然收敛。
    "importance_filter": {
        "enabled": True,
        "min_relative": 0.05,   # 保留 importance >= 5% × 最大 importance 的因子
        "min_keep": 3,          # 至少保留（防阈值过严误杀）
    },

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
        # [G1] stride 提高到持有期 H=5：DK_L 标签为 5 日前瞻收益，相邻交易日的标签窗口重叠，
        # IC 序列存在强序列自相关。隔 5 日采样使"用于 IC 的标签窗口互不重叠"，消除乐观偏差。
        "stride": 5,
        "top_k_ratio": 0.4,           # [量价差异化] 类别内线性 IC 粗筛保留比例 0.6→0.4（大族抗多重检验）
        # [G2] 非线性信号补充：线性 IC 无法捕捉"量价形态"（阈值型/非单调）子族，
        # 另按非线性信号强度保留一定比例并取【并集】，避免线性粗筛误杀非线性因子。
        "nonlinear_ratio": 0.3,       # 非线性评分保留比例（并集补充）
        "nonlinear_bins": 10,         # 非线性评分分桶数
        "min_keep": 3,                # 至少保留因子数（IC 粗筛保底）
        "stable_ic": True,            # 跨窗口稳定 IC（首窗口退化为单窗口 |IC|）
    },

    # --- [P2-整合] 置换检验（来自 train_tree-doubao，向量化批量 Spearman）---
    # 对"嵌入法重要性阈值"后的候选因子做显著性检验，剔除纯噪声因子（p >= 阈值）；
    # 显著因子过少时保留原结果（min_keep 保护，防因子池崩空）。
    # [P1-4 口径说明] BH-FDR 的控制范围 = **本候选子集**（非全族 215），属"条件 FDR"：
    # 之前的 IC 粗筛（比例截断）引入的选择偏差不在本步校正范围内，故不宣称"全族多重
    # 检验已受控"。此处选 BH 而非 Bonferroni，是为适配"候选内近亲多、相关性高"的特性。
    "permutation_test": {
        "enabled": True,
        "n_permutations": 500,        # [量价差异化] 置换次数 200→500，提升 FDR 阈值下 p 值粒度
        "pvalue_threshold": 0.05,     # [量价差异化] FDR 控制水平 q（作用于候选子集，非全族名义阈值）
        "fdr": True,                  # [量价差异化] 启用 Benjamini-Hochberg FDR 多重检验校正
        # [G1] 置换前按 stride 采样日期（=持有期 H=5）：行级打乱置换要求样本近似独立，
        # 而 5 日重叠标签使相邻交易日高度相关，直接打乱会低估零分布方差、p 值偏乐观。
        # 隔 H 日采样后标签窗口互不重叠，行级置换零分布才成立。
        "stride": 5,
        "min_keep": 3,                # 至少保留显著因子数
    },

    # --- [P2-整合] 因子中性化显著性筛选（来自 train_tree 的 factor_neutralization）---
    # 对嵌入法+置换检验后的候选因子做行业+市值 Ridge 中性化，只保留中性化后
    # |IC| 仍 > min_ic_after_neutralize 的"纯 Alpha"因子。默认关闭——树模型路线下
    # 纯 alpha 口径由标签侧 DK_L 保证，因子侧仅做标准化；开启时因子侧也剥离风格。
    "factor_neutralization": {
        "enabled": False,                 # 默认关，对齐训练端 DK_L 口径
        "industry_field": "sw_l1",
        "market_cap_field": "circ_mv",
        "log_mc": True,
        "min_ic_after_neutralize": 0.005,
        "min_keep": 2,
        "min_keep_ratio": 0.3,
    },

    # --- [P2-整合] 因子体检（来自 train_tree 的 _compute_ic_decay / _compute_group_ic）---
    # 跨窗口聚合后对最终入选因子做诊断：IC 衰减曲线 + 行业/市值分组 IC。
    # 仅输出诊断报告，不改变筛选结果。
    "factor_health_check": {
        "enabled": True,
        "horizons": [1, 3, 5, 10, 20],              # IC 衰减前瞻周期
        "analysis_types": ["industry", "marketcap"],  # 分组 IC 类型
    },

    # --- [量价差异化] IC 衰减硬门槛（由"体检诊断"升级为"淘汰条件"）---
    # 周频调仓 = 持有 5 个交易日（LABEL_EXPR 的 H5 即持有期末）。因此淘汰锚点必须落在
    # 持有期末：要求因子在 H5 仍保有相对短周期（H1/H3）的预测力，即"信号能活到卖出时点"。
    # [P0-3 修复] 原门槛用 |IC_H20|/max(|IC_H1..H5|) —— H20 已超出持有期 4 倍，与周频 P&L
    # 无关，会误杀"持有期内有效、之后正常衰减"的量价因子；现改为锚定 H5，H20 仅留体检诊断。
    "decay_gate": {
        "enabled": True,
        "horizons": [1, 3, 5, 10, 20],              # 衰减曲线前瞻周期（与体检一致）
        "holding_horizon": 5,                       # 持有期前瞻周期（周频=5交易日，与 LABEL_EXPR 对齐）
        "decay_floor": 0.30,                        # |IC_H5| / max(|IC_H1|,|IC_H3|) 下限，低于则判"持有期崩塌"
        "short_horizons": [1, 3],                   # 持有期内早期信号，作为衰减参照基准
        "sign_consistency_horizons": [1, 3, 5],     # 方向须同号的前瞻周期（动量-反转混叠判定）
        "min_valid_horizons": 3,                    # 至少多少个 horizon 有有效 IC 才做门控（不足则放行）
    },

    # --- 滚动窗口因子筛选（研究用途，不对接下游选股/训练脚本日期）---
    # 窗口唯一职责 = 为"跨期稳健性投票"提供**互不重叠**的独立时间切片：
    # 因子需在 >= min_window_ratio(0.5 → 3 窗即 2/3) 的窗口中入选才进候选池。
    # 采用 3 个互不重叠的 ~2 年窗口，覆盖最近 ~6.7 年、跨越多个 regime；
    # 末窗止于最新可用样本日（数据到 2026-09-11，去 5 日前瞻后 ≈ 2026-09-04）。
    # 重跑前滚方式见文件头[运行频率与窗口前滚约定]。
    "rolling_windows": [
        {
            "name": "Window_A",
            "train": ("2020-01-01", "2021-12-20"),  # 2020 疫情 / 2021 动量崩塌
        },
        {
            "name": "Window_B",
            "train": ("2022-01-01", "2023-12-20"),  # 2022 熊市 / 2023 修复
        },
        {
            "name": "Window_C",
            "train": ("2024-01-01", "2026-09-04"),  # 2024 反弹 / 2025-26 近期（权重偏高）
        }
    ],

    # 标签（共享配置，与 screen_factors / train_tree-doubao / 评测端一致）
    "label_expr": LABEL_EXPR,
    "label_name": LABEL_NAME,

    # 冗余检测（阈值引用 pipeline_config 单一事实源，与粗筛/训练端统一）
    "redundancy_check": True,
    "redundancy_threshold": 0.85,  # [量价差异化] 类内 ρ 收紧 0.90→0.85（滚动统计近亲多）

    # [归档 2026-10-08] 因子库 YAML 内 category 已是 Barra 大类终态，无需类别归并。

    # [P0 修复] 全局跨类冗余兜底：类内 check_redundancy 只覆盖本类因子列，
    # 跨类冗余（如 circ_mv[市值与流动性] vs t0otal_mv_log[风格]，ρ≈0.99）无人管控；
    # 跨窗口聚合后对入选因子做一次全局 Spearman 去重（阈值严于类内 0.90）。
    "global_redundancy_check": True,
    "global_redundancy_threshold": 0.65,  # [量价差异化] 全局跨类冗余 0.70→0.65

    # ICIR 稳定性校验（引用共享配置，与粗筛/训练端统一）
    "icir_stability": True,
    "icir_window": ICIR_WINDOW,
    "icir_keep_ratio": ICIR_KEEP_RATIO,

    # 跨窗口聚合：因子在 >= min_window_ratio 比例的窗口中入选才最终选中
    "min_window_ratio": 0.5,

    # 输出：None 自动生成时间戳文件名
    "output": None,

    # [P1-6] 精选结果回写候选池（Alpha Book）作为准入建议（不直接 admitted）
    # 候选池的最终准入仍由 run_eval.py --admit 三关检验唯一决定。
    "write_pool": False,

    # 缓存
    "clean_start": False,
}


# ==============================================================================
# 因子库加载与分组
# ==============================================================================
def resolve_factor_files(factor_files) -> List[str]:
    """解析因子文件列表。支持 'all' 表示自动发现所有活跃家族文件。"""
    if factor_files == "all":
        return discover_factor_files()
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


# ==============================================================================
# 数据准备与标准化
# ==============================================================================
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
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    为单个滚动窗口准备训练数据：切片时间范围 → 合并标签 → 标准化 → 过滤。

    返回: (x_train, y_train, x_train_raw)
    - x_train: 已 CSRankNorm 的因子矩阵（供筛选主流程复用）
    - y_train: DK_L 标签 Series（中性化 + 分位化）
    - x_train_raw: 原始因子值矩阵（供因子中性化显著性在原始值上做 Ridge 中性化）
    """
    # 1. 从缓存切片特征数据（按需惰性合并）
    warehouse_df = global_feature_cache.get_warehouse_df(
        selected_names=all_factor_names,
        start_time=train_start,
        end_time=train_end,
    )
    if warehouse_df.empty:
        # [P0-1 修复] 返回三元组占位，与函数声明的 (x_train, y_train, x_train_raw) 对齐；
        # 原实现只返回 2 个值，调用方 3 元解包会触发 ValueError（被 except 误吞）。
        return pd.DataFrame(), pd.Series(dtype="float64"), pd.DataFrame()
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
    # [修复] 参数名 filter_untradeable_labels(bool) 遮蔽了模块级导入的同名函数，
    # 导致 bool 被当函数调用 → TypeError；改为局部导入别名（与 screen_factors._load_label_series 一致）
    if filter_untradeable_labels:
        from qlworks.factors.filter_utils import filter_untradeable_labels as _filter_untradeable_labels
        label_raw = _filter_untradeable_labels(
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
    x_raw = x_raw.loc[valid_mask]

    del warehouse_df, label_raw, label_flat, full_train_frame, y_mi
    gc.collect()

    return x_norm, y, x_raw


def _neutralize_label_series(
    label_df: pd.DataFrame,
    label_name: str,
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
    log_mc: bool = True,
) -> pd.Series:
    """[P2 修复] 对裸收益标签做行业+市值 Ridge 中性化，得到与 DK_L 同口径的纯 alpha 标签。

    用途：体检模块(IC 衰减/分组 IC)原先直接用裸收益标签算 IC，混入市值/行业 beta，
    与筛选阶段(DK_L 标签)口径不一致；统一为中性化标签后，体检 IC 可与筛选 IC 直接对比。

    注：compute_ic 为 Spearman 秩相关，对单调变换(如 CSQuantileNorm)不变，故无需再分位化。

    输入:
    - label_df: 单列裸收益 DataFrame，MultiIndex (datetime, instrument)
    - label_name: 标签列名
    - industry_field / market_cap_field / log_mc: CSNeutralize 中性化参数

    输出:
    - 中性化后的标签 Series（(datetime, instrument) 对齐，含 NaN）
    """
    y_mi = label_df.copy()
    y_mi.columns = pd.MultiIndex.from_tuples([("label", label_name)])
    y_mi = CSNeutralize(
        fields_group="label",
        industry_field=industry_field,
        market_cap_field=market_cap_field,
        log_mc=log_mc,
    ).__call__(y_mi)
    return y_mi[("label", label_name)].rename(label_name)


# ==============================================================================
# 类别筛选链
# ==============================================================================
def _mean_daily_ic(
    factor_frame: pd.DataFrame,
    label_series: pd.Series,
) -> pd.Series:
    """逐日截面 Spearman IC 的均值（quant 标准 IC 口径），index = 因子名。

    [G3 口径统一] 全 panel pooled Spearman 会把"时间维的标签方差"混入 IC，口径偏乐观；
    逐日截面 IC 再取时间均值，才是与 ICIR 校验(check_icir_stability)/评测端一致的 IC 定义。
    本函数收敛该口径，供 IC 粗筛与 IC 衰减复用，避免各阶段口径漂移。

    输入:
    - factor_frame: 因子面板（MultiIndex 含 'datetime' level）
    - label_series: DK_L 标签 Series（与 factor_frame 行对齐）
    输出:
    - Series（index=因子列名，values=逐日截面 IC 的时间均值；无有效截面则 NaN）
    """
    daily = compute_daily_ic_frame(factor_frame, label_series)
    if daily.empty:
        return pd.Series(dtype=float)
    return daily.mean(axis=0, skipna=True)


def _nonlinear_signal_score(
    feat: pd.Series,
    lab: pd.Series,
    bins: int = 10,
) -> float:
    """[G2] 非线性信号评分：因子分位分桶后各桶标签均值的离散度（不假设单调）。

    线性 Spearman IC 只度量单调关系，会低估"量价形态"这类阈值型/非单调子族
    （如 U 型、反转拐点）。本评分衡量"条件标签均值随因子分桶的整体离散度"，
    单调与非单调信号均能被捕捉，用于在 IC 粗筛阶段为非线性因子保留名额。

    输入:
    - feat: 单因子值（已 CSRankNorm，取值近似 [0,1]，可比桶）
    - lab:  DK_L 标签 Series（与 feat 行对齐）
    - bins: 分桶数
    输出:
    - 非负评分（越大越有信号）；样本不足或无有效分桶返回 0.0
    """
    df = pd.concat([feat.rename("f"), lab.rename("y")], axis=1).dropna()
    if len(df) < 50:
        return 0.0
    try:
        q = pd.qcut(df["f"], bins, labels=False, duplicates="drop")
    except Exception:
        return 0.0
    if q.nunique(dropna=True) < 2:
        return 0.0
    gm = df.groupby(q.values)["y"].mean()
    return float(gm.std())


def _ic_coarse_select(
    x_sub: pd.DataFrame,
    y_sub: pd.Series,
    factor_names: List[str],
    ic_history: Optional[List[Dict]] = None,
    stride: int = 5,
    top_k_ratio: float = 0.4,
    min_keep: int = 3,
    stable_ic: bool = True,
    nonlinear_ratio: float = 0.3,
    nonlinear_bins: int = 10,
) -> Tuple[List[str], Dict[str, float]]:
    """
    类别内 IC 粗筛：逐因子计算与 DK_L 标签的【逐日截面 Spearman IC 均值】(G3 统一口径)，
    支持跨窗口稳定 IC（方向一致性 + 均值 |IC|）与隔日 stride 采样。

    输入:
    - x_sub: 类别因子矩阵（MultiIndex (datetime, instrument) × 因子列，已 CSRankNorm）
    - y_sub: DK_L 标签 Series（与 x_sub 行对齐）
    - factor_names: 类别内因子名列表（待粗筛）
    - ic_history: 跨窗口 IC 历史（list of dict，每元素 = {因子名: 该窗口IC}，时间升序）
    - stride: 隔日采样步长（[G1] 取持有期 H=5，令标签窗口互不重叠，消除序列自相关污染）
    - top_k_ratio: 线性 IC 保留比例（候选数 = max(ceil(len * ratio), min_keep)）
    - min_keep: 至少保留因子数
    - stable_ic: 是否启用跨窗口稳定 IC 评分
    - nonlinear_ratio: [G2] 非线性评分保留比例（与线性集合取并集，保护量价形态子族）
    - nonlinear_bins: [G2] 非线性评分分桶数

    输出:
    - (候选因子名列表[按线性 IC 降序]，本窗口 {因子名: 逐日截面 IC 均值})
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

    # 逐因子 IC：G3 统一为"逐日截面 Spearman IC 的时间均值"（与 ICIR 校验/评测端同口径）
    y_aligned = y_ic.reindex(x_ic.index)
    _y_mask = y_aligned.notna().to_numpy()
    _counts = (x_ic.notna().to_numpy() & _y_mask[:, None]).sum(axis=0)
    eligible = [c for c, n in zip(avail, _counts) if n >= 50]
    ics: Dict[str, float] = {}
    if eligible:
        ic_mean = _mean_daily_ic(x_ic[eligible], y_aligned)
        for col in eligible:
            v = ic_mean.get(col, np.nan)
            if pd.notna(v):
                ics[col] = float(v)

    # [G2] 非线性信号评分（仅对通过覆盖率门的因子计算），用于保护"量价形态"非线性子族
    nl_score: Dict[str, float] = {
        col: _nonlinear_signal_score(x_ic[col], y_aligned, bins=nonlinear_bins)
        for col in ics
    }

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
    if not ranked:
        return [], ics
    # 线性 IC 名额
    keep_n = max(int(np.ceil(len(ranked) * top_k_ratio)), min_keep)
    linear_set = set(ranked[:keep_n])
    # [G2] 非线性评分名额，与线性集合取并集（防止线性粗筛误杀非线性因子）
    nl_ranked = sorted([k for k in nl_score if nl_score[k] > 0],
                       key=lambda k: nl_score[k], reverse=True)
    keep_nl = int(np.ceil(len(nl_ranked) * nonlinear_ratio)) if nl_ranked else 0
    keep_set = linear_set | set(nl_ranked[:keep_nl])
    coarse = [k for k in ranked if k in keep_set]
    if len(coarse) < min_keep:
        coarse = ranked[:min_keep]
    return coarse, ics


def _benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """[量价差异化] Benjamini-Hochberg FDR 多重检验校正，返回显著布尔 mask。

    BH 步骤：p 值升序排序，找到最大 k 使 p_(k) <= (k/m)*q，则排序后前 k 个判显著。
    相比 Bonferroni（单阈值 q/m），BH 对近亲/相关检验更温和，适合 momentum_price
    这类"大族+近亲多"的因子族（有效独立检验数远小于 215，Bonferroni 会全盘误杀）。

    输入:
    - p_values: 一维 p 值数组
    - q: FDR 控制水平（预期误报比例上限）

    输出:
    - 与 p_values 等长的布尔数组，True 表示经 FDR 校正后显著
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    k = int(np.max(np.where(below)[0])) + 1 if below.any() else 0
    significant = np.zeros(m, dtype=bool)
    significant[order[:k]] = True
    return significant


def _permutation_significance(
    x_sub: pd.DataFrame,
    y_sub: pd.Series,
    selected_names: List[str],
    n_perms: int = 200,
    pvalue_threshold: float = 0.05,
    min_keep: int = 3,
    fdr: bool = False,
    stride: int = 5,
) -> List[str]:
    """
    置换检验：对选中的因子做双尾显著性检验（向量化批量 Spearman）。

    [量价差异化] fdr=True 时，pvalue_threshold 语义由"单因子名义阈值"变为
    Benjamini-Hochberg FDR 控制水平 q，对批量 p 值做多重检验校正。
    [P1-4 口径说明] FDR 控制范围 = 本候选子集（selected_names），属"条件 FDR"；
    IC 粗筛引入的选择偏差不在本步校正范围内。
    [G1 口径说明] 行级打乱置换要求样本近似独立；5 日重叠标签使相邻交易日高度相关，
    直接打乱会低估零分布方差、p 值偏乐观。故按 stride 采样日期（默认=持有期 H=5）令
    标签窗口互不重叠，再行级打乱。本检验使用 pooled Spearman 作为【检验统计量】
    （非 IC 排序口径），IC 的排序/衰减口径已由 G3 统一为逐日截面 IC。

    输入:
    - x_sub: 类别因子矩阵（含 selected_names 列）
    - y_sub: DK_L 标签 Series（与 x_sub 行对齐）
    - selected_names: 待检验因子名列表
    - n_perms: 置换次数
    - pvalue_threshold: 双尾 p 值阈值（fdr=True 时为 FDR 水平 q）
    - min_keep: 显著因子少于该值时保留原列表（防因子池崩空）
    - fdr: 是否启用 Benjamini-Hochberg FDR 多重检验校正
    - stride: [G1] 置换前的日期采样步长（默认 5=持有期，令标签不重叠）

    输出:
    - 显著因子名列表（保序；显著不足时返回原列表）
    """
    from scipy.stats import rankdata

    avail = [c for c in selected_names if c in x_sub.columns]
    if len(avail) < 2:
        return selected_names
    comb = x_sub[avail].join(y_sub.rename("_y"), how="inner").dropna()
    # [G1] 按 stride 采样日期：使参与置换的标签窗口互不重叠，行级打乱零分布才成立
    if stride > 1 and len(comb) > 0:
        _dates = comb.index.get_level_values("datetime").unique()[::stride]
        comb = comb[comb.index.get_level_values("datetime").isin(_dates)]
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

    if fdr:
        # [量价差异化] BH-FDR 多重检验校正（q=pvalue_threshold，控制预期误报比例）
        sig_mask = _benjamini_hochberg(p_values, q=pvalue_threshold)
        significant = [feat_cols[i] for i in range(n_factors) if sig_mask[i]]
        print(f"    [置换检验] 显著 {len(significant)}/{n_factors} 个 "
              f"(BH-FDR q<{pvalue_threshold}，候选子集内)")
    else:
        significant = [feat_cols[i] for i in range(n_factors) if p_values[i] < pvalue_threshold]
        print(f"    [置换检验] 显著 {len(significant)}/{n_factors} 个 (p<{pvalue_threshold})")

    n_sig = len(significant)
    if n_sig < min_keep:
        print(f"    [置换检验] 显著因子过少({n_sig}<{min_keep})，保留原筛选结果")
        return selected_names
    return significant


def _neutralize_significance(
    x_sub: pd.DataFrame,
    y_sub: pd.Series,
    selected_names: List[str],
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
    log_mc: bool = True,
    min_ic_after_neutralize: float = 0.005,
    min_keep: int = 2,
    min_keep_ratio: float = 0.3,
) -> List[str]:
    """[AQR] 因子中性化后显著性筛选：对因子做行业+市值 Ridge 中性化后，
    只保留中性化后仍有预测力（|IC| > min_ic_after_neutralize）的因子。

    作用：剥离因子的行业/市值风格暴露，确保最终入选的是"纯 Alpha"因子，
    而非靠市值/行业 beta 蹭 IC 的风格因子（收敛于 train_tree 的 factor_neutralization）。

    输入:
    - x_sub: 类别因子矩阵（(datetime, instrument) × 因子列，**原始值**，未 CSRankNorm）
    - y_sub: DK_L 标签 Series（与 x_sub 行对齐）
    - selected_names: 待检验因子名列表（通常为嵌入法+置换检验后的候选）
    - industry_field / market_cap_field / log_mc: CSNeutralize 中性化参数
    - min_ic_after_neutralize: 中性化后 |IC| 下限
    - min_keep / min_keep_ratio: 显著因子过少时的保底（防因子池崩空）

    输出:
    - 中性化后仍显著因子名列表（按中性化 |IC| 降序；显著不足时返回原列表）
    """
    avail = [c for c in selected_names if c in x_sub.columns]
    if len(avail) < 2:
        return selected_names

    # CSNeutralize 需要 (fields_group, factor_name) 二级列结构
    x_neut_in = x_sub[avail].copy()
    x_neut_in.columns = pd.MultiIndex.from_tuples([("feature", c) for c in x_neut_in.columns])
    try:
        neutralized = CSNeutralize(
            fields_group="feature",
            industry_field=industry_field,
            market_cap_field=market_cap_field,
            log_mc=log_mc,
        ).__call__(x_neut_in)
    except Exception as e:
        print(f"    [中性化显著性] 中性化失败: {e}，保留原筛选结果")
        return selected_names
    x_neut = neutralized["feature"]

    ic_map: Dict[str, float] = {}
    for col in avail:
        feat = x_neut[col].dropna()
        lab = y_sub.reindex(feat.index).dropna()
        common = feat.index.intersection(lab.index)
        if len(common) < 50:
            continue
        try:
            ic_val = compute_ic(feat.loc[common], lab.loc[common])
        except Exception:
            continue
        if ic_val is None or (isinstance(ic_val, float) and np.isnan(ic_val)):
            continue
        ic_map[col] = abs(float(ic_val))

    keep = [c for c in avail if ic_map.get(c, 0.0) > min_ic_after_neutralize]
    keep.sort(key=lambda c: ic_map.get(c, 0.0), reverse=True)

    min_neut_keep = max(int(len(selected_names) * min_keep_ratio), min_keep)
    print(f"    [中性化显著性] 中性化后 |IC|>{min_ic_after_neutralize} 的因子 "
          f"{len(keep)}/{len(avail)} 个")
    if len(keep) < min_neut_keep:
        print(f"    [中性化显著性] 显著因子过少({len(keep)}<{min_neut_keep})，保留原筛选结果")
        return selected_names
    return keep


def run_single_category_selection(
    cat_name: str,
    factors: List[Dict],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fs_method: str,
    fs_algo: str,
    top_k: int,
    label_name: str,
    x_train_raw: Optional[pd.DataFrame] = None,  # [P2] 原始因子值（供中性化显著性用原始值）
    redundancy_check: bool = True,
    redundancy_threshold: float = REDUNDANCY_THRESHOLD,
    icir_stability: bool = True,
    icir_rolling_window: int = ICIR_WINDOW,
    icir_keep_ratio: float = ICIR_KEEP_RATIO,
    ic_history: Optional[List[Dict]] = None,   # [P2] 跨窗口 IC 历史（函数内 append 本窗口 IC）
    ic_coarse_conf: Optional[Dict] = None,     # [P2] IC 粗筛配置（stride/稳定 IC/比例）
    perm_conf: Optional[Dict] = None,          # [P2] 置换检验配置
    neut_conf: Optional[Dict] = None,          # [P2] 因子中性化显著性筛选配置（默认关）
    imp_conf: Optional[Dict] = None,           # [量价差异化] 重要性阈值筛选配置（embedded 阈值驱动）
) -> Optional[pd.DataFrame]:
    """
    在单个类别上运行特征选择。

    [性能优化] 不再创建 Qlib dataset，直接从预准备的 x_train/y_train 中按因子名切片。
    CSRankNorm 已在主流程中运行一次，此处直接复用结果。

    [P2-整合] 完整筛选链：IC 粗筛(逐日截面 IC + 跨窗口稳定 + stride[H=5]) → 嵌入法(重要性阈值驱动)
    → 置换检验(显著性) → 因子中性化显著性(可选，默认关) → 冗余检测 → ICIR 稳定校验。
    [G1-G4] IC 口径统一为逐日截面 IC；IC 粗筛对线性 IC 与非线性评分取并集，保护量价形态子族；
    置换/衰减按持有期 H=5 对齐重叠标签；信号持续性(换手)维度在因子体检中输出。

    输入:
    - cat_name: 类别名称
    - factors: 该类别因子列表 [{name, expression, meaning, source_file}, ...]
    - x_train: 已标准化 (CSRankNorm) 的特征矩阵
    - y_train: DK_L 标签 Series
    - x_train_raw: 原始因子值矩阵（未标准化；中性化显著性默认关，仅启用时使用）
    - fs_method, fs_algo, top_k: 特征选择参数
    - label_name: 标签列名（用于 ICIR 校验）
    - redundancy_check: 是否做冗余检测
    - redundancy_threshold: Spearman 相关系数阈值
    - icir_stability: 是否做 ICIR 稳定性校验
    - icir_rolling_window: ICIR 滚动窗口天数
    - icir_keep_ratio: ICIR 保留比例
    - ic_history: 跨窗口 IC 历史（list of dict，时间升序；本窗口 IC 会追加到列表尾）
    - ic_coarse_conf / perm_conf / neut_conf: [P2-整合] 新增筛选阶段配置
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

    # 从全局特征矩阵切片（含全部可用因子列，供 IC 粗筛使用）
    x_cat = x_train[available]
    y_cat = y_train

    # [P2-整合] IC 粗筛：跨窗口稳定 IC + stride（候选池收缩，剔除低/不稳定 IC 因子）
    ic_cfg = ic_coarse_conf or {}
    perm_cfg = perm_conf or {}
    neut_cfg = neut_conf or {}
    coarse_pool = available
    window_ic = {}
    if ic_cfg.get("enabled", True) and len(available) > ic_cfg.get("min_keep", 3):
        coarse_pool, window_ic = _ic_coarse_select(
            x_cat, y_cat, available,
            ic_history=ic_history,
            stride=ic_cfg.get("stride", 5),
            top_k_ratio=ic_cfg.get("top_k_ratio", 0.4),
            min_keep=ic_cfg.get("min_keep", 3),
            stable_ic=ic_cfg.get("stable_ic", True),
            nonlinear_ratio=ic_cfg.get("nonlinear_ratio", 0.3),
            nonlinear_bins=ic_cfg.get("nonlinear_bins", 10),
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
    # [P2 修复] 中性化显著性需在原始因子值上做 Ridge 中性化（而非 CSRankNorm 分位数），
    # 否则剥离的是"分位数上的行业/市值暴露"，口径弱于原始值中性化。
    x_raw_cat = x_train_raw[coarse_pool] if x_train_raw is not None else x_cat

    print(f"      >>> {x_cat.shape[0]} 行, {x_cat.shape[1]} 个特征")

    # [量价差异化] embedded 已改为阈值驱动（不再用 max_features 硬截断），
    # top_k 仅对 filter/wrapper 类方法生效（本脚本默认 embedded，不读取）。

    # 特征选择
    try:
        if fs_method == "embedded":
            # [量价差异化] 不传 max_features，返回全部候选的重要性排序，
            # 由下方 importance_filter 按相对阈值保留，数量成为输出而非预设。
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo, threshold=0.0,
                model_kwargs={"importance_type": "gain"},
                remove_collinearity=False,
                use_cache=False,  # [P2 性能] 全量 tobytes MD5 指纹(GB级内存复制)成本 > GPU 训练本身
            )
        elif fs_method == "filter":
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                k=min(top_k, len(coarse_pool)),
                remove_collinearity=False,
                use_cache=False,
            )
        else:
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                model_kwargs={"max_features": min(top_k, len(coarse_pool))},
                remove_collinearity=False,
                use_cache=False,
            )
    except Exception as e:
        print(f"    [错误] 特征选择失败: {e}")
        return None

    scores = fs_result.feature_scores
    if fs_method == "embedded":
        # [量价差异化] 重要性阈值驱动：按 importance 相对阈值保留，替代固定 top_k 截断
        imp_cfg = imp_conf or {}
        max_imp = float(scores.max()) if len(scores) else 0.0
        if imp_cfg.get("enabled", True) and max_imp > 0:
            thr = max_imp * imp_cfg.get("min_relative", 0.05)
            kept_imp = scores[scores >= thr].index.tolist()
            if len(kept_imp) < imp_cfg.get("min_keep", 3):
                kept_imp = scores.index.tolist()[:imp_cfg.get("min_keep", 3)]
            selected_factor_names = kept_imp
            selected_set = set(kept_imp)
            print(f"    [重要性阈值] 阈值={thr:.5f}(相对 {imp_cfg.get('min_relative', 0.05)})，"
                  f"保留 {len(kept_imp)}/{len(scores)} 个")
        else:
            selected_set = set(fs_result.selected_features)
            selected_factor_names = list(selected_set)
    else:
        selected_set = set(fs_result.selected_features)
        selected_factor_names = list(selected_set)

    # [P2-整合] 置换检验：剔除纯噪声因子（p >= 阈值，双尾）
    if perm_cfg.get("enabled", True) and len(selected_factor_names) >= 3:
        try:
            selected_factor_names = _permutation_significance(
                x_cat, y_cat, selected_factor_names,
                n_perms=perm_cfg.get("n_permutations", 200),
                pvalue_threshold=perm_cfg.get("pvalue_threshold", 0.05),
                min_keep=perm_cfg.get("min_keep", 3),
                fdr=perm_cfg.get("fdr", False),
                stride=perm_cfg.get("stride", 5),
            )
            selected_set = set(selected_factor_names)
        except Exception as e:
            print(f"    [置换检验] 异常: {e}，保留原筛选结果")

    # [P2-整合] 因子中性化显著性：剥离行业/市值后仍显著才保留（默认关，对齐 DK_L 口径）
    if neut_cfg.get("enabled", False) and len(selected_factor_names) >= 3:
        try:
            selected_factor_names = _neutralize_significance(
                x_raw_cat, y_cat, selected_factor_names,
                industry_field=neut_cfg.get("industry_field", "sw_l1"),
                market_cap_field=neut_cfg.get("market_cap_field", "circ_mv"),
                log_mc=neut_cfg.get("log_mc", True),
                min_ic_after_neutralize=neut_cfg.get("min_ic_after_neutralize", 0.005),
                min_keep=neut_cfg.get("min_keep", 2),
                min_keep_ratio=neut_cfg.get("min_keep_ratio", 0.3),
            )
            selected_set = set(selected_factor_names)
        except Exception as e:
            print(f"    [中性化显著性] 异常: {e}，保留原筛选结果")

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


# ==============================================================================
# 因子体检
# ==============================================================================
def _compute_ic_decay(
    feature_cache,
    selected_factors: List[str],
    instruments,
    start_time: str,
    end_time: str,
    horizons: List[int],
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
    log_mc: bool = True,
) -> pd.DataFrame:
    """[因子体检] IC 衰减诊断：计算入选因子在不同前瞻周期的【逐日截面 IC 均值】，
    输出衰减曲线（行=horizon，列=因子，值=IC）。[G3] 口径与 IC 粗筛/ICIR/评测端统一。

    用途：识别"短周期强、长周期迅速衰减"的因子，评估持有期匹配度
    （收敛于 train_tree 的 _compute_ic_decay）。不改变入选因子，仅输出诊断。

    输入:
    - feature_cache: 全局特征缓存
    - selected_factors: 最终入选因子名列表
    - instruments: 股票列表
    - start_time / end_time: 全期时间范围
    - horizons: 前瞻周期列表，如 [1, 3, 5, 10, 20]

    输出:
    - decay_df: DataFrame（index=H{h}，columns=因子名，values=IC）
    """
    if not selected_factors:
        print("    [IC 衰减] 无入选因子，跳过")
        return pd.DataFrame()

    factor_df = feature_cache.get_warehouse_df(
        selected_factors, start_time=start_time, end_time=end_time
    )
    if factor_df.empty:
        print("    [IC 衰减] 因子面板为空，跳过")
        return pd.DataFrame()
    if isinstance(factor_df.columns, pd.MultiIndex):
        factor_df.columns = factor_df.columns.droplevel(0)
    if isinstance(factor_df.index, pd.MultiIndex) and factor_df.index.names == ["instrument", "datetime"]:
        factor_df = factor_df.swaplevel().sort_index()

    decay: Dict[str, Dict[str, float]] = {}
    for horizon in horizons:
        # LABEL_EXPR 的前瞻周期泛化（horizon=5 时即 LABEL_EXPR 本身），
        # 保持"收盘对开盘"的收益口径一致，仅改变持有期长度。
        label_expr = f"Ref($close, -{horizon}) / Ref($open, -1) - 1"
        try:
            label_raw = D.features(instruments, [label_expr], start_time, end_time)
            if label_raw.empty:
                continue
            if isinstance(label_raw.columns, pd.MultiIndex):
                label_raw.columns = label_raw.columns.droplevel(1)
            label_raw = label_raw.rename(columns={label_raw.columns[0]: f"LABEL_H{horizon}"})
            # [P2 修复] 统一 DK_L 口径：行业+市值中性化后再算 IC，剔除市值/行业 beta，
            # 使体检 IC 可与筛选阶段(DK_L 标签)直接对比（见 _neutralize_label_series）。
            label_series = _neutralize_label_series(
                label_raw, f"LABEL_H{horizon}",
                industry_field=industry_field,
                market_cap_field=market_cap_field,
                log_mc=log_mc,
            ).sort_index()
            if (isinstance(label_series.index, pd.MultiIndex)
                    and label_series.index.names == ["instrument", "datetime"]):
                label_series = label_series.swaplevel().sort_index()

            # [G3] IC 口径统一为"逐日截面 Spearman IC 的时间均值"（与 IC 粗筛/ICIR/评测端一致），
            # 一次算出全因子截面 IC 再取均值，替代原逐因子 pooled Spearman。
            present = [c for c in selected_factors if c in factor_df.columns]
            ic_mean = _mean_daily_ic(factor_df[present], label_series) if present else pd.Series(dtype=float)
            row: Dict[str, float] = {}
            for col in selected_factors:
                if col in present:
                    v = ic_mean.get(col, np.nan)
                    row[col] = float(v) if pd.notna(v) else np.nan
                else:
                    row[col] = np.nan
            decay[f"H{horizon}"] = row
            print(f"    [IC 衰减] H{horizon}: mean|IC|={np.nanmean(np.abs(list(row.values()))):.4f}")
        except Exception as e:
            print(f"    [IC 衰减] H{horizon}: 计算失败 ({e})")

    return pd.DataFrame(decay).T if decay else pd.DataFrame()


def _compute_group_ic(
    factor_df: pd.DataFrame,
    label_series: pd.Series,
    instruments,
    selected_factors: List[str],
    analysis_type: str = "industry",
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
) -> pd.DataFrame:
    """[因子体检] 分组 IC 诊断：按行业(sw_l1)或市值(circ_mv)分组，计算各入选因子的
    组内 IC，识别对特定风格段依赖过强的因子（收敛于 train_tree 的 _compute_group_ic）。

    不改变入选因子，仅输出诊断。

    输入:
    - factor_df: (datetime, instrument) × 因子面板（原始值）
    - label_series: DK_L 口径标签 Series（行业+市值中性化，(datetime, instrument) 对齐）
    - instruments: 股票列表
    - selected_factors: 入选因子名列表
    - analysis_type: "industry" 或 "marketcap"
    - industry_field / market_cap_field: 分组字段名（与 CONFIG 对齐，从 Qlib 拉取 $字段）

    输出:
    - group_ic_df: DataFrame（index=因子名，columns=分组名，values=组内 IC）
    """
    if not selected_factors or factor_df.empty or label_series.empty:
        return pd.DataFrame()

    dates = sorted(factor_df.index.get_level_values("datetime").unique())
    if not dates:
        return pd.DataFrame()
    ref_date = str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1])[:10]

    group_field = f"${industry_field}" if analysis_type == "industry" else f"${market_cap_field}"
    try:
        group_raw = D.features(instruments, [group_field], ref_date, ref_date)
    except Exception as e:
        print(f"    [分组 IC] 拉取 {analysis_type} 分组字段失败: {e}")
        return pd.DataFrame()
    if group_raw.empty:
        return pd.DataFrame()
    if isinstance(group_raw.columns, pd.MultiIndex):
        group_raw.columns = group_raw.columns.droplevel(1)

    group_series = group_raw[group_raw.columns[0]]
    # 单日截面 → 归一为 (instrument) 级分组映射
    if isinstance(group_series.index, pd.MultiIndex):
        group_series = group_series.groupby(level="instrument").last()

    if analysis_type == "industry":
        group_map = group_series.to_dict()
    else:
        mkt_group = pd.qcut(group_series.rank(method="first"), 3,
                            labels=["小市值", "中市值", "大市值"])
        group_map = mkt_group.to_dict()

    inst_group = factor_df.index.get_level_values("instrument").map(group_map)
    result: Dict[str, Dict[str, float]] = {}
    for col in selected_factors:
        if col not in factor_df.columns:
            continue
        tmp = pd.DataFrame({
            "group": inst_group,
            "feat": factor_df[col].values,
            "label": label_series.reindex(factor_df.index).values,
        }, index=factor_df.index).dropna()
        if tmp.empty:
            continue
        ics: Dict[str, float] = {}
        for g, grp in tmp.groupby("group"):
            if len(grp) < 30:
                continue
            try:
                ics[str(g)] = compute_ic(grp["feat"], grp["label"])
            except Exception:
                continue
        if ics:
            result[col] = ics

    return pd.DataFrame(result).T if result else pd.DataFrame()


def _compute_signal_persistence(
    factor_df: pd.DataFrame,
    selected_factors: List[str],
    lag: int = 1,
) -> pd.DataFrame:
    """[G4] 信号持续性 / 换手诊断：因子截面秩的 lag 日自相关 + 换手代理。

    周频调仓下，量价因子的有效性依赖"信号能否在一周内保持稳定"：持续性越低，
    换手越高、越依赖交易成本容忍度。本函数给出该缺失维度，供成本感知的因子取舍。

    [定位] 仅作诊断维度，**不作为淘汰硬门槛**——有效性（IC）与成本（换手）是两个正交
    评价维度，把换手当有效性门槛会误杀"高 IC 但快衰减"的合法因子。

    输入:
    - factor_df: 因子面板（index=(datetime, instrument)，原始值）
    - selected_factors: 入选因子名列表
    - lag: 自相关滞后天数（默认 1 日）
    输出:
    - DataFrame（index=因子名，columns=[signal_autocorr, turnover_proxy]）
      signal_autocorr = 相邻日截面秩相关均值（∈[-1,1]，越大越稳）；
      turnover_proxy  = 相邻日截面秩的平均绝对变动（∈[0,1]，越小越省成本）。
    """
    present = [c for c in selected_factors if c in factor_df.columns]
    if not present:
        return pd.DataFrame()
    ranked = factor_df[present].groupby(level="datetime").rank(pct=True).sort_index()
    lagged = ranked.groupby(level="instrument").shift(lag)
    rows: Dict[str, Dict[str, float]] = {}
    for col in present:
        a, b = ranked[col], lagged[col]
        m = a.notna() & b.notna()
        if int(m.sum()) < 50:
            continue
        av = a[m].to_numpy(dtype=float)
        bv = b[m].to_numpy(dtype=float)
        if np.std(av) == 0 or np.std(bv) == 0:
            continue
        rows[col] = {
            "signal_autocorr": float(np.corrcoef(av, bv)[0, 1]),
            "turnover_proxy": float(np.mean(np.abs(av - bv))),
        }
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def _run_factor_health_check(
    feature_cache,
    selected_factors: List[str],
    instruments,
    start_time: str,
    end_time: str,
    horizons: List[int],
    analysis_types: List[str],
    industry_field: str = "sw_l1",
    market_cap_field: str = "circ_mv",
    log_mc: bool = True,
) -> pd.DataFrame:
    """[因子体检] 对最终入选因子做诊断（IC 衰减 + 行业/市值分组 IC）。

    基于 DK_L 口径标签（行业+市值中性化），不改变筛选结果，仅输出诊断报告。
    收敛于 train_tree 的 _compute_ic_decay / _compute_group_ic。

    [量价差异化] 返回 IC 衰减曲线 decay_df，供下游 decay_gate 硬门槛复用（避免重复计算）。

    输入:
    - feature_cache: 全局特征缓存（注意：调用方需确保尚未释放）
    - selected_factors: 最终入选因子名列表
    - instruments: 股票列表
    - start_time / end_time: 全期时间范围
    - horizons: IC 衰减的前瞻周期列表
    - analysis_types: 分组 IC 类型，如 ["industry", "marketcap"]
    - industry_field / market_cap_field / log_mc: 标签中性化参数
    """
    if not selected_factors:
        print("\n[因子体检] 无入选因子，跳过")
        return pd.DataFrame()

    print(f"\n{'=' * 60}\n  因子体检（IC 衰减 + 分组 IC）\n{'=' * 60}")
    print(f"  入选因子 {len(selected_factors)} 个, horizons={horizons}, 分组={analysis_types}")

    # 加载 5 日标签并统一 DK_L 口径（行业+市值中性化），供分组 IC 用
    # （标签表达式引用 pipeline_config.LABEL_EXPR 单一事实源，与筛选/训练端一致）
    label_expr = LABEL_EXPR
    label_series = pd.Series(dtype="float64")
    try:
        label_raw = D.features(instruments, [label_expr], start_time, end_time)
        if not label_raw.empty:
            if isinstance(label_raw.columns, pd.MultiIndex):
                label_raw.columns = label_raw.columns.droplevel(1)
            label_raw = label_raw.rename(columns={label_raw.columns[0]: LABEL_NAME})
            # [P2 修复] 统一 DK_L 口径：中性化后再算分组 IC，剔除市值/行业 beta
            label_series = _neutralize_label_series(
                label_raw, LABEL_NAME,
                industry_field=industry_field,
                market_cap_field=market_cap_field,
                log_mc=log_mc,
            ).sort_index()
            if (isinstance(label_series.index, pd.MultiIndex)
                    and label_series.index.names == ["instrument", "datetime"]):
                label_series = label_series.swaplevel().sort_index()
    except Exception as e:
        print(f"  [分组 IC] 标签加载失败: {e}，仅做 IC 衰减")

    # 1) IC 衰减
    decay_df = _compute_ic_decay(
        feature_cache, selected_factors, instruments, start_time, end_time, horizons,
        industry_field=industry_field,
        market_cap_field=market_cap_field,
        log_mc=log_mc,
    )
    if not decay_df.empty:
        print(f"\n  [IC 衰减曲线] (行=horizon, 列=因子, 值=IC)")
        print(decay_df.round(4).to_string())

    # 2) 因子面板（供分组 IC 与信号持续性诊断复用，避免重复拉取）
    factor_df = feature_cache.get_warehouse_df(
        selected_factors, start_time=start_time, end_time=end_time
    )
    if not factor_df.empty:
        if isinstance(factor_df.columns, pd.MultiIndex):
            factor_df.columns = factor_df.columns.droplevel(0)
        if (isinstance(factor_df.index, pd.MultiIndex)
                and factor_df.index.names == ["instrument", "datetime"]):
            factor_df = factor_df.swaplevel().sort_index()

    # 2a) 分组 IC（依赖标签）
    if not factor_df.empty and not label_series.empty:
        for atype in analysis_types:
            gic = _compute_group_ic(
                factor_df, label_series, instruments, selected_factors, atype,
                industry_field=industry_field,
                market_cap_field=market_cap_field,
            )
            if not gic.empty:
                print(f"\n  [{atype} 分组 IC] (行=因子, 列=分组, 值=组内 IC)")
                print(gic.round(4).to_string())

    # 2b) [G4] 信号持续性 / 换手诊断（仅需因子面板，独立于标签）
    if not factor_df.empty:
        persist = _compute_signal_persistence(factor_df, selected_factors)
        if not persist.empty:
            print("\n  [信号持续性/换手诊断] (signal_autocorr 越大越稳，turnover_proxy 越小越省成本)")
            print(persist.round(4).to_string())

    return decay_df


# ==============================================================================
# 量价差异：IC 衰减硬门槛 + 方向一致性门控
# ==============================================================================
def _apply_decay_gate(
    decay_df: pd.DataFrame,
    selected_factors: List[str],
    horizons: List[int],
    decay_floor: float = 0.30,
    sign_consistency_horizons: Optional[List[int]] = None,
    min_valid_horizons: int = 3,
    holding_horizon: int = 5,
    short_horizons: Optional[List[int]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """[量价差异化] IC 衰减硬门槛 + 方向一致性门控。

    对跨窗口聚合后最终入选的每个因子，基于 IC 衰减曲线做两道淘汰判定：
      1) 持有期崩塌：|IC_H{holding}| / max(|IC_short|) < decay_floor → 淘汰
         （[P0-3] 锚定实际持有期末，周频=H5，与 LABEL_EXPR 对齐。短周期参照默认 H1/H3。
          原实现用 H20 会误杀"持有期内有效、之后正常衰减"的量价因子）。
      2) 方向翻转：sign_consistency_horizons（默认 H1/H3/H5）的 IC 符号不一致 → 淘汰
         （动量-反转混叠，方向不稳定，量价因子的大忌）。

    仅当有效 horizon 数 >= min_valid_horizons 才做门控；数据不足时放行（不误杀）。

    输入:
    - decay_df: IC 衰减曲线（index=H{h}，columns=因子名，values=IC），来自 _compute_ic_decay
    - selected_factors: 待门控的入选因子名列表
    - horizons: 衰减前瞻周期（升序）
    - decay_floor: 持有期 |IC| 相对短周期峰值的保底比例
    - sign_consistency_horizons: 方向一致性判定用的 horizon 列表
    - min_valid_horizons: 至少有效 horizon 数（不足则放行）
    - holding_horizon: 持有期前瞻周期（周频=5，与标签 H5 对齐）
    - short_horizons: 持有期内早期参照周期（默认 [1, 3]）

    输出:
    - (保留因子名列表, {淘汰因子名: 淘汰原因})
    """
    if decay_df is None or decay_df.empty or not selected_factors:
        return selected_factors, {}

    if sign_consistency_horizons is None:
        sign_consistency_horizons = [1, 3, 5]
    if short_horizons is None:
        short_horizons = [1, 3]

    kept: List[str] = []
    dropped: Dict[str, str] = {}
    h_cols = [f"H{h}" for h in horizons]
    sign_cols = [f"H{h}" for h in sign_consistency_horizons]

    for factor in selected_factors:
        if factor not in decay_df.columns:
            kept.append(factor)  # 无衰减数据，放行
            continue
        row = decay_df[factor].reindex(h_cols).astype(float)
        valid = row.dropna()
        if len(valid) < min_valid_horizons:
            kept.append(factor)  # 有效 horizon 不足，放行
            continue

        # 方向一致性：H1/H3/H5 同号（0 视为无效符号）
        signs = []
        for hc in sign_cols:
            if hc in row.index and pd.notna(row[hc]) and row[hc] != 0:
                signs.append(1 if row[hc] > 0 else -1)
        if signs and len(set(signs)) > 1:
            dropped[factor] = "方向翻转(动量-反转混叠)"
            continue

        # [P0-3] 持有期崩塌：|IC_H{holding}| / max(|短周期 IC|) < decay_floor
        # 锚定持有期末（周频=H5），衡量"信号能否活到卖出时点"，而非超出持有期的 H20。
        short_vals = [abs(row.get(f"H{h}", np.nan)) for h in short_horizons
                      if pd.notna(row.get(f"H{h}", np.nan))]
        hold_val = abs(row.get(f"H{holding_horizon}", np.nan))
        if short_vals and pd.notna(hold_val):
            ratio = hold_val / max(short_vals)
            if ratio < decay_floor:
                dropped[factor] = (f"持有期崩塌(IC_H{holding_horizon}/短周期max={ratio:.2f}"
                                   f"<{decay_floor})")
                continue

        kept.append(factor)

    return kept, dropped


# ==============================================================================
# 结果输出与候选池回写
# ==============================================================================
def _write_pool_suggestions(final_df: pd.DataFrame) -> None:
    """[P1-6] 精选结果回写候选池（Alpha Book）建议区。

    精选因子作为"准入建议"写入 candidate_pool.json 的 suggestions 区，
    状态为 suggested，**不直接 admitted**——最终准入由
    run_eval.py --admit 三关检验（相关性/边际贡献/方向一致）唯一决定。

    输入：
    - final_df: 跨窗口聚合结果（含 factor_name / importance / n_windows_selected 等列）
    """
    from datetime import datetime
    pool_path = (Path(__file__).resolve().parents[2]
                 / "factor_data" / "registry" / "candidate_pool.json")
    if not pool_path.exists():
        print(f"[write_pool] 候选池不存在: {pool_path}，跳过回写（先运行 run_eval.py --admit 评测即入库）")
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
        print("           建议因子需经 run_eval.py --admit 三关检验后方可 admitted（训练因子池）")
    else:
        print("[write_pool] 无新增建议因子（均已存在）")


# ==============================================================================
# 主流程
# ==============================================================================
class _Tee:
    """标准输出分流器：同时写入控制台与日志文件，实现进度实时落盘。

    功能：包装原始 stdout，每次 write 后立即 flush 日志文件，做到"边跑边写"。
    入参：stream——原始 stdout；log_file——已打开的可写文件对象（utf-8）。
    边界：日志文件已关闭时 flush 静默跳过，避免退出阶段报错。
    注意：仅代理 write/flush/isatty/fileno，防止第三方库访问其它属性异常。
    """

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        if not self._log_file.closed:
            self._log_file.write(data)
            self._log_file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        if not self._log_file.closed:
            self._log_file.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()


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

    # [归档 2026-10-08] 因子库已按 5+1 家族整理，YAML 内 category 即 Barra 大类终态，
    # 无需再归并；load_factors_by_category 直接按 category 分组，同名因子天然去重。

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
            x_train, y_train, x_train_raw = _prepare_window_data(
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
                # [P0-2 修复] 因子数不足最小评估规模时，**不做无验证的"整类直通"**：
                # 未过任何有效性标准的因子一律不判为有效（selected=False），仅登记到
                # 结果明细供审计，避免其被误写入候选池（有效性标准须由门控链证明）。
                print(f"\n  [{cat_index}/{len(categories)}] 类别 '{cat_name}' ({len(factors)} 个因子) — 因子数 < {CONFIG['min_factors']}，未评估（不判为有效）")
                rows = []
                for f in factors:
                    rows.append({
                        "category": cat_name, "factor_name": f["name"],
                        "selected": False, "importance": 0.0, "rank": 0,
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
                x_train_raw=x_train_raw,
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
                neut_conf=CONFIG.get("factor_neutralization", {}),
                imp_conf=CONFIG.get("importance_filter", {}),
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

    # [移位] 原此处释放全局缓存；全局跨类冗余兜底需重载入选因子面板，延后至聚合后释放

    # ─────────────────────────────────────────────────────────────────────────
    # 跨窗口聚合：因子在 >= min_window_ratio 的窗口中入选才最终选中
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

    # ─────────────────────────────────────────────────────────────────────────
    # [P0 修复] 全局跨类冗余兜底：类内 check_redundancy 只覆盖本类因子列，
    # 跨类冗余（circ_mv vs total_mv_log、动量族分散 4 类等）无人管控。
    # 对跨窗口聚合后的入选因子重载全期面板，做一次全局 Spearman 去重。
    # rank = 跨窗口入选次数 × 10 + 平均重要性（稳定性优先、强度次之）。
    # ─────────────────────────────────────────────────────────────────────────
    if CONFIG.get("global_redundancy_check", True) and final_df["selected"].sum() > 2:
        _sel_before = int(final_df["selected"].sum())
        print(f"\n[全局冗余] 对 {_sel_before} 个入选因子做跨类 Spearman 去重 "
              f"(threshold={CONFIG['global_redundancy_threshold']})...")
        dropped: List[str] = []
        try:
            sel_names = final_df.loc[final_df["selected"], "factor_name"].tolist()
            global_df = global_feature_cache.get_warehouse_df(
                selected_names=sel_names,
                start_time=full_start,
                end_time=full_end,
            )
            if not global_df.empty:
                if isinstance(global_df.columns, pd.MultiIndex):
                    global_df.columns = global_df.columns.droplevel(0)
                if (isinstance(global_df.index, pd.MultiIndex)
                        and global_df.index.names == ["instrument", "datetime"]):
                    global_df = global_df.swaplevel().sort_index()
                # [P1-5 口径统一] 与类内 check_redundancy 一致：面板先逐日 CSRankNorm 再算
                # Spearman。原始值 pooled 秩相关会混入时间趋势成分，可能误删截面各异但同向漂移
                # 的因子；逐日截面秩化后，pooled 相关近似"平均截面相关"，口径与类内一致。
                global_df = _apply_cs_rank_norm(global_df)
                _rank = {
                    row["factor_name"]: row["n_windows_selected"] * 10 + row["importance"]
                    for _, row in final_df[final_df["selected"]].iterrows()
                }
                kept_global = check_redundancy(
                    global_df, sel_names,
                    threshold=CONFIG["global_redundancy_threshold"],
                    rank=_rank,
                )
                dropped = [f for f in sel_names if f not in kept_global]
                if dropped:
                    print(f"    [全局冗余] 剔除 {len(dropped)} 个跨类冗余因子:")
                    for f in dropped:
                        print(f"      - {f}")
                else:
                    print("    [全局冗余] 无跨类冗余因子")
            else:
                print("    [全局冗余] 因子面板加载为空，跳过（保留原结果）")
        except Exception as e:
            print(f"    [全局冗余] 异常: {e}，跳过（保留原结果）")
        final_df["global_redundant"] = final_df["factor_name"].isin(dropped)
        if dropped:
            final_df.loc[final_df["global_redundant"], "selected"] = False
    else:
        final_df["global_redundant"] = False

    # [P2-整合] 因子体检：对最终入选因子做 IC 衰减 + 行业/市值分组 IC 诊断
    # （基于 DK_L 口径标签，不改变筛选结果，仅在释放全局缓存前输出诊断报告）
    hc_cfg = CONFIG.get("factor_health_check", {})
    decay_cfg = CONFIG.get("decay_gate", {})
    sel_final = final_df.loc[final_df["selected"], "factor_name"].tolist()
    decay_df = pd.DataFrame()
    if hc_cfg.get("enabled", True):
        decay_df = _run_factor_health_check(
            feature_cache=global_feature_cache,
            selected_factors=sel_final,
            # [修复] D.features 只接受 list/dict/ndarray，不接受市场名字符串；
            # 传 resolved_instruments（已解析股票列表），与标签加载/建缓存口径一致
            instruments=global_feature_cache.resolved_instruments,
            start_time=full_start,
            end_time=full_end,
            horizons=hc_cfg.get("horizons", [1, 3, 5, 10, 20]),
            analysis_types=hc_cfg.get("analysis_types", ["industry", "marketcap"]),
            industry_field=CONFIG.get("industry_field", "sw_l1"),
            market_cap_field=CONFIG.get("market_cap_field", "circ_mv"),
            log_mc=CONFIG.get("log_mc", True),
        )

    # [量价差异化] IC 衰减硬门槛 + 方向一致性门控：将不合格因子从最终结果中淘汰
    if decay_cfg.get("enabled", True) and sel_final:
        if decay_df.empty:
            # 体检关闭时单独计算衰减曲线（仅在需要门控时兜底，避免重复拉取）
            decay_df = _compute_ic_decay(
                global_feature_cache, sel_final,
                global_feature_cache.resolved_instruments,
                full_start, full_end, decay_cfg.get("horizons", [1, 3, 5, 10, 20]),
                industry_field=CONFIG.get("industry_field", "sw_l1"),
                market_cap_field=CONFIG.get("market_cap_field", "circ_mv"),
                log_mc=CONFIG.get("log_mc", True),
            )
        _, dropped_gate = _apply_decay_gate(
            decay_df, sel_final,
            horizons=decay_cfg.get("horizons", [1, 3, 5, 10, 20]),
            decay_floor=decay_cfg.get("decay_floor", 0.30),
            sign_consistency_horizons=decay_cfg.get("sign_consistency_horizons", [1, 3, 5]),
            min_valid_horizons=decay_cfg.get("min_valid_horizons", 3),
            holding_horizon=decay_cfg.get("holding_horizon", 5),
            short_horizons=decay_cfg.get("short_horizons", [1, 3]),
        )
        final_df["decay_gate_failed"] = final_df["factor_name"].isin(dropped_gate)
        if dropped_gate:
            print(f"\n[量价门控] IC 衰减/方向门控淘汰 {len(dropped_gate)} 个因子:")
            for f, reason in dropped_gate.items():
                print(f"      - {f}: {reason}")
            final_df.loc[final_df["decay_gate_failed"], "selected"] = False
        else:
            print("\n[量价门控] 所有入选因子通过 IC 衰减/方向门控")
    else:
        final_df["decay_gate_failed"] = False

    # 释放全局缓存（[移位] 原在聚合前释放；全局冗余兜底需重载入选因子面板）
    try:
        del global_feature_cache
    except NameError:
        pass
    gc.collect()

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
    # [日志] 运行时同步生成日志文件：stdout 分流写控制台与日志，进度动态落盘。
    # 日志文件名与本次运行时间戳绑定，落盘目录与 CSV 输出一致（脚本同目录）。
    _run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = os.path.join(os.path.dirname(__file__), f"select_momentum_price_{_run_ts}.log")
    _log_file = open(_log_path, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_file)
    try:
        print(f"[日志] 运行日志同步写入: {_log_path}")
        main()
    finally:
        # 无论正常结束还是异常/ sys.exit，都恢复 stdout 并关闭日志文件
        sys.stdout = _orig_stdout
        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass
