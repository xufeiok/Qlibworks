"""
factor_screen.py — 统一因子筛选引擎（一套脚本 + 家族配置驱动）· 唯一评测脚本

[定位]
  融合本项目既有脚本优势，替代分散筛选流程的唯一入口，三大模块：
    模块 A（家族筛选）    两阶段防 OOM / BH 多重检验 / 分年度 IC 报告 / 日志 Tee
                          + 家族差异化质检链（FAMILY_DEFS 单一事实源驱动）
    模块 B（深评 --deep-eval）  单因子完整评测（分层回测/衰减/稳健性/walk-forward 估计-外推）
    模块 C（准入 --admit/--admit-only）  三关准入 → 候选池 v2（factor_stats 唯一口径）
  差异质检完全由 FAMILY_DEFS 家族配置驱动：不同家族使用不同 IC/ICIR 门槛、
  不同特殊检验项开关，杜绝"全因子混洗统一阈值"。

[家族配置]（对齐《量化因子家族管理规范》第四章，头部机构差异化质检）
  家族          IC 门槛   ICIR 门槛   特殊检验项                    配额参考(50池)
  quality       0.015     0.30       跨牛熊同号/熊市保护/52周同号/中性化强制  16-18 (32-36%)
  value         0.025     0.50       描述量一致性/跨牛熊放宽/中性化强制       10-12 (20-24%)
  momentum_price 0.02     0.40       FDR置换/IC衰减门/冗余0.50/t≥3.0        14-16 (28-32%)
  sentiment     0.012     0.40       跳过ICIR稳定/52周同号/覆盖率放宽         4-6   (8-12%)
  risk          不强制    不强制     只查风险区分度(不参与Alpha比拼)          2-4   (4-8%)
  硬性红线：量价家族(含 rolling/形态子类)占比严禁超过 35%

[股票池]
  默认 INSTRUMENTS = main_board（沪深主板：沪 600/601/603/605，深 000/001/002/003），
  由 pipeline_config 单一事实源统一，满足"仅交易沪深主板"约束。

[用法]
  # 模块 A：家族筛选
  python factor_screen.py --family momentum_price     # 仅量价家族
  python factor_screen.py --all                       # 五大家族全部
  python factor_screen.py --all --period-windows 2023,2024,2025   # 多周期分段稳健性检验
  # 模块 B/C：单因子评测（深评 + 准入）
  python factor_screen.py --deep-eval --factor KDJ_K   # 仅深评
  python factor_screen.py --deep-eval --admit --factor KDJ_K   # 深评即准入（标准主路径）
  python factor_screen.py --admit-only --factor KDJ_K           # 复用 registry 旧 IC 补准入
  python factor_screen.py --factor-list <A模块名单.json/csv> --deep-eval --admit  # 全链一条命令

[输出]
  模块 A：factor_data/reports/screening/ 下
    family_screen_card_{ts}.csv     全因子粗筛卡（含 family / 各门判定 / 分年度 IC）
    family_candidates_{ts}.json     候选清单（按家族分组）
  模块 B：factor_data/reports/{tier}/（深评报告 html + _summary.csv）、
          factor_data/qualified_factors/{tier}/{因子名}.ref.json
  模块 C：factor_data/registry/candidate_pool.json（候选池 v2）
  运行日志（每次运行自动生成，实时记录进度，问题核查以此为准）：
    runtime/logs/factor_screen_{YYYYmmdd_HHMMSS}.log   历史日志（每次运行独立，不覆盖）
    runtime/logs/factor_screen_latest.log              最新一次运行（实时同步，快速核查）
"""

import os
import sys
import gc
import json
import time
import logging
import warnings
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

# Conda site-packages 优先，避免 Roaming 覆盖
sp = list(sys.path)
conda_sp = [p for p in sp if 'Anaconda' in p and 'site-packages' in p]
roaming_sp = [p for p in sp if 'Roaming' in p]
other_sp = [p for p in sp if p not in conda_sp and p not in roaming_sp]
sys.path = conda_sp + other_sp + roaming_sp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qlib
from qlib.data import D

from qlworks.evaluation.selector import (
    compute_factor_ics, compute_daily_ic_frame, select_top_by_abs,
    check_redundancy, check_icir_stability,
)
from qlworks.features.dataset import build_custom_feature_cache
from qlworks.config import QLIB_DATA_DIR
from qlworks.pipeline_config import (
    LABEL_EXPR, LABEL_NAME, INSTRUMENTS, START_TIME, END_TIME,
    ICIR_WINDOW, FILTER_ST, FILTER_NEW_STOCKS,
    ADMISSION_CUTOFF_DATE, ADMISSION_WINDOW_YEARS,
    FILTER_LIMIT_UPDOWN, FILTER_SUSPENDED,
)
# [单脚本化 2026-10-12] 深评 + 准入（模块 B/C）——唯一评测脚本 = 本文件。
# factor_stats.py：唯一统计口径（IC/ICIR-NW/相关矩阵/三关/候选池 v2 读写）。
# deep_eval.py：深评编排核心（仓库优先 → 数据质量 → FactorEvaluator 评测）。
from qlworks.evaluation.factor_stats import (
    run_admit_decision as _run_admit_decision,
    load_candidate_pool as _load_candidate_pool,
    load_label_frame as _fs_load_label_frame,
    normalize_factor_family as _normalize_factor_family,
)
from qlworks.evaluation import deep_eval as _deep_eval
from qlworks.evaluation import FactorStore as _FactorStore
from qlworks.evaluation import DEFAULT_CONFIG as _DEFAULT_CONFIG
from qlworks.evaluation.runner import FactorEvaluator as _FactorEvaluator
# 复用既有实现（同一目录，模块级代码无副作用）
# [P1 归一化 2026-10-13] 依赖已迁入 src/qlworks/evaluation/_legacy/（原 training 脚本已归档）
from qlworks.evaluation._legacy.select_factors import (
    _permutation_significance, build_global_bundle, load_factors_by_category,
)
from qlworks.evaluation._legacy.screen_factors import (
    _load_label_series, _build_tradeable_mask, _apply_tradeable_mask, _apply_dk_label,
    _load_factor_batch, _annual_ic_report, _print_annual_report, _Tee,
)
# [量价移植 2026-10-11] 复用 select_momentum_price.py 的动量家族专属检验
# （BH-FDR 置换 / IC 衰减硬门槛 / 方向一致性门控）。select_momentum_price.py 已冻结
# 为参考实现，其差异化检验统一由本引擎 FAMILY_DEFS["momentum_price"]["special"] 配置驱动。
from qlworks.evaluation._legacy.select_momentum_price import (
    _permutation_significance as _mp_permutation_significance,  # fdr+stride 版本
    _compute_ic_decay as _mp_ic_decay,
    _apply_decay_gate as _mp_decay_gate,
    _compute_signal_persistence as _mp_signal_persistence,
)

# ==============================================================================
# [家族配置] 单一事实源 —— 差异化质检的唯一入口（对齐规范第四章）
# ==============================================================================
FAMILY_DEFS = {
    "quality": {
        "label": "Quality 质量基本面",
        "ic_min": 0.015,          # 周频 IC 均值门槛（允许偏低但求稳）
        "icir_min": 0.30,         # 周频 ICIR 门槛（[P0-核实] 0.35→0.30：慢变量口径，QMJ 美国 Sharpe 仅 0.40）
        "win_rate_min": 0.52,     # 正 IC 占比（稳定性代理）
        "special": {
            "cross_regime": True,   # 跨牛熊分段：分年度 IC 同号占比 >= regime_min_same_ratio
            "crowding": True,       # [P0-核实] False→True：QMJ Table VIII 质量价格负向预测未来 QMJ 收益
            "risk_rank": False,
            "min_coverage": 0.75,   # [P0-核实] 0.85→0.75：财报披露后 1-2 月覆盖率跌破 0.85（慢变量）
            "regime_min_same_ratio": 0.65,  # [P0-核实] 0.60→0.65：QMJ 22/24 国=92%、3/3 子区间=100%
            "factor_neutralize": True,      # [P0-核实] 开行业+市值中性：MSCI sector-relative z-score
            "min_ic_after_neutralize": 0.01,# 中性化后 |IC| 门槛（对齐 MSCI 行业内 z-score 口径）
            "permutation_enabled": True,    # [P0-核实] 引擎默认置换检验（200 次，HLZ 多重检验口径）
            "t_min": 2.5,                   # [P1-1] 家族级 |t| 软门槛
            "rolling_ic_window_weeks": 52,  # [P1-2a] 52 周滚动 IC 正占比（慢变量持续性）
            "rolling_positive_ratio": 0.60, #   rolling 正周占比下限
            "bear_regime_protection": True, # [P1-2b] 熊市保护：熊市 regime 子样本 IC 不得显著为负
            "bear_ic_min_t": -1.5,
            "bear_index_code": "000300.SH", # CH 宽基指数（2002 起）；机构"12月收益<−25%"口径 A 股适配
            "bear_cum_return_52w": -0.15,   # 沪深300 52 周累计收益 < -15% 计为熊市 regime
            "bear_5d_ret_threshold": -0.01, # 回退口径：5日窗标签等权月均值 < -1%（≈月跌5-6%）
        },
        "quota": (16, 18),        # 配额参考（50 池）
        "note": "信号慢噪声低、半衰期6-12月、极端行情稳健",
    },
    "value": {
        "label": "Value 价值估值",
        "ic_min": 0.025,          # [P0-核实] 0.015→0.025：对齐中证月度 RankIC 0.059-0.072 的周频折算
        "icir_min": 0.50,         # [P0-核实] 0.35→0.50：AQR 单资产 Sharpe 0.29-0.57 对应 ICIR 量级
        "win_rate_min": 0.53,     # [P0-核实] 0.52→0.53：对应中证月度胜率 61-65%
        "special": {
            "cross_regime": True,
            "crowding": False,
            "risk_rank": False,
            "regime_min_same_ratio": 0.50,  # [P0-核实] 0.60→0.50：价值有 12 年 39% 回撤，4 年负 IC 是正常 regime
            "factor_neutralize": True,      # [P0-核实] 强制开行业+市值中性：Barra CNE5 32 行业 WLS 口径
            "min_ic_after_neutralize": 0.01,# 中性化后 |IC| 门槛（防银行/公用事业行业暴露主导裸 IC）
            "permutation_enabled": True,    # [P0-核实] 引擎默认置换检验（200 次，HLZ 多重检验口径）
            "t_min": 2.5,                   # [P1-1] 家族级 |t| 软门槛
            "descriptor_consistency": {     # [P1-5] 多描述量一致性（家族资格检验）
                "min_pass": 2,
                "min_ic_each": 0.010,
                "direction_map": {          # 描述量方向归一：价值低→收益高
                    "pb_lf": -1,            # BTOP = 1/pb
                    "pe_ttm": -1,           # ETOP = 1/pe
                    "ps_ttm": -1,           # S/P  = 1/ps
                    "dv_ttm": 1,            # D/P  = 股息率
                },
            },
        },
        "quota": (10, 12),
        "note": "均值回归、需控行业市值暴露（风格漂移另由组合层约束）",
    },
    "momentum_price": {
        "label": "Momentum&Price 量价趋势",
        "ic_min": 0.020,          # 量价信号快，门槛最高
        "icir_min": 0.40,
        "win_rate_min": 0.54,
        "special": {
            "cross_regime": False,   # 量价半衰期短，分年度一致性要求放宽
            "crowding": True,        # 额外拥挤度检测（同家族平均相关上限）
            "risk_rank": False,
            # [量价移植 2026-10-11] 以下来自 select_momentum_price.py 的差异化检验，
            # 由引擎主流程按本配置挂载；其他家族无此键，不受影响。
            "permutation_fdr": {      # 置换检验升级：BH-FDR + stride 去标签重叠
                "enabled": True,
                "n_permutations": 200,
                "pvalue_threshold": 0.05,   # fdr=True 时语义为 FDR 水平 q
                "stride": 5,                # 隔持有期采样，令标签窗口互不重叠
                "min_keep": 3,
            },
            "decay_gate": {           # IC 衰减硬门槛 + 方向一致性门控
                "enabled": True,
                "horizons": [1, 3, 5, 10, 20],
                "holding_horizon": 5,        # 周频持有期，锚定 H5
                "decay_floor": 0.30,         # |IC_H5| / max(|IC_H1,H3|) 保底比例
                "sign_consistency_horizons": [1, 3, 5],  # 同号门控
            },
            "redundancy_inner": 0.50,  # [P3] 量价族内冗余 0.85→0.50（报告 P0：WW 平均成对相关 15.9%，0.7 仅拒收红线）
            "redundancy_global": 0.65, # 量价全局跨类收紧（vs 引擎默认 0.70）
            "representative_max": 16,  # [P3] 粗筛后贪心聚类代表上限（报告 P0：215 同族每簇留代表 ≤16）
            "t_min": 3.0,              # [P1-1] 家族级 |t| 硬门槛（HLZ 2016：新因子 |t|>3.0）
        },
        "quota": (14, 16),
        "note": "信号快、半衰期1-4周、拥挤度高、需高频监控；占比红线≤35%",
    },
    "sentiment": {
        "label": "Sentiment 预期情绪",
        "ic_min": 0.012,          # [P0-核实] 0.018→0.012：文本/修正类 IC 普遍 0.01-0.03（Tetlock/LM/CJL）
        "icir_min": 0.40,         # [P0-核实] 0.38→0.40：用稳定性补偿低 IC
        "win_rate_min": 0.52,     # [P0-核实] 0.53→0.52
        "special": {
            "cross_regime": False,  # [P0-核实] True→False：SYU 2012 证明情绪利润 70-80% 集中高情绪期，
                                    # 要求年度同号>=0.60 与理论冲突（即家族=0 的主因）；
                                    # P1/P2 替换为"高/低情绪期 IC 差"检验（需情绪状态指标数据前置）
            "crowding": False,
            "risk_rank": False,
            "min_coverage": 0.55,        # [P1] 0.70→0.55：两融标的子池 ~61%、龙虎榜/北向另类因子天然低覆盖
            "crowding_max_avg_corr": 0.65,  # [P0-核实] 0.55→0.65：同源情绪因子天然相关 0.5-0.7
            "t_min": 2.5,                # [P1-1] 家族级 |t| 软门槛
            "permutation_enabled": True, # [P3] 置换检验全家族默认开（McLean-Pontiff 随机基线口径）
            "permutation_n_perms": 200,
            "skip_icir_stable": True,    # [P1] 事件驱动：滚动ICIR稳定<80%是脉冲IC本质，跳过（用 rolling 同号占比替代）
            "rolling_ic_window_weeks": 52,  # [P1-2a] 52 周滚动 IC 同号占比（情绪因子持续性）
            "rolling_positive_ratio": 0.60,
        },
        "quota": (4, 6),
        "note": "事件驱动、财报季效应显著；当前 4 因子（资金流/两融/换手/北向），待 P2-1 sentiment_split 高低情绪期 IC 差检验",
    },
    "risk": {
        "label": "Risk 风险流动性",
        "ic_min": None,           # 不做 IC 强制要求
        "icir_min": None,
        "win_rate_min": None,
        "special": {
            "cross_regime": False,
            "crowding": False,
            "risk_rank": True,      # 只校验风险区分度/稳定性
            # [P1-3] 风险因子三钩子（报告《家族检验标准核实报告》§2.4）
            "risk_layer_separation": {   # 分层分离度：五分层标签均值单调 + KW 显著
                "min_mono_corr": 0.5,
                "kruskal_wallis_p": 0.05,
            },
            "risk_rolling_stability": {  # rolling 稳定性：相邻周截面秩相关 52 周均值
                "window_weeks": 52,
                "cross_section_rank_corr_min": 0.70,
            },
            "risk_direction_check": {    # 方向一致性：因子级期望方向，错配告警（人工复核）
                "direction_map": {
                    "volatility_1m": "neg",    # 低波动异象：高波动→低收益
                    "beta": "neg",
                    "pledge_ratio": "neg",     # 高质押→暴雷风险
                    "shareholder_change": "neg",  # 股东数减少=筹码集中（正向信号则反转）
                    "illiquidity_amihud": "pos",  # Amihud 非流动性溢价
                    "circ_mv": "neg",          # 小市值效应
                    "total_mv_log": "neg",
                    "turnover_rate_1m": "neg", # 高换手折价
                    # volume_surge 方向不定，不标注
                },
            },
        },
        "quota": (2, 4),
        "note": "仅作风控约束，不参与 Alpha 比拼",
    },
}

# 跨家族去冗余保留优先级（规范 5.2：基本面 > 预期 > 量价 > 风险）
FAMILY_PRIORITY = ["quality", "value", "sentiment", "momentum_price", "risk"]

# 家族 → 因子库 yaml 文件名（与 factor_library 归档一致）
FAMILY_FILE = {fam: f"{fam}_factors.yaml" for fam in FAMILY_DEFS}
FAMILY_FILE["other"] = "other_factors.yaml"

# ==============================================================================
# [全局配置]
# ==============================================================================
CONFIG = {
    # 股票池：沪深主板（用户可交易范围），pipeline_config 单一事实源
    "instruments": INSTRUMENTS,
    "start_time": START_TIME,
    "end_time": END_TIME,
    "label_expr": LABEL_EXPR,
    "label_name": LABEL_NAME,
    # 阶段一：分批 IC 粗筛（防 OOM）
    "batch_size": 20,
    "top_k": 60,               # 每家族进入阶段二的候选数上限
    "min_samples": 50,
    # 阶段二：通用基础门槛（规范 4.1）
    "min_coverage": 0.85,      # 截面缺失率 < 15% → 覆盖率 >= 85%
    "min_nunique": 50,
    "correction_alpha": 0.05,  # BH 多重检验
    "correction_method": "fdr_bh",  # [P3] 原"bh"无效（statsmodels 仅认 fdr_bh），致 BH 长期走 except 假全过
    "icir_window": ICIR_WINDOW,
    "icir_keep_ratio": 0.8,      # [P3] 滚动 ICIR 稳定性 keep_ratio（支持家族 special 覆盖）
    # 冗余（规范 5：家族内 |r|>=0.7 优先；全局兜底同阈值）
    "redundancy_threshold": 0.70,
    "global_redundancy_threshold": 0.70,
    # 特殊检验参数
    "crowding_max_avg_corr": 0.55,   # 拥挤度：同家族平均 |corr| 上限
    "regime_min_same_ratio": 0.60,   # 跨牛熊：分年度 IC 同号占比下限
    # 可选检验（默认关闭，成本高或依赖额外数据）
    "permutation_test": {"enabled": False, "n_permutations": 200, "pvalue_threshold": 0.05},
    "monotonicity": {"enabled": False, "n_layers": 10, "min_mono_corr": 0.7},
    # 多周期分段稳健性检验（对齐 select_factors：窗口独立筛选 → 跨窗口聚合）
    # 语义：并行分片重复检验（sub-period robustness），非时序滚动外推。
    # 默认单窗口全时段；--period-windows 显式开启（旧名 --rolling-windows 兼容）
    "rolling_windows": None,          # 如 ["2023", "2024", "2025"] 或 ["2023-01-01/2023-12-31", ...]
    "min_window_ratio": 0.5,          # 因子在 >= 该比例窗口中通过才最终入选
    # 树模型嵌入法（对齐 select_factors/train_tree：LGBM importance 组合视角精选）
    "embedded_selection": {"enabled": False, "importance_type": "gain",
                           "n_estimators": 200, "min_keep": 3},
    # 因子中性化（对齐 train_tree-doubao：Ridge 剥离行业+市值后重算 IC）
    "factor_neutralization": {"enabled": False, "min_ic_after_neutralize": 0.005,
                              "min_keep_ratio": 0.3},
    # 可交易性过滤（全链路统一开关）
    "filter_st": FILTER_ST,
    "filter_new_stocks": FILTER_NEW_STOCKS,
    # [P3] 滚动窗口模式边界治理：purge=标签重叠窗剔除 / embargo=窗口起点冷却
    "wf_purge_days": 5,
    "wf_embargo_days": 10,
    # 输出（[P3] 迁移到 factor_data/reports/screening，与代码目录隔离）
    "output_dir": str(PROJECT_ROOT / "factor_data" / "reports" / "screening"),
}


# ==============================================================================
# [家族因子加载] 直接按 family 读 factor_library yaml（不经过 category 分组）
# ==============================================================================

def load_family_factors(family: str) -> list[dict]:
    """加载指定家族的因子定义列表。

    返回: [{name, expression(qlib), category, source_file, meaning}, ...]
    """
    import yaml

    fname = FAMILY_FILE.get(family)
    if not fname:
        return []
    path = Path(__file__).resolve().parents[2] / "factor_data" / "factor_library" / fname
    if not path.exists():
        print(f"  [警告] 家族文件不存在: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = []
    for factor in data.get("factors") or []:
        name = factor.get("name")
        if not name:
            continue
        expr_raw = factor.get("expression", {})
        qlib_expr = expr_raw.get("qlib", "") if isinstance(expr_raw, dict) else str(expr_raw)
        if not qlib_expr:
            continue
        out.append({
            "name": name,
            "expression": qlib_expr,
            "category": factor.get("category", ""),
            "source_file": fname,
            "meaning": factor.get("meaning", ""),
        })
    return out


# ==============================================================================
# [阶段一] 家族内分批 IC 粗筛（按家族 IC 门槛）
# ==============================================================================

def _family_batch_ic(cache, factor_names, name_expr_map, label_series,
                     start_time, end_time, cfg, tradeable_mask, family):
    """分批计算家族内全部因子的 Spearman IC，按 |IC| 降序取 top_k。

    对 Risk 家族不做 IC 门槛（ic_min=None），但仍计算 IC 供报告。
    """
    all_ic: dict = {}
    n = len(factor_names)
    for i in range(0, n, cfg["batch_size"]):
        batch_names = factor_names[i:i + cfg["batch_size"]]
        batch_df = _load_factor_batch(cache, name_expr_map, batch_names, start_time, end_time)
        if batch_df.empty:
            continue
        batch_df = _apply_tradeable_mask(batch_df, tradeable_mask)
        if batch_df.empty:
            continue
        common = batch_df.index.intersection(label_series.index)
        if len(common) < 100:
            continue
        ics = compute_factor_ics(batch_df.loc[common], label_series.loc[common],
                                 min_samples=cfg["min_samples"])
        all_ic.update(ics.to_dict())
        del batch_df
        gc.collect()

    ic_series = pd.Series(all_ic, dtype=float)
    if ic_series.empty:
        return ic_series, []
    # 有 IC 门槛的家族：先用 |IC| 门槛预筛（减半放宽，最终由阶段二判定）
    ic_min = FAMILY_DEFS.get(family, {}).get("ic_min")
    ranked = ic_series.reindex(sorted(ic_series.index, key=lambda x: abs(ic_series[x]), reverse=True))
    if ic_min is not None:
        ranked = ranked[ranked.abs() >= ic_min * 0.5]  # 门槛减半预筛，仅收敛候选规模
    selected = ranked.index[:cfg["top_k"]].tolist() if len(ranked) else []
    return ic_series, selected


# ==============================================================================
# [阶段二] 家族内 5 道门 + 家族特殊检验
# ==============================================================================

def _run_family_gates(factor_frame, labels, all_factor_names, cfg, family, daily_ic=None):
    """对家族因子执行完整质检链，返回逐因子判定 DataFrame。

    门顺序（家族化）：
      ① 数据质量门（覆盖率/常数）
      ② IC/ICIR 门槛（家族差异化）
      ③ ICIR 滚动稳定（家族窗口一致）
      ④ 家族内冗余剔除（|r| >= 0.7，保留 IC 强者）
      ⑤ BH 多重检验（n_tested = 全库因子数，避免低估选择偏差）
      + 家族特殊检验（拥挤度 / 跨牛熊分段 / 风险区分度）
    """
    fd = FAMILY_DEFS.get(family, {})
    spec = fd.get("special", {})   # [量价移植] 提前提取，供冗余阈值/特殊检验共用
    ic_min = fd.get("ic_min")
    icir_min = fd.get("icir_min")
    win_rate_min = fd.get("win_rate_min")

    card = pd.DataFrame(index=factor_frame.columns)
    card.index.name = "factor_name"

    # ① 质量门：覆盖率 & 非全常数
    # [P0-核实] 覆盖率门槛支持家族覆盖（quality 0.75 / sentiment 0.70：慢变量、低披露）
    n_total_rows = len(factor_frame)
    cov = factor_frame.notna().mean(axis=0)
    nunique = factor_frame.nunique(dropna=True, axis=0)
    card["coverage"] = cov
    card["nunique"] = nunique
    card["quality_ok"] = (cov >= spec.get("min_coverage", cfg["min_coverage"])) & (nunique >= cfg["min_nunique"])

    # ② IC / ICIR / win_rate —— 统一"周频口径"（每 5 个交易日一期）
    # 规范阈值（IC≥0.015 / ICIR≥0.35 等）定义在周频多因子语境下：
    #   ic    = 周频 IC 均值（直接与 0.015 / 0.02 等门槛可比）
    #   icir  = 周频 IC 均值 / 周频 IC 标准差（不年化，与规范 0.35/0.4 同量级）
    #   win_rate = 周频 IC > 0 占比
    if daily_ic is None:
        daily_ic = compute_daily_ic_frame(factor_frame, labels)
    if daily_ic is not None and not daily_ic.empty:
        n_rows = len(daily_ic)
        week_idx = np.arange(n_rows) // 5
        weekly = daily_ic.groupby(week_idx).mean()
        # 索引用每期最后一个交易日日期（保持真实日期语义，避免整数被误解析）
        date_idx = pd.to_datetime(daily_ic.index)
        last_dates = date_idx.to_series().groupby(week_idx).last()
        weekly.index = pd.DatetimeIndex(last_dates.values)
        ic_mean = weekly.mean(axis=0)
        ic_std = weekly.std(axis=0, ddof=1).replace(0, np.nan)
        card["ic"] = ic_mean
        # [P3] ICIR 用 Newey-West 自相关修正（5 日重叠标签 → 原生 mean/std 虚高 √5 量级；
        #      与 run_eval.py --admit 的 icir_nw 口径一致，筛选端/准入端口径统一）
        from qlworks.evaluation.ic_analysis import calc_newey_west_tstat
        nw_std_map: dict = {}
        for _c in weekly.columns:
            _ic_clean = weekly[_c].dropna()
            if len(_ic_clean) >= 10:
                _nw = calc_newey_west_tstat(_ic_clean)
                _se = _nw.get("nw_std")
                _n = _nw.get("n_obs", len(_ic_clean))
                _s = _se * np.sqrt(_n) if (_se and _se > 1e-12 and _n > 0) else float(_ic_clean.std())
            else:
                _s = float(_ic_clean.std()) if len(_ic_clean) > 1 else np.nan
            nw_std_map[_c] = _s if _s and _s > 1e-12 else np.nan
        card["icir"] = ic_mean / pd.Series(nw_std_map)
        card["icir_raw"] = ic_mean / ic_std  # 保留原生口径供对比（虚高参考）
        # win_rate 按因子主符号方向计算"同号占比"：负向稳定因子不被误杀
        sign = ic_mean >= 0
        win_pos = (weekly > 0).mean(axis=0)
        win_neg = (weekly < 0).mean(axis=0)
        card["win_rate"] = win_pos.where(sign, win_neg)
        daily_ic = weekly  # 后续分年度/跨年度同号检验统一用周频面板
    else:
        card["ic"] = np.nan
        card["icir"] = np.nan
        card["win_rate"] = np.nan

    # ②' 家族 IC/ICIR 门槛（Risk 家族不做 IC 强制）
    # 机构口径：因子方向可变，用 |IC| / |ICIR|（负向稳定因子同样有效）
    if ic_min is not None and icir_min is not None and win_rate_min is not None:
        card["ic_gate"] = (card["ic"].abs() >= ic_min) & (card["icir"].abs() >= icir_min) \
                          & (card["win_rate"] >= win_rate_min)
    else:
        card["ic_gate"] = True  # risk：仅作风险区分度

    # ③ ICIR 滚动稳定（复用 selector.check_icir_stability）
    stable_names = []
    try:
        feat = [c for c in factor_frame.columns if card.loc[c, "ic_gate"]]
        if len(feat) > 2:
            stable_names = check_icir_stability(
                factor_frame[feat], labels, feat,
                rolling_window=cfg["icir_window"],
                keep_ratio=cfg.get("icir_keep_ratio", 0.8), min_keep=1,
            )
    except Exception:
        stable_names = feat if 'feat' in dir() else []
    card["icir_stable"] = [c in set(stable_names) for c in card.index]
    # 事件驱动家族（sentiment）：滚动 ICIR 稳定占比 <80% 是脉冲式 IC 的本质，改用 rolling 同号占比（P1-2a）
    if spec.get("skip_icir_stable"):
        card["icir_stable"] = True

    # ④ 家族内冗余（默认 |r| >= 0.7；量价家族由 special.redundancy_inner 收紧到 0.50）
    card["redundant"] = False
    try:
        kept = check_redundancy(
            factor_frame[card.index.tolist()], card.index.tolist(),
            threshold=spec.get("redundancy_inner", cfg["redundancy_threshold"]),
            rank={c: (card.loc[c, "icir"] if pd.notna(card.loc[c, "icir"]) else 0)
                  for c in card.index},
        )
        card["redundant"] = ~card.index.isin(set(kept))
    except Exception:
        pass

    # ⑤ BH 多重检验（[P3] 真实 t 检验 p 值 + 全库试验数 m）
    # 旧实现用 |IC| 相对分位当 p 值（伪 p 值，循环论证）；现改为周频 IC 的
    # t 检验 p 值（ic_mean/(ic_std/√n) → 正态近似 p），BH 按全库试验数 m 校正：
    # 未参与本家族检验的其他家族因子记为 p=1.0（最不显著），体现跨家族多重检验。
    card["significant"] = False
    try:
        from statsmodels.stats.multitest import multipletests
        n_weeks = len(daily_ic) if daily_ic is not None and not daily_ic.empty else 0
        if n_weeks >= 20:
            ic_m = daily_ic.mean(axis=0)
            ic_s = daily_ic.std(axis=0, ddof=1).replace(0, np.nan)
            t_all = ic_m / (ic_s / np.sqrt(n_weeks))
            # 向量化双侧正态近似 p（标量 math.erf 对 Series 抛 TypeError，曾致 BH 走 except 假全过）
            from scipy.stats import norm
            p_all = pd.Series(2 * (1 - norm.cdf(t_all.abs().values)), index=t_all.index)
        else:
            p_all = pd.Series(1.0, index=card.index)
        pvals = p_all.reindex(card.index).fillna(1.0)
        m = max(len(all_factor_names), len(pvals))              # 全库试验数
        pvals_ext = np.concatenate([pvals.values, np.ones(m - len(pvals))])
        _, qvals, _, _ = multipletests(pvals_ext, alpha=cfg["correction_alpha"],
                                       method=cfg["correction_method"])
        card["significant"] = qvals[: len(pvals)] < cfg["correction_alpha"]
        card["p_value"] = pvals
        print(f"  [BH] 显著 {int(card['significant'].sum())}/{len(card)} (真实t检验p, m={m}全库)")
    except Exception:
        card["significant"] = True  # 降级：不做多重检验时全通过（日志告警）

    # + 家族特殊检验（spec 已在函数开头提取）
    card["special_ok"] = True
    # [P0-核实] 跨年度同号占比门槛支持家族覆盖：
    #   value 放宽 0.50（12 年 39% 回撤的正常 regime） / quality 收紧 0.65（QMJ 22/24 国=92%）
    if spec.get("cross_regime") and daily_ic is not None and not daily_ic.empty:
        # daily_ic 索引可能为字符串日期（qlib datetime level），统一转 Timestamp
        years = pd.to_datetime(daily_ic.index).year
        annual = daily_ic.groupby(years).mean()
        n_years = annual.shape[0]
        if n_years >= 2:
            # 每个因子的"正 IC 年度占比"（跨年度同号稳定性）
            sign_ratio = (annual > 0).mean(axis=0)
            card["regime_same_ratio"] = sign_ratio.reindex(card.index).fillna(0)
            card["special_ok"] &= card["regime_same_ratio"] >= spec.get(
                "regime_min_same_ratio", cfg["regime_min_same_ratio"])
    if spec.get("crowding"):
        corr = factor_frame[card.index.tolist()].corr(method="spearman").abs()
        avg_corr = corr.mean(axis=0)
        card["avg_corr"] = avg_corr.reindex(card.index).fillna(0)
        # 拥挤度定位：组合层监控指标，不参与筛选一票否决——
        # ① 入选因子两两 |r|<0.7 已由家族内冗余门保证（可分散性底线）
        # ② 同质但有效的因子族（如波动反转）是机构接受的"驱动组"，
        #    一刀切会误杀全部；拥挤度高于阈值仅作预警，交由组合层配额/权重约束
        # [P0-核实] 阈值支持家族覆盖（sentiment 放宽 0.65：同源情绪因子天然相关 0.5-0.7）
        crowd_th = spec.get("crowding_max_avg_corr", cfg["crowding_max_avg_corr"])
        high_crowding = card.loc[card["avg_corr"] > crowd_th].index.tolist()
        if high_crowding:
            print(f"  [拥挤度预警] {len(high_crowding)} 个因子同家族平均相关偏高(>{crowd_th}): "
                  f"{high_crowding[:6]}{'...' if len(high_crowding) > 6 else ''}（组合层建议控制同驱动权重）")
    if spec.get("risk_rank"):
        # 风险家族：不做 IC 门槛，仅查分层区分度（用 IC 面板的波动区分代理）
        card["special_ok"] &= card["coverage"] >= cfg["min_coverage"]
        if "ic" in card.columns:
            card["special_ok"] &= card["ic"].notna()

    # ==============================================================================
    # [P1 家族专属钩子]（2026-10-11 落地，全部 special 开关驱动）
    # 注意：② 已将 daily_ic 替换为周频 IC 面板（index=DatetimeIndex，列=因子）
    # ==============================================================================

    # [P1-1] 家族级 |t| 软门槛（HLZ 2016 多重检验口径；momentum_price 3.0，其余 2.5）
    t_min = spec.get("t_min")
    if t_min and daily_ic is not None and not daily_ic.empty and len(daily_ic) >= 20:
        try:
            n_weeks = len(daily_ic)
            ic_m = daily_ic.mean(axis=0)
            ic_s = daily_ic.std(axis=0, ddof=1).replace(0, np.nan)
            card["t_stat"] = ic_m / (ic_s / np.sqrt(n_weeks))
            card["t_min_ok"] = card["t_stat"].abs() >= t_min
            # [P3] t 门槛接入 passed（此前仅打印不筛选；HLZ 2016 硬门槛生效）
            card["special_ok"] &= card["t_min_ok"]
            print(f"  [t 门槛] {int(card['t_min_ok'].sum())}/{len(card)} 个通过 (|t|>={t_min})")
        except Exception:
            card["t_min_ok"] = True
    else:
        card["t_min_ok"] = True

    # [P1-2a] 慢变量 52 周滚动 IC 同号占比（quality/sentiment：单期 IC 低但持续为正/负 → 持续性门槛）
    #          按因子主符号方向计算：负向稳定因子用"负周占比"，不被误杀
    rwin = spec.get("rolling_ic_window_weeks")
    if rwin and daily_ic is not None and not daily_ic.empty and len(daily_ic) >= rwin:
        try:
            pos_ratio = daily_ic.rolling(rwin, min_periods=max(int(rwin * 0.5), 26)) \
                .apply(lambda x: (x > 0).mean(), raw=True)
            sign = daily_ic.mean(axis=0) >= 0
            same_ratio = pos_ratio.where(sign, 1 - pos_ratio)  # 负向因子取负占比
            card["rolling_pos_ratio"] = same_ratio.mean(axis=0).reindex(card.index)
            rp_min = spec.get("rolling_positive_ratio", 0.60)
            card["special_ok"] &= card["rolling_pos_ratio"] >= rp_min
            print(f"  [rolling持续性] {int((card['rolling_pos_ratio'] >= rp_min).sum())}/{len(card)} 个通过 ({rwin}周同号占比>={rp_min})")
        except Exception as e:
            print(f"  [rolling持续性] 失败({e})，跳过")

    # [P1-2b] 熊市保护：熊市 regime 子样本 IC 不得显著为负
    #          优先 CH 宽基指数（000300.SH，机构"12月收益<−25%"口径）；失败回退标签截面均值代理
    if spec.get("bear_regime_protection") and daily_ic is not None and not daily_ic.empty:
        try:
            bear_weeks = None
            idx_code = spec.get("bear_index_code", "000300.SH")
            bear_cum = spec.get("bear_cum_return_52w", -0.15)
            bear_weeks = _load_ch_bear_weeks(idx_code, cfg["start_time"], cfg["end_time"], bear_cum)
            ic_period_w = pd.to_datetime(daily_ic.index).to_period("W")
            if bear_weeks:
                bear_rows = ic_period_w.isin(bear_weeks)
            else:
                # 回退：标签截面均值（5 日前瞻窗口）月序列 < bear_5d_ret_threshold
                lab_dates = pd.to_datetime(labels.index.get_level_values("datetime"))
                mkt_month = labels.groupby(lab_dates.to_period("M")).mean()
                bear_th = spec.get("bear_5d_ret_threshold", -0.01)
                bear_months = mkt_month[mkt_month < bear_th].index
                ic_period_m = pd.to_datetime(daily_ic.index).to_period("M")
                bear_rows = ic_period_m.isin(bear_months)
            if int(bear_rows.sum()) >= 10:
                bear_ic = daily_ic.loc[bear_rows]
                b_mean = bear_ic.mean(axis=0)
                b_std = bear_ic.std(axis=0, ddof=1).replace(0, np.nan)
                bear_t = b_mean / (b_std / np.sqrt(int(bear_rows.sum())))
                card["bear_t"] = bear_t.reindex(card.index)
                bmin = spec.get("bear_ic_min_t", -1.5)
                card["special_ok"] &= card["bear_t"] > bmin
                print(f"  [熊市保护] {int((card['bear_t'] > bmin).sum())}/{len(card)} 个通过 (熊市{int(bear_rows.sum())}周, t>{bmin})")
            else:
                print(f"  [熊市保护] 熊市周不足 {int(bear_rows.sum())}（<10），跳过")
        except Exception as e:
            print(f"  [熊市保护] 失败({e})，跳过")

    # [P1-3a] 分层分离度（risk）：五分层标签均值单调 + 分层收益差异显著（KW）
    if spec.get("risk_layer_separation") and spec.get("risk_rank"):
        rls = spec["risk_layer_separation"]
        try:
            from scipy.stats import kruskal, spearmanr
            sep_ok = {}
            for f in card.index.tolist():
                pair = pd.DataFrame({"f": factor_frame[f], "y": labels.reindex(factor_frame.index)}).dropna()
                if len(pair) < 500:
                    sep_ok[f] = True  # 样本不足放行
                    continue
                try:
                    q = pd.qcut(pair["f"], 5, labels=False, duplicates="drop")
                except Exception:
                    sep_ok[f] = True
                    continue
                if q.nunique(dropna=True) < 3:
                    sep_ok[f] = True
                    continue
                gm = pair.groupby(q.values)["y"].mean()
                mono = abs(spearmanr(list(range(len(gm))), gm.values)[0]) if len(gm) >= 3 else 0.0
                try:
                    kp = kruskal(*[pair["y"].values[q.values == i] for i in sorted(set(q.values))])[1]
                except Exception:
                    kp = 0.0
                sep_ok[f] = (mono >= rls.get("min_mono_corr", 0.5)) and (kp < rls.get("kruskal_wallis_p", 0.05))
            card["layer_sep_ok"] = [sep_ok.get(c, True) for c in card.index]
            card["special_ok"] &= pd.Series(card["layer_sep_ok"], index=card.index)
            print(f"  [分层分离度] {int(sum(sep_ok.values()))}/{len(card)} 个通过 (五分层单调|corr|>={rls.get('min_mono_corr', 0.5)} + KW p<{rls.get('kruskal_wallis_p', 0.05)})")
        except Exception as e:
            print(f"  [分层分离度] 失败({e})，跳过")

    # [P1-3b] rolling 稳定性（risk）：逐因子相邻周截面秩相关（列间 Spearman），滚动 52 周均值
    if spec.get("risk_rolling_stability") and spec.get("risk_rank"):
        rrs = spec["risk_rolling_stability"]
        try:
            win_weeks = rrs.get("window_weeks", 52)
            stab = {}
            for f in card.index.tolist():
                wide = factor_frame[f].unstack("datetime")  # inst × date
                if wide.shape[1] < 30:
                    stab[f] = np.nan
                    continue
                weekly = wide.resample("W", axis=1).last()  # 周末截面
                wr = weekly.rank(axis=0)                    # 每周截面秩（列=周）
                corrs = [wr.iloc[:, i - 1].corr(wr.iloc[:, i]) for i in range(1, wr.shape[1])]
                corrs = [c for c in corrs if c == c]        # 去 NaN（停牌致公共样本不足）
                if len(corrs) >= 26:
                    stab[f] = pd.Series(corrs).rolling(win_weeks, min_periods=26).mean().mean()
                else:
                    stab[f] = np.nan
            card["rank_stability"] = pd.Series(stab).reindex(card.index)
            rs_min = rrs.get("cross_section_rank_corr_min", 0.70)
            card["special_ok"] &= card["rank_stability"] >= rs_min
            print(f"  [rolling稳定] {int((card['rank_stability'] >= rs_min).sum())}/{len(card)} 个通过 ({win_weeks}周秩相关>={rs_min})")
        except Exception as e:
            print(f"  [rolling稳定] 失败({e})，跳过")

    # [P1-3c] 方向一致性（risk）：因子级期望方向标注，错配 → 告警人工复核（不否决）
    if spec.get("risk_direction_check") and spec.get("risk_rank"):
        rdc = spec["risk_direction_check"]
        try:
            dir_map = rdc.get("direction_map", {})
            if dir_map and daily_ic is not None and not daily_ic.empty:
                ic_med = daily_ic.median(axis=0)
                mismatch = []
                for name, expect in dir_map.items():
                    if name not in ic_med.index or pd.isna(ic_med[name]):
                        continue
                    got_pos = ic_med[name] > 0
                    want_pos = (expect == "pos")
                    if got_pos != want_pos and abs(ic_med[name]) > 1e-6:
                        mismatch.append(f"{name}(期望{'正' if want_pos else '负'},实际{'正' if got_pos else '负'})")
                if mismatch:
                    print(f"  [方向一致性] ⚠ 错配 {len(mismatch)} 个（连续 12 月错配需人工复核）: {mismatch[:6]}")
        except Exception as e:
            print(f"  [方向一致性] 失败({e})")

    # [P1-5] value 多描述量一致性（家族资格检验）：≥min_pass 个描述量 |IC|≥min_ic_each 且方向归一后同号
    if spec.get("descriptor_consistency"):
        dc = spec["descriptor_consistency"]
        try:
            dir_map = dc.get("direction_map", {})
            if dir_map and daily_ic is not None and not daily_ic.empty:
                ic_m = daily_ic.mean(axis=0)
                passed_desc = []
                for name, d in dir_map.items():
                    if name in ic_m.index and pd.notna(ic_m[name]) \
                            and abs(ic_m[name]) >= dc.get("min_ic_each", 0.010):
                        passed_desc.append((name, ic_m[name], d))
                n_pass = len(passed_desc)
                value_signal = [ic_m[n] * d for n, _, d in passed_desc]
                same_sign = all(s > 0 for s in value_signal)
                ok = (n_pass >= dc.get("min_pass", 2)) and same_sign
                if 0 < n_pass < dc.get("min_pass", 2):
                    print(f"  [描述量一致性] ⚠ 仅 {n_pass} 个描述量通过（<min_pass={dc.get('min_pass', 2)}），家族降级观察")
                else:
                    print(f"  [描述量一致性] {n_pass} 个描述量 |IC|>={dc.get('min_ic_each', 0.010)}，方向归一后同号={same_sign} → 家族{'通过' if ok else '不通过'}")
                card["descriptor_ok"] = ok
                card["special_ok"] &= ok  # 家族级判定广播到全部因子
        except Exception as e:
            print(f"  [描述量一致性] 失败({e})，跳过")

    # 汇总判定：质量门 & IC 门 & 稳定 & 非冗余 & 显著 & 特殊
    card["passed"] = (
        card["quality_ok"]
        & card["ic_gate"]
        & card["icir_stable"]
        & ~card["redundant"]
        & card["significant"]
        & card["special_ok"]
    )
    return card, daily_ic


def _load_ch_bear_weeks(index_code: str, start_time: str, end_time: str,
                        cum_return_52w: float = -0.15, min_periods: int = 26) -> list:
    """从局域网 ClickHouse 加载宽基指数，返回 52 周累计收益 < 阈值 的熊市周 Period 列表。
    数据源：quant_db.index_daily（含 000300.SH 2002 起 / 000905.SH / 000852.SH 2004 起）。
    口径：机构"过去 12 月收益 <−25%"的 A 股适配（沪深300 累计 -15%）。
    失败时返回空列表（调用方回退标签代理）。"""
    try:
        import clickhouse_connect
        from qlworks.config import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
        cli = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, user=CH_USER,
                                            password=CH_PASSWORD, database=CH_DATABASE,
                                            connect_timeout=15)
        rows = cli.query(
            "SELECT trade_date, close FROM index_daily "
            "WHERE ts_code = {code:String} AND trade_date >= {start:String} AND trade_date <= {end:String} "
            "ORDER BY trade_date",
            parameters={"code": index_code, "start": start_time, "end": end_time}).result_rows
        if not rows:
            return []
        df = pd.DataFrame(rows, columns=["trade_date", "close"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["ret"] = df["close"].pct_change()
        ret_w = df.set_index("trade_date")["ret"].resample("W").sum()
        cum = ret_w.rolling(52, min_periods=min_periods).sum()
        bear = cum[cum < cum_return_52w].index.to_period("W").tolist()  # PeriodIndex（对齐调用方 isin）
        if bear:
            print(f"  [熊市保护] CH 指数 {index_code}：{len(bear)} 个熊市周 (52周累计<-{abs(cum_return_52w):.0%})")
        return bear
    except Exception as e:
        print(f"  [熊市保护] CH 指数加载失败({e})，回退标签代理")
        return []

def _normal_cdf(x):
    """标准正态 CDF（免 scipy 依赖的轻量实现）。"""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ==============================================================================
# [全局去冗余] 跨家族兜底（保留优先级：基本面>预期>量价>风险）
# ==============================================================================

def _global_dedup(candidates_by_family: dict, cache, start_time, end_time, cfg) -> list[str]:
    """对全部入选因子做跨家族 Spearman 去重（默认阈值 0.70）。

    保留优先级 FAMILY_PRIORITY；同一优先级内保留 IC 高者。
    [量价移植] 全局阈值取"有入选因子的家族中 special.redundancy_global 最低值"：
    momentum_price 配 0.65 时全局收紧到 0.65，其他家族未配时保持引擎默认 0.70。
    """
    all_names = []
    for fam in FAMILY_PRIORITY:
        all_names.extend(candidates_by_family.get(fam, []))
    if len(all_names) <= 2:
        return all_names

    # [量价移植] 有效全局去冗余阈值 = 各家族 special.redundancy_global 最小值
    eff_threshold = cfg["global_redundancy_threshold"]
    for fam, names in candidates_by_family.items():
        if not names:
            continue
        fam_th = FAMILY_DEFS.get(fam, {}).get("special", {}).get("redundancy_global")
        if fam_th is not None and fam_th < eff_threshold:
            eff_threshold = fam_th

    df = cache.get_warehouse_df(all_names, start_time=start_time, end_time=end_time)
    if df.empty:
        return all_names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    rank = {}
    for fam in FAMILY_PRIORITY:
        for name in candidates_by_family.get(fam, []):
            rank[name] = FAMILY_PRIORITY.index(fam) * 100  # 家族优先级为主序
    for i, fam in enumerate(FAMILY_PRIORITY):
        base = i * 100
        for j, name in enumerate(candidates_by_family.get(fam, [])):
            rank[name] = base + (len(candidates_by_family.get(fam, [])) - j)  # 同族内列表序=IC降序
    try:
        kept = check_redundancy(df, all_names, threshold=eff_threshold,
                                rank=rank)
        return [n for n in all_names if n in kept]
    except Exception:
        return all_names


# ==============================================================================
# [多周期分段稳健性检验（旧名"滚动筛选"）/ 嵌入法 / 因子中性化]
# ==============================================================================

def _parse_rolling_windows(spec) -> list[tuple[str, str, str]]:
    """解析多周期分段窗口配置 → [(start, end, name), ...]。

    语义澄清：这是"多周期分段稳健性检验"（sub-period robustness）——
    各窗口彼此独立、并行重复完整检验，再跨窗口聚合；不是时序滚动外推。
    与深评 --walk-forward（train→valid 时序外推）是两种不同检验。
    支持两种写法：
      --period-windows 2023,2024,2025          每年全年窗口
      --period-windows 2023-01-01/2023-12-31,2024-01-01/2024-12-31
    （旧名 --rolling-windows 保留为兼容别名）
    """
    out: list[tuple[str, str, str]] = []
    if not spec:
        return out
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    for p in parts:
        if "/" in p:
            s, e = p.split("/", 1)
            out.append((s.strip(), e.strip(), f"{s.strip()[:7]}~{e.strip()[:7]}"))
        elif len(p) == 4 and p.isdigit():
            out.append((f"{p}-01-01", f"{p}-12-31", p))
        else:
            # 容忍 "2023-01-01" 单点：取该年全年
            year = p[:4]
            out.append((f"{year}-01-01", f"{year}-12-31", year))
    return out


def _embedded_selection(factor_frame: pd.DataFrame, labels: pd.Series,
                        candidates: list[str], effective_top_k: int,
                        min_keep: int = 3) -> list[str]:
    """树模型嵌入法精选（LGBM importance，组合视角）。

    语义：单因子 IC 门只检验"各自预测力"，嵌入法补"组合/非线性贡献"——
    IC 粗筛出的候选在树上共同拟合标签，importance(gain) 高者与其余因子互补。
    对齐 select_factors / train_tree 的嵌入法用法（LGBM gain 重要性）。

    防御：候选过少 / 样本不足 / lightgbm 缺失时保留原名单，不崩管线。
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [嵌入法] lightgbm 未安装，跳过嵌入法精选，保留基础门通过名单")
        return candidates
    if len(candidates) <= min_keep:
        return candidates
    comb = factor_frame[candidates].join(labels.rename("__y__"), how="inner").dropna()
    if len(comb) < 200:
        print(f"  [嵌入法] 样本不足({len(comb)})，跳过")
        return candidates
    X, y = comb[candidates], comb["__y__"]
    model = lgb.LGBMRegressor(
        n_estimators=CONFIG["embedded_selection"]["n_estimators"],
        learning_rate=0.05, num_leaves=15, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1,
        importance_type=CONFIG["embedded_selection"]["importance_type"],
    )
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=candidates).sort_values(ascending=False)
    keep = imp.index[:effective_top_k].tolist()
    if len(keep) < min_keep:
        print(f"  [嵌入法] 精选因子过少({len(keep)})，保留基础门名单")
        return candidates
    print(f"  [嵌入法] 组合重要性 Top {len(keep)}: "
          + ", ".join(f"{n}({imp[n]:.0f})" for n in keep[:6])
          + ("..." if len(keep) > 6 else ""))
    return keep


def _factor_neutralize_select(factor_frame: pd.DataFrame, labels: pd.Series,
                              candidates: list[str], cfg: dict,
                              spec: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """因子中性化（Ridge 剥离行业+市值暴露）后重算 IC，保留仍有预测力者。

    对齐 train_tree-doubao 的 factor_neutralization 语义：因子与行业/市值正交后，
    其 IC 反映的是纯 Alpha 而非风格暴露。默认关闭（树模型场景常跳过）。
    [P0-核实] spec 家族覆盖：value/quality 配置 factor_neutralize=True 强制开，
    防裸 IC 被金融/公用事业行业暴露主导（Barra CNE5 / MSCI sector-relative 口径）。
    """
    neut_conf = cfg["factor_neutralization"]
    if spec:
        neut_conf = {
            **neut_conf,
            "enabled": spec.get("factor_neutralize", neut_conf["enabled"]),
            "min_ic_after_neutralize": spec.get(
                "min_ic_after_neutralize", neut_conf["min_ic_after_neutralize"]),
        }
    if not neut_conf.get("enabled") or len(candidates) <= 2:
        return factor_frame, candidates
    from qlworks.processors.neutralize import CSNeutralize
    # 构建 MultiIndex（CSNeutralize 需要 fields_group 结构）
    neut_df = factor_frame[candidates].copy()
    neut_df.columns = pd.MultiIndex.from_product([["feature"], neut_df.columns])
    try:
        neutralizer = CSNeutralize(fields_group="feature", industry_field="sw_l1",
                                   market_cap_field="circ_mv", log_mc=True)
        result = neutralizer(neut_df)
        neut_feat = result["feature"].copy()
        # 中性化后重算周频 IC（与 _run_family_gates 同口径）
        combined = neut_feat.join(labels.rename("__y__"), how="inner").dropna()
        if len(combined) < 200:
            print("  [因子中性化] 样本不足，跳过")
            return factor_frame, candidates
        ics = compute_factor_ics(combined, combined["__y__"], min_samples=50)
        kept = [n for n in candidates if abs(ics.get(n, 0)) > neut_conf["min_ic_after_neutralize"]]
        # 自适应：至少保留 30%，且不少于 2 个
        min_keep = max(int(len(candidates) * neut_conf["min_keep_ratio"]), 2)
        if len(kept) >= min_keep:
            print(f"  [因子中性化] 剥离行业/市值后仍有预测力: {len(kept)}/{len(candidates)} "
                  f"(|IC|>{neut_conf['min_ic_after_neutralize']})")
            return neut_feat, kept
        print(f"  [因子中性化] 有效因子过少({len(kept)})，保留原名单")
        return factor_frame, candidates
    except Exception as e:
        print(f"  [因子中性化] 失败({e})，保留原名单")
        return factor_frame, candidates


def _representative_select(factor_frame: pd.DataFrame, candidates: list[str],
                           ic_rank: dict, corr_threshold: float = 0.50,
                           max_rep: int = 16) -> list[str]:
    """[P3] 贪心聚类代表选择：粗筛候选 → 每簇留 |IC| 最高代表，簇间 |corr|<阈值。

    依据（报告 P0）：momentum_price 215 个因子 89.2% 同族，多数为参数变体
    （两两相关 0.7-0.95）；机构（WorldQuant 101 Alphas）入库 alpha 平均成对相关
    仅 15.9%，0.7 只是拒收红线。用 |IC| 降序贪心 + 0.50 相关阈值收敛到 ≤max_rep 个代表，
    保证进入质检链的候选具备族内多样性，避免 60 个全为同一模板变体白算后段门。
    """
    if not candidates or len(candidates) <= max_rep:
        return candidates
    frame = factor_frame[candidates].replace([np.inf, -np.inf], np.nan)
    ranked = sorted(candidates, key=lambda n: abs(ic_rank.get(n, 0.0)), reverse=True)
    kept: list[str] = []
    for name in ranked:
        if len(kept) >= max_rep:
            break
        if not kept:
            kept.append(name)
            continue
        cur = frame[name].dropna()
        corrs = []
        for k in kept:
            pair = pd.concat([cur, frame[k].dropna()], axis=1, join="inner").dropna()
            if len(pair) < 30:
                continue
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
            if pd.notna(r):
                corrs.append(abs(r))
        if not corrs or max(corrs) < corr_threshold:
            kept.append(name)
    print(f"  [聚类代表] {len(candidates)} → {len(kept)} 个代表 (|corr|<{corr_threshold}, 上限{max_rep})")
    return kept


def _purge_embargo_label(label: pd.Series, purge_days: int = 5, embargo_days: int = 10) -> pd.Series:
    """[P3] 滚动窗口边界治理：剔除窗口首 embargo_days 个交易日（上窗口冷却）
    与尾 purge_days 个交易日（其 5 日标签 Ref(-5) 已越出窗口/与下窗口重叠）。
    返回清洗后的标签序列（同索引类型）。"""
    dates = pd.Index(sorted(set(label.index.get_level_values("datetime"))))
    if len(dates) <= (purge_days + embargo_days):
        return label
    keep_dates = dates[embargo_days: len(dates) - purge_days]
    mask = label.index.get_level_values("datetime").isin(set(keep_dates))
    return label[mask]


def _screen_family_in_window(fam, fs, feature_cache, name_expr_map, label_series,
                             start_time, end_time, cfg, tradeable_mask,
                             all_factor_names, family_factors):
    """单个窗口内对单个家族执行完整质检链。

    流程：家族内分批 IC 粗筛 → 因子中性化(可选) → 置换检验(可选)
          → 家族差异化 5 门 + 特殊检验 → 树模型嵌入法精选(可选)。
    返回 (card_df, passed_names)；card 含 family 列，window 列由调用方按模式追加。
    """
    if not fs:
        return pd.DataFrame(), []
    fam_names = [f["name"] for f in fs]
    fd = FAMILY_DEFS[fam]
    spec = fd.get("special", {})   # [P0-核实] 家族 special 配置（置换/中性化/衰减门共用）
    print(f"\n{'─' * 66}\n  [{fam}] {fd['label']} ({len(fam_names)} 因子, "
          f"IC≥{fd['ic_min'] or '—'} ICIR≥{fd['icir_min'] or '—'})\n{'─' * 66}")

    # 阶段一：家族内分批 IC 粗筛（收敛候选规模）
    ic_all, top_names = _family_batch_ic(
        feature_cache, fam_names, name_expr_map, label_series,
        start_time, end_time, cfg, tradeable_mask, fam,
    )
    if not top_names:
        print(f"    [跳过] {fam} 无因子通过 IC 粗筛")
        return pd.DataFrame(), []
    print(f"    [粗筛] 家族内 {len(fam_names)} → top {len(top_names)}")

    # 阶段二：全量加载 → 家族化质检链
    batch_df = _load_factor_batch(feature_cache, name_expr_map, top_names,
                                  start_time, end_time)
    batch_df = _apply_tradeable_mask(batch_df, tradeable_mask)
    common = batch_df.index.intersection(label_series.index)
    if len(common) < 100:
        print(f"    [跳过] {fam} 有效样本不足")
        return pd.DataFrame(), []
    frame = batch_df.loc[common]
    labels = label_series.loc[common]

    # [P3] 粗筛后贪心聚类代表（momentum 家族：215 同族 → ≤16 代表，簇间 |corr|<redundancy_inner）
    ic_rank = {n: v for n, v in ic_all.items()}
    rep_max = spec.get("representative_max")
    if rep_max and len(top_names) > rep_max:
        top_names = _representative_select(
            frame, top_names, ic_rank,
            corr_threshold=spec.get("redundancy_inner", 0.50), max_rep=rep_max,
        )
        frame = frame[top_names]
        if frame.empty or len(frame.columns) < 2:
            print(f"    [跳过] {fam} 聚类代表后有效因子不足")
            return pd.DataFrame(), []

    # 因子中性化（可选，Ridge 剥离行业+市值后重算 IC；[P0-核实] 家族覆盖：value/quality 强制开）
    frame, top_names = _factor_neutralize_select(frame, labels, top_names, cfg,
                                                 spec=fd.get("special", {}))

    # 置换检验（可选，López de Prado）
    # [量价移植] 量价家族走 special.permutation_fdr：BH-FDR 多重检验校正 +
    # stride=5 采样消除 5 日重叠标签的序列自相关（select_momentum_price 口径）；
    # 其他家族保持引擎默认置换检验。
    perm_fdr = fd.get("special", {}).get("permutation_fdr")
    if perm_fdr and perm_fdr.get("enabled") and len(top_names) >= 3:
        try:
            kept_perm = _mp_permutation_significance(
                frame, labels, top_names,
                n_perms=perm_fdr["n_permutations"],
                pvalue_threshold=perm_fdr["pvalue_threshold"],
                min_keep=perm_fdr.get("min_keep", 3),
                fdr=True,
                stride=perm_fdr.get("stride", 5),
            )
            print(f"    [置换检验·BH-FDR] {len(top_names)} → {len(kept_perm)} 显著 (q<{perm_fdr['pvalue_threshold']}, stride={perm_fdr.get('stride', 5)})")
            top_names = kept_perm
        except Exception as e:
            print(f"    [置换检验·BH-FDR] 失败({e})，保留原结果")
    elif (spec.get("permutation_enabled", cfg["permutation_test"].get("enabled"))
          and len(top_names) >= 3):
        try:
            kept_perm = _permutation_significance(
                frame, labels, top_names,
                n_perms=spec.get("permutation_n_perms",
                                 cfg["permutation_test"]["n_permutations"]),
                pvalue_threshold=cfg["permutation_test"]["pvalue_threshold"],
                min_keep=3,
            )
            print(f"    [置换检验] {len(top_names)} → {len(kept_perm)} 显著")
            top_names = kept_perm
        except Exception as e:
            print(f"    [置换检验] 失败({e})，保留原结果")

    card, daily_ic = _run_family_gates(frame, labels, all_factor_names, cfg, fam)
    card["family"] = fam

    passed = card.loc[card["passed"]].index.tolist()   # reset 前 index 即 factor_name

    # 树模型嵌入法（可选，组合重要性精选）
    emb_conf = cfg["embedded_selection"]
    if emb_conf.get("enabled") and len(passed) > emb_conf["min_keep"]:
        effective_top_k = max(int(len(passed) * 0.6), emb_conf["min_keep"])
        kept_emb = _embedded_selection(frame, labels, passed, effective_top_k,
                                       emb_conf["min_keep"])
        # 嵌入法结果回注 card：仅保留"基础门通过 且 嵌入法保留"的因子
        card["embedded_kept"] = card.index.isin(set(kept_emb))
        card["passed"] = card["passed"] & card["embedded_kept"]
        passed = card.loc[card["passed"]].index.tolist()

    # [量价移植] IC 衰减硬门槛 + 方向一致性门控（select_momentum_price 口径）：
    # 持有期 H5 内 IC 崩塌或 H1/H3/H5 方向翻转的因子直接淘汰，防"短周期强、活不到卖出"。
    dg = fd.get("special", {}).get("decay_gate")
    if dg and dg.get("enabled") and passed:
        try:
            # D.features 的 instrument 必须是解析后的真实股票代码列表：
            # "main_board" 是静态池名，直接传会得到空面板（qlib 不识别）。
            # 优先取 feature_cache.resolved_instruments（与 select_momentum_price 原版一致）。
            _insts = getattr(feature_cache, "resolved_instruments", None)
            if not _insts:
                _insts = [cfg["instruments"]] if isinstance(cfg["instruments"], str) else cfg["instruments"]
            decay_df = _mp_ic_decay(
                feature_cache, passed, _insts,
                start_time, end_time, dg["horizons"],
            )
            kept_dg, dropped_dg = _mp_decay_gate(
                decay_df, passed, dg["horizons"],
                decay_floor=dg["decay_floor"],
                sign_consistency_horizons=dg["sign_consistency_horizons"],
                holding_horizon=dg["holding_horizon"],
            )
            if dropped_dg:
                _d_str = "; ".join(f"{k}:{v}" for k, v in list(dropped_dg.items())[:6])
                print(f"    [IC衰减门] 淘汰 {len(dropped_dg)} 个（持有期崩塌/方向翻转）: {_d_str}")
            card["decay_gate_kept"] = card.index.isin(set(kept_dg))
            card["passed"] = card["passed"] & card["decay_gate_kept"]
            passed = kept_dg
        except Exception as e:
            print(f"    [IC衰减门] 失败({e})，保留原结果")

    print(f"    [结果] 通过 {len(passed)}/{len(top_names)} 个: {passed[:8]}{'...' if len(passed) > 8 else ''}")
    if daily_ic is not None and not daily_ic.empty:
        _ic_for_annual = daily_ic.copy()
        _ic_for_annual.index = pd.to_datetime(_ic_for_annual.index)
        annual = _annual_ic_report(_ic_for_annual)
        if not annual.empty:
            _print_annual_report(annual)

    return card.reset_index(), passed


# ==============================================================================
# [主流程]
# ==============================================================================

# ==============================================================================
# [单脚本化 2026-10-12] 模块 B/C：深评 + 三关准入（唯一评测脚本模式）
# ==============================================================================

def _find_factor_in_library(m, name):
    """在因子库全部策略中查找因子定义（含 duckdb_expr 自动计算字段）。"""
    for s in [s for s in m.list_strategies() if "dictionary" not in s]:
        try:
            sfg = m.load_strategy_config(s)
            for fd in sfg.get("factors", []):
                if fd.get("name") == name:
                    expr = fd.get("expression", "")
                    duckdb_expr = ""
                    if isinstance(expr, dict):
                        duckdb_expr = str(expr.get("duckdb", ""))
                        expr = str(expr.get("qlib", str(expr)))
                    return {"name": name, "expr": str(expr),
                            "duckdb_expr": duckdb_expr, "category": s,
                            "sub_category": fd.get("sub_category", ""),
                            "meaning": fd.get("meaning", ""),
                            "_source_file": fd.get("_source_file", "")}
        except Exception:
            pass
    return None


def _read_registry_metrics(name: str):
    """从 registry.json 读取因子最新 IC 指标（--admit-only 复用深评历史结果）。"""
    import json as _json
    reg_path = Path(__file__).resolve().parents[2] / "factor_data" / "registry" / "registry.json"
    # 权威路径：E:\Quant\Qlibworks\factor_data\registry\registry.json（scripts/evaluation 旧物已删）
    if not reg_path.exists():
        return None, None
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = _json.load(f)
    except Exception:
        return None, None
    reg_factors = reg.get("factors") or []
    rentry = None
    if isinstance(reg_factors, dict):
        rentry = reg_factors.get(name) or {}
    elif isinstance(reg_factors, list):
        rentry = next((e for e in reg_factors if e.get("name") == name), {}) or {}
    ic_mean = rentry.get("ic_mean")
    icir_nw = rentry.get("icir_nw") or rentry.get("icir")
    return ic_mean, icir_nw


def _run_deep_eval_mode(args, cfg):
    """模块 B（深评）+ 模块 C（三关准入）——唯一评测脚本模式。

    数据流（机构范式：WW BRAIN 模拟器→提交校验一气呵成；DE Shaw OOS+评审即入池）：
      收集因子（--factor / --factor-list / --family 全家族）→
      deep_eval.evaluate_factor_list（仓库优先 → 数据质量 → FactorEvaluator 完整评测）→
      --admit/--admit-only：factor_stats.run_admit_decision 三关准入 → 写候选池 v2。

    统计口径唯一：IC/ICIR-NW/相关矩阵/三关/候选池读写全部走 factor_stats（单一事实源）。
    深评实现唯一：FactorEvaluator（runner.py），本脚本只做编排（机构"薄脚本 + 厚核心库"）。
    """
    from qlworks.factors import FactorLibraryManager

    log = logging.getLogger("factor_screen.deep")

    # ── 收集因子 ──
    m = FactorLibraryManager()
    factors: list[dict] = []
    if args.factor_list:
        names = _deep_eval.read_factor_list(args.factor_list)
        for name in names:
            info = _find_factor_in_library(m, name)
            if info:
                factors.append(info)
            else:
                log.warning(f"{name}: 因子库中未找到定义，跳过")
    elif args.factor:
        info = _find_factor_in_library(m, args.factor)
        if info:
            factors.append(info)
    elif args.family:
        if args.family not in FAMILY_DEFS:
            print(f"[错误] 未知家族: {args.family}，可选 {list(FAMILY_DEFS)}")
            sys.exit(1)
        for fd in load_family_factors(args.family):
            expr = fd.get("expression", "")
            duckdb_expr = ""
            if isinstance(expr, dict):
                duckdb_expr = str(expr.get("duckdb", ""))
                expr = str(expr.get("qlib", str(expr)))
            # 类别口径统一：category=家族名；sub_category=YAML 中文子类；_source_file=家族文件 stem
            factors.append({"name": fd["name"], "expr": str(expr),
                            "duckdb_expr": duckdb_expr, "category": args.family,
                            "sub_category": fd.get("category", ""),
                            "meaning": fd.get("meaning", ""),
                            "_source_file": fd.get("source_file", "")})
    else:
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            try:
                sfg = m.load_strategy_config(s)
                for fd in sfg.get("factors", []):
                    name = fd.get("name")
                    if not name:
                        continue
                    expr = fd.get("expression", "")
                    duckdb_expr = ""
                    if isinstance(expr, dict):
                        duckdb_expr = str(expr.get("duckdb", ""))
                        expr = str(expr.get("qlib", str(expr)))
                    factors.append({"name": name, "expr": str(expr),
                                    "duckdb_expr": duckdb_expr, "category": s,
                                    "sub_category": fd.get("sub_category", ""),
                                    "meaning": fd.get("meaning", ""),
                                    "_source_file": fd.get("_source_file", "")})
            except Exception:
                pass

    if not factors:
        print("[错误] 未收集到任何因子（--factor / --factor-list / --family 至少指定其一）")
        sys.exit(1)

    # [类别口径统一] 归一化类别为五大家族口径，保证筛查端→候选池→训练端前后一致
    for f in factors:
        f["category"] = _normalize_factor_family(f.get("category", ""))

    _tm = time.monotonic()
    print("=" * 66)
    print("  单因子评测（深评 + 三关准入）· 唯一评测脚本 factor_screen.py")
    shown = ", ".join(f["name"] for f in factors[:8]) + ("..." if len(factors) > 8 else "")
    print("  因子数: {}（{}）".format(len(factors), shown))
    print("  区间  : {} ~ {} | 股票池: {}".format(
        cfg["start_time"], cfg["end_time"], args.pool or cfg["instruments"]))
    print("  模式  : {} {}".format("Walk-Forward 估计-外推" if args.walk_forward else "全时段深评",
                                    "+ 三关准入 → 候选池" if (args.admit or args.admit_only) else ""))
    print("=" * 66)

    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading")

    # ── 模块 B：深评（仅统计，不建任何树模型；机构：单因子评测阶段无模型）──
    if args.admit_only:
        print("\n[准入模式] 跳过深评，直接从 registry 读取 IC 指标执行三关准入...")
        results = []
        for f in factors:
            results.append({"name": f["name"], "factor_info": f,
                            "tier": args.tier or "satellite",
                            "ic_mean": None, "icir_nw": None, "result": None})
    else:
        print("\n[深评] 开始完整深评（仅统计，不建任何树模型；机构：单因子评测阶段无模型）...")
        results = _deep_eval.evaluate_factor_list(
            factors,
            start_time=cfg["start_time"], end_time=cfg["end_time"],
            pool=args.pool or cfg["instruments"],
            recompute=args.recompute,
            walk_forward=args.walk_forward,
            qlib_data_dir=QLIB_DATA_DIR,
            instruments_dir=QLIB_DATA_DIR / "instruments",
            log=log,
        )
        print(f"[深评] 完成（{time.monotonic() - _tm:.1f}s）")

    # ── 模块 C：三关准入 → 候选池 v2（机构：低相关 + 增量贡献 + 方向一致）──
    if args.admit or args.admit_only:
        thresholds = {
            "max_correlation_existing": CONFIG["redundancy_threshold"],  # 0.70（WW self-corr 拒收红线）
            "max_correlation_barra": 0.50,
            "min_oos_icir": 0.5,
            "min_recent_3y_ic_positive_ratio": 0.60,
            "direction_consistency_required": True,
            "icir_improve_min": 0.0,
            "admission_cutoff_date": ADMISSION_CUTOFF_DATE,
            "admission_window_years": ADMISSION_WINDOW_YEARS,
        }
        _fd_root = Path(__file__).resolve().parents[2] / "factor_data"
        pool_path = _fd_root / "registry" / "candidate_pool.json"
        warehouse_root = _fd_root / "warehouse"
        # 准入窗口标签（严格早于 ADMISSION_CUTOFF_DATE；window 由 factor_stats 内部按年切片）
        from qlworks.evaluation.factor_stats import build_admission_window as _build_win
        _win = _build_win(thresholds["admission_cutoff_date"], thresholds["admission_window_years"])
        label_frame = _fs_load_label_frame(
            cfg["label_expr"], cfg["label_name"], cfg["instruments"],
            QLIB_DATA_DIR, _win["corr_start"], _win["corr_end"],
        )

        # ── [seed-quota] 首批建池种子配额（机构 seed alpha 策略）──
        # 仅当候选池为空（种子阶段）且启用配额时生效：每家族最多 N 个、按 |ICIR_NW| 降序截断，
        # 防止首批因子全堆一族；池非空后走正常增量三关（增量边际检验自然接管）。
        _pool_now = _load_candidate_pool(pool_path, thresholds)
        _admit_queue = list(results)
        if args.seed_quota and args.seed_quota > 0 and not _pool_now.get("factors"):
            # 先补全每个候选的 ICIR_NW（admit_only 从 registry 读）
            _ranked = []
            for r in results:
                if args.admit_only:
                    _im, _inw = _read_registry_metrics(r["name"])
                    r = dict(r, ic_mean=_im, icir_nw=_inw)
                _ranked.append(r)
            # 按家族分组 → |ICIR_NW| 降序 → 每家族取前 N
            from collections import defaultdict
            _by_fam: dict = defaultdict(list)
            for r in _ranked:
                _fam = (r.get("factor_info") or {}).get("category") or "unknown"
                _by_fam[_fam].append(r)
            _keep: list = []
            for _fam in sorted(_by_fam):
                _grp = sorted(_by_fam[_fam],
                              key=lambda x: abs(x.get("icir_nw") or 0.0), reverse=True)
                _cut = _grp[:args.seed_quota]
                _keep.extend(_cut)
                print(f"  [seed-quota] {_fam}: {len(_grp)} 个候选 → 按|ICIR_NW|取前 {args.seed_quota} 个"
                      + (f"（跳过 {len(_grp) - len(_cut)} 个: "
                         + ", ".join(x["name"] for x in _grp[args.seed_quota:][:5])
                         + ("..." if len(_grp) - len(_cut) > 5 else "")
                         if len(_grp) - len(_cut) > 0 else ""))
            _admit_queue = _keep
        elif args.seed_quota and args.seed_quota > 0:
            print(f"  [seed-quota] 候选池非空（{len(_pool_now.get('factors', []))} 个），种子配额不生效，走正常增量三关")

        admitted = 0
        for i, r in enumerate(_admit_queue, 1):
            name = r["name"]
            tier = args.tier or r.get("tier") or "satellite"
            print(f"  [admit {i}/{len(_admit_queue)}] {name} (tier={tier}) ...")
            if tier not in ("core", "satellite"):
                print(f"  [admit] {name}: 评测档位={tier}（非 core/satellite），不入候选池"
                      f"——先调因子再评测（机构范式：reject 不进 Alpha Book）")
                continue
            if args.admit_only:
                ic_mean, icir_nw = _read_registry_metrics(name)
                if ic_mean is None:
                    print(f"  [跳过] {name}: registry 无 IC 指标——请用 `--deep-eval --admit` 评测即入库（机构范式主路径）")
                    continue
            else:
                ic_mean = r.get("ic_mean")
                icir_nw = r.get("icir_nw")
            ok = _run_admit_decision(
                pool_path, thresholds, warehouse_root,
                name, r.get("factor_info") or {}, tier, ic_mean, icir_nw,
                label_frame=label_frame,
            )
            if ok:
                admitted += 1
        print(f"\n[准入汇总] {admitted}/{len(_admit_queue)} 个因子进入候选池"
              + (f"（seed-quota 每家族 {args.seed_quota} 个）" if args.seed_quota else ""))
    else:
        print("\n[提示] 未指定 --admit，仅输出深评报告（候选池写入需 --admit）")

    print(f"\n[完成] 深评模式结束，总耗时 {time.monotonic() - _tm:.1f}s | 运行日志: {_RUN_LOG_PATH}")
    return 0


_RUN_LOG_PATH = ""  # 最近一次运行日志绝对路径（_setup_run_log 写入，供各模式尾部提示）


def _setup_run_log() -> str:
    """初始化每次运行独立的动态运行日志（模块 A/B/C 全覆盖，Windows 终端不可靠时以日志为准）。

    机制：
      - 每次运行生成 runtime/logs/factor_screen_{YYYYmmdd_HHMMSS}.log（历史留存，绝不覆盖）；
      - 同时实时同步写 factor_screen_latest.log（始终为最近一次运行，快速核查用）；
      - 通道1：_Tee 捕获 print 全量输出（终端 + 双日志文件，逐行 flush 防丢失）；
      - 通道2：logging（深评内部 logger 默认 propagate 到 root → 同文件，
              带时间戳/级别，异常 traceback 经 logging.exception 也落盘）。
    返回日志绝对路径。
    """
    log_dir = PROJECT_ROOT / "runtime" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"factor_screen_{ts}.log"
    latest_path = log_dir / "factor_screen_latest.log"

    # 通道1：Tee 三流（终端 + 主日志 + latest 实时同步）
    _log_file = open(log_path, "w", encoding="utf-8")
    _latest_file = open(latest_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_file, _latest_file)
    sys.stderr = _Tee(sys.stderr, _log_file, _latest_file)

    global _RUN_LOG_PATH
    _RUN_LOG_PATH = str(log_path)

    # 通道2：logging（root FileHandler；深评内部 log.info/exception 同源落盘）
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    _fh = logging.FileHandler(log_path, encoding="utf-8")
    _fh.setFormatter(_fmt)
    logging.getLogger().addHandler(_fh)
    logging.getLogger().setLevel(logging.INFO)

    # 头部：命令行回显（复现用）+ 环境信息
    print("=" * 72)
    print(f"  factor_screen.py 运行日志  |  {ts}")
    print(f"  命令: {sys.argv[0]} {' '.join(sys.argv[1:])}")
    import platform
    print(f"  Python: {platform.python_version()}  |  Qlib 数据: {QLIB_DATA_DIR}")
    print(f"  日志文件: {log_path}")
    print("=" * 72)
    return str(log_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="统一因子筛选引擎（家族配置驱动，沪深主板）")
    parser.add_argument("--family", default=None, help="仅筛选指定家族 quality/value/momentum_price/sentiment/risk")
    parser.add_argument("--all", action="store_true", help="筛选全部五大家族")
    parser.add_argument("--instruments", default=CONFIG["instruments"])
    parser.add_argument("--start-time", default=CONFIG["start_time"])
    parser.add_argument("--end-time", default=CONFIG["end_time"])
    parser.add_argument("--top-k", type=int, default=CONFIG["top_k"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--redundancy-threshold", type=float, default=CONFIG["redundancy_threshold"])
    parser.add_argument("--enable-permutation", action="store_true", help="开启置换检验（默认关）")
    parser.add_argument("--period-windows", "--rolling-windows", dest="period_windows", default=None,
                        help="多周期分段稳健性检验（旧名 --rolling-windows，保留兼容）："
                             "如 2023,2024,2025（全年窗口）或 2023-01-01/2023-12-31,2024-01-01/2024-12-31。"
                             "每窗口独立粗筛+质检，跨窗口聚合看因子在不同时期是否都稳定——"
                             "非时序滚动，与深评 --walk-forward（时序外推）不同")
    parser.add_argument("--min-window-ratio", type=float, default=CONFIG["min_window_ratio"],
                        help="跨窗口聚合阈值：因子在 >= 该比例窗口通过才入选（默认 0.5）")
    parser.add_argument("--enable-embedded", action="store_true", help="开启树模型嵌入法精选（LGBM gain，默认关）")
    parser.add_argument("--enable-factor-neutralize", action="store_true",
                        help="开启因子中性化（Ridge 剥离行业+市值后重算 IC，默认关）")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    # ── [单脚本化 2026-10-12] 深评 + 准入模式（模块 B/C）──────────────────
    g_deep = parser.add_mutually_exclusive_group()
    g_deep.add_argument("--deep-eval", action="store_true",
                        help="深评模式：对 --factor / --factor-list / --family 名单执行完整深评"
                             "（分层回测/衰减/稳健性子池/walk-forward，仅统计、不建树模型）")
    g_deep.add_argument("--admit", action="store_true",
                        help="深评 + 三关准入：评测合格（core/satellite）因子直接写候选池（Alpha Book v2）")
    g_deep.add_argument("--admit-only", action="store_true",
                        help="仅准入：对候选名单执行三关准入（跳过重新深评，取 registry 已有 IC 指标）")
    parser.add_argument("--factor", "-f", default=None, help="深评单个因子（因子库名称）")
    parser.add_argument("--factor-list", default=None,
                        help="深评/准入名单文件（txt 每行一个因子名；CSV 优先 factor_name 列）——"
                             "粗筛名单 → 深评 → 准入 一条命令全链")
    parser.add_argument("--tier", default=None, choices=["core", "satellite", "archive"],
                        help="准入档位（默认取深评 qual_result.tier，非 core/satellite 不入池）")
    parser.add_argument("--walk-forward", action="store_true",
                        help="深评启用 Walk-Forward 估计-外推验证：train 段估计因子方向签名，"
                             "valid 段按签名方向检验外推实绩（机构口径；含 purge/embargo 防泄漏）")
    parser.add_argument("--pool", default=None, help="深评股票池（默认 main_board，与训练端对齐）")
    parser.add_argument("--recompute", action="store_true",
                        help="深评强制从 ClickHouse 重算因子数据（否则仓库优先）")
    parser.add_argument("--seed-quota", type=int, default=0,
                        help="首批建池种子配额（每家族最多 N 个，按|ICIR_NW|降序；仅候选池为空时生效，"
                             "池非空走正常增量三关）。0=不启用（机构 seed alpha 策略：防首批全堆一族）")
    args = parser.parse_args(argv)

    cfg = dict(CONFIG)
    cfg.update(vars(args))
    if args.enable_permutation:
        cfg["permutation_test"] = dict(CONFIG["permutation_test"], enabled=True)
    if args.enable_embedded:
        cfg["embedded_selection"] = dict(CONFIG["embedded_selection"], enabled=True)
    if args.enable_factor_neutralize:
        cfg["factor_neutralization"] = dict(CONFIG["factor_neutralization"], enabled=True)
    if args.period_windows:
        cfg["rolling_windows"] = args.period_windows

    # ── 运行日志：每次运行独立动态日志（模块 A/B/C 全覆盖）──
    # 历史：runtime/logs/factor_screen_{ts}.log；最新：factor_screen_latest.log
    _log_path = _setup_run_log()
    _t0 = time.monotonic()

    # ── [单脚本化 2026-10-12] 深评 + 准入模式（模块 B/C）────────────────
    # 机构范式：单因子评测（Phase 1 深评）与三关准入（Phase 2 入候选池）同一脚本一气呵成。
    # 此模式不走家族筛选（模块 A），直接 收集因子 → 深评 → 准入。
    if args.deep_eval or args.admit or args.admit_only:
        return _run_deep_eval_mode(args, cfg)

    if args.family and args.family not in FAMILY_DEFS:
        print(f"[错误] 未知家族: {args.family}，可选 {list(FAMILY_DEFS)}")
        sys.exit(1)
    # 默认全家族（未指定 --family/--all 时五大家族全部筛选）
    families = [args.family] if args.family else list(FAMILY_DEFS)

    # 日志已由 _setup_run_log 统一接管（runtime/logs/，模块 A/B/C 全覆盖）

    print("=" * 66)
    print("  统一因子筛选引擎 · 家族配置驱动")
    print("  股票池: {}（沪深主板，仅可交易范围）".format(cfg["instruments"]))
    print("  时间  : {} ~ {}".format(cfg["start_time"], cfg["end_time"]))
    print("  家族  : {}".format(", ".join(families)))
    print("=" * 66)

    # 1. Qlib 初始化
    print("\n[1] 初始化 Qlib...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading")
    print(f"  [1] Qlib 初始化完成（{time.monotonic() - _t0:.1f}s）")

    # 2. 按家族加载因子
    print("\n[2] 加载因子库（按家族）...")
    family_factors: dict[str, list[dict]] = {}
    all_factor_names: list[str] = []
    name_expr_map: dict = {}
    for fam in families:
        fs = load_family_factors(fam)
        family_factors[fam] = fs
        for f in fs:
            name_expr_map[f["name"]] = f["expression"]
        all_factor_names.extend(f["name"] for f in fs)
        print(f"    {fam:<16s}: {len(fs):3d} 个因子"
              + ("（⚠ 0 因子，需先接入数据源）" if not fs else ""))
    if not all_factor_names:
        print("[错误] 未加载到任何因子")
        sys.exit(1)
    print(f"  [2] 因子库加载完成（{time.monotonic() - _t0:.1f}s）")

    # 3. 构建全局特征缓存（仅覆盖本批家族所需字段）
    print(f"\n[3] 构建特征缓存...")
    categories_for_bundle = {fam: fs for fam, fs in family_factors.items()}
    global_bundle = build_global_bundle(categories_for_bundle, cfg["label_expr"], cfg["label_name"])
    feature_cache = build_custom_feature_cache(
        instruments=cfg["instruments"],
        feature_bundle=global_bundle,
        factor_cache_names=[],
        start_time=cfg["start_time"], end_time=cfg["end_time"],
        use_dynamic_filter=False,
    )
    print(f"  [3] 特征缓存构建完成（{time.monotonic() - _t0:.1f}s）")

    # 4. 标签 + 可交易掩码（沪深主板 + ST/次新过滤）
    # 标签 DK_L 管线（与 select_factors / train_tree-doubao 逐位对齐）：
    #   CSNeutralize(行业+市值, Ridge) → CSQuantileNorm(截面分位化)
    # 筛选端 IC 与训练端标签口径一致（剥离风格暴露的纯 alpha 标签）。
    print("\n[4] 加载标签与可交易掩码（DK_L 管线：CSNeutralize 行业+市值 → CSQuantileNorm）...")
    label_series = _load_label_series(feature_cache, cfg["label_expr"], cfg["label_name"],
                                      cfg["start_time"], cfg["end_time"])
    dates = label_series.index.get_level_values("datetime").unique()
    tradeable_mask = _build_tradeable_mask(
        feature_cache, dates,
        filter_new_stocks=cfg["filter_new_stocks"], filter_st=cfg["filter_st"],
    )
    label_series = _apply_dk_label(label_series, tradeable_mask)
    print(f"  [4] 标签与掩码加载完成（{time.monotonic() - _t0:.1f}s）")

    # 5. 逐家族筛选（单窗口全时段 / 多周期分段窗口两种模式）
    print("\n[5] 逐家族执行差异化质检...")
    _t5 = time.monotonic()
    card_frames: list[pd.DataFrame] = []
    candidates_by_family: dict[str, list[str]] = {}
    family_detail: dict[str, dict] = {}
    window_cards: dict[str, list[pd.DataFrame]] = {}   # 滚动模式：{window: [card...]}
    window_passed: dict[str, dict[str, list[str]]] = {}  # {window: {family: [names]}}

    rolling_windows = _parse_rolling_windows(cfg.get("rolling_windows"))
    mode_desc = "多周期分段稳健性检验（{} 个窗口，跨窗口聚合 ≥{}）".format(
        len(rolling_windows), cfg["min_window_ratio"]) if rolling_windows \
        else "单窗口全时段（{} ~ {}）".format(cfg["start_time"], cfg["end_time"])
    print(f"  [筛选模式] {mode_desc}")

    if not rolling_windows:
        # ── 单窗口全时段模式 ──
        for fam in families:
            _tf = time.monotonic()
            card, passed = _screen_family_in_window(
                fam, family_factors[fam], feature_cache, name_expr_map, label_series,
                cfg["start_time"], cfg["end_time"], cfg, tradeable_mask,
                all_factor_names, family_factors,
            )
            print(f"    [{fam}] 质检完成（{time.monotonic() - _tf:.1f}s）")
            card_frames.append(card)
            candidates_by_family[fam] = passed
            family_detail[fam] = {"total": len(family_factors[fam]),
                                  "screened": len(passed), "passed": len(passed)}
        print(f"  [5] 逐家族筛选完成（{time.monotonic() - _t5:.1f}s）")
    else:
        # ── 多周期分段稳健性检验模式（各窗口独立，互不污染）──
        # 每个窗口独立执行"粗筛→质检链→(嵌入法/置换/中性化)"，窗口间互不污染；
        # 跨窗口聚合：因子在 >= min_window_ratio 的窗口通过才最终入选（对齐
        # select_factors.aggregate_across_windows 语义，防单窗口偶然性）。
        for win_idx, (win_start, win_end, win_name) in enumerate(rolling_windows, 1):
            _tw = time.monotonic()
            print(f"\n=== 分段窗口 {win_idx}/{len(rolling_windows)}: {win_name} "
                  f"({win_start} ~ {win_end}) ===")
            win_label = _load_label_series(feature_cache, cfg["label_expr"], cfg["label_name"],
                                           win_start, win_end)
            # [P3] 滚动窗口边界治理：embargo=剔除窗口首 10 交易日（上窗口冷却）、
            #       purge=剔除窗口尾 5 交易日（5 日标签越界，防跨窗口信息重叠）
            n_before = win_label.index.get_level_values("datetime").nunique()
            win_label = _purge_embargo_label(
                win_label, purge_days=cfg["wf_purge_days"], embargo_days=cfg["wf_embargo_days"])
            n_after = win_label.index.get_level_values("datetime").nunique()
            print(f"  [边界治理] {win_name}: 交易日 {n_before} → {n_after} "
                  f"(purge {cfg['wf_purge_days']}日 / embargo {cfg['wf_embargo_days']}日)")
            win_dates = win_label.index.get_level_values("datetime").unique()
            win_mask = _build_tradeable_mask(
                feature_cache, win_dates,
                filter_new_stocks=cfg["filter_new_stocks"], filter_st=cfg["filter_st"],
            )
            win_label = _apply_dk_label(win_label, win_mask)
            window_cards[win_name] = []
            window_passed[win_name] = {}
            for fam in families:
                card, passed = _screen_family_in_window(
                    fam, family_factors[fam], feature_cache, name_expr_map, win_label,
                    win_start, win_end, cfg, win_mask,
                    all_factor_names, family_factors,
                )
                card["window"] = win_name
                window_cards[win_name].append(card)
                window_passed[win_name][fam] = passed
            card_frames.extend(window_cards[win_name])
            print(f"  [窗口 {win_name}] 完成（{time.monotonic() - _tw:.1f}s）")
        print(f"  [5] 多窗口筛选完成（{time.monotonic() - _t5:.1f}s）")

        # 跨窗口聚合：因子在 >= min_window_ratio 的窗口通过才最终入选
        n_windows = len(rolling_windows)
        min_wins = max(1, int(np.ceil(n_windows * cfg["min_window_ratio"])))
        print(f"\n[跨窗口聚合] 阈值 {cfg['min_window_ratio']} → 需在 ≥{min_wins}/{n_windows} 窗口通过")
        agg_count: dict[str, int] = {}
        for fam in families:
            for win_name in window_passed:
                for n in window_passed[win_name].get(fam, []):
                    agg_count[n] = agg_count.get(n, 0) + 1
        for fam in families:
            candidates_by_family[fam] = [
                n for n, cnt in agg_count.items()
                if cnt >= min_wins and n in [f["name"] for f in family_factors[fam]]
            ]
            print(f"    {fam:<16s}: {len(candidates_by_family[fam])} 个因子跨窗口稳定通过")
        family_detail["_rolling_windows"] = {
            "windows": [w[2] for w in rolling_windows],
            "min_window_ratio": cfg["min_window_ratio"],
            "n_window_passed": agg_count,
        }

    # 6. 全局跨家族去冗余（保留优先级）
    # [量价移植] 显示实际生效阈值（与 _global_dedup 同口径：各家族 special.redundancy_global 最小值）
    _eff_global = cfg["global_redundancy_threshold"]
    for _fam, _names in candidates_by_family.items():
        if _names:
            _ft = FAMILY_DEFS.get(_fam, {}).get("special", {}).get("redundancy_global")
            if _ft is not None and _ft < _eff_global:
                _eff_global = _ft
    print(f"\n[6] 全局跨家族去冗余（阈值 {_eff_global}，优先级 基本面>预期>量价>风险）...")
    final_names = _global_dedup(candidates_by_family, feature_cache,
                                cfg["start_time"], cfg["end_time"], cfg)
    removed_global = [n for fam, names in candidates_by_family.items() for n in names
                      if n not in final_names]
    print(f"    [全局去冗余] 入选 {len(final_names)} 个（剔除 {len(removed_global)} 个跨家族冗余）")
    # 7. 家族分布（信息展示）
    # [关注点分离] 筛查端只负责"单因子是否有效"，不施加类别配额；
    # 不同类别因子的配额约束下移至组合/训练阶段（见 train_tree-doubao.py 读取候选池后）。
    final_by_family: dict[str, list[str]] = {fam: [] for fam in families}
    for n in final_names:
        for fam in families:
            if n in [f["name"] for f in family_factors.get(fam, [])]:
                final_by_family[fam].append(n)
                break
    card_df = pd.concat(card_frames, ignore_index=True) if card_frames else pd.DataFrame()
    print("\n[7] 家族分布（配额约束已下移至组合/训练阶段）:")
    for fam in families:
        cnt = len(final_by_family[fam])
        q = FAMILY_DEFS[fam].get("quota")
        q_str = f"参考配额 {q[0]}-{q[1]}" if q else "无配额（风控）"
        print(f"    {fam:<16s}: {cnt:3d} 个  {q_str}")
    print(f"    合计: {sum(len(v) for v in final_by_family.values())} 个")

    # 8. 输出
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg["output_dir"]
    card_path = os.path.join(out_dir, f"family_screen_card_{ts}.csv")
    if not card_df.empty:
        card_df.to_csv(card_path, index=False, encoding="utf-8-sig")
        print(f"\n  粗筛卡已保存: {card_path}")

    result = {
        "meta": {"updated_at": ts, "instruments": cfg["instruments"],
                 "start_time": cfg["start_time"], "end_time": cfg["end_time"],
                 "families": families, "family_defs": {f: FAMILY_DEFS[f]["label"] for f in families}},
        "family_detail": family_detail,
        "candidates_by_family": {f: final_by_family[f] for f in families},
        "global_dedup_removed": removed_global,
    }
    json_path = os.path.join(out_dir, f"family_candidates_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  候选清单已保存: {json_path}")

    print("\n" + "=" * 66)
    print(f"  完成：共入选 {len(final_names)} 个因子（{len(families)} 家族）")
    print(f"  总耗时: {time.monotonic() - _t0:.1f}s")
    print(f"  运行日志: {_log_path}")
    print("  提示：粗筛名单 → 深评 → 三关准入一条命令：")
    print("        factor_screen.py --factor-list <粗筛名单> --deep-eval --admit")
    print("=" * 66)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # sys.exit(1) 正常路径，不兜底
    except Exception:
        # 未捕获异常：完整 traceback 落运行日志（Tee 已激活），退出码非零便于脚本级核查
        import traceback
        logging.getLogger("factor_screen").exception("未捕获异常，运行终止")
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        # 双保险：确保 latest 与主日志一致（Tee 已实时同步，此处兜底收尾）
        try:
            import shutil
            if _RUN_LOG_PATH and os.path.exists(_RUN_LOG_PATH):
                shutil.copyfile(_RUN_LOG_PATH,
                                os.path.join(os.path.dirname(_RUN_LOG_PATH),
                                             "factor_screen_latest.log"))
        except Exception:
            pass
