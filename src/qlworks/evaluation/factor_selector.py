"""
因子筛选：三级等级评定 + 注册表管理 + 候选池。

数据导出已拆分到 factor_store.link_factor_to_tier()，
这里只负责等级判定和状态管理，不做数据文件写入。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .config import EvalConfig, DEFAULT_CONFIG
from .factor_def import LifecycleStage
from .lifecycle import LifecycleManager


def evaluate_qualification(ic_stats, ls_stats, config=None,
                           decay_df=None, turnover_stats=None,
                           coverage_pct=1.0):
    """综合评估因子等级，返回合格判定和分类信息。

    增强评分公式（满分100分）：
      IC 均值        × 20  （基础预测力）
      ICIR           × 15  （预测稳定性）
      胜率           × 15  （方向正确率）
      多空年化收益   × 15  （实际选股收益）
      多空夏普       × 10  （收益风险比）
      IC 衰减速度    × 10  （信号持久性）
      换手率         × 5   （交易成本）
      数据覆盖率     × 10  （样本代表性）
    满分 = 100

    Returns:
        {"qualified": bool, "tier": str, "reasons": [str],
         "scores": {str: float}, "composite_score": float}
        tier: "core" | "satellite" | "archive"
    """
    if config is None:
        config = DEFAULT_CONFIG

    reasons = []
    scores = {}

    # 1. IC 均值（权重 20）
    ic_mean = abs(ic_stats.get("ic_mean", 0))
    scores["ic"] = min(ic_mean / config.ic_threshold, 1.0) if config.ic_threshold > 0 else 0
    if ic_mean < config.ic_threshold:
        reasons.append(f"IC {ic_mean:.4f} < {config.ic_threshold}")

    # 2. ICIR（权重 15）
    icir = abs(ic_stats.get("icir", 0))
    scores["icir"] = min(icir / config.icir_threshold, 1.0) if config.icir_threshold > 0 else 0
    if icir < config.icir_threshold:
        reasons.append(f"ICIR {icir:.4f} < {config.icir_threshold}")

    # 3. 胜率（权重 15）
    wr = ic_stats.get("win_rate", 0)
    scores["win_rate"] = min(wr / config.win_rate_threshold, 1.0) if config.win_rate_threshold > 0 else 0
    if wr < config.win_rate_threshold:
        reasons.append(f"胜率 {wr:.2%} < {config.win_rate_threshold:.0%}")

    # 4. 多空年化收益（权重 15）
    ls_ret = ls_stats.get("annual_return", 0)
    threshold_ret = config.ls_annual_return_threshold * 100
    scores["ls_return"] = min(max(ls_ret / threshold_ret, 0), 1.0) if threshold_ret > 0 else 0
    if ls_ret < threshold_ret:
        reasons.append(f"多空年化 {ls_ret:.2f}% < {threshold_ret:.0f}%")

    # 5. 多空夏普（权重 10）
    ls_sharpe = ls_stats.get("sharpe", 0)
    scores["ls_sharpe"] = min(max(ls_sharpe / config.ls_sharpe_threshold, 0), 1.0) if config.ls_sharpe_threshold > 0 else 0
    if ls_sharpe < config.ls_sharpe_threshold:
        reasons.append(f"多空夏普 {ls_sharpe:.4f} < {config.ls_sharpe_threshold}")

    # 6. IC 衰减速度（权重 10）— 信号越持久越好
    decay_score = 1.0
    if decay_df is not None and not decay_df.empty and len(decay_df) > 1:
        first_ic = abs(decay_df["ic_mean"].iloc[0])
        last_ic = abs(decay_df["ic_mean"].iloc[-1])
        if first_ic > 1e-8:
            decay_ratio = last_ic / first_ic
            decay_score = min(decay_ratio * 2, 1.0)  # 保留50%以上给满分
    scores["decay"] = decay_score
    if decay_score < 0.3:
        reasons.append(f"IC衰减过快({decay_score:.2f}), 信号不持久")

    # 7. 换手率惩罚（权重 5）— 换手越低越好
    turnover_score = 1.0
    if turnover_stats:
        avg_to = turnover_stats.get("avg_turnover", 0)
        turnover_score = max(1.0 - avg_to * 2, 0.0)  # 换手率超过50%开始扣分
    scores["turnover"] = turnover_score
    if turnover_score < 0.5:
        reasons.append(f"换手率过高({turnover_stats.get('avg_turnover',0):.2%}), 交易成本大")

    # 8. 数据覆盖率（权重 10）— 覆盖越广越好
    coverage_score = min(coverage_pct / 0.5, 1.0)  # 覆盖50%以上给满分
    scores["coverage"] = coverage_score
    if coverage_pct < 0.3:
        reasons.append(f"数据覆盖率仅{coverage_pct:.0%}, 样本不足")

    # 加权总分
    weights = {
        "ic": 20, "icir": 15, "win_rate": 15,
        "ls_return": 15, "ls_sharpe": 10,
        "decay": 10, "turnover": 5, "coverage": 10,
    }
    composite_score = round(sum(
        scores.get(k, 0) * w for k, w in weights.items()
    ), 2) if scores else 0

    n_failed = len(reasons)

    if n_failed == 0:
        tier = "core"
        qualified = True
    elif composite_score >= config.satellite_composite_min:
        tier = "satellite"
        qualified = False
    else:
        tier = "archive"
        qualified = False

    return {
        "qualified": qualified,
        "tier": tier,
        "reasons": reasons,
        "scores": scores,
        "composite_score": composite_score,
    }


def update_factor_tier(name: str, tier: str, config: Optional[EvalConfig] = None):
    """
    更新因子等级引用（不复制数据，只写引用标记）。
    替代原 export_factor_by_status。

    实际数据写入由 factor_store.FactorStore.link_factor_to_tier() 完成。
    """
    from .factor_store import FactorStore
    cfg = config or DEFAULT_CONFIG
    store = FactorStore(cfg)
    return store.link_factor_to_tier(name, tier)


def update_factor_registry(registry_path, factor_name, eval_result, ic_stats, ls_stats,
                           lifecycle_manager=None):
    """更新因子注册表，记录评测结果、生命周期阶段。"""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"factors": {}, "last_updated": "", "version": "3.0"}

    prev = registry["factors"].get(factor_name, {})
    prev_lifecycle = prev.get("lifecycle_stage", "")
    tier = eval_result.get("tier", "archive")

    tier_to_stage = {
        "core": LifecycleStage.ACTIVE,
        "satellite": LifecycleStage.EXPLORATION,
        "archive": LifecycleStage.ARCHIVED,
    }
    new_stage = tier_to_stage.get(tier, LifecycleStage.EXPLORATION)

    vh = prev.get("version_history", [])
    vh.append({
        "eval_date": str(datetime.now()),
        "tier": tier,
        "composite_score": eval_result.get("composite_score", 0),
        "ic_mean": ic_stats.get("ic_mean"),
        "icir": ic_stats.get("icir"),
    })

    registry["factors"][factor_name] = {
        "tier": tier,
        "lifecycle_stage": new_stage,
        "version_history": vh,
        "composite_score": eval_result.get("composite_score", 0),
        "last_eval_date": str(datetime.now()),
        "ic_mean": ic_stats.get("ic_mean"),
        "icir": ic_stats.get("icir"),
        "win_rate": ic_stats.get("win_rate"),
        "ls_annual_return": ls_stats.get("annual_return"),
        "ls_sharpe": ls_stats.get("sharpe"),
        "ls_max_drawdown": ls_stats.get("max_drawdown"),
        "reasons": eval_result.get("reasons", []),
    }
    registry["last_updated"] = str(datetime.now())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    if lifecycle_manager and prev_lifecycle != new_stage:
        lifecycle_manager.transition(factor_name, prev_lifecycle or "none", new_stage,
                                      reason=f"等级={tier}, 评分={eval_result.get('composite_score', 0):.1f}")


def handle_candidate_pool_entry(factor_name, ic_stats, ls_stats, config=None):
    """处理候选因子池。"""
    from .candidate_pool import CandidatePool
    cfg = config or DEFAULT_CONFIG
    pool = CandidatePool(cfg.registry_dir)
    metrics = {
        "ic_mean": abs(ic_stats.get("ic_mean", 0)),
        "ic_positive_ratio": ic_stats.get("ic_positive_ratio", ic_stats.get("win_rate", 0)),
        "ir": ic_stats.get("icir", 0),
        "ic_std": ic_stats.get("ic_std", 0),
        "sharpe": ls_stats.get("sharpe", 0),
        "monotonicity": ls_stats.get("monotonicity", 0),
        "missing_rate": 0.0, "n_years": 5.0, "valid_pct": 1.0,
    }
    screening = pool.full_screening(metrics)
    pool.add_candidate(factor_name, metrics, screening)