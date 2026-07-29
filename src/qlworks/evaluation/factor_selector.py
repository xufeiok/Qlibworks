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


def calc_scenario_robustness(scenario_results: Optional[dict] = None) -> tuple:
    """计算场景稳健性得分 [0, 1]。

    评估因子在不同市场环境下的表现一致性：
      - 分市值一致性（大/中/小盘）
      - 牛熊市鲁棒性（牛市赚钱、熊市不亏钱）
      - 行业板块普适性（覆盖多少个板块）

    注意：当数据缺失时返回 0.5（中性分）而非 1.0，
    避免默认满分掩盖真实问题。

    Returns:
        (score_0_to_1, reasons_list)
    """
    if not scenario_results:
        return 0.5, ["场景压力测试数据不可用，未参与评分（默认 0.5 中性分）"]

    reasons = []
    penalties = []  # each element is a deduction ratio [0, 1]

    # 1. 分市值一致性（权重 40%）
    mc_df = scenario_results.get('market_cap_ic', pd.DataFrame())
    if isinstance(mc_df, pd.DataFrame) and not mc_df.empty and len(mc_df) >= 2:
        ic_vals = [abs(r.get('ic_mean', 0)) for _, r in mc_df.iterrows()]
        min_ic = min(ic_vals)
        max_ic = max(ic_vals)
        # 检查是否有负 IC（方向反转）
        has_negative = any(r.get('ic_mean', 0) < 0 for _, r in mc_df.iterrows())
        if has_negative:
            penalties.append(0.6)
            reasons.append("分市值检验中存在 IC 为负的市值组（因子方向不一致）")
        elif max_ic > 1e-8:
            ratio = min_ic / max_ic
            if ratio > 0.5:
                penalties.append(1.0)
            elif ratio > 0.3:
                penalties.append(0.7)
                reasons.append(f"分市值 IC 差异较大（最小/最大={ratio:.2f}），不同市值组表现存在分化")
            else:
                penalties.append(0.4)
                reasons.append(f"分市值 IC 差异大（最小/最大={ratio:.2f}），因子在不同市值股票上表现不一致")
        else:
            penalties.append(1.0)
    else:
        penalties.append(0.5)  # 无数据 → 中性
        reasons.append("分市值检验数据不可用")

    # 2. 牛熊市鲁棒性（权重 40%）
    #   - 检查熊市 IC 是否有效
    #   - 同时检查熊市多空收益是否为负（IC 为正但 LS 为负说明排序稳定性差）
    regime_df = scenario_results.get('market_regime', pd.DataFrame())
    if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
        bear_rows = regime_df[regime_df['regime'] == '熊市']
        bull_rows = regime_df[regime_df['regime'] == '牛市']
        if not bear_rows.empty:
            bear_ic = bear_rows['ic_mean'].mean()
            # 熊市 IC 方向校验
            if bear_ic < 0:
                penalties.append(0.3)
                reasons.append(f"熊市因子 IC 为负（{bear_ic:.4f}），因子在熊市完全失效")
            elif bear_ic < 0.005:
                penalties.append(0.6)
                reasons.append(f"熊市因子 IC 接近零（{bear_ic:.4f}），因子在熊市无预测力")
            else:
                # IC 为正，但还要看熊市下的 LS 收益
                if 'ls_annual_return' in bear_rows.columns:
                    bear_ls = bear_rows['ls_annual_return'].mean()
                    if bear_ls < -5:
                        penalties.append(0.5)
                        reasons.append(f"熊市 IC 为正（{bear_ic:.4f}）但多空收益为负（{bear_ls:.1f}%），排序稳定性差")
                    elif bear_ls < 0:
                        penalties.append(0.7)
                    else:
                        penalties.append(1.0)
                else:
                    penalties.append(1.0)
        else:
            penalties.append(0.5)
            reasons.append("回测区间内无熊市时段数据")

        if not bull_rows.empty and not bear_rows.empty:
            bull_ic = bull_rows['ic_mean'].mean()
            bear_ic = bear_rows['ic_mean'].mean()
            if bull_ic > 1e-8 and bear_ic / bull_ic < 0.3:
                reasons.append(f"熊市 IC/牛市 IC = {bear_ic/bull_ic:.2f}，因子过度依赖牛市行情")
    else:
        penalties.append(0.5)
        reasons.append("牛熊市分段检验数据不可用")

    # 3. 行业板块普适性（权重 20%）
    sector_df = scenario_results.get('industry_sector', pd.DataFrame())
    if isinstance(sector_df, pd.DataFrame) and not sector_df.empty and len(sector_df) >= 2:
        n_positive = (sector_df['ic_mean'] > 0).sum()
        n_total = len(sector_df)
        ratio = n_positive / n_total
        if ratio >= 0.8:
            penalties.append(1.0)
        elif ratio >= 0.5:
            penalties.append(0.7)
            reasons.append(f"因子在 {n_positive}/{n_total} 个板块 IC 为正，行业通用性一般")
        else:
            penalties.append(0.3)
            reasons.append(f"因子仅在 {n_positive}/{n_total} 个板块有效，行业局限性强")
    else:
        penalties.append(0.5)
        reasons.append("行业板块检验数据不可用")

    score = float(np.mean(penalties)) if penalties else 0.5
    return round(score, 4), reasons


def calc_residual_independence(control_results: Optional[dict] = None,
                                ic_mean_original: float = 0.0) -> tuple:
    """计算残差独立性得分 [0, 1]。

    评估因子在剔除已知风险因子（行业/市值）后是否仍有独立预测能力：
      - 残差因子 IC 保留率
      - 双变量分组单调性

    注意：当数据缺失时返回 0.5（中性分）而非 1.0。

    Returns:
        (score_0_to_1, reasons_list)
    """
    if not control_results:
        return 0.5, ["控制变量对冲数据不可用，未参与评分（默认 0.5 中性分）"]

    reasons = []
    penalties = []

    # 1. 残差因子 IC 保留率（权重 60%）
    residual = control_results.get('residual', {})
    if residual:
        res_ic = abs(residual.get('residual_ic_stats', {}).get('ic_mean', 0))
        orig_ic = abs(ic_mean_original)
        if orig_ic > 1e-8:
            retention = res_ic / orig_ic
            if retention > 0.8:
                penalties.append(1.0)
            elif retention > 0.5:
                penalties.append(0.7)
                reasons.append(f"残差因子 IC 保留率 {retention:.0%}，市值/行业解释部分预测力")
            elif retention > 0.2:
                penalties.append(0.4)
                reasons.append(f"残差因子 IC 保留率仅 {retention:.0%}，因子与市值/行业高度相关")
            else:
                penalties.append(0.1)
                reasons.append(f"残差因子 IC 保留率 {retention:.0%}，因子收益几乎完全来自市值/行业暴露")
        else:
            penalties.append(1.0)
    else:
        penalties.append(0.5)
        reasons.append("残差因子检验数据不可用")

    # 2. 双变量分组单调性（权重 40%）
    bv_df = control_results.get('bivariate', pd.DataFrame())
    if isinstance(bv_df, pd.DataFrame) and not bv_df.empty and 'monotonicity' in bv_df.columns:
        mono = bv_df['monotonicity'].mean()
        if mono > 0.5:
            penalties.append(1.0)
        elif mono > 0:
            penalties.append(0.6)
            reasons.append(f"双变量分组单调性偏弱（{mono:.2f}），控制市值后因子区分度下降")
        else:
            penalties.append(0.2)
            reasons.append("双变量分组无单调性，控制市值后因子丧失预测力")
    else:
        penalties.append(0.5)
        reasons.append("双变量分组检验数据不可用")

    score = float(np.mean(penalties)) if penalties else 0.5
    return round(score, 4), reasons


def _classify_reasons(reasons: list, turnover_stats: dict = None,
                       decay_df=None, scenario_results=None) -> tuple:
    """将未达标项分为「核心缺陷」和「特征提示」两类。

    核心缺陷（红色）：影响因子有效性的根本问题 — 否决候选
    特征提示（灰色）：因子固有属性或数据限制 — 仅影响评分，不作为否决项

    Returns:
        (core_failures, characteristic_notes)
    """
    # 被认定为「特征提示」的 reason 前缀列表
    characteristic_patterns = [
        "衰减", "换手", "数据不可用", "数据缺失",
    ]
    core_failures = []
    characteristic_notes = []

    for r in reasons:
        is_char = any(p in r for p in characteristic_patterns)
        if is_char:
            characteristic_notes.append(r)
        else:
            core_failures.append(r)

    return core_failures, characteristic_notes


_RECOMMENDATIONS = {
    "core":      ("★★★ 强烈推荐", "可直接作为多因子核心底仓，纳入主模型"),
    "satellite": ("★★ 推荐", "可作为多因子卫星成分，推荐纳入候选池"),
    "watch":     ("★ 观察", "需进一步验证或改进，暂不推荐入库"),
    "reject":    ("不推荐", "因子无效或存在根本性缺陷，拒绝入库"),
}


def evaluate_qualification(ic_stats, ls_stats, config=None,
                           decay_df=None, turnover_stats=None,
                           coverage_pct=1.0,
                           scenario_results=None,
                           control_results=None,
                           recent_ic_stats=None):  # 近2年 IC 统计（双门槛准入）
    """综合评估因子等级，输出三级推荐（强烈推荐/推荐/观察/不推荐）。

    评分公式（满分100分，10 维评分）：
      IC 均值        × 16  （基础预测力）
      ICIR           × 12  （预测稳定性）
      胜率           × 12  （方向正确率）
      多空年化收益   × 12  （实际选股收益）
      多空夏普       × 8   （收益风险比）
      IC 衰减速度    × 8   （信号持久性，特征提示不否决）
      换手率         × 4   （交易成本，特征提示不否决）
      数据覆盖率     × 8   （样本代表性）
      场景稳健性     × 12  （跨市值/牛熊/行业的表现一致性）
      残差独立性     × 8   （剔除市值/行业后的纯因子预测力）
    满分 = 100

    双门槛准入（近2年 + 全样本）：
      - 近2年 ICIR ≥ 0.3 **且** 全样本 ICIR ≥ 0.15 → 核心因子，高权重
      - 全样本 ICIR < 0.15 **但** 近2年 ICIR ≥ 0.3 → 辅助因子，低权重（satellite）
      - 仅近半年突然走强，长期为负 → 直接淘汰（reject）

    Returns:
        {"qualified": bool, "tier": str, "reasons": [str],
         "core_failures": [str], "characteristic_notes": [str],
         "scores": {str: float}, "composite_score": float,
         "recommendation": str, "recommendation_detail": str}
        tier: "core" | "satellite" | "watch" | "reject"
    """
    if config is None:
        config = DEFAULT_CONFIG

    reasons = []
    scores = {}

    # 1. IC 均值（权重 16）— 核心否决项
    ic_mean = abs(ic_stats.get("ic_mean", 0))
    scores["ic"] = min(ic_mean / config.ic_threshold, 1.0) if config.ic_threshold > 0 else 0
    if ic_mean < config.ic_threshold:
        reasons.append(f"IC {ic_mean:.4f} < {config.ic_threshold}")

    # 跨维度校验：IC 为正但多空收益显著为负 → 因子方向性矛盾，惩罚 IC 评分
    ls_ret = ls_stats.get("annual_return", 0)
    if ic_mean > config.ic_threshold * 0.5 and ls_ret < -2.0:
        ic_penalty = max(1.0 - abs(ls_ret) / 30.0, 0.3)
        scores["ic"] = min(scores["ic"], ic_penalty)
        reasons.append(f"IC 为正（{ic_mean:.4f}）但多空收益为负（{ls_ret:.1f}%），因子方向矛盾")

    # 2. ICIR（权重 12）— 核心否决项
    icir = abs(ic_stats.get("icir", 0))
    scores["icir"] = min(icir / config.icir_threshold, 1.0) if config.icir_threshold > 0 else 0
    if icir < config.icir_threshold:
        reasons.append(f"ICIR {icir:.4f} < {config.icir_threshold}")

    # 3. 胜率（权重 12）— 核心否决项
    wr = ic_stats.get("win_rate", 0)
    scores["win_rate"] = min(wr / config.win_rate_threshold, 1.0) if config.win_rate_threshold > 0 else 0
    if wr < config.win_rate_threshold:
        reasons.append(f"胜率 {wr:.2%} < {config.win_rate_threshold:.0%}")

    # 4. 多空年化收益（权重 12）— 核心否决项
    ls_ret = ls_stats.get("annual_return", 0)
    threshold_ret = config.ls_annual_return_threshold * 100
    scores["ls_return"] = min(max(ls_ret / threshold_ret, 0), 1.0) if threshold_ret > 0 else 0
    if ls_ret < threshold_ret:
        reasons.append(f"多空年化 {ls_ret:.2f}% < {threshold_ret:.0f}%")

    # 5. 多空夏普（权重 8）— 核心否决项
    ls_sharpe = ls_stats.get("sharpe", 0)
    scores["ls_sharpe"] = min(max(ls_sharpe / config.ls_sharpe_threshold, 0), 1.0) if config.ls_sharpe_threshold > 0 else 0
    if ls_sharpe < config.ls_sharpe_threshold:
        reasons.append(f"多空夏普 {ls_sharpe:.4f} < {config.ls_sharpe_threshold}")

    # 6. IC 衰减速度（权重 8）— 特征提示，不否决
    decay_score = 1.0
    if decay_df is not None and not decay_df.empty and len(decay_df) > 1:
        first_ic = abs(decay_df["ic_mean"].iloc[0])
        last_ic = abs(decay_df["ic_mean"].iloc[-1])
        if first_ic > 1e-8:
            decay_ratio = last_ic / first_ic
            decay_score = min(decay_ratio * 2, 1.0)
    scores["decay"] = decay_score
    # 仅当衰减得分极低时添加特征提示（不否决）
    if decay_score < 0.3:
        reasons.append(f"IC衰减偏快({decay_score:.2f})，属于因子策略属性，不影响入库判断")

    # 7. 换手率（权重 4）— 特征提示，不否决
    turnover_score = 1.0
    if turnover_stats:
        avg_to = turnover_stats.get("avg_turnover", 0)
        turnover_score = max(1.0 - avg_to * 2, 0.0)
    scores["turnover"] = turnover_score
    if turnover_score < 0.5:
        reasons.append(f"换手率偏高({turnover_stats.get('avg_turnover',0):.2%})，属于因子策略属性，不影响入库判断")

    # 8. 数据覆盖率（权重 8）
    coverage_score = min(coverage_pct / 0.5, 1.0)
    scores["coverage"] = coverage_score
    if coverage_pct < 0.3:
        reasons.append(f"数据覆盖率仅{coverage_pct:.0%}, 样本不足")

    # ── 9. 场景稳健性（权重 12）──
    scenario_score, scenario_reasons = calc_scenario_robustness(scenario_results)
    scores["scenario_robustness"] = scenario_score
    reasons.extend(scenario_reasons)

    # ── 10. 残差独立性（权重 8）──
    independence_score, ind_reasons = calc_residual_independence(control_results, ic_mean)
    scores["residual_independence"] = independence_score
    reasons.extend(ind_reasons)

    # 加权总分（10 维，满分 100）
    weights = {
        "ic": 16, "icir": 12, "win_rate": 12,
        "ls_return": 12, "ls_sharpe": 8,
        "decay": 8, "turnover": 4, "coverage": 8,
        "scenario_robustness": 12,
        "residual_independence": 8,
    }
    composite_score = round(sum(
        scores.get(k, 0) * w for k, w in weights.items()
    ), 2) if scores else 0

    # 将 reasons 分为「核心缺陷」和「特征提示」
    core_failures, characteristic_notes = _classify_reasons(
        reasons, turnover_stats, decay_df, scenario_results
    )

    # ── 双门槛准入（近2年 + 全样本 ICIR） ──
    # [Citadel Alpha Lab] 解决长短周期冲突，防止短命脉冲因子入库
    if recent_ic_stats is not None:
        recent_icir = abs(recent_ic_stats.get("icir_nw", recent_ic_stats.get("icir", 0)))
        full_icir = abs(ic_stats.get("icir_nw", ic_stats.get("icir", 0)))
        full_ic_mean = abs(ic_stats.get("ic_mean", 0))

        # 情形 2：仅近半年突然走强，长期为负 → 直接淘汰
        if full_ic_mean < 0.01 and recent_icir >= 0.3:
            # 检查全样本 IC 是否长期接近零或为负
            core_failures.append(f"全样本 IC 长期微弱 ({full_ic_mean:.4f})，仅近2年走强 (ICIR={recent_icir:.2f})，属行情脉冲，淘汰")
            composite_score = min(composite_score, config.watch_composite_min - 1)

        # 情形 1：全样本偏弱但近2年稳定 → 降级为辅助因子
        elif full_icir < 0.15 and recent_icir >= 0.3:
            core_failures.append(f"全样本 ICIR={full_icir:.2f} < 0.15，近2年 ICIR={recent_icir:.2f} ≥ 0.3，定位辅助因子、低权重")
            # 强制降级到 satellite（不高于 satellite）
            if composite_score >= config.satellite_composite_min:
                composite_score = config.satellite_composite_min - 1

        # 情形 3：双达标 → 加分鼓励
        elif full_icir >= 0.15 and recent_icir >= 0.3:
            characteristic_notes.append(f"双门槛达标：全样本 ICIR={full_icir:.2f} ≥ 0.15，近2年 ICIR={recent_icir:.2f} ≥ 0.3")
            composite_score = min(composite_score + 2, 100)

    # 等级判定：三档推荐 + watchdog
    n_core = len(core_failures)

    if n_core == 0 and composite_score >= 90:
        tier = "core"
    elif composite_score >= config.satellite_composite_min:
        tier = "satellite"
    elif composite_score >= config.watch_composite_min:
        tier = "watch"
    else:
        tier = "reject"

    # qualified=True 仅用于核心/卫星（推荐入库）
    qualified = tier in ("core", "satellite")
    rec_label, rec_detail = _RECOMMENDATIONS.get(tier, ("不推荐", ""))

    return {
        "qualified": qualified,
        "tier": tier,
        "reasons": reasons,
        "core_failures": core_failures,
        "characteristic_notes": characteristic_notes,
        "scores": scores,
        "composite_score": composite_score,
        "recommendation": rec_label,
        "recommendation_detail": rec_detail,
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