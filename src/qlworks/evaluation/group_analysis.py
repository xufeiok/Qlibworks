"""
分层回测分析：分组收益率、单调性检验、多空组合。
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── 默认参数（可统一修改） ──
_N_GROUPS: int = 10          # 分层回测组数
_QR_WINDOW: int = 5          # 分层收益计算窗口
_QR_MIN_OBS: int = 20        # 每组最小观测数


def safe_quantile_assign(series, n=5):
    """用 rank 分位数替代 pd.qcut，避免重复边界报错。"""
    ranks = series.rank(method="first")
    result = np.floor((ranks - 1) / len(series) * n).clip(0, n - 1).astype(int)
    return result


# ──────────── Q1 vs Q10 差异显著性检验 ────────────

def calc_q1_q10_significance(q_df: pd.DataFrame, label_col: str = "mean") -> dict:
    """检验 Q1（最低组）与 Q10（最高组）的收益率差异是否显著。

    使用两种方法：
      1. Welch's t-test：两组均值差异的 t 检验（不假设方差齐性）
      2. Mann-Whitney U 检验：非参数秩和检验（不假设正态分布）

    如果两者都显著（p < 0.05），说明因子的分层能力是统计显著的，
    不是随机噪声。

    Args:
        q_df: 分层回测结果（含 datetime, quantile, mean）
        label_col: 收益列名（默认 mean）

    Returns:
        dict: t_stat, t_pvalue, mw_stat, mw_pvalue,
              q1_mean, q10_mean, diff_mean,
              t_significant, mw_significant
    """
    from scipy import stats

    if q_df.empty or "quantile" not in q_df.columns:
        return {"note": "数据为空", "t_significant": False, "mw_significant": False}

    q_min = int(q_df["quantile"].min())
    q_max = int(q_df["quantile"].max())

    q1 = q_df[q_df["quantile"] == q_min][label_col].dropna().values
    q10 = q_df[q_df["quantile"] == q_max][label_col].dropna().values

    if len(q1) < 5 or len(q10) < 5:
        return {"note": "样本不足", "t_significant": False, "mw_significant": False,
                "q1_n": len(q1), "q10_n": len(q10)}

    # Welch's t-test
    t_stat, t_pvalue = stats.ttest_ind(q1, q10, equal_var=False)

    # Mann-Whitney U
    mw_stat, mw_pvalue = stats.mannwhitneyu(q1, q10, alternative="two-sided")

    return {
        "q1_mean": round(float(np.mean(q1)), 6),
        "q10_mean": round(float(np.mean(q10)), 6),
        "diff_mean": round(float(np.mean(q1) - np.mean(q10)), 6),
        "q1_std": round(float(np.std(q1, ddof=1)), 6),
        "q10_std": round(float(np.std(q10, ddof=1)), 6),
        "t_stat": round(float(t_stat), 4),
        "t_pvalue": round(float(t_pvalue), 6),
        "t_significant": bool(t_pvalue < 0.05),
        "mw_stat": round(float(mw_stat), 4),
        "mw_pvalue": round(float(mw_pvalue), 6),
        "mw_significant": bool(mw_pvalue < 0.05),
        "q1_n": int(len(q1)),
        "q10_n": int(len(q10)),
    }


def quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    quantiles: int = _N_GROUPS,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """每日分层收益率（使用 rank 分位数，更稳定）。

    返回 DataFrame with columns: datetime, quantile, mean, count, instruments
    """
    rows = []
    for dt, grp in df.groupby("datetime"):
        f = grp[factor_col]
        l = grp[label_col]
        # 过滤 inf/-inf
        l = l.replace([np.inf, -np.inf], np.nan)
        valid = f.notna() & l.notna()
        if valid.sum() < quantiles * 2:
            continue

        instrument_list = grp.get("instrument", pd.Series(index=grp.index))
        try:
            if group_col and group_col in grp.columns:
                q = grp.groupby(group_col)[factor_col].transform(
                    lambda x: safe_quantile_assign(x, quantiles)
                )
            else:
                q = pd.Series(index=grp.index, dtype="Int64")
                q[valid] = safe_quantile_assign(f[valid], quantiles)

            result = pd.DataFrame({
                "quantile": q,
                "return": l,
                "instrument": instrument_list,
            })
            stats = result.groupby("quantile")["return"].agg(["mean", "count"]).reset_index()
            stats["datetime"] = dt
            # 记录每层的成分股列表用于精确换手率
            stats["instruments"] = result.groupby("quantile")["instrument"].apply(list).values
            rows.append(stats)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["datetime", "quantile", "mean", "count"])
    return pd.concat(rows, ignore_index=True)


def calc_monotonicity_score(q_df: pd.DataFrame) -> float:
    """计算分层单调性得分 [-1, 1]。

    q_df 包含 datetime / quantile / mean / count 四列（扁平格式）。
    """
    scores = []
    for dt, grp in q_df.groupby("datetime"):
        # 按 quantile 排序
        means = grp.sort_values("quantile")["mean"].values
        if len(means) < 2:
            continue
        diffs = np.diff(means)
        n_pos = (diffs > 0).sum()
        n_neg = (diffs < 0).sum()
        total = n_pos + n_neg
        if total > 0:
            scores.append((n_pos - n_neg) / total)
    return float(np.mean(scores)) if scores else 0.0


def long_short_returns(
    q_df: pd.DataFrame,
    long_quantile: int = _N_GROUPS - 1,
    short_quantile: int = 0,
    cost: float = 0.0,
) -> pd.DataFrame:
    """多空组合每日收益。

    返回 DataFrame with index=datetime, columns=[ls_return]。
    """
    records = []
    for dt, grp in q_df.groupby("datetime"):
        long_row = grp[grp["quantile"] == long_quantile]
        short_row = grp[grp["quantile"] == short_quantile]
        if long_row.empty or short_row.empty:
            continue
        ls_ret = float(long_row["mean"].iloc[0]) - float(short_row["mean"].iloc[0])
        ls_ret -= 2 * cost  # 双边交易成本
        records.append({"datetime": dt, "ls_return": ls_ret})

    return pd.DataFrame(records).set_index("datetime") if records else pd.DataFrame()


def calc_ls_stats(ls_df: pd.DataFrame, annual_factor: float = 252.0, label_horizon: int = 5) -> dict:
    """计算多空组合统计量。

    Args:
        ls_df: 多空收益数据，含 ls_return 列
        annual_factor: 年化因子（日度=252）
        label_horizon: 标签收益率对应的持有期（交易日），
                       如 5 日标签 label_horizon=5，则年化时除以 5，累积用日频等效

    Returns:
        含 annual_return, annual_vol, sharpe, max_drawdown, cumulative 的字典
    """
    if ls_df.empty or len(ls_df) < 5:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "cumulative": pd.Series()}

    returns = ls_df["ls_return"].dropna()
    if len(returns) < 5:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "cumulative": pd.Series()}

    # 将多期标签收益率转为日频等效后再年化
    # 如 5日标签: 日频等效 = 标签值 / 5
    daily_equiv = returns / label_horizon if label_horizon > 1 else returns
    ann_ret = float(daily_equiv.mean()) * annual_factor
    ann_vol = float(returns.std()) / np.sqrt(label_horizon) * np.sqrt(annual_factor) if label_horizon > 1 else float(returns.std()) * np.sqrt(annual_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    # 累积用日频等效收益，避免多期标签的复利爆炸
    cum = (1 + daily_equiv).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max.where(running_max > 0, np.nan)
    mdd = float(np.nanmin(dd)) if not dd.isna().all() else 0.0

    return {
        "annual_return": round(np.nan_to_num(ann_ret * 100, nan=0.0, neginf=0.0, posinf=0.0), 2),
        "annual_vol": round(np.nan_to_num(ann_vol * 100, nan=0.0, neginf=0.0, posinf=0.0), 2),
        "sharpe": round(np.nan_to_num(sharpe, nan=0.0, neginf=0.0, posinf=0.0), 4),
        "max_drawdown": round(np.nan_to_num(mdd * 100, nan=0.0, neginf=0.0, posinf=0.0), 2),
        "cumulative": cum,
    }


def calc_group_avg_returns(q_df: pd.DataFrame) -> pd.Series:
    """各分组的平均收益率序列。

    q_df 包含 datetime / quantile / mean / count 四列（扁平格式）。
    """
    return q_df.groupby("quantile")["mean"].mean()


def calc_holding_period_returns(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    quantiles: int = 5,
    horizons: list = None,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """多期持有收益分析：在不同调仓频率下因子的分层收益。

    标签已是多日 forward return（如 Ref($close,-5)/Ref($open,-1)-1），
    对 horizon > 1 时将相同标签值重复应用到后续每日截面。
    该分析显示如果按不同周期调仓，因子的分层效果如何。

    Args:
        df: DataFrame, 含 datetime, instrument, factor_col, label_col
        factor_col: 因子列名
        label_col: 标签列名
        quantiles: 分组数
        horizons: 调仓周期列表（单位：交易日）

    Returns:
        DataFrame: [horizon, q0_mean, qN_mean, ls_return, monotonicity, n_days]
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    results = []
    for h in horizons:
        if h <= 1:
            # 每日调仓：直接用每日截面
            q_df = quantile_returns(df, factor_col, label_col, quantiles)
        else:
            # 每 h 日调仓：只在调仓日计算截面收益
            df_h = df.sort_values(["datetime", "instrument"]).copy()
            dates = sorted(df_h["datetime"].unique())
            rebalance_dates = set(dates[::h])  # 每 h 天选一个调仓日
            df_h = df_h[df_h["datetime"].isin(rebalance_dates)]
            q_df = quantile_returns(df_h, factor_col, label_col, quantiles) if not df_h.empty else pd.DataFrame()

        if q_df.empty:
            continue
        q_means = calc_group_avg_returns(q_df)
        ls_ret = q_means.get(quantiles - 1, 0) - q_means.get(0, 0) if len(q_means) >= 2 else 0
        mono = calc_monotonicity_score(q_df) if not q_df.empty else 0

        results.append({
            "horizon": h,
            "q0_mean": q_means.get(0, 0),
            f"q{quantiles - 1}_mean": q_means.get(quantiles - 1, 0),
            "ls_return": ls_ret - cost_bps / 10000.0 * 252,  # net of slippage
            "monotonicity": mono,
            "n_days": q_df["datetime"].nunique() if not q_df.empty else 0,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()

def calc_group_cumulative_returns(q_df: pd.DataFrame) -> pd.DataFrame:
    """计算每个分位组的累计净值曲线（分层净值曲线）。

    q_df 包含 datetime / quantile / mean 三列（扁平格式）。
    返回 DataFrame: index=datetime, columns=[G1, G2, ..., GN],
    每个单元格为当日为止的累计净值。

    这是「分层净值曲线」的数据基础，用于直观观察：
      - 分层是否长期分化（G10 >> G1）
      - 是否阶段性失效（某时间段所有组混在一起）
    """
    if q_df.empty:
        return pd.DataFrame()

    # 透视：每行=日期，每列=分组
    piv = q_df.pivot_table(index="datetime", columns="quantile", values="mean", aggfunc="mean")
    piv = piv.sort_index()

    # 将多期标签收益转为日频等效（与 calc_ls_stats 保持一致）
    # 从已有的 label_horizon 无法获取，这里保守不缩放，用原始 cumprod
    # 因为报告显示的是相对趋势而非绝对年化，不影响分层分化判断
    # 关键：用 fillna(0) 替代 dropna()，避免尾部因 label_horizon 产生的 NaN 导致大量数据被丢弃
    # 若某天某分组无成分股（mean=NaN），视同当日收益为 0，不中断累计净值
    piv = piv.fillna(0.0)
    cum = (1 + piv).cumprod()
    cum.columns = [f"G{int(c)+1}" for c in cum.columns]
    return cum


def calc_turnover(q_df: pd.DataFrame) -> dict:
    """计算分层组合的月均换手率（精确追踪成分股ID变化）。

    换手率 = 每月组合内成分股变更比例的平均值。
    直接追踪每层每月的 instrument 集合，计算 Jaccard 距离。
    Virtu 视角：高换手 = 高交易成本吞噬收益。

    Args:
        q_df: 分层回测结果，含 datetime / quantile / mean / count / instruments

    Returns:
        {monthly_turnover_by_q: {q0: 0.3, ...},
         avg_turnover: 0.25,
         max_turnover: 0.45}
    """
    if q_df.empty or "datetime" not in q_df.columns:
        return {"monthly_turnover_by_q": {}, "avg_turnover": 0.0, "max_turnover": 0.0}

    if "instruments" not in q_df.columns:
        return {"monthly_turnover_by_q": {}, "avg_turnover": 0.0, "max_turnover": 0.0}

    df = q_df.copy()
    df["month"] = pd.to_datetime(df["datetime"]).dt.to_period("M")

    turnovers = {}
    for q, grp in df.groupby("quantile"):
        monthly = grp.groupby("month")["instruments"].apply(
            lambda x: set(sum(x, []))  # 合并当月所有成分股
        )
        if len(monthly) < 2:
            continue
        changes = []
        prev_set = monthly.iloc[0]
        for curr_set in monthly.iloc[1:]:
            if not prev_set or not curr_set:
                continue
            union = prev_set | curr_set
            intersect = prev_set & curr_set
            to = 1 - len(intersect) / len(union) if union else 0
            changes.append(to)
            prev_set = curr_set
        if changes:
            turnovers[f"q{q}"] = round(float(np.mean(changes)), 4)

    avg = float(np.mean(list(turnovers.values()))) if turnovers else 0.0
    mx = float(max(turnovers.values())) if turnovers else 0.0

    return {
        "monthly_turnover_by_q": turnovers,
        "avg_turnover": round(avg, 4),
        "max_turnover": round(mx, 4),
    }

# ── A 股交易约束过滤 ──

def filter_ashare_constraints(df, factor_col=None, limit_up_pct=0.095, limit_down_pct=-0.095, filter_suspended=True, volume_col='volume'):
    import numpy as np
    result = df.copy()
    n_before = len(result)
    reasons = []
    if limit_up_pct is not None and 'change_pct' in result.columns:
        limit_up_mask = result['change_pct'] >= limit_up_pct
        if volume_col in result.columns:
            vol = pd.to_numeric(result[volume_col], errors='coerce').fillna(0)
            limit_up_mask = limit_up_mask & (vol < result[volume_col].quantile(0.5))
        result = result[~limit_up_mask]
        reasons.append(f'涨跌停过滤: {limit_up_mask.sum()} 行')
    if filter_suspended and volume_col in result.columns:
        vol = pd.to_numeric(result[volume_col], errors='coerce').fillna(0)
        suspended = vol == 0
        result = result[~suspended]
        reasons.append(f'停牌过滤: {suspended.sum()} 行')
    n_removed = n_before - len(result)
    if n_removed > 0 and factor_col:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f'[A 股约束] {factor_col}: 去除 {n_removed} 行')
    return result


def calc_capacity_analysis(df, factor_col, label_col, aum_levels=None, turnover_rate=0.5, annual_factor=252.0):
    import numpy as np
    import pandas as pd
    if aum_levels is None:
        aum_levels = [1e8, 5e8, 1e9, 5e9, 1e10]
    q_df = quantile_returns(df, factor_col, label_col)
    ls_df = long_short_returns(q_df, cost=0.0)
    ls_stats = calc_ls_stats(ls_df, annual_factor)
    base_ret = ls_stats.get('annual_return', 0)
    base_sharpe = ls_stats.get('sharpe', 0)
    if 'amount' in df.columns:
        avg_daily_volume = pd.to_numeric(df['amount'], errors='coerce').median()
    else:
        avg_daily_volume = 1e8
    rows = []
    for aum in aum_levels:
        daily_turnover = aum * turnover_rate
        participation_rate = daily_turnover / avg_daily_volume if avg_daily_volume > 0 else 0
        participation_rate = min(participation_rate, 1.0)
        impact_cost = np.sqrt(participation_rate) * 0.001
        annual_impact = impact_cost * annual_factor * turnover_rate
        adj_return = base_ret - annual_impact
        capacity_ratio = adj_return / base_ret if abs(base_ret) > 1e-8 else 0
        rows.append({
            'aum': aum,
            'aum_label': f'{aum/1e8:.0f}亿',
            'annual_return': round(adj_return, 4),
            'sharpe': round(base_sharpe * max(0, capacity_ratio), 4),
            'capacity_ratio': round(capacity_ratio, 4),
            'participation_rate': round(participation_rate, 4),
            'impact_cost_bps': round(impact_cost * 10000, 1),
        })
    return pd.DataFrame(rows)
