"""
分层回测分析：分组收益率、单调性检验、多空组合。
"""

import numpy as np
import pandas as pd
from typing import Optional


def safe_quantile_assign(series, n=5):
    """用 rank 分位数替代 pd.qcut，避免重复边界报错。"""
    ranks = series.rank(method="first")
    result = np.floor((ranks - 1) / len(series) * n).clip(0, n - 1).astype(int)
    return result


def quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    quantiles: int = 5,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """每日分层收益率（使用 rank 分位数，更稳定）。

    返回 DataFrame with columns: datetime, quantile, mean, count, instruments
    """
    rows = []
    for dt, grp in df.groupby("datetime"):
        f = grp[factor_col]
        l = grp[label_col]
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
    long_quantile: int = 4,
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


def calc_ls_stats(ls_df: pd.DataFrame, annual_factor: float = 252.0) -> dict:
    """计算多空组合统计量。

    Args:
        ls_df: 多空收益数据，含 ls_return 列
        annual_factor: 年化因子（日度=252）

    Returns:
        含 annual_return, annual_vol, sharpe, max_drawdown, cumulative 的字典

    注意事项:
        - 自动丢弃 ls_return 中的 NaN，避免统计量传播为 NaN
        - 非数值（NaN/Inf）统一替换为 0.0，确保后续格式化不报错
    """
    if ls_df.empty or len(ls_df) < 5:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "cumulative": pd.Series()}

    returns = ls_df["ls_return"].dropna()
    if len(returns) < 5:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "cumulative": pd.Series()}

    ann_ret = float(returns.mean()) * annual_factor
    ann_vol = float(returns.std()) * np.sqrt(annual_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    cum = (1 + returns).cumprod()
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
            "ls_return": ls_ret,
            "monotonicity": mono,
            "n_days": q_df["datetime"].nunique() if not q_df.empty else 0,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()

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

