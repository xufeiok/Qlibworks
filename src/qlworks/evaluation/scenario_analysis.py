"""
场景压力测试与控制变量对冲：扩展稳健性检验，覆盖专家方法论步骤 4+5。

包含功能：
  1. test_by_market_cap_buckets  — 分市值分组（大/中/小盘）分别跑 IC + 分层
  2. test_by_market_regime       — 牛熊/震荡市分段评测
  3. test_by_industry_sector     — 分行业板块（周期/消费/制造/金融/科技）测评
  4. bivariate_sort              — 双变量分组（先按市值分层，再按因子二次分层）
  5. residual_factor_test        — 残差因子评测（剔除行业/市值干扰后的纯因子）
  6. size_neutral_test           — 规模中性化后的因子评测
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

from .ic_analysis import calc_daily_ic, calc_ic_stats
from .group_analysis import (
    quantile_returns,
    long_short_returns,
    calc_ls_stats,
    calc_monotonicity_score,
    calc_group_avg_returns,
)


# ── 行业板块映射（申万一级→五大板块） ──

SECTOR_MAP = {
    # 周期
    "化工": "周期", "钢铁": "周期", "有色金属": "周期", "建筑材料": "周期",
    "采掘": "周期", "交通运输": "周期", "公用事业": "周期",
    # 消费
    "食品饮料": "消费", "医药生物": "消费", "商贸零售": "消费",
    "农林牧渔": "消费", "家用电器": "消费", "纺织服装": "消费",
    "轻工制造": "消费", "汽车": "消费", "休闲服务": "消费",
    # 制造
    "电气设备": "制造", "机械设备": "制造", "国防军工": "制造",
    "建筑装饰": "制造", "综合": "制造",
    # 金融
    "银行": "金融", "非银金融": "金融", "房地产": "金融",
    # 科技
    "电子": "科技", "计算机": "科技", "通信": "科技",
    "传媒": "科技",
}


# ── 牛熊市分段（基于沪深300 全收益指数） ──

MARKET_REGIMES = [
    ("2014-07-01", "2015-06-12", "牛市"),
    ("2015-06-15", "2015-08-26", "熊市"),
    ("2015-08-27", "2015-12-31", "震荡"),
    ("2016-01-04", "2016-01-28", "熊市"),
    ("2016-01-29", "2016-11-28", "震荡"),
    ("2016-11-29", "2017-12-31", "震荡"),
    ("2018-01-02", "2018-12-31", "熊市"),
    ("2019-01-02", "2020-01-23", "牛市"),
    ("2020-02-03", "2020-03-23", "熊市"),
    ("2020-03-24", "2021-02-10", "牛市"),
    ("2021-02-18", "2021-12-31", "震荡"),
    ("2022-01-04", "2022-10-31", "熊市"),
    ("2022-11-01", "2023-05-09", "震荡"),
    ("2023-05-10", "2024-02-05", "熊市"),
    ("2024-02-06", "2024-05-20", "震荡"),
    ("2024-05-21", "2024-09-30", "熊市"),
    ("2024-10-08", "2026-06-30", "震荡"),
]


def _sector_from_industry(industry_name: str) -> str:
    if pd.isna(industry_name):
        return "其他"
    return SECTOR_MAP.get(industry_name.strip(), "其他")


def _calc_bucket_ic_stats(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    method: str = "spearman",
    annual_factor: float = 252.0,
) -> dict:
    ic_s = calc_daily_ic(df, factor_col, label_col, method)
    stats = calc_ic_stats(ic_s, annual_factor)
    return stats


def _calc_bucket_ls_stats(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    quantiles: int = 10,
    annual_factor: float = 252.0,
    label_horizon: int = 5,
    cost: float = 0.001,
) -> dict:
    q_df = quantile_returns(df, factor_col, label_col, quantiles)
    if q_df.empty:
        return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0, "monotonicity": 0}
    mono = calc_monotonicity_score(q_df)
    ls_df = long_short_returns(q_df, quantiles - 1, 0, cost=cost)
    ls_s = calc_ls_stats(ls_df, annual_factor, label_horizon)
    ls_s["monotonicity"] = round(mono, 4)
    return ls_s


# ═══════════════════════════════════════════
# 1. 分市值分组检验
# ═══════════════════════════════════════════

def test_by_market_cap_buckets(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    mkt_cap_col: str = "mkt_cap",
    quantiles: int = 10,
    annual_factor: float = 252.0,
    label_horizon: int = 5,
) -> pd.DataFrame:
    """分市值分组评测：每日按市值分为大/中/小 3 组，分别跑 IC + 分层。

    Returns:
        DataFrame: [bucket, ic_mean, icir, win_rate, ls_ann_ret, ls_sharpe, monotonicity, n_days]
    """
    if mkt_cap_col not in df.columns:
        logger.warning(f"[分市值] 缺少{mkt_cap_col}列，跳过")
        return pd.DataFrame()

    tmp = df.copy()
    def _assign_bucket(x):
        if x.nunique() < 6:
            return pd.Series(["未知"] * len(x))
        return pd.qcut(x.rank(method="first"), q=3, labels=["小盘", "中盘", "大盘"])

    tmp["_cap_bucket"] = tmp.groupby("datetime")[mkt_cap_col].transform(_assign_bucket)

    rows = []
    for bucket in ["小盘", "中盘", "大盘"]:
        sub = tmp[tmp["_cap_bucket"] == bucket]
        if sub["datetime"].nunique() < 20:
            continue
        ic_s = _calc_bucket_ic_stats(sub, factor_col, label_col, "spearman", annual_factor)
        ls_s = _calc_bucket_ls_stats(sub, factor_col, label_col, quantiles, annual_factor, label_horizon)
        rows.append({
            "bucket": bucket,
            "ic_mean": ic_s["ic_mean"],
            "icir": ic_s["icir"],
            "win_rate": ic_s["win_rate"],
            "ls_ann_ret": ls_s["annual_return"],
            "ls_sharpe": ls_s["sharpe"],
            "max_drawdown": ls_s.get("max_drawdown", 0),
            "monotonicity": ls_s.get("monotonicity", 0),
            "n_days": sub["datetime"].nunique(),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════
# 2. 牛熊分段检验
# ═══════════════════════════════════════════

def test_by_market_regime(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    regimes: list = None,
    quantiles: int = 10,
    annual_factor: float = 252.0,
    label_horizon: int = 5,
) -> pd.DataFrame:
    """按牛熊/震荡市分段评测因子表现。

    Args:
        regimes: [(start, end, regime_name), ...]

    Returns:
        DataFrame: [regime, period, ic_mean, icir, win_rate, ls_ann_ret, ls_sharpe, monotonicity, n_days]
    """
    if regimes is None:
        regimes = MARKET_REGIMES

    rows = []
    for start, end, regime in regimes:
        sub = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
        if sub["datetime"].nunique() < 10:
            continue
        ic_s = _calc_bucket_ic_stats(sub, factor_col, label_col, "spearman", annual_factor)
        ls_s = _calc_bucket_ls_stats(sub, factor_col, label_col, quantiles, annual_factor, label_horizon)
        rows.append({
            "regime": regime,
            "period": f"{start}~{end}",
            "ic_mean": ic_s["ic_mean"],
            "icir": ic_s["icir"],
            "win_rate": ic_s["win_rate"],
            "ls_ann_ret": ls_s["annual_return"],
            "ls_sharpe": ls_s["sharpe"],
            "max_drawdown": ls_s.get("max_drawdown", 0),
            "monotonicity": ls_s.get("monotonicity", 0),
            "n_days": sub["datetime"].nunique(),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════
# 3. 分行业板块检验
# ═══════════════════════════════════════════

def test_by_industry_sector(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    industry_col: str = "industry",
    sector_map: dict = None,
    quantiles: int = 10,
    annual_factor: float = 252.0,
    label_horizon: int = 5,
) -> pd.DataFrame:
    """按行业板块（周期/消费/制造/金融/科技）分别评测因子。

    Returns:
        DataFrame: [sector, ic_mean, icir, win_rate, ls_ann_ret, ls_sharpe, monotonicity, n_stocks, n_days]
    """
    if industry_col not in df.columns:
        logger.warning(f"[分行业] 缺少{industry_col}列，跳过")
        return pd.DataFrame()

    if sector_map is None:
        sector_map = SECTOR_MAP

    tmp = df.copy()
    tmp["_sector"] = tmp[industry_col].map(_sector_from_industry)

    rows = []
    for sector in ["周期", "消费", "制造", "金融", "科技"]:
        sub = tmp[tmp["_sector"] == sector]
        if sub["datetime"].nunique() < 20 or len(sub) < 500:
            continue
        ic_s = _calc_bucket_ic_stats(sub, factor_col, label_col, "spearman", annual_factor)
        ls_s = _calc_bucket_ls_stats(sub, factor_col, label_col, quantiles, annual_factor, label_horizon)
        rows.append({
            "sector": sector,
            "ic_mean": ic_s["ic_mean"],
            "icir": ic_s["icir"],
            "win_rate": ic_s["win_rate"],
            "ls_ann_ret": ls_s["annual_return"],
            "ls_sharpe": ls_s["sharpe"],
            "max_drawdown": ls_s.get("max_drawdown", 0),
            "monotonicity": ls_s.get("monotonicity", 0),
            "n_stocks": sub["instrument"].nunique() if "instrument" in sub.columns else 0,
            "n_days": sub["datetime"].nunique(),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════
# 4. 双变量分组（剔除市值干扰）
# ═══════════════════════════════════════════

def bivariate_sort(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    primary_col: str = "mkt_cap",
    primary_n: int = 5,
    secondary_n: int = 5,
    annual_factor: float = 252.0,
    label_horizon: int = 5,
) -> pd.DataFrame:
    """双变量分组：先按 primary_col 分 primary_n 组，再在每个组内按 factor 分 secondary_n 组。

    目的是检验因子在控制 primary_col（如市值）后是否仍有独立预测力。

    Returns:
        DataFrame: [primary_group, q0_mean, qN_mean, ls_return, monotonicity, n_days]
    """
    if primary_col not in df.columns:
        logger.warning(f"[双变量] 缺少{primary_col}列，跳过")
        return pd.DataFrame()

    tmp = df.copy()
    results = []

    for dt, grp in tmp.groupby("datetime"):
        f = grp[factor_col]
        l = grp[label_col]
        p = grp[primary_col]
        valid = f.notna() & l.notna() & p.notna()
        if valid.sum() < primary_n * secondary_n * 2:
            continue

        p_valid = p[valid]
        p_ranks = p_valid.rank(method="first")
        p_groups = np.floor((p_ranks - 1) / len(p_ranks) * primary_n).clip(0, primary_n - 1).astype(int)

        for pg in range(primary_n):
            mask = valid.copy()
            mask[mask] = p_groups == pg
            if mask.sum() < secondary_n * 2:
                continue

            f_pg = f[mask]
            l_pg = l[mask]
            s_ranks = f_pg.rank(method="first")
            s_groups = np.floor((s_ranks - 1) / len(s_ranks) * secondary_n).clip(0, secondary_n - 1).astype(int)

            q_means = {}
            for sg in range(secondary_n):
                sg_mask = s_groups == sg
                if sg_mask.sum() > 0:
                    q_means[sg] = float(l_pg[sg_mask].mean())

            if len(q_means) >= 2:
                q0 = q_means.get(0, 0)
                qN = q_means.get(secondary_n - 1, 0)
                results.append({
                    "primary_group": pg,
                    "datetime": dt,
                    "q0_mean": q0,
                    "qN_mean": qN,
                    "ls_return": qN - q0,
                })

    if not results:
        return pd.DataFrame()

    rdf = pd.DataFrame(results)
    summary = rdf.groupby("primary_group").agg(
        q0_mean=("q0_mean", "mean"),
        qN_mean=("qN_mean", "mean"),
        ls_return=("ls_return", "mean"),
        n_days=("datetime", "nunique"),
    ).reset_index()

    ls_vals = summary.sort_values("primary_group")["ls_return"].values
    diffs = np.diff(ls_vals)
    n_pos = (diffs > 0).sum()
    n_neg = (diffs < 0).sum()
    summary["monotonicity"] = (n_pos - n_neg) / max(n_pos + n_neg, 1)

    return summary


# ═══════════════════════════════════════════
# 5. 残差因子评测（控制变量回归取残差）
# ═══════════════════════════════════════════

def residual_factor_test(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    control_cols: list = None,
    quantiles: int = 10,
    annual_factor: float = 252.0,
    label_horizon: int = 5,
) -> dict:
    """残差因子评测：用控制变量（行业/市值）对因子做截面回归，取残差作为纯因子，重新跑 IC + 分层。

    这是最严格的「控制变量对冲」检验——验证因子收益是否来自已知风险因子。

    Args:
        control_cols: 需要剔除的变量列表，如 ["mkt_cap", "industry"]

    Returns:
        dict with keys: residual_ic_stats, residual_ls_stats, residual_group_means, control_cols
    """
    if not control_cols:
        control_cols = ["mkt_cap"]

    from sklearn.linear_model import Ridge

    tmp = df.copy()

    resid_list = []
    for dt, grp in tmp.groupby("datetime"):
        f = grp[factor_col].values.astype(float)
        y_valid = ~(np.isnan(f) | np.isinf(f))

        control_dfs = []
        for col in control_cols:
            if col == "industry" and col in grp.columns:
                dummies = pd.get_dummies(grp[col].astype(str), prefix="ind", drop_first=False)
                control_dfs.append(dummies)
            elif col in grp.columns:
                control_dfs.append(grp[[col]].apply(pd.to_numeric, errors="coerce"))

        if not control_dfs:
            continue

        X = pd.concat(control_dfs, axis=1).values.astype(float)
        X_valid = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        valid_mask = y_valid & X_valid

        if valid_mask.sum() < 20:
            continue

        reg = Ridge(alpha=1e-5, fit_intercept=True)
        reg.fit(X[valid_mask], f[valid_mask])
        residuals = np.full(len(grp), np.nan)
        residuals[valid_mask] = f[valid_mask] - reg.predict(X[valid_mask])

        sub = grp.copy()
        sub["_residual"] = residuals
        resid_list.append(sub)

    if not resid_list:
        return {"residual_ic_stats": {}, "residual_ls_stats": {}, "residual_group_means": pd.Series(), "control_cols": control_cols}

    resid_df = pd.concat(resid_list, ignore_index=True)

    ic_s = calc_daily_ic(resid_df, "_residual", label_col, "spearman")
    ic_stats = calc_ic_stats(ic_s, annual_factor)

    q_df = quantile_returns(resid_df, "_residual", label_col, quantiles)
    group_means = calc_group_avg_returns(q_df) if not q_df.empty else pd.Series()
    ls_df = long_short_returns(q_df, quantiles - 1, 0, cost=0.001)
    ls_stats = calc_ls_stats(ls_df, annual_factor, label_horizon)
    mono = calc_monotonicity_score(q_df) if not q_df.empty else 0
    ls_stats["monotonicity"] = round(mono, 4)

    return {
        "residual_ic_stats": ic_stats,
        "residual_ls_stats": ls_stats,
        "residual_group_means": group_means,
        "control_cols": control_cols,
    }


# ═══════════════════════════════════════════
# 6. 规模分组检验（市值中性化验证）
# ═══════════════════════════════════════════

def size_neutral_test(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    mkt_cap_col: str = "mkt_cap",
    quantiles: int = 5,
    annual_factor: float = 252.0,
    label_horizon: int = 5,
) -> dict:
    """规模分组检验：按市值分 5 组，在每组内查看因子值的分布，
    验证中性化是否到位（理想情况：各组因子均值接近 0，避免分层收益只是市值带来的）。

    Returns:
        dict with keys:
          - cap_group_factor_stats: DataFrame [cap_group, factor_mean, factor_std, label_mean, count]
          - cap_group_ic: DataFrame [cap_group, ic_mean, icir, n_days]
    """
    if mkt_cap_col not in df.columns:
        logger.warning(f"[规模分组] 缺少{mkt_cap_col}列，跳过")
        return {}

    tmp = df.copy()

    def _assign_cap_group(x):
        if x.nunique() < 10:
            return pd.Series(["未知"] * len(x))
        return pd.qcut(x.rank(method="first"), q=5, labels=["Q1小", "Q2", "Q3", "Q4", "Q5大"])

    tmp["_cap_group"] = tmp.groupby("datetime")[mkt_cap_col].transform(_assign_cap_group)

    factor_stats = tmp.groupby("_cap_group").agg(
        factor_mean=(factor_col, "mean"),
        factor_std=(factor_col, "std"),
        label_mean=(label_col, "mean"),
        count=(label_col, "count"),
    ).reset_index()
    factor_stats.columns = ["cap_group", "factor_mean", "factor_std", "label_mean", "count"]

    ic_rows = []
    for grp_name, sub in tmp.groupby("_cap_group"):
        if sub["datetime"].nunique() < 10:
            continue
        ic_s = calc_daily_ic(sub, factor_col, label_col, "spearman")
        stats = calc_ic_stats(ic_s, annual_factor)
        ic_rows.append({
            "cap_group": grp_name,
            "ic_mean": stats["ic_mean"],
            "icir": stats["icir"],
            "n_days": len(ic_s.dropna()),
        })

    return {
        "cap_group_factor_stats": factor_stats,
        "cap_group_ic": pd.DataFrame(ic_rows) if ic_rows else pd.DataFrame(),
    }
