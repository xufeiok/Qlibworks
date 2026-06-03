"""
稳健性检验：子时段、子股票池、参数敏感性。
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional

from .ic_analysis import calc_daily_ic, calc_ic_stats
from .group_analysis import (
    quantile_returns,
    long_short_returns,
    calc_ls_stats,
    calc_monotonicity_score,
)


def test_sub_periods(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    periods: list,
    annual_factor: float = 252.0,
) -> pd.DataFrame:
    """子时段检验。"""
    rows = []
    for start, end in periods:
        sub = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
        if sub.empty:
            continue
        ic_s = calc_daily_ic(sub, factor_col, label_col)
        stats = calc_ic_stats(ic_s, annual_factor)
        q_df = quantile_returns(sub, factor_col, label_col)
        ls_df = long_short_returns(q_df, cost=0.001)
        ls_s = calc_ls_stats(ls_df, annual_factor)
        rows.append({
            "period": f"{start}~{end}",
            "ic_mean": stats["ic_mean"],
            "icir": stats["icir"],
            "win_rate": stats["win_rate"],
            "ls_ann_ret": ls_s["annual_return"],
            "ls_sharpe": ls_s["sharpe"],
            "monotonicity": round(calc_monotonicity_score(q_df), 4),
            "n_days": len(sub["datetime"].unique()),
        })
    return pd.DataFrame(rows)


def test_sub_pools(
    accessor_call: Callable,
    factor_col: str,
    label_col: str,
    start_time: str,
    end_time: str,
    pools: list,
    label_expr: str,
    field_exprs: dict,
    annual_factor: float = 252.0,
) -> pd.DataFrame:
    """子股票池检验。"""
    rows = []
    for pool in pools:
        try:
            spec = accessor_call(pool, list(field_exprs.values()), start_time, end_time)
            df = spec
            # 重命名列
            rename = {v: k for k, v in field_exprs.items()}
            df = df.rename(columns=rename)

            ic_s = calc_daily_ic(df, factor_col, label_col)
            stats = calc_ic_stats(ic_s, annual_factor)
            q_df = quantile_returns(df, factor_col, label_col)
            ls_df = long_short_returns(q_df)

            n_stocks = len(df.index.get_level_values("instrument").unique()) if isinstance(df.index, pd.MultiIndex) else 0
            ls_s = calc_ls_stats(ls_df, annual_factor)

            rows.append({
                "pool": pool,
                "ic_mean": stats["ic_mean"],
                "icir": stats["icir"],
                "win_rate": stats["win_rate"],
                "ls_ann_ret": ls_s["annual_return"],
                "ls_sharpe": ls_s["sharpe"],
                "monotonicity": round(calc_monotonicity_score(q_df), 4),
                "n_stocks": n_stocks,
            })
        except Exception as e:
            rows.append({"pool": pool, "error": str(e)})

    return pd.DataFrame(rows)


def test_parameter_sensitivity(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    param_grid: dict,
    eval_func: Callable,
    annual_factor: float = 252.0,
) -> pd.DataFrame:
    """参数敏感性检验。"""
    rows = []
    for param_name, values in param_grid.items():
        for val in values:
            try:
                metrics = eval_func(df, factor_col, label_col, **{param_name: val})
                if isinstance(metrics, dict):
                    metrics["param"] = str(val)
                    metrics["param_name"] = param_name
                    rows.append(metrics)
            except Exception:
                continue
    return pd.DataFrame(rows)
