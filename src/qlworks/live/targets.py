"""
目标持仓生成逻辑

职责:
- 从日度分数表中筛出指定交易日的目标持仓
- 按等权方式分配目标权重

说明:
- 首版保持和回测一致，先只做“目标组合生成”，不在此层处理账户持仓差额
- 若无股票通过阈值，则返回空表，由执行层选择保持空仓或仅做卖出
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def build_daily_target_positions(
    score_df: pd.DataFrame,
    trade_date: str | pd.Timestamp,
    top_k: int,
    score_threshold: float,
    buy_pct: float,
) -> pd.DataFrame:
    """
    根据指定日期的打分结果生成目标持仓表。

    输入:
    - score_df: 至少包含 datetime/instrument/score，可选 raw_score
    - trade_date: 目标交易日
    - top_k: 最大持仓数
    - score_threshold: 分数阈值
    - buy_pct: 总资金使用率

    输出:
    - DataFrame: trade_date/instrument/score/raw_score/target_weight/rank
    """
    required_cols = {"datetime", "instrument", "score"}
    missing_cols = required_cols - set(score_df.columns)
    if missing_cols:
        raise ValueError(f"score_df 缺少必要列: {sorted(missing_cols)}")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not 0 < buy_pct <= 1:
        raise ValueError("buy_pct 必须在 (0, 1] 区间内")

    trade_ts = pd.Timestamp(trade_date).normalize()
    frame = score_df.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()

    day_df = frame.loc[frame["datetime"] == trade_ts].copy()
    if day_df.empty:
        return _empty_target_frame()

    day_df = day_df.loc[day_df["score"].notna()]
    day_df = day_df.loc[day_df["score"] >= score_threshold]
    if day_df.empty:
        return _empty_target_frame()

    day_df = day_df.sort_values(["score", "instrument"], ascending=[False, True]).head(top_k).reset_index(drop=True)
    target_weight = buy_pct / len(day_df)

    day_df["trade_date"] = trade_ts
    if "raw_score" not in day_df.columns:
        day_df["raw_score"] = day_df["score"]
    day_df["target_weight"] = target_weight
    day_df["rank"] = range(1, len(day_df) + 1)

    return day_df[["trade_date", "instrument", "score", "raw_score", "target_weight", "rank"]]


def _empty_target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["trade_date", "instrument", "score", "raw_score", "target_weight", "rank"]
    )


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    将 DataFrame 转为 records，便于脚本层做 JSON/日志输出。
    """
    if df.empty:
        return []
    return df.to_dict(orient="records")
