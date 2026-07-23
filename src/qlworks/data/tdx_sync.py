"""
通达信 -> ClickHouse 日频补数辅助模块

职责:
- 将通达信返回的宽表行情转换为 ClickHouse 目标长表
- 对通达信前复权因子做按股锚点缩放，保证能与现有 daily_adj_factors 连续拼接

注意:
- 通达信 ForwardFactor 与 ClickHouse(Tushare) adj_factor 的绝对尺度不同
- 不能直接拼接，必须通过重叠日期做缩放对齐
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pandas as pd


def _to_decimal_4(value: float | int | None) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    return Decimal(str(float(value))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _stack_wide_frame(wide_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    stacked = wide_df.stack(future_stack=True).reset_index()
    stacked.columns = ["trade_date", "ts_code", value_name]
    stacked["trade_date"] = pd.to_datetime(stacked["trade_date"]).dt.normalize()
    return stacked


def build_daily_prices_frame(market_data: dict[str, pd.DataFrame], start_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    将通达信 get_market_data 返回值转换为 daily_prices 表结构。
    """
    required = ["Open", "High", "Low", "Close", "Volume", "Amount"]
    missing = [key for key in required if key not in market_data]
    if missing:
        raise ValueError(f"market_data 缺少必要字段: {missing}")

    close_df = market_data["Close"].sort_index()
    open_df = market_data["Open"].sort_index().reindex_like(close_df)
    high_df = market_data["High"].sort_index().reindex_like(close_df)
    low_df = market_data["Low"].sort_index().reindex_like(close_df)
    vol_df = market_data["Volume"].sort_index().reindex_like(close_df)
    amount_df = market_data["Amount"].sort_index().reindex_like(close_df)

    pre_close_df = close_df.shift(1)
    change_df = close_df - pre_close_df
    pct_chg_df = change_df / pre_close_df * 100

    frames = [
        _stack_wide_frame(open_df, "open"),
        _stack_wide_frame(high_df, "high"),
        _stack_wide_frame(low_df, "low"),
        _stack_wide_frame(close_df, "close"),
        _stack_wide_frame(pre_close_df, "pre_close"),
        _stack_wide_frame(change_df, "change"),
        _stack_wide_frame(pct_chg_df, "pct_chg"),
        _stack_wide_frame(vol_df, "vol"),
        _stack_wide_frame(amount_df, "amount"),
    ]

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["trade_date", "ts_code"], how="left")

    start_ts = pd.Timestamp(start_date).normalize()
    merged = merged.loc[merged["trade_date"] >= start_ts].copy()
    merged = merged.dropna(subset=["open", "high", "low", "close"])

    for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "amount"]:
        merged[col] = merged[col].map(_to_decimal_4)
    merged["vol"] = pd.to_numeric(merged["vol"], errors="coerce").astype(float)

    return merged[
        ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    ]


def scale_forward_factors(
    forward_factor_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    将通达信 ForwardFactor 按重叠锚点缩放到 ClickHouse adj_factor 尺度。

    anchor_df 字段要求:
    - ts_code
    - trade_date
    - adj_factor
    """
    if forward_factor_df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    if anchor_df.empty:
        raise ValueError("anchor_df 为空，无法进行复权因子尺度对齐")

    ff_long = _stack_wide_frame(forward_factor_df.sort_index(), "forward_factor")
    ff_long = ff_long.dropna(subset=["forward_factor"]).copy()

    anchor = anchor_df.copy()
    anchor["trade_date"] = pd.to_datetime(anchor["trade_date"]).dt.normalize()
    anchor["adj_factor"] = pd.to_numeric(anchor["adj_factor"], errors="coerce")

    overlap = ff_long.merge(anchor, on=["ts_code", "trade_date"], how="inner")
    overlap = overlap.dropna(subset=["forward_factor", "adj_factor"])
    if overlap.empty:
        raise ValueError("未找到通达信与 ClickHouse 的复权因子重叠锚点")

    overlap = overlap.sort_values(["ts_code", "trade_date"])
    scale_map = (
        overlap.groupby("ts_code")
        .tail(1)
        .assign(scale=lambda df: df["adj_factor"] / df["forward_factor"])
        .set_index("ts_code")["scale"]
        .to_dict()
    )

    ff_long["scale"] = ff_long["ts_code"].map(scale_map)
    ff_long = ff_long.dropna(subset=["scale"]).copy()
    ff_long["adj_factor"] = (ff_long["forward_factor"] * ff_long["scale"]).map(_to_decimal_4)
    ff_long["trade_date"] = pd.to_datetime(ff_long["trade_date"]).dt.normalize()

    start_ts = pd.Timestamp(start_date).normalize()
    ff_long = ff_long.loc[ff_long["trade_date"] >= start_ts].copy()
    return ff_long[["ts_code", "trade_date", "adj_factor"]]


def build_index_daily_frame(market_data: dict[str, pd.DataFrame], start_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    将指数行情转换为 index_daily 表结构。
    """
    daily_prices = build_daily_prices_frame(market_data, start_date=start_date).copy()
    daily_prices["vol"] = daily_prices["vol"].map(_to_decimal_4)
    return daily_prices[
        ["ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_chg", "vol", "amount"]
    ]
