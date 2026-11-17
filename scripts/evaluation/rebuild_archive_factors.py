# -*- coding: utf-8 -*-
"""
按 GTJA191 归档公式重建主板因子。

功能:
- 从 ClickHouse 读取前复权 OHLC、成交量和成交额。
- 按归档表达式语义计算 13 个归档因子。
- 输出到指定仓库目录，并更新元数据。

边界条件:
- 滚动窗口不足的位置保持 NaN。
- 横截面 Rank 使用按交易日升序的百分位排名。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlworks.data.api import QuantDataAPI
from qlworks.evaluation import DEFAULT_CONFIG, FactorStore
from qlworks.evaluation.factor_store import _sanitize_values


ARCHIVE_FACTORS = {
    "Alpha2": "-1 * Delta(((($close-$low)-($high-$close))/($high-$low)),1)",
    "Alpha6": "(Rank(Sign(Delta((($open*0.85)+($high*0.15)),4)))*-1)",
    "Alpha13": "((($high*$low)^0.5)-$vwap)",
    "Alpha17": "Rank(($vwap-Max($vwap,15)))^Delta($close,5)",
    "Alpha37": "-1*Rank((Sum($open,5)*Sum($close/Ref($close,1)-1,5)-Ref(Sum($open,5)*Sum($close/Ref($close,1)-1,5),10)))",
    "Alpha42": "-1*Rank(Std($high,10))*Corr($high,$volume,10)",
    "Alpha48": "-1*(Rank(Sign($close-Ref($close,1))+Sign(Ref($close,1)-Ref($close,2))+Sign(Ref($close,2)-Ref($close,3)))*Sum($volume,5)/Sum($volume,20))",
    "Alpha81": "Sma($volume,21,2)",
    "Alpha83": "-1*Rank(Cov(Rank($high),Rank($volume),5))",
    "Alpha98": "If(Delta(Sum($close,100)/100,100)/Ref($close,100)<=0.05, -1*($close-Min($close,100)), -1*Delta($close,3))",
    "Alpha137": "16*($close-Ref($close,1)+($close-$open)/2+Ref($close,1)-Ref($open,1))/(If((Abs($high-Ref($close,1))>Abs($low-Ref($close,1)))&(Abs($high-Ref($close,1))>Abs($high-Ref($low,1))),Abs($high-Ref($close,1))+Abs($low-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4,If((Abs($low-Ref($close,1))>Abs($high-Ref($low,1)))&(Abs($low-Ref($close,1))>Abs($high-Ref($close,1))),Abs($low-Ref($close,1))+Abs($high-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4,Abs($high-Ref($low,1))+Abs(Ref($close,1)-Ref($open,1))/4)))*Max(Abs($high-Ref($close,1)),Abs($low-Ref($close,1)))",
    "Alpha158": "(($high-Sma($close,15,2))-($low-Sma($close,15,2)))/$close",
    "Alpha185": "Rank((-1*((1-($open/$close))^2)))",
}


def _read_codes(path: Path) -> list[str]:
    """读取主板股票代码。"""
    codes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            codes.append(parts[0].upper())
    return sorted(set(codes))


def _load_raw(api: QuantDataAPI, stocks: list[str], start: str, end: str) -> pd.DataFrame:
    """读取前复权行情和估值字段。"""
    store = FactorStore(DEFAULT_CONFIG)
    sql = store._build_adj_sql(start, end, stocks)
    frame = api.query(sql)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.set_index(["ts_code", "trade_date"]).sort_index()
    frame.index.names = ["instrument", "datetime"]
    return frame


def _cross_rank(series: pd.Series) -> pd.Series:
    """按交易日计算截面百分位排名。"""
    return series.groupby(level="datetime").rank(pct=True)


def _rolling(series: pd.Series, window: int, method: str, min_periods: int | None = None) -> pd.Series:
    """按股票执行滚动统计。"""
    if min_periods is None:
        min_periods = window
    grouped = series.groupby(level="instrument", group_keys=False)
    if method == "sum":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    if method == "mean":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    if method == "min":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).min())
    if method == "max":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).max())
    if method == "std":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=1))
    raise ValueError(f"未知滚动统计: {method}")


def _shift(series: pd.Series, periods: int) -> pd.Series:
    """按股票滞后序列。"""
    return series.groupby(level="instrument", group_keys=False).shift(periods)


def _sma(series: pd.Series, n: int, m: int) -> pd.Series:
    """实现中国市场常用 Sma(X,N,M) 递推平滑。"""
    return series.groupby(level="instrument", group_keys=False).transform(
        lambda s: s.ewm(alpha=m / n, adjust=False, min_periods=n).mean()
    )


def _grouped_rolling_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """按股票分组计算滚动相关系数。"""
    result = pd.Series(np.nan, index=left.index, dtype=np.float64)
    for instrument, index in left.groupby(level="instrument", sort=False).groups.items():
        result.loc[index] = left.loc[index].rolling(window).corr(right.loc[index]).to_numpy()
    return result


def _grouped_rolling_cov(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """按股票分组计算滚动协方差。"""
    result = pd.Series(np.nan, index=left.index, dtype=np.float64)
    for instrument, index in left.groupby(level="instrument", sort=False).groups.items():
        result.loc[index] = left.loc[index].rolling(window).cov(right.loc[index]).to_numpy()
    return result


def _compute_factor(raw: pd.DataFrame, name: str) -> pd.Series:
    """按归档公式数学语义计算单因子。"""
    open_ = raw["open"]
    high = raw["high"]
    low = raw["low"]
    close = raw["close"]
    volume = raw["volume"]
    vwap = raw["vwap"]

    if name == "Alpha2":
        value = -((close - low) - (high - close)) / (high - low)
        return _shift(value, 1) - value

    if name == "Alpha6":
        delta = (open_ * 0.85 + high * 0.15) - _shift(open_ * 0.85 + high * 0.15, 4)
        return -_cross_rank(np.sign(delta))

    if name == "Alpha13":
        return np.sqrt(high * low) - vwap

    if name == "Alpha17":
        base = vwap - _rolling(vwap, 15, "max")
        exponent = close - _shift(close, 5)
        return np.power(base, exponent)

    if name == "Alpha37":
        ret = close / _shift(close, 1) - 1
        base = _rolling(open_, 5, "sum") * _rolling(ret, 5, "sum")
        return -_cross_rank(base - _shift(base, 10))

    if name == "Alpha42":
        std10 = _rolling(high, 10, "std")
        corr10 = _grouped_rolling_corr(high, volume, 10)
        return -_cross_rank(std10) * corr10

    if name == "Alpha48":
        sign_sum = (
            np.sign(close - _shift(close, 1))
            + np.sign(_shift(close, 1) - _shift(close, 2))
            + np.sign(_shift(close, 2) - _shift(close, 3))
        )
        return -_cross_rank(sign_sum) * _rolling(volume, 5, "sum") / _rolling(volume, 20, "sum")

    if name == "Alpha81":
        return _sma(volume, 21, 2)

    if name == "Alpha83":
        rank_high = _cross_rank(high)
        rank_volume = _cross_rank(volume)
        cov5 = _grouped_rolling_cov(rank_high, rank_volume, 5)
        return -_cross_rank(cov5)

    if name == "Alpha98":
        mean100 = _rolling(close, 100, "mean")
        condition = (mean100 - _shift(mean100, 100)) / _shift(close, 100) <= 0.05
        return pd.Series(
            np.where(
                condition,
                -1 * (close - _rolling(close, 100, "min")),
                -1 * (close - _shift(close, 3)),
            ),
            index=close.index,
        )

    if name == "Alpha137":
        prev_close = _shift(close, 1)
        prev_open = _shift(open_, 1)
        prev_low = _shift(low, 1)
        high_close = (high - prev_close).abs()
        low_close = (low - prev_close).abs()
        high_low = (high - prev_low).abs()
        close_open = (prev_close - prev_open).abs()
        condition1 = (high_close > low_close) & (high_close > high_low)
        condition2 = (low_close > high_low) & (low_close > high_close)
        denominator = np.where(
            condition1,
            high_close + low_close / 2 + close_open / 4,
            np.where(
                condition2,
                low_close + high_close / 2 + close_open / 4,
                high_low + close_open / 4,
            ),
        )
        numerator = close - prev_close + (close - open_) / 2 + prev_close - prev_open
        return 16 * numerator / denominator * np.maximum(high_close, low_close)

    if name == "Alpha158":
        return (high - low) / close

    if name == "Alpha185":
        return _cross_rank(-1 * np.power(1 - open_ / close, 2))

    raise ValueError(f"未实现归档因子: {name}")


def _save_factor(store: FactorStore, name: str, values: pd.Series) -> int:
    """按年份保存单因子。"""
    frame = values.rename("value").to_frame().dropna()
    if frame.empty:
        return 0
    frame = _sanitize_values(frame, "value")
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return 0
    frame = frame.sort_index()
    frame["_year"] = frame.index.get_level_values("datetime").year
    total = 0
    for year, part in frame.groupby("_year"):
        part = part.drop(columns=["_year"])
        part = part.astype({"value": "float32"})
        store._save_warehouse_year(name, int(year), part)
        total += len(part)
    store._update_warehouse_meta(name)
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建 GTJA191 归档因子")
    parser.add_argument("--qlib-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-09-11")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    qlib_dir = Path(args.qlib_dir).resolve()
    output_root = Path(args.output_root).resolve()
    stocks = _read_codes(qlib_dir / "instruments" / "main_board.txt")

    from dataclasses import replace

    config = replace(DEFAULT_CONFIG, warehouse_dir=str(output_root))
    store = FactorStore(config)

    api = QuantDataAPI()
    print(f"[1] 读取主板行情: {len(stocks)} 只")
    raw = _load_raw(api, stocks, args.start_date, args.end_date)
    print(f"[2] 行情数据: {raw.shape}")

    report = {}
    for index, name in enumerate(ARCHIVE_FACTORS, 1):
        print(f"[3.{index}] {name}")
        values = _compute_factor(raw, name)
        rows = _save_factor(store, name, values)
        report[name] = rows
        if rows == 0:
            raise RuntimeError(f"归档因子无有效数据: {name}")

    report_path = output_root / "archive_rebuild_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[4] 完成: {report_path}")


if __name__ == "__main__":
    main()
