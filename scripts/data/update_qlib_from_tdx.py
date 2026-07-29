"""
用通达信日线数据增量更新本地 Qlib .day.bin

本脚本只更新本地 Qlib，不写 ClickHouse。

当前覆盖字段：
- open
- high
- low
- close
- volume
- amount

当前不更新：
- total_mv
- circ_mv
- sw_l1 / sw_l2 / sw_l3

原因：
- 通达信可稳定提供 OHLCV 和 Amount
- 市值/行业仍建议走专门的数据源更新
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, r"D:\chenxu\TDX_MONI\PYPlugins\user")

from tqcenter import tq


QLIB_DATA_DIR = PROJECT_ROOT / "qlib_data"
FEATURES_DIR = QLIB_DATA_DIR / "features"
CALENDAR_PATH = QLIB_DATA_DIR / "calendars" / "day.txt"
FIELD_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "amount": "Amount",
}


def load_calendar() -> list[str]:
    with open(CALENDAR_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_calendar(calendar_list: list[str]) -> None:
    with open(CALENDAR_PATH, "w", encoding="utf-8") as f:
        for date_str in calendar_list:
            f.write(f"{date_str}\n")


def read_bin(bin_path: Path) -> tuple[int | None, np.ndarray]:
    if not bin_path.exists():
        return None, np.array([], dtype=np.float32)
    raw = bin_path.read_bytes()
    if len(raw) < 4:
        return None, np.array([], dtype=np.float32)
    start_idx = int(struct.unpack("<f", raw[:4])[0])
    data = np.frombuffer(raw, dtype="<f4")[1:].copy()
    return start_idx, data


def write_bin(bin_path: Path, start_idx: int, values: np.ndarray) -> None:
    header = np.array([start_idx], dtype="<f4").tobytes()
    body = values.astype("<f4").tobytes()
    with open(bin_path, "wb") as f:
        f.write(header + body)


def merge_bin_updates(bin_path: Path, start_idx: int, updates: dict[int, float]) -> None:
    if not updates:
        return

    old_start, old_data = read_bin(bin_path)
    if old_start is None:
        new_end = max(updates)
        merged = np.full(new_end - start_idx + 1, np.nan, dtype=np.float32)
        for idx, value in updates.items():
            merged[idx - start_idx] = float(value)
        write_bin(bin_path, start_idx, merged)
        return

    merge_start = min(old_start, start_idx)
    merge_end = max(old_start + len(old_data) - 1, max(updates))
    merged = np.full(merge_end - merge_start + 1, np.nan, dtype=np.float32)
    merged[old_start - merge_start : old_start - merge_start + len(old_data)] = old_data
    for idx, value in updates.items():
        merged[idx - merge_start] = float(value)
    write_bin(bin_path, merge_start, merged)


def get_local_value(bin_path: Path, calendar_index: int) -> float | None:
    start_idx, data = read_bin(bin_path)
    if start_idx is None:
        return None
    offset = calendar_index - start_idx
    if offset < 0 or offset >= len(data):
        return None
    value = data[offset]
    if np.isnan(value):
        return None
    return float(value)


def get_existing_symbols(limit: int | None = None) -> list[str]:
    symbols = sorted(d.name.upper() for d in FEATURES_DIR.iterdir() if d.is_dir())
    return symbols[:limit] if limit else symbols


def fetch_market_data(stock_list: list[str], start_date: str, end_date: str, batch_size: int) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i : i + batch_size]
        data = tq.get_market_data(
            stock_list=batch,
            start_time=start_date.replace("-", ""),
            end_time=end_date.replace("-", ""),
            period="1d",
            dividend_type="none",
        )
        if not isinstance(data, dict):
            raise RuntimeError(f"通达信返回格式异常: {type(data)}")
        for key, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
            merged[key] = df if key not in merged else pd.concat([merged[key], df], axis=1)
        print(f"  已抓取批次 {i // batch_size + 1}: {len(batch)} 只")
    return merged


def resolve_anchor_date(
    symbol: str,
    market_data: dict[str, pd.DataFrame],
    calendar_list: list[str],
    calendar_map: dict[str, int],
    latest_local_date: str,
) -> str | None:
    close_df = market_data["Close"]
    if symbol not in close_df.columns:
        return None
    candidates = [d for d in close_df.index.tolist() if d <= latest_local_date]
    candidates.sort(reverse=True)
    close_bin = FEATURES_DIR / symbol.lower() / "close.day.bin"
    for date_str in candidates:
        raw_close = close_df.at[date_str, symbol]
        if pd.isna(raw_close) or raw_close == 0:
            continue
        local_close = get_local_value(close_bin, calendar_map[date_str])
        if local_close is None or local_close == 0:
            continue
        return date_str
    return None


def build_scales(symbol: str, anchor_date: str, market_data: dict[str, pd.DataFrame], calendar_map: dict[str, int]) -> dict[str, float]:
    stock_dir = FEATURES_DIR / symbol.lower()
    anchor_idx = calendar_map[anchor_date]

    local_close = get_local_value(stock_dir / "close.day.bin", anchor_idx)
    local_volume = get_local_value(stock_dir / "volume.day.bin", anchor_idx)
    local_amount = get_local_value(stock_dir / "amount.day.bin", anchor_idx)

    raw_close = float(market_data["Close"].at[anchor_date, symbol])
    raw_volume = float(market_data["Volume"].at[anchor_date, symbol])
    raw_amount = float(market_data["Amount"].at[anchor_date, symbol])

    if local_close is None or raw_close == 0:
        raise ValueError(f"{symbol} 无法构建价格锚点")
    if local_volume is None or raw_volume == 0:
        raise ValueError(f"{symbol} 无法构建成交量锚点")
    if local_amount is None or raw_amount == 0:
        raise ValueError(f"{symbol} 无法构建成交额锚点")

    return {
        "price_scale": local_close / raw_close,
        "volume_scale": local_volume / raw_volume,
        "amount_scale": local_amount / raw_amount,
    }


def update_symbol_bins(
    symbol: str,
    market_data: dict[str, pd.DataFrame],
    new_dates: list[str],
    calendar_map: dict[str, int],
    scales: dict[str, float],
) -> int:
    stock_dir = FEATURES_DIR / symbol.lower()
    stock_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for qlib_field, tdx_field in FIELD_MAP.items():
        df = market_data[tdx_field]
        if symbol not in df.columns:
            continue

        updates: dict[int, float] = {}
        for date_str in new_dates:
            if date_str not in df.index:
                continue
            raw_value = df.at[date_str, symbol]
            if pd.isna(raw_value):
                continue
            raw_value = float(raw_value)
            if qlib_field in {"open", "high", "low", "close"}:
                value = raw_value * scales["price_scale"]
            elif qlib_field == "volume":
                value = raw_value * scales["volume_scale"]
            else:
                value = raw_value * scales["amount_scale"]
            updates[calendar_map[date_str]] = value

        if updates:
            existing_start, _ = read_bin(stock_dir / f"{qlib_field}.day.bin")
            start_idx = existing_start if existing_start is not None else min(updates)
            merge_bin_updates(stock_dir / f"{qlib_field}.day.bin", start_idx=start_idx, updates=updates)
            written += len(updates)

    return written


def run_update(
    end_date: str,
    batch_size: int,
    lookback_days: int,
    limit: int | None = None,
    start_date: str | None = None,
) -> None:
    calendar_list = load_calendar()
    latest_local_date = calendar_list[-1]
    latest_local_ts = pd.Timestamp(latest_local_date)
    end_ts = pd.Timestamp(end_date)
    start_ts = pd.Timestamp(start_date) if start_date else latest_local_ts + pd.Timedelta(days=1)

    if end_ts < start_ts:
        print(f"本地 Qlib 已更新到 {latest_local_date}，无需处理。")
        return

    fetch_start = (min(latest_local_ts, start_ts) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    stock_list = get_existing_symbols(limit=limit)
    print("=" * 60)
    print("通达信 -> 本地 Qlib 增量更新")
    print(f"股票数: {len(stock_list)}")
    print(f"抓取范围: {fetch_start} ~ {end_date}")
    print(f"本地最新日历: {latest_local_date}")
    print("=" * 60)

    tq.initialize(__file__)
    try:
        market_data = fetch_market_data(stock_list, start_date=fetch_start, end_date=end_date, batch_size=batch_size)
    finally:
        tq.close()

    if "Close" not in market_data or market_data["Close"].empty:
        raise RuntimeError("通达信未返回有效日线数据")

    update_dates = sorted(d for d in market_data["Close"].index.tolist() if pd.Timestamp(d) >= start_ts and pd.Timestamp(d) <= end_ts)
    if not update_dates:
        print("通达信数据中没有处于目标区间的交易日。")
        return

    missing_dates = [d for d in update_dates if d not in calendar_list]
    if missing_dates:
        calendar_list = calendar_list + missing_dates
        save_calendar(calendar_list)
    calendar_map = {date_str: idx for idx, date_str in enumerate(calendar_list)}
    if missing_dates:
        print(f"追加日历 {len(missing_dates)} 天: {missing_dates[0]} ~ {missing_dates[-1]}")
    else:
        print("日历无需追加，直接刷新目标区间数据。")

    success = 0
    failed = 0
    written_points = 0
    for idx, symbol in enumerate(stock_list, 1):
        try:
            anchor_date = resolve_anchor_date(symbol, market_data, calendar_list, calendar_map, latest_local_date)
            if anchor_date is None:
                failed += 1
                print(f"[跳过] {symbol}: 未找到重叠锚点")
                continue
            scales = build_scales(symbol, anchor_date, market_data, calendar_map)
            points = update_symbol_bins(symbol, market_data, update_dates, calendar_map, scales)
            success += 1
            written_points += points
            if idx % 200 == 0:
                print(f"  已处理 {idx}/{len(stock_list)} 只")
        except Exception as exc:
            failed += 1
            print(f"[失败] {symbol}: {exc}")

    print("=" * 60)
    print(f"完成：成功 {success} 只，失败 {failed} 只，写入数据点 {written_points}")
    print(f"最新日历已到: {calendar_list[-1]}")
    print("提示：本次只补了 Qlib 本地日线字段，市值/行业字段未更新。")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="用通达信日线增量更新本地 Qlib")
    parser.add_argument("--end-date", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="截止日期，默认今天")
    parser.add_argument("--batch-size", type=int, default=200, help="通达信抓取批大小")
    parser.add_argument("--lookback-days", type=int, default=20, help="为寻找锚点回看多少自然日")
    parser.add_argument("--limit", type=int, default=None, help="仅更新前 N 只股票，用于抽样验证")
    parser.add_argument("--start-date", type=str, default=None, help="显式指定要刷新的起始日期；用于补跑已追加到日历但未写完的区间")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_update(
        end_date=args.end_date,
        batch_size=args.batch_size,
        lookback_days=args.lookback_days,
        limit=args.limit,
        start_date=args.start_date,
    )
