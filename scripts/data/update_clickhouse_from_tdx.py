"""
用通达信日线数据补齐 ClickHouse 到最新收盘日。

当前覆盖范围：
- daily_prices
- daily_adj_factors
- index_daily（默认仅 000905.SH）

不覆盖：
- daily_indicators

原因：
- 通达信可稳定提供 OHLCV 和前复权因子
- 但无法完整提供历史 daily_basic/估值类字段
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, r"D:\chenxu\TDX_MONI\PYPlugins\user")

from qlworks.data.api import QuantDataAPI
from qlworks.data.tdx_sync import build_daily_prices_frame, build_index_daily_frame, scale_forward_factors
from tqcenter import tq


DEFAULT_INDEX_CODES = ["000905.SH"]


def _read_main_board_codes() -> list[str]:
    path = PROJECT_ROOT / "qlib_data" / "instruments" / "main_board.txt"
    if not path.exists():
        raise FileNotFoundError(f"找不到股票池文件: {path}")

    codes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            code = line.split()[0].strip()
            if "." not in code:
                continue
            raw, market = code.split(".", 1)
            codes.append(f"{raw}.{market.upper()}")
    return sorted(set(codes))


def _fetch_tdx_market_data(stock_list: list[str], start_date: str, end_date: str, batch_size: int) -> dict[str, pd.DataFrame]:
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
            merged[key] = df if key not in merged else pd.concat([merged[key], df], axis=1)
        print(f"  已抓取批次 {i // batch_size + 1}: {len(batch)} 只")
    return merged


def _get_latest_date(api: QuantDataAPI, table_name: str) -> pd.Timestamp:
    df = api.query(f"SELECT max(trade_date) AS latest FROM {table_name}")
    return pd.Timestamp(df.iloc[0, 0]).normalize()


def _query_adj_anchor(api: QuantDataAPI, stock_list: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    all_parts = []
    for i in range(0, len(stock_list), 300):
        batch = stock_list[i : i + 300]
        batch_sql = ",".join(f"'{code}'" for code in batch)
        sql = f"""
        SELECT ts_code, trade_date, adj_factor
        FROM daily_adj_factors
        WHERE trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
          AND ts_code IN ({batch_sql})
        """
        all_parts.append(api.query(sql))
    if not all_parts:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    return pd.concat(all_parts, ignore_index=True)


def _check_overlap(api: QuantDataAPI, table_name: str, start_date: str, end_date: str) -> int:
    sql = f"SELECT count() AS n FROM {table_name} WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
    return int(api.query(sql).iloc[0, 0])


def _insert_df(api: QuantDataAPI, table_name: str, df: pd.DataFrame, execute: bool):
    if df.empty:
        print(f"[跳过] {table_name}: 无新增数据")
        return
    print(f"[准备写入] {table_name}: {len(df)} 行")
    if execute:
        api._get_ch_client().insert_df(table_name, df)
        print(f"[完成写入] {table_name}: {len(df)} 行")


def run_update(end_date: str, batch_size: int, execute: bool):
    with QuantDataAPI() as api:
        price_latest = _get_latest_date(api, "daily_prices")
        adj_latest = _get_latest_date(api, "daily_adj_factors")
        index_latest = _get_latest_date(api, "index_daily")

        price_start = (price_latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        adj_start = (adj_latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        index_start = (index_latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print("=" * 60)
        print("ClickHouse -> 通达信补数计划")
        print(f"daily_prices:      {price_start} ~ {end_date}")
        print(f"daily_adj_factors: {adj_start} ~ {end_date}")
        print(f"index_daily:       {index_start} ~ {end_date}")
        print("=" * 60)

        stock_list = _read_main_board_codes()
        print(f"主板股票池: {len(stock_list)} 只")

        tq.initialize(__file__)
        try:
            if pd.Timestamp(price_start) <= pd.Timestamp(end_date):
                fetch_start = min(price_latest, adj_latest).strftime("%Y-%m-%d")
                market_data = _fetch_tdx_market_data(stock_list, start_date=fetch_start, end_date=end_date, batch_size=batch_size)

                prices_df = build_daily_prices_frame(market_data, start_date=price_start)
                if _check_overlap(api, "daily_prices", price_start, end_date) > 0:
                    raise RuntimeError("daily_prices 目标日期区间已存在数据，当前脚本为安全起见不做覆盖写入")
                _insert_df(api, "daily_prices", prices_df, execute=execute)

                anchor_df = _query_adj_anchor(
                    api,
                    stock_list=stock_list,
                    start_date=adj_latest.strftime("%Y-%m-%d"),
                    end_date=min(price_latest, pd.Timestamp(end_date)).strftime("%Y-%m-%d"),
                )
                adj_df = scale_forward_factors(
                    forward_factor_df=market_data["ForwardFactor"],
                    anchor_df=anchor_df,
                    start_date=adj_start,
                )
                if _check_overlap(api, "daily_adj_factors", adj_start, end_date) > 0:
                    raise RuntimeError("daily_adj_factors 目标日期区间已存在数据，当前脚本为安全起见不做覆盖写入")
                _insert_df(api, "daily_adj_factors", adj_df, execute=execute)
            else:
                print("[跳过] daily_prices / daily_adj_factors 已是最新")

            if pd.Timestamp(index_start) <= pd.Timestamp(end_date):
                index_data = _fetch_tdx_market_data(DEFAULT_INDEX_CODES, start_date=index_start, end_date=end_date, batch_size=len(DEFAULT_INDEX_CODES))
                index_df = build_index_daily_frame(index_data, start_date=index_start)
                if _check_overlap(api, "index_daily", index_start, end_date) > 0:
                    raise RuntimeError("index_daily 目标日期区间已存在数据，当前脚本为安全起见不做覆盖写入")
                _insert_df(api, "index_daily", index_df, execute=execute)
            else:
                print("[跳过] index_daily 已是最新")
        finally:
            tq.close()

    print("=" * 60)
    if execute:
        print("通达信补数已写入 ClickHouse。")
    else:
        print("预演完成：未写入 ClickHouse。加 --execute 可正式写入。")
    print("注意：daily_indicators 仍未更新，若今晚需要完整重训，还需补齐估值/换手率等日频指标。")
    print("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(description="用通达信日线数据补齐 ClickHouse")
    parser.add_argument("--end-date", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="截止日期，默认今天")
    parser.add_argument("--batch-size", type=int, default=200, help="通达信分批抓取股票数量")
    parser.add_argument("--execute", action="store_true", help="正式写入 ClickHouse；默认仅预演")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_update(end_date=args.end_date, batch_size=args.batch_size, execute=args.execute)
