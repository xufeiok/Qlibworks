# -*- coding: utf-8 -*-
"""
数据同步脚本 - 用腾讯实时快照补齐 ClickHouse daily_prices 最新交易日。

背景：TDX 补数（update_clickhouse_from_tdx.py）依赖通达信盘后数据文件，
盘中/收盘后当日数据无法即时入库。本脚本通过腾讯行情接口（qt.gtimg.cn）
批量拉取全市场实时快照，收盘后即为当日终值，可第一时间补齐最新交易日。

覆盖范围：
- daily_prices（仅缺失的 trade_date，默认不覆盖已有日期）

不覆盖：
- daily_adj_factors / index_daily / daily_indicators（复权与估值类字段由 TDX/Tushare 维护）

风险控制：
- 默认 dry-run（预演），加 --execute 才正式写入
- 写入前做日期重叠校验，目标日期已有数据则拒绝写入
- 停牌（price<=0）、字段缺失、OHLC 非法的股票整行剔除

用法：
    # 预演（不写入，查看补数计划与样例统计）
    python -m scripts.data.sync_web_quotes

    # 正式写入最新交易日
    python -m scripts.data.sync_web_quotes --execute

    # 指定截止日期（默认今天）与每批数量
    python -m scripts.data.sync_web_quotes --end-date 2026-10-05 --batch-size 60 --execute
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlworks.data.api import QuantDataAPI

# 腾讯行情接口
TX_QUOTE_URL = "http://qt.gtimg.cn/q="
# 单次请求最多携带的股票数量（URL 长度限制）
DEFAULT_BATCH_SIZE = 60
# 时间字段索引（YYYYMMDDHHMMSS）
FIELD_DATE = 30


def _to_tx_code(ts_code: str) -> str:
    """ts_code(600000.SH) -> 腾讯代码(sh600000)，仅支持沪深京三市"""
    code, market = ts_code.split(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(market.upper())
    if not prefix:
        raise ValueError(f"不支持的市场代码: {ts_code}")
    return f"{prefix}{code}"


def _to_ts_code(tx_code: str) -> str:
    """腾讯代码(sh600000) -> ts_code(600000.SH)"""
    code = tx_code[2:]
    market = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(tx_code[:2].lower())
    if not market:
        raise ValueError(f"无法识别的腾讯代码: {tx_code}")
    return f"{code}.{market}"


def _parse_quote(text: str, tx_code: str) -> dict | None:
    """
    解析腾讯单只快照 v_sh600000="1~名称~代码~...~"。
    返回 OHLCV 字段字典；停牌/字段缺失返回 None。

    字段索引（split('~')）：3=现价 4=昨收 5=今开 6=成交量(手) 31=涨跌 32=涨跌幅% 33=最高 34=最低 37=成交额(万元)
    输出单位与 ClickHouse daily_prices 一致：vol=手、amount=千元。
    """
    prefix = f'v_{tx_code}="'
    start = text.find(prefix)
    if start == -1:
        return None
    start += len(prefix)
    end = text.find('"', start)
    if end == -1:
        return None
    fields = text[start:end].split("~")
    if len(fields) < 38:
        return None

    def _f(idx: int) -> float | None:
        try:
            v = float(fields[idx])
            return v if v == v else None
        except (ValueError, IndexError):
            return None

    price = _f(3)
    if not price or price <= 0:
        return None  # 停牌或未开盘
    date_str = (fields[FIELD_DATE] or "")[:8]
    try:
        trade_date = datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        return None

    open_p, high, low = _f(5), _f(33), _f(34)
    pre_close = _f(4) or 0
    change = _f(31)
    pct_chg = _f(32)
    vol_hand = _f(6) or 0
    amount_wan = _f(37) or 0
    # 涨跌幅兜底（个别接口字段缺失时按昨收计算）
    if pct_chg is None and pre_close:
        pct_chg = (price - pre_close) / pre_close * 100
    if pct_chg is None:
        pct_chg = 0.0
    # 合法性校验：OHLC 非负、最高>=最低；停牌股腾讯返回 OHLC=0，整行剔除
    if None in (open_p, high, low) or high < low or open_p <= 0 or high <= 0 or low <= 0:
        return None
    return {
        "ts_code": _to_ts_code(tx_code),
        "trade_date": trade_date,
        "open": round(open_p, 4),
        "high": round(high, 4),
        "low": round(low, 4),
        "close": round(price, 4),
        "pre_close": round(pre_close, 4),
        "change": round(change if change is not None else (price - pre_close), 4),
        "pct_chg": round(pct_chg, 4),
        "vol": round(vol_hand, 0),          # 手（与 ClickHouse daily_prices 单位一致）
        "amount": round(amount_wan * 10, 2),  # 万元 -> 千元（与 ClickHouse daily_prices 单位一致）
    }


def _read_main_board_codes() -> list[str]:
    """读取主板股票池（与 update_clickhouse_from_tdx.py 一致）"""
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


def _get_latest_date(api: QuantDataAPI) -> date:
    df = api.query("SELECT max(trade_date) AS latest FROM daily_prices")
    val = df.iloc[0, 0]
    return pd.Timestamp(val).date() if val is not None else date(2000, 1, 1)


def _fetch_quotes(codes: list[str], batch_size: int, session) -> list[dict]:
    """分批拉取腾讯快照，返回合法行情行列表"""
    rows: list[dict] = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i : i + batch_size]
        tx_codes = ",".join(_to_tx_code(c) for c in batch)
        try:
            r = session.get(TX_QUOTE_URL + tx_codes, timeout=15)
            r.encoding = "gbk"
        except Exception as e:
            print(f"  [WARN] 批次 {i // batch_size + 1} 请求失败: {e}")
            continue
        batch_rows = []
        for c in batch:
            parsed = _parse_quote(r.text, _to_tx_code(c))
            if parsed:
                batch_rows.append(parsed)
        if batch_rows:
            rows.extend(batch_rows)
        print(f"  已抓取批次 {i // batch_size + 1}: {len(batch)} 只，有效 {len(batch_rows)} 只")
    return rows


def _check_overlap(api: QuantDataAPI, table_name: str, trade_date: str) -> int:
    sql = f"SELECT count() AS n FROM {table_name} WHERE trade_date = '{trade_date}'"
    return int(api.query(sql).iloc[0, 0])


def run_sync(end_date: date, batch_size: int, execute: bool):
    """主流程：查最新日 -> 拉腾讯快照 -> 过滤 -> 校验 -> （可选）写入"""
    import requests

    print("=" * 60)
    print("腾讯实时快照 -> ClickHouse daily_prices 补数")
    print("=" * 60)

    stock_list = _read_main_board_codes()
    print(f"主板股票池: {len(stock_list)} 只")

    with QuantDataAPI() as api:
        latest = _get_latest_date(api)
        print(f"daily_prices 最新日期: {latest}")
        if latest >= end_date:
            print(f"[跳过] 目标日期 {end_date} 已有数据，无需补数。")
            return

        print(f"补数目标日期: {end_date}")
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
        })
        rows = _fetch_quotes(stock_list, batch_size, session)
        if not rows:
            print("[中止] 未获取到任何有效行情，请检查网络或稍后重试。")
            return

        df = pd.DataFrame(rows)
        # 只保留目标日期的数据（避免快照日期与目标不符的脏数据）
        df = df[df["trade_date"] == end_date]
        print(f"目标日期 {end_date} 有效行数: {len(df)}")

        # 合理性校验：close>0、OHLC 全正、high>=low、pct_chg 在 ±50% 内
        df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
        df = df[(df["high"] >= df["low"]) & (df["pct_chg"].between(-50, 50))]
        # 与已有最近数据交叉验证：剔除疑似除权日异常（pct_chg 极端但量能异常）由人工复核，此处仅统计
        print(f"校验后有效行数: {len(df)}")

        overlap = _check_overlap(api, "daily_prices", end_date.strftime("%Y-%m-%d"))
        if overlap > 0:
            raise RuntimeError(f"daily_prices 目标日期 {end_date} 已存在 {overlap} 行，拒绝覆盖写入")

        # 预览样例
        sample = df.head(5)[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]]
        print("\n[样例数据] 前5行：")
        print(sample.to_string(index=False))

        print("=" * 60)
        if execute:
            api._get_ch_client().insert_df("daily_prices", df[["ts_code", "trade_date", "open", "high", "low",
                                                               "close", "pre_close", "change", "pct_chg", "vol", "amount"]])
            print(f"[完成写入] daily_prices: {len(df)} 行 @ {end_date}")
        else:
            print(f"[预演] 共 {len(df)} 行待写入 @ {end_date}，加 --execute 正式写入。")
        print("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(description="用腾讯实时快照补齐 ClickHouse daily_prices 最新交易日")
    parser.add_argument("--end-date", type=str, default=date.today().strftime("%Y-%m-%d"), help="补数日期，默认今天")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="腾讯单次请求股票数量")
    parser.add_argument("--execute", action="store_true", help="正式写入 ClickHouse；默认仅预演")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_sync(end_date=pd.Timestamp(args.end_date).date(), batch_size=args.batch_size, execute=args.execute)
