# -*- coding: utf-8 -*-
"""
用 Tushare Pro 增量补齐本地 Qlib .day.bin 的最新交易日。

适用场景：ClickHouse 不可达、本地通达信数据文件陈旧时，仍能把 Qlib 内核数据更新到当天收盘。

覆盖字段（与现有 bin 口径保持一致）：
- open / high / low / close / volume / amount：Tushare daily（不复权价、volume 单位手、amount 单位千元）
- total_mv / circ_mv：Tushare daily_basic（万元）
- sw_l1 / sw_l2 / sw_l3：沿用该股票 bin 中最后一个有效行业编码（申万行业短期内不变）

输入：
- --start-date：追加起始日；默认取本地日历最后一日的下一交易日
- --end-date：追加截止日；默认今天
- --execute：正式写盘；缺省为 dry-run（只打印计划）

输出：
- 扩展 qlib_data/calendars/day.txt
- 追加 qlib_data/features/<code>/*.day.bin

边界与异常：
- 目标日期若非交易日（trade_cal 校验）自动跳过
- bin 结束位置与旧日历末尾不一致的股票（退市/长期停牌）一律跳过，避免日期错位
- Tushare 无当日行情的股票（停牌）写入 NaN，保持与日历等长
- 任一步失败直接抛错退出，不做静默降级

性能/安全注意：
- 每次只追加增量交易日，逐股票重写 bin（约 3500 个目录，秒级完成）
- 写盘前做全量预演统计；不覆盖历史已有值（只追加）
"""
from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QLIB_DATA_DIR = PROJECT_ROOT / "qlib_data"
FEATURES_DIR = QLIB_DATA_DIR / "features"
CALENDAR_PATH = QLIB_DATA_DIR / "calendars" / "day.txt"

# 凭证统一从环境变量 / 项目根 .env 读取（qlworks.config 负责加载 .env），禁止硬编码
from qlworks.config import TUSHARE_TOKEN
PRICE_FIELDS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "vol",
    "amount": "amount",
}
MV_FIELDS = {"total_mv": "total_mv", "circ_mv": "circ_mv"}
INDUSTRY_FIELDS = ("sw_l1", "sw_l2", "sw_l3")
INDEX_TS_CODES = {"SH000905": "000905.SH", "SH000300": "000300.SH"}


def load_calendar() -> list[str]:
    """读取本地交易日历。"""
    with open(CALENDAR_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_bin(bin_path: Path) -> tuple[int, np.ndarray]:
    """读取 qlib .day.bin：4 字节起始索引头 + float32 数组。"""
    raw = bin_path.read_bytes()
    start = int(struct.unpack("<f", raw[:4])[0])
    return start, np.frombuffer(raw[4:], dtype="<f4").copy()


def write_bin(bin_path: Path, start: int, values: np.ndarray) -> None:
    """写回 qlib .day.bin（float32 小端）。"""
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<f", float(start)))
        f.write(values.astype("<f4").tobytes())


def _to_ts_code(dir_name: str) -> str:
    """目录名 600000.sh -> Tushare 代码 600000.SH。"""
    code, market = dir_name.split(".")
    return f"{code}.{market.upper()}"


def resolve_new_dates(pro, last_cal_date: str, end_date: str) -> list[str]:
    """校验并返回需要追加的交易日（YYYY-MM-DD）。"""
    start = (pd.Timestamp(last_cal_date) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    cal = pro.trade_cal(
        exchange="SSE",
        start_date=start,
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        fields="cal_date,is_open",
    )
    if cal.empty:
        return []
    return [
        pd.Timestamp(str(d)).strftime("%Y-%m-%d")
        for d in cal[cal["is_open"] == 1].sort_values("cal_date")["cal_date"]
    ]


def fetch_block(pro, new_dates: list[str]) -> tuple[dict, dict]:
    """按交易日拉取全市场日行情与市值，返回 {ts_code: {date: row}} 结构。"""
    price_map: dict[str, dict[str, dict]] = {}
    mv_map: dict[str, dict[str, dict]] = {}
    for d in new_dates:
        key = d.replace("-", "")
        daily = pro.daily(
            trade_date=key,
            fields="ts_code,open,high,low,close,vol,amount",
        )
        basic = pro.daily_basic(
            trade_date=key,
            fields="ts_code,total_mv,circ_mv",
        )
        for row in daily.to_dict("records"):
            price_map.setdefault(row["ts_code"], {})[d] = row
        for row in basic.to_dict("records"):
            mv_map.setdefault(row["ts_code"], {})[d] = row
    return price_map, mv_map


def append_stock(code_dir: Path, ts_code: str, new_dates: list[str],
                 old_len: int, price_map: dict, mv_map: dict) -> str:
    """对单只股票追加增量交易日；返回状态标识。"""
    close_path = code_dir / "close.day.bin"
    if not close_path.exists():
        return "no_close"
    start, close_arr = read_bin(close_path)
    if start + len(close_arr) != old_len:
        # 退市或长期停牌的股票，bin 未对齐旧日历末尾，跳过避免日期错位
        return "unaligned"

    k = len(new_dates)
    stock_price = price_map.get(ts_code, {})
    stock_mv = mv_map.get(ts_code, {})

    for field, src in PRICE_FIELDS.items():
        p = code_dir / f"{field}.day.bin"
        if not p.exists():
            continue
        _, arr = read_bin(p)
        vals = [stock_price.get(d, {}).get(src, np.nan) for d in new_dates]
        write_bin(p, start, np.concatenate([arr, np.asarray(vals, dtype="<f4")]))

    for field, src in MV_FIELDS.items():
        p = code_dir / f"{field}.day.bin"
        if not p.exists():
            continue
        _, arr = read_bin(p)
        vals = [stock_mv.get(d, {}).get(src, np.nan) for d in new_dates]
        write_bin(p, start, np.concatenate([arr, np.asarray(vals, dtype="<f4")]))

    # 行业编码沿用最后一期有效值（申万行业短期不变）
    for field in INDUSTRY_FIELDS:
        p = code_dir / f"{field}.day.bin"
        if not p.exists():
            continue
        _, arr = read_bin(p)
        valid = arr[~np.isnan(arr)]
        last_val = float(valid[-1]) if len(valid) else np.nan
        write_bin(p, start, np.concatenate([arr, np.full(k, last_val, dtype="<f4")]))

    return "ok"


def append_index(pro, calendar: list[str]) -> dict:
    """补齐基准指数（中证500/沪深300）的 OHLCV；允许 bin 落后日历，自动补齐缺口。"""
    cal_map = {d: i for i, d in enumerate(calendar)}
    stats = {}
    for qlib_name, ts_code in INDEX_TS_CODES.items():
        d = FEATURES_DIR / qlib_name.lower()
        close_path = d / "close.day.bin"
        if not close_path.exists():
            stats[qlib_name] = "missing"
            continue
        start, close_arr = read_bin(close_path)
        cur_end = start + len(close_arr)          # 已覆盖到的日历下标（不含）
        target_end = len(calendar)                # 目标：与日历等长
        if cur_end >= target_end:
            stats[qlib_name] = "already"
            continue

        df = pro.index_daily(
            ts_code=ts_code,
            start_date=calendar[cur_end].replace("-", ""),
            end_date=calendar[-1].replace("-", ""),
            fields="trade_date,open,high,low,close,vol",
        )
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        row_map = {r["trade_date"]: r for r in df.to_dict("records")}

        for field, src in {"open": "open", "high": "high", "low": "low",
                           "close": "close", "volume": "vol"}.items():
            p = d / f"{field}.day.bin"
            if not p.exists():
                continue
            _, arr = read_bin(p)
            # 先把已有数组对齐到 start，再按缺口长度补齐（缺口日写 NaN）
            full = np.full(target_end - start, np.nan, dtype="<f4")
            full[:len(arr)] = arr[: min(len(arr), target_end - start)]
            for day in calendar[cur_end:target_end]:
                idx = cal_map[day] - start
                full[idx] = row_map.get(day, {}).get(src, np.nan)
            write_bin(p, start, full)
        stats[qlib_name] = f"ok(+{target_end - cur_end})"
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="用 Tushare 增量补齐 Qlib .day.bin")
    parser.add_argument("--start-date", type=str, default=None,
                        help="追加起始日；默认本地日历最后一日之后的交易日")
    parser.add_argument("--end-date", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                        help="追加截止日，默认今天")
    parser.add_argument("--execute", action="store_true", help="正式写盘（缺省 dry-run）")
    return parser.parse_args()


def main():
    args = parse_args()
    import tushare as ts

    token = TUSHARE_TOKEN
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未配置：请在环境变量或项目根 .env 中设置")
    pro = ts.pro_api(token)

    cal = load_calendar()
    last_cal = cal[-1]
    old_len = len(cal)

    new_dates = resolve_new_dates(pro, last_cal, args.end_date)
    if args.start_date:
        new_dates = [d for d in new_dates if d >= args.start_date]
    print(f"本地日历: {old_len} 天，最后一日 {last_cal}")
    if not new_dates:
        print(f"无需更新：{last_cal} 已是最新交易日（截止 {args.end_date}）")
        return
    print(f"待追加交易日({len(new_dates)}): {new_dates}")
    if not args.execute:
        print("dry-run：未写盘，确认无误后加 --execute 执行")
        return

    price_map, mv_map = fetch_block(pro, new_dates)
    print(f"行情覆盖股票数: {len(price_map)}；市值覆盖股票数: {len(mv_map)}")

    stats = {"ok": 0, "unaligned": 0, "no_close": 0}
    for code_dir in sorted(FEATURES_DIR.iterdir()):
        if not code_dir.is_dir() or code_dir.name.lower() in {k.lower() for k in INDEX_TS_CODES}:
            continue
        status = append_stock(
            code_dir, _to_ts_code(code_dir.name), new_dates, old_len, price_map, mv_map
        )
        stats[status] = stats.get(status, 0) + 1
    print(f"股票 bin 追加统计: {stats}")

    new_calendar = cal + new_dates
    idx_stats = append_index(pro, new_calendar)
    print(f"指数 bin 追加统计: {idx_stats}")

    with open(CALENDAR_PATH, "a", encoding="utf-8") as f:
        for d in new_dates:
            f.write(f"{d}\n")
    print(f"日历已扩展至 {new_dates[-1]}（{old_len} -> {old_len + len(new_dates)}）")


if __name__ == "__main__":
    main()
