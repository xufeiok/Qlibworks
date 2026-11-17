# -*- coding: utf-8 -*-
"""
修复 Qlib .bin 前复权锚点不一致，并重算受影响股票的因子仓库数据。

背景：
    incremental_sync 以 append 模式写入新段，新段使用"当时"的 latest_adj_factor 作为
    前复权锚点，旧段不重算；若此后该股发生除权，新旧两段锚点不一致，跨断点的滚动窗口
    因子（MA5/MA60 等）会被伪跳空污染。

流程：
    1. 扫描：本地 close.day.bin 与 ClickHouse 当前前复权口径逐样本日比对，
       偏差超过 0.1% 即判定锚点不一致；
    2. 修 .bin：按全历史重新拉取前复权行情，覆盖写入 8 个内核字段（open/high/low/close/
       volume/amount/total_mv/circ_mv）；
    3. 重算因子：对受影响股票重算仓库中全部因子，按索引去重合并（不删除其他股票数据）；
    4. 复检：重复步骤 1，确认偏差归零。

用法：
    python scripts/data/repair_qlib_adj_anchor.py --dry-run        # 只扫描，不写数据
    python scripts/data/repair_qlib_adj_anchor.py                  # 扫描 + 修复 + 复检
    python scripts/data/repair_qlib_adj_anchor.py --skip-factors   # 只修 .bin，不重算因子

注意：
    - 执行前请确保没有其它 Qlib 进程正在读取 features 目录；
    - ClickHouse 是唯一真实数据源（SSOT），.bin 与因子均由它重算，无需额外备份。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import clickhouse_connect

from qlworks.config import (
    CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE,
    QLIB_DATA_DIR, WAREHOUSE_DIR,
)
from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer
from qlworks.evaluation.factor_store import FactorStore

DEV_TOL = 0.001          # 偏差容忍度：0.1%
SAMPLE_MONTHS = (1, 4, 7, 10)  # 每年按季度抽样比对
FACTOR_FIRST_DATE = "2020-01-02"  # 仓库因子起始日（早于该日期的重算结果不写回）


def load_calendar():
    """读取本地 Qlib 交易日历（day.txt）。"""
    path = Path(QLIB_DATA_DIR) / "calendars" / "day.txt"
    return [d.strip() for d in path.read_text().splitlines() if d.strip()]


def build_sample_dates(calendar):
    """构造抽样比对日期：2020 年起每季度首个交易日 + 最新交易日。"""
    dates = []
    for year in range(2020, int(calendar[-1][:4]) + 1):
        for month in SAMPLE_MONTHS:
            hit = [d for d in calendar if d.startswith(f"{year}-{month:02d}")]
            if hit and hit[0] <= calendar[-1]:
                dates.append(hit[0])
    dates.append(calendar[-1])
    return sorted(set(dates))


def load_local_close(calendar, sample_dates):
    """读取本地 close.day.bin，还原为 {股票: {日期: 收盘价}}。"""
    cal_idx = {d: i for i, d in enumerate(calendar)}
    feature_dir = Path(QLIB_DATA_DIR) / "features"
    local = {}
    for stock_dir in feature_dir.iterdir():
        if not stock_dir.is_dir():
            continue
        bin_path = stock_dir / "close.day.bin"
        if not bin_path.exists():
            continue
        raw = np.fromfile(bin_path, dtype="<f4")
        if len(raw) < 2:
            continue
        start_idx, values = int(raw[0]), raw[1:]
        rec = {}
        for date in sample_dates:
            idx = cal_idx.get(date)
            if idx is None:
                continue
            offset = idx - start_idx
            if 0 <= offset < len(values) and np.isfinite(values[offset]) and values[offset] > 0:
                rec[date] = float(values[offset])
        if rec:
            local[stock_dir.name.upper()] = rec
    return local


def query_ch_qfq(client, sample_dates):
    """从 ClickHouse 取样本日的前复权收盘价（与 api.get_daily_data 同口径）。"""
    date_in = ", ".join(f"'{d}'" for d in sample_dates)
    sql = f"""
        SELECT p.ts_code AS ts_code, p.trade_date AS trade_date,
               p.close * COALESCE(NULLIF(a.adj_factor, 0), 1)
                       / COALESCE(NULLIF(latest.adj_factor, 0), 1) AS qfq_close
        FROM daily_prices p
        LEFT JOIN daily_adj_factors a
               ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
        LEFT JOIN (SELECT ts_code, argMax(adj_factor, trade_date) AS adj_factor
                   FROM daily_adj_factors GROUP BY ts_code) latest
               ON p.ts_code = latest.ts_code
        WHERE p.trade_date IN ({date_in})
    """
    df = client.query_df(sql)
    df["trade_date"] = df["trade_date"].astype(str).str.slice(0, 10)
    df["qfq_close"] = df["qfq_close"].astype(float)
    return {
        code: dict(zip(grp["trade_date"], grp["qfq_close"]))
        for code, grp in df.groupby("ts_code")
    }


def scan_affected(client, calendar):
    """
    扫描锚点不一致的股票。

    Returns:
        {股票代码: {"max_abs_dev": float, "last_bad": str}}，空字典表示无异常。
    """
    sample_dates = build_sample_dates(calendar)
    local = load_local_close(calendar, sample_dates)
    ch_map = query_ch_qfq(client, sample_dates)

    affected = {}
    for code, rec in local.items():
        devs = []
        for date, local_close in rec.items():
            ch_close = ch_map.get(code, {}).get(date)
            if ch_close and ch_close > 0:
                devs.append((date, local_close / ch_close - 1.0))
        bad = [(d, r) for d, r in devs if abs(r) > DEV_TOL]
        if bad:
            affected[code] = {
                "max_abs_dev": max(abs(r) for _, r in bad),
                "last_bad": bad[-1][0],
            }
    return affected


def repair_bins(codes, calendar):
    """按全历史重新拉取前复权行情，覆盖写入 8 个内核字段的 .bin 文件。"""
    api = QuantDataAPI()
    api.clear_query_cache(max_age_sec=None)  # 清空查询缓存，避免读到除权前的旧复权结果
    sync = QlibSynchronizer(api)
    calendar_map = {d: i for i, d in enumerate(calendar)}
    sync._sync_features(
        stocks=codes,
        calendar_list=calendar,
        calendar_map=calendar_map,
        start_date=calendar[0],
        end_date=calendar[-1],
        append=False,  # 覆盖模式：全历史重写，统一采用当前 latest_adj_factor 锚点
    )


def repair_factors(codes, start_date, end_date, missing_only=False):
    """对受影响股票重算仓库中全部因子，按索引去重合并写回。

    Args:
        codes: 受影响股票代码列表
        start_date/end_date: 计算区间
        missing_only: 仅处理今日尚未重算过的因子（按 meta.updated_at 判断），用于断点续跑
    """
    store = FactorStore()
    meta_files = sorted(Path(WAREHOUSE_DIR).glob("*/meta.json"))
    print(f"\n[因子] 候选因子 {len(meta_files)} 个" + ("（仅未重算过的）" if missing_only else ""))

    for meta_file in meta_files:
        name = meta_file.parent.name
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if missing_only and str(meta.get("updated_at", "")).startswith(
                    pd.Timestamp.now().strftime("%Y-%m-%d")):
                continue
            expr = (meta.get("expression") or {}).get("qlib")
            if not expr:
                print(f"  - {name}: 缺少 qlib 表达式，跳过")
                continue
            duckdb_expr = (meta.get("expression") or {}).get("duckdb")
            first_date = (meta.get("data_range") or {}).get("first_date") or FACTOR_FIRST_DATE

            df = store._compute(name, expr, start_date, end_date, codes,
                                duckdb_expr=duckdb_expr)
            if df is None or df.empty:
                print(f"  - {name}: 无计算结果，跳过")
                continue
            if df.index.names != ["instrument", "datetime"]:
                df = df.set_index(["instrument", "datetime"])
            df = df[df.index.get_level_values("datetime") >= pd.Timestamp(first_date)]
            if df.empty:
                print(f"  - {name}: 目标区间无数据，跳过")
                continue

            # replace_years=None → 按索引去重合并（新值覆盖旧值），不影响其他股票
            stats = store.save_warehouse_frame(name, df, replace_years=None)
            print(f"  - {name}: 写入 {len(df)} 行（受影响的 {len(codes)} 只股票），"
                  f"仓库总行数 {sum(stats.values())}")
        except Exception as exc:
            # 单因子失败不中断整批：如 inv_turn 等需外部脚本补数据的因子
            print(f"  - {name}: 重算失败，跳过（{exc}）")


def print_affected(affected):
    """打印受影响股票清单（按偏差降序）。"""
    if not affected:
        print("  ✓ 未发现锚点不一致的股票")
        return
    rows = [
        {"ts_code": code, "max_abs_dev": f"{v['max_abs_dev']:.4%}", "last_bad": v["last_bad"]}
        for code, v in sorted(affected.items(), key=lambda kv: -kv[1]["max_abs_dev"])
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="修复 Qlib 前复权锚点不一致")
    parser.add_argument("--dry-run", action="store_true", help="只扫描，不写数据")
    parser.add_argument("--skip-bins", action="store_true", help="跳过 .bin 重写（已修复过时使用）")
    parser.add_argument("--skip-factors", action="store_true", help="只修 .bin，不重算因子")
    parser.add_argument("--missing-only", action="store_true",
                        help="仅处理今日尚未重算过的因子（按 meta.updated_at 判断），推荐用于断点续跑")
    parser.add_argument("--stocks", default=None,
                        help="逗号分隔的股票列表；指定后不再依赖扫描结果（用于 .bin 已修复后重跑因子）")
    args = parser.parse_args()

    calendar = load_calendar()
    calendar_start, calendar_end = calendar[0], calendar[-1]
    print(f"日历 {len(calendar)} 天：{calendar_start} ~ {calendar_end}")

    client = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER,
        password=CH_PASSWORD, database=CH_DATABASE
    )

    print("\n[1] 扫描锚点不一致的股票...")
    affected = scan_affected(client, calendar)
    print(f"  受影响 {len(affected)} 只")
    print_affected(affected)

    if args.stocks:
        codes = sorted(c.strip().upper() for c in args.stocks.split(",") if c.strip())
        print(f"  使用命令行指定的 {len(codes)} 只股票")
    else:
        if not affected:
            return
        codes = sorted(affected)

    if args.dry_run:
        print("\n--dry-run 模式，未写入任何数据")
        return
    if args.skip_bins:
        print("\n[2] 跳过 .bin 重写（--skip-bins）")
    else:
        print(f"\n[2] 重写 {len(codes)} 只股票的 .bin（全历史覆盖）...")
        repair_bins(codes, calendar)

    if not args.skip_factors:
        print("\n[3] 重算受影响股票的仓库因子...")
        repair_factors(codes, calendar_start, calendar_end,
                       missing_only=args.missing_only)

    print("\n[4] 复检...")
    remaining = scan_affected(client, calendar)
    print(f"  仍有偏差 {len(remaining)} 只")
    print_affected(remaining)
    print("\n完成：" + ("修复成功，偏差已归零" if not remaining else "仍有残留，请检查日志"))


if __name__ == "__main__":
    main()
