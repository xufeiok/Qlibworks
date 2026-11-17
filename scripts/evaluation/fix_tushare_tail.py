# -*- coding: utf-8 -*-
"""
补齐 3 个 Tushare 专属因子到最新交易日。

覆盖因子:
- inv_turn               : 存货周转率（季报公告日 forward-fill 到日频）
- block_trade_premium_20d: 大宗交易折溢价 20 次事件滚动均值（稀疏事件日序列）
- unlock_pressure_60d    : 未来 60 日限售解禁比例合计（日频，仅保留 pressure>0）

实现口径与原脚本一致:
- inv_turn 复用 rebuild_inv_turn_tushare._build_daily_series；
- block_trade 逐票取最近 30 笔事件，按 ts_code/trade_date 求 premium 均值后
  rolling(20, min_periods=1)（按事件日滚动）；
- unlock 扫描 share_float 公告事件，若 float_date ∈ (d, d+60d] 则计入 d 日压力。

写入方式: FactorStore.save_warehouse_frame（按索引去重合并，不删除历史数据）。

用法:
  python -m scripts.evaluation.fix_tushare_tail --factor inv_turn
  python -m scripts.evaluation.fix_tushare_tail --factor block_trade
  python -m scripts.evaluation.fix_tushare_tail --factor unlock
  python -m scripts.evaluation.fix_tushare_tail --factor all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dotenv

# 先加载 .env，保证 TUSHARE_TOKEN 可用（不覆盖已有环境变量）
dotenv.load_dotenv(dotenv_path=str(ROOT / ".env"), override=False)

import tushare as ts

from qlworks.evaluation import DEFAULT_CONFIG, FactorStore
from rebuild_inv_turn_tushare import _build_daily_series, _read_codes

TARGET_DATE = "2026-11-06"
WAREHOUSE = ROOT / "factor_data" / "warehouse"
CACHE_ROOT = ROOT / "factor_data" / "cache"
CH_CONFIG = dict(host="192.168.10.102", port=18123, username="reader",
                 password="123", database="quant_db")


# ────────────────────────── 公共工具 ──────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补齐 Tushare 专属因子")
    parser.add_argument("--factor", default="all",
                        choices=["all", "inv_turn", "block_trade", "unlock"])
    parser.add_argument("--rebuild", action="store_true",
                        help="对 block_trade/unlock 执行全量重建（修复 Tushare 1000 行截断）")
    parser.add_argument("--rebuild-start", default="2015-01-01")
    parser.add_argument("--end-date", default=TARGET_DATE)
    parser.add_argument("--qlib-dir", default=str(ROOT / "qlib_data"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--scan-start", default="2024-01-01",
                        help="unlock 公告日扫描起点")
    parser.add_argument("--validate-samples", type=int, default=60,
                        help="unlock 抽样校验股票数")
    return parser.parse_args()


def _calendar(qlib_dir: Path) -> pd.DatetimeIndex:
    """读取 Qlib 交易日历。"""
    path = qlib_dir / "calendars" / "day.txt"
    return pd.DatetimeIndex([
        pd.Timestamp(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ])


def _store() -> FactorStore:
    """返回指向项目默认仓库的 FactorStore。"""
    return FactorStore(DEFAULT_CONFIG)


def _existing(store: FactorStore, name: str) -> pd.DataFrame:
    """读取仓库存量数据（不存在返回空 DataFrame）。"""
    df = store._load_warehouse_data(name)
    return df if df is not None else pd.DataFrame()


def _merge(store: FactorStore, name: str, df: pd.DataFrame) -> None:
    """增量合并写回并刷新 meta。"""
    if df.empty:
        print(f"  [WARN] {name} 无新增数据")
        return
    if df.index.names != ["instrument", "datetime"]:
        df = df.set_index(["instrument", "datetime"])
    df = df.sort_index().astype({"value": "float32"})
    stats = store.save_warehouse_frame(name, df, replace_years=None, dedup_keep="last")
    total = sum(stats.values())
    meta = store.get_warehouse_meta(name) or {}
    rng = meta.get("data_range", {})
    print(f"  [OK] {name}: 新增 {len(df):,} 行 | 合并后 {total:,} 行 | "
          f"{rng.get('first_date')} ~ {rng.get('last_date')} | "
          f"股票 {rng.get('unique_stocks')}")


def _fetch_parallel(pro, tasks, worker_fn, workers: int, desc: str):
    """通用并发拉取：tasks 为参数列表，worker_fn 返回 (key, error)。"""
    failures = {}
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {pool.submit(worker_fn, pro, task): task for task in tasks}
        done = 0
        for fut in as_completed(futures):
            key, err = fut.result()
            if err:
                failures[key] = err
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {desc}: {done}/{len(tasks)}，失败 {len(failures)}")
    return failures


# ────────────────────────── inv_turn ──────────────────────────

def _cache_last_ann(path: Path) -> pd.Timestamp | None:
    """读取缓存文件中最新公告日。"""
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if frame.empty or "ann_date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["ann_date"], errors="coerce").dropna()
    return dates.max() if len(dates) else None


def _refresh_inv_turn(pro, task):
    """重拉单只股票 inv_turn 全历史并覆盖缓存。"""
    code, start, end, cache_dir, sleep = task
    path = cache_dir / f"{code}.parquet"
    last_error = None
    for attempt in range(3):
        try:
            frame = pro.fina_indicator(
                ts_code=code, start_date=start, end_date=end,
                fields="ts_code,ann_date,end_date,inv_turn",
            )
            frame = frame if frame is not None else pd.DataFrame()
            frame.to_parquet(path, index=False, compression="zstd")
            time.sleep(sleep)
            return code, None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(sleep * (attempt + 1))
    return code, last_error


def update_inv_turn(pro, store: FactorStore, args) -> None:
    """刷新过期缓存后重建 inv_turn 日频序列并合并。"""
    name = "inv_turn"
    end = pd.Timestamp(args.end_date)
    start = pd.Timestamp("2020-01-01")  # 与原始构建口径一致
    stale_before = end - pd.Timedelta(days=20)

    cache_dir = CACHE_ROOT / "tushare_inv_turn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    codes = _read_codes(Path(args.qlib_dir) / "instruments" / "main_board.txt")
    stale = []
    for code in codes:
        path = cache_dir / f"{code}.parquet"
        if not path.exists():
            stale.append(code)
            continue
        last = _cache_last_ann(path)
        if last is None or last < stale_before:
            stale.append(code)
    print(f"[inv_turn] 股票 {len(codes)} 只，缓存需刷新 {len(stale)} 只")

    if stale:
        tasks = [(code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
                  cache_dir, args.sleep_seconds) for code in stale]
        failures = _fetch_parallel(pro, tasks, _refresh_inv_turn,
                                   args.workers, "刷新缓存")
        if failures:
            print(f"  [WARN] 刷新失败 {len(failures)} 只，沿用旧缓存")

    calendar = _calendar(Path(args.qlib_dir))
    parts, empty = [], 0
    for code in codes:
        part = _build_daily_series(cache_dir, code, calendar, start, end)
        if part is None:
            empty += 1
        else:
            parts.append(part)
    if not parts:
        raise RuntimeError("inv_turn 未生成任何数据")
    result = pd.concat(parts).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    print(f"  重建序列 {len(result):,} 行，空股票 {empty} 只")
    _merge(store, name, result)


# ────────────────────────── block_trade ──────────────────────────

def _ch_close(codes: list[str], min_date: str) -> pd.DataFrame:
    """从 ClickHouse 读取指定股票集合的收盘价。"""
    import clickhouse_connect

    client = clickhouse_connect.get_client(**CH_CONFIG)
    frames = []
    for i in range(0, len(codes), 300):
        chunk = codes[i:i + 300]
        code_list = ",".join(f"'{c}'" for c in chunk)
        sql = (
            "SELECT ts_code, trade_date, close FROM daily_prices "
            f"WHERE ts_code IN ({code_list}) AND trade_date >= '{min_date}'"
        )
        rows = client.query(sql).result_rows
        if rows:
            frames.append(pd.DataFrame(rows, columns=["ts_code", "trade_date", "close"]))
    client.close()
    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out


def _fetch_block_trade(pro, task):
    """逐票拉取大宗交易（Tushare 单次最多 1000 行，足够覆盖最近 30 笔）。"""
    code, cache_dir, sleep = task
    path = cache_dir / f"{code}.parquet"
    last_error = None
    for attempt in range(3):
        try:
            frame = pro.block_trade(ts_code=code)
            frame = frame if frame is not None else pd.DataFrame()
            frame.to_parquet(path, index=False, compression="zstd")
            time.sleep(sleep)
            return code, None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(sleep * (attempt + 1))
    return code, last_error


def update_block_trade(pro, store: FactorStore, args) -> None:
    """补齐大宗交易折溢价因子的新增事件日。"""
    name = "block_trade_premium_20d"
    end = pd.Timestamp(args.end_date)
    existing = _existing(store, name)
    if existing.empty:
        raise RuntimeError(f"{name} 仓库为空，需先全量构建")
    last = existing.index.get_level_values("datetime").max()
    calendar = _calendar(Path(args.qlib_dir))
    new_dates = [d for d in calendar if last < d <= end]
    print(f"[{name}] 仓库末日 {last.date()}，待补 {len(new_dates)} 个交易日")
    if not new_dates:
        return

    # 1) 先按交易日拿到新增事件的股票集合
    rows = []
    for d in new_dates:
        frame = pro.block_trade(trade_date=d.strftime("%Y%m%d"))
        if frame is not None and len(frame):
            rows.append(frame)
        time.sleep(args.sleep_seconds)
    if not rows:
        print("  [WARN] 目标区间无大宗交易事件")
        return
    events = pd.concat(rows, ignore_index=True)
    events["trade_date"] = pd.to_datetime(events["trade_date"])
    codes = sorted(events["ts_code"].unique())
    print(f"  新增事件 {len(events)} 笔，涉及股票 {len(codes)} 只")

    # 2) 逐票取历史事件（缓存），计算逐事件折溢价
    cache_dir = CACHE_ROOT / "tushare_block_trade"
    cache_dir.mkdir(parents=True, exist_ok=True)
    failures = _fetch_parallel(
        pro, [(c, cache_dir, args.sleep_seconds) for c in codes],
        _fetch_block_trade, args.workers, "拉取事件")
    if failures:
        print(f"  [WARN] 拉取失败 {len(failures)} 只: {list(failures)[:5]}")

    parts = [pd.read_parquet(cache_dir / f"{c}.parquet") for c in codes
             if (cache_dir / f"{c}.parquet").exists()]
    if not parts:
        raise RuntimeError("未取得任何大宗交易事件")
    bt = pd.concat(parts, ignore_index=True)
    bt = bt.drop_duplicates(subset=["ts_code", "trade_date", "price", "vol"])
    bt["trade_date"] = pd.to_datetime(bt["trade_date"])
    bt["price"] = pd.to_numeric(bt["price"], errors="coerce")
    # 仅保留每票最近 30 笔事件：rolling(20) 只需最近 20 笔，30 笔可同时校验存量口径
    bt = (bt.sort_values(["ts_code", "trade_date"])
            .groupby("ts_code").tail(30).reset_index(drop=True))

    # 3) 合并收盘价并计算 premium
    close = _ch_close(codes, (bt["trade_date"].min()).strftime("%Y%m%d"))
    bt = bt.merge(close, on=["ts_code", "trade_date"], how="left")
    bt["premium"] = (bt["price"] - bt["close"]) / bt["close"].replace(0, np.nan)
    agg = (bt.groupby(["ts_code", "trade_date"])["premium"].mean()
             .reset_index().sort_values(["ts_code", "trade_date"]))
    agg["premium_20d"] = agg.groupby("ts_code")["premium"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())

    # 4) 仅保留新增日期的数据
    new_set = set(new_dates)
    inc = agg[agg["trade_date"].isin(new_set)].dropna(subset=["premium_20d"])
    inc = (inc.rename(columns={"ts_code": "instrument", "trade_date": "datetime"})
              .set_index(["instrument", "datetime"])[["premium_20d"]]
              .rename(columns={"premium_20d": "value"}))

    # 5) 重叠区间校验（取最近 5 个已存在事件日）
    check = _validate_block_trade(agg, existing, last)
    print(f"  口径校验: 比对 {check['n']} 行，最大偏差 {check['max_diff']:.2e}，"
          f"1e-6 匹配率 {check['match']:.2%}")
    _merge(store, name, inc)


def _validate_block_trade(agg: pd.DataFrame, existing: pd.DataFrame,
                          last: pd.Timestamp) -> dict:
    """用逐票重算结果校验仓库存量（最近若干事件日）。"""
    recent_dates = sorted(existing.index.get_level_values("datetime").unique())[-5:]
    sub = agg[agg["trade_date"].isin(recent_dates)]
    sub = sub.rename(columns={"ts_code": "instrument", "trade_date": "datetime"})
    sub = sub.set_index(["instrument", "datetime"])["premium_20d"]
    old = existing.loc[existing.index.isin(sub.index), "value"]
    both = pd.concat([old.rename("old"), sub.rename("new")], axis=1).dropna()
    if both.empty:
        return {"n": 0, "max_diff": float("nan"), "match": float("nan")}
    diff = (both["old"] - both["new"]).abs()
    return {"n": len(both), "max_diff": float(diff.max()),
            "match": float((diff < 1e-6).mean())}


# ────────────────────────── 全量重建（修复 1000 行截断） ──────────────────────────

def _ch_close_by_dates(events: pd.DataFrame) -> pd.DataFrame:
    """按事件日逐日读取收盘价（仅取当日涉事股票，避免全市场拉取）。"""
    import clickhouse_connect

    client = clickhouse_connect.get_client(**CH_CONFIG)
    frames = []
    for date, grp in events.groupby("trade_date"):
        codes = sorted(grp["ts_code"].unique())
        for i in range(0, len(codes), 500):
            chunk = codes[i:i + 500]
            code_list = ",".join(f"'{c}'" for c in chunk)
            sql = (
                "SELECT ts_code, trade_date, close FROM daily_prices "
                f"WHERE trade_date = '{date:%Y-%m-%d}' AND ts_code IN ({code_list})"
            )
            rows = client.query(sql).result_rows
            if rows:
                frames.append(pd.DataFrame(rows, columns=["ts_code", "trade_date", "close"]))
    client.close()
    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.drop_duplicates(subset=["ts_code", "trade_date"])


def _fetch_block_trade_day(pro, task):
    """按交易日拉取全市场大宗交易（单日行数远小于 1000，无截断）。"""
    day, cache_dir, sleep = task
    path = cache_dir / f"{day}.parquet"
    if path.exists():
        return day, None
    last_error = None
    for attempt in range(3):
        try:
            frame = pro.block_trade(trade_date=day)
            frame = frame if frame is not None else pd.DataFrame()
            frame.to_parquet(path, index=False, compression="zstd")
            time.sleep(sleep)
            return day, None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(sleep * (attempt + 1))
    return day, last_error


def rebuild_block_trade(pro, store: FactorStore, args) -> None:
    """用逐日全市场事件全量重建 block_trade_premium_20d。"""
    name = "block_trade_premium_20d"
    end = pd.Timestamp(args.end_date)
    start = pd.Timestamp(args.rebuild_start)
    calendar = _calendar(Path(args.qlib_dir))
    days = [d for d in calendar if start <= d <= end]
    cache_dir = CACHE_ROOT / "tushare_block_trade_daily"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] 全量重建：扫描 {len(days)} 个交易日（{days[0].date()} ~ {days[-1].date()}）")

    tasks = [(d.strftime("%Y%m%d"), cache_dir, args.sleep_seconds) for d in days]
    failures = _fetch_parallel(pro, tasks, _fetch_block_trade_day,
                               args.workers, "拉取日频事件")
    if failures:
        raise RuntimeError(
            f"仍有 {len(failures)} 个交易日拉取失败，已缓存成功部分，请降低并发后重跑")

    parts = [pd.read_parquet(cache_dir / f"{d}.parquet") for d in
             (x.strftime("%Y%m%d") for x in days) if (cache_dir / f"{d}.parquet").exists()]
    if not parts:
        raise RuntimeError("未取得任何大宗交易事件")
    ev = pd.concat(parts, ignore_index=True)
    ev = ev.drop_duplicates()
    ev["trade_date"] = pd.to_datetime(ev["trade_date"])
    ev["price"] = pd.to_numeric(ev["price"], errors="coerce")
    ev = ev.dropna(subset=["price"]).sort_values(["ts_code", "trade_date"])
    print(f"  事件 {len(ev):,} 笔，股票 {ev['ts_code'].nunique():,} 只，"
          f"交易日 {ev['trade_date'].nunique()} 天")

    close = _ch_close_by_dates(ev[["ts_code", "trade_date"]])
    ev = ev.merge(close, on=["ts_code", "trade_date"], how="left")
    ev["premium"] = (ev["price"] - ev["close"]) / ev["close"].replace(0, np.nan)
    agg = (ev.groupby(["ts_code", "trade_date"])["premium"].mean()
             .reset_index().sort_values(["ts_code", "trade_date"]))
    agg["premium_20d"] = agg.groupby("ts_code")["premium"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    out = (agg.dropna(subset=["premium_20d"])
              .rename(columns={"ts_code": "instrument", "trade_date": "datetime"})
              .set_index(["instrument", "datetime"])[["premium_20d"]]
              .rename(columns={"premium_20d": "value"})
              .sort_index())
    print(f"  重建结果 {len(out):,} 行，股票 "
          f"{out.index.get_level_values('instrument').nunique():,} 只")

    years = set(range(start.year, end.year + 1))
    store.save_warehouse_frame(name, out.astype({"value": "float32"}),
                               replace_years=years, dedup_keep="last")
    meta = store.get_warehouse_meta(name) or {}
    print(f"  [OK] {name}: {meta.get('data_range')}")


def _fetch_share_float_stock(pro, task):
    """逐票拉取限售解禁全历史事件（限流失败时退避重试）。"""
    code, cache_dir, sleep = task
    path = cache_dir / f"{code}.parquet"
    if path.exists():
        return code, None
    last_error = None
    for attempt in range(4):
        try:
            frame = pro.share_float(ts_code=code)
            frame = frame if frame is not None else pd.DataFrame()
            frame.to_parquet(path, index=False, compression="zstd")
            time.sleep(sleep)
            return code, None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(3.0 * (attempt + 1))
    return code, last_error


def rebuild_unlock(pro, store: FactorStore, args) -> None:
    """用逐票全历史解禁事件全量重建 unlock_pressure_60d。"""
    name = "unlock_pressure_60d"
    end = pd.Timestamp(args.end_date)
    start = pd.Timestamp(args.rebuild_start)
    codes = _read_codes(Path(args.qlib_dir) / "instruments" / "all.txt")
    cache_dir = CACHE_ROOT / "tushare_share_float_by_stock"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] 全量重建：股票 {len(codes)} 只")

    failures = _fetch_parallel(pro, [(c, cache_dir, args.sleep_seconds) for c in codes],
                               _fetch_share_float_stock, args.workers, "拉取解禁事件")
    if failures:
        raise RuntimeError(
            f"仍有 {len(failures)} 只股票拉取失败，已缓存成功部分，请降低并发后重跑")

    calendar = _calendar(Path(args.qlib_dir))
    dates = np.array([d for d in calendar if start <= d <= end], dtype="datetime64[ns]")
    rows = []
    for code in codes:
        path = cache_dir / f"{code}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["float_date"] = pd.to_datetime(frame["float_date"], errors="coerce")
        frame["float_ratio"] = pd.to_numeric(frame["float_ratio"], errors="coerce")
        frame = frame.dropna(subset=["float_date", "float_ratio"])
        if frame.empty:
            continue
        frame = frame.sort_values("float_date")
        fd = frame["float_date"].to_numpy(dtype="datetime64[ns]")
        cum = np.concatenate([[0.0], np.cumsum(frame["float_ratio"].to_numpy(dtype="float64"))])
        # pressure = sum(float_ratio where float_date ∈ (d, d+60d])
        i = np.searchsorted(fd, dates, side="right")
        j = np.searchsorted(fd, dates + np.timedelta64(60, "D"), side="right")
        pressure = cum[j] - cum[i]
        mask = pressure > 0
        if not mask.any():
            continue
        rows.append(pd.DataFrame({
            "instrument": code,
            "datetime": pd.DatetimeIndex(dates[mask]),
            "value": pressure[mask],
        }))
    if not rows:
        raise RuntimeError("未生成任何解禁压力数据")
    out = (pd.concat(rows, ignore_index=True)
             .set_index(["instrument", "datetime"])
             .sort_index())
    print(f"  重建结果 {len(out):,} 行，股票 "
          f"{out.index.get_level_values('instrument').nunique():,} 只")

    years = set(range(start.year, end.year + 1))
    store.save_warehouse_frame(name, out.astype({"value": "float32"}),
                               replace_years=years, dedup_keep="last")
    meta = store.get_warehouse_meta(name) or {}
    print(f"  [OK] {name}: {meta.get('data_range')}")


# ────────────────────────── unlock ──────────────────────────

def _fetch_share_float(pro, task):
    """按公告日拉取限售解禁事件。"""
    day, cache_dir, sleep = task
    path = cache_dir / f"{day}.parquet"
    if path.exists():
        return day, None
    last_error = None
    for attempt in range(3):
        try:
            frame = pro.share_float(ann_date=day)
            frame = frame if frame is not None else pd.DataFrame()
            frame.to_parquet(path, index=False, compression="zstd")
            time.sleep(sleep)
            return day, None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(sleep * (attempt + 1))
    return day, last_error


def update_unlock(pro, store: FactorStore, args) -> None:
    """补齐未来 60 日限售解禁压力。"""
    name = "unlock_pressure_60d"
    end = pd.Timestamp(args.end_date)
    existing = _existing(store, name)
    if existing.empty:
        raise RuntimeError(f"{name} 仓库为空，需先全量构建")
    last = existing.index.get_level_values("datetime").max()
    calendar = _calendar(Path(args.qlib_dir))
    new_dates = [d for d in calendar if last < d <= end]
    print(f"[{name}] 仓库末日 {last.date()}，待补 {len(new_dates)} 个交易日")
    if not new_dates:
        return

    # 1) 扫描公告事件（公告日可能落在非交易日，需按自然日扫描）
    cache_dir = CACHE_ROOT / "tushare_share_float"
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = [d.strftime("%Y%m%d") for d in
            pd.date_range(args.scan_start, end, freq="D")]
    tasks = [(d, cache_dir, args.sleep_seconds) for d in days]
    print(f"  扫描公告日 {len(days)} 天（{args.scan_start} ~ {args.end_date}）")
    failures = _fetch_parallel(pro, tasks, _fetch_share_float,
                               args.workers, "扫描公告")
    if failures:
        print(f"  [WARN] 公告日拉取失败 {len(failures)} 天")

    parts = [pd.read_parquet(cache_dir / f"{d}.parquet") for d in days
             if (cache_dir / f"{d}.parquet").exists()]
    if not parts:
        raise RuntimeError("未取得任何解禁事件")
    ev = pd.concat(parts, ignore_index=True)
    ev = ev.drop_duplicates(subset=["ts_code", "ann_date", "float_date",
                                    "holder_name", "float_ratio"])
    ev["ann_date"] = pd.to_datetime(ev["ann_date"], errors="coerce")
    ev["float_date"] = pd.to_datetime(ev["float_date"], errors="coerce")
    ev["float_ratio"] = pd.to_numeric(ev["float_ratio"], errors="coerce")
    ev = ev.dropna(subset=["float_date", "float_ratio"])

    # 2) 只保留可能影响目标日期的未来解禁事件
    horizon = max(new_dates) + pd.Timedelta(days=60)
    win = ev[(ev["float_date"] > last) & (ev["float_date"] <= horizon)]
    print(f"  事件总数 {len(ev):,}，命中窗口 {len(win)} 笔，"
          f"涉及股票 {win['ts_code'].nunique()} 只")

    # 3) 逐票汇总未来 60 日解禁比例
    rows = []
    for code, grp in win.groupby("ts_code"):
        for d in new_dates:
            pressure = grp.loc[(grp["float_date"] > d) &
                               (grp["float_date"] <= d + pd.Timedelta(days=60)),
                               "float_ratio"].sum()
            if pressure > 0:
                rows.append((code, d, float(pressure)))
    inc = (pd.DataFrame(rows, columns=["instrument", "datetime", "value"])
             .set_index(["instrument", "datetime"])
             .sort_index() if rows else pd.DataFrame(
                 columns=["value"], index=pd.MultiIndex.from_arrays(
                     [[], []], names=["instrument", "datetime"])))

    # 4) 完整性校验：随机抽样逐票全历史比对
    _validate_unlock(pro, win, inc, existing, new_dates, args)
    _merge(store, name, inc)


def _validate_unlock(pro, win: pd.DataFrame, inc: pd.DataFrame,
                     existing: pd.DataFrame, new_dates: list[pd.Timestamp],
                     args) -> None:
    """抽样逐票拉全历史事件，校验公告窗口扫描的完整性。"""
    universe = sorted(existing.index.get_level_values("instrument").unique())
    n = min(args.validate_samples, len(universe))
    rng = np.random.default_rng(42)
    sample = rng.choice(universe, size=n, replace=False)
    mismatch = 0
    for code in sample:
        try:
            hist = pro.share_float(ts_code=str(code))
        except Exception:
            continue
        time.sleep(args.sleep_seconds)
        if hist is None or hist.empty:
            continue
        hist = hist.dropna(subset=["float_date", "float_ratio"]).copy()
        hist["float_date"] = pd.to_datetime(hist["float_date"])
        hist["float_ratio"] = pd.to_numeric(hist["float_ratio"], errors="coerce")
        for d in new_dates:
            expected = hist.loc[(hist["float_date"] > d) &
                                (hist["float_date"] <= d + pd.Timedelta(days=60)),
                                "float_ratio"].sum()
            key = (code, d)
            got = inc.loc[key, "value"] if key in inc.index else 0.0
            if abs(float(expected) - float(got)) > 1e-6:
                mismatch += 1
                print(f"  [DIFF] {code} {d.date()} 期望 {expected:.6f} 实得 {got:.6f}")
    print(f"  抽样校验 {n} 只，不一致 {mismatch} 项")


# ────────────────────────── 入口 ──────────────────────────

def main() -> None:
    args = _parse_args()
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    pro = ts.pro_api(token)
    store = _store()
    print(f"仓库目录: {store.warehouse_dir}")

    if args.factor in ("all", "inv_turn"):
        update_inv_turn(pro, store, args)
    if args.factor in ("all", "block_trade"):
        if args.rebuild:
            rebuild_block_trade(pro, store, args)
        else:
            update_block_trade(pro, store, args)
    if args.factor in ("all", "unlock"):
        if args.rebuild:
            rebuild_unlock(pro, store, args)
        else:
            update_unlock(pro, store, args)


if __name__ == "__main__":
    main()
