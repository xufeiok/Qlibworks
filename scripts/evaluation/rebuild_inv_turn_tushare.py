# -*- coding: utf-8 -*-
"""
从 Tushare 补齐主板 inv_turn 因子。

功能:
- 按主板股票逐票获取 fina_indicator.inv_turn。
- 将公告数据按交易日 forward-fill 后写入指定因子仓库。
- 成功响应按股票缓存，支持断点续跑。

边界条件:
- 银行等无存货指标的股票允许为空。
- 仅生成目标因子目录，不修改 ClickHouse。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tushare as ts

from qlworks.evaluation import DEFAULT_CONFIG, FactorStore


def _read_codes(path: Path) -> list[str]:
    """读取 Qlib instruments 股票代码。"""
    codes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            codes.append(parts[0].upper())
    return sorted(set(codes))


def _fetch_one(
    pro,
    code: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    sleep_seconds: float,
) -> tuple[str, str | None]:
    """获取单只股票的 inv_turn，成功结果缓存为 Parquet。"""
    cache_path = cache_dir / f"{code}.parquet"
    if cache_path.exists():
        return code, None

    last_error = None
    for attempt in range(3):
        try:
            frame = pro.fina_indicator(
                ts_code=code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                fields="ts_code,ann_date,end_date,inv_turn",
            )
            frame.to_parquet(cache_path, index=False, compression="zstd")
            time.sleep(sleep_seconds)
            return code, None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(sleep_seconds * (attempt + 1))
    return code, last_error


def _build_daily_series(
    cache_dir: Path,
    code: str,
    calendar: list[pd.Timestamp],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """将公告日因子值按交易日 forward-fill。"""
    cache_path = cache_dir / f"{code}.parquet"
    if not cache_path.exists():
        return None
    frame = pd.read_parquet(cache_path)
    if frame.empty or "inv_turn" not in frame.columns:
        return None

    frame = frame.dropna(subset=["inv_turn", "ann_date"]).copy()
    if frame.empty:
        return None
    frame["ann_date"] = pd.to_datetime(frame["ann_date"])
    frame = (
        frame.sort_values(["ann_date", "end_date"])
        .drop_duplicates(subset=["ann_date"], keep="last")
        .reset_index(drop=True)
    )

    dates = pd.DatetimeIndex(
        [date for date in calendar if start_date <= date <= end_date]
    )
    if len(dates) == 0:
        return None

    report_dates = frame["ann_date"].to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(
        report_dates,
        dates.to_numpy(dtype="datetime64[ns]"),
        side="right",
    ) - 1
    valid = positions >= 0
    if not valid.any():
        return None

    values = np.full(len(dates), np.nan, dtype=np.float32)
    report_values = frame["inv_turn"].to_numpy(dtype=np.float32)
    values[valid] = report_values[positions[valid]]
    result = pd.DataFrame(
        {
            "instrument": code,
            "datetime": dates,
            "value": values,
        }
    ).dropna(subset=["value"])
    if result.empty:
        return None
    return result.set_index(["instrument", "datetime"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Tushare 补齐 inv_turn")
    parser.add_argument("--qlib-dir", default=str(PROJECT_ROOT / "qlib_data"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-09-11")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.6)
    parser.add_argument("--code", nargs="*", default=None, help="仅处理指定股票")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    pro = ts.pro_api(token)

    qlib_dir = Path(args.qlib_dir).resolve()
    codes = _read_codes(qlib_dir / "instruments" / "main_board.txt")
    if args.code:
        requested = {code.upper() for code in args.code}
        codes = [code for code in codes if code in requested]
    calendar = pd.DatetimeIndex(
        [
            pd.Timestamp(line.strip())
            for line in (qlib_dir / "calendars" / "day.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    )
    cache_dir = PROJECT_ROOT / "factor_data" / "cache" / "tushare_inv_turn"
    cache_dir.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    print(f"[1] 获取 inv_turn: {len(codes)} 只主板股票")
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = [
            executor.submit(
                _fetch_one,
                pro,
                code,
                args.start_date,
                args.end_date,
                cache_dir,
                args.sleep_seconds,
            )
            for code in codes
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Tushare inv_turn",
        ):
            code, error = future.result()
            if error:
                failures[code] = error

    print(f"[2] 组装交易日序列，失败 {len(failures)} 只")
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    parts: list[pd.DataFrame] = []
    for code in tqdm(codes, desc="对齐交易日"):
        part = _build_daily_series(
            cache_dir,
            code,
            calendar,
            start_date,
            end_date,
        )
        if part is not None:
            parts.append(part)

    if not parts:
        raise RuntimeError("Tushare 未生成任何 inv_turn 数据")

    result = pd.concat(parts).sort_index()
    result = result[~result.index.duplicated(keep="last")].astype({"value": "float32"})

    config = replace(DEFAULT_CONFIG, warehouse_dir=str(output_root))
    store = FactorStore(config)
    factor_dir = store.warehouse_dir / "inv_turn"
    for path in factor_dir.glob("*.parquet"):
        path.unlink()
    factor_dir.mkdir(parents=True, exist_ok=True)
    result["_year"] = result.index.get_level_values("datetime").year
    for year, part in result.groupby("_year"):
        part = part.drop(columns=["_year"])
        store._save_warehouse_year("inv_turn", int(year), part)
    store._update_warehouse_meta("inv_turn")

    print(
        f"[3] inv_turn 完成: {len(result):,} 行，"
        f"{result.index.get_level_values('instrument').nunique()} 只股票"
    )
    if failures:
        failure_path = output_root / "inv_turn_failures.json"
        failure_path.write_text(
            pd.Series(failures, dtype="object").to_json(
                orient="index",
                force_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"失败明细: {failure_path}")


if __name__ == "__main__":
    main()
