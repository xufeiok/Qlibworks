# -*- coding: utf-8 -*-
"""
主板优先的 Qlib 分阶段重建脚本。

功能:
- 以现有 main_board.txt 为股票范围，从 ClickHouse 重建 Qlib 核心字段。
- 输出到独立目录，不覆盖现有 qlib_data。
- 重建后校验日历、字段完整性、OHLC 关系和源库抽样一致性。

入参:
- --output-root: 重建输出根目录。
- --start-date: 重建起始日期，默认 2010-01-01。
- --end-date: 重建结束日期，默认取 ClickHouse 最新交易日。

边界条件:
- 现有 main_board.txt 中无源库行情记录的股票会被跳过并记录。
- 脚本只负责生成新目录，正式切换需在验收通过后单独执行。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer


EXISTING_MAIN_BOARD = PROJECT_ROOT / "qlib_data" / "instruments" / "main_board.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "qlib_data_rebuild_main_board"
FIELDS = ("open", "high", "low", "close", "volume", "amount", "total_mv", "circ_mv")


def _load_main_board_codes() -> list[str]:
    """读取现有主板股票池，保持代码为大写标准格式。"""
    if not EXISTING_MAIN_BOARD.exists():
        raise FileNotFoundError(f"主板股票池不存在: {EXISTING_MAIN_BOARD}")

    codes: list[str] = []
    for line in EXISTING_MAIN_BOARD.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            codes.append(parts[0].upper())
    return sorted(set(codes))


def _query_source_lifecycle(
    api: QuantDataAPI,
    codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    查询每只股票的行情覆盖与基础生命周期。

    返回字段: ts_code, first_trade, last_trade, list_date, delist_date, has_data。
    """
    parts: list[pd.DataFrame] = []
    for offset in range(0, len(codes), 500):
        batch = codes[offset:offset + 500]
        code_sql = ", ".join(f"'{code}'" for code in batch)
        sql = f"""
        SELECT
            coalesce(nullIf(b.ts_code, ''), p.ts_code) AS ts_code,
            b.list_date AS list_date,
            b.delist_date AS delist_date,
            p.first_trade AS first_trade,
            p.last_trade AS last_trade,
            b.ts_code != '' AS has_basic
        FROM
        (
            SELECT ts_code, list_date, delist_date
            FROM stock_basic
            WHERE ts_code IN ({code_sql})
        ) b
        FULL OUTER JOIN
        (
            SELECT
                ts_code,
                MIN(trade_date) AS first_trade,
                MAX(trade_date) AS last_trade
            FROM daily_prices
            WHERE ts_code IN ({code_sql})
              AND trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            GROUP BY ts_code
        ) p
        ON b.ts_code = p.ts_code
        """
        part = api.query(sql)
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(
            columns=["ts_code", "first_trade", "last_trade", "list_date", "delist_date"]
        )

    lifecycle = pd.concat(parts, ignore_index=True)
    lifecycle["ts_code"] = lifecycle["ts_code"].astype(str).str.upper()
    lifecycle = lifecycle.drop_duplicates(subset=["ts_code"], keep="last")
    lifecycle["has_data"] = lifecycle["first_trade"].notna()
    return lifecycle


def _write_main_board_file(
    output_root: Path,
    lifecycle: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    """写入仅包含主板股票的重建版 instruments/main_board.txt。"""
    target = output_root / "instruments" / "main_board.txt"
    target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for row in lifecycle.itertuples(index=False):
        list_timestamp = (
            pd.Timestamp(row.list_date)
            if pd.notna(row.list_date)
            else pd.Timestamp(row.first_trade)
        )
        list_date = list_timestamp.strftime("%Y-%m-%d")
        if list_date < start_date:
            list_date = start_date

        delist_timestamp = pd.Timestamp(row.delist_date) if pd.notna(row.delist_date) else None
        if (
            delist_timestamp is None
            or delist_timestamp <= pd.Timestamp("1970-01-01")
            or delist_timestamp >= pd.Timestamp("2099-01-01")
        ):
            if not bool(row.has_basic) and pd.Timestamp(row.last_trade) < pd.Timestamp(end_date):
                delist_date = pd.Timestamp(row.last_trade).strftime("%Y-%m-%d")
            else:
                delist_date = "9999-12-31"
        else:
            delist_date = delist_timestamp.strftime("%Y-%m-%d")
        if delist_date < list_date:
            raise ValueError(f"生命周期倒置: {row.ts_code} {list_date} ~ {delist_date}")
        rows.append(f"{row.ts_code}\t{list_date}\t{delist_date}")

    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_bin(path: Path) -> tuple[int, np.ndarray]:
    """读取 Qlib bin 文件，返回起始索引和值数组。"""
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"bin 文件过小: {path}")
    start_index = int(struct.unpack("<f", raw[:4])[0])
    return start_index, np.frombuffer(raw, dtype="<f4", offset=4).copy()


def _validate_qlib(
    api: QuantDataAPI,
    output_root: Path,
    lifecycle: pd.DataFrame,
    end_date: str,
) -> dict:
    """校验重建目录的关键结构和抽样数值。"""
    features_dir = output_root / "features"
    calendar = [
        line.strip()
        for line in (output_root / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not calendar:
        raise RuntimeError("重建日历为空")
    if calendar[-1] != end_date:
        raise RuntimeError(f"重建日历末日异常: {calendar[-1]} != {end_date}")

    expected_codes = set(lifecycle["ts_code"].astype(str))
    actual_codes = {path.name.upper() for path in features_dir.iterdir() if path.is_dir()}
    missing_codes = sorted(expected_codes - actual_codes)
    if missing_codes:
        raise RuntimeError(f"重建后仍缺少 {len(missing_codes)} 只股票: {missing_codes[:10]}")

    issues: list[str] = []
    infinite_count = 0
    ohlc_violation_count = 0
    active_stale: list[str] = []
    calendar_map = {date: index for index, date in enumerate(calendar)}

    for code in sorted(expected_codes):
        stock_dir = features_dir / code.lower()
        values: dict[str, tuple[int, np.ndarray]] = {}
        for field in FIELDS:
            path = stock_dir / f"{field}.day.bin"
            if not path.exists():
                issues.append(f"{code} 缺少 {field}")
                continue
            values[field] = _read_bin(path)

        if len(values) != len(FIELDS):
            continue

        for field, (_, array) in values.items():
            infinite_count += int(np.isinf(array).sum())

        starts = [values[field][0] for field in ("open", "high", "low", "close")]
        arrays = [values[field][1] for field in ("open", "high", "low", "close")]
        base = min(starts)
        size = max(start + len(array) for start, array in zip(starts, arrays)) - base
        aligned = []
        for start, array in zip(starts, arrays):
            target = np.full(size, np.nan, dtype=np.float32)
            target[start - base:start - base + len(array)] = array
            aligned.append(target)

        open_arr, high_arr, low_arr, close_arr = aligned
        valid = (
            np.isfinite(open_arr)
            & np.isfinite(high_arr)
            & np.isfinite(low_arr)
            & np.isfinite(close_arr)
        )
        bad_ohlc = valid & (
            (open_arr <= 0)
            | (high_arr <= 0)
            | (low_arr <= 0)
            | (close_arr <= 0)
            | (high_arr < np.maximum.reduce([open_arr, low_arr, close_arr]))
            | (low_arr > np.minimum.reduce([open_arr, high_arr, close_arr]))
        )
        ohlc_violation_count += int(bad_ohlc.sum())

        close_start, close_array = values["close"]
        valid_indexes = np.flatnonzero(np.isfinite(close_array))
        if valid_indexes.size == 0:
            active_stale.append(code)
            continue
        last_valid_index = close_start + int(valid_indexes[-1])
        last_valid_date = calendar[min(last_valid_index, len(calendar) - 1)]
        expected_last = lifecycle.loc[
            lifecycle["ts_code"] == code, "last_trade"
        ].iloc[0]
        if pd.notna(expected_last):
            expected_last_date = pd.Timestamp(expected_last).strftime("%Y-%m-%d")
            if last_valid_date != expected_last_date:
                active_stale.append(
                    f"{code}(实际={last_valid_date}, 源库={expected_last_date})"
                )

    if infinite_count:
        issues.append(f"无穷值: {infinite_count}")
    if ohlc_violation_count:
        issues.append(f"OHLC 关系异常: {ohlc_violation_count}")
    if active_stale:
        issues.append(f"末日不一致: {len(active_stale)} 只 {active_stale[:5]}")

    # 抽取最近交易日与源库逐股对照 close/volume。
    sample_codes = sorted(expected_codes)[::max(len(expected_codes) // 80, 1)][:80]
    sample_dates = calendar[-20:]
    code_sql = ", ".join(f"'{code}'" for code in sample_codes)
    date_sql = ", ".join(f"'{date}'" for date in sample_dates)
    source = api.query(f"""
        WITH latest AS (
            SELECT ts_code, argMax(adj_factor, trade_date) AS adj_factor
            FROM daily_adj_factors
            GROUP BY ts_code
        )
        SELECT
            p.ts_code AS ts_code,
            p.trade_date AS trade_date,
            CAST(p.close * a.adj_factor / latest.adj_factor AS DOUBLE) AS src_close,
            CAST(p.vol AS DOUBLE) AS src_volume
        FROM daily_prices p
        JOIN daily_adj_factors a
          ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
        JOIN latest ON p.ts_code = latest.ts_code
        WHERE p.ts_code IN ({code_sql})
          AND p.trade_date IN ({date_sql})
    """)

    mismatch = 0
    compared = 0
    for row in source.itertuples(index=False):
        code = str(row.ts_code)
        date = pd.Timestamp(row.trade_date).strftime("%Y-%m-%d")
        index = calendar_map[date]
        close_start, close_array = _read_bin(features_dir / code.lower() / "close.day.bin")
        volume_start, volume_array = _read_bin(features_dir / code.lower() / "volume.day.bin")
        if not (close_start <= index < close_start + len(close_array)):
            mismatch += 1
            continue
        if not (volume_start <= index < volume_start + len(volume_array)):
            mismatch += 1
            continue

        local_close = float(close_array[index - close_start])
        local_volume = float(volume_array[index - volume_start])
        source_close = float(row.src_close)
        source_volume = float(row.src_volume)
        compared += 1
        if not np.isclose(local_close, source_close, rtol=1e-5, atol=1e-5):
            mismatch += 1
        elif not np.isclose(local_volume, source_volume, rtol=1e-5, atol=1e-5):
            mismatch += 1

    if mismatch:
        issues.append(f"源库抽样不一致: {mismatch}/{compared}")

    return {
        "calendar_days": len(calendar),
        "start_date": calendar[0],
        "end_date": calendar[-1],
        "feature_stocks": len(actual_codes),
        "infinite_count": infinite_count,
        "ohlc_violation_count": ohlc_violation_count,
        "source_sample_compared": compared,
        "source_sample_mismatch": mismatch,
        "issues": issues,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="主板优先的 Qlib 分阶段重建")
    parser.add_argument("--start-date", default="2010-01-01", help="重建起始日期")
    parser.add_argument("--end-date", default=None, help="重建结束日期，默认取源库最新交易日")
    parser.add_argument("--output-root", default=None, help="输出根目录")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path(f"{DEFAULT_OUTPUT_ROOT}_{timestamp}")
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"输出目录非空，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("主板优先 Qlib 分阶段重建")
    print(f"输出目录: {output_root}")
    print("=" * 72)

    api = QuantDataAPI()
    source_end = api.query("SELECT max(trade_date) AS end_date FROM daily_prices").iloc[0, 0]
    end_date = args.end_date or pd.Timestamp(source_end).strftime("%Y-%m-%d")

    codes = _load_main_board_codes()
    print(f"[1] 现有主板代码: {len(codes)}")
    lifecycle = _query_source_lifecycle(api, codes, args.start_date, end_date)
    lifecycle = lifecycle[lifecycle["has_data"]].copy()
    lifecycle["last_trade"] = pd.to_datetime(lifecycle["last_trade"])
    skipped = sorted(set(codes) - set(lifecycle["ts_code"]))
    print(f"[2] 有源库行情: {len(lifecycle)}，跳过: {len(skipped)} {skipped[:10]}")

    instruments_dict = {
        row.ts_code: max(
            args.start_date,
            pd.Timestamp(row.first_trade).strftime("%Y-%m-%d"),
        )
        for row in lifecycle.itertuples(index=False)
    }

    sync = QlibSynchronizer(api)
    sync.qlib_dir = output_root
    sync.features_dir = output_root / "features"
    sync.instruments_dir = output_root / "instruments"
    sync.calendars_dir = output_root / "calendars"
    sync.full_sync(
        start_date=args.start_date,
        end_date=end_date,
        instruments=sorted(lifecycle["ts_code"].tolist()),
        instruments_dict=instruments_dict,
    )
    _write_main_board_file(output_root, lifecycle, args.start_date, end_date)

    print("\n[7] 校验重建结果...")
    report = _validate_qlib(api, output_root, lifecycle, end_date)
    report.update(
        {
            "source_end_date": end_date,
            "requested_main_board": len(codes),
            "rebuilt_main_board": len(lifecycle),
            "skipped_codes": skipped,
            "output_root": str(output_root),
        }
    )
    report_path = output_root / "rebuild_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["issues"]:
        raise RuntimeError(f"重建校验未通过，详见: {report_path}")
    print(f"\n重建校验通过: {report_path}")


if __name__ == "__main__":
    main()
