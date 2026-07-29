"""
2024 年 OHLCV 数据修复脚本

问题诊断：
- 标签 Ref($close,-5)/Ref($open,-1)-1 在 2024 年出现极大异常值（max|label|=9918）
- 根因：部分日期的 open 数据被写成了 ~0.0001 级的极小值（正常值 ~1-50）
- 最严重日期为 2024-10-10 前后（next_open 畸小）

修复策略：
1. 逐股扫描 open.day.bin / close.day.bin 在 2024 年的数据
2. 认定异常：当前值 < 0.1，且前后紧邻值均 > 1.0 → 确实是写坏而非退市股
3. 修复方式：将该值设为前一个有效值的前向填充
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QLIB_DATA_DIR = PROJECT_ROOT / "qlib_data"
FEATURES_DIR = QLIB_DATA_DIR / "features"
CALENDAR_PATH = QLIB_DATA_DIR / "calendars" / "day.txt"

FIX_START = "2024-01-02"  # 2024-01-01 非交易日
FIX_END = "2024-12-31"
MIN_PRICE_THRESHOLD = 0.1  # 低于此值的价格视为异常
ADJACENT_MIN = 1.0  # 前后紧邻值必须大于此值才判定为异常

FIELDS_TO_FIX = ["open", "close"]


def load_calendar() -> list[str]:
    with open(CALENDAR_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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


def fix_stock_field(
    stock_dir: Path,
    field: str,
    cal_slice: slice,
    cal_offset: int,
) -> int:
    """修复单只股票单个字段。返回修复的异常值数量。"""
    bin_path = stock_dir / f"{field}.day.bin"
    if not bin_path.exists():
        return 0

    start_idx, data = read_bin(bin_path)
    if start_idx is None or len(data) == 0:
        return 0

    # 只操作 2024 年范围内的数据
    local_start = max(0, cal_offset - start_idx)
    local_end = min(len(data), cal_offset + (cal_slice.stop - cal_slice.start) - start_idx)
    if local_start >= local_end or local_start >= len(data):
        return 0

    segment = data[local_start:local_end]
    if len(segment) == 0:
        return 0

    fix_count = 0
    for i in range(len(segment)):
        abs_idx = local_start + i
        val = float(data[abs_idx])

        # 跳过 NaN 或正常值
        if np.isnan(val) or val >= MIN_PRICE_THRESHOLD:
            continue

        # 检查前后紧邻值，确认这是写坏而非退市股
        prev_ok = (
            abs_idx > 0
            and not np.isnan(data[abs_idx - 1])
            and float(data[abs_idx - 1]) >= ADJACENT_MIN
        )
        next_ok = (
            abs_idx < len(data) - 1
            and not np.isnan(data[abs_idx + 1])
            and float(data[abs_idx + 1]) >= ADJACENT_MIN
        )
        if not (prev_ok or next_ok):
            continue

        # 修复：前向填充
        if prev_ok:
            data[abs_idx] = data[abs_idx - 1]
        elif next_ok:
            data[abs_idx] = data[abs_idx + 1]
        fix_count += 1

    if fix_count > 0:
        write_bin(bin_path, start_idx, data)

    return fix_count


def main() -> None:
    print("=" * 60)
    print("  2024 年 OHLCV 数据修复")
    print("=" * 60)

    calendar = load_calendar()
    try:
        cal_start = calendar.index(FIX_START)
        cal_end = calendar.index(FIX_END)
    except ValueError as e:
        print(f"日历中找不到日期边界: {e}")
        sys.exit(1)

    cal_slice = slice(cal_start, cal_end + 1)
    cal_offset = cal_start

    stock_dirs = sorted(FEATURES_DIR.iterdir())
    stock_dirs = [d for d in stock_dirs if d.is_dir() and len(d.name) == 9]

    total_fixed = {f: 0 for f in FIELDS_TO_FIX}
    total_checked = 0
    total_affected_stocks = {f: 0 for f in FIELDS_TO_FIX}

    print(f"\n日历索引范围: {cal_start} ~ {cal_end} ({FIX_START} ~ {FIX_END})")
    print(f"待扫描股票: {len(stock_dirs)} 只\n")

    for stock_dir in stock_dirs:
        symbol = stock_dir.name
        total_checked += 1

        for field in FIELDS_TO_FIX:
            fixed = fix_stock_field(stock_dir, field, cal_slice, cal_offset)
            if fixed > 0:
                total_fixed[field] += fixed
                total_affected_stocks[field] += 1

        if total_checked % 500 == 0:
            print(f"  已扫描 {total_checked}/{len(stock_dirs)} 只...")

    print(f"\n扫描完成: {total_checked} 只股票")
    for field in FIELDS_TO_FIX:
        print(f"  {field}: 修复 {total_fixed[field]} 个异常值，涉及 {total_affected_stocks[field]} 只股票")

    # 写入修复日志
    log_path = PROJECT_ROOT / "runtime" / "diagnostics" / "fix_2024_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"2024 OHLCV 数据修复日志\n")
        f.write(f"扫描股票: {total_checked}\n")
        for field in FIELDS_TO_FIX:
            f.write(f"{field}: 修复 {total_fixed[field]} 个异常值, {total_affected_stocks[field]} 只股票\n")
    print(f"\n修复日志: {log_path}")


if __name__ == "__main__":
    main()
