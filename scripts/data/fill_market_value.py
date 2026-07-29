#!/usr/bin/env python
"""
补全 Qlib features 中缺失的 total_mv / circ_mv 数据

数据来源:
  - 总股本/流通股本: 通达信 gpcw 财务数据文件 (cw/gpcw*.zip)
  - 收盘价(不复权): 通达信日线数据 (sh/lday/*.day, sz/lday/*.day)
  - 收盘价(前复权, 后备): Qlib 现有的 close.day.bin

计算方法（与 Tushare 一致）:
  total_mv(万元) = 总股本(股) × 不复权收盘价(元) / 10000
  circ_mv(万元)  = 流通股本(股) × 不复权收盘价(元) / 10000

依赖: pip install pytdx pandas numpy tqdm

用法: python fill_market_value.py
"""

import os
import struct
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pytdx.reader import HistoryFinancialReader

# ======================== 配置 ========================
TDX_BASE = r"D:\chenxu\TDX_MONI\vipdoc"
TDX_CW_DIR = os.path.join(TDX_BASE, "cw")
TDX_SH_LDAY = os.path.join(TDX_BASE, "sh", "lday")
TDX_SZ_LDAY = os.path.join(TDX_BASE, "sz", "lday")
QLIB_FEATURES_DIR = r"E:\Quant\Qlibworks\qlib_data\features"
CALENDAR_FILE = r"E:\Quant\Qlibworks\qlib_data\calendars\day.txt"
# =====================================================


def load_calendar(path: str) -> tuple:
    """加载交易日历"""
    with open(path, "r") as f:
        cal = [d.strip() for d in f.read().splitlines() if d.strip()]
    cal_map = {d: i for i, d in enumerate(cal)}
    return cal, cal_map


def parse_gpcw_files(cw_dir: str) -> pd.DataFrame:
    """
    解析所有 gpcw zip 文件，提取总股本(col238)和流通股本(col239)
    返回 DataFrame: code, report_date, total_share, float_share
    """
    zips = sorted([
        f for f in os.listdir(cw_dir)
        if f.startswith("gpcw") and f.endswith(".zip")
    ])
    print(f"找到 {len(zips)} 个 gpcw 财务数据文件 ({zips[0]} ~ {zips[-1]})")

    reader = HistoryFinancialReader()
    all_records = []

    for zf in tqdm(zips, desc="解析 gpcw 文件"):
        zip_path = os.path.join(cw_dir, zf)
        report_date = zf.replace("gpcw", "").replace(".zip", "")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(tmp)
                    extracted = os.listdir(tmp)
                    if not extracted:
                        continue
                    dat_path = os.path.join(tmp, extracted[0])

                    df = reader.get_df(dat_path)
                    if df is None:
                        continue

                    for code, row in df.iterrows():
                        total_share = float(row["col238"])
                        float_share = float(row["col239"])
                        if total_share > 0 and float_share > 0:
                            all_records.append({
                                "code": int(code),
                                "report_date": report_date,
                                "total_share": total_share,  # 单位: 股
                                "float_share": float_share,  # 单位: 股
                            })
        except Exception:
            continue

    result = pd.DataFrame(all_records)
    if not result.empty:
        result = result.drop_duplicates(subset=["code", "report_date"])
        result = result.sort_values(["code", "report_date"]).reset_index(drop=True)
    return result


def code_to_qlib(code: int) -> str:
    """TDX 6位代码转 Qlib 代码（小写后缀）"""
    s = f"{code:06d}"
    if s.startswith("6"):
        return f"{s}.sh"
    return f"{s}.sz"


def qlib_to_tdx_path(stock: str) -> str:
    """Qlib 代码转 TDX 日线文件路径"""
    code, ext = stock.split(".")
    if ext == "sh":
        return os.path.join(TDX_SH_LDAY, f"sh{code}.day")
    else:
        return os.path.join(TDX_SZ_LDAY, f"sz{code}.day")


def read_tdx_daily(filepath: str) -> dict:
    """
    读取 TDX 日线文件，返回 {date_str: close_price} 字典
    TDX 日线格式: 每条记录32字节
      日期(int32,YYYYMMDD) + 开盘(int32) + 最高 + 最低 + 收盘 + 成交额 + 成交量 + 保留
    价格: raw_int32 / 100 = 实际价格（不复权）
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "rb") as f:
        data = f.read()

    result = {}
    n = len(data) // 32
    for i in range(n):
        rec = data[i * 32:(i + 1) * 32]
        d = struct.unpack("<I", rec[0:4])[0]
        close_raw = struct.unpack("<I", rec[16:20])[0]
        yr = d // 10000
        mo = (d % 10000) // 100
        dy = d % 100
        date_str = f"{yr:04d}-{mo:02d}-{dy:02d}"
        result[date_str] = close_raw / 100.0
    return result


def read_qlib_bin(filepath: str) -> tuple:
    """读取 Qlib .day.bin 文件，返回 (start_index, values_array)"""
    if not os.path.exists(filepath):
        return None, None
    with open(filepath, "rb") as f:
        raw = f.read()
    if len(raw) < 4:
        return None, None
    start_idx = int(struct.unpack("<f", raw[:4])[0])
    values = np.frombuffer(raw[4:], dtype="<f4")
    return start_idx, values


def write_qlib_bin(filepath: str, start_index: int, values: np.ndarray):
    """写入 Qlib .day.bin 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = np.array([start_index], dtype="<f4").tobytes()
    data = values.astype("<f4").tobytes()
    with open(filepath, "wb") as f:
        f.write(header + data)


def merge_market_value_series(
    existing_start_idx: int | None,
    existing_vals: np.ndarray | None,
    new_start_idx: int,
    new_vals: np.ndarray,
) -> tuple[int, np.ndarray, int]:
    """
    合并旧市值序列与新计算序列。

    输入:
    - existing_start_idx: 旧 bin 的起始交易日索引；不存在时为 None。
    - existing_vals: 旧 bin 数组；不存在时为 None。
    - new_start_idx: 新计算数组起始交易日索引。
    - new_vals: 新计算数组，NaN 表示该日无新值。

    输出:
    - merged_start_idx: 合并后序列起始交易日索引。
    - merged_vals: 合并后的完整序列。
    - overwrite_count: 本次实际用新值覆盖的记录数。

    边界:
    - 旧文件长度落后于新日历时，需要自动扩展而不是直接布尔索引。
    - 旧文件起始日早于/晚于新序列时，都要保留重叠区间外已有数据。
    """
    overwrite_count = int((~np.isnan(new_vals)).sum())
    if existing_vals is None or existing_start_idx is None:
        return new_start_idx, new_vals.astype("<f4").copy(), overwrite_count

    merged_start_idx = min(existing_start_idx, new_start_idx)
    existing_end_idx = existing_start_idx + len(existing_vals)
    new_end_idx = new_start_idx + len(new_vals)
    merged_len = max(existing_end_idx, new_end_idx) - merged_start_idx

    merged_vals = np.full(merged_len, np.nan, dtype="<f4")

    existing_offset = existing_start_idx - merged_start_idx
    merged_vals[existing_offset:existing_offset + len(existing_vals)] = existing_vals.astype("<f4")

    new_offset = new_start_idx - merged_start_idx
    target_slice = slice(new_offset, new_offset + len(new_vals))
    new_mask = ~np.isnan(new_vals)
    target_vals = merged_vals[target_slice]
    target_vals[new_mask] = new_vals[new_mask].astype("<f4")
    merged_vals[target_slice] = target_vals

    return merged_start_idx, merged_vals, overwrite_count


def build_daily_share_series(
    share_df: pd.DataFrame,
    calendar: list,
    start_idx: int,
    n_dates: int,
) -> tuple:
    """
    将季度股本数据 forward-fill 到日频（向量化实现）
    返回: (daily_total_share, daily_float_share)
    """
    daily_total = np.full(n_dates, np.nan, dtype="<f4")
    daily_float = np.full(n_dates, np.nan, dtype="<f4")
    if share_df.empty:
        return daily_total, daily_float

    report_dates = share_df["report_date"].values.astype(int)
    total_shares = share_df["total_share"].values.astype(np.float64)
    float_shares = share_df["float_share"].values.astype(np.float64)

    cal_dates = np.array([
        int(calendar[start_idx + i].replace("-", ""))
        for i in range(n_dates)
    ])

    indices = np.searchsorted(report_dates, cal_dates, side="right") - 1
    valid = indices >= 0
    daily_total[valid] = total_shares[indices[valid]]
    daily_float[valid] = float_shares[indices[valid]]
    return daily_total, daily_float


def build_tdx_close_array(
    tdx_dict: dict,
    calendar: list,
    start_idx: int,
    n_dates: int,
) -> np.ndarray:
    """
    构建对齐的收盘价数组：仅使用 TDX 不复权数据
    无 TDX 数据的日期保持 NaN，不会被写入
    """
    result = np.full(n_dates, np.nan, dtype="<f4")
    for i in range(n_dates):
        date = calendar[start_idx + i]
        if date in tdx_dict:
            result[i] = tdx_dict[date]
    return result


def main():
    print("=" * 60)
    print("补全 Qlib features 缺失的总市值/流通市值")
    print("（使用 TDX 不复权日线数据 + gpcw 股本数据）")
    print("=" * 60)

    # 1. 加载日历
    print("\n[1] 加载交易日历...")
    calendar, cal_map = load_calendar(CALENDAR_FILE)
    print(f"    交易日历: {len(calendar)} 天 ({calendar[0]} ~ {calendar[-1]})")

    # 2. 解析 gpcw 财务数据
    print("\n[2] 解析通达信 gpcw 财务数据...")
    share_df = parse_gpcw_files(TDX_CW_DIR)
    if share_df.empty:
        print("    未解析到股本数据，退出")
        return
    print(f"    共 {len(share_df)} 条股本记录，覆盖 {share_df['code'].nunique()} 只股票")
    share_df["qlib_code"] = share_df["code"].apply(code_to_qlib)

    # 3. 获取 Qlib 股票列表
    print("\n[3] 扫描 Qlib features 目录...")
    stocks = sorted([
        d for d in os.listdir(QLIB_FEATURES_DIR)
        if os.path.isdir(os.path.join(QLIB_FEATURES_DIR, d))
    ])
    print(f"    共 {len(stocks)} 只股票")

    # 4. 遍历补全市值
    print("\n[4] 开始补全市值...")
    stats = {
        "has_data": 0, "filled": 0, "skipped_noshare": 0,
        "skipped_tdx": 0, "overwrite": 0,
    }

    for stock in tqdm(stocks, desc="补全市值"):
        stock_dir = os.path.join(QLIB_FEATURES_DIR, stock)
        close_file = os.path.join(stock_dir, "close.day.bin")
        mv_file = os.path.join(stock_dir, "total_mv.day.bin")
        circ_file = os.path.join(stock_dir, "circ_mv.day.bin")

        # 读取 Qlib close（前复权，作为后备）
        start_idx, close_qlib = read_qlib_bin(close_file)
        if close_qlib is None or len(close_qlib) == 0:
            stats["skipped_tdx"] += 1
            continue
        n_dates = len(close_qlib)

        # 读取 TDX 日线（不复权，作为主力）
        tdx_path = qlib_to_tdx_path(stock)
        tdx_dict = read_tdx_daily(tdx_path)
        if not tdx_dict:
            # TDX 也无数据，跳到下一个
            stats["skipped_tdx"] += 1
            continue

        # 构建对齐的收盘价：仅使用 TDX 不复权数据
        close_vals = build_tdx_close_array(
            tdx_dict, calendar, start_idx, n_dates
        )
        valid_close = ~np.isnan(close_vals) & (close_vals > 0)
        if not valid_close.any():
            stats["skipped_tdx"] += 1
            continue

        # 获取该股票的股本数据
        stock_shares = share_df[share_df["qlib_code"] == stock].copy()
        if stock_shares.empty:
            stats["skipped_noshare"] += 1
            continue

        # Forward-fill 季度股本到日频
        daily_total, daily_float = build_daily_share_series(
            stock_shares, calendar, start_idx, n_dates
        )

        # 计算市值（万元 = 股 × 元 / 10000）
        total_mv = np.where(
            valid_close & ~np.isnan(daily_total),
            (daily_total.astype(np.float64) * close_vals.astype(np.float64)) / 10000.0,
            np.nan,
        ).astype("<f4")

        circ_mv = np.where(
            valid_close & ~np.isnan(daily_float),
            (daily_float.astype(np.float64) * close_vals.astype(np.float64)) / 10000.0,
            np.nan,
        ).astype("<f4")

        if not np.any(~np.isnan(total_mv)):
            stats["skipped_noshare"] += 1
            continue

        def merge_with_tdx(filepath, new_vals):
            nonlocal stats
            existing_si, existing_vals = read_qlib_bin(filepath)
            merged_start_idx, merged_vals, overwrite_count = merge_market_value_series(
                existing_start_idx=existing_si,
                existing_vals=existing_vals,
                new_start_idx=start_idx,
                new_vals=new_vals,
            )
            if overwrite_count > 0:
                write_qlib_bin(filepath, merged_start_idx, merged_vals)
                stats["filled"] += overwrite_count
            if existing_vals is not None:
                stats["has_data"] += 1
            else:
                stats["overwrite"] += 1

        merge_with_tdx(mv_file, total_mv)
        merge_with_tdx(circ_file, circ_mv)


    # 5. 统计结果
    print("\n" + "=" * 60)
    print("补全完成！")
    print(f"  有股本数据: {stats['has_data']} 只")
    print(f"  补全市值记录: {stats['filled']} 条")
    print(f"  新增写入: {stats['overwrite']} 只")
    print(f"  跳过(无TDX日线): {stats['skipped_tdx']} 只")
    print(f"  跳过(无股本): {stats['skipped_noshare']} 只")
    print("=" * 60)


if __name__ == "__main__":
    main()
