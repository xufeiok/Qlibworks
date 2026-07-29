"""
补全 Qlib features close/open 数据 — 前复权版

通过 QuantDataAPI 一次查询前复权价格写入 .day.bin

用法：
  cd E:\Quant\Qlibworks
  python -u scripts/data/fill_close_open.py
"""
from __future__ import annotations
import sys, struct, time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(_project_root) / "src"))

import numpy as np
import pandas as pd
from qlworks.data.api import QuantDataAPI

QLIB_DATA_DIR = r"E:\Quant\Qlibworks\qlib_data"


def write_bin(fp, cal_map, series):
    valid = [(cal_map[d], float(v)) for d, v in series.items() if d in cal_map and pd.notna(v)]
    if not valid:
        return False
    valid.sort(key=lambda x: x[0])
    si = valid[0][0]
    arr = np.full(valid[-1][0] - si + 1, np.nan, dtype=np.float32)
    for idx, val in valid:
        arr[idx - si] = val
    with open(fp, "wb") as f:
        f.write(np.array([si], dtype="<f4").tobytes() + arr.astype("<f4").tobytes())
    return True


def main():
    t0 = time.time()
    features_dir = Path(QLIB_DATA_DIR) / "features"
    existing_dirs = sorted([d.name for d in features_dir.iterdir()
                            if d.is_dir() and (d / "close.day.bin").exists()])
    print(f"[1] 股票目录: {len(existing_dirs)} 只")

    cal_path = Path(QLIB_DATA_DIR) / "calendars" / "day.txt"
    with open(cal_path) as f:
        cal_list = [l.strip() for l in f if l.strip()]
    cal_map = {d: i for i, d in enumerate(cal_list)}
    print(f"[2] 日历: {cal_list[0]} ~ {cal_list[-1]} ({len(cal_list)} 天)")

    # 用 ClickHouse 的 LIMIT 1 BY 一次性查前复权价格
    # 前复权: adj_price = raw * adj_today / adj_latest
    print(f"[3] 查询前复权 close/open ...", flush=True)
    sql = """
        SELECT dp.ts_code AS ts_code, dp.trade_date AS trade_date,
            dp.close * COALESCE(NULLIF(daf.adj_factor,0),1) / l.af AS close,
            dp.open  * COALESCE(NULLIF(daf.adj_factor,0),1) / l.af AS open
        FROM daily_prices dp
        LEFT JOIN daily_adj_factors daf
            ON dp.ts_code = daf.ts_code AND dp.trade_date = daf.trade_date
        LEFT JOIN (
            SELECT ts_code, adj_factor AS af
            FROM daily_adj_factors
            ORDER BY ts_code, trade_date DESC
            LIMIT 1 BY ts_code
        ) l ON dp.ts_code = l.ts_code
        WHERE dp.trade_date >= '2010-01-04' AND dp.trade_date <= '2025-12-31'
        ORDER BY dp.ts_code, dp.trade_date
    """
    with QuantDataAPI() as api:
        df = api.query(sql)

    print(f"    完成: {len(df):,} 行, {df['ts_code'].nunique()} 只股票, 耗时 {time.time()-t0:.0f}s", flush=True)

    # 写入 .day.bin（数据已按 ts_code 排序，顺序遍历即可）
    print(f"[4] 写入 .day.bin ...", flush=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

    dir_to_code = {d: d.upper() for d in existing_dirs}
    code_to_dir = {v: k for k, v in dir_to_code.items()}
    stock_set = set(dir_to_code.values())

    total_ok = 0
    skipped = 0
    i = 0
    n = len(df)

    while i < n:
        tc = df.iloc[i]["ts_code"]
        # 找到本股票连续块
        j = i + 1
        while j < n and df.iloc[j]["ts_code"] == tc:
            j += 1

        if tc in stock_set:
            stock_dir = code_to_dir[tc]
            sub = df.iloc[i:j].drop_duplicates(subset=["trade_date"], keep="last").set_index("trade_date")
            c_ok = write_bin(features_dir / stock_dir / "close.day.bin", cal_map, sub["close"])
            o_ok = write_bin(features_dir / stock_dir / "open.day.bin", cal_map, sub["open"])
            if c_ok and o_ok:
                total_ok += 1
            else:
                skipped += 1
        else:
            skipped += 1

        i = j

    elapsed = time.time() - t0
    print(f"    {total_ok} 只写入, {skipped} 只跳过, 耗时 {elapsed:.0f}s", flush=True)

    # 抽样验证
    print(f"\n[5] 抽样验证:")
    sample = sorted(set(existing_dirs[:5] + existing_dirs[::len(existing_dirs)//5][:5]))
    print(f"{'股票':<16} {'字段':<8} {'起始':>12} {'数据点':>8} {'有效值':>8} {'最小值':>14} {'最大值':>14}")
    print("-" * 76)
    for code in sample:
        for field in ["close", "open"]:
            fp = features_dir / code / f"{field}.day.bin"
            if not fp.exists(): continue
            with open(fp, "rb") as f:
                raw = f.read()
            if len(raw) < 4: continue
            si = int(struct.unpack("<f", raw[:4])[0])
            data = np.frombuffer(raw, dtype="<f4")
            valid = data[1:][~np.isnan(data[1:])]
            sd = cal_list[si] if si < len(cal_list) else "?"
            print(f"{code:<16} {field:<8} {sd:>12} {len(data)-1:>8} {len(valid):>8} "
                  f"{str(float(valid.min()))[:14] if len(valid)>0 else 'N/A':>14} "
                  f"{str(float(valid.max()))[:14] if len(valid)>0 else 'N/A':>14}")

    target = "000638.sz"
    if (features_dir / target).exists():
        for field in ["close", "open"]:
            fp = features_dir / target / f"{field}.day.bin"
            if fp.exists():
                with open(fp, "rb") as f:
                    raw = f.read()
                si = int(struct.unpack("<f", raw[:4])[0])
                data = np.frombuffer(raw, dtype="<f4")
                valid = data[1:][~np.isnan(data[1:])]
                sd = cal_list[si] if si < len(cal_list) else "?"
                ed = cal_list[si + len(data) - 2] if si + len(data) - 2 < len(cal_list) else "?"
                print(f"\n{target} {field}: {len(valid)} 有效值, {sd} ~ {ed}")

    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
