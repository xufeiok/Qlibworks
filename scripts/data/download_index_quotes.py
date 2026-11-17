# -*- coding: utf-8 -*-
"""
下载宽基指数日线行情（中证500/沪深300）写入 qlib_data，供回测基准对比。

用法:
    E:\\Conda_env\\Qlib_env\\python.exe scripts\\data\\download_index_quotes.py

说明:
- tushare index_daily 拉取 000905.SH（中证500）、000300.SH（沪深300）
- 写入 qlib_data/features/<SH000905|SH000300>/{open,high,low,close,volume}.day.bin
- bin 格式与 qlib dump_bin 一致: 4 字节头 + float32 数组（按日历 day.txt 对齐）
- token 从环境变量 TUSHARE_TOKEN 或项目根 .env 读取（经 qlworks.config 加载）
"""
import os
import sys
import numpy as np
import pandas as pd

from qlworks.config import TUSHARE_TOKEN

QLIB_DATA_DIR = r"E:\Quant\Qlibworks\qlib_data"
CALENDAR_FILE = os.path.join(QLIB_DATA_DIR, "calendars", "day.txt")

# 指数代码（qlib instrument 名）→ tushare ts_code
INDEXES = {
    "SH000905": "000905.SH",   # 中证500
    "SH000300": "000300.SH",   # 沪深300
}
START_DATE = "20100101"
END_DATE = "20261026"


def write_day_bin(path, values, dates, calendar):
    """按 qlib dump_bin 格式写 .day.bin：4字节头 + float32，行数与日历对齐。"""
    arr = np.full(len(calendar), np.nan, dtype="<f4")
    date_set = set(dates)
    for i, d in enumerate(calendar):
        if d in date_set:
            v = values[d]
            arr[i] = v
    # 缺失日填充为 NaN（qlib 读取时自动处理）
    with open(path, "wb") as f:
        f.write(np.array([0], dtype="<f4").tobytes())  # 4 字节头
        f.write(arr.tobytes())
    print(f"    -> {path} ({len(calendar)} 行, 有效 {np.isfinite(arr).sum()})")


def main():
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    with open(CALENDAR_FILE) as f:
        calendar = [ln.strip() for ln in f if ln.strip()]

    for qlib_name, ts_code in INDEXES.items():
        print(f"[1] {qlib_name} ({ts_code}) 下载 {START_DATE}~{END_DATE} ...")
        df = pro.index_daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
        if df is None or df.empty:
            print(f"    [错误] {ts_code} 无数据，跳过")
            continue
        df = df.sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates("trade_date")
        print(f"    共 {len(df)} 个交易日: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")

        out_dir = os.path.join(QLIB_DATA_DIR, "features", qlib_name)
        os.makedirs(out_dir, exist_ok=True)

        # qlib 字段映射（volume 用 tushare vol，单位手；仅作基准对比参考）
        for qf, tf in [("open", "open"), ("high", "high"), ("low", "low"),
                       ("close", "close"), ("volume", "vol")]:
            vals = df.set_index("trade_date")[tf]
            write_day_bin(os.path.join(out_dir, f"{qf}.day.bin"), vals, df["trade_date"].tolist(), calendar)

    print("\n完成。回测基准读取示例:")
    print("  from qlib.data import D")
    print("  bm = D.features(['SH000905'], ['$open','$high','$low','$close','$volume'], start_time=..., end_time=...)")


if __name__ == "__main__":
    main()
