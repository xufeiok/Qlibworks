# -*- coding: utf-8 -*-
"""P2 tushare 因子（优化版）：限售解禁压力 + 大宗交易折溢价。"""
import sys, time, json
from pathlib import Path
import pandas as pd
import numpy as np
import tushare as ts
import clickhouse_connect

# 凭证统一从环境变量 / 项目根 .env 读取（qlworks.config 负责加载 .env），禁止硬编码
from qlworks.config import TUSHARE_TOKEN

PROJECT = Path(r"E:\Quant\Qlibworks")
WAREHOUSE = PROJECT / "factor_data" / "warehouse"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

def save_factor(name, s, meaning, category, expression, usage_scenario, strategy_hint):
    if s is None or s.empty:
        print(f"  [SKIP] {name}: 空")
        return
    df = s.rename("value").reset_index()
    df.columns = ["instrument", "datetime", "value"]
    df = df.dropna(subset=["value"]).sort_values(["instrument", "datetime"]).set_index(["instrument", "datetime"])
    fdir = WAREHOUSE / name
    fdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fdir / "data.parquet", compression="zstd")
    dates = df.index.get_level_values("datetime")
    stocks = df.index.get_level_values("instrument").unique()
    meta = {
        "factor_name": name, "data_version": "1.0",
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_range": {"first_date": str(dates.min())[:10], "last_date": str(dates.max())[:10],
                        "total_records": len(df), "unique_stocks": len(stocks)},
        "lifecycle_stage": "active", "meaning": meaning, "category": category,
        "expression": {"tushare": expression},
        "usage_scenario": usage_scenario, "strategy_hint": strategy_hint,
    }
    (fdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK] {name}: {len(df)}行, {len(stocks)}股, {str(dates.min())[:10]}~{str(dates.max())[:10]}")

def main():
    t0 = time.time()

    # ========== 1. unlock_pressure_60d ==========
    print("\n=== 1. unlock_pressure_60d（优化向量化）===")
    all_float = []
    for year in range(2015, 2027):
        try:
            df = pro.share_float(start_date=f"{year}0101", end_date=f"{year}1231")
            if df is not None and len(df) > 0:
                all_float.append(df)
            time.sleep(0.3)
        except:
            pass
    float_df = pd.concat(all_float, ignore_index=True)
    float_df["float_date"] = pd.to_datetime(float_df["float_date"])
    float_df["float_ratio"] = pd.to_numeric(float_df["float_ratio"], errors="coerce")
    float_df = float_df.dropna(subset=["float_ratio"])
    print(f"  解禁事件总数: {len(float_df)}")

    # 向量化：对每只股票，解禁事件按日期排序
    # 对每个交易日，未来60天解禁 = sum(float_ratio where float_date in (d, d+60d])
    # 用 numpy searchsorted
    ch = clickhouse_connect.get_client(host="192.168.10.102", port=18123, username="reader", password="123", database="quant_db")
    trade_dates = pd.to_datetime([r[0] for r in ch.query("SELECT DISTINCT trade_date FROM daily_prices ORDER BY trade_date").result_rows])
    print(f"  交易日数: {len(trade_dates)}")

    rows = []
    for ts_code, grp in float_df.groupby("ts_code"):
        grp = grp.sort_values("float_date")
        dates_arr = grp["float_date"].values
        ratios_arr = grp["float_ratio"].values
        cum = np.cumsum(ratios_arr)
        # 对每个交易日 d，找 float_date > d 的第一个位置 i，float_date <= d+60d 的最后位置 j
        # 解禁压力 = cum[j] - cum[i-1]
        dts = trade_dates.values
        i = np.searchsorted(dates_arr, dts, side="right")  # float_date > d 的位置
        j_dates = dts + np.timedelta64(60, "D")
        j = np.searchsorted(dates_arr, j_dates, side="right")  # float_date <= d+60d 的位置
        # 未来60天解禁 = cum[j-1] - cum[i-1]
        pressure = np.zeros(len(dts))
        for k in range(len(dts)):
            if i[k] < j[k]:
                pressure[k] = cum[j[k]-1] - (cum[i[k]-1] if i[k] > 0 else 0)
        for k in range(len(dts)):
            if pressure[k] > 0:
                rows.append((ts_code, trade_dates[k], pressure[k]))

    unlock = pd.DataFrame(rows, columns=["ts_code","trade_date","unlock_pressure_60d"])
    s = unlock.set_index(["ts_code","trade_date"])["unlock_pressure_60d"]
    s.index.names = ["instrument","datetime"]
    save_factor("unlock_pressure_60d", s,
                "未来60日限售解禁比例合计。解禁=潜在抛压，高解禁压力股票需回避。",
                "风险", "sum(float_ratio) 未来60日",
                "解禁前抛压预判，大额解禁前1-2月股价承压。",
                "反向因子（解禁压力大=回避），日频监控。")

    # ========== 2. block_trade_premium_20d ==========
    print("\n=== 2. block_trade_premium_20d ===")
    all_bt = []
    for year in range(2015, 2027):
        try:
            df = pro.block_trade(start_date=f"{year}0101", end_date=f"{year}1231")
            if df is not None and len(df) > 0:
                all_bt.append(df)
            time.sleep(0.3)
        except:
            pass
    bt_df = pd.concat(all_bt, ignore_index=True)
    bt_df["trade_date"] = pd.to_datetime(bt_df["trade_date"])
    bt_df["price"] = pd.to_numeric(bt_df["price"], errors="coerce")
    print(f"  大宗交易笔数: {len(bt_df)}")

    close = ch.query("SELECT ts_code, trade_date, close FROM daily_prices ORDER BY ts_code, trade_date").result_rows
    cl_df = pd.DataFrame(close, columns=["ts_code","trade_date","close"])
    cl_df["trade_date"] = pd.to_datetime(cl_df["trade_date"])
    cl_df["close"] = pd.to_numeric(cl_df["close"], errors="coerce")
    bt_df = bt_df.merge(cl_df, on=["ts_code","trade_date"], how="left")
    bt_df["premium"] = (bt_df["price"] - bt_df["close"]) / bt_df["close"].replace(0, np.nan)
    bt_agg = bt_df.groupby(["ts_code","trade_date"])["premium"].mean().reset_index()
    bt_agg = bt_agg.sort_values(["ts_code","trade_date"])
    bt_agg["premium_20d"] = bt_agg.groupby("ts_code")["premium"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    s = bt_agg.dropna(subset=["premium_20d"]).set_index(["ts_code","trade_date"])["premium_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("block_trade_premium_20d", s,
                "大宗交易折溢价率20日均值。大宗折价=机构出货信号，溢价=机构抢筹。",
                "资金情绪", "mean((block_price-close)/close) 近20日",
                "机构大宗交易动向，折价率持续高=机构急于出货。",
                "反向因子（折价深=利空），周频跟踪。")

    elapsed = time.time() - t0
    print(f"\n完成，耗时 {elapsed:.0f}s")

if __name__ == "__main__":
    main()
