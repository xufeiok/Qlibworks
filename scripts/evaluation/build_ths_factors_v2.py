# -*- coding: utf-8 -*-
"""补充同花顺因子：重算涨停池2个因子(延长到2020-07) + 新增3个因子(炸板/跌停/游资)。"""
import sys, time, json
from pathlib import Path
import pandas as pd
import numpy as np
import clickhouse_connect

PROJECT = Path(r"E:\Quant\Qlibworks")
sys.path.insert(0, str(PROJECT / "src"))

CH = dict(host="192.168.10.102", port=18123, username="reader", password="123", database="quant_db")
WAREHOUSE = PROJECT / "factor_data" / "warehouse"

def get_ch():
    return clickhouse_connect.get_client(**CH)

def save_factor(name, df_value, meaning, category, expression):
    if df_value is None or df_value.empty:
        print(f"  [SKIP] {name}: 数据为空")
        return
    df = df_value.rename("value").reset_index()
    df.columns = ["instrument", "datetime", "value"]
    df = df.dropna(subset=["value"])
    df = df.sort_values(["instrument", "datetime"]).set_index(["instrument", "datetime"])
    fdir = WAREHOUSE / name
    fdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fdir / "data.parquet", compression="zstd")
    dates = df.index.get_level_values("datetime")
    stocks = df.index.get_level_values("instrument").unique()
    meta = {
        "factor_name": name,
        "data_version": "1.1",
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_range": {
            "first_date": str(dates.min())[:10],
            "last_date": str(dates.max())[:10],
            "total_records": len(df),
            "unique_stocks": len(stocks),
        },
        "lifecycle_stage": "active",
        "meaning": meaning,
        "category": category,
        "expression": {"ch": expression},
    }
    (fdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK] {name}: {len(df)} 行, {len(stocks)} 股, {str(dates.min())[:10]} ~ {str(dates.max())[:10]}")

def ticker_to_code(t):
    t = str(t).zfill(6)
    if t.startswith(("0", "3")):
        return f"{t}.SZ"
    elif t.startswith("6"):
        return f"{t}.SH"
    elif t.startswith(("8", "4")):
        return f"{t}.BJ"
    return f"{t}.SZ"

def main():
    ch = get_ch()
    t0 = time.time()
    results = []

    # ========== 1. 重算 limit_up_height_20d（2020-07起） ==========
    print("\n=== 重算 limit_up_height_20d（延长到2020-07） ===")
    lu = ch.query("""
        SELECT trade_date, ticker, continue_day_cnt, seal_money
        FROM ths_limit_up_pool_daily ORDER BY trade_date
    """).result_rows
    lu_df = pd.DataFrame(lu, columns=["trade_date","ticker","continue_day_cnt","seal_money"])
    lu_df["trade_date"] = pd.to_datetime(lu_df["trade_date"])
    lu_df["ts_code"] = lu_df["ticker"].apply(ticker_to_code)
    for c in ["continue_day_cnt","seal_money"]:
        lu_df[c] = pd.to_numeric(lu_df[c], errors="coerce")

    # limit_up_height_20d: 近20日连板高度最大值
    lu_height = lu_df.groupby(["ts_code","trade_date"])["continue_day_cnt"].max().reset_index()
    lu_height = lu_height.sort_values(["ts_code","trade_date"])
    lu_height["height_20d"] = lu_height.groupby("ts_code")["continue_day_cnt"].transform(
        lambda x: x.rolling(20, min_periods=1).max())
    s = lu_height.set_index(["ts_code","trade_date"])["height_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("limit_up_height_20d", s,
                "近20日涨停连板高度最大值。打板情绪周期位置。", "龙虎榜/涨停/热搜",
                "continue_day_cnt 近20日最大值")
    results.append("limit_up_height_20d")

    # ========== 2. 重算 limit_seal_strength（2020-07起） ==========
    print("\n=== 重算 limit_seal_strength（延长到2020-07） ===")
    # 需要 circ_mv
    circ = ch.query("SELECT ts_code, trade_date, circ_mv FROM daily_indicators ORDER BY ts_code, trade_date").result_rows
    circ_df = pd.DataFrame(circ, columns=["ts_code","trade_date","circ_mv"])
    circ_df["trade_date"] = pd.to_datetime(circ_df["trade_date"])
    circ_df["circ_mv"] = pd.to_numeric(circ_df["circ_mv"], errors="coerce")
    lu_df = lu_df.merge(circ_df, on=["ts_code","trade_date"], how="left")
    lu_df["seal_ratio"] = lu_df["seal_money"] / lu_df["circ_mv"].replace(0, np.nan)
    lu_seal = lu_df.groupby(["ts_code","trade_date"])["seal_ratio"].mean().reset_index()
    lu_seal = lu_seal.sort_values(["ts_code","trade_date"])
    lu_seal["seal_strength"] = lu_seal.groupby("ts_code")["seal_ratio"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    s = lu_seal.set_index(["ts_code","trade_date"])["seal_strength"]
    s.index.names = ["instrument","datetime"]
    save_factor("limit_seal_strength", s,
                "涨停封板资金强度（封单金额/流通市值）。封板力度=做多决心。", "龙虎榜/涨停/热搜",
                "seal_money/circ_mv 近20日均值")
    results.append("limit_seal_strength")

    # ========== 3. 新增 break_count_20d（炸板次数） ==========
    print("\n=== 新增 break_count_20d ===")
    br = ch.query("""
        SELECT trade_date, ticker, open_times
        FROM ths_limit_break_pool_daily ORDER BY trade_date
    """).result_rows
    br_df = pd.DataFrame(br, columns=["trade_date","ticker","open_times"])
    br_df["trade_date"] = pd.to_datetime(br_df["trade_date"])
    br_df["ts_code"] = br_df["ticker"].apply(ticker_to_code)
    br_df["open_times"] = pd.to_numeric(br_df["open_times"], errors="coerce")
    # 每只股票每日炸板次数（open_times = 开板次数）
    br_count = br_df.groupby(["ts_code","trade_date"])["open_times"].sum().reset_index()
    br_count = br_count.sort_values(["ts_code","trade_date"])
    br_count["break_20d"] = br_count.groupby("ts_code")["open_times"].transform(
        lambda x: x.rolling(20, min_periods=1).sum())
    s = br_count.set_index(["ts_code","trade_date"])["break_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("break_count_20d", s,
                "近20日炸板次数（开板次数累计）。炸板多=封板不坚决=上方抛压重。", "龙虎榜/涨停/热搜",
                "ths_limit_break_pool_daily.open_times 近20日累计")
    results.append("break_count_20d")

    # ========== 4. 新增 limit_down_count_20d（跌停次数） ==========
    print("\n=== 新增 limit_down_count_20d ===")
    ld = ch.query("""
        SELECT trade_date, ticker
        FROM ths_limit_down_pool_daily ORDER BY trade_date
    """).result_rows
    ld_df = pd.DataFrame(ld, columns=["trade_date","ticker"])
    ld_df["trade_date"] = pd.to_datetime(ld_df["trade_date"])
    ld_df["ts_code"] = ld_df["ticker"].apply(ticker_to_code)
    ld_count = ld_df.groupby(["ts_code","trade_date"]).size().reset_index(name="ld_cnt")
    ld_count = ld_count.sort_values(["ts_code","trade_date"])
    ld_count["ld_20d"] = ld_count.groupby("ts_code")["ld_cnt"].transform(
        lambda x: x.rolling(20, min_periods=1).sum())
    s = ld_count.set_index(["ts_code","trade_date"])["ld_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("limit_down_count_20d", s,
                "近20日跌停次数。跌停多=恐慌抛售=弱势股。", "龙虎榜/涨停/热搜",
                "ths_limit_down_pool_daily 近20日跌停次数")
    results.append("limit_down_count_20d")

    # ========== 5. 新增 hot_money_net_rate（游资净买入率） ==========
    print("\n=== 新增 hot_money_net_rate_20d ===")
    hm = ch.query("""
        SELECT trade_date, ticker, net_value, buy_value, sell_value
        FROM ths_dragon_tiger_hot_money_daily ORDER BY trade_date
    """).result_rows
    hm_df = pd.DataFrame(hm, columns=["trade_date","ticker","net_value","buy_value","sell_value"])
    hm_df["trade_date"] = pd.to_datetime(hm_df["trade_date"])
    hm_df["ts_code"] = hm_df["ticker"].apply(ticker_to_code)
    for c in ["net_value","buy_value","sell_value"]:
        hm_df[c] = pd.to_numeric(hm_df[c], errors="coerce")
    # 每只股票每日游资净买入率 = sum(net_value) / sum(buy_value+sell_value)
    hm_agg = hm_df.groupby(["ts_code","trade_date"]).agg(
        net_sum=("net_value","sum"),
        total=("buy_value","sum")).reset_index()
    hm_agg["net_rate"] = hm_agg["net_sum"] / hm_agg["total"].replace(0, np.nan)
    hm_agg = hm_agg.sort_values(["ts_code","trade_date"])
    hm_agg["net_rate_20d"] = hm_agg.groupby("ts_code")["net_rate"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    s = hm_agg.set_index(["ts_code","trade_date"])["net_rate_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("hot_money_net_rate_20d", s,
                "近20日龙虎榜游资净买入率。游资净流入强度=短线资金追捧。", "龙虎榜/涨停/热搜",
                "sum(net_value)/sum(buy_value) 近20日均值")
    results.append("hot_money_net_rate_20d")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"补充计算完成: {len(results)} 个因子, 耗时 {elapsed:.0f}s")
    for r in results:
        print(f"  ✓ {r}")

if __name__ == "__main__":
    main()
