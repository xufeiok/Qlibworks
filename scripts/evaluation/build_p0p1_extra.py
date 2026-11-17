# -*- coding: utf-8 -*-
"""P0+P1 因子补充：质押/北向/融资/换手率/上市天数/行业动量。"""
import sys, time, json
from pathlib import Path
import pandas as pd
import numpy as np
import clickhouse_connect

PROJECT = Path(r"E:\Quant\Qlibworks")
sys.path.insert(0, str(PROJECT / "src"))
CH = dict(host="192.168.10.102", port=18123, username="reader", password="123", database="quant_db")
WAREHOUSE = PROJECT / "factor_data" / "warehouse"

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
        "expression": {"ch": expression},
        "usage_scenario": usage_scenario, "strategy_hint": strategy_hint,
    }
    (fdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK] {name}: {len(df)}行, {len(stocks)}股, {str(dates.min())[:10]}~{str(dates.max())[:10]}")

def main():
    ch = clickhouse_connect.get_client(**CH)
    t0 = time.time()
    results = []

    # ========== 1. pledge_ratio 股权质押比例 ==========
    print("\n=== 1. pledge_ratio ===")
    pl = ch.query("SELECT ts_code, end_date, pledge_ratio FROM pledge_stat ORDER BY ts_code, end_date").result_rows
    pl_df = pd.DataFrame(pl, columns=["ts_code","end_date","pledge_ratio"])
    pl_df["end_date"] = pd.to_datetime(pl_df["end_date"])
    pl_df["pledge_ratio"] = pd.to_numeric(pl_df["pledge_ratio"], errors="coerce")
    s = pl_df.dropna(subset=["pledge_ratio"]).set_index(["ts_code","end_date"])["pledge_ratio"]
    s.index.names = ["instrument","datetime"]
    save_factor("pledge_ratio", s,
                "股权质押比例。大股东质押股份占总股本比例，高质押=爆仓风险。",
                "风险", "pledge_stat.pledge_ratio",
                "A股特色风险排雷，高质押股在下跌时容易强平。",
                "反向因子（质押比例高=风险大=回避），季度频。")
    results.append("pledge_ratio")

    # ========== 2. north_hold_chg_20d 北向持仓变化 ==========
    print("\n=== 2. north_hold_chg_20d ===")
    di = ch.query("SELECT ts_code, trade_date, north_hold FROM daily_indicators ORDER BY ts_code, trade_date").result_rows
    di_df = pd.DataFrame(di, columns=["ts_code","trade_date","north_hold"])
    di_df["trade_date"] = pd.to_datetime(di_df["trade_date"])
    di_df["north_hold"] = pd.to_numeric(di_df["north_hold"], errors="coerce")
    di_df["north_chg_20d"] = di_df.groupby("ts_code")["north_hold"].transform(
        lambda x: x / x.shift(20) - 1)
    s = di_df.dropna(subset=["north_chg_20d"]).set_index(["ts_code","trade_date"])["north_chg_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("north_hold_chg_20d", s,
                "北向持仓比例20日变化率。北向资金被称为聪明钱，持续加仓=外资看好。",
                "资金情绪", "MA20变化: north_hold / Ref20(north_hold) - 1",
                "跟踪外资动向，北向加仓股往往中期跑赢。",
                "正向因子（北向加仓=外资看好），周频调仓。")
    results.append("north_hold_chg_20d")

    # ========== 3. rzye_chg_20d 融资余额变化 ==========
    print("\n=== 3. rzye_chg_20d ===")
    di2 = ch.query("SELECT ts_code, trade_date, rzye FROM daily_indicators ORDER BY ts_code, trade_date").result_rows
    di2_df = pd.DataFrame(di2, columns=["ts_code","trade_date","rzye"])
    di2_df["trade_date"] = pd.to_datetime(di2_df["trade_date"])
    di2_df["rzye"] = pd.to_numeric(di2_df["rzye"], errors="coerce")
    di2_df["rzye_chg_20d"] = di2_df.groupby("ts_code")["rzye"].transform(
        lambda x: x / x.shift(20) - 1)
    s = di2_df.dropna(subset=["rzye_chg_20d"]).set_index(["ts_code","trade_date"])["rzye_chg_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("rzye_chg_20d", s,
                "融资余额20日变化率。融资盘是杠杆做多力量，余额急升=散户情绪高涨。",
                "资金情绪", "MA20变化: rzye / Ref20(rzye) - 1",
                "杠杆资金情绪指标，融资盘急升后往往见顶（反向）。",
                "反向因子（融资余额急升=情绪过热=短期见顶）。")
    results.append("rzye_chg_20d")

    # ========== 4. turnover_chg_20d 换手率变化 ==========
    print("\n=== 4. turnover_chg_20d ===")
    di3 = ch.query("SELECT ts_code, trade_date, turnover_rate FROM daily_indicators ORDER BY ts_code, trade_date").result_rows
    di3_df = pd.DataFrame(di3, columns=["ts_code","trade_date","turnover_rate"])
    di3_df["trade_date"] = pd.to_datetime(di3_df["trade_date"])
    di3_df["turnover_rate"] = pd.to_numeric(di3_df["turnover_rate"], errors="coerce")
    di3_df["turn_chg_20d"] = di3_df.groupby("ts_code")["turnover_rate"].transform(
        lambda x: x / x.rolling(60, min_periods=20).mean() - 1)
    s = di3_df.dropna(subset=["turn_chg_20d"]).set_index(["ts_code","trade_date"])["turn_chg_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("turnover_chg_20d", s,
                "换手率相对60日均值偏离度。交投活跃度突变=资金关注度变化。",
                "流动性", "turnover_rate / MA60(turnover_rate) - 1",
                "识别放量启动/缩量冷却，配合价格方向判断。",
                "正向因子（放量上涨=资金关注），需结合价格方向。")
    results.append("turnover_chg_20d")

    # ========== 5. list_age_days 上市天数 ==========
    print("\n=== 5. list_age_days ===")
    sb = ch.query("SELECT ts_code, list_date FROM stock_basic WHERE list_status='L'").result_rows
    sb_df = pd.DataFrame(sb, columns=["ts_code","list_date"])
    sb_df["list_date"] = pd.to_datetime(sb_df["list_date"])
    # 每个交易日的上市天数 = 当日 - list_date
    # 用 daily_prices 的交易日序列
    dates = ch.query("SELECT DISTINCT trade_date FROM daily_prices ORDER BY trade_date").result_rows
    trade_dates = pd.to_datetime([d[0] for d in dates])
    rows = []
    for _, r in sb_df.iterrows():
        age_series = pd.Series(trade_dates, index=trade_dates)
        ages = (age_series - r["list_date"]).dt.days
        for d, a in zip(trade_dates, ages):
            if a >= 0:
                rows.append((r["ts_code"], d, a))
    la_df = pd.DataFrame(rows, columns=["ts_code","trade_date","list_age_days"])
    s = la_df.set_index(["ts_code","trade_date"])["list_age_days"]
    s.index.names = ["instrument","datetime"]
    save_factor("list_age_days", s,
                "上市天数（自然日）。次新股（<250天）有显著溢价效应。",
                "风格", "days_since_listing",
                "次新股溢价/次新股破发风险识别，上市初期波动大。",
                "非线性因子：上市<250天为次新股，需单独分组合并。")
    results.append("list_age_days")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"P0+P1 完成: {len(results)} 个因子, 耗时 {elapsed:.0f}s")
    for r in results:
        print(f"  ✓ {r}")

if __name__ == "__main__":
    main()
