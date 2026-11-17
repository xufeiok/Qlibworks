# -*- coding: utf-8 -*-
"""P0 因子补充：成长类 + 应计利润 + Amihud 流动性。"""
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

    # ========== 1. 成长类因子（financial_indicators 直接有列） ==========
    print("\n=== 成长类因子 ===")
    fi = ch.query("""
        SELECT ts_code, ann_date, tr_yoy, netprofit_yoy, q_sales_yoy, dt_netprofit_yoy
        FROM financial_indicators ORDER BY ts_code, ann_date
    """).result_rows
    fi_df = pd.DataFrame(fi, columns=["ts_code","ann_date","tr_yoy","netprofit_yoy","q_sales_yoy","dt_netprofit_yoy"])
    fi_df["ann_date"] = pd.to_datetime(fi_df["ann_date"])
    for c in ["tr_yoy","netprofit_yoy","q_sales_yoy","dt_netprofit_yoy"]:
        fi_df[c] = pd.to_numeric(fi_df[c], errors="coerce")

    # revenue_growth_yoy: 营收同比增速
    s = fi_df.dropna(subset=["tr_yoy"]).set_index(["ts_code","ann_date"])["tr_yoy"]
    s.index.names = ["instrument","datetime"]
    save_factor("revenue_growth_yoy", s,
                "营业收入同比增长率（YoY）。营收持续高增长=公司扩张。", "成长",
                "financial_indicators.tr_yoy",
                "成长股筛选、业绩拐点识别。",
                "正向因子（营收增速高=成长性强），季度频。")
    results.append("revenue_growth_yoy")

    # profit_growth_yoy: 净利润同比增速
    s = fi_df.dropna(subset=["netprofit_yoy"]).set_index(["ts_code","ann_date"])["netprofit_yoy"]
    s.index.names = ["instrument","datetime"]
    save_factor("profit_growth_yoy", s,
                "净利润同比增长率（YoY）。利润持续高增长=盈利能力增强。", "成长",
                "financial_indicators.netprofit_yoy",
                "成长股筛选、业绩超预期识别。",
                "正向因子（利润增速高=盈利能力强），季度频。")
    results.append("profit_growth_yoy")

    # q_revenue_growth_yoy: 单季营收同比
    s = fi_df.dropna(subset=["q_sales_yoy"]).set_index(["ts_code","ann_date"])["q_sales_yoy"]
    s.index.names = ["instrument","datetime"]
    save_factor("q_revenue_growth_yoy", s,
                "单季营业收入同比增速。更敏感的成长拐点信号。", "成长",
                "financial_indicators.q_sales_yoy",
                "季度业绩拐点捕捉，比年度增速更灵敏。",
                "正向因子（单季加速=成长动能增强）。")
    results.append("q_revenue_growth_yoy")

    # dt_profit_growth_yoy: 扣非净利润同比
    s = fi_df.dropna(subset=["dt_netprofit_yoy"]).set_index(["ts_code","ann_date"])["dt_netprofit_yoy"]
    s.index.names = ["instrument","datetime"]
    save_factor("dt_profit_growth_yoy", s,
                "扣非净利润同比增长率。剔除非经常性损益的真实盈利增长。", "成长",
                "financial_indicators.dt_netprofit_yoy",
                "识别主业真实成长性，剔除卖资产/补贴等一次性收益。",
                "正向因子（扣非高增=主业真实成长）。")
    results.append("dt_profit_growth_yoy")

    # ========== 2. 应计利润（Sloan 1996） ==========
    print("\n=== 应计利润 Sloan ===")
    inc = ch.query("SELECT ts_code, ann_date, n_income FROM income_statement ORDER BY ts_code, ann_date").result_rows
    cf = ch.query("SELECT ts_code, ann_date, n_cashflow_act FROM cashflow_statement ORDER BY ts_code, ann_date").result_rows
    bs = ch.query("SELECT ts_code, ann_date, total_assets FROM balance_sheet ORDER BY ts_code, ann_date").result_rows
    inc_df = pd.DataFrame(inc, columns=["ts_code","ann_date","n_income"])
    cf_df = pd.DataFrame(cf, columns=["ts_code","ann_date","ocf"])
    bs_df = pd.DataFrame(bs, columns=["ts_code","ann_date","total_assets"])
    for df in [inc_df, cf_df, bs_df]:
        df["ann_date"] = pd.to_datetime(df["ann_date"])
    for c in ["n_income","ocf","total_assets"]:
        pass
    inc_df["n_income"] = pd.to_numeric(inc_df["n_income"], errors="coerce")
    cf_df["ocf"] = pd.to_numeric(cf_df["ocf"], errors="coerce")
    bs_df["total_assets"] = pd.to_numeric(bs_df["total_assets"], errors="coerce")

    acc_df = inc_df.merge(cf_df, on=["ts_code","ann_date"], how="inner").merge(bs_df, on=["ts_code","ann_date"], how="inner")
    # Sloan 应计利润 = (净利润 - 经营现金流) / 平均总资产
    acc_df["accruals"] = (acc_df["n_income"] - acc_df["ocf"]) / acc_df["total_assets"].replace(0, np.nan)
    s = acc_df.dropna(subset=["accruals"]).set_index(["ts_code","ann_date"])["accruals"]
    s.index.names = ["instrument","datetime"]
    save_factor("accruals_sloan", s,
                "Sloan应计利润=(净利润-经营现金流)/总资产。应计高=利润质量差。", "质量",
                "(n_income - n_cashflow_act)/total_assets",
                "识别利润含金量，应计高的公司未来收益低（Sloan 1996）。",
                "反向因子（应计高=利润质量差=未来跑输），季度频。")
    results.append("accruals_sloan")

    # ========== 3. Amihud 非流动性 ==========
    print("\n=== Amihud 非流动性 ===")
    dp = ch.query("SELECT ts_code, trade_date, pct_chg, amount FROM daily_prices ORDER BY ts_code, trade_date").result_rows
    dp_df = pd.DataFrame(dp, columns=["ts_code","trade_date","pct_chg","amount"])
    dp_df["trade_date"] = pd.to_datetime(dp_df["trade_date"])
    dp_df["pct_chg"] = pd.to_numeric(dp_df["pct_chg"], errors="coerce")
    dp_df["amount"] = pd.to_numeric(dp_df["amount"], errors="coerce")  # 单位：千元
    # Amihud: |收益%| / 成交额(百万)
    dp_df["illiq"] = dp_df["pct_chg"].abs() / (dp_df["amount"]/100).replace(0, np.nan)
    dp_df["illiq_20d"] = dp_df.groupby("ts_code")["illiq"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    s = dp_df.dropna(subset=["illiq_20d"]).set_index(["ts_code","trade_date"])["illiq_20d"]
    s.index.names = ["instrument","datetime"]
    save_factor("amihud_illiq_20d", s,
                "Amihud非流动性指标20日均值。|日收益|/成交额，值高=流动性差。", "流动性",
                "MA20(|pct_chg|/amount)",
                "流动性溢价选股，非流动性高的股票要求更高收益补偿。",
                "正向因子（流动性差=长期收益补偿高），日频周调仓。")
    results.append("amihud_illiq_20d")

    # ========== 4. 杠杆（顺手补） ==========
    print("\n=== 杠杆因子 ===")
    fi2 = ch.query("SELECT ts_code, ann_date, debt_to_assets FROM financial_indicators ORDER BY ts_code, ann_date").result_rows
    fi2_df = pd.DataFrame(fi2, columns=["ts_code","ann_date","debt_to_assets"])
    fi2_df["ann_date"] = pd.to_datetime(fi2_df["ann_date"])
    fi2_df["debt_to_assets"] = pd.to_numeric(fi2_df["debt_to_assets"], errors="coerce")
    s = fi2_df.dropna(subset=["debt_to_assets"]).set_index(["ts_code","ann_date"])["debt_to_assets"]
    s.index.names = ["instrument","datetime"]
    save_factor("debt_to_assets", s,
                "资产负债率。总负债/总资产，杠杆水平。", "杠杆",
                "financial_indicators.debt_to_assets",
                "财务风险排雷，高杠杆公司在衰退期脆弱。",
                "反向因子（高杠杆=风险高），季度频。")
    results.append("debt_to_assets")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"P0 补充完成: {len(results)} 个因子, 耗时 {elapsed:.0f}s")
    for r in results:
        print(f"  ✓ {r}")

if __name__ == "__main__":
    main()
