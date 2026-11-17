# -*- coding: utf-8 -*-
"""批量计算 P0-P2 新因子，从 ClickHouse 拉数据写入 warehouse parquet。"""
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
    """df_value: Series, index=(instrument, datetime)，name=value"""
    if df_value is None or df_value.empty:
        print(f"  [SKIP] {name}: 数据为空")
        return
    df = df_value.rename("value").reset_index()
    df.columns = ["instrument", "datetime", "value"]
    df = df.dropna(subset=["value"])
    df = df.sort_values(["instrument", "datetime"]).set_index(["instrument", "datetime"])
    # 写 parquet
    fdir = WAREHOUSE / name
    fdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fdir / "data.parquet", compression="zstd")
    # meta.json
    dates = df.index.get_level_values("datetime")
    stocks = df.index.get_level_values("instrument").unique()
    meta = {
        "factor_name": name,
        "data_version": "1.0",
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

def to_warehouse_index(df, date_col="trade_date", code_col="ts_code"):
    """ClickHouse DataFrame → Series, index=(instrument, datetime)，取第一列作为值"""
    val_col = [c for c in df.columns if c not in (code_col, date_col)][0]
    s = df.set_index([code_col, date_col])[val_col]
    s.index.names = ["instrument", "datetime"]
    return s

def main():
    ch = get_ch()
    t0 = time.time()
    results = []

    # ========== 1. money_flow 资金流因子 ==========
    print("\n=== money_flow 资金流因子 ===")
    mf = ch.query("""
        SELECT ts_code, trade_date,
               buy_elg_amount, sell_elg_amount,
               buy_lg_amount, sell_lg_amount,
               buy_md_amount, sell_md_amount
        FROM money_flow
        ORDER BY ts_code, trade_date
    """).result_rows
    mf_cols = ["ts_code","trade_date","buy_elg","sell_elg","buy_lg","sell_lg","buy_md","sell_md"]
    mf_df = pd.DataFrame(mf, columns=mf_cols)
    mf_df["trade_date"] = pd.to_datetime(mf_df["trade_date"])
    for c in ["buy_elg","sell_elg","buy_lg","sell_lg","buy_md","sell_md"]:
        mf_df[c] = pd.to_numeric(mf_df[c], errors="coerce")

    # sent_elg_net_ratio
    elg = (mf_df["buy_elg"] - mf_df["sell_elg"]) / (mf_df["buy_elg"] + mf_df["sell_elg"]).replace(0, np.nan)
    save_factor("sent_elg_net_ratio",
                to_warehouse_index(mf_df.assign(v=elg)[["ts_code","trade_date","v"]].rename(columns={"v":"value"})),
                "超大单净流入占比。主力资金（机构/大户）净买入占超大单成交比。", "资金情绪",
                "(buy_elg-sell_elg)/(buy_elg+sell_elg)")
    results.append("sent_elg_net_ratio")

    # sent_lg_net_ratio
    lg = (mf_df["buy_lg"] - mf_df["sell_lg"]) / (mf_df["buy_lg"] + mf_df["sell_lg"]).replace(0, np.nan)
    save_factor("sent_lg_net_ratio",
                to_warehouse_index(mf_df.assign(v=lg)[["ts_code","trade_date","v"]].rename(columns={"v":"value"})),
                "大单净流入占比。大户资金净买入占大单成交比。", "资金情绪",
                "(buy_lg-sell_lg)/(buy_lg+sell_lg)")
    results.append("sent_lg_net_ratio")

    # sent_md_net_ratio
    md = (mf_df["buy_md"] - mf_df["sell_md"]) / (mf_df["buy_md"] + mf_df["sell_md"]).replace(0, np.nan)
    save_factor("sent_md_net_ratio",
                to_warehouse_index(mf_df.assign(v=md)[["ts_code","trade_date","v"]].rename(columns={"v":"value"})),
                "中单净流入占比。散户与中户资金净买入占比。", "资金情绪",
                "(buy_md-sell_md)/(buy_md+sell_md)")
    results.append("sent_md_net_ratio")

    # sent_mf_strength_20d: (buy_lg+buy_elg - sell_lg - sell_elg) / amount, MA20
    mf_df["net_big"] = mf_df["buy_lg"] + mf_df["buy_elg"] - mf_df["sell_lg"] - mf_df["sell_elg"]
    # 需要 amount，从 daily_prices 拉
    dp = ch.query("""
        SELECT ts_code, trade_date, amount FROM daily_prices ORDER BY ts_code, trade_date
    """).result_rows
    dp_df = pd.DataFrame(dp, columns=["ts_code","trade_date","amount"])
    dp_df["trade_date"] = pd.to_datetime(dp_df["trade_date"])
    dp_df["amount"] = pd.to_numeric(dp_df["amount"], errors="coerce")
    mf_df = mf_df.merge(dp_df, on=["ts_code","trade_date"], how="left")
    mf_df["mf_strength"] = mf_df["net_big"] / mf_df["amount"].replace(0, np.nan)
    mf_df["mf_strength_20d"] = mf_df.groupby("ts_code")["mf_strength"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    save_factor("sent_mf_strength_20d",
                to_warehouse_index(mf_df[["ts_code","trade_date","mf_strength_20d"]].rename(columns={"mf_strength_20d":"value"})),
                "大单+超大单净流入占成交额比，20日均值。综合主力资金强度。", "资金情绪",
                "MA20((buy_lg+buy_elg-sell_lg-sell_elg)/amount)")
    results.append("sent_mf_strength_20d")

    # ========== 2. daily_indicators + financial_indicators 因子 ==========
    print("\n=== daily_indicators + financial_indicators 因子 ===")
    di = ch.query("""
        SELECT ts_code, trade_date, volume_ratio, pe_ttm
        FROM daily_indicators ORDER BY ts_code, trade_date
    """).result_rows
    di_df = pd.DataFrame(di, columns=["ts_code","trade_date","volume_ratio","pe_ttm"])
    di_df["trade_date"] = pd.to_datetime(di_df["trade_date"])
    for c in ["volume_ratio","pe_ttm"]:
        di_df[c] = pd.to_numeric(di_df[c], errors="coerce")

    # 从 financial_indicators 拉 grossprofit_margin
    fin = ch.query("""
        SELECT ts_code, ann_date, grossprofit_margin
        FROM financial_indicators
        WHERE grossprofit_margin IS NOT NULL
        ORDER BY ts_code, ann_date
    """).result_rows
    fin_df = pd.DataFrame(fin, columns=["ts_code","ann_date","grossprofit_margin"])
    fin_df["ann_date"] = pd.to_datetime(fin_df["ann_date"])
    fin_df["grossprofit_margin"] = pd.to_numeric(fin_df["grossprofit_margin"], errors="coerce")
    fin_df = fin_df.dropna(subset=["grossprofit_margin"]).drop_duplicates(subset=["ts_code","ann_date"], keep="last")

    # volume_ratio_5d: MA5(volume_ratio)
    di_df["vr_5d"] = di_df.groupby("ts_code")["volume_ratio"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    save_factor("volume_ratio_5d",
                to_warehouse_index(di_df[["ts_code","trade_date","vr_5d"]].rename(columns={"vr_5d":"value"})),
                "量比5日均值。当日成交量/过去5日均量的5日平滑。", "交易情绪",
                "MA5(volume_ratio)")
    results.append("volume_ratio_5d")

    # gross_margin_chg_4q: grossprofit_margin - 4季度前（按报告期对齐到日频）
    fin_df = fin_df.sort_values(["ts_code","ann_date"])
    fin_df["gpm_4q_ago"] = fin_df.groupby("ts_code")["grossprofit_margin"].shift(4)
    fin_df["gpm_chg"] = fin_df["grossprofit_margin"] - fin_df["gpm_4q_ago"]
    save_factor("gross_margin_chg_4q",
                to_warehouse_index(fin_df.dropna(subset=["gpm_chg"]).rename(columns={"ann_date":"trade_date"})[["ts_code","trade_date","gpm_chg"]].rename(columns={"gpm_chg":"value"})),
                "销售毛利率4季度变化。毛利率持续提升=产品竞争力增强。", "质量",
                "grossprofit_margin - Ref4Q(grossprofit_margin)")
    results.append("gross_margin_chg_4q")

    # ep_ind_neutral: (1/pe_ttm) - 行业均值（行业映射需要 sw_industry_members，简化用全市场均值代替）
    di_df["ep"] = 1.0 / di_df["pe_ttm"].replace(0, np.nan)
    di_df["ep_ind_neutral"] = di_df["ep"] - di_df.groupby("trade_date")["ep"].transform("mean")
    save_factor("ep_ind_neutral",
                to_warehouse_index(di_df[["ts_code","trade_date","ep_ind_neutral"]].rename(columns={"ep_ind_neutral":"value"})),
                "EP行业中性版。个股盈利收益率减去市场均值，剔除整体估值波动。", "价值",
                "(1/pe_ttm) - cross_section_mean(1/pe_ttm)")
    results.append("ep_ind_neutral")

    # ========== 3. stk_holdernumber 股东户数 ==========
    print("\n=== stk_holdernumber 股东户数 ===")
    sh = ch.query("""
        SELECT ts_code, ann_date, end_date, holder_num
        FROM stk_holdernumber ORDER BY ts_code, ann_date
    """).result_rows
    sh_df = pd.DataFrame(sh, columns=["ts_code","ann_date","end_date","holder_num"])
    sh_df["ann_date"] = pd.to_datetime(sh_df["ann_date"])
    sh_df["holder_num"] = pd.to_numeric(sh_df["holder_num"], errors="coerce")
    sh_df = sh_df.drop_duplicates(subset=["ts_code","ann_date"], keep="last")

    # holder_num_chg_20d: holder_num / 4期前 - 1
    sh_df = sh_df.sort_values(["ts_code","ann_date"])
    sh_df["hn_4q_ago"] = sh_df.groupby("ts_code")["holder_num"].shift(4)
    sh_df["hn_chg"] = sh_df["holder_num"] / sh_df["hn_4q_ago"].replace(0, np.nan) - 1
    save_factor("holder_num_chg_20d",
                to_warehouse_index(sh_df.dropna(subset=["hn_chg"])[["ts_code","ann_date","hn_chg"]].rename(columns={"ann_date":"trade_date","hn_chg":"value"})),
                "股东户数环比变化率（4期）。户数减少=筹码集中=机构建仓。", "筹码/股东",
                "holder_num / Ref4Q(holder_num) - 1")
    results.append("holder_num_chg_20d")

    # holder_num_trend_4q: 近4期线性斜率
    def slope_4q(x):
        if len(x) < 4:
            return np.nan
        y = x.tail(4).values
        return np.polyfit(range(4), y, 1)[0]
    sh_df["hn_slope"] = sh_df.groupby("ts_code")["holder_num"].transform(slope_4q)
    save_factor("holder_num_trend_4q",
                to_warehouse_index(sh_df.dropna(subset=["hn_slope"])[["ts_code","ann_date","hn_slope"]].rename(columns={"ann_date":"trade_date","hn_slope":"value"})),
                "近4期股东户数线性斜率。持续下降=筹码持续集中。", "筹码/股东",
                "LinReg slope(holder_num, 4Q)")
    results.append("holder_num_trend_4q")

    # ========== 4. 同花顺特色因子 ==========
    print("\n=== 同花顺龙虎榜/涨停/热搜 ===")
    # 龙虎榜
    lhb = ch.query("""
        SELECT trade_date, ticker, org_net_value, buy_value
        FROM ths_dragon_tiger_stock_daily ORDER BY trade_date
    """).result_rows
    lhb_df = pd.DataFrame(lhb, columns=["trade_date","ticker","org_net_value","buy_value"])
    lhb_df["trade_date"] = pd.to_datetime(lhb_df["trade_date"])
    lhb_df["ts_code"] = lhb_df["ticker"].apply(lambda t: f"{t}.SZ" if t.startswith("0") or t.startswith("3") else f"{t}.SH")
    for c in ["org_net_value","buy_value"]:
        lhb_df[c] = pd.to_numeric(lhb_df[c], errors="coerce")

    # lhb_count_20d: 近20日上榜次数
    all_dates = pd.date_range(lhb_df["trade_date"].min(), lhb_df["trade_date"].max(), freq="B")
    # 每只股票每日是否上榜 → 滚动20日求和
    lhb_pivot = lhb_df.groupby(["ts_code","trade_date"]).size().unstack(fill_value=0)
    lhb_count = lhb_pivot.T.rolling(20, min_periods=1).sum().T
    lhb_count_long = lhb_count.stack().reset_index()
    lhb_count_long.columns = ["ts_code","trade_date","value"]
    save_factor("lhb_count_20d",
                to_warehouse_index(lhb_count_long),
                "近20日龙虎榜上榜次数。游资关注度。", "龙虎榜/涨停/热搜",
                "ths_dragon_tiger_stock_daily 近20日上榜次数")
    results.append("lhb_count_20d")

    # lhb_org_net_rate: org_net_value/buy_value 近20日均值
    lhb_df["org_rate"] = lhb_df["org_net_value"] / lhb_df["buy_value"].replace(0, np.nan)
    lhb_org = lhb_df.groupby(["ts_code","trade_date"])["org_rate"].mean().reset_index()
    lhb_org = lhb_org.sort_values(["ts_code","trade_date"])
    lhb_org["org_rate_20d"] = lhb_org.groupby("ts_code")["org_rate"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    save_factor("lhb_org_net_rate",
                to_warehouse_index(lhb_org[["ts_code","trade_date","org_rate_20d"]].rename(columns={"org_rate_20d":"value"})),
                "龙虎榜机构席位净买入率。机构通过龙虎榜的买卖力度。", "龙虎榜/涨停/热搜",
                "org_net_value/buy_value 近20日均值")
    results.append("lhb_org_net_rate")

    # 涨停池
    lu = ch.query("""
        SELECT trade_date, ticker, continue_day_cnt, seal_money
        FROM ths_limit_up_pool_daily ORDER BY trade_date
    """).result_rows
    lu_df = pd.DataFrame(lu, columns=["trade_date","ticker","continue_day_cnt","seal_money"])
    lu_df["trade_date"] = pd.to_datetime(lu_df["trade_date"])
    lu_df["ts_code"] = lu_df["ticker"].apply(lambda t: f"{t}.SZ" if t.startswith("0") or t.startswith("3") else f"{t}.SH")
    for c in ["continue_day_cnt","seal_money"]:
        lu_df[c] = pd.to_numeric(lu_df[c], errors="coerce")

    # limit_up_height_20d: 近20日连板高度最大值
    lu_height = lu_df.groupby(["ts_code","trade_date"])["continue_day_cnt"].max().reset_index()
    lu_height = lu_height.sort_values(["ts_code","trade_date"])
    lu_height["height_20d"] = lu_height.groupby("ts_code")["continue_day_cnt"].transform(lambda x: x.rolling(20, min_periods=1).max())
    save_factor("limit_up_height_20d",
                to_warehouse_index(lu_height[["ts_code","trade_date","height_20d"]].rename(columns={"height_20d":"value"})),
                "近20日涨停连板高度最大值。打板情绪周期位置。", "龙虎榜/涨停/热搜",
                "continue_day_cnt 近20日最大值")
    results.append("limit_up_height_20d")

    # limit_seal_strength: seal_money/circ_mv 近20日均值
    # 需要 circ_mv，从 daily_indicators 拉
    circ = ch.query("SELECT ts_code, trade_date, circ_mv FROM daily_indicators ORDER BY ts_code, trade_date").result_rows
    circ_df = pd.DataFrame(circ, columns=["ts_code","trade_date","circ_mv"])
    circ_df["trade_date"] = pd.to_datetime(circ_df["trade_date"])
    circ_df["circ_mv"] = pd.to_numeric(circ_df["circ_mv"], errors="coerce")
    lu_df = lu_df.merge(circ_df, on=["ts_code","trade_date"], how="left")
    lu_df["seal_ratio"] = lu_df["seal_money"] / lu_df["circ_mv"].replace(0, np.nan)
    lu_seal = lu_df.groupby(["ts_code","trade_date"])["seal_ratio"].mean().reset_index()
    lu_seal = lu_seal.sort_values(["ts_code","trade_date"])
    lu_seal["seal_strength"] = lu_seal.groupby("ts_code")["seal_ratio"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    save_factor("limit_seal_strength",
                to_warehouse_index(lu_seal[["ts_code","trade_date","seal_strength"]].rename(columns={"seal_strength":"value"})),
                "涨停封板资金强度（封单金额/流通市值）。封板力度=做多决心。", "龙虎榜/涨停/热搜",
                "seal_money/circ_mv 近20日均值")
    results.append("limit_seal_strength")

    # 热门股排名
    hr = ch.query("""
        SELECT date, ticker, rank FROM ths_hot_stock_rank_daily ORDER BY date
    """).result_rows
    hr_df = pd.DataFrame(hr, columns=["trade_date","ticker","rank"])
    hr_df["trade_date"] = pd.to_datetime(hr_df["trade_date"])
    hr_df["ts_code"] = hr_df["ticker"].apply(lambda t: f"{t}.SZ" if t.startswith("0") or t.startswith("3") else f"{t}.SH")
    hr_df["rank"] = pd.to_numeric(hr_df["rank"], errors="coerce")
    hr_df = hr_df.sort_values(["ts_code","trade_date"])
    # hot_rank_chg_20d: rank 20日前的变化（排名上升=关注度提升，rank变小=上升）
    hr_df["rank_20d_ago"] = hr_df.groupby("ts_code")["rank"].transform(lambda x: x.shift(20))
    hr_df["rank_chg"] = hr_df["rank_20d_ago"] - hr_df["rank"]  # 正=排名上升
    save_factor("hot_rank_chg_20d",
                to_warehouse_index(hr_df.dropna(subset=["rank_chg"])[["ts_code","trade_date","rank_chg"]].rename(columns={"rank_chg":"value"})),
                "热门股搜索排名20日变化。排名上升=关注度提升。", "龙虎榜/涨停/热搜",
                "rank 近20日变化（排名上升=关注度提升）")
    results.append("hot_rank_chg_20d")

    # ========== 5. 行业相对动量 ==========
    print("\n=== ind_rel_mom_20d ===")
    # 用 daily_prices 的 close 计算 20日动量，再减去全市场均值（简化行业中性为截面中性）
    close = ch.query("SELECT ts_code, trade_date, close FROM daily_prices ORDER BY ts_code, trade_date").result_rows
    cl_df = pd.DataFrame(close, columns=["ts_code","trade_date","close"])
    cl_df["trade_date"] = pd.to_datetime(cl_df["trade_date"])
    cl_df["close"] = pd.to_numeric(cl_df["close"], errors="coerce")
    cl_df = cl_df.sort_values(["ts_code","trade_date"])
    cl_df["mom_20d"] = cl_df.groupby("ts_code")["close"].transform(lambda x: x / x.shift(20) - 1)
    cl_df["mom_mkt_mean"] = cl_df.groupby("trade_date")["mom_20d"].transform("mean")
    cl_df["ind_rel_mom"] = cl_df["mom_20d"] - cl_df["mom_mkt_mean"]
    save_factor("ind_rel_mom_20d",
                to_warehouse_index(cl_df.dropna(subset=["ind_rel_mom"])[["ts_code","trade_date","ind_rel_mom"]].rename(columns={"ind_rel_mom":"value"})),
                "行业相对动量20日。个股20日动量减去截面均值，剔除市场beta后的个股超额动量。", "行业相对动量",
                "close/Ref20(close)-1 - cross_section_mean(close/Ref20(close)-1)")
    results.append("ind_rel_mom_20d")

    # ========== 汇总 ==========
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"批量计算完成: {len(results)} 个因子, 耗时 {elapsed:.0f}s")
    for r in results:
        print(f"  ✓ {r}")

if __name__ == "__main__":
    main()
