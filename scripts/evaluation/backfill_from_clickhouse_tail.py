# -*- coding: utf-8 -*-
"""基于 ClickHouse 的因子仓库尾部补齐。

背景：momentum_price 家族之外仍有 28 个因子停留在 2026-10-30/31，
其表达式为 `CH: ...` 伪语法，无法走 Qlib/DuckDB 自动计算路径。

做法：按原始构建脚本（build_new_factors / build_p0_factors /
build_p0p1_extra / build_ths_factors_v2 / build_sentiment_factors）
的同一口径，只从 ClickHouse 拉取尾部窗口重算并追加，
历史数据保持不动（同索引以新值覆盖）。

输入：ClickHouse quant_db（已更新至 2026-11-03/04）
输出：factor_data/warehouse/<name>/{data.parquet, meta.json}

边界条件：
  - 尾部窗口默认 2025-06-01 起（20/60 日滚动 + 4 季 shift 均有足够预热）；
  - 季度类因子窗口 2024-03-01 起，保证 shift(4) 前值存在；
  - 表缺失（block_trade / share_float）的因子不在本脚本范围。
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import clickhouse_connect

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from qlworks.config import (CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE,
                            WAREHOUSE_DIR)

TARGET = "2026-11-03"
DAILY_START = "2025-06-01"
QUARTER_START = "2024-03-01"


def get_ch():
    """建立 ClickHouse 连接（reader 只读账号）。"""
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, user=CH_USER,
                                         password=CH_PASSWORD, database=CH_DATABASE,
                                         connect_timeout=30)


def qdf(cli, sql):
    """执行查询并返回 DataFrame。"""
    return cli.query_df(sql)


def ticker_to_code(t):
    """同花顺 ticker（6 位）→ ts_code（.SH/.SZ/.BJ）。"""
    t = str(t).zfill(6)
    if t.startswith(("0", "3")):
        return f"{t}.SZ"
    if t.startswith("6"):
        return f"{t}.SH"
    if t.startswith(("8", "4")):
        return f"{t}.BJ"
    return f"{t}.SZ"


def to_series(df, value_col):
    """DataFrame(ts_code, trade_date, value) → Series[instrument, datetime]。"""
    s = df.set_index(["ts_code", "trade_date"])[value_col].astype("float64")
    s.index.names = ["instrument", "datetime"]
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna().sort_index()


def append_factor(name, new_s):
    """把尾部新值合并进 warehouse，返回 (旧末日期, 新末日期, 新增行数)。"""
    fdir = WAREHOUSE_DIR / name
    pq = fdir / "data.parquet"
    if new_s is None or len(new_s) == 0:
        print(f"  [SKIP] {name}: 尾部无数据", flush=True)
        return None, None, 0

    new_s = new_s.rename("value").astype("float64")
    new_s.index.names = ["instrument", "datetime"]
    new_df = new_s.to_frame()
    cut = new_df.index.get_level_values("datetime").min()

    if pq.exists():
        old = pd.read_parquet(pq)
        old.index.names = ["instrument", "datetime"]
        old_last = str(old.index.get_level_values("datetime").max())[:10]
        old = old[old.index.get_level_values("datetime") < cut]
        merged = pd.concat([old, new_df])
        del old
        gc.collect()
    else:
        old_last = "-"
        merged = new_df
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged["value"] = merged["value"].astype("float32")
    merged.to_parquet(pq, compression="zstd")

    dates = merged.index.get_level_values("datetime")
    meta_path = fdir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"factor_name": name}
    meta["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["data_range"] = {
        "first_date": str(dates.min())[:10],
        "last_date": str(dates.max())[:10],
        "total_records": int(len(merged)),
        "unique_stocks": int(merged.index.get_level_values("instrument").nunique()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    new_last = meta["data_range"]["last_date"]
    added = len(new_df)
    del merged, new_df
    gc.collect()
    print(f"  [OK] {name}: {old_last} → {new_last} | 尾部 {added:,} 行", flush=True)
    return old_last, new_last, added


# ---------------------------- 日频因子 ----------------------------

def build_daily(cli):
    """日频因子：money_flow / daily_indicators / daily_prices。"""
    out = []

    mf = qdf(cli, f"""
        SELECT ts_code, trade_date, net_mf_amount,
               buy_elg_amount, sell_elg_amount, buy_lg_amount, sell_lg_amount,
               buy_md_amount, sell_md_amount
        FROM money_flow WHERE trade_date >= '{DAILY_START}'
        ORDER BY ts_code, trade_date""")
    mf["trade_date"] = pd.to_datetime(mf["trade_date"])
    for c in mf.columns[2:]:
        mf[c] = pd.to_numeric(mf[c], errors="coerce")

    for name, buy, sell in [("sent_elg_net_ratio", "buy_elg_amount", "sell_elg_amount"),
                            ("sent_lg_net_ratio", "buy_lg_amount", "sell_lg_amount"),
                            ("sent_md_net_ratio", "buy_md_amount", "sell_md_amount")]:
        v = (mf[buy] - mf[sell]) / (mf[buy] + mf[sell]).replace(0, np.nan)
        out.append((name, to_series(mf.assign(v=v)[["ts_code", "trade_date", "v"]], "v")))

    # 主力净流入率
    dp = qdf(cli, f"""SELECT ts_code, trade_date, amount, pct_chg FROM daily_prices
                      WHERE trade_date >= '{DAILY_START}' ORDER BY ts_code, trade_date""")
    dp["trade_date"] = pd.to_datetime(dp["trade_date"])
    dp["amount"] = pd.to_numeric(dp["amount"], errors="coerce")
    dp["pct_chg"] = pd.to_numeric(dp["pct_chg"], errors="coerce")

    m = mf.merge(dp[["ts_code", "trade_date", "amount"]], on=["ts_code", "trade_date"], how="inner")
    out.append(("sent_mf_net_ratio",
                to_series(m.assign(v=m["net_mf_amount"] / (m["amount"] * 10000.0)
                                   .replace(0, np.nan))[["ts_code", "trade_date", "v"]], "v")))

    m["net_big"] = m["buy_lg_amount"] + m["buy_elg_amount"] - m["sell_lg_amount"] - m["sell_elg_amount"]
    m["strength"] = m["net_big"] / m["amount"].replace(0, np.nan)
    m = m.sort_values(["ts_code", "trade_date"])
    m["strength_20d"] = m.groupby("ts_code")["strength"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    out.append(("sent_mf_strength_20d", to_series(m, "strength_20d")))
    del mf, m
    gc.collect()

    di = qdf(cli, f"""
        SELECT ts_code, trade_date, turnover_rate, volume_ratio, pe_ttm, circ_mv, rzye, north_hold
        FROM daily_indicators WHERE trade_date >= '{DAILY_START}'
        ORDER BY ts_code, trade_date""")
    di["trade_date"] = pd.to_datetime(di["trade_date"])
    for c in ["turnover_rate", "volume_ratio", "pe_ttm", "circ_mv", "rzye", "north_hold"]:
        di[c] = pd.to_numeric(di[c], errors="coerce")
    di = di.sort_values(["ts_code", "trade_date"])

    tr = di["turnover_rate"].astype(float)
    di["turn_dev"] = tr / tr.groupby(di["ts_code"]).transform(lambda x: x.rolling(20, min_periods=5).mean()) - 1
    out.append(("sent_turnover_dev", to_series(di, "turn_dev")))

    di["vr_5d"] = di.groupby("ts_code")["volume_ratio"].transform(
        lambda x: x.rolling(5, min_periods=3).mean())
    out.append(("volume_ratio_5d", to_series(di, "vr_5d")))

    di["turn_chg_20d"] = tr / tr.groupby(di["ts_code"]).transform(
        lambda x: x.rolling(60, min_periods=20).mean()) - 1
    out.append(("turnover_chg_20d", to_series(di, "turn_chg_20d")))

    # north_hold 自 2026-07 起被填充为 0（北向日频停发），统一按缺失处理
    for name, col, zero_nan in [("sent_margin_chg_20d", "rzye", True), ("rzye_chg_20d", "rzye", False),
                                ("sent_north_chg_20d", "north_hold", True),
                                ("north_hold_chg_20d", "north_hold", True)]:
        v = di[col].astype(float)
        if zero_nan:
            v = v.replace(0, np.nan)
        chg = v / v.groupby(di["ts_code"]).transform(lambda x: x.shift(20)) - 1
        out.append((name, to_series(di.assign(v=chg), "v")))

    di["ep"] = 1.0 / di["pe_ttm"].replace(0, np.nan)
    di["ep_neutral"] = di["ep"] - di.groupby("trade_date")["ep"].transform("mean")
    out.append(("ep_ind_neutral", to_series(di, "ep_neutral")))

    # Amihud 非流动性：|pct_chg| / (amount/100)
    dp["illiq"] = dp["pct_chg"].abs() / (dp["amount"] / 100).replace(0, np.nan)
    dp = dp.sort_values(["ts_code", "trade_date"])
    dp["illiq_20d"] = dp.groupby("ts_code")["illiq"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    out.append(("amihud_illiq_20d", to_series(dp, "illiq_20d")))

    # 上市天数
    sb = qdf(cli, "SELECT ts_code, list_date FROM stock_basic WHERE list_status='L'")
    dates = qdf(cli, f"""SELECT DISTINCT trade_date FROM daily_prices
                         WHERE trade_date >= '{DAILY_START}' ORDER BY trade_date""")
    cr = sb.merge(dates, how="cross")
    cr["v"] = (cr["trade_date"] - cr["list_date"]).dt.days
    cr = cr[cr["v"] >= 0]
    out.append(("list_age_days", to_series(cr, "v")))
    return out


# ---------------------------- 季度/财报因子 ----------------------------

def build_quarterly(cli):
    """财报类因子：股东户数、毛利率变化、单季营收、应计利润。"""
    out = []

    sh = qdf(cli, f"""SELECT ts_code, ann_date, holder_num FROM stk_holdernumber
                      WHERE ann_date >= '{QUARTER_START}' ORDER BY ts_code, ann_date""")
    sh["ann_date"] = pd.to_datetime(sh["ann_date"])
    sh["holder_num"] = pd.to_numeric(sh["holder_num"], errors="coerce")
    sh = sh.drop_duplicates(subset=["ts_code", "ann_date"], keep="last").sort_values(["ts_code", "ann_date"])

    sh["hn_4q_ago"] = sh.groupby("ts_code")["holder_num"].shift(4)
    sh["hn_chg"] = sh["holder_num"] / sh["hn_4q_ago"].replace(0, np.nan) - 1

    def slope_4q(x):
        if len(x) < 4:
            return np.nan
        return np.polyfit(range(4), x.tail(4).values, 1)[0]

    sh["hn_slope"] = sh.groupby("ts_code")["holder_num"].transform(slope_4q)
    sh = sh.rename(columns={"ann_date": "trade_date"})
    out.append(("holder_num_chg_20d", to_series(sh, "hn_chg")))
    out.append(("holder_num_trend_4q", to_series(sh, "hn_slope")))

    fi = qdf(cli, f"""SELECT ts_code, ann_date, grossprofit_margin, q_sales_yoy
                      FROM financial_indicators WHERE ann_date >= '{QUARTER_START}'
                      ORDER BY ts_code, ann_date""")
    fi["ann_date"] = pd.to_datetime(fi["ann_date"])
    for c in ["grossprofit_margin", "q_sales_yoy"]:
        fi[c] = pd.to_numeric(fi[c], errors="coerce")
    fi = fi.drop_duplicates(subset=["ts_code", "ann_date"], keep="last").sort_values(["ts_code", "ann_date"])

    fi["gpm_4q_ago"] = fi.groupby("ts_code")["grossprofit_margin"].shift(4)
    fi["gpm_chg"] = fi["grossprofit_margin"] - fi["gpm_4q_ago"]
    fi = fi.rename(columns={"ann_date": "trade_date"})
    out.append(("gross_margin_chg_4q", to_series(fi, "gpm_chg")))
    out.append(("q_revenue_growth_yoy", to_series(fi[fi["q_sales_yoy"].notna()], "q_sales_yoy")))

    inc = qdf(cli, f"SELECT ts_code, ann_date, n_income FROM income_statement WHERE ann_date >= '{QUARTER_START}'")
    cf = qdf(cli, f"SELECT ts_code, ann_date, n_cashflow_act FROM cashflow_statement WHERE ann_date >= '{QUARTER_START}'")
    bs = qdf(cli, f"SELECT ts_code, ann_date, total_assets FROM balance_sheet WHERE ann_date >= '{QUARTER_START}'")
    for df in (inc, cf, bs):
        df["ann_date"] = pd.to_datetime(df["ann_date"])
        for c in df.columns[2:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    acc = inc.merge(cf, on=["ts_code", "ann_date"], how="inner") \
             .merge(bs, on=["ts_code", "ann_date"], how="inner")
    acc["accruals"] = (acc["n_income"] - acc["n_cashflow_act"]) / acc["total_assets"].replace(0, np.nan)
    acc = acc.rename(columns={"ann_date": "trade_date"})
    out.append(("accruals_sloan", to_series(acc, "accruals")))
    return out


# ---------------------------- 同花顺情绪因子 ----------------------------

def build_ths(cli):
    """同花顺涨停池/炸板/跌停/龙虎榜/热搜因子。"""
    out = []

    lu = qdf(cli, f"""SELECT trade_date, ticker, continue_day_cnt, seal_money
                      FROM ths_limit_up_pool_daily WHERE trade_date >= '{DAILY_START}'
                      ORDER BY trade_date""")
    lu["trade_date"] = pd.to_datetime(lu["trade_date"])
    lu["ts_code"] = lu["ticker"].map(ticker_to_code)
    lu["continue_day_cnt"] = pd.to_numeric(lu["continue_day_cnt"], errors="coerce")
    lu["seal_money"] = pd.to_numeric(lu["seal_money"], errors="coerce")

    h = lu.groupby(["ts_code", "trade_date"])["continue_day_cnt"].max().reset_index()
    h = h.sort_values(["ts_code", "trade_date"])
    h["v"] = h.groupby("ts_code")["continue_day_cnt"].transform(
        lambda x: x.rolling(20, min_periods=1).max())
    out.append(("limit_up_height_20d", to_series(h, "v")))

    circ = qdf(cli, f"""SELECT ts_code, trade_date, circ_mv FROM daily_indicators
                        WHERE trade_date >= '{DAILY_START}'""")
    circ["trade_date"] = pd.to_datetime(circ["trade_date"])
    circ["circ_mv"] = pd.to_numeric(circ["circ_mv"], errors="coerce")
    seal = lu.merge(circ, on=["ts_code", "trade_date"], how="left")
    seal["seal_ratio"] = seal["seal_money"] / seal["circ_mv"].replace(0, np.nan)
    seal = seal.groupby(["ts_code", "trade_date"])["seal_ratio"].mean().reset_index()
    seal = seal.sort_values(["ts_code", "trade_date"])
    seal["v"] = seal.groupby("ts_code")["seal_ratio"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    out.append(("limit_seal_strength", to_series(seal, "v")))

    br = qdf(cli, f"""SELECT trade_date, ticker, open_times FROM ths_limit_break_pool_daily
                      WHERE trade_date >= '{DAILY_START}'""")
    br["trade_date"] = pd.to_datetime(br["trade_date"])
    br["ts_code"] = br["ticker"].map(ticker_to_code)
    br["open_times"] = pd.to_numeric(br["open_times"], errors="coerce")
    br = br.groupby(["ts_code", "trade_date"])["open_times"].sum().reset_index()
    br = br.sort_values(["ts_code", "trade_date"])
    br["v"] = br.groupby("ts_code")["open_times"].transform(lambda x: x.rolling(20, min_periods=1).sum())
    out.append(("break_count_20d", to_series(br, "v")))

    ld = qdf(cli, f"""SELECT trade_date, ticker FROM ths_limit_down_pool_daily
                      WHERE trade_date >= '{DAILY_START}'""")
    ld["trade_date"] = pd.to_datetime(ld["trade_date"])
    ld["ts_code"] = ld["ticker"].map(ticker_to_code)
    ld = ld.groupby(["ts_code", "trade_date"]).size().reset_index(name="ld_cnt")
    ld = ld.sort_values(["ts_code", "trade_date"])
    ld["v"] = ld.groupby("ts_code")["ld_cnt"].transform(lambda x: x.rolling(20, min_periods=1).sum())
    out.append(("limit_down_count_20d", to_series(ld, "v")))

    hm = qdf(cli, f"""SELECT trade_date, ticker, net_value, buy_value
                      FROM ths_dragon_tiger_hot_money_daily WHERE trade_date >= '{DAILY_START}'""")
    hm["trade_date"] = pd.to_datetime(hm["trade_date"])
    hm["ts_code"] = hm["ticker"].map(ticker_to_code)
    for c in ["net_value", "buy_value"]:
        hm[c] = pd.to_numeric(hm[c], errors="coerce")
    hm = hm.groupby(["ts_code", "trade_date"]).agg(net_sum=("net_value", "sum"),
                                                   total=("buy_value", "sum")).reset_index()
    hm["net_rate"] = hm["net_sum"] / hm["total"].replace(0, np.nan)
    hm = hm.sort_values(["ts_code", "trade_date"])
    hm["v"] = hm.groupby("ts_code")["net_rate"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    out.append(("hot_money_net_rate_20d", to_series(hm, "v")))

    lhb = qdf(cli, f"""SELECT trade_date, ticker, org_net_value, buy_value
                       FROM ths_dragon_tiger_stock_daily WHERE trade_date >= '{DAILY_START}'
                       ORDER BY trade_date""")
    lhb["trade_date"] = pd.to_datetime(lhb["trade_date"])
    lhb["ts_code"] = lhb["ticker"].map(ticker_to_code)
    for c in ["org_net_value", "buy_value"]:
        lhb[c] = pd.to_numeric(lhb[c], errors="coerce")

    lhb_pivot = lhb.groupby(["ts_code", "trade_date"]).size().unstack(fill_value=0)
    lhb_cnt = lhb_pivot.T.rolling(20, min_periods=1).sum().T
    cnt_long = lhb_cnt.stack().reset_index()
    cnt_long.columns = ["ts_code", "trade_date", "v"]
    out.append(("lhb_count_20d", to_series(cnt_long, "v")))

    lhb["org_rate"] = lhb["org_net_value"] / lhb["buy_value"].replace(0, np.nan)
    org = lhb.groupby(["ts_code", "trade_date"])["org_rate"].mean().reset_index()
    org = org.sort_values(["ts_code", "trade_date"])
    org["v"] = org.groupby("ts_code")["org_rate"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    out.append(("lhb_org_net_rate", to_series(org, "v")))

    hr = qdf(cli, f"""SELECT date AS trade_date, ticker, rank FROM ths_hot_stock_rank_daily
                      WHERE date >= '{DAILY_START}' ORDER BY date""")
    hr["trade_date"] = pd.to_datetime(hr["trade_date"])
    hr["ts_code"] = hr["ticker"].map(ticker_to_code)
    hr["rank"] = pd.to_numeric(hr["rank"], errors="coerce")
    hr = hr.sort_values(["ts_code", "trade_date"])
    hr["rank_20d_ago"] = hr.groupby("ts_code")["rank"].transform(lambda x: x.shift(20))
    hr["v"] = hr["rank_20d_ago"] - hr["rank"]
    out.append(("hot_rank_chg_20d", to_series(hr, "v")))
    return out


def main():
    ap = argparse.ArgumentParser(description="ClickHouse 尾部补齐因子仓库")
    ap.add_argument("--groups", default="daily,quarterly,ths", help="daily,quarterly,ths 组合")
    ap.add_argument("--only", default="", help="仅处理指定因子，逗号分隔")
    ap.add_argument("--daily-start", default="", help="覆盖日频尾部窗口起始日")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    global DAILY_START
    if args.daily_start:
        DAILY_START = args.daily_start

    t0 = time.time()
    cli = get_ch()
    print(f"ClickHouse {CH_HOST}:{CH_PORT}/{CH_DATABASE} 已连接", flush=True)

    items = []
    if "daily" in groups:
        print("== 日频因子 ==", flush=True)
        items += build_daily(cli)
    if "quarterly" in groups:
        print("== 财报类因子 ==", flush=True)
        items += build_quarterly(cli)
    if "ths" in groups:
        print("== 同花顺情绪因子 ==", flush=True)
        items += build_ths(cli)

    print(f"共 {len(items)} 个因子待写入", flush=True)
    only = {n.strip() for n in args.only.split(",") if n.strip()}
    if only:
        items = [(n, s) for n, s in items if n in only]
        print(f"按 --only 过滤后 {len(items)} 个", flush=True)
    ok, fail = 0, []
    for name, s in items:
        try:
            append_factor(name, s)
            ok += 1
        except Exception as e:
            fail.append((name, str(e)[:200]))
            print(f"  [ERR] {name}: {str(e)[:200]}", flush=True)
    print(f"\n完成：成功={ok} | 失败={len(fail)} | 耗时={time.time()-t0:.0f}s", flush=True)
    for n, e in fail:
        print(f"  失败 {n}: {e}", flush=True)


if __name__ == "__main__":
    main()
