# -*- coding: utf-8 -*-
"""sentiment 家族因子构建：ClickHouse(同花顺) → factor_data/warehouse/<name>/{data.parquet, meta.json}

因子清单（全部个股截面级，可进 factor_screen 筛选）：
  1. sent_mf_net_ratio    主力净流入率   = money_flow.net_mf_amount / (daily_prices.amount*10000)
  2. sent_turnover_dev    换手率偏离     = turnover_rate / Mean20(turnover_rate) - 1
  3. sent_margin_chg_20d  两融余额20日变化 = rzye / Ref20(rzye) - 1
  4. sent_north_chg_20d   北向持股20日变化 = north_hold / Ref20(north_hold) - 1  [2024-08 后断更]

数据窗口：2020-01-01 ~ 2026-10-30（对齐 pipeline_config 主窗口；底层表历史更长可扩展）
"""
import sys, json, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\Quant\Qlibworks")
import clickhouse_connect
from qlworks.config import (CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE,
                            WAREHOUSE_DIR, PROJECT_ROOT)

START = "2020-01-01"
END = "2026-10-30"
FAMILY = "sentiment"
LIB_DIR = PROJECT_ROOT / "factor_data" / "factor_library"

cli = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, user=CH_USER,
                                    password=CH_PASSWORD, database=CH_DATABASE,
                                    connect_timeout=30)


def fetch_month(table, cols, ym):
    ym_next = (pd.Period(ym, "M") + 1).strftime("%Y-%m")
    sql = (f"SELECT {cols} FROM {table} "
           f"WHERE trade_date >= '{ym}-01' AND trade_date < '{ym_next}-01' ORDER BY ts_code, trade_date")
    return cli.query(sql).result_rows


def load_table(table, cols, start=START, end=END):
    """按月循环加载（避免大查询超时）"""
    months = pd.period_range(start, end, freq="M")
    frames = []
    for m in months:
        ym = m.strftime("%Y-%m")
        rows = fetch_month(table, cols, ym)
        if rows:
            frames.append(pd.DataFrame(rows, columns=cols.split(",")))
        time.sleep(0.05)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def save_factor(name, series: pd.Series, meaning, qlib_expr, source_desc):
    """series: MultiIndex (instrument, datetime)，值 float32"""
    d = WAREHOUSE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    df = series.rename("value").astype("float32").to_frame()
    df = df.sort_index()
    df.to_parquet(d / "data.parquet", index=True)
    years = sorted({ts.year for ts in df.index.get_level_values("datetime")})
    meta = {
        "factor_name": name,
        "data_version": "1.0",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_range": {
            "years": years,
            "first_date": str(df.index.get_level_values("datetime").min().date()),
            "last_date": str(df.index.get_level_values("datetime").max().date()),
            "total_records": int(len(df)),
            "unique_stocks": int(df.index.get_level_values("instrument").nunique()),
        },
        "usage_scenario": "市场情绪/资金情绪截面因子（sentiment 家族）。",
        "expression": {"qlib": qlib_expr, "source": source_desc},
        "category": "sentiment",
        "strategy_hint": "情绪类因子：高换手折价、主力净流入溢价、杠杆与北向资金动量。",
        "meaning": meaning,
        "family": FAMILY,
        "lifecycle_stage": "active",
    }
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {name}: {len(df):,} 行, {df.index.get_level_values('instrument').nunique()} 只, "
          f"{meta['data_range']['first_date']} ~ {meta['data_range']['last_date']}")
    return df


def main():
    print("== 加载 ClickHouse 数据 ==")
    t0 = time.time()
    # money_flow + amount
    mf = load_table("money_flow", "ts_code,trade_date,net_mf_amount")
    px = load_table("daily_prices", "ts_code,trade_date,amount")
    di = load_table("daily_indicators", "ts_code,trade_date,turnover_rate,rzye,north_hold")
    print(f"  加载完成: money_flow {len(mf):,} / daily_prices {len(px):,} / daily_indicators {len(di):,} 行, "
          f"{time.time()-t0:.0f}s")

    def to_panel(df, col):
        p = df.pivot_table(index=["ts_code", "trade_date"], values=col, aggfunc="first")
        p.index.names = ["instrument", "datetime"]
        return p.iloc[:, 0]

    print("\n== 构建因子 ==")

    # 1. 主力净流入率 = net_mf_amount / (amount*10000)
    m = mf.merge(px, on=["ts_code", "trade_date"], how="inner")
    s = (m["net_mf_amount"].astype(float) / (m["amount"].astype(float) * 10000.0)).replace([np.inf, -np.inf], np.nan)
    s.index = pd.MultiIndex.from_frame(m[["ts_code", "trade_date"]])
    s.index.names = ["instrument", "datetime"]
    save_factor("sent_mf_net_ratio", s,
                "主力资金净流入/成交额（资金情绪：主力净买入占比，>0 资金流入）。",
                "CH: money_flow.net_mf_amount / (daily_prices.amount*1e4)",
                "ClickHouse 同花顺 money_flow + daily_prices，2010 起")

    # 2. 换手率偏离 = turnover_rate / Mean20(turnover_rate) - 1
    tr = to_panel(di, "turnover_rate").astype(float).replace(0, np.nan)
    m20 = tr.groupby(level="instrument").transform(lambda x: x.rolling(20, min_periods=5).mean())
    s2 = (tr / m20 - 1.0).rename("v")
    save_factor("sent_turnover_dev", s2,
                "换手率 / 20日均换手 - 1（换手异动情绪：>0 放量、<0 缩量）。",
                "CH: daily_indicators.turnover_rate / Mean20 - 1",
                "ClickHouse 同花顺 daily_indicators，1990 起")

    # 3. 两融余额 20 日变化 = rzye / Ref20 - 1
    rz = to_panel(di, "rzye").astype(float).replace(0, np.nan)
    rz20 = rz.groupby(level="instrument").transform(lambda x: x.shift(20))
    s3 = (rz / rz20 - 1.0).rename("v")
    save_factor("sent_margin_chg_20d", s3,
                "融资余额 20 日变化率（杠杆情绪：融资买入动能，>0 加杠杆）。",
                "CH: daily_indicators.rzye / Ref20 - 1",
                "ClickHouse 同花顺 daily_indicators.rzye，2010-03 起")

    # 4. 北向持股 20 日变化（2024-08 后监管停发日频）
    nh = to_panel(di, "north_hold").astype(float).replace(0, np.nan)
    nh20 = nh.groupby(level="instrument").transform(lambda x: x.shift(20))
    s4 = (nh / nh20 - 1.0).rename("v")
    save_factor("sent_north_chg_20d", s4,
                "北向持股 20 日变化率（外资情绪）。注意：2024-08 起日频停发，样本止于 2024-06。",
                "CH: daily_indicators.north_hold / Ref20 - 1",
                "ClickHouse 同花顺 daily_indicators.north_hold，2017 起 ~ 2024-06 断更")

    print(f"\n完成，耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
