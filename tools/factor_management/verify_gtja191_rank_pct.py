# -*- coding: utf-8 -*-
"""GTJA191 对拍样例：Alpha3 / Alpha4 / Alpha42 研报式(1..N) + 百分位入库 vs 仓库现状。

口径：
- 价格/成交量取自 FactorStore._build_adj_sql（与生产链路完全一致，前复权）
- 研报式实现：RANK 使用 1..N，STD 使用样本标准差 STDDEV_SAMP
- 入库版：对最终因子值做当日截面百分位 (RANK()-1)/(N-1)

用法（只读校验，不写仓库）：
    python tools/factor_management/verify_gtja191_rank_pct.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\Quant\Qlibworks\src")
from qlworks.data import QuantDataAPI
from qlworks.evaluation.factor_store import FactorStore

START, END = "2020-01-01", "2026-10-23"
WH = Path(r"E:\Quant\Qlibworks\factor_data\warehouse")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("拉取 OHLCV（与生产同口径 _build_adj_sql）")
api = QuantDataAPI()
raw = api.query(FactorStore._build_adj_sql(None, START, END))
log(f"原始数据 {len(raw)} 行，{raw['ts_code'].nunique()} 只")

# 统一到仓库实际股票池（Alpha3），保证截面百分位可比
_tmp = duckdb.connect()
_pq = (WH / "Alpha3" / "data.parquet").as_posix()
universe = set(_tmp.execute(f"SELECT DISTINCT instrument FROM read_parquet('{_pq}')").df()["instrument"])
_tmp.close()
raw = raw[raw["ts_code"].isin(universe)].reset_index(drop=True)
n_dup = int(raw.duplicated(subset=["ts_code", "trade_date"]).sum())
raw = (raw.sort_values(["ts_code", "trade_date"])
          .drop_duplicates(["ts_code", "trade_date"], keep="first")
          .reset_index(drop=True))
log(f"限定仓库股票池后 {len(raw)} 行，{raw['ts_code'].nunique()} 只（重复 stock-date 行 {n_dup} 已去重）")

conn = duckdb.connect()
conn.register("_raw", raw)

SQL_A3 = """
CREATE OR REPLACE TABLE a3_raw AS
WITH s AS (
    SELECT ts_code, trade_date, close, high, low,
           LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) AS pc
    FROM _raw
)
SELECT ts_code, trade_date,
       SUM(CASE WHEN close = pc THEN 0.0
                WHEN close > pc THEN close - LEAST(low, pc)
                ELSE close - GREATEST(high, pc) END)
           OVER (PARTITION BY ts_code ORDER BY trade_date
                 ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS value
FROM s
WHERE pc IS NOT NULL
"""

# 研报原式：第三层用 (SUM(CLOSE,2)/2)/(SUM(CLOSE,8)/8) 与 1、1.2 比较
SQL_A4 = """
CREATE OR REPLACE TABLE a4_raw AS
WITH s AS (
    SELECT ts_code, trade_date,
           AVG(close) OVER w2 AS ma2,
           AVG(close) OVER w8 AS ma8,
           STDDEV_SAMP(close) OVER w8 AS sd8
    FROM _raw
    WINDOW w2 AS (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW),
           w8 AS (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW)
)
SELECT ts_code, trade_date,
       CASE WHEN sd8 IS NULL THEN NULL
            WHEN ma8 + sd8 < ma2 THEN -1.0
            WHEN ma2 < ma8 - sd8 THEN 1.0
            WHEN ma2 / ma8 > 1 AND ma2 / ma8 < 1.2 THEN 1.0
            WHEN ma2 / ma8 > 1.2 THEN -1.0
            ELSE 0.0 END AS value
FROM s
"""

SQL_A42 = """
CREATE OR REPLACE TABLE a42_raw AS
WITH s AS (
    SELECT ts_code, trade_date,
           STDDEV_SAMP(high) OVER w AS sd10,
           CORR(high, volume) OVER w AS corr_hv
    FROM _raw
    WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
)
SELECT ts_code, trade_date,
       (-1.0) * RANK() OVER (PARTITION BY trade_date ORDER BY sd10) * corr_hv AS value
FROM s
WHERE sd10 IS NOT NULL AND corr_hv IS NOT NULL
"""

PCT = """
CREATE OR REPLACE TABLE {dst} AS
SELECT ts_code, trade_date,
       (RANK() OVER (PARTITION BY trade_date ORDER BY value) - 1)
       / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1, 0) AS value
FROM {src}
WHERE value IS NOT NULL AND isfinite(value)
"""


def build(name: str) -> None:
    t0 = time.time()
    conn.execute({"A3": SQL_A3, "A4": SQL_A4, "A42": SQL_A42}[name])
    src = {"A3": "a3_raw", "A4": "a4_raw", "A42": "a42_raw"}[name]
    conn.execute(PCT.format(dst=f"{src[:-4]}_pct", src=src))
    n = conn.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
    log(f"{name} 计算完成 {n} 行，耗时 {time.time() - t0:.1f}s")


for k in ("A3", "A4", "A42"):
    build(k)

# ── 现有仓库值 ──
for k, fn in (("A3", "Alpha3"), ("A4", "Alpha4"), ("A42", "Alpha42")):
    conn.execute(f"""
        CREATE OR REPLACE VIEW old_{k.lower()} AS
        SELECT CAST(instrument AS VARCHAR) AS ts_code,
               CAST(datetime AS DATE) AS trade_date,
               CAST(value AS DOUBLE) AS value
        FROM read_parquet('{(WH / fn / "data.parquet").as_posix()}')
    """)
log("仓库现有数据已注册")


def compare(new_tbl: str, old_view: str) -> dict:
    """原值口径：行数覆盖 + 逐日 Spearman / Pearson。"""
    base = f"""
        SELECT n.ts_code, n.trade_date, n.value AS nv, o.value AS ov
        FROM {new_tbl} n JOIN {old_view} o USING (ts_code, trade_date)
        WHERE n.value IS NOT NULL AND o.value IS NOT NULL
          AND isfinite(n.value) AND isfinite(o.value)
    """
    n_new = conn.execute(f"SELECT COUNT(*) FROM {new_tbl}").fetchone()[0]
    n_old = conn.execute(f"SELECT COUNT(*) FROM {old_view}").fetchone()[0]
    n_j = conn.execute(f"SELECT COUNT(*) FROM ({base})").fetchone()[0]
    daily = conn.execute(f"""
        WITH j AS ({base}),
        r AS (SELECT trade_date, nv, ov,
                     RANK() OVER (PARTITION BY trade_date ORDER BY nv) AS rn,
                     RANK() OVER (PARTITION BY trade_date ORDER BY ov) AS ro
              FROM j)
        SELECT trade_date, COUNT(*) AS n, corr(rn, ro) AS sp, corr(nv, ov) AS pe
        FROM r GROUP BY trade_date
    """).df()
    return {
        "n_new": n_new, "n_old": n_old, "n_joined": n_j,
        "days": len(daily),
        "sp_mean": daily["sp"].mean(), "sp_min": daily["sp"].min(),
        "sp_p05": daily["sp"].quantile(0.05),
        "pe_mean": daily["pe"].mean(),
    }


def pct_diff(new_pct: str, old_pct: str) -> dict:
    """百分位口径：平均/最大绝对差、完全相等占比。"""
    r = conn.execute(f"""
        WITH j AS (
            SELECT n.value AS nv, o.value AS ov
            FROM {new_pct} n JOIN {old_pct} o USING (ts_code, trade_date)
            WHERE n.value IS NOT NULL AND o.value IS NOT NULL
        )
        SELECT COUNT(*) AS n, AVG(ABS(nv - ov)) AS mae, MAX(ABS(nv - ov)) AS mx,
               AVG(CASE WHEN ABS(nv - ov) < 1e-6 THEN 1.0 ELSE 0.0 END) AS exact
        FROM j
    """).fetchdf().iloc[0]
    return {"pct_n": int(r["n"]), "pct_mae": r["mae"],
            "pct_max": r["mx"], "pct_exact": r["exact"]}


# 仓库值也转百分位，才能和"百分位入库"版本同尺度比较
for k in ("a3", "a4", "a42"):
    conn.execute(f"""
        CREATE OR REPLACE VIEW old_{k}_pct AS
        SELECT ts_code, trade_date,
               (RANK() OVER (PARTITION BY trade_date ORDER BY value) - 1)
               / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1, 0) AS value
        FROM old_{k}
        WHERE value IS NOT NULL AND isfinite(value)
    """)

rows = []
for k, fn in (("A3", "Alpha3"), ("A4", "Alpha4"), ("A42", "Alpha42")):
    r = {"因子": fn}
    r.update(compare(f"a{k[1:]}_raw", f"old_{k.lower()}"))
    r.update(pct_diff(f"a{k[1:]}_pct", f"old_{k.lower()}_pct"))
    rows.append(r)

df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\n=== 对拍汇总（原值 Spearman/Pearson；百分位差异） ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── Alpha4 取值分布（-1/0/1）──
dist = conn.execute("""
    SELECT '研报式' AS src, value, COUNT(*) AS n FROM a4_raw GROUP BY value
    UNION ALL
    SELECT '仓库' AS src, value, COUNT(*) AS n FROM old_a4 GROUP BY value
    ORDER BY src, value
""").df()
print("\nAlpha4 取值分布：")
print(dist.to_string(index=False))

# ── Alpha42：1..N 与百分位两种尺度之间的截面排序差异 ──
d = conn.execute("""
    WITH j AS (
        SELECT a.ts_code, a.trade_date, a.value AS rv, b.value AS pv
        FROM a42_raw a JOIN a42_pct b USING (ts_code, trade_date)
    ),
    r AS (SELECT trade_date, rv, pv,
                 RANK() OVER (PARTITION BY trade_date ORDER BY rv) AS r1,
                 RANK() OVER (PARTITION BY trade_date ORDER BY pv) AS r2 FROM j)
    SELECT trade_date, COUNT(*) AS n, corr(r1, r2) AS sp FROM r GROUP BY trade_date
""").df()
print(f"\nAlpha42 内部：1..N 版 vs 百分位版 逐日Spearman 均值={d['sp'].mean():.4f} "
      f"最小={d['sp'].min():.4f} 非完全一致天数={(d['sp'] < 0.9999).sum()}/{len(d)}")

# ── Alpha3 原值逐行相对误差（判断 qlib Less/Greater 是否真的算错）──
a3 = conn.execute("""
    WITH j AS (
        SELECT n.ts_code, n.trade_date, n.value AS nv, o.value AS ov
        FROM a3_raw n JOIN old_a3 o USING (ts_code, trade_date)
        WHERE n.value IS NOT NULL AND o.value IS NOT NULL
    )
    SELECT COUNT(*) AS n,
           AVG(CASE WHEN ABS(nv - ov) <= 1e-5 * GREATEST(1.0, ABS(ov)) THEN 1.0 ELSE 0.0 END) AS rel_exact,
           MAX(ABS(nv - ov)) AS max_abs,
           AVG(ABS(nv - ov)) AS mae
    FROM j
""").fetchdf().iloc[0]
print(f"\nAlpha3 原值逐行：样本={int(a3['n'])} 相对一致率={a3['rel_exact']:.4f} "
      f"平均绝对差={a3['mae']:.6f} 最大绝对差={a3['max_abs']:.4f}")

print("\nAlpha3 差异最大的 5 条：")
print(conn.execute("""
    WITH j AS (
        SELECT n.ts_code, n.trade_date, n.value AS nv, o.value AS ov
        FROM a3_raw n JOIN old_a3 o USING (ts_code, trade_date)
        WHERE n.value IS NOT NULL AND o.value IS NOT NULL
    )
    SELECT ts_code, trade_date, ROUND(nv, 4) AS 研报式, ROUND(ov, 4) AS 仓库,
           ROUND(ABS(nv - ov), 4) AS diff
    FROM j ORDER BY ABS(nv - ov) DESC LIMIT 5
""").df().to_string(index=False))

# ── Alpha3 差异集中度：多少只股票贡献了偏差 ──
conc = conn.execute("""
    WITH j AS (
        SELECT n.ts_code, ABS(n.value - o.value) AS d
        FROM a3_raw n JOIN old_a3 o USING (ts_code, trade_date)
        WHERE n.value IS NOT NULL AND o.value IS NOT NULL
    )
    SELECT COUNT(*) AS n_all,
           COUNT(*) FILTER (WHERE d > 0.5) AS n_big,
           COUNT(DISTINCT CASE WHEN d > 0.5 THEN ts_code END) AS n_stock_big,
           COUNT(DISTINCT ts_code) AS n_stock_all
    FROM j
""").fetchdf().iloc[0]
print(f"\nAlpha3 偏差集中度：|diff|>0.5 的行={int(conc['n_big'])}/{int(conc['n_all'])} "
      f"({conc['n_big']/conc['n_all']:.2%})，涉及股票={int(conc['n_stock_big'])}/{int(conc['n_stock_all'])}")
print(conn.execute("""
    WITH j AS (
        SELECT n.ts_code, ABS(n.value - o.value) AS d
        FROM a3_raw n JOIN old_a3 o USING (ts_code, trade_date)
        WHERE n.value IS NOT NULL AND o.value IS NOT NULL
    )
    SELECT ts_code, COUNT(*) AS n_big, ROUND(MAX(d), 2) AS max_d
    FROM j WHERE d > 0.5 GROUP BY ts_code ORDER BY n_big DESC LIMIT 10
""").df().to_string(index=False))

# ── Alpha42 排序最差的交易日 ──
worst = conn.execute("""
    WITH j AS (
        SELECT a.ts_code, a.trade_date, a.value AS nv, b.value AS ov
        FROM a42_raw a JOIN old_a42 b USING (ts_code, trade_date)
        WHERE a.value IS NOT NULL AND b.value IS NOT NULL
    ),
    r AS (SELECT trade_date, nv, ov,
                 RANK() OVER (PARTITION BY trade_date ORDER BY nv) AS r1,
                 RANK() OVER (PARTITION BY trade_date ORDER BY ov) AS r2 FROM j)
    SELECT trade_date, COUNT(*) AS n, corr(r1, r2) AS sp FROM r GROUP BY trade_date
    ORDER BY sp ASC LIMIT 5
""").df()
print("\nAlpha42 逐日排序最差的 5 天：")
print(worst.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── 抽样：2026-10-23 截面 Top5 ──
for k in ("a3", "a4", "a42"):
    s = conn.execute(f"""
        SELECT ts_code, ROUND(value, 6) AS value FROM {k}_pct
        WHERE trade_date = DATE '2026-10-23' ORDER BY value DESC LIMIT 5
    """).df()
    print(f"\n{k}_pct 2026-10-23 Top5:\n{s.to_string(index=False)}")

conn.close()
log("完成")
