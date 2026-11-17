# -*- coding: utf-8 -*-
"""GTJA191 全量对拍：研报式(RANK 1..N) + 百分位入库 vs 仓库现状。

口径（与 verify_gtja191_rank_pct.py 保持一致）：
- 价格/成交量来自 ClickHouse（FactorStore._build_adj_sql，前复权，与生产链路一致）
- 研报式实现：字典里的 PERCENT_RANK() 统一替换为 RANK()，即横截面 1..N 排名
- 入库版：对最终因子值再做当日截面百分位 (RANK()-1)/(N-1)
- 对拍基准：仓库 factor_data/warehouse/{因子}/data.parquet，同样转百分位后比较

范围：
- 191 个因子全部计算，记录行数/日期覆盖率/异常
- 仅对仓库已有数据的因子做 A/B 对拍（逐日 Spearman + 百分位 MAE）

用法（只读校验，不写仓库）：
    python tools/factor_management/batch_verify_gtja191.py
输出：
    factor_data/archive_verify/gtja191_rank_pct_verify_<时间戳>.csv
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import duckdb
import pandas as pd
import yaml

sys.path.insert(0, r"E:\Quant\Qlibworks\src")
from qlworks.data import QuantDataAPI
from qlworks.evaluation.factor_store import FactorStore

START, END = "2020-01-01", "2026-10-23"
ROOT = Path(r"E:\Quant\Qlibworks")
WH = ROOT / "factor_data" / "warehouse"
DICT = ROOT / "factor_data" / "factor_library" / "archive" / "gtja191_factor_dictionary.yaml"
OUT_DIR = ROOT / "factor_data" / "archive_verify"
OUT_CSV = OUT_DIR / f"gtja191_rank_pct_verify_{time.strftime('%Y%m%d_%H%M')}.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_expressions() -> dict[str, str]:
    """读取字典中的 duckdb 表达式，并把百分位排名改成研报式 1..N 排名。"""
    data = yaml.safe_load(DICT.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for f in data["factors"]:
        expr = ((f.get("expression") or {}).get("duckdb") or "").strip().rstrip(";")
        if not expr:
            out[f["name"]] = ""
            continue
        out[f["name"]] = expr.replace("PERCENT_RANK()", "RANK()")
    return out


def build_source(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """拉取一次原始行情并按 (ts_code, trade_date) 去重。"""
    api = QuantDataAPI()
    raw = api.query(FactorStore._build_adj_sql(None, START, END))
    n0 = len(raw)
    raw = (raw.sort_values(["ts_code", "trade_date"])
              .drop_duplicates(["ts_code", "trade_date"], keep="first")
              .reset_index(drop=True))
    conn.register("_raw_all", raw)
    log(f"原始行情 {n0} 行 → 去重后 {len(raw)} 行，{raw['ts_code'].nunique()} 只")
    return len(raw), raw["ts_code"].nunique()


def factor_instruments(conn: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame | None:
    """取仓库该因子的股票池；无仓库数据返回 None。"""
    pq = WH / name / "data.parquet"
    if not pq.exists():
        return None
    conn.execute(f"""
        CREATE OR REPLACE VIEW t_old AS
        SELECT CAST(instrument AS VARCHAR) AS ts_code,
               CAST(datetime AS DATE) AS trade_date,
               CAST(value AS DOUBLE) AS value
        FROM read_parquet('{pq.as_posix()}')
    """)
    return conn.execute("SELECT DISTINCT ts_code FROM t_old").df()


def old_meta(name: str) -> str:
    """仓库该因子当初用哪种后端算的（qlib / duckdb）。"""
    p = WH / name / "meta.json"
    if not p.exists():
        return ""
    import json
    meta = json.loads(p.read_text(encoding="utf-8"))
    duck = ((meta.get("expression") or {}).get("duckdb") or "")
    return "duckdb" if duck and "暂未实现" not in duck else "qlib"


def compute_new(conn: duckdb.DuckDBPyConnection, expr: str) -> None:
    """按研报式计算原始值，并生成百分位入库版。"""
    inner = expr if expr.upper().startswith("WITH") else \
        f"SELECT ts_code, trade_date, ({expr}) AS value FROM _raw"
    conn.execute(f"""
        CREATE OR REPLACE TABLE t_raw AS
        SELECT ts_code, trade_date, CAST(value AS DOUBLE) AS value
        FROM ({inner})
    """)
    conn.execute("""
        CREATE OR REPLACE TABLE t_pct AS
        SELECT ts_code, trade_date,
               (RANK() OVER (PARTITION BY trade_date ORDER BY value) - 1)
               / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1, 0) AS value
        FROM t_raw
        WHERE value IS NOT NULL AND isfinite(value)
    """)


def compare(conn: duckdb.DuckDBPyConnection) -> dict:
    """原值口径逐日 Spearman/Pearson + 百分位口径差异。"""
    base = """
        SELECT n.ts_code, n.trade_date, n.value AS nv, o.value AS ov
        FROM t_raw n JOIN t_old o USING (ts_code, trade_date)
        WHERE n.value IS NOT NULL AND o.value IS NOT NULL
          AND isfinite(n.value) AND isfinite(o.value)
    """
    n_new = conn.execute("SELECT COUNT(*) FROM t_pct").fetchone()[0]
    n_old = conn.execute("SELECT COUNT(*) FROM t_old").fetchone()[0]
    n_j = conn.execute(f"SELECT COUNT(*) FROM ({base})").fetchone()[0]
    res = {"n_new": int(n_new), "n_old": int(n_old), "n_joined": int(n_j),
           "n_days": 0, "sp_mean": None, "sp_min": None, "sp_p05": None,
           "pct_mae": None, "pct_max": None}
    if n_j == 0:
        return res

    daily = conn.execute(f"""
        WITH j AS ({base}),
        r AS (SELECT trade_date, nv, ov,
                     RANK() OVER (PARTITION BY trade_date ORDER BY nv) AS rn,
                     RANK() OVER (PARTITION BY trade_date ORDER BY ov) AS ro
              FROM j)
        SELECT trade_date, COUNT(*) AS n, corr(rn, ro) AS sp FROM r GROUP BY trade_date
    """).df()
    res["n_days"] = len(daily)
    if len(daily):
        res["sp_mean"] = float(daily["sp"].mean())
        res["sp_min"] = float(daily["sp"].min())
        res["sp_p05"] = float(daily["sp"].quantile(0.05))

    pct = conn.execute("""
        WITH n AS (
            SELECT ts_code, trade_date,
                   (RANK() OVER (PARTITION BY trade_date ORDER BY value) - 1)
                   / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1, 0) AS v
            FROM t_pct
        ),
        o AS (
            SELECT ts_code, trade_date,
                   (RANK() OVER (PARTITION BY trade_date ORDER BY value) - 1)
                   / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1, 0) AS v
            FROM t_old
            WHERE value IS NOT NULL AND isfinite(value)
        )
        SELECT COUNT(*) AS n, AVG(ABS(n.v - o.v)) AS mae, MAX(ABS(n.v - o.v)) AS mx
        FROM n JOIN o USING (ts_code, trade_date)
    """).fetchdf().iloc[0]
    if pct["n"]:
        res["pct_mae"] = float(pct["mae"])
        res["pct_max"] = float(pct["mx"])
    return res


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exprs = load_expressions()
    conn = duckdb.connect()
    build_source(conn)

    # 默认股票池：Alpha3 的仓库股票池（与单因子样例口径一致）
    default_univ = factor_instruments(conn, "Alpha3")
    if default_univ is None:
        raise RuntimeError("仓库缺少 Alpha3，无法确定默认股票池")
    conn.register("_univ_tmp", default_univ)
    conn.execute("CREATE OR REPLACE TABLE univ_default AS SELECT * FROM _univ_tmp")
    conn.unregister("_univ_tmp")

    rows: list[dict] = []
    t_all = time.time()
    names = sorted(exprs, key=lambda s: int(s.replace("Alpha", "")))
    for i, name in enumerate(names, 1):
        row: dict = {"factor": name, "status": "ok", "error": "",
                     "old_backend": old_meta(name), "has_warehouse": False}
        try:
            univ = factor_instruments(conn, name)
            if univ is None:
                conn.execute("CREATE OR REPLACE TABLE univ_f AS SELECT * FROM univ_default")
            else:
                row["has_warehouse"] = True
                conn.register("_univ_tmp", univ)
                conn.execute("CREATE OR REPLACE TABLE univ_f AS SELECT * FROM _univ_tmp")
                conn.unregister("_univ_tmp")
            conn.execute("""
                CREATE OR REPLACE VIEW _raw AS
                SELECT r.* FROM _raw_all r
                JOIN univ_f u ON r.ts_code = u.ts_code
            """)
            expr = exprs[name]
            if not expr:
                raise ValueError("字典中无 duckdb 表达式")
            compute_new(conn, expr)
            n_raw = conn.execute("SELECT COUNT(*) FROM t_raw").fetchone()[0]
            row["n_raw"] = int(n_raw)
            row["n_new"] = int(conn.execute("SELECT COUNT(*) FROM t_pct").fetchone()[0])
            if row["has_warehouse"]:
                row.update(compare(conn))
            log(f"{i}/{len(names)} {name} 行={row['n_raw']} "
                f"sp_mean={row.get('sp_mean')}")
        except Exception as exc:  # 单因子失败不影响整批
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            log(f"{i}/{len(names)} {name} 失败：{row['error']}")
            traceback.print_exc(limit=1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    ok = df[df["status"] == "ok"]
    wh = ok[ok["has_warehouse"]]
    print("\n===== 全量体检 =====")
    print(f"因子总数={len(df)}  计算成功={len(ok)}  失败={len(df) - len(ok)}  "
          f"有仓库基准={len(wh)}  耗时={time.time() - t_all:.0f}s")
    print(f"结果明细：{OUT_CSV}")

    if len(df[df["status"] != "ok"]):
        print("\n计算失败清单：")
        print(df[df["status"] != "ok"][["factor", "error"]].to_string(index=False))

    if len(wh):
        print("\n===== 对拍分布（逐日 Spearman 均值）=====")
        bins = [
            ("≥0.99 基本一致", wh["sp_mean"] >= 0.99),
            ("0.95~0.99 轻微差异", (wh["sp_mean"] >= 0.95) & (wh["sp_mean"] < 0.99)),
            ("0.80~0.95 明显差异", (wh["sp_mean"] >= 0.80) & (wh["sp_mean"] < 0.95)),
            ("<0.80 定义/口径不一致", wh["sp_mean"] < 0.80),
        ]
        for label, mask in bins:
            print(f"  {label}: {int(mask.sum())}")

        cols = ["factor", "old_backend", "n_joined", "n_days", "sp_mean",
                "sp_min", "pct_mae", "pct_max"]
        print("\n最差 20 个（按 sp_mean 升序）：")
        print(wh.sort_values("sp_mean")[cols].head(20).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n最好 10 个（按 sp_mean 降序）：")
        print(wh.sort_values("sp_mean", ascending=False)[cols].head(10).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))

    conn.close()
    log("完成")


if __name__ == "__main__":
    main()
