import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qlworks.config import DUCKDB_PATH, QLIB_DATA_DIR  # noqa: E402


_SYMBOL_KEYS = ["symbol", "code", "ticker", "instrument", "secid", "ts_code"]
_DATE_KEYS = ["date", "datetime", "trade_date", "timestamp", "time"]
_OPEN_KEYS = ["open", "open_price"]
_HIGH_KEYS = ["high", "high_price"]
_LOW_KEYS = ["low", "low_price"]
_CLOSE_KEYS = ["close", "close_price", "last"]
_VOLUME_KEYS = ["volume", "vol"]
_AMOUNT_KEYS = ["amount", "turnover", "value"]


def _colmap(columns: List[str], keys: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in columns}
    for k in keys:
        if k in cols:
            return cols[k]
    return None


def _detect_ohlcv_columns(columns: List[str]) -> Optional[Dict[str, str]]:
    m: Dict[str, str] = {}
    s = _colmap(columns, _SYMBOL_KEYS)
    d = _colmap(columns, _DATE_KEYS)
    o = _colmap(columns, _OPEN_KEYS)
    h = _colmap(columns, _HIGH_KEYS)
    l = _colmap(columns, _LOW_KEYS)
    cl = _colmap(columns, _CLOSE_KEYS)
    v = _colmap(columns, _VOLUME_KEYS)
    if not (s and d and o and h and l and cl and v):
        return None
    m["symbol"] = s
    m["date"] = d
    m["open"] = o
    m["high"] = h
    m["low"] = l
    m["close"] = cl
    m["volume"] = v
    amt = _colmap(columns, _AMOUNT_KEYS)
    if amt:
        m["amount"] = amt
    return m


def _list_user_tables(con: duckdb.DuckDBPyConnection) -> List[Tuple[str, str]]:
    q = """
    select table_schema, table_name
    from information_schema.tables
    where table_schema not in ('pg_catalog', 'information_schema')
    order by 1, 2
    """
    rows = con.execute(q).fetchall()
    return [(str(s), str(t)) for s, t in rows]


def _table_columns(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> List[str]:
    rows = con.execute(f"pragma table_info({schema}.{table})").fetchall()
    # pragma table_info returns: (column_id, name, type, null, default, pk)
    return [str(r[1]) for r in rows]


def _get_symbols(con: duckdb.DuckDBPyConnection, sch: str, tbl: str, sym_col: str) -> List[str]:
    rows = con.execute(f"select distinct {sym_col} as s from {sch}.{tbl} order by 1").fetchall()
    return [str(r[0]) for r in rows]


def _export_parquet_for_symbol(
    con: duckdb.DuckDBPyConnection,
    sch: str,
    tbl: str,
    cmap: Dict[str, str],
    symbol: str,
    out_dir: Path,
) -> Path:
    out = out_dir / f"{symbol}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    q = f"""
    copy (
      select
        {cmap['date']}::timestamp as date,
        {cmap['open']} as open,
        {cmap['high']} as high,
        {cmap['low']} as low,
        {cmap['close']} as close,
        {cmap['volume']} as volume,
        1.0 as factor
      from {sch}.{tbl}
      where {cmap['symbol']} = '{symbol.replace("'", "''")}'
      order by {cmap['date']}
    ) to '{out.as_posix()}' (format csv, header true)
    """
    con.execute(q)
    return out


def _run_dump_bin(temp_parquet_dir: Path, qlib_dir: Path) -> None:
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    if qlib_dir.exists():
        shutil.rmtree(qlib_dir)
    qlib_dir.mkdir(parents=True, exist_ok=True)
    vendor = ROOT / "scripts" / "vendor_dump_bin.py"
    argv = [
        "dump_bin.py",
        "dump_all",
        "--data_path",
        str(temp_parquet_dir),
        "--qlib_dir",
        str(qlib_dir),
        "--include_fields",
        "open,close,high,low,volume,factor",
        "--file_suffix",
        ".csv",
        "--date_field_name",
        "date",
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(vendor), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> int:
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        tables = _list_user_tables(con)
        target: Optional[Tuple[str, str]] = None
        cmap: Optional[Dict[str, str]] = None
        for sch, tbl in tables:
            cols = _table_columns(con, sch, tbl)
            m = _detect_ohlcv_columns(cols)
            if m:
                target = (sch, tbl)
                cmap = m
                break
        if not (target and cmap):
            print("no OHLCV-like table detected")
            return 3
        sch, tbl = target
        print(f"table={sch}.{tbl}")
        symbols = _get_symbols(con, sch, tbl, cmap["symbol"])
        lim = os.environ.get("QLIB_BUILD_LIMIT")
        if lim:
            try:
                n = int(lim)
                symbols = symbols[: max(0, n)]
                print(f"limit={n}")
            except Exception:
                pass
        print(f"symbols={len(symbols)}")
        with tempfile.TemporaryDirectory() as td:
            temp_parquet_dir = Path(td) / "parquet"
            temp_parquet_dir.mkdir(parents=True, exist_ok=True)
            for s in symbols:
                _export_parquet_for_symbol(con, sch, tbl, cmap, s, temp_parquet_dir)
            _run_dump_bin(temp_parquet_dir, QLIB_DATA_DIR)
    finally:
        con.close()
    print(str(QLIB_DATA_DIR))
    p = QLIB_DATA_DIR
    exists = p.exists() and any(p.iterdir())
    print(f"built={exists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
