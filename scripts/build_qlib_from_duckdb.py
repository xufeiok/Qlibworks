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
        {cmap['amount']} as amount,
        {cmap['vwap']} as vwap,
        {cmap['pe']} as pe,
        {cmap['pe_ttm']} as pe_ttm,
        {cmap['pb']} as pb,
        {cmap['ps']} as ps,
        {cmap['ps_ttm']} as ps_ttm,
        {cmap['total_mv']} as total_mv,
        {cmap['circ_mv']} as circ_mv,
        {cmap['turnover_rate']} as turnover_rate,
        {cmap['roe_ttm']} as roe_ttm,
        {cmap['roa']} as roa,
        {cmap['grossprofit_margin']} as grossprofit_margin,
        {cmap['netprofit_margin']} as netprofit_margin,
        {cmap['netprofit_yoy']} as netprofit_yoy,
        {cmap['tr_yoy']} as tr_yoy,
        {cmap['basic_eps_yoy']} as basic_eps_yoy,
        {cmap['debt_to_assets']} as debt_to_assets,
        {cmap['current_ratio']} as current_ratio,
        {cmap['inv_turn']} as inv_turn,
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
        "open,close,high,low,volume,amount,vwap,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate,roe_ttm,roa,grossprofit_margin,netprofit_margin,netprofit_yoy,tr_yoy,basic_eps_yoy,debt_to_assets,current_ratio,inv_turn,factor",
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
    import clickhouse_connect
    try:
        ch_client = clickhouse_connect.get_client(
            host="192.168.10.102",
            port=18123,
            user="xufei",
            password="xf1987216",
            database="quant_db"
        )
    except Exception as e:
        print(f"Failed to connect to ClickHouse: {e}")
        return 1

    con = duckdb.connect()
    try:
        # [AQR 改进] 将整个 daily_prices 表与 stock_universe 在 ClickHouse 中直接过滤关联
        # 返回满足条件的 Arrow 数据并注册给 DuckDB，这样就不用通过本地 DuckDB 连接了
        print("Fetching data from ClickHouse...")
        query_all = """
            SELECT 
                dp.ts_code as symbol,
                dp.trade_date as date,
                dp.open as open,
                dp.high as high,
                dp.low as low,
                dp.close as close,
                dp.vol as volume,
                dp.amount as amount,
                (dp.amount * 10) / NULLIF(dp.vol, 0) as vwap,
                di.pe as pe,
                di.pe_ttm as pe_ttm,
                di.pb as pb,
                di.ps as ps,
                di.ps_ttm as ps_ttm,
                di.total_mv as total_mv,
                di.circ_mv as circ_mv,
                di.turnover_rate as turnover_rate,
                fi.roe as roe_ttm,
                fi.roa as roa,
                fi.grossprofit_margin as grossprofit_margin,
                fi.netprofit_margin as netprofit_margin,
                fi.netprofit_yoy as netprofit_yoy,
                fi.tr_yoy as tr_yoy,
                fi.basic_eps_yoy as basic_eps_yoy,
                fi.debt_to_assets as debt_to_assets,
                fi.current_ratio as current_ratio,
                fi.inv_turn as inv_turn
            FROM daily_prices dp
            JOIN stock_universe su ON dp.ts_code = su.ts_code
            LEFT JOIN daily_indicators di ON dp.ts_code = di.ts_code AND dp.trade_date = di.trade_date
            ASOF LEFT JOIN financial_indicators fi ON dp.ts_code = fi.ts_code AND dp.trade_date >= fi.ann_date
            WHERE su.market = '主板'
            ORDER BY dp.ts_code, dp.trade_date
        """
        
        arrow_table = ch_client.query_arrow(query_all)
        con.register("kline_data", arrow_table)
        
        target = ("main", "kline_data")
        cmap = {
            "symbol": "symbol",
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
            "vwap": "vwap",
            "pe": "pe",
            "pe_ttm": "pe_ttm",
            "pb": "pb",
            "ps": "ps",
            "ps_ttm": "ps_ttm",
            "total_mv": "total_mv",
            "circ_mv": "circ_mv",
            "turnover_rate": "turnover_rate",
            "roe_ttm": "roe_ttm",
            "roa": "roa",
            "grossprofit_margin": "grossprofit_margin",
            "netprofit_margin": "netprofit_margin",
            "netprofit_yoy": "netprofit_yoy",
            "tr_yoy": "tr_yoy",
            "basic_eps_yoy": "basic_eps_yoy",
            "debt_to_assets": "debt_to_assets",
            "current_ratio": "current_ratio",
            "inv_turn": "inv_turn"
        }
        
        sch, tbl = target
        print(f"Virtual table={sch}.{tbl}")
        
        # 获取符合条件的 symbols
        query_symbols = f"SELECT DISTINCT symbol FROM {tbl}"
        rows = con.execute(query_symbols).fetchall()
        symbols = [str(r[0]) for r in rows]
        
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
