from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from .config import DUCKDB_PATH


_SYMBOL_KEYS = ["symbol", "code", "ticker", "instrument", "secid", "ts_code"]
_DATE_KEYS = ["date", "datetime", "trade_date", "timestamp", "time"]
_OPEN_KEYS = ["open", "open_price"]
_HIGH_KEYS = ["high", "high_price"]
_LOW_KEYS = ["low", "low_price"]
_CLOSE_KEYS = ["close", "close_price", "last"]
_VOLUME_KEYS = ["volume", "vol"]
_AMOUNT_KEYS = ["amount", "turnover", "value"]


def _lower(s: str) -> str:
    return s.lower().strip()


def _colmap(columns: List[str], keys: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in columns}
    for k in keys:
        if k in cols:
            return cols[k]
    return None


def _detect_ohlcv_columns(columns: List[str]) -> Optional[Dict[str, str]]:
    m = {}
    s = _colmap(columns, _SYMBOL_KEYS)
    d = _colmap(columns, _DATE_KEYS)
    o = _colmap(columns, _OPEN_KEYS)
    h = _colmap(columns, _HIGH_KEYS)
    l = _colmap(columns, _LOW_KEYS)
    cl = _colmap(columns, _CLOSE_KEYS)
    v = _colmap(columns, _VOLUME_KEYS)
    mnt = _colmap(columns, _AMOUNT_KEYS)
    if not (s and d and o and h and l and cl):
        return None
    m["symbol"] = s
    m["date"] = d
    m["open"] = o
    m["high"] = h
    m["low"] = l
    m["close"] = cl
    if v:
        m["volume"] = v
    if mnt:
        m["amount"] = mnt
    return m


def _list_user_tables(con: duckdb.DuckDBPyConnection) -> List[Tuple[str, str]]:
    q = """
    select table_schema, table_name
    from information_schema.tables
    where table_schema not in ('pg_catalog', 'information_schema')
    order by 1, 2
    """
    df = con.execute(q).fetch_df()
    out = []
    for _, r in df.iterrows():
        out.append((str(r["table_schema"]), str(r["table_name"])))
    return out


def _table_columns(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> List[str]:
    df = con.execute(f"pragma table_info({schema}.{table})").fetch_df()
    return [str(x) for x in df["name"].tolist()]


@dataclass
class DuckDBOHLCV:
    db_path: Path = DUCKDB_PATH
    table: Optional[Tuple[str, str]] = None
    column_map: Optional[Dict[str, str]] = None

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _auto_detect(self, con: duckdb.DuckDBPyConnection) -> Tuple[Tuple[str, str], Dict[str, str]]:
        tables = _list_user_tables(con)
        for sch, tbl in tables:
            cols = _table_columns(con, sch, tbl)
            m = _detect_ohlcv_columns(cols)
            if m:
                return (sch, tbl), m
        raise RuntimeError("No OHLCV-like table detected")

    def load(self) -> pd.DataFrame:
        con = self._connect()
        try:
            table = self.table
            cmap = self.column_map
            if table is None or cmap is None:
                table, cmap = self._auto_detect(con)
                self.table = table
                self.column_map = cmap
            sch, tbl = table
            cols = list(cmap.values())
            q = f"select {', '.join([f'{sch}.{tbl}.{c}' for c in cols])} from {sch}.{tbl}"
            df = con.execute(q).fetch_df()
        finally:
            con.close()
        ren = {cmap[k]: k for k in cmap}
        df = df.rename(columns=ren)
        if "amount" not in df.columns:
            df["amount"] = np.nan
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"])
        df = df[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]]
        return df

    def export_per_symbol_csv(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.load()
        for sym, g in df.groupby("symbol"):
            path = out_dir / f"{sym}.csv"
            g[["date", "open", "high", "low", "close", "volume", "amount"]].to_csv(path, index=False)
        return out_dir
