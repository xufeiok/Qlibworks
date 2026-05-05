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
    # 飞牛OS ClickHouse 连接配置
    ch_host: str = "192.168.10.102"
    ch_port: int = 18123
    ch_user: str = "xufei"
    ch_password: str = "xf1987216"
    ch_database: str = "quant_db"
    
    db_path: Path = DUCKDB_PATH  # 保留兼容性
    table: Optional[Tuple[str, str]] = None
    column_map: Optional[Dict[str, str]] = None

    def _connect_clickhouse(self):
        import clickhouse_connect
        return clickhouse_connect.get_client(
            host=self.ch_host,
            port=self.ch_port,
            user=self.ch_user,
            password=self.ch_password,
            database=self.ch_database
        )

    def _auto_detect(self) -> Tuple[Tuple[str, str], Dict[str, str]]:
        # 在 ClickHouse 中自动检测 OHLCV 表
        ch_client = self._connect_clickhouse()
        tables_res = ch_client.query("SHOW TABLES").result_rows
        tables = [row[0] for row in tables_res]
        
        for tbl in tables:
            cols_res = ch_client.query(f"DESCRIBE {tbl}").result_rows
            cols = [row[0] for row in cols_res]
            m = _detect_ohlcv_columns(cols)
            if m:
                return (self.ch_database, tbl), m
        raise RuntimeError("No OHLCV-like table detected in ClickHouse")

    def load(self) -> pd.DataFrame:
        ch_client = self._connect_clickhouse()
        
        table = self.table
        cmap = self.column_map
        if table is None or cmap is None:
            table, cmap = self._auto_detect()
            self.table = table
            self.column_map = cmap
            
        sch, tbl = table
        cols = list(cmap.values())
        
        # 1. 从 ClickHouse 取【中间结果集】获取 Arrow 格式（超快）
        query = f"SELECT {', '.join(cols)} FROM {tbl}"
        arrow_table = ch_client.query_arrow(query)
        
        # 2. 直接注入 DuckDB（零拷贝、瞬间完成）
        con = duckdb.connect()
        con.register("kline_data", arrow_table)
        
        # 3. 从 DuckDB 中查询并返回 DataFrame
        q = f"SELECT * FROM kline_data"
        df = con.execute(q).df()
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
