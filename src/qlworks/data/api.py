"""
统一数据访问 API - QuantDataAPI

功能概述：
- 作为整个项目的唯一数据访问入口
- 封装 ClickHouse 查询、DuckDB 缓存、Qlib 数据同步、Parquet 特征管理
- 遵循 SSOT（唯一真实数据源）原则：ClickHouse 是唯一数据源
- 自动化缓存管理和数据一致性检查
- 支持 Tushare 作为后备数据源，当 ClickHouse 数据缺失时自动补充
"""
from __future__ import annotations

import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import threading
import atexit
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import duckdb
import clickhouse_connect

# Tushare 作为后备数据源
try:
    import tushare as ts
    _TUSHARE_AVAILABLE = True
except ImportError:
    _TUSHARE_AVAILABLE = False

from qlworks.config import (
    CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE,
    DUCKDB_PATH, QLIB_DATA_DIR, FS_CACHE_DIR,
    FORCE_ADJUSTED_PRICES, FINANCIAL_USE_ANNOUNCEMENT_DATE, ADJUSTED_PRICE_TYPE
)

# Tushare Token（必须从环境变量获取，不可硬编码）
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    import warnings
    warnings.warn("TUSHARE_TOKEN 未设置，Tushare 后备数据源不可用。请在 .env 文件中配置 TUSHARE_TOKEN")

# ==================== 现代统一数据 API ====================

class QuantDataAPI:
    """
    统一数据访问接口 - 整个项目的数据核心枢纽

    架构原则：
    1. ClickHouse 是唯一真实数据源 (SSOT)
    2. DuckDB 作为查询缓存层（针对不变历史数据，避免重复查询 ClickHouse）
    3. Tushare 作为后备数据源
    """

    # 类级别共享 DuckDB 连接（避免多实例创建多个连接、进程崩溃后残留锁文件）
    _shared_duckdb_conn = None
    _duckdb_lock = threading.Lock()

    @classmethod
    def _close_shared_duckdb(cls):
        """进程退出时释放共享 DuckDB 连接。"""
        conn = cls._shared_duckdb_conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            cls._shared_duckdb_conn = None

    def __init__(self):
        """
        初始化 QuantDataAPI
        """
        self._ch_client = None
        self._init_duckdb()
        self._tushare_pro = None  # Tushare Pro 客户端
        # Tushare 限速器（实例级别，避免每次调用 get_daily_data 重建）
        self._ts_rate_limit_sem = threading.Semaphore(4)  # 最多 4 个并发
        self._ts_last_requests: deque = deque(maxlen=80)
        self._ts_executor = ThreadPoolExecutor(max_workers=4)
        # Parquet 查询缓存目录
        self._query_cache_dir = Path(str(FS_CACHE_DIR)) / "query_cache"
        self._query_cache_dir.mkdir(parents=True, exist_ok=True)
        self.clear_query_cache(max_age_sec=604800)

    def _init_duckdb(self):
        """初始化 DuckDB 连接（单例模式，所有实例共享同一连接）"""
        DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
        if QuantDataAPI._shared_duckdb_conn is None:
            with QuantDataAPI._duckdb_lock:
                if QuantDataAPI._shared_duckdb_conn is None:
                    try:
                        QuantDataAPI._shared_duckdb_conn = duckdb.connect(str(DUCKDB_PATH))
                    except duckdb.IOException as e:
                        if "Cannot open file" in str(e) or "already open" in str(e):
                            logger.warning(f"DuckDB 文件被锁（{e}），尝试只读模式...")
                            QuantDataAPI._shared_duckdb_conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
                        else:
                            raise
                    atexit.register(QuantDataAPI._close_shared_duckdb)
        self._duckdb_conn = QuantDataAPI._shared_duckdb_conn
        # 注意：查询缓存已改用 Parquet 文件存储（query_cached 方法），
        # DuckDB 保留用于后续元数据管理功能。
    
    def _get_ch_client(self):
        """获取 ClickHouse 客户端（单例模式，带指数退避重试）"""
        if self._ch_client is not None:
            return self._ch_client

        last_exc = None
        for attempt in range(3):
            try:
                self._ch_client = clickhouse_connect.get_client(
                    host=CH_HOST,
                    port=CH_PORT,
                    user=CH_USER,
                    password=CH_PASSWORD,
                    database=CH_DATABASE,
                    connect_timeout=15,
                )
                return self._ch_client
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"[重试] ClickHouse 连接失败 (第{attempt+1}次)，{wait}s 后重试: {e}")
                    time.sleep(wait)
        raise ConnectionError(f"ClickHouse 连接失败，已重试 3 次: {last_exc}")

    # ==================== Tushare 后备数据源 ====================
    
    def _get_tushare_pro(self):
        """获取 Tushare Pro 客户端（单例模式）"""
        if not _TUSHARE_AVAILABLE:
            return None
        
        if self._tushare_pro is None:
            try:
                self._tushare_pro = ts.pro_api(TUSHARE_TOKEN)
            except Exception as e:
                print(f"❌ Tushare 初始化失败: {e}")
                return None
        return self._tushare_pro
    
    def _format_tushare_date(self, date_str: str) -> str:
        """
        统一日期格式为 Tushare 要求的 YYYYMMDD
        
        Args:
            date_str: 日期字符串，支持 YYYY-MM-DD 或 YYYYMMDD 格式
            
        Returns:
            格式化后的日期字符串 YYYYMMDD，失败返回空字符串
        """
        if not date_str:
            return ''
        
        # 已经是 YYYYMMDD 格式
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
        
        # YYYY-MM-DD 格式
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str.replace('-', '')
        
        # 尝试使用 pandas 解析
        try:
            return pd.to_datetime(date_str).strftime('%Y%m%d')
        except Exception:
            return ''
    
    def _fetch_from_tushare(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Tushare 获取日线数据（作为 ClickHouse 数据缺失时的后备）
        返回的价格数据已经处理为前复权格式，与 ClickHouse 数据保持一致
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期（支持 YYYY-MM-DD 或 YYYYMMDD）
            end_date: 结束日期（支持 YYYY-MM-DD 或 YYYYMMDD）
            
        Returns:
            日线数据 DataFrame（前复权），如果获取失败返回空 DataFrame
        """
        pro = self._get_tushare_pro()
        if pro is None:
            return pd.DataFrame()
        
        try:
            # 统一日期格式为 Tushare 要求的 YYYYMMDD
            start_str = self._format_tushare_date(start_date)
            end_str = self._format_tushare_date(end_date)
            
            if not start_str or not end_str:
                print(f"⚠️ 日期格式错误 ({ts_code}): {start_date} - {end_date}")
                return pd.DataFrame()
            
            # 1. 获取日线数据
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_str,
                end_date=end_str
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 2. 获取复权因子（用于计算前复权价格）
            adj_df = pro.adj_factor(
                ts_code=ts_code,
                start_date=start_str,
                end_date=end_str
            )
            
            if adj_df.empty:
                # 如果没有复权因子，直接返回原始数据（会有警告）
                print(f"⚠️ Tushare 未获取到复权因子 ({ts_code})，返回原始价格")
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
                df = df.sort_values('trade_date').reset_index(drop=True)
                return df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
            
            # 3. 合并日线数据和复权因子
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'], format='%Y%m%d')
            
            df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
            
            # 4. 高效获取最新复权因子（只获取最近30天数据，而非全部历史）
            current_latest_factor = df['adj_factor'].iloc[-1] if not df['adj_factor'].empty else 1.0
            latest_adj_factor = current_latest_factor
            
            try:
                # 只获取最近30天的复权因子来获取最新值，提高效率
                from datetime import datetime, timedelta
                recent_end = datetime.now().strftime('%Y%m%d')
                recent_start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                
                recent_adj_df = pro.adj_factor(
                    ts_code=ts_code,
                    start_date=recent_start,
                    end_date=recent_end
                )
                
                if not recent_adj_df.empty:
                    recent_adj_df['trade_date'] = pd.to_datetime(recent_adj_df['trade_date'], format='%Y%m%d')
                    recent_adj_df = recent_adj_df.sort_values('trade_date')
                    latest_adj_factor = recent_adj_df['adj_factor'].iloc[-1]
                    
            except Exception as e:
                # 如果获取最近数据失败，回退到查询全部历史
                try:
                    all_adj_df = pro.adj_factor(ts_code=ts_code)
                    if not all_adj_df.empty:
                        all_adj_df['trade_date'] = pd.to_datetime(all_adj_df['trade_date'], format='%Y%m%d')
                        all_adj_df = all_adj_df.sort_values('trade_date')
                        latest_adj_factor = all_adj_df['adj_factor'].iloc[-1]
                except Exception as fallback_e:
                    print(f"⚠️ 获取最新复权因子失败 ({ts_code}): {fallback_e}")
            
            # 5. 计算前复权价格
            # 前复权公式: 前复权价格 = 原始价格 × 当日复权因子 / 最新复权因子
            if latest_adj_factor > 0:
                df['open'] = df['open'] * df['adj_factor'] / latest_adj_factor
                df['high'] = df['high'] * df['adj_factor'] / latest_adj_factor
                df['low'] = df['low'] * df['adj_factor'] / latest_adj_factor
                df['close'] = df['close'] * df['adj_factor'] / latest_adj_factor
                # 复权后的成交额 = 复权收盘价 × 成交量
                df['amount'] = df['close'] * df['vol']
            
            # 6. 格式化日期（保持与 ClickHouse 一致的 datetime64 类型）
            df['trade_date'] = df['trade_date'].dt.normalize()
            
            # 7. 按日期排序
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            print(f"✅ Tushare 获取并处理前复权数据 ({ts_code}): {len(df)} 条")
            
            # 确保数值类型一致（转换为 float64，与 ClickHouse 数据保持一致）
            numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'adj_factor']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype('float64')
            
            return df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'adj_factor']]
        
        except Exception as e:
            print(f"⚠️ Tushare 获取数据失败 ({ts_code}): {e}")
            return pd.DataFrame()
    
    # ==================== ClickHouse 查询与缓存 ====================

    # ── Parquet 文件缓存（替代旧的 DuckDB BLOB 缓存） ──
    # 历史数据天然不可变（只读账号），缓存永不过期。
    # 文件存储在 .fs_cache/query_cache/{hash}.parquet，zstd 压缩。
    # 每次初始化自动清理超过 7 天的缓存文件。

    @staticmethod
    def _query_cache_key(sql: str, params: Optional[list] = None) -> str:
        raw = sql + (str(params) if params else '')
        return hashlib.md5(raw.encode()).hexdigest()[:24]

    def _query_cache_get(self, key: str, max_age_sec: int = 86400):
        cache_file = self._query_cache_dir / f"{key}.parquet"
        if not cache_file.exists():
            return None
        age = time.time() - cache_file.stat().st_mtime
        if age > max_age_sec:
            cache_file.unlink(missing_ok=True)
            return None
        return pd.read_parquet(cache_file)

    def _query_cache_set(self, key: str, df: pd.DataFrame) -> None:
        cache_file = self._query_cache_dir / f"{key}.parquet"
        df.to_parquet(cache_file, index=False, compression='zstd')

    def query_cached(self, sql: str, ttl: int = 86400) -> pd.DataFrame:
        """
        带 Parquet 缓存的查询：先查缓存，命中直接返回；未命中查 ClickHouse 后缓存。
        """
        key = self._query_cache_key(sql)
        cached = self._query_cache_get(key, max_age_sec=ttl)
        if cached is not None:
            return cached
        df = self.query(sql)
        if not df.empty:
            self._query_cache_set(key, df)
        return df

    def clear_query_cache(self, max_age_sec: Optional[int] = None) -> int:
        """清理查询缓存。max_age_sec=None 全部删除，否则删除超过指定秒数的文件。"""
        if not self._query_cache_dir.exists():
            return 0
        removed = 0
        now = time.time()
        for f in self._query_cache_dir.iterdir():
            if not f.name.endswith('.parquet'):
                continue
            if max_age_sec is None or (now - f.stat().st_mtime) > max_age_sec:
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            print(f"[缓存] 清理 {removed} 个过期查询缓存文件")
        return removed

    def query(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
        """
        执行 SQL 查询（直连 ClickHouse）。

        Args:
            sql: SQL 查询语句
            params: SQL 参数列表

        Returns:
            查询结果的 DataFrame
        """
        sql, params = self._convert_placeholders(sql, params)
        client = self._get_ch_client()
        try:
            return client.query_df(sql, params, use_numpy=True)
        except TypeError:
            return client.query_df(sql, params)

    def _convert_placeholders(self, sql: str, params: list) -> tuple:
        """将 ? 占位符转换为 ClickHouse 的 $n 格式"""
        if not params:
            return sql, params
        param_idx = 0
        result = []
        for i, char in enumerate(sql):
            if char == '?':
                param_idx += 1
                result.append(f'${param_idx}')
            else:
                result.append(char)
        new_sql = ''.join(result)
        return new_sql, params
    
    # ==================== 基础数据查询 ====================
    
    def get_daily_data(
        self,
        ts_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        adj: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        获取日线数据（默认强制前复权）
        
        前复权价格计算公式（Tushare 标准）：
            前复权价格 = (当日收盘价 × 当日复权因子) / 最新复权因子
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期 YYYYMMDD 或 YYYY-MM-DD
            end_date: 结束日期 YYYYMMDD 或 YYYY-MM-DD
            fields: 字段列表，None 表示全部
            adj: 是否使用前复权，默认由 FORCE_ADJUSTED_PRICES 配置决定
            
        Returns:
            日线数据 DataFrame（包含前复权价格和指标）
        """
        if FORCE_ADJUSTED_PRICES and adj is False:
            import warnings
            warnings.warn(
                "FORCE_ADJUSTED_PRICES=True，已强制使用前复权价格，忽略 adj=False",
                UserWarning
            )
        use_adj = FORCE_ADJUSTED_PRICES if adj is None else adj

        if use_adj:
            adj_divisor = "COALESCE(NULLIF(latest.adj_factor, 0), 1)"
            sql = f"""
                SELECT
                    p.ts_code AS ts_code, p.trade_date AS trade_date,
                    p.open * COALESCE(NULLIF(a.adj_factor, 0), 1) / {adj_divisor} AS open,
                    p.high * COALESCE(NULLIF(a.adj_factor, 0), 1) / {adj_divisor} AS high,
                    p.low * COALESCE(NULLIF(a.adj_factor, 0), 1) / {adj_divisor} AS low,
                    p.close * COALESCE(NULLIF(a.adj_factor, 0), 1) / {adj_divisor} AS close,
                    p.vol AS vol,
                    p.close * COALESCE(NULLIF(a.adj_factor, 0), 1) / {adj_divisor} * p.vol AS amount,
                    i.pe AS pe, i.pe_ttm AS pe_ttm, i.pb AS pb, i.ps AS ps, i.ps_ttm AS ps_ttm,
                    i.total_mv AS total_mv, i.circ_mv AS circ_mv, i.dv_ttm AS dv_ttm,
                    a.adj_factor AS adj_factor,
                    latest.adj_factor AS latest_adj_factor
                FROM daily_prices p
                LEFT JOIN daily_indicators i ON p.ts_code = i.ts_code AND p.trade_date = i.trade_date
                LEFT JOIN daily_adj_factors a ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
                LEFT JOIN (
                    SELECT ts_code, argMax(adj_factor, trade_date) AS adj_factor
                    FROM daily_adj_factors
                    GROUP BY ts_code
                ) latest ON p.ts_code = latest.ts_code
            """
        else:
            sql = f"""
                SELECT
                    p.ts_code AS ts_code, p.trade_date AS trade_date,
                    p.open AS open, p.high AS high, p.low AS low, p.close AS close, p.vol AS vol, p.amount AS amount,
                    i.pe AS pe, i.pe_ttm AS pe_ttm, i.pb AS pb, i.ps AS ps, i.ps_ttm AS ps_ttm,
                    i.total_mv AS total_mv, i.circ_mv AS circ_mv, i.dv_ttm AS dv_ttm
                FROM daily_prices p
                LEFT JOIN daily_indicators i ON p.ts_code = i.ts_code AND p.trade_date = i.trade_date
            """
        
        sql += " WHERE 1=1"

        if ts_codes:
            ts_list = ", ".join([f"'{c}'" for c in ts_codes])
            sql += f" AND p.ts_code IN ({ts_list})"
        if start_date:
            sql += f" AND p.trade_date >= '{start_date}'"
        if end_date:
            sql += f" AND p.trade_date <= '{end_date}'"

        sql += " ORDER BY p.ts_code, p.trade_date"
        
        # 1. 首先从 ClickHouse 查询（DuckDB 缓存）
        df_ch = self.query_cached(sql)
        
        # 2. 如果 ClickHouse 数据为空，尝试从 Tushare 获取（带限速并发）
        if df_ch.empty and _TUSHARE_AVAILABLE and ts_codes:
            print(f"⚠️ ClickHouse 数据为空，尝试从 Tushare 获取前复权数据...")

            # 标准化日期格式
            start_dt = start_date if start_date else '2000-01-01'
            end_dt = end_date if end_date else datetime.now().strftime('%Y-%m-%d')

            tushare_dfs = []
            fut_map = {self._ts_executor.submit(self._rate_limited_tushare_fetch, c, start_dt, end_dt): c for c in ts_codes}
            for fut in as_completed(fut_map):
                    try:
                        df_tu = fut.result()
                        if not df_tu.empty:
                            tushare_dfs.append(df_tu)
                    except Exception as e:
                        print(f"⚠️ Tushare 并发获取失败 ({fut_map[fut]}): {e}")

            if tushare_dfs:
                df_ch = pd.concat(tushare_dfs, ignore_index=True)
                
                # 补充缺失的字段，保持与 ClickHouse 数据格式一致
                missing_fields = ['pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 
                                 'total_mv', 'circ_mv', 'dv_ttm', 'latest_adj_factor']
                for field in missing_fields:
                    if field not in df_ch.columns:
                        df_ch[field] = np.nan
                
                # 设置 latest_adj_factor = adj_factor（最后一条记录的复权因子）
                if 'adj_factor' in df_ch.columns and 'latest_adj_factor' in df_ch.columns:
                    latest_adjs = df_ch.groupby('ts_code')['adj_factor'].last().to_dict()
                    df_ch['latest_adj_factor'] = df_ch['ts_code'].map(latest_adjs)
                
                print(f"📊 从 Tushare 获取了 {len(df_ch)} 条前复权数据")
        
        return df_ch
    
    def get_calendar(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日历 DataFrame
        """
        sql = "SELECT DISTINCT trade_date FROM daily_prices"

        if start_date:
            sql += f" WHERE trade_date >= '{start_date}'"
        if end_date:
            if not start_date:
                sql += f" WHERE trade_date <= '{end_date}'"
            else:
                sql += f" AND trade_date <= '{end_date}'"

        sql += " ORDER BY trade_date"
        return self.query(sql, params=[])
    
    def get_stock_list(
        self,
        market: Optional[str] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            market: 市场板块
            status: 上市状态（L 上市/D 退市）

        Returns:
            股票列表 DataFrame
        """
        sql = "SELECT ts_code, symbol, name, industry, area, market, list_date, delist_date, list_status FROM stock_basic WHERE 1=1"

        if market:
            sql += f" AND market = '{market}'"
        if status:
            sql += f" AND list_status = '{status}'"

        return self.query(sql, params=[])
    
    def get_financial_data(
        self,
        ts_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        use_ann_date: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        获取财务数据（财报指标）
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期（公告日期）
            end_date: 结束日期（公告日期）
            fields: 字段列表，None 表示全部
            use_ann_date: 是否使用公告日期，默认强制使用
            
        Returns:
            财务数据 DataFrame，使用 ann_date 作为主日期列
        """
        if FINANCIAL_USE_ANNOUNCEMENT_DATE and use_ann_date is False:
            import warnings
            warnings.warn(
                "FINANCIAL_USE_ANNOUNCEMENT_DATE=True，已强制使用公告日期，忽略 use_ann_date=False",
                UserWarning
            )
        use_ann = FINANCIAL_USE_ANNOUNCEMENT_DATE if use_ann_date is None else use_ann_date
        
        base_fields = [
            'ts_code', 'ann_date', 'end_date',
            'roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
            'debt_to_assets', 'current_ratio', 'eps', 'ocfps',
            'netprofit_yoy', 'dt_netprofit_yoy', 'basic_eps_yoy',
            'tr_yoy', 'stk_holdernumber', 'pledge_ratio', 'eps_forecast'
        ]
        # 确保 ann_date 和 ts_code 总是被包含（用于日期处理和数据合并）
        if fields is None:
            select_fields = base_fields
        else:
            select_fields = ['ts_code', 'ann_date'] + [f for f in fields if f in base_fields and f not in ['ts_code', 'ann_date']]

        sql = f"SELECT {', '.join(select_fields)} FROM financial_indicators WHERE 1=1"

        if ts_codes:
            ts_list = ", ".join([f"'{c}'" for c in ts_codes])
            sql += f" AND ts_code IN ({ts_list})"
        if start_date:
            date_col = 'ann_date' if use_ann else 'end_date'
            sql += f" AND {date_col} >= '{start_date}'"
        if end_date:
            date_col = 'ann_date' if use_ann else 'end_date'
            sql += f" AND {date_col} <= '{end_date}'"

        sql += " ORDER BY ts_code, ann_date" if use_ann else " ORDER BY ts_code, end_date"
        df = self.query_cached(sql)
        
        if use_ann and 'end_date' in df.columns and 'ann_date' in df.columns:
            df = df.drop(columns=['end_date'])
        
        return df

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_connections()

    def _close_connections(self):
        """关闭所有连接（共享 DuckDB 连接由进程退出时自动释放）"""
        if self._ch_client:
            self._ch_client.close()
            self._ch_client = None
        # DuckDB 连接是类级别共享的，不在此关闭
        self._ts_executor.shutdown(wait=False)
