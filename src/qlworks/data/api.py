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
import io
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from datetime import datetime, timedelta

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
    DUCKDB_PATH, QLIB_DATA_DIR,
    FORCE_ADJUSTED_PRICES, FINANCIAL_USE_ANNOUNCEMENT_DATE, ADJUSTED_PRICE_TYPE
)

# Tushare Token（必须从环境变量获取，不可硬编码）
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    import warnings
    warnings.warn("TUSHARE_TOKEN 未设置，Tushare 后备数据源不可用。请在 .env 文件中配置 TUSHARE_TOKEN")


class QuantDataAPI:
    """
    统一数据访问接口 - 整个项目的数据核心枢纽
    
    架构原则：
    1. ClickHouse 是唯一真实数据源 (SSOT)
    2. DuckDB 作为智能缓存层
    3. Qlib/Parquet 作为专用格式导出
    4. 自动化缓存管理和一致性检查
    """
    
    def __init__(self, cache_ttl: int = 86400):
        """
        初始化 QuantDataAPI
        
        Args:
            cache_ttl: 缓存过期时间（秒），默认 24 小时
        """
        self.cache_ttl = cache_ttl
        self._ch_client = None
        self._duckdb_conn = None
        self._tushare_pro = None  # Tushare Pro 客户端
        self._init_duckdb()
    
    def _init_duckdb(self):
        """初始化 DuckDB 连接和系统表"""
        self._duckdb_conn = duckdb.connect(str(DUCKDB_PATH))
        
        # 创建查询缓存表
        self._duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash VARCHAR PRIMARY KEY,
                data BLOB,
                created_at TIMESTAMP,
                sql_text VARCHAR
            )
        """)
        
        # 创建特征元数据表
        self._duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR,
                version VARCHAR,
                description VARCHAR,
                created_at TIMESTAMP,
                start_date DATE,
                end_date DATE,
                num_stocks INTEGER,
                columns VARCHAR,
                file_path VARCHAR,
                PRIMARY KEY (feature_name, version)
            )
        """)
        
        # 创建数据同步日志表
        self._duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_log (
                sync_id INTEGER PRIMARY KEY,
                sync_type VARCHAR,
                target_table VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                rows_synced INTEGER,
                status VARCHAR,
                error_message VARCHAR
            )
        """)
    
    def _get_ch_client(self):
        """获取 ClickHouse 客户端（单例模式）"""
        if self._ch_client is None:
            self._ch_client = clickhouse_connect.get_client(
                host=CH_HOST,
                port=CH_PORT,
                user=CH_USER,
                password=CH_PASSWORD,
                database=CH_DATABASE
            )
        return self._ch_client
    
    def _close_connections(self):
        """关闭所有连接"""
        if self._ch_client:
            self._ch_client.close()
            self._ch_client = None
        if self._duckdb_conn:
            self._duckdb_conn.close()
            self._duckdb_conn = None
    
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
    
    def query(self, sql: str, use_cache: bool = True, params: Optional[List] = None) -> pd.DataFrame:
        """
        执行 SQL 查询，自动使用 DuckDB 缓存
        
        Args:
            sql: SQL 查询语句
            use_cache: 是否使用缓存
            params: SQL 参数列表
            
        Returns:
            查询结果的 DataFrame
        """
        # 转换占位符格式
        sql, params = self._convert_placeholders(sql, params)

        if not use_cache:
            return self._get_ch_client().query_df(sql, params)

        # 生成查询哈希
        query_str = sql + str(params) if params else sql
        query_hash = hashlib.md5(query_str.encode()).hexdigest()

        # 从 ClickHouse 查询（暂时禁用缓存，避免 DuckDB 二进制数据存储问题）
        df = self._get_ch_client().query_df(sql, params)

        return df

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
    
    def clear_cache(self, ttl: Optional[int] = None):
        """
        清理过期缓存

        Args:
            ttl: 自定义 TTL，不传则使用默认值
        """
        ttl = ttl or self.cache_ttl
        result = self._duckdb_conn.execute(f"""
            DELETE FROM query_cache WHERE created_at < NOW() - INTERVAL '{ttl}' SECOND
        """)
        print(f"清理了 {result.rowcount} 条缓存记录")
    
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
                    p.open * COALESCE(a.adj_factor, 1) / {adj_divisor} AS open,
                    p.high * COALESCE(a.adj_factor, 1) / {adj_divisor} AS high,
                    p.low * COALESCE(a.adj_factor, 1) / {adj_divisor} AS low,
                    p.close * COALESCE(a.adj_factor, 1) / {adj_divisor} AS close,
                    p.vol AS vol,
                    p.close * COALESCE(a.adj_factor, 1) / {adj_divisor} * p.vol AS amount,
                    i.pe AS pe, i.pe_ttm AS pe_ttm, i.pb AS pb, i.ps AS ps, i.ps_ttm AS ps_ttm,
                    i.total_mv AS total_mv, i.circ_mv AS circ_mv, i.dv_ttm AS dv_ttm,
                    a.adj_factor AS adj_factor,
                    latest.adj_factor AS latest_adj_factor
                FROM daily_prices p
                LEFT JOIN daily_indicators i ON p.ts_code = i.ts_code AND p.trade_date = i.trade_date
                LEFT JOIN daily_adj_factors a ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
                LEFT JOIN (
                    SELECT ts_code, adj_factor
                    FROM daily_adj_factors
                    WHERE trade_date = (SELECT MAX(trade_date) FROM daily_adj_factors WHERE ts_code = daily_adj_factors.ts_code)
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
        
        # 1. 首先从 ClickHouse 查询
        df_ch = self.query(sql, params=[], use_cache=False)
        
        # 2. 如果 ClickHouse 数据为空，尝试从 Tushare 获取
        if df_ch.empty and _TUSHARE_AVAILABLE and ts_codes:
            print(f"⚠️ ClickHouse 数据为空，尝试从 Tushare 获取前复权数据...")
            
            # 标准化日期格式
            start_dt = start_date if start_date else '2000-01-01'
            end_dt = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
            
            # 从 Tushare 获取每只股票的数据
            tushare_dfs = []
            for ts_code in ts_codes:
                df_tu = self._fetch_from_tushare(ts_code, start_dt, end_dt)
                if not df_tu.empty:
                    tushare_dfs.append(df_tu)
            
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
        sql = "SELECT ts_code, symbol, name, industry, area, market, list_date, list_status FROM stock_universe WHERE 1=1"

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
        df = self.query(sql, params=[])
        
        if use_ann and 'end_date' in df.columns and 'ann_date' in df.columns:
            df = df.drop(columns=['end_date'])
        
        return df
    
    # ==================== Qlib 数据同步 ====================
    
    def sync_qlib_full(self, start_date: str, end_date: str, instruments: Optional[List[str]] = None):
        """
        全量同步 Qlib 数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            instruments: 股票列表，None 表示全部主板股票
        """
        from qlworks.data.qlib_sync import QlibSynchronizer
        
        syncer = QlibSynchronizer(self)
        syncer.full_sync(start_date, end_date, instruments)
    
    def sync_qlib_incremental(self):
        """增量同步 Qlib 数据"""
        from qlworks.data.qlib_sync import QlibSynchronizer
        
        syncer = QlibSynchronizer(self)
        syncer.incremental_sync()
    
    # ==================== Parquet 特征管理 ====================
    
    def save_feature(
        self,
        df: pd.DataFrame,
        name: str,
        version: str,
        description: str = "",
        category: str = "experimental",
        date_column: str = "trade_date"
    ) -> Path:
        """
        保存特征数据，自动记录元数据
        
        Args:
            df: 特征数据 DataFrame
            name: 特征名称
            version: 版本号
            description: 描述
            category: 分类 (fundamental/technical/alternative/experimental)
            date_column: 日期列名，默认 trade_date，财务数据应传 ann_date
            
        Returns:
            保存的文件路径
        """
        feature_dir = QLIB_DATA_DIR / "features" / category
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = f"{name}_v{version}.parquet"
        file_path = feature_dir / file_name
        
        # 保存数据
        df.to_parquet(file_path, index=False)
        
        # 确定日期列（兼容不同数据格式）
        date_col = date_column if date_column in df.columns else None
        if date_col is None:
            for col in ['ann_date', 'trade_date', 'date']:
                if col in df.columns:
                    date_col = col
                    break
        
        # 保存元数据
        self._duckdb_conn.execute("""
            INSERT OR REPLACE INTO feature_metadata
            (feature_name, version, description, created_at, start_date, end_date, num_stocks, columns, file_path)
            VALUES (?, ?, ?, NOW(), ?, ?, ?, ?, ?)
        """, [
            name, version, description,
            df[date_col].min() if date_col and date_col in df.columns else None,
            df[date_col].max() if date_col and date_col in df.columns else None,
            df["ts_code"].nunique() if "ts_code" in df.columns else 0,
            ",".join(df.columns),
            str(file_path)
        ])
        
        return file_path
    
    def load_feature(self, name: str, version: Optional[str] = None) -> pd.DataFrame:
        """
        加载特征数据，自动获取最新版本
        
        Args:
            name: 特征名称
            version: 版本号，None 表示最新版本
            
        Returns:
            特征数据 DataFrame
        """
        if version is None:
            version_result = self._duckdb_conn.execute("""
                SELECT version FROM feature_metadata 
                WHERE feature_name = ? ORDER BY created_at DESC LIMIT 1
            """, [name]).fetchone()
            if not version_result:
                raise FileNotFoundError(f"未找到特征：{name}")
            version = version_result[0]
        
        file_path_result = self._duckdb_conn.execute("""
            SELECT file_path FROM feature_metadata 
            WHERE feature_name = ? AND version = ?
        """, [name, version]).fetchone()
        
        if not file_path_result:
            raise FileNotFoundError(f"未找到特征：{name} v{version}")
        
        return pd.read_parquet(file_path_result[0])
    
    def list_features(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        列出所有特征
        
        Args:
            category: 分类过滤
            
        Returns:
            特征元数据 DataFrame
        """
        if category:
            return self._duckdb_conn.execute("""
                SELECT * FROM feature_metadata 
                WHERE file_path LIKE ?
                ORDER BY feature_name, created_at DESC
            """, [f"%/{category}/%"]).df()
        return self._duckdb_conn.execute("SELECT * FROM feature_metadata ORDER BY feature_name, created_at DESC").df()
    
    # ==================== 数据一致性检查 ====================
    
    def check_consistency(self) -> Dict[str, bool]:
        """
        检查数据一致性
        
        Returns:
            一致性检查结果字典
        """
        results = {}
        
        # 检查 Qlib 与 ClickHouse 一致性
        try:
            qlib_init = False
            try:
                import qlib
                qlib.init(provider_uri=str(QLIB_DATA_DIR))
                qlib_init = True
            except:
                pass
            
            if qlib_init:
                from qlib.data import D
                qlib_latest = D.calendar()[-1]
                ch_latest = self.query("SELECT MAX(trade_date) FROM daily_prices").iloc[0, 0]
                
                if qlib_latest != ch_latest:
                    results["qlib_sync"] = False
                    print(f"⚠️ Qlib 数据不一致：Qlib 最新{qlib_latest}，ClickHouse 最新{ch_latest}")
                else:
                    results["qlib_sync"] = True
                    print("✅ Qlib 数据一致")
            else:
                results["qlib_sync"] = None
                print("⚠️ Qlib 未初始化，跳过检查")
        except Exception as e:
            results["qlib_sync"] = False
            print(f"⚠️ Qlib 一致性检查失败：{e}")
        
        # 检查特征文件完整性
        missing_result = self._duckdb_conn.execute("""
            SELECT file_path FROM feature_metadata 
            WHERE file_path IS NOT NULL
        """).df()
        
        missing_files = []
        for _, row in missing_result.iterrows():
            if not Path(row["file_path"]).exists():
                missing_files.append(row["file_path"])
        
        if missing_files:
            results["feature_files"] = False
            print(f"⚠️ 缺失特征文件：{missing_files}")
        else:
            results["feature_files"] = True
            print("✅ 特征文件完整")
        
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_connections()
