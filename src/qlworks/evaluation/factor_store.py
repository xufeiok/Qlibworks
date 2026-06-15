"""
因子数据存储层 — 统一数据仓库 + 增量追加

架构：
  factor_data/warehouse/{因子名}/
    ├── 2018.parquet          ← 按年分文件，一次性生成
    ├── 2019.parquet
    ├── ...
    ├── 2026.parquet          ← 每年持续追加
    └── meta.json             ← 最后计算日期、行数等元信息

  factor_data/qualified_factors/{tier}/{年份}/{因子名}.parquet
    ← 软链/引用，由评测结果驱动，不存实际数据

使用方式：
  # 首次批量计算
  store.compute_to_warehouse("KDJ_K", expr, "2018-01-01", "2025-12-31")

  # 每周增量追加
  store.append_to_warehouse("KDJ_K", expr, "2026-01-01")

  # 评测时读取
  df = store.load_from_warehouse("KDJ_K", "2020-01-01", "2025-12-31")
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
import pandas as pd

logger = logging.getLogger(__name__)

_CHUNK_YEARS = 1

# daily_indicators 表的日频字段（与 OHLCV 同频，直接 JOIN 可用）
DAILY_INDICATOR_FIELDS = {
    'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
    'total_mv', 'circ_mv', 'dv_ttm',
    'turnover_rate',
}

# financial_indicators 表的季频字段（稀疏，需向前填充到日频）
# 以下字段经 DESCRIBE financial_indicators 验证，逐个确认存在
FINANCIAL_FIELDS = {
    'roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
    'debt_to_assets', 'current_ratio', 'eps', 'ocfps',
    'netprofit_yoy', 'dt_netprofit_yoy', 'basic_eps_yoy',
    'tr_yoy', 'stk_holdernumber', 'pledge_ratio', 'eps_forecast',
    'inv_turn', 'ar_turn',
}  # 注意：q_profit_yoy/ocf/net_profit/eps_last 在 ClickHouse 中无对应列，会退化到 Qlib

# cashflow_statement + income_statement 补充字段（用于 ocf_to_netprofit 等复合因子）
# 这些字段通过 LEFT JOIN 从其他表获取
EXTRA_FINANCIAL_TABLES = {
    'n_cashflow_act': ('cashflow_statement', 'n_cashflow_act'),
    'n_income_attr_p': ('income_statement', 'n_income_attr_p'),
}

# 综合所有字段来源，用于判断因子类型
ALL_KNOWN_FIELDS = DAILY_INDICATOR_FIELDS | FINANCIAL_FIELDS | set(EXTRA_FINANCIAL_TABLES.keys())


class FactorStore:
    """因子数据仓库：统一存储 + 增量追加 + 快速读取。"""

    def __init__(self, config=None):
        if config is None:
            from .config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self.config = config
        self.warehouse_dir = Path(config.warehouse_dir)
        self.factors_dir = Path(config.factors_dir)
        self.cache_dir = Path(config.cache_dir)
        for d in [self.warehouse_dir, self.factors_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._qlib_inited = False

    # ──────────── 仓库接口（公有） ────────────

    def list_warehouse_factors(self) -> List[str]:
        """列出仓库中所有因子名。"""
        return sorted(d.name for d in self.warehouse_dir.iterdir()
                      if d.is_dir() and (d / "meta.json").exists())

    def get_warehouse_meta(self, name: str) -> Optional[dict]:
        """获取仓库中某个因子的元数据。"""
        p = self._warehouse_meta_path(name)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_warehouse_years(self, name: str) -> List[int]:
        """获取仓库中某个因子已有的年份列表。"""
        d = self._warehouse_dir(name)
        if not d.exists():
            return []
        return sorted(int(f.stem) for f in d.glob("*.parquet") if f.stem.isdigit())

    def inject_warehouse_meta(self, name: str, yaml_meta: dict):
        """
        向已有 warehouse 因子注入 YAML 语义元数据。
        
        不影响数据覆盖范围等自动计算的字段，只补充：version, category, sub_category,
        expression, function_description, theory_background, applicable_conditions,
        reference, lifecycle_stage, meaning, usage_scenario, strategy_hint.

        Args:
            name: 因子名
            yaml_meta: 从 YAML 因子库提取的元数据字典
        """
        meta = self.get_warehouse_meta(name) or {}
        semantic_keys = {
            "version", "category", "sub_category",
            "expression", "function_description", "theory_background",
            "applicable_conditions", "reference", "lifecycle_stage",
            "meaning", "usage_scenario", "strategy_hint",
        }
        for k in semantic_keys:
            if k in yaml_meta and yaml_meta[k]:
                meta[k] = yaml_meta[k]

        # 确保 data_version
        if "data_version" not in meta:
            meta["data_version"] = "3.0"

        p = self._warehouse_meta_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[仓库] 已注入 YAML 元数据到 {name}")

    def compute_to_warehouse(self, name: str, expr: str,
                               start_date: str, end_date: str,
                               stocks: Optional[List[str]] = None,
                               overwrite: bool = False,
                               duckdb_expr: Optional[str] = None) -> Dict[int, int]:
        """
        计算因子全量数据并写入仓库（按年分文件）。
        首次批量生成用这个方法。之后增量用 append_to_warehouse().

        Args:
            name: 因子名
            expr: Qlib 表达式
            start_date/end_date: 日期范围
            stocks: 可选股票代码列表
            overwrite: 是否覆盖已有年份
            duckdb_expr: DuckDB 表达式（YAML 中定义的 duckdb 字段），优先使用

        Returns:
            {年份: 行数} 字典
        """
        existing_years = set(self.get_warehouse_years(name))
        target_years = set(range(int(start_date[:4]), int(end_date[:4]) + 1))

        years_needed = target_years - existing_years if not overwrite else target_years
        if not years_needed:
            logger.info(f"[仓库] {name} 所有年份已存在，跳过。需覆盖请用 overwrite=True")
            return {}

        # 分批按年计算
        stats = {}
        for year in sorted(years_needed):
            ys = f"{year}-01-01"
            ye = f"{year}-12-31"
            logger.info(f"[仓库] 计算 {name} {year}...")
            df = self._compute(name, expr, ys, ye, stocks, duckdb_expr=duckdb_expr)
            if df is None or df.empty:
                logger.warning(f"[仓库] {name} {year} 无数据")
                continue
            self._save_warehouse_year(name, year, df)
            stats[year] = len(df)
            logger.info(f"[仓库] {name} {year}: {len(df)} 行")

        # 更新 meta
        self._update_warehouse_meta(name)
        return stats

    def batch_compute(self, factors: List[Tuple[str, str]],
                       start_date: str, end_date: str,
                       stocks: Optional[List[str]] = None,
                       overwrite: bool = False) -> Dict[str, Dict[int, int]]:
        """
        批量计算多个价格类因子（共享同一份 OHLCV 数据），大幅减少 ClickHouse 查询次数。

        Args:
            factors: [(因子名, duckdb_表达式), ...] 列表
            start_date/end_date: 日期范围
            stocks: 可选股票列表
            overwrite: 是否覆盖已有年份

        Returns:
            {因子名: {年份: 行数}} 字典
        """
        if not factors:
            return {}

        existing_all = True
        to_compute = []
        for name, ds in factors:
            existing_years = set(self.get_warehouse_years(name))
            target_years = set(range(int(start_date[:4]), int(end_date[:4]) + 1))
            if overwrite or not target_years.issubset(existing_years):
                to_compute.append((name, ds))
                existing_all = False

        if existing_all and not overwrite:
            logger.info("[批量] 所有因子已存在，跳过")
            return {}

        try:
            import duckdb
            from qlworks.data import QuantDataAPI
        except ImportError:
            return {}

        api = QuantDataAPI()
        sd, ed = start_date, end_date

        # Split into 3-year chunks to avoid single massive query timeout
        n_chunks = (int(ed[:4]) - int(sd[:4])) // 3 + 1
        if n_chunks > 1:
            logger.info("""[批量] 分 %d 批查询 (%s ~ %s)，%d 个因子""" % (n_chunks, sd, ed, len(to_compute)))
        else:
            logger.info("""[批量] 一次 ClickHouse 查询 (%s ~ %s)，%d 个因子""" % (sd, ed, len(to_compute)))

        all_raw = []
        for ci in range(n_chunks):
            cs = str(int(sd[:4]) + ci * 3) + '-01-01'
            ce = str(min(int(sd[:4]) + (ci + 1) * 3 - 1, int(ed[:4]))) + '-12-31'
            logger.info("""  [批次 %d/%d] %s ~ %s""" % (ci + 1, n_chunks, cs, ce))
            sql = self._build_adj_sql(cs, ce, stocks)
            try:
                raw = api.query(sql)
                if not raw.empty:
                    all_raw.append(raw)
            except Exception as e:
                logger.warning("""  [批次 %d/%d] 查询失败: %s""" % (ci + 1, n_chunks, e))

        if not all_raw:
            logger.warning("[批量] ClickHouse 返回空数据")
            return {}
        import pandas as pd
        raw = pd.concat(all_raw, ignore_index=True)

        conn = duckdb.connect()
        conn.register("_raw", raw)

        # 分离 CTE 和非 CTE 表达式（CTE 需要单独执行，不能嵌入 SELECT）
        cte_factors = [(n, ds) for n, ds in to_compute if ds.strip().upper().startswith("WITH")]
        simple_factors = [(n, ds) for n, ds in to_compute if not ds.strip().upper().startswith("WITH")]

        stats_all = {}

        # 注册行业数据（缓存，供 CTE 中引用 _sw_industry）
        self._register_industry(conn)

        # 1. 非 CTE 表达式：合并为一次查询
        if simple_factors:
            select_parts = [f"{ds} AS \"{name}\"" for name, ds in simple_factors]
            sql_all = f"SELECT ts_code, trade_date, {', '.join(select_parts)} FROM _raw WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL"
            try:
                result = conn.execute(sql_all).df()
                if not result.empty:
                    result["trade_date"] = pd.to_datetime(result["trade_date"])
                    for name, _ in simple_factors:
                        if name in result.columns:
                            stats_all[name] = self._save_factor_result(name, result)
            except Exception as e:
                logger.error(f"[批量] 简单表达式计算失败: {e}")

        # 2. CTE 表达式：每个单独执行
        for name, ds in cte_factors:
            try:
                part = conn.execute(ds).df()
                if not part.empty and "value" in part.columns:
                    part["trade_date"] = pd.to_datetime(part["trade_date"])
                    part = part.rename(columns={"ts_code": "instrument", "value": name})
                    part = part.set_index(["instrument", "trade_date"])
                    part.index.names = ["instrument", "datetime"]
                    part = part.astype({name: "float32"})
                    part = part.sort_index()
                    stats_all[name] = self._save_single_factor(name, part)
                else:
                    logger.warning(f"[批量] {name} CTE 结果为空")
            except Exception as e:
                logger.error(f"[批量] {name} CTE 执行失败: {e}")

        conn.close()
        return stats_all

    def _save_single_factor(self, name: str, df: pd.DataFrame) -> Dict[int, int]:
        """将单因子 DataFrame 按年份写入 warehouse 并更新 meta。"""
        factor_dir = self._warehouse_dir(name)
        factor_dir.mkdir(parents=True, exist_ok=True)
        years = {}
        df["_year"] = df.index.get_level_values("datetime").year
        for year, grp in df.groupby("_year"):
            grp = grp.drop(columns=["_year"])
            grp.to_parquet(factor_dir / f"{year}.parquet", compression="zstd")
            years[int(year)] = len(grp)
        self._update_warehouse_meta(name)
        total = sum(years.values())
        logger.info(f"[批量] {name}: {total:,} 行")
        return years

    def _save_factor_result(self, name: str, result: pd.DataFrame) -> Dict[int, int]:
        """从批量结果中提取单因子写入 warehouse。"""
        factor_df = result[["ts_code", "trade_date", name]].copy()
        factor_df = factor_df.dropna(subset=[name])
        factor_df = factor_df.rename(columns={"ts_code": "instrument", name: "value"})
        factor_df = factor_df.set_index(["instrument", "trade_date"])
        factor_df.index.names = ["instrument", "datetime"]
        factor_df = factor_df.astype({"value": "float32"})
        factor_df = factor_df.sort_index()
        if factor_df.empty:
            return {}
        return self._save_single_factor(name, factor_df)

    def append_to_warehouse(self, name: str, expr: str,
                              start_date: Optional[str] = None,
                              stocks: Optional[List[str]] = None,
                              duckdb_expr: Optional[str] = None) -> int:
        """
        增量追加新数据到仓库（每周/每月调用）。
        只计算 start_date 之后的新数据，与已有数据按日去重。

        Args:
            name: 因子名
            expr: Qlib/DuckDB 表达式
            start_date: 起始日期，None 则自动从仓库最后日期+1天开始
            stocks: 可选股票代码列表

        Returns:
            新增行数
        """
        warehouse_meta = self.get_warehouse_meta(name)

        if start_date is None:
            if warehouse_meta and warehouse_meta.get("data_range", {}).get("last_date"):
                # 从最后日期之后开始
                from datetime import datetime, timedelta
                last = datetime.strptime(warehouse_meta["data_range"]["last_date"], "%Y-%m-%d").date()
                start_date = (last + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                # 仓库为空，退化为全量计算
                start_date = self.config.start_time

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        if start_date >= today:
            logger.info(f"[仓库] {name} 已是最新（最后日期={warehouse_meta.get('data_range', {}).get('last_date','?')}），无需追加")
            return 0

        logger.info(f"[仓库] 增量追加 {name}: {start_date} ~ {today}")
        df = self._compute(name, expr, start_date, today, stocks, duckdb_expr=duckdb_expr)
        if df is None or df.empty:
            logger.info(f"[仓库] {name} 无新数据")
            return 0

        # 按年写回，去重
        df = df.reset_index()
        df["_year"] = pd.to_datetime(df["datetime"]).dt.year
        total_new = 0
        for year, grp in df.groupby("_year"):
            year_df = grp.drop(columns=["_year"]).set_index(["instrument", "datetime"])
            year_df = year_df.sort_index().astype({"value": "float32"})
            existing = self._load_warehouse_year(name, int(year))
            if existing is not None and not existing.empty:
                # 去重：以现有数据为准，只追加新日期的数据
                combined = pd.concat([existing, year_df])
                combined = combined[~combined.index.duplicated(keep="first")]
                combined = combined.sort_index()
            else:
                combined = year_df
            self._save_warehouse_year(name, int(year), combined)
            n_new = len(combined) - (len(existing) if existing is not None else 0)
            total_new += n_new
            logger.info(f"[仓库] {name} {year}: {n_new} 行新增 (总计 {len(combined)} 行)")

        self._update_warehouse_meta(name)
        logger.info(f"[仓库] {name} 增量追加完成: {total_new} 行")
        return total_new

    def load_from_warehouse(self, name: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              filter_alive: bool = True) -> Optional[pd.DataFrame]:
        """
        从仓库读取因子数据。

        Args:
            name: 因子名
            start_date/end_date: 日期范围
            filter_alive: 是否过滤退市股（避免幸存者偏差）

        Returns:
            MultiIndex [instrument, datetime] 的 DataFrame，列 ['value']
        """
        wh_dir = self._warehouse_dir(name)
        if not wh_dir.exists():
            meta = self.get_warehouse_meta(name)
            if meta is None:
                return None

        years = self.get_warehouse_years(name)
        if not years:
            return None

        # 按需过滤年份
        if start_date:
            sy = int(start_date[:4])
            years = [y for y in years if y >= sy]
        if end_date:
            ey = int(end_date[:4])
            years = [y for y in years if y <= ey]
        if not years:
            return None

        parts = []
        for year in years:
            part = self._load_warehouse_year(name, year)
            if part is not None and not part.empty:
                parts.append(part)

        if not parts:
            return None

        df = pd.concat(parts)
        df = df.sort_index()

        if start_date:
            df = df[df.index.get_level_values("datetime") >= start_date]
        if end_date:
            df = df[df.index.get_level_values("datetime") <= end_date]
        # 过滤退市股：按每日上市状态过滤（Point-in-Time），避免幸存者偏差
        if filter_alive:
            try:
                import clickhouse_connect
                from qlworks.config import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
                ch = clickhouse_connect.get_client(
                    host=CH_HOST, port=CH_PORT,
                    user=CH_USER, password=CH_PASSWORD,
                    database=CH_DATABASE, connect_timeout=10,
                )
                dates = df.index.get_level_values("datetime").unique()
                date_list = sorted(dates)
                all_alive = set()
                for batch_start in range(0, len(date_list), 200):
                    batch_dates = date_list[batch_start:batch_start+200]
                    dr_str = ",".join(f"'{d.date()}'" for d in batch_dates)
                    try:
                        univ = ch.query_df(f"""
                            SELECT ts_code, trade_date
                            FROM stock_universe_daily
                            WHERE trade_date IN ({dr_str})
                                  AND list_status = 'L'
                        """)
                        if not univ.empty:
                            for _, row in univ.iterrows():
                                key = str(row["ts_code"]) + "@" + str(row["trade_date"])
                                all_alive.add(key)
                    except Exception:
                        pass
                if all_alive:
                    idx = df.index.get_level_values("instrument") + "@" + df.index.get_level_values("datetime").astype(str)
                    n_before = len(df)
                    df = df[[x in all_alive for x in idx]]
                    removed = n_before - len(df)
                    if removed > 0:
                        logger.info(f"[仓库] {name} PIT 退市股过滤: 去除 {removed} 行，保留 {len(df)} 行")
                else:
                    logger.warning(f"[仓库] {name} stock_universe_daily 查询为空，跳过过滤")
                ch.close()
            except Exception as e:
                try:
                    logger.warning(f"[仓库] {name} PIT 过滤不可用，回退到当前状态: {e}")
                    ch = clickhouse_connect.get_client(
                        host=CH_HOST, port=CH_PORT,
                        user=CH_USER, password=CH_PASSWORD,
                        database=CH_DATABASE, connect_timeout=10,
                    )
                    universe = ch.query_df("SELECT DISTINCT ts_code FROM stock_universe WHERE list_status='L'")
                    ch.close()
                    if not universe.empty:
                        alive_codes = set(universe["ts_code"].tolist())
                        mask = df.index.get_level_values("instrument").isin(alive_codes)
                        n_before = len(df)
                        df = df[mask]
                        removed = n_before - len(df)
                        if removed > 0:
                            logger.warning(f"[仓库] {name} 当前状态过滤: 去除 {removed} 行")
                except Exception as e2:
                    logger.warning(f"[仓库] {name} 退市股过滤全部跳过: {e2}")


        return df

    def load(self, name: str, expr: str,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        统一加载接口（兼容旧接口）：
        仓库 > 缓存 > 计算

        [Citadel Alpha Lab] 优先使用仓库数据避免重复计算。
        """
        # 1. 优先仓库
        df = self.load_from_warehouse(name, start_date, end_date)
        if df is not None and not df.empty:
            return df

        # 2. 退到缓存
        r = self._load_cache(name, start_date, end_date)
        if r is not None:
            return r

        # 3. 实时计算（并自动写入仓库）
        logger.info(f"[因子] {name} 仓库无缓存，实时计算...")
        sd = start_date or self.config.start_time
        ed = end_date or self.config.end_time
        df = self._compute(name, expr, sd, ed)
        if df is not None and not df.empty:
            # 写入仓库供后续复用
            self._save_warehouse_chunk(name, df)
            self._update_warehouse_meta(name)
        return df if df is not None else pd.DataFrame()

    # ──────────── 等级管理 ────────────

    def link_factor_to_tier(self, name: str, tier: str) -> str:
        """
        将仓库中的因子链接到 tier 目录（不复制数据，只写 registry 记录）。
        原 export_factor_by_status 的替换。

        Returns:
            目标目录路径
        """
        from datetime import datetime
        meta = self.get_warehouse_meta(name)
        out_dir = self.factors_dir / tier
        out_dir.mkdir(parents=True, exist_ok=True)

        # 在 tier 目录下写入引用标记（轻量 meta）
        ref = {
            "factor_name": name,
            "tier": tier,
            "warehouse_path": str(self._warehouse_dir(name)),
            "last_updated": str(datetime.now()),
            "years": self.get_warehouse_years(name),
            "total_rows": meta.get("statistics", {}).get("total_records", 0) if meta else 0,
            "last_date": meta.get("data_range", {}).get("last_date", "") if meta else "",
            "data_version": "3.0",
        }
        ref_path = out_dir / f"{name}.ref.json"
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref, f, ensure_ascii=False, indent=2)

        return str(out_dir)

    def list_evaluated(self) -> List[dict]:
        """列出所有已评测（有 ref 标记）的因子。"""
        result = []
        for tier in ["core", "satellite", "archive"]:
            td = self.factors_dir / tier
            if not td.exists():
                continue
            for ref_file in sorted(td.glob("*.ref.json")):
                try:
                    with open(ref_file) as f:
                        meta = json.load(f)
                    result.append({
                        "name": meta.get("factor_name", ref_file.stem),
                        "tier": tier,
                    })
                except Exception:
                    continue
        return result

    def get_evaluated_meta(self, name: str) -> Optional[dict]:
        """获取已评测因子的 meta。"""
        for tier in ["core", "satellite", "archive"]:
            ref_path = self.factors_dir / tier / f"{name}.ref.json"
            if ref_path.exists():
                with open(ref_path) as f:
                    return json.load(f)
        return None

    def get_evaluated(self, name: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """兼容旧接口：读取已评测因子数据（实际走仓库）。"""
        # 先确认有评测记录
        meta = self.get_evaluated_meta(name)
        if meta is None:
            return None
        return self.load_from_warehouse(name, start_date, end_date)

    # ──────────── 仓库内部方法 ────────────

    def _warehouse_dir(self, name: str) -> Path:
        return self.warehouse_dir / name

    def _warehouse_meta_path(self, name: str) -> Path:
        return self._warehouse_dir(name) / "meta.json"

    def _warehouse_year_path(self, name: str, year: int) -> Path:
        return self._warehouse_dir(name) / f"{year}.parquet"

    def _save_warehouse_year(self, name: str, year: int, df: pd.DataFrame):
        p = self._warehouse_year_path(name, year)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, compression="zstd")

    def _load_warehouse_year(self, name: str, year: int) -> Optional[pd.DataFrame]:
        p = self._warehouse_year_path(name, year)
        if p.exists():
            return pd.read_parquet(p)
        return None

    def _save_warehouse_chunk(self, name: str, df: pd.DataFrame):
        """将计算结果的 DataFrame 按年拆分写入仓库。"""
        if df.index.names != ["instrument", "datetime"]:
            df = df.set_index(["instrument", "datetime"])
        df = df.sort_index()
        years = set(df.index.get_level_values("datetime").year)
        for year in years:
            chunk = df[df.index.get_level_values("datetime").year == year]
            existing = self._load_warehouse_year(name, year)
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, chunk])
                combined = combined[~combined.index.duplicated(keep="first")]
                combined = combined.sort_index()
            else:
                combined = chunk
            self._save_warehouse_year(name, year, combined)

    def _update_warehouse_meta(self, name: str):
        """
        更新仓库元信息。
        
        保留已有的 YAML 语义元数据（表达式、含义、理论背景等），
        只更新数据覆盖范围与基本统计。
        """
        years = self.get_warehouse_years(name)
        total_rows = 0
        first_date = None
        last_date = None
        n_unique_stocks = 0

        for y in years:
            df = self._load_warehouse_year(name, y)
            if df is None or df.empty:
                continue
            total_rows += len(df)
            dates = df.index.get_level_values("datetime")
            if first_date is None or dates.min() < pd.Timestamp(first_date):
                first_date = dates.min().strftime("%Y-%m-%d")
            if last_date is None or dates.max() > pd.Timestamp(last_date):
                last_date = dates.max().strftime("%Y-%m-%d")
            n_unique_stocks = max(n_unique_stocks, df.index.get_level_values("instrument").nunique())

        # 读取已有的 meta.json，保留 YAML 语义字段
        existing_meta = {}
        p = self._warehouse_meta_path(name)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except Exception:
                pass

        # 从已有 meta 中提取 YAML 语义字段（这些由 YAML 因子库注入，不应被覆盖）
        semantic_fields = {
            "version", "category", "sub_category",
            "expression", "function_description", "theory_background",
            "applicable_conditions", "reference", "lifecycle_stage",
            "meaning", "usage_scenario", "strategy_hint",
        }

        meta = {
            "factor_name": name,
            "data_version": "3.0",
            "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_range": {
                "years": years,
                "first_date": first_date or "",
                "last_date": last_date or "",
                "total_records": total_rows,
                "unique_stocks": n_unique_stocks,
            },
        }

        # 注入已有的 YAML 语义字段
        for field in semantic_fields:
            if field in existing_meta and existing_meta[field]:
                meta[field] = existing_meta[field]

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ──────────── 计算引擎（复用旧代码） ────────────

    @staticmethod
    def _chunk_dates(start_date, end_date, chunk_years=_CHUNK_YEARS):
        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)
        chunks = []
        cur = s
        while cur < e:
            nxt = cur + pd.DateOffset(years=chunk_years)
            if nxt > e:
                nxt = e
            chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
            cur = nxt + pd.Timedelta(days=1)
        return chunks
    def _build_adj_sql(self, start_date, end_date, stocks=None):
        """????? OHLCV + ???? + ?? + ?????? SQL?"""
        adj = ", ".join(f"CAST(p.{f} * a.adj_factor / latest.adj_factor AS DOUBLE) AS {f}"
                        for f in ["open", "high", "low", "close"])
        indicator_fields = ", ".join(
            f"CAST(i.{f} AS DOUBLE) AS {f}"
            for f in sorted(DAILY_INDICATOR_FIELDS)
        )
        stock_filter = ""
        if stocks:
            codes = ", ".join(f"'{c}'" for c in stocks)
            stock_filter = f" AND p.ts_code IN ({codes})"
        return f"""SELECT p.ts_code AS ts_code, p.trade_date AS trade_date, {adj},
       CAST(p.vol AS DOUBLE) AS volume, CAST(p.amount AS DOUBLE) AS amount,
       CAST(p.pre_close AS DOUBLE) AS pre_close, CAST(p.pct_chg AS DOUBLE) AS change_pct,
       {indicator_fields},
       sw.l1_name AS industry, sw.l1_code AS industry_code
FROM daily_prices p
JOIN daily_adj_factors a ON p.ts_code=a.ts_code AND p.trade_date=a.trade_date
JOIN (SELECT ts_code, argMax(adj_factor, trade_date) AS adj_factor FROM daily_adj_factors GROUP BY ts_code) latest ON p.ts_code=latest.ts_code
LEFT JOIN daily_indicators i ON p.ts_code=i.ts_code AND p.trade_date=i.trade_date
LEFT JOIN sw_industry_members sw ON p.ts_code=sw.ts_code
WHERE p.trade_date>='{start_date}' AND p.trade_date<='{end_date}'{stock_filter}
ORDER BY p.ts_code, p.trade_date"""

    @staticmethod
    def _is_financial_factor(duckdb_expr: str) -> bool:
        """判断 DuckDB 表达式是否引用财务季频字段（需要 forward-fill）。"""
        # 提取表达式中所有可能的字段名（字母/数字/下划线组成的 token）
        tokens = set(re.findall(r'[a-zA-Z_]\w*', duckdb_expr))
        # 过滤掉 SQL 关键字和 OHLCV 基础字段
        skip = {'over', 'partition', 'by', 'order', 'rows', 'between',
                'and', 'preceding', 'current', 'row', 'from', 'where',
                'select', 'as', 'on', 'lag', 'lead', 'avg', 'sum',
                'stddev', 'std', 'min', 'max', 'count', 'abs',
                'cast', 'null', 'not', 'is', 'in', 'case', 'when',
                'then', 'else', 'end', 'desc', 'asc', 'nullif',
                'coalesce', 'round', 'floor', 'ceil', 'ln', 'log',
                'exp', 'power', 'sqrt', 'sign', 'ref'}
        skip |= {'open', 'high', 'low', 'close', 'volume', 'vol', 'amount',
                 'ts_code', 'trade_date', 'adj_factor', 'vwap', 'returns'}
        tokens = tokens - skip
        # 如果有 token 命中 FINANCIAL_FIELDS 或 EXTRA_FINANCIAL_TABLES，则为财务因子
        return bool(tokens & (FINANCIAL_FIELDS | set(EXTRA_FINANCIAL_TABLES.keys())))

    def _try_duckdb_financial(self, name: str, expr: str,
                                start_date: str, end_date: str,
                                stocks: Optional[List[str]] = None,
                                duckdb_expr: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        DuckDB 计算财务因子路径。

        财务数据在 ClickHouse 中是季度频率（financial_indicators 表，以 ann_date 为准）。
        需要：
          1. 从 ClickHouse 拉取原始财务数据
          2. 用 DuckDB 向前填充（forward-fill）到日频
          3. 应用因子表达式计算

        Args:
            name: 因子名
            expr: Qlib 表达式（来自 YAML 的 expression.qlib）
            duckdb_expr: DuckDB 表达式（来自 YAML 的 expression.duckdb，优先使用）
            start_date/end_date: 日期范围
            stocks: 可选股票列表
        """
        try:
            import duckdb
            from qlworks.data import QuantDataAPI
        except ImportError:
            return None

        # 优先使用 YAML 中定义的 duckdb 表达式，否则从 Qlib 表达式转换
        ds = duckdb_expr or self._qlib_to_duckdb(expr)
        if ds is None:
            return None

        sd = start_date
        ed = end_date
        logger.info(f"[财务] DuckDB forward-fill 计算 {name}: {ds}")

        api = QuantDataAPI()

        # 1. 从 ClickHouse 查询 financial_indicators 原始数据
        fin_sql = f"""SELECT ts_code, ann_date, {', '.join(sorted(FINANCIAL_FIELDS))}
FROM financial_indicators
WHERE ann_date >= '{sd}' AND ann_date <= '{ed}'"""
        if stocks:
            ts_list = ", ".join(f"'{c}'" for c in stocks)
            fin_sql += f" AND ts_code IN ({ts_list})"
        fin_sql += " ORDER BY ts_code, ann_date"

        try:
            fin_raw = api.query(fin_sql)
        except Exception as e:
            logger.warning(f"[财务] {name} financial_indicators 查询失败: {e}")
            return None
        if fin_raw.empty:
            logger.warning(f"[财务] {name}: financial_indicators 无数据")
            return None
        fin_raw["ann_date"] = pd.to_datetime(fin_raw["ann_date"])
        # ClickHouse Decimal 列需要转为 float64，否则 DuckDB 注册时可能溢出
        dec_cols = [c for c in fin_raw.columns if c not in ('ts_code', 'ann_date', 'end_date')]
        for c in dec_cols:
            if fin_raw[c].dtype in (object,):
                fin_raw[c] = pd.to_numeric(fin_raw[c], errors='coerce').astype('float32')
            elif 'decimal' in str(fin_raw[c].dtype).lower():
                fin_raw[c] = fin_raw[c].astype('float32')
        fin_raw = fin_raw.fillna(0.0)

        # 1.5 从 EXTRA_FINANCIAL_TABLES 补充字段（cashflow_statement, income_statement 等）
        extra_cols = []
        for field_name, (table_name, col_name) in EXTRA_FINANCIAL_TABLES.items():
            try:
                extra_sql = f"""SELECT ts_code, ann_date, {col_name} AS {field_name}
FROM {table_name}
WHERE ann_date >= '{sd}' AND ann_date <= '{ed}'"""
                if stocks:
                    ts_list = ", ".join(f"'{c}'" for c in stocks)
                    extra_sql += f" AND ts_code IN ({ts_list})"
                extra_sql += " ORDER BY ts_code, ann_date"
                extra_df = api.query(extra_sql)
                if not extra_df.empty:
                    extra_df["ann_date"] = pd.to_datetime(extra_df["ann_date"])
                    for c in extra_df.columns:
                        if c not in ('ts_code', 'ann_date'):
                            extra_df[c] = pd.to_numeric(extra_df[c], errors='coerce').astype('float32')
                    extra_df = extra_df.fillna(0.0)
                    # 按 ts_code + ann_date 合并到 fin_raw
                    fin_raw = fin_raw.merge(extra_df, on=["ts_code", "ann_date"], how="left", suffixes=("", "_extra"))
                    # 清除多余的 _extra 列
                    for c in list(fin_raw.columns):
                        if c.endswith("_extra"):
                            del fin_raw[c]
                    extra_cols.append(field_name)
                    logger.info(f"[财务] 已补充 {field_name} 来自 {table_name}.{col_name}: {len(extra_df)} 行")
            except Exception as e:
                logger.warning(f"[财务] {field_name} 从 {table_name} 查询失败: {e}")

        # 填充 null 为 0
        fin_raw = fin_raw.fillna(0.0)

        # 2. 获取交易日历
        cal_sql = f"SELECT DISTINCT trade_date FROM daily_prices WHERE trade_date >= '{sd}' AND trade_date <= '{ed}' ORDER BY trade_date"
        calendar = api.query(cal_sql)
        if calendar.empty:
            return None
        calendar["trade_date"] = pd.to_datetime(calendar["trade_date"])

        # 3. 获取股票列表
        if stocks:
            stock_list = stocks
        else:
            stk_df = fin_raw[["ts_code"]].drop_duplicates()
            stock_list = stk_df["ts_code"].tolist()

        # 4. 在 DuckDB 中 forward-fill 并计算因子
        conn = duckdb.connect()
        conn.register("_fin", fin_raw)
        conn.register("_cal", calendar)

        try:
            # forward-fill: 对每个股票 × 每天，取最近一次公告的财务数据
            # 注意：ClickHouse Decimal 类型在 DuckDB 中可能溢出，先 CAST 为 DOUBLE
            # 先用简单字段尝试（适用于 roe, roa 等单字段因子）
            ff_sql = f"""
            WITH
            stocks AS (
                SELECT DISTINCT ts_code FROM _fin
            ),
            ff AS (
                SELECT
                    s.ts_code,
                    c.trade_date,
                    LAST(fi.{ds} ORDER BY fi.ann_date) AS value
                FROM stocks s
                CROSS JOIN _cal c
                LEFT JOIN _fin fi
                    ON s.ts_code = fi.ts_code
                    AND fi.ann_date <= c.trade_date
                GROUP BY s.ts_code, c.trade_date
            )
            SELECT ts_code, trade_date, value
            FROM ff
            WHERE value IS NOT NULL AND NOT isnan(value)
            ORDER BY ts_code, trade_date
            """
            result = conn.execute(ff_sql).df()
        except Exception as e:
            # 如果简单字段失败（如复合表达式 ocf/net_profit、含窗口函数等），
            # 尝试先 forward-fill 所有原始字段，再在每日结果上应用表达式
            logger.info(f"[财务] {name} 简单字段 failed, 尝试全字段 forward-fill: {e}")
            try:
                ff_fields = sorted(set(FINANCIAL_FIELDS) | set(EXTRA_FINANCIAL_TABLES.keys()))
                ff_all_sql = f"""
                WITH
                stocks AS (SELECT DISTINCT ts_code FROM _fin),
                calendar AS (SELECT trade_date FROM _cal),
                ff_all AS (
                    SELECT
                        s.ts_code,
                        c.trade_date,
                        {', '.join(
                            f'LAST(fi.{f} ORDER BY fi.ann_date) AS {f}'
                            for f in ff_fields
                        )}
                    FROM stocks s
                    CROSS JOIN calendar c
                    LEFT JOIN _fin fi
                        ON s.ts_code = fi.ts_code
                        AND fi.ann_date <= c.trade_date
                    GROUP BY s.ts_code, c.trade_date
                )
                SELECT ts_code, trade_date, {ds} AS value
                FROM ff_all
                WHERE {ds} IS NOT NULL AND NOT isnan({ds})
                ORDER BY ts_code, trade_date
                """
                result = conn.execute(ff_all_sql).df()
            except Exception as e2:
                # 如果全字段 forward-fill 仍失败，且表达式含 LAG（如 stk_holdernumber / lag(stk_holdernumber, 20) - 1），
                # 用两步法：先 forward-fill 所有字段，再在外层应用 LAG 窗口函数
                if 'lag(' in ds.lower():
                    logger.info(f"[财务] {name} 包含 LAG 窗口函数，尝试两步 forward-fill: {e2}")
                    try:
                        ff_fields = sorted(set(FINANCIAL_FIELDS) | set(EXTRA_FINANCIAL_TABLES.keys()))
                        ff_base_sql = f"""
                        WITH
                        stocks AS (SELECT DISTINCT ts_code FROM _fin),
                        calendar AS (SELECT trade_date FROM _cal),
                        ff_all AS (
                            SELECT
                                s.ts_code,
                                c.trade_date,
                                {', '.join(
                                    f'LAST(fi.{f} ORDER BY fi.ann_date) AS {f}'
                                    for f in ff_fields
                                )}
                            FROM stocks s
                            CROSS JOIN calendar c
                            LEFT JOIN _fin fi
                                ON s.ts_code = fi.ts_code
                                AND fi.ann_date <= c.trade_date
                            GROUP BY s.ts_code, c.trade_date
                        ),
                        calc AS (
                            SELECT ts_code, trade_date, {ds} AS value
                            FROM ff_all
                        )
                        SELECT ts_code, trade_date, value
                        FROM calc
                        WHERE value IS NOT NULL AND NOT isnan(value)
                        ORDER BY ts_code, trade_date
                        """
                        result = conn.execute(ff_base_sql).df()
                    except Exception as e3:
                        logger.warning(f"[财务] {name} 两步 forward-fill 也失败: {e3}")
                        conn.close()
                        return None
                else:
                    logger.warning(f"[财务] {name} 全字段 forward-fill 也失败: {e2}")
                    conn.close()
                    return None

        conn.close()

        if result.empty:
            logger.warning(f"[财务] {name} forward-fill 后无有效数据")
            return None

        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = result.set_index(["ts_code", "trade_date"])
        result.index.names = ["instrument", "datetime"]
        result = result.astype({"value": "float32"})
        result = result.sort_index()

        logger.info(f"[财务] DuckDB {name}: {len(result)} 行 (forward-fill)")
        return result

    def _compute(self, name, expr, start_date=None, end_date=None, stocks=None,
                 duckdb_expr=None):
        """计算因子（DuckDB 价格指标路径 → DuckDB 财务路径 → Qlib 兜底）。"""
        sd = start_date or self.config.start_time
        ed = end_date or self.config.end_time

        # 检查是否强制使用 Qlib
        force_qlib = os.environ.get("FORCE_QLIB", "false").lower() == "true"
        if not force_qlib:
            # [Bloomberg Data Pipeline] 第 1 层：DuckDB + daily_prices/daily_indicators
            # 覆盖量价因子 + 估值/市值类因子（pe_ttm, pb, circ_mv 等）
            df = self._try_duckdb(name, expr, sd, ed, stocks, duckdb_expr=duckdb_expr)
            if df is not None and not FactorStore._is_degenerate(df):
                return df
            if df is not None:
                logger.warning(f"[计算] {name} DuckDB 结果退化（全零/常数），回退到 Qlib...")

            # [AQR] 第 2 层：DuckDB + financial_indicators forward-fill
            # 覆盖财务因子（roe, eps, netprofit_yoy 等）
            ds = duckdb_expr or self._qlib_to_duckdb(expr)
            if ds is not None and self._is_financial_factor(ds):
                df = self._try_duckdb_financial(name, expr, sd, ed, stocks, duckdb_expr)
                if df is not None and not FactorStore._is_degenerate(df):
                    return df
                if df is not None:
                    logger.warning(f"[计算] {name} DuckDB 财务结果退化，回退到 Qlib...")
        else:
            logger.info(f"[计算] {name} 强制使用 Qlib (FORCE_QLIB=true)")

        # 第 3 层：Qlib 兜底（最慢但最全）
        df = self._try_qlib(name, expr, sd, ed, stocks)
        if df is not None:
            return df
        raise RuntimeError(f"无法计算因子 {name}: {expr}")

    @staticmethod

    @staticmethod
    def _detect_suspension(df: pd.DataFrame) -> pd.Series:
        """?????????? 0 ?????????? bool Series?"""
        if df is None or df.empty:
            return pd.Series(dtype=bool)
        vol = df.get("volume", df.get("vol", None))
        if vol is None:
            return pd.Series(False, index=df.index)
        vol = pd.to_numeric(vol, errors="coerce").fillna(0)
        return vol == 0

    def _is_degenerate(df: pd.DataFrame) -> bool:
        """检测因子计算结果是否退化（全零/常数）。
        
        当 ClickHouse 字段不存在或全部为 NULL 时，DuckDB 计算可能返回全零，
        此时应回退到 Qlib。
        """
        if df is None or df.empty:
            return True
        v = df["value"]
        if len(v) < 10:
            return False  # 样本太少，无法判断
        # 检查是否全零（或 99%+ 零）
        zero_ratio = (v == 0).sum() / len(v) if hasattr(v, '__len__') else 0
        if zero_ratio > 0.99:
            return True
        # 检查是否常数（std=0）
        std_val = v.std()
        if pd.isna(std_val) or std_val < 1e-12:
            return True
        return False

    _industry_cache: Optional[pd.DataFrame] = None

    def _register_industry(self, conn) -> bool:
        """将行业数据注册到 DuckDB 连接（缓存，仅查询一次）。"""
        if FactorStore._industry_cache is None:
            try:
                from qlworks.data import QuantDataAPI
                api = QuantDataAPI()
                ind = api.query("""
                    SELECT ts_code, l1_code AS sw_l1, l1_name AS sw_l1_name
                    FROM sw_industry_members
                """)
                if not ind.empty:
                    FactorStore._industry_cache = ind
            except Exception:
                return False
        if FactorStore._industry_cache is not None:
            conn.register("_sw_industry", FactorStore._industry_cache)
            return True
        return False

    def _try_duckdb(self, name, expr, start_date=None, end_date=None, stocks=None, duckdb_expr=None):
        try:
            import duckdb
            from qlworks.data import QuantDataAPI
        except ImportError:
            return None

        sd = start_date or self.config.start_time
        ed = end_date or self.config.end_time
        ds = duckdb_expr or FactorStore._qlib_to_duckdb(expr)
        if ds is None:
            return None

        api = QuantDataAPI()
        chunks = FactorStore._chunk_dates(sd, ed)
        logger.info(f"[计算] DuckDB {name}: {len(chunks)} chunks ({sd} ~ {ed})")

        all_parts = []
        for ck_s, ck_e in chunks:
            sql = self._build_adj_sql(ck_s, ck_e, stocks)
            try:
                raw = api.query(sql)
            except Exception:
                logger.warning(f"  chunk {ck_s}~{ck_e} 查询失败, 跳过")
                continue
            if raw.empty:
                continue

            conn = duckdb.connect()
            conn.register("_raw", raw)
            self._register_industry(conn)
            try:
                if ds.strip().upper().startswith("WITH"):
                    # CTE 表达式：直接执行完整 SQL（里面自己引用 _raw）
                    part = conn.execute(ds).df()
                else:
                    part = conn.execute(
                        f"SELECT ts_code, trade_date, {ds} AS value FROM _raw "
                        f"WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL "
                        f"ORDER BY ts_code, trade_date"
                    ).df()
            except Exception as e:
                logger.warning(f"  chunk {ck_s}~{ck_e} 计算失败: {e}")
                conn.close()
                continue
            conn.close()

            if not part.empty:
                part = part.dropna(subset=["value"])
                all_parts.append(part)

        if not all_parts:
            return None

        r = pd.concat(all_parts, ignore_index=True)
        r["trade_date"] = pd.to_datetime(r["trade_date"])
        r = r.set_index(["ts_code", "trade_date"])
        r.index.names = ["instrument", "datetime"]
        r = r.astype({"value": "float32"})
        r = r.sort_index()

        logger.info(f"[计算] DuckDB {name}: {len(r)} 行")
        return r

    @staticmethod
    def _qlib_to_duckdb(expr):
        if not expr:
            return None
        e = expr.strip()

        def _r(m):
            f = m.group(1).lstrip("$")
            off = abs(int(m.group(2)))
            return f"LAG({f},{off}) OVER (PARTITION BY ts_code ORDER BY trade_date)"

        e = re.sub(r"Ref\((\$\w+),\s*(-?\d+)\)", _r, e)

        def _m(m):
            f = m.group(1).lstrip("$")
            n = int(m.group(2))
            return f"AVG({f}) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {n-1} PRECEDING AND CURRENT ROW)"

        e = re.sub(r"Mean\((\$\w+),\s*(\d+)\)", _m, e)

        def _s(m):
            f = m.group(1).lstrip("$")
            n = int(m.group(2))
            return f"STDDEV({f}) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {n-1} PRECEDING AND CURRENT ROW)"

        e = re.sub(r"Std\((\$\w+),\s*(\d+)\)", _s, e)

        def _minmax(m, func):
            f = m.group(1).lstrip("$")
            n = int(m.group(2))
            return f"{func}({f}) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {n-1} PRECEDING AND CURRENT ROW)"

        e = re.sub(r"Min\((\$\w+),\s*(\d+)\)", lambda m: _minmax(m, "MIN"), e)
        e = re.sub(r"Max\((\$\w+),\s*(\d+)\)", lambda m: _minmax(m, "MAX"), e)
        e = re.sub(r"Sum\((\$\w+),\s*(\d+)\)", lambda m: _minmax(m, "SUM"), e)
        e = re.sub(r"Ts_Max\((\$\w+),\s*(\d+)\)", lambda m: _minmax(m, "MAX"), e)

        # Delta($field, N) -> field - LAG(field, N) OVER (...)
        def _delta(m):
            f = m.group(1).lstrip("$")
            off = abs(int(m.group(2)))
            return f"{f} - LAG({f},{off}) OVER (PARTITION BY ts_code ORDER BY trade_date)"

        e = re.sub(r"Delta\((\$\w+),\s*(-?\d+)\)", _delta, e)

        # If(cond, then, else) -> CASE WHEN cond THEN then ELSE else END
        e = re.sub(r"If\((.+?),\s*(.+?),\s*(.+?)\)", r"CASE WHEN \1 THEN \2 ELSE \3 END", e)
        # Greater(a, b) -> CASE WHEN a > b THEN 1 ELSE 0 END
        e = re.sub(r"Greater\((.+?),\s*(.+?)\)", r"CASE WHEN \1 > \2 THEN 1 ELSE 0 END", e)
        # Less(a, b) -> CASE WHEN a < b THEN 1 ELSE 0 END
        e = re.sub(r"Less\((.+?),\s*(.+?)\)", r"CASE WHEN \1 < \2 THEN 1 ELSE 0 END", e)
        # Abs($field) -> ABS(field)
        e = re.sub(r"Abs\((\$\w+)\)", r"ABS(\1)", e)
        # Sign($field) -> SIGN(field)
        e = re.sub(r"Sign\((\$\w+)\)", r"SIGN(\1)", e)

        # 转换 Log($field) → ln(nullif(field, 0)) 防止 log(0) 报错
        e = re.sub(r"Log\((\$\w+)\)", r"ln(nullif(\1, 0))", e)
        e = re.sub(r"\$(\w+)", r"\1", e)
        # 如果 Mean/Std 还有未匹配的（复杂表达式含窗口函数嵌套），退回 Qlib
        if 'Mean(' in e or 'Std(' in e or 'Corr(' in e or 'Rank(' in e or 'Ts_Rank(' in e:
            return None
        for kw in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]:
            if kw in e.upper():
                return None
        return e

    def _try_qlib(self, name, expr, start_date=None, end_date=None, stocks=None):
        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D
            from qlworks.config import QLIB_DATA_DIR
        except ImportError:
            return None
        sd = start_date or self.config.start_time
        ed = end_date or self.config.end_time
        try:
            if not self._qlib_inited:
                qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)
                self._qlib_inited = True
            # Windows 单线程加速
            from qlib.config import C as _QC
            _QC.dataloader_workers = 0
        except Exception:
            pass

        pool = stocks
        if not pool:
            ifile = Path(str(QLIB_DATA_DIR)) / "instruments" / "all.txt"
            pool = []
            if ifile.exists():
                with open(ifile, encoding="utf-8") as f:
                    for line in f:
                        p = line.strip().split()
                        if p:
                            pool.append(p[0])
        if not pool:
            return None

        # 分批加载股票，避免 Qlib multiprocessing 在 Windows 上过慢
        batch_size = 500
        all_parts = []
        for i in range(0, len(pool), batch_size):
            batch = pool[i:i + batch_size]
            for ck_s, ck_e in FactorStore._chunk_dates(sd, ed):
                try:
                    df = D.features(batch, [expr], ck_s, ck_e)
                except Exception:
                    continue
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.rename(columns={expr: "value"})
                df = df.reset_index()
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["instrument"] = df["instrument"].astype(str)
                df = df.dropna(subset=["value"])
                all_parts.append(df)

        if not all_parts:
            return None
        r = pd.concat(all_parts, ignore_index=True)
        r = r.set_index(["instrument", "datetime"])[["value"]]
        r = r.sort_index().astype({"value": "float32"})
        return r

    def _save_cache(self, name, df):
        p = self.cache_dir / f"{name}.parquet"
        df.to_parquet(p, compression="zstd")

    def _load_cache(self, name, start_date=None, end_date=None):
        cf = self.cache_dir / f"{name}.parquet"
        if cf.exists():
            df = pd.read_parquet(cf)
            if start_date:
                df = df[df.index.get_level_values("datetime") >= start_date]
            if end_date:
                df = df[df.index.get_level_values("datetime") <= end_date]
            return df
        return None

    # ──────────── 兼容旧接口 ────────────

    def load_multi(self, names, start_date=None, end_date=None):
        """兼容旧接口。"""
        frames = {}
        for name in names:
            try:
                df = self.load(name, "", start_date, end_date)
                if df is not None and not df.empty:
                    frames[name] = df["value"]
            except Exception as e:
                logger.warning(f"load {name} failed: {e}")
        if not frames:
            return pd.DataFrame()
        r = pd.concat(frames, axis=1)
        r.columns = frames.keys()
        return r

