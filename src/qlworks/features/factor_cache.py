"""
DuckDB + Parquet 因子预计算缓存

将因子计算从 Qlib 逐股票逐表达式求值中剥离，改为批量管道：
  1. 从 ClickHouse 一次拉取全量原始 OHLCV 宽表
  2. 载入 DuckDB 用 SQL 窗口函数计算因子表达式
  3. 每个因子存为一个独立 Parquet 文件（zstd 压缩）
  4. 后续加载只需读 Parquet（ms 级），永不重复计算

种子因子（3 个，用于跑通流程）：
  - ret_1d           1 日收益率
  - ma_5             5 日移动平均
  - price_position_20  20 日价格位置

添加新因子只需在 SEED_FACTORS 中追加一条定义即可。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import duckdb

from qlworks.config import FACTOR_CACHE_DIR


SEED_FACTORS: Dict[str, dict] = {
    "ret_1d": {
        "name": "ret_1d",
        "description": "1日收益率: close / Ref(close, 1) - 1",
        "source_fields": ["ts_code", "trade_date", "close"],
        "sql": "close / LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1",
    },
    "ma_5": {
        "name": "ma_5",
        "description": "5日移动平均: Mean(close, 5)",
        "source_fields": ["ts_code", "trade_date", "close"],
        "sql": "AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)",
    },
    "price_position_20": {
        "name": "price_position_20",
        "description": "20日价格位置: (close - min_20) / (max_20 - min_20)",
        "source_fields": ["ts_code", "trade_date", "close"],
        "sql": """
            (close - MIN(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))
            / NULLIF(MAX(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            - MIN(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)
        """,
    },
}


class FactorCache:
    """
    因子预计算缓存

    架构：
      ClickHouse（原始 OHLCV，一次批量查询）
          │
          ▼
      DuckDB（SQL 窗口函数批量计算）
          │
          ▼
      Parquet 文件（每个因子独立，zstd 压缩）
          │
          ▼
      读取（pd.read_parquet，ms 级）
    """

    def __init__(self, api):
        self.api = api
        self.cache_dir = Path(str(FACTOR_CACHE_DIR))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 公开方法 ──────────────────────────────────────────

    def list_factors(self) -> List[str]:
        """返回已缓存的因子列表。"""
        return sorted(set(
            f.stem for f in self.cache_dir.glob("*.parquet")
            if f.stem in SEED_FACTORS
        ))

    def has_factor(self, name: str) -> bool:
        return (self.cache_dir / f"{name}.parquet").exists()

    def load_factor(self, name: str, start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        从 Parquet 加载预计算因子。

        Returns:
            DataFrame with MultiIndex [instrument, datetime], single factor column
        """
        path = self.cache_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"因子 {name!r} 未缓存。可用: {self.list_factors()}。"
                f"请先调用 compute_factor({name!r})"
            )
        df = pd.read_parquet(path)
        if start_date:
            df = df[df.index.get_level_values("datetime") >= start_date]
        if end_date:
            df = df[df.index.get_level_values("datetime") <= end_date]
        return df

    @staticmethod
    def _build_adj_close_sql(start_date: str, end_date: str, stock_filter: str = "") -> str:
        """
        构建前复权收盘价查询 SQL。

        前复权公式: adj_close = close * adj_factor / latest_adj_factor
        使用 ClickHouse 的 argMax 窗口函数高效获取每只股票最新复权因子。
        """
        return f"""
            SELECT p.ts_code AS ts_code, p.trade_date AS trade_date,
                   toFloat64(p.close * a.adj_factor / latest.adj_factor) AS close
            FROM daily_prices p
            JOIN daily_adj_factors a ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
            JOIN (
                SELECT ts_code, argMax(adj_factor, trade_date) AS adj_factor
                FROM daily_adj_factors
                GROUP BY ts_code
            ) latest ON p.ts_code = latest.ts_code
            WHERE p.trade_date >= '{start_date}' AND p.trade_date <= '{end_date}'
            {stock_filter}
            ORDER BY p.ts_code, p.trade_date
        """

    def compute_factor(self, name: str, start_date: str = "2010-01-01",
                       end_date: Optional[str] = None,
                       overwrite: bool = False,
                       stocks: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从 ClickHouse 拉取原始数据 → DuckDB 计算 → 缓存为 Parquet。

        Args:
            name: 因子名，必须在 SEED_FACTORS 中
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期，None 表示最新
            overwrite: 是否覆盖已有缓存
            stocks: 股票代码列表，None 表示全部
        """
        if name not in SEED_FACTORS:
            raise ValueError(f"未知因子: {name!r}，可用: {list(SEED_FACTORS.keys())}")

        path = self.cache_dir / f"{name}.parquet"
        if path.exists() and not overwrite:
            print(f"  [缓存] {name} 已存在，跳过。如需重算请用 overwrite=True")
            return self.load_factor(name)

        spec = SEED_FACTORS[name]

        if end_date is None:
            end_df = self.api.query("SELECT MAX(trade_date) AS d FROM daily_prices")
            end_date = str(end_df["d"].iloc[0])[:10]

        label = f"{spec['name']}"
        print(f"\n  [计算] {label}: {spec['description']}")
        print(f"    拉取数据 {start_date} ~ {end_date} ...")

        stock_filter = ""
        if stocks:
            codes = ", ".join(f"'{c}'" for c in stocks)
            stock_filter = f" AND p.ts_code IN ({codes})"

        raw = self.api.query(self._build_adj_close_sql(start_date, end_date, stock_filter))

        if raw.empty:
            raise RuntimeError(f"ClickHouse 返回空数据: {start_date} ~ {end_date}")

        print(f"    原始数据: {len(raw)} 行")

        conn = duckdb.connect()
        conn.register("_raw", raw)

        result = conn.execute(f"""
            SELECT ts_code, trade_date, {spec["sql"]} AS {name}
            FROM _raw
            WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL
            ORDER BY ts_code, trade_date
        """).df()

        conn.close()

        result = result.dropna(subset=[name])
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = result.set_index(["ts_code", "trade_date"])
        result.index.names = ["instrument", "datetime"]
        result = result.astype({name: "float32"})

        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, compression="zstd")
        print(f"    已缓存: {path} ({len(result)} 行)")

        return result

    def extend_factor(self, name: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      stocks: Optional[List[str]] = None) -> pd.DataFrame:
        """
        增量扩展因子：补算新股票或新日期范围，与已有缓存合并去重。

        Args:
            name: 因子名
            start_date: 开始日期，None 表示从已有缓存最新日期+1天
            end_date: 结束日期，None 表示最新
            stocks: 要追加的股票列表，None 时仅补充新日期

        Returns:
            合并后的完整因子 DataFrame
        """
        path = self.cache_dir / f"{name}.parquet"
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()

        new = self.compute_factor(name, start_date=start_date or "2005-01-01",
                                  end_date=end_date, overwrite=False,
                                  stocks=stocks)

        if existing.empty:
            return new
        if new.empty:
            print(f"  [最新] {name} 无需扩展")
            return existing

        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_parquet(path, compression="zstd")
        print(f"    扩展完成: {name} (现共 {len(combined)} 行)")
        return combined

    def rebuild_all(self, start_date: str = "2010-01-01",
                    end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """计算所有注册因子（覆盖已有）。"""
        out = {}
        for name in SEED_FACTORS:
            out[name] = self.compute_factor(name, start_date, end_date, overwrite=True)
        return out

    def append_factor(self, name: str) -> pd.DataFrame:
        """
        增量追加：只计算已有缓存之后的最新数据。

        Returns:
            合并后的完整因子 DataFrame
        """
        path = self.cache_dir / f"{name}.parquet"
        if not path.exists():
            return self.compute_factor(name)

        existing = pd.read_parquet(path)
        latest = existing.index.get_level_values("datetime").max()
        next_day = (pd.to_datetime(latest) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        new = self.compute_factor(name, start_date=next_day)
        if new.empty:
            print(f"  [最新] {name} 已是最新")
            return existing

        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_parquet(path, compression="zstd")
        print(f"    追加完成: {name} (+{len(new)} 行)")
        return combined
