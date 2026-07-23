"""
warehouse 增量同步辅助工具

职责:
- 解析因子 YAML，提取适合本地 Qlib 续算的因子定义
- 统一推导因子仓库的增量起始日
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd

from qlworks.factors.manager import FactorLibraryManager


def resolve_append_start_date(meta: Optional[dict], default_start_date: str) -> str:
    """
    根据 warehouse meta 解析增量起始日。
    """
    if meta and meta.get("data_range", {}).get("last_date"):
        last_date = pd.Timestamp(meta["data_range"]["last_date"]).normalize()
        return (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    return default_start_date


def load_factor_definitions(strategy_name: str, repo_path: str | None = None) -> list[dict]:
    """
    从 YAML 因子库读取因子定义，只保留带 qlib 表达式的因子。

    输出字段:
    - name
    - qlib_expr
    - duckdb_expr
    - meta
    """
    manager = FactorLibraryManager(repo_path=repo_path)
    config = manager.load_strategy_config(strategy_name)

    factors = []
    for factor in config.get("factors", []):
        name = factor.get("name")
        expr = factor.get("expression", {})
        qlib_expr = expr.get("qlib", "") if isinstance(expr, dict) else str(expr or "")
        duckdb_expr = expr.get("duckdb", "") if isinstance(expr, dict) else ""
        if not name or not qlib_expr:
            continue
        factors.append(
            {
                "name": name,
                "qlib_expr": str(qlib_expr),
                "duckdb_expr": str(duckdb_expr or ""),
                "meta": factor,
            }
        )
    return factors
