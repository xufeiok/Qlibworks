#!/usr/bin/env python3
"""
基于本地 Qlib 的 warehouse 增量更新脚本。

用途：
- 在 ClickHouse 不可写、但本地 Qlib 已更新到最新收盘日时
- 将 YAML 因子库中的因子增量续算到 factor_data/warehouse
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlworks.evaluation import DEFAULT_CONFIG, FactorStore
from qlworks.evaluation.warehouse_sync import load_factor_definitions, resolve_append_start_date


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("update_warehouse_from_local_qlib")


def parse_args():
    parser = argparse.ArgumentParser(description="基于本地 Qlib 增量更新 warehouse")
    parser.add_argument("--strategy", type=str, default="selected_good_factors", help="因子 YAML 文件名，不含 .yaml")
    parser.add_argument("--repo-path", type=str, default=None, help="因子库目录，默认项目 factor_data/factor_library")
    parser.add_argument("--start-date", type=str, default=None, help="显式起始日期；默认按各因子 meta 推断")
    parser.add_argument("--factor", nargs="*", default=None, help="只更新指定因子名，可多个")
    parser.add_argument("--skip-meta", action="store_true", help="跳过向 warehouse 注入 YAML 元数据")
    return parser.parse_args()


def main():
    args = parse_args()
    store = FactorStore(DEFAULT_CONFIG)
    factor_defs = load_factor_definitions(strategy_name=args.strategy, repo_path=args.repo_path)

    if args.factor:
        requested = set(args.factor)
        factor_defs = [item for item in factor_defs if item["name"] in requested]

    if not factor_defs:
        logger.warning("没有找到可更新的因子定义。")
        return

    logger.info("策略=%s | 因子数=%d", args.strategy, len(factor_defs))

    success = 0
    failed = 0
    total_rows = 0
    for idx, item in enumerate(factor_defs, 1):
        name = item["name"]
        expr = item["qlib_expr"]
        meta = store.get_warehouse_meta(name)
        start_date = args.start_date or resolve_append_start_date(meta=meta, default_start_date=store.config.start_time)

        try:
            logger.info("[%d/%d] %s | start=%s", idx, len(factor_defs), name, start_date)
            appended = store.append_to_warehouse(
                name=name,
                expr=expr,
                start_date=start_date,
                duckdb_expr=None,
                compute_backend="qlib",
            )
            if not args.skip_meta:
                store.inject_warehouse_meta(name, item["meta"])
            total_rows += appended
            success += 1
        except Exception as exc:
            failed += 1
            logger.error("[%d/%d] %s 失败: %s", idx, len(factor_defs), name, exc)

    logger.info("完成：成功=%d | 失败=%d | 新增行数=%d", success, failed, total_rows)


if __name__ == "__main__":
    main()
