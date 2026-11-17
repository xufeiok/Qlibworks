# -*- coding: utf-8 -*-
"""
补跑 warehouse 中 61 个未更新因子（非 momentum_price_factors 类别）。
复用 FactorStore.append_to_warehouse，auto 后端（DuckDB 优先 → 财务 forward-fill → Qlib 兜底）。
"""
import sys, os, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("backfill_other_categories")

from qlworks.evaluation import DEFAULT_CONFIG, FactorStore
from qlworks.evaluation.warehouse_sync import load_factor_definitions, resolve_append_start_date

WH = r"E:\Quant\Qlibworks\factor_data\warehouse"
STRATEGIES = ["quality_factors", "value_factors", "risk_factors", "growth_factors", "other_factors", "sentiment_factors"]

def stale_names():
    names = []
    for d in os.listdir(WH):
        p = os.path.join(WH, d, "meta.json")
        if os.path.isdir(os.path.join(WH, d)) and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            last = (m.get("data_range") or {}).get("last_date", "")
            if last < "2026-10-30":
                names.append(d)
    return set(names)

def main():
    store = FactorStore(DEFAULT_CONFIG)
    targets = stale_names()
    logger.info("待更新因子数: %d", len(targets))

    total_ok, total_fail, total_rows = 0, 0, 0
    for strat in STRATEGIES:
        defs = load_factor_definitions(strategy_name=strat)
        cands = [it for it in defs if it["name"] in targets]
        if not cands:
            logger.info("== %s: 无待更新因子 ==", strat)
            continue
        logger.info("== %s: 待更新 %d ==", strat, len(cands))
        for item in cands:
            name = item["name"]
            expr = item["qlib_expr"]
            meta = store.get_warehouse_meta(name)
            start_date = resolve_append_start_date(meta=meta, default_start_date=store.config.start_time)
            try:
                appended = store.append_to_warehouse(
                    name=name, expr=expr, start_date=start_date,
                    duckdb_expr=item["duckdb_expr"], compute_backend="auto",
                )
                store.inject_warehouse_meta(name, item["meta"])
                total_ok += 1; total_rows += appended
                logger.info("    OK %s +%d 行", name, appended)
            except Exception as exc:
                total_fail += 1
                logger.error("    FAIL %s: %s", name, exc)
    logger.info("完成：成功=%d | 失败=%d | 新增行数=%d", total_ok, total_fail, total_rows)

if __name__ == "__main__":
    main()
