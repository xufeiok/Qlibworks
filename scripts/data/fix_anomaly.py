# -*- coding: utf-8 -*-
"""
修复 warehouse 异常因子：
A类(归一化污染): 删除 10-26/27 行 + auto 重算(全市场口径)
C类(qlib口径残留): 删除 10-28~30 行 + auto 重算(全市场口径)
用法: python scripts/data/fix_anomaly.py --class A|C|both [--list 额外名单文件]
修复前自动备份原 parquet 到 factor_data/_backup_anomaly_YYYYMMDD/
"""
from __future__ import annotations
import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from qlworks.evaluation import DEFAULT_CONFIG, FactorStore
from qlworks.evaluation.warehouse_sync import load_factor_definitions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("fix_anomaly")

WH = Path(r"E:\Quant\Qlibworks\factor_data\warehouse")
A_LIST = r"E:\Quant\Qlibworks\scripts\data\_fix_A.txt"
C_LIST = r"E:\Quant\Qlibworks\scripts\data\_fix_C.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="cls", default="both", choices=["A", "C", "both"])
    args = parser.parse_args()

    store = FactorStore(DEFAULT_CONFIG)
    defs = {it["name"]: it for it in load_factor_definitions(strategy_name="momentum_price_factors")}

    fix_names = []
    if args.cls in ("A", "both"):
        fix_names += [l.strip() for l in open(A_LIST, encoding="utf-8") if l.strip()]
    if args.cls in ("C", "both"):
        fix_names += [l.strip() for l in open(C_LIST, encoding="utf-8") if l.strip()]
    fix_names = sorted(set(fix_names))

    # 备份
    bak_root = Path(r"E:\Quant\Qlibworks\factor_data\_backup_anomaly_20261101")
    bak_root.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for name in fix_names:
        try:
            src = WH / name / "data.parquet"
            if not src.exists():
                logger.warning("跳过 %s: 无 data.parquet", name)
                continue
            # 备份
            bak = bak_root / name
            bak.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(bak / "data.parquet"))
            # 删行
            df = store._load_warehouse_data(name)
            if df is None or df.empty:
                logger.warning("跳过 %s: 仓库为空", name)
                continue
            dates = df.index.get_level_values("datetime")
            if args.cls in ("A", "both"):
                drop = (dates == pd.Timestamp("2026-10-26")) | (dates == pd.Timestamp("2026-10-27"))
                recalc_start = "2026-10-26"
            else:
                drop = dates >= pd.Timestamp("2026-10-28")
                recalc_start = "2026-10-28"
            ndrop = int(drop.sum())
            if ndrop == 0:
                logger.info("跳过 %s: 无待删行", name)
                continue
            df = df[~drop]
            store._write_warehouse_data(name, df)
            store._update_warehouse_meta(name)
            logger.info("[%s] 删除 %d 行 (10-26/27 或 10-28~30) | 余 %d 行", name, ndrop, len(df))

            # 重算
            item = defs.get(name)
            if item is None:
                logger.warning("跳过重算 %s: YAML 无定义", name)
                ok += 1
                continue
            appended = store.append_to_warehouse(
                name=name,
                expr=item["qlib_expr"],
                start_date=recalc_start,
                duckdb_expr=item.get("duckdb_expr"),
                compute_backend="auto",
            )
            store.inject_warehouse_meta(name, item["meta"])
            meta = store.get_warehouse_meta(name)
            logger.info("[%s] 重算完成 +%d 行 | last=%s total=%s", name, appended,
                        meta["data_range"]["last_date"], meta["data_range"]["total_records"])
            ok += 1
        except Exception as exc:
            fail += 1
            logger.error("[%s] 失败: %s", name, exc)
    logger.info("修复完成：成功=%d 失败=%d", ok, fail)


if __name__ == "__main__":
    main()
