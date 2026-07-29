#!/usr/bin/env python3
"""
用 Qlib 原生计算剩余的复杂 Alpha 因子（跳过 DuckDB 转换）。

使用方式（在 Qlib_env 中运行）：
  cd e:\Quant\Qlibworks
  E:\Conda_env\Qlib_env\python.exe scripts\evaluation\compute_missing_qlib.py
"""
import sys, logging, warnings, os
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compute_qlib")

os.environ["FORCE_QLIB"] = "true"

import yaml
import pandas as pd
import numpy as np
from qlworks.evaluation import FactorStore, DEFAULT_CONFIG

YAML_PATH = Path(r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml")
WAREHOUSE_DIR = Path(r"e:\Quant\Qlibworks\factor_data\warehouse")
START = "2010-01-01"
END = "2026-12-31"


def get_missing_factors():
    """获取缺失的因子列表及其 Qlib 表达式。"""
    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    missing = []
    for fd in data.get("factors", []):
        name = fd.get("name", "")
        if not name:
            continue
        fdir = WAREHOUSE_DIR / name
        if fdir.is_dir() and list(fdir.glob("*.parquet")):
            continue
        expr = fd.get("expression", {}).get("qlib", "")
        if expr:
            missing.append((name, str(expr)))

    return missing


def main():
    store = FactorStore(DEFAULT_CONFIG)

    # 初始化 Qlib（确保在 Qlib_env 中）
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.config import QLIB_DATA_DIR

        qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN,
                  joblib_backend="threading", maxtasksperchild=1)
        from qlib.config import C as _QC
        _QC.dataloader_workers = 0
        _QC.joblib_backend = "threading"
        _QC.maxtasksperchild = 1
        logger.info("Qlib 初始化成功")
    except Exception as e:
        logger.error(f"Qlib 初始化失败: {e}")
        return

    missing = get_missing_factors()
    logger.info(f"共 {len(missing)} 个缺失因子")

    # 获取全市场股票列表
    ifile = Path(str(QLIB_DATA_DIR)) / "instruments" / "all.txt"
    pool = []
    if ifile.exists():
        with open(ifile, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if p:
                    pool.append(p[0])
    logger.info(f"股票池: {len(pool)} 只")

    if not pool:
        logger.error("无法加载股票池")
        return

    # 分批计算（每次 500 只股票）
    batch_size = 500
    succeeded = 0
    failed = 0

    for idx, (name, expr) in enumerate(missing, 1):
        logger.info(f"[{idx}/{len(missing)}] {name}: {expr[:60]}...")

        try:
            all_parts = []
            for i in range(0, len(pool), batch_size):
                batch = pool[i:i + batch_size]
                try:
                    df = D.features(batch, [expr], START, END)
                except Exception as e:
                    logger.warning(f"  批次 {i//batch_size+1} 失败: {e}")
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
                logger.error(f"  ❌ Qlib 返回空数据")
                failed += 1
                continue

            r = pd.concat(all_parts, ignore_index=True)
            r = r.set_index(["instrument", "datetime"])[["value"]]
            r = r.sort_index().astype({"value": "float32"})

            # 按年保存到 warehouse
            r = r.reset_index()
            r["_year"] = pd.to_datetime(r["datetime"]).dt.year
            fdir = WAREHOUSE_DIR / name
            fdir.mkdir(parents=True, exist_ok=True)

            for year, grp in r.groupby("_year"):
                year_df = grp.drop(columns=["_year"]).set_index(["instrument", "datetime"])
                year_df = year_df.sort_index()
                pf = fdir / f"{year}.parquet"
                year_df.to_parquet(pf, compression="zstd")

            # 写 meta
            meta = {
                "factor_name": name,
                "total_rows": len(r),
                "data_range": {
                    "start_date": str(r["datetime"].min().date()),
                    "last_date": str(r["datetime"].max().date()),
                },
            }
            import json
            with open(fdir / "meta.json", "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"  ✅ {len(r):,} 行")
            succeeded += 1

        except Exception as e:
            logger.error(f"  ❌ {e}")
            failed += 1

    logger.info(f"\n完成→ 成功: {succeeded}, 失败: {failed}")


if __name__ == "__main__":
    main()
