#!/usr/bin/env python3
"""
用 Qlib 原生计算剩余的复杂 Alpha 因子。

使用方式：
  cd e:\Quant\Qlibworks
  E:\Conda_env\Qlib_env\python.exe scripts\evaluation\compute_alphas_qlib.py
"""
import sys, logging, warnings, json
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("qlib_alpha")

import yaml, pandas as pd, numpy as np
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR

# 初始化 Qlib
qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN,
          joblib_backend="threading", maxtasksperchild=1)
from qlib.config import C as _QC
_QC.dataloader_workers = 0

# 读取股票池
ifile = Path(str(QLIB_DATA_DIR)) / "instruments" / "all.txt"
pool = []
with open(ifile) as f:
    for line in f:
        p = line.strip().split()
        if p:
            pool.append(p[0])
logger.info(f"股票池: {len(pool)} 只")

YAML_PATH = Path(r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml")
WAREDIR = Path(r"e:\Quant\Qlibworks\factor_data\warehouse")
START = "2010-01-01"
END = "2026-12-31"


def get_missing():
    """获取缺失的复杂 Alpha 因子。"""
    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    missing = []
    for fd in data.get("factors", []):
        name = fd.get("name", "")
        if not name:
            continue
        fdir = WAREDIR / name
        if fdir.is_dir() and list(fdir.glob("*.parquet")):
            continue
        expr = fd.get("expression", {}).get("qlib", "")
        if expr:
            missing.append((name, str(expr)))
    return missing


def save_to_warehouse(name, df):
    """将结果按年保存到 warehouse。"""
    fdir = WAREDIR / name
    fdir.mkdir(parents=True, exist_ok=True)

    df = df.reset_index()
    df["_year"] = pd.to_datetime(df["datetime"]).dt.year
    total_rows = 0
    for year, grp in df.groupby("_year"):
        year_df = grp.drop(columns=["_year"]).set_index(["instrument", "datetime"])
        year_df = year_df.sort_index()
        pf = fdir / f"{year}.parquet"
        year_df.to_parquet(pf, compression="zstd")
        total_rows += len(year_df)

    # 写 meta
    meta = {
        "factor_name": name,
        "total_rows": total_rows,
        "data_range": {
            "start_date": str(df["datetime"].min().date()),
            "last_date": str(df["datetime"].max().date()),
        },
    }
    with open(fdir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return total_rows


def compute_alpha(name, expr):
    """用 Qlib D.features() 计算单个因子。"""
    logger.info(f"计算 {name}: {expr[:60]}...")

    batch_size = 500
    all_parts = []

    for i in range(0, len(pool), batch_size):
        batch = pool[i:i + batch_size]
        for ck_s, ck_e in [("2010-01-01", "2016-12-31"), ("2017-01-01", "2021-12-31"),
                            ("2022-01-01", "2026-12-31")]:
            try:
                df = D.features(batch, [expr], ck_s, ck_e)
            except Exception as e:
                logger.warning(f"  chunk {ck_s}~{ck_e} 失败: {str(e)[:60]}")
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
        raise RuntimeError("Qlib 返回空数据")

    r = pd.concat(all_parts, ignore_index=True)
    r = r.set_index(["instrument", "datetime"])[["value"]]
    r = r.sort_index().astype({"value": "float32"})
    return r


def main():
    missing = get_missing()
    logger.info(f"缺失因子数: {len(missing)}")

    succeeded = 0
    failed = 0

    for idx, (name, expr) in enumerate(missing, 1):
        logger.info(f"[{idx}/{len(missing)}] {name}")
        try:
            result = compute_alpha(name, expr)
            total = save_to_warehouse(name, result)
            logger.info(f"  OK {total:,} 行")
            succeeded += 1
        except Exception as e:
            logger.error(f"  FAIL: {e}")
            failed += 1

    logger.info(f"\n完成 -> 成功: {succeeded}, 失败: {failed}")


if __name__ == "__main__":
    main()
