#!/usr/bin/env python3
"""
纯因子排序打分脚本（替代 ML 集成方案）

策略：
  1. 用 t-stat 最强的 5 个因子
  2. D.features() 计算价格类因子 + warehouse 直读财务类因子
  3. 每日截面百分位排序 + 等权组合
  4. 输出与 tree_label_test.py 兼容的 score CSV

用法：
  C:\\xfworks\\Qlib_venv\\Scripts\\python.exe factor_rank_score.py
  输出：score_factor_rank.csv
"""

import os, sys, warnings, argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import qlib
from qlib.data import D
from qlib.config import REG_CN
from qlworks.config import QLIB_DATA_DIR

QLIB_DATA_PATH = Path(QLIB_DATA_DIR)
FACTOR_DATA_PATH = Path(r"C:\xfworks\Qlibworks\factor_data")

warnings.filterwarnings("ignore")

START_DATE = "2023-01-01"
END_DATE = "2026-06-29"
OUTPUT_FILE = "score_factor_rank.csv"

# ── 全部 5 个因子 ──
FACTOR_NAMES = ["EXTREME_REV", "pe_ttm", "roe_ttm", "STR_5d", "illiquidity_amihud"]

# D.features() 可算的因子
FACTOR_EXPRESSIONS = {
    "EXTREME_REV": "If(Abs($close/Ref($close,1)-1)>0.095, -($close/Ref($close,1)-1), 0)",
    "illiquidity_amihud": "Abs($close/Ref($close,1)-1)/($amount*1e6+1)",
}

# warehouse 直读的因子
WAREHOUSE_FACTORS = {"pe_ttm", "roe_ttm", "STR_5d"}

# 方向：1=正向（值越大越好），-1=反向
FACTOR_DIRECTIONS = {
    "EXTREME_REV": 1,
    "pe_ttm": -1,
    "roe_ttm": 1,
    "STR_5d": -1,
    "illiquidity_amihud": 1,
}


def read_warehouse_factor(factor_name: str, start: str, end: str) -> pd.DataFrame:
    """从 warehouse parquet 直读，返回 (date×instrument) 宽表"""
    wh_dir = FACTOR_DATA_PATH / "warehouse" / factor_name
    parts = sorted(wh_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "date" in cl:
            col_map[c] = "date"
        elif "code" in cl or "instrument" in cl:
            col_map[c] = "instrument"
    df = df.rename(columns=col_map)

    value_cols = [c for c in df.columns if c not in ("date", "instrument")]
    vc = value_cols[0]
    df = df[["date", "instrument", vc]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str).str.lower()

    mask = (df["date"] >= start) & (df["date"] <= end)
    df = df[mask]
    return df.pivot_table(index="date", columns="instrument", values=vc, aggfunc="first")


def compute_qlib_factor(expr: str, start: str, end: str) -> pd.DataFrame:
    """用 D.features() 计算因子，返回 (date×instrument) 宽表"""
    inst = D.instruments("main_board")
    df = D.features(inst, [expr], start_time=start, end_time=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    col = df.columns[0]
    df = df[col].unstack(level="instrument").astype(float)
    return df


def main():
    parser = argparse.ArgumentParser(description="纯因子排序打分")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    print("=" * 60)
    print("  纯因子排序打分（Factor Rank Score）")
    print("=" * 60)
    print(f"  因子: {', '.join(FACTOR_NAMES)}")
    print()

    qlib.init(provider_uri=str(QLIB_DATA_PATH).replace("\\", "/"), region=REG_CN)

    factor_data = {}
    for fname in FACTOR_NAMES:
        direction = FACTOR_DIRECTIONS.get(fname, 1)
        print(f"  [{fname}]...", end=" ", flush=True)
        try:
            if fname in WAREHOUSE_FACTORS:
                df = read_warehouse_factor(fname, args.start, args.end)
                print(f"warehouse {df.shape}", end="")
            elif fname in FACTOR_EXPRESSIONS:
                df = compute_qlib_factor(FACTOR_EXPRESSIONS[fname], args.start, args.end)
                print(f"D.features {df.shape}", end="")
            else:
                print(f"✗ 未知因子", end="")
                continue

            rank_df = df.rank(axis=1, pct=True, na_option="keep")
            if direction == -1:
                rank_df = 1.0 - rank_df
            factor_data[fname] = rank_df
            print(" ✓")
        except Exception as e:
            print(f"✗ {e}")

    if len(factor_data) < 3:
        print(f"\n  错误: 仅成功加载 {len(factor_data)} 个因子 (<3)")
        sys.exit(1)

    # 等权组合
    print(f"\n  等权组合 {len(factor_data)} 个因子排名...")
    combined = sum(factor_data.values()) / len(factor_data)
    valid_dates = combined.dropna(how="all").index
    print(f"  共同交易日: {len(valid_dates)} 天")

    # 转换 score CSV
    print("  转换 score CSV...")
    scores = combined.stack().dropna().reset_index()
    scores.columns = ["datetime", "instrument", "score"]
    scores["datetime"] = scores["datetime"].astype(str)
    scores = scores.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / args.output
    scores.to_csv(output_path, index=False)

    print(f"\n  总记录: {len(scores):,}")
    print(f"  score 均值: {scores['score'].mean():.4f}")
    print(f"  score 标准差: {scores['score'].std():.4f}")
    print(f"  日期范围: {scores['datetime'].min()} ~ {scores['datetime'].max()}")
    print(f"  股票数: {scores['instrument'].nunique()}")
    print(f"  ➤ 保存至: {output_path}")


if __name__ == "__main__":
    main()
