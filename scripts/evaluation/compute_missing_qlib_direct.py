#!/usr/bin/env python3
"""
用 Qlib D.features() 计算剩余的复杂 Alpha 因子。

表达式修复（Qlib Windows 兼容）:
  Rank(X) -> Rank(X, 1)    # 横截面排名必须带 N
  Ts_Max(X,N) -> Max(X,N)  # Ts_Max 未注册
  Ts_Min(X,N) -> Min(X,N)  # Ts_Min 未注册
  Ts_Sum(X,N) -> Sum(X,N)  # Ts_Sum 未注册

使用方式:
  cd e:\Quant\Qlibworks
  E:\Conda_env\Qlib_env\python.exe scripts\evaluation\compute_missing_qlib_direct.py
"""
import sys, warnings, json, re, logging
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("qlib_direct")

import yaml, pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR

qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN,
          joblib_backend="threading", maxtasksperchild=1)
from qlib.config import C as _QC
_QC.dataloader_workers = 0

YAML_PATH = Path(r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml")
WAREDIR = Path(r"e:\Quant\Qlibworks\factor_data\warehouse")
START = "2010-01-01"
END = "2026-12-31"

# 股票池
pool = [p.strip().split()[0] for p in open(str(QLIB_DATA_DIR)+"/instruments/all.txt") if p.strip()]
logger.info(f"股票池: {len(pool)} 只")


def _extract_func_args(expr: str, func_name: str):
    """提取函数参数（支持嵌套括号）。"""
    pattern = re.compile(rf'\b{func_name}\(')
    for m in pattern.finditer(expr):
        start = m.start()
        depth = 1
        i = m.end()
        args = []
        cur = i
        while depth > 0 and i < len(expr):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    args.append(expr[cur:i].strip())
            elif expr[i] == ',' and depth == 1:
                args.append(expr[cur:i].strip())
                cur = i + 1
            i += 1
        if depth == 0:
            yield (start, i, args)


def _split_top_level_args(s: str) -> list:
    """按顶层逗号分割函数参数（忽略括号内的逗号）。"""
    args = []
    depth = 0
    cur = []
    for ch in s:
        if ch == ',' and depth == 0:
            args.append(''.join(cur).strip())
            cur = []
        else:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            cur.append(ch)
    args.append(''.join(cur).strip())
    return args


def _fix_rank_recursive(s: str) -> str:
    """递归修复 Rank(X) -> Rank(X,1)，从内到外处理。

    遇到 Rank(...) 时，先递归修复内部表达式中的 Rank 调用，
    再根据修复后的参数数量决定是否补 ,1。
    """
    i = 0
    result = []
    while i < len(s):
        idx = s.find('Rank(', i)
        if idx == -1:
            result.append(s[i:])
            break
        # 添加 Rank 前面的内容
        result.append(s[i:idx])
        # 定位匹配的右括号（支持括号嵌套）
        depth = 1
        j = idx + 5
        while j < len(s) and depth > 0:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
            j += 1
        # 递归修复 Rank 内部表达式
        inner = s[idx + 5:j - 1]
        fixed_inner = _fix_rank_recursive(inner)
        # 判断是否需要补 ,1
        args = _split_top_level_args(fixed_inner)
        if len(args) == 1:
            result.append(f"Rank({args[0]}, 1)")
        else:
            result.append(f"Rank({fixed_inner})")
        i = j
    return ''.join(result)


def fix_expr(expr: str) -> str:
    """修复 Qlib 表达式使其兼容 Windows 版 Qlib。

    Rank(X) -> Rank(X, 1)  # 给无 N 参数的 Rank 添加 N=1
    Ts_Max  -> Max         # Ts_Max/Ts_Min/Ts_Sum 未注册
    Ts_Min  -> Min
    Ts_Sum  -> Sum
    """
    e = expr
    # Ts_Max/Ts_Min/Ts_Sum -> Max/Min/Sum（无偏移问题）
    for fn, repl in [('Ts_Max', 'Max'), ('Ts_Min', 'Min'), ('Ts_Sum', 'Sum')]:
        e = e.replace(fn, repl)
    # Rank(X) -> Rank(X,1)：递归从内到外处理，无位置偏移问题
    e = _fix_rank_recursive(e)
    return e


def get_missing():
    """获取缺失因子。"""
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
    """按年保存到 warehouse。"""
    fdir = WAREDIR / name
    fdir.mkdir(parents=True, exist_ok=True)
    df = df.reset_index()
    df["_year"] = pd.to_datetime(df["datetime"]).dt.year
    total = 0
    for year, grp in df.groupby("_year"):
        yd = grp.drop(columns=["_year"]).set_index(["instrument", "datetime"]).sort_index()
        n = len(yd)
        yd.to_parquet(fdir / f"{year}.parquet", compression="zstd")
        total += n
    meta = {"factor_name": name, "total_rows": total,
            "data_range": {"start_date": str(df["datetime"].min().date()),
                           "last_date": str(df["datetime"].max().date())}}
    with open(fdir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return total


def compute_one(name, expr, batch_size=500):
    """用 Qlib D.features() 计算单个因子。"""
    fixed = fix_expr(expr)
    logger.info(f"  Qlib 原始: {expr[:60]}...")
    logger.info(f"  修复后: {fixed[:60]}...")

    chunks = [("2010-01-01", "2012-12-31"), ("2013-01-01", "2015-12-31"),
              ("2016-01-01", "2018-12-31"), ("2019-01-01", "2021-12-31"),
              ("2022-01-01", "2026-12-31")]

    all_parts = []
    for ck_s, ck_e in chunks:
        for i in range(0, len(pool), batch_size):
            batch = pool[i:i + batch_size]
            try:
                df = D.features(batch, [fixed], ck_s, ck_e)
            except Exception:
                continue
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns={fixed: "value"})
            df = df.reset_index()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["instrument"] = df["instrument"].astype(str)
            df = df.dropna(subset=["value"])
            all_parts.append(df)

    if not all_parts:
        raise RuntimeError("Qlib 返回空数据")

    r = pd.concat(all_parts, ignore_index=True)
    r = r.set_index(["instrument", "datetime"])[["value"]].sort_index()
    r = r.astype({"value": "float32"})
    return r


def main():
    missing = get_missing()
    logger.info(f"缺失因子数: {len(missing)}")

    ok = 0
    fail = 0
    for idx, (name, expr) in enumerate(missing, 1):
        logger.info(f"[{idx}/{len(missing)}] {name}")
        try:
            result = compute_one(name, expr)
            total = save_to_warehouse(name, result)
            logger.info(f"  OK {total:,} rows")
            ok += 1
        except Exception as e:
            logger.error(f"  FAIL: {e}")
            fail += 1

    logger.info(f"\n完成 -> OK: {ok}, FAIL: {fail}")


if __name__ == "__main__":
    main()
