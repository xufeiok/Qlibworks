#!/usr/bin/env python3
"""
因子批量计算与增量追加脚本。

用法：
  # 首次批量计算：从 YAML 因子库计算所有因子并存入仓库
  python batch_compute.py --all --start 2018-01-01 --end 2025-12-31

  # 计算指定分类下的因子
  python batch_compute.py --category price_volume_factors

  # 计算单个因子
  python batch_compute.py --factor KDJ_K

  # 增量追加最新数据（每周/每月执行）
  python batch_compute.py --all --append

  # 查看仓库状态
  python batch_compute.py --status
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

# 路径处理
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_compute")

from qlworks.evaluation import FactorStore, EvalConfig, DEFAULT_CONFIG
from qlworks.factors import FactorLibraryManager


def _collect_factors(args):
    """从 YAML 因子库收集因子列表。"""
    m = FactorLibraryManager()
    factors = []

    if args.factor:
        info = _find_factor(m, args.factor)
        if info:
            factors.append(info)
    elif args.category:
        cfg = m.load_strategy_config(args.category)
        for fd in cfg.get("factors", []):
            name = fd.get("name")
            if not name:
                continue
            expr = fd.get("expression", "")
            duckdb_expr = ""
            if isinstance(expr, dict):
                duckdb_expr = str(expr.get("duckdb", ""))
                expr = str(expr.get("qlib", str(expr)))
            factors.append({"name": name, "expr": str(expr), "duckdb_expr": duckdb_expr})
    elif args.all:
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            try:
                cfg = m.load_strategy_config(s)
                for fd in cfg.get("factors", []):
                    name = fd.get("name")
                    if not name:
                        continue
                    expr = fd.get("expression", "")
                    duckdb_expr = ""
                    if isinstance(expr, dict):
                        duckdb_expr = str(expr.get("duckdb", ""))
                        expr = str(expr.get("qlib", str(expr)))
                    factors.append({"name": name, "expr": str(expr), "duckdb_expr": duckdb_expr})
            except Exception:
                continue

    return factors


def _find_factor(m, name):
    """在因子库中查找单个因子。"""
    for s in [s for s in m.list_strategies() if "dictionary" not in s]:
        try:
            cfg = m.load_strategy_config(s)
            for fd in cfg.get("factors", []):
                if fd.get("name") == name:
                    expr = fd.get("expression", "")
                    duckdb_expr = ""
                    if isinstance(expr, dict):
                        duckdb_expr = str(expr.get("duckdb", ""))
                        expr = str(expr.get("qlib", str(expr)))
                    return {"name": name, "expr": str(expr), "duckdb_expr": duckdb_expr}
        except Exception:
            continue
    return None


def show_warehouse_status(store):
    """显示仓库状态。"""
    factors = store.list_warehouse_factors()
    if not factors:
        print("\n  仓库为空，请先运行 --all 批量计算。")
        return

    print(f"\n  {'因子名':30s} {'年份':20s} {'总行数':>10s}  {'最后日期':14s}")
    print(f"  {'-'*30} {'-'*20} {'-'*10}  {'-'*14}")
    for name in factors:
        meta = store.get_warehouse_meta(name)
        if meta:
            years = ",".join(str(y) for y in (meta.get("years") or []))
            total = meta.get("total_rows", 0)
            last = meta.get("last_date", "")
            print(f"  {name:30s} {years:20s} {total:>10,}  {last:14s}")
    print(f"\n  共 {len(factors)} 个因子")


def main():
    parser = argparse.ArgumentParser(description="因子批量计算与增量追加")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--factor", "-f", help="单个因子名")
    g.add_argument("--category", "-c", help="分类 YAML 文件")
    g.add_argument("--all", "-a", action="store_true", help="所有因子")
    g.add_argument("--status", "-s", action="store_true", help="查看仓库状态")
    parser.add_argument("--append", action="store_true", help="增量追加（默认全量计算）")
    parser.add_argument("--start", default="2018-01-01", help="开始日期")
    parser.add_argument("--end", default="2025-12-31", help="结束日期")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有数据")
    args = parser.parse_args()

    # 使用全局默认配置，确保仓库/缓存目录落在项目统一路径下。
    config = DEFAULT_CONFIG
    store = FactorStore(config)

    # ── 只看状态 ──
    if args.status:
        show_warehouse_status(store)
        return

    # ── 收集因子 ──
    factors = _collect_factors(args)
    if not factors:
        print("未找到因子，请检查 YAML 因子库。")
        return

    print(f"\n{'='*60}")
    print(f"{'因子仓库批量计算':^60}")
    print(f"{'='*60}")
    print(f"  因子数: {len(factors)}")
    print(f"  时间:   {args.start} ~ {args.end}")
    print(f"  模式:   {'增量追加' if args.append else '全量计算'}")
    print(f"{'='*60}")

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for i, f in enumerate(factors, 1):
        name = f["name"]
        expr = f["expr"]
        duckdb_expr = f.get("duckdb_expr", "")
        try:
            if args.append:
                # [Bloomberg Data Pipeline] 增量追加：只算新数据
                n = store.append_to_warehouse(name, expr, args.start, duckdb_expr=duckdb_expr)
                status = f"+{n}行" if n else "已最新"
                print(f"  [{i}/{len(factors)}] {name:25s} {status}")
                total_ok += 1
            else:
                # 全量计算
                stats = store.compute_to_warehouse(
                    name, expr, args.start, args.end,
                    overwrite=args.overwrite,
                    duckdb_expr=duckdb_expr,
                )
                if stats:
                    yr = ",".join(f"{y}({n})" for y, n in stats.items())
                    print(f"  [{i}/{len(factors)}] {name:25s} 已计算: {yr}")
                    total_ok += 1
                else:
                    print(f"  [{i}/{len(factors)}] {name:25s} 已存在，跳过(加 --overwrite 覆盖)")
                    total_skip += 1
        except Exception as e:
            print(f"  [{i}/{len(factors)}] {name:25s} X 失败: {e}")
            total_fail += 1

    print(f"\n{'='*60}")
    print(f"  完成: {total_ok} 成功, {total_skip} 跳过, {total_fail} 失败")
    print(f"{'='*60}")

    show_warehouse_status(store)


if __name__ == "__main__":
    main()
