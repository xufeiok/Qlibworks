#!/usr/bin/env python3
"""
单因子评测入口脚本 — 仓库优先（v3.0）

数据流：
  1. 首次：batch_compute.py --all  → 将因子写入 warehouse/
  2. 每周：batch_compute.py --all --append → 增量追加新数据
  3. 评测：run_eval.py --factor KDJ_K → 从 warehouse 读取，只做评测

用法：
  # 评测单个因子（从 warehouse 读取数据）
  python run_eval.py --factor KDJ_K

  # 评测某个分类
  python run_eval.py --category price_volume_factors

  # 评测仓库中所有因子
  python run_eval.py --all

  # 自定义参数
  python run_eval.py --factor KDJ_K --pool csi500 --start 2020-01-01

  # 仅查看因子列表（不评测）
  python run_eval.py --status

  # 强制从 ClickHouse 实时计算并写入仓库（不读取缓存）
  python run_eval.py --factor KDJ_K --recompute
"""
import argparse
from dataclasses import replace
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")

from qlworks.evaluation import FactorEvaluator, FactorStore, DEFAULT_CONFIG
from qlworks.factors import FactorLibraryManager


def _collect_factors(args):
    """从 YAML 因子库收集因子列表（含 duckdb_expr 用于自动计算）。"""
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
            factors.append({"name": name, "expr": str(expr),
                           "duckdb_expr": duckdb_expr, "category": args.category})
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
                    factors.append({"name": name, "expr": str(expr),
                                   "duckdb_expr": duckdb_expr, "category": s})
            except Exception:
                pass

    return factors


def _find_factor(m, name):
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
                    return {"name": name, "expr": str(expr),
                            "duckdb_expr": duckdb_expr, "category": s}
        except Exception:
            pass
    return None


def _meta_val(meta: dict, key: str, default=None):
    """兼容新旧 meta 格式读取辅助函数。"""
    # 新格式在 data_range 下
    dr = meta.get("data_range", {}) if meta else {}
    if key in dr and dr[key]:
        return dr[key]
    # 旧格式在顶层
    if meta and key in meta and meta[key]:
        return meta[key]
    return default


def show_status(store):
    """显示仓库中所有因子的状态。"""
    factors = store.list_warehouse_factors()
    print(f"\n  仓库中有 {len(factors)} 个因子:")
    print(f"  {'因子名':30s} {'年份':20s} {'总行数':>10s}  {'最后日期':14s}  {'等级':12s}")
    print(f"  {'-'*30} {'-'*20} {'-'*10}  {'-'*14}  {'-'*12}")
    for name in factors:
        meta = store.get_warehouse_meta(name)
        eval_meta = store.get_evaluated_meta(name)
        tier = eval_meta.get("tier", "未评测") if eval_meta else "未评测"
        if meta:
            years_list = _meta_val(meta, "years", [])
            years = ",".join(str(y) for y in (years_list or []))
            total = _meta_val(meta, "total_records", 0) or _meta_val(meta, "total_rows", 0)
            last = _meta_val(meta, "last_date", "")
            print(f"  {name:30s} {years:20s} {total:>10,}  {last:14s}  {tier:12s}")
        else:
            print(f"  {name:30s} {'无元数据':20s}")
    print()


def main():
    parser = argparse.ArgumentParser(description="单因子评测系统（仓库优先 v3.0）")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--factor", "-f", help="指定单因子名称")
    g.add_argument("--category", "-c", help="指定分类 YAML 文件")
    g.add_argument("--all", "-a", action="store_true", help="评测仓库中所有因子")
    g.add_argument("--status", "-s", action="store_true", help="查看仓库状态")
    g.add_argument("--demo", action="store_true", help="演示模式")
    parser.add_argument("--pool", default="csi500")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--recompute", action="store_true",
                        help="强制从 ClickHouse 实时计算并写入仓库（跳过仓库缓存）")
    parser.add_argument("--walk-forward", action="store_true", help="启用 Walk-Forward 滚动外推验证")
    parser.add_argument("--dry-run", action="store_true", help="仅列出因子，不评测")
    args = parser.parse_args()

    if not any([args.factor, args.category, args.all, args.demo, args.status]):
        parser.print_help()
        return

    store = FactorStore()
    config = replace(
        DEFAULT_CONFIG,
        instruments=args.pool,
        start_time=args.start,
        end_time=args.end,
    )

    # ── 状态模式 ──
    if args.status:
        show_status(store)
        return

    # ── 收集因子 ──
    factors = _collect_factors(args)

    if args.dry_run:
        print(f"共 {len(factors)} 个因子:")
        for f in factors:
            print(f"  {f['name']:30s} {f['expr'][:60]}")
        return

    # ── 实例化评测器 ──
    evaluator = FactorEvaluator(config)

    # ── 执行评测 ──
    for i, f in enumerate(factors, 1):
        name = f["name"]
        expr = f["expr"]
        print(f"\n[{i}/{len(factors)}] {name}")

        try:
            # [Citadel Alpha Lab] 仓库优先：先确保数据在仓库中
            if args.recompute or not store.get_warehouse_meta(name):
                duckdb_expr = f.get("duckdb_expr", "")
                logger.info(f"{name} 仓库无数据或强制重算，从 ClickHouse 计算...")
                store.compute_to_warehouse(name, expr, args.start, args.end,
                                           duckdb_expr=duckdb_expr)
            else:
                warehouse_meta = store.get_warehouse_meta(name)
                wh_start = _meta_val(warehouse_meta, "first_date", "")
                wh_end = _meta_val(warehouse_meta, "last_date", "")
                total_rows_val = _meta_val(warehouse_meta, "total_records", 0) or _meta_val(warehouse_meta, "total_rows", 0)
                logger.info(f"{name} 从仓库读取 ({wh_start} ~ {wh_end}, {total_rows_val:,} 行)")

            # 从仓库加载数据（仓库列名为 value，需重命名为因子名）
            df = store.load_from_warehouse(name, args.start, args.end)
            if df is None or df.empty:
                logger.warning(f"{name} 无数据，跳过")
                continue
            df = df.rename(columns={"value": name})

            # 合并标签
            df_label = evaluator._load_labels(args.start, args.end)
            if df_label is not None:
                df = df.join(df_label, how="inner")

            # [AQR] 完整评测流水线
            if args.walk_forward:
                result = evaluator.walk_forward_evaluate(
                    name, df.reset_index(),
                    config_override={"category": f.get("category", "satellite")},
                )
            else:
                result = evaluator.evaluate(
                    name, df.reset_index(),
                    category=f.get("category", "satellite"),
                )
            q = result["qual_result"]
            icon = {"core": "[OK]", "satellite": "[~]", "archive": "[ ]"}.get(q["tier"], "[ ]")
            # 读取最新 meta 信息（兼容新旧格式）
            cur_meta = store.get_warehouse_meta(name) or {}
            total_str = _meta_val(cur_meta, "total_records", 0) or _meta_val(cur_meta, "total_rows", 0)
            if total_str:
                total_str = f"{total_str:,}"
            last_dt = _meta_val(cur_meta, "last_date", "?")
            last_rows = f"({total_str}行, 至{last_dt})" if not args.recompute else ""
            print(f"  {icon} 等级={q['tier']:10s} IC={result['ic_stats']['ic_mean']:.4f} ICIR={result['ic_stats']['icir']:.2f} 评分={q['composite_score']:.1f} {last_rows}")

        except Exception as e:
            print(f"  [X] {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n完成! 报告: {config.report_dir}")


if __name__ == "__main__":
    main()
