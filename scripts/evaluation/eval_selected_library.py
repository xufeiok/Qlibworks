#!/usr/bin/env python3
"""
评测选定因子库中已在 warehouse 有数据的因子，股票池为沪深主板。

用法:
  python eval_selected_library.py
  python eval_selected_library.py --start 2018-01-01 --end 2025-12-31
  python eval_selected_library.py --dry-run    # 仅列出符合条件的因子
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
warnings.filterwarnings("ignore")

from qlworks.factors import FactorLibraryManager
from qlworks.evaluation import FactorStore
from qlworks.evaluation.config import EvalConfig, DEFAULT_CONFIG
from qlworks.evaluation.runner import FactorEvaluator
from dataclasses import replace
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_selected")

# 要扫描的 YAML 文件白名单（排除 archive/）
# 通过 --batch N 可追加 pv_batch_N.yaml
BASE_STRATEGIES = [
    "quality_factors", "reversal_momentum_factors",
    "risk_factors", "sentiment_factors", "style_factors", "other_factors",
]

# 价量因子分批文件（pv_batch_1 ~ pv_batch_5）
PV_BATCH_NAMES = [f"pv_batch_{i}" for i in range(1, 6)]

# 已知当前数据源无法计算的因子（ClickHouse/Qlib 均无对应字段）
KNOWN_SKIP_FACTORS = {"q_profit_yoy", "eps_forecast_yoy"}

_INSTRUMENTS_DIR = Path(__file__).resolve().parents[2] / "qlib_data" / "instruments"


def get_pool_stocks(pool_name: str) -> set:
    """从本地 instruments 目录加载股票池。"""
    pool_file = _INSTRUMENTS_DIR / f"{pool_name}.txt"
    if not pool_file.exists():
        logger.warning(f"股票池文件不存在: {pool_file}")
        return set()
    stocks = set()
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                stocks.add(parts[0])
    logger.info(f"[股票池] {pool_name}: {len(stocks)} 只股票")
    return stocks

def collect_factors(mgr, strategy_names=None):
    """
    扫描指定 YAML 文件，收集所有因子定义。
    warehouse 数据不够的因子会在后续评测时自动从 qlib 兜底计算。

    Args:
        strategy_names: 策略名列表，None 则使用 BASE_STRATEGIES
    """
    if strategy_names is None:
        strategy_names = BASE_STRATEGIES
    factors = []
    for strategy in strategy_names:
        try:
            config = mgr.load_strategy_config(strategy)
        except FileNotFoundError:
            logger.warning(f"跳过 {strategy}.yaml（未找到）")
            continue
        factor_list = config.get("factors") or []
        for fd in factor_list:
            name = fd.get("name")
            if not name:
                continue
            expr = fd.get("expression", "")
            duckdb_expr = ""
            if isinstance(expr, dict):
                duckdb_expr = str(expr.get("duckdb", ""))
                expr = str(expr.get("qlib", str(expr)))
            factors.append({
                "name": name,
                "expr": str(expr),
                "duckdb_expr": duckdb_expr,
                "category": strategy,
                "meaning": fd.get("meaning", ""),
            })
    return factors

def main():
    parser = argparse.ArgumentParser(description="评测选定因子库中 warehouse 已有的因子（沪深主板）")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--dry-run", "-n", action="store_true", help="仅列出因子不评测")
    parser.add_argument("--walk-forward", action="store_true", help="启用 Walk-Forward 验证")
    parser.add_argument("--pool", default="main_board", help="股票池（默认 main_board=沪深主板）")
    parser.add_argument("--batch", type=int, default=0, choices=range(0, 6),
                        help="价量因子分批评测: 1~5 对应 pv_batch_N.yaml, 0 仅评测基础策略(默认)")
    parser.add_argument("--yaml", type=str, default="",
                        help="直接指定 YAML 文件名（不含路径和扩展名），覆盖 --batch")
    args = parser.parse_args()

    # 构建策略列表
    if args.yaml:
        strategy_names = [args.yaml]
        print(f"\n评测模式: 仅 {args.yaml}.yaml (自定义策略)")
    elif args.batch and 1 <= args.batch <= 5:
        strategy_names = [PV_BATCH_NAMES[args.batch - 1]]
        print(f"\n评测模式: 仅 pv_batch_{args.batch} (独立分批评测)")
    else:
        strategy_names = BASE_STRATEGIES
        print("\n评测模式: 仅基础策略")

    mgr = FactorLibraryManager()
    store = FactorStore()
    _REPORT_DIR = DEFAULT_CONFIG.report_dir

    # 1. 收集所有因子（不跳过 warehouse 无数据的）
    logger.info("扫描因子库...")
    all_factors = collect_factors(mgr, strategy_names)
    # 过滤已知不可计算的因子
    factors = [f for f in all_factors if f["name"] not in KNOWN_SKIP_FACTORS]
    skipped = len(all_factors) - len(factors)
    if skipped:
        skipped_names = KNOWN_SKIP_FACTORS & {f["name"] for f in all_factors}
        print(f"  跳过 {skipped} 个已知不可计算因子: {sorted(skipped_names)}")
    print(f"\n共 {len(factors)} 个因子待评测\n")

    if args.dry_run:
        print(f"\n{'因子名':30s} {'分类':25s} {'含义'}")
        print("-" * 100)
        for f in factors:
            meaning = (f.get("meaning") or "")[:50]
            print(f"  {f['name']:30s} {f['category']:25s} {meaning}")
        return

    # 2. 获取股票池（本地文件，不依赖 ClickHouse）
    pool_stocks = get_pool_stocks(args.pool)
    print(f"  {args.pool} 共 {len(pool_stocks)} 只股票")

    # 3. 初始化评测器（使用 DEFAULT_CONFIG 确保 report_dir 正确）
    config = replace(DEFAULT_CONFIG, start_time=args.start, end_time=args.end)
    evaluator = FactorEvaluator(config)

    # 4. 预加载标签数据（只加载一次，避免循环内重复加载）
    logger.info("预加载标签数据...")
    df_label = evaluator._load_labels(args.start, args.end)
    if df_label is not None:
        logger.info(f"  标签: {len(df_label)} 行, "
                    f"{df_label.index.get_level_values('instrument').nunique()} 只股票")

    # 5. 逐因子评测（warehouse 优先，数据不够则 qlib 兜底）
    results = []
    for i, f in enumerate(factors, 1):
        name = f["name"]
        print(f"\n[{i}/{len(factors)}] {name} ({f['category']})")

        try:
            # ── 路径 A：从 warehouse 加载 ──
            df = store.load_from_warehouse(name, args.start, args.end)
            if df is not None and not df.empty:
                logger.info(f"  warehouse 加载: {len(df):,} 行")
            else:
                # ── 路径 B：warehouse 数据不够，用 ClickHouse/Qlib 兜底计算 ──
                logger.info(f"  warehouse 无数据，从 ClickHouse 计算...")
                store.compute_to_warehouse(name, f["expr"], args.start, args.end,
                                           duckdb_expr=f.get("duckdb_expr", ""))
                df = store.load_from_warehouse(name, args.start, args.end)
                if df is None or df.empty:
                    logger.warning(f"  {name} 计算后仍无数据，跳过")
                    continue
                logger.info(f"  兜底计算完成: {len(df):,} 行")

            df = df.rename(columns={"value": name})

            # ── 过滤到指定股票池 ──
            codes = df.index.get_level_values("instrument")
            before = len(df)
            df = df[codes.isin(pool_stocks)]
            removed = before - len(df)
            if removed > 0:
                logger.info(f"  股票池过滤: 去除 {removed:,} 行非 {args.pool}")

            # ── 补充市值和行业字段（本地 Qlib 优先） ──
            from qlworks.evaluation.enrich import enrich_with_extra_fields
            df = enrich_with_extra_fields(df, args.start, args.end)

            # ── 合并标签（使用预加载的缓存标签） ──
            if df_label is not None:
                df = df.join(df_label, how="inner")

            # ── 执行评测 ──
            result = evaluator.evaluate(name, df.reset_index(), category=f["category"])
            q = result["qual_result"]
            tier = q["tier"]
            icon = {"core": "✓", "satellite": "~", "archive": " "}.get(tier, " ")

            # 报告路径（evaluate 内部生成的 HTML）
            report_dir = Path(_REPORT_DIR) / tier
            report_files = sorted(report_dir.glob(f"{name}_*.html"))
            report_path = f"file:///{report_files[-1].as_posix()}" if report_files else "(未生成)"

            print(f"  {icon} 评级={tier:9s} IC={result['ic_stats']['ic_mean']:.4f} "
                  f"ICIR={result['ic_stats']['icir']:.2f} 评分={q['composite_score']:.1f}")
            print(f"  报告: {report_path}")
            result["report_path"] = report_path
            results.append(result)

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback; traceback.print_exc()

    # 5. 汇总
    qualified = [r for r in results if r.get("qual_result", {}).get("qualified")]
    report_count = sum(1 for r in results if r.get("report_path") and r["report_path"] != "(未生成)")
    print(f"\n{'='*60}")
    print(f"评测完成! 总计 {len(results)} | 合格 {len(qualified)} | 失败 {len(factors)-len(results)}")
    print(f"生成报告: {report_count} 份（目录: file:///{_REPORT_DIR}）")
    if qualified:
        print(f"\n{'因子名':28s} {'评级':10s} {'IC':>8s} {'ICIR':>6s} {'评分':>6s} {'报告'}")
        print("-" * 90)
        for r in qualified:
            tier = r.get("qual_result", {}).get("tier", "?")
            fname = r['factor_name']
            rp = f"file:///{_REPORT_DIR}/{tier}/{fname}_*.html"
            print(f"  ✓ {fname:28s} {tier:10s} "
                  f"{r['ic_stats']['ic_mean']:.4f} {r['ic_stats']['icir']:.2f} "
                  f"{r['qual_result']['composite_score']:.1f}  {rp}")

if __name__ == "__main__":
    main()
