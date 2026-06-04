#!/usr/bin/env python3
"""
因子批量计算与增量追加脚本。

数据来源：
  ClickHouse (192.168.10.102:18123) daily_prices / daily_indicators / financial_indicators

用法：
  # 首次批量计算（逐因子，每个因子独立查询 ClickHouse，较慢但兼容性好）
  python batch_compute.py --all --start 2018-01-01 --end 2025-12-31

  # 批量快速计算（一次 ClickHouse 查询计算所有价格类因子，速度快 N 倍）[推荐]
  python batch_compute.py --all --batch --start 2018-01-01 --end 2025-12-31

  # 计算指定分类下的因子
  python batch_compute.py --category price_volume_factors
  python batch_compute.py --category reversal_momentum_factors --batch

  # 计算单个因子
  python batch_compute.py --factor KDJ_K
  python batch_compute.py --factor STR_20d --overwrite

  # 增量追加最新数据（每周/每月执行，逐因子追加）
  python batch_compute.py --all --append

  # 向已有 warehouse 因子注入语义元数据（从 YAML 因子库）
  python batch_compute.py --all --inject-meta

  # 查看仓库状态
  python batch_compute.py --status

说明：
  --batch 模式已完全覆盖 scripts/factors/calculate_reversal_momentum.py 的全部功能，
  包括因子的批次计算和 YAML 语义元数据注入。后者已废弃。
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


def _meta_val(meta: dict, key: str, default=None):
    """兼容新旧 meta 格式读取辅助函数。"""
    dr = meta.get("data_range", {}) if meta else {}
    if key in dr and dr[key]:
        return dr[key]
    if meta and key in meta and meta[key]:
        return meta[key]
    return default


def _inject_yaml_meta(store, factors):
    """从 YAML 因子库提取语义元数据并注入到 warehouse。"""
    m = FactorLibraryManager()
    for f in factors:
        name = f["name"]
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            try:
                cfg = m.load_strategy_config(s)
                for fd in cfg.get("factors", []):
                    if fd.get("name") != name:
                        continue
                    yaml_meta = {
                        "version": fd.get("version", "1.0"),
                        "category": fd.get("category", ""),
                        "sub_category": fd.get("sub_category", ""),
                        "expression": fd.get("expression", {}),
                        "function_description": fd.get("meaning", ""),
                        "theory_background": fd.get("logic", {}).get("theory", ""),
                        "applicable_conditions": {
                            "market": "中国A股市场",
                            "frequency": fd.get("parameters", {}).get("freq", "daily"),
                            "lookback_period": f"{fd.get('parameters', {}).get('lookback', 0)} 交易日" if fd.get("parameters", {}).get("lookback") else "",
                            "expected_direction": fd.get("logic", {}).get("expected_direction", ""),
                            "data_requirement": "需要前复权收盘价数据",
                        },
                        "usage_scenario": fd.get("usage_scenario", ""),
                        "strategy_hint": fd.get("strategy_hint", ""),
                        "reference": fd.get("ref", ""),
                        "lifecycle_stage": fd.get("lifecycle_stage", "exploration"),
                    }
                    # 如果 YAML 中没写 usage_scenario/strategy_hint，按分类推导
                    if not yaml_meta["usage_scenario"]:
                        cat = fd.get("category", "")
                        sub = fd.get("sub_category", "")
                        usage_map = {
                            "reversal": {"short_term": "短周期（5-20日）均值回归策略，适合高频或中高频交易",
                                          "long_term": "长周期（年频）反转策略，适合低频配置型组合",
                                          "volatility": "波动率择时，识别高波动后的均值回归机会",
                                          "extreme": "极端行情（涨停/跌停）后的反向交易信号"},
                            "momentum": {"medium_term": "中期趋势跟踪，适合 3-6 个月持仓周期的动量策略",
                                          "long_term": "长期趋势跟踪，适合年度调仓的动量因子组合",
                                          "risk_adjusted": "跳过短期噪音的中长期动量，适合组合中的 alpha 增厚",
                                          "industry_neutral": "行业中性动量，在行业内选股，控制行业暴露",
                                          "acceleration": "动量加速信号，识别趋势加强阶段"},
                        }
                        yaml_meta["usage_scenario"] = usage_map.get(cat, {}).get(sub, "")
                    if not yaml_meta["strategy_hint"]:
                        cat = fd.get("category", "")
                        sub = fd.get("sub_category", "")
                        hint_map = {
                            "reversal": {"short_term": "配合低波动率因子使用效果更佳；注意避开财报密集发布期",
                                          "long_term": "与动量因子形成天然对冲，适合多空组合",
                                          "volatility": "适合在市场恐慌情绪高点时入场",
                                          "extreme": "连续极端值后信号更强，但需警惕跌停无法成交的流动性风险"},
                            "momentum": {"medium_term": "注意避开大市值股票，动量在小市值中更显著",
                                          "long_term": "长期动量可能存在趋势衰竭风险，建议设置止盈条件",
                                          "risk_adjusted": "跳过一个月有效规避了短期反转噪音，信号更稳定",
                                          "industry_neutral": "适合行业配置型组合，降低行业轮动干扰",
                                          "acceleration": "趋势加速信号通常伴随高波动，需配合风险管理"},
                        }
                        yaml_meta["strategy_hint"] = hint_map.get(cat, {}).get(sub, "")
                    store.inject_warehouse_meta(name, yaml_meta)
                    print(f"  {name:25s} 元数据已注入")
                    break
            except Exception:
                continue


def show_warehouse_status(store):
    """显示仓库状态。"""
    factors = store.list_warehouse_factors()
    if not factors:
        print("\n  仓库为空，请先运行 --all 批量计算。")
        return

    print(f"\n  {'因子名':30s} {'年份':20s} {'总行数':>10s}  {'最后日期':14s}  {'分类':12s}")
    print(f"  {'-'*30} {'-'*20} {'-'*10}  {'-'*14}  {'-'*12}")
    for name in factors:
        meta = store.get_warehouse_meta(name)
        if meta:
            years_list = _meta_val(meta, "years", [])
            years = ",".join(str(y) for y in (years_list or []))
            total = _meta_val(meta, "total_records", 0) or _meta_val(meta, "total_rows", 0)
            last = _meta_val(meta, "last_date", "")
            cat = meta.get("category", "") or meta.get("sub_category", "") or ""
            print(f"  {name:30s} {years:20s} {total:>10,}  {last:14s}  {cat:12s}")
    print(f"\n  共 {len(factors)} 个因子")


def main():
    parser = argparse.ArgumentParser(description="因子批量计算与增量追加")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--factor", "-f", help="单个因子名")
    g.add_argument("--category", "-c", help="分类 YAML 文件")
    g.add_argument("--all", "-a", action="store_true", help="所有因子")
    g.add_argument("--status", "-s", action="store_true", help="查看仓库状态")
    parser.add_argument("--append", action="store_true", help="增量追加（默认全量计算）")
    parser.add_argument("--batch", action="store_true", help="批量模式：一次 ClickHouse 查询计算多个因子（更快，仅限价格类因子）")
    parser.add_argument("--inject-meta", action="store_true", help="向已有 warehouse 因子注入 YAML 语义元数据")
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
    print(f"  模式:   {'增量追加' if args.append else '注入元数据' if args.inject_meta else '批量' if args.batch else '逐因子'}")
    print(f"{'='*60}")

    total_ok = 0
    total_skip = 0
    total_fail = 0

    # ── 注入语义元数据模式 ──
    if args.inject_meta:
        print("\n[可选] 向已有 warehouse 因子注入 YAML 语义元数据...")
        _inject_yaml_meta(store, factors)
        print("\n元数据注入完成")
        show_warehouse_status(store)
        return

    # ── 批量快速计算模式（一次 ClickHouse 查询）[推荐] ──
    if args.batch and not args.append:
        duckdb_factors = [(f["name"], f.get("duckdb_expr", "")) for f in factors if f.get("duckdb_expr", "").strip()]
        if not duckdb_factors:
            print("  [警告] 所有因子均无 DuckDB 表达式，无法使用 --batch 模式。请去掉 --batch 用逐因子模式。")
        else:
            skipped = len(factors) - len(duckdb_factors)
            if skipped:
                print(f"  [提示] {skipped} 个因子无 DuckDB 表达式，已在 batch 中跳过（将回退到逐因子模式）")
            stats_all = store.batch_compute(
                duckdb_factors, args.start, args.end,
                overwrite=args.overwrite,
            )
            for name, yr_stats in stats_all.items():
                if yr_stats:
                    yr = ",".join(f"{y}({n})" for y, n in yr_stats.items())
                    print(f"  {name:25s} 已计算: {yr}")
                    total_ok += 1
                else:
                    print(f"  {name:25s} 已存在")
                    total_skip += 1
        # 注入 YAML 语义元数据
        _inject_yaml_meta(store, factors)
    else:
        # ── 逐因子计算模式（默认 / --append） ──
        for i, f in enumerate(factors, 1):
            name = f["name"]
            expr = f["expr"]
            duckdb_expr = f.get("duckdb_expr", "")
            try:
                if args.append:
                    n = store.append_to_warehouse(name, expr, args.start, duckdb_expr=duckdb_expr)
                    status = f"+{n}行" if n else "已最新"
                    print(f"  [{i}/{len(factors)}] {name:25s} {status}")
                    total_ok += 1
                else:
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
                        print(f"  [{i}/{len(factors)}] {name:25s} 已存在，跳过")
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
