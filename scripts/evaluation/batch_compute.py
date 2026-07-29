#!/usr/bin/env python3
"""
因子批量计算与增量追加脚本。

数据来源：
  ClickHouse (192.168.10.102:18123) daily_prices / daily_indicators / financial_indicators

用法：
  # 首次批量计算（逐因子，每个因子独立查询 ClickHouse，较慢但兼容性好）
  python batch_compute.py --all --start 2010-01-01 --end 2026-12-31

  # 批量快速计算（一次 ClickHouse 查询计算所有价格类因子，速度快 N 倍）[推荐]
  python batch_compute.py --all --batch --start 2010-01-01 --end 2026-12-31

  # 计算指定分类下的因子
  python batch_compute.py --category price_volume_factors
  python batch_compute.py --category reversal_momentum_factors --batch

  # 计算单个因子
  python batch_compute.py --factor KDJ_K

  # 计算多个因子（空格分隔）
  python batch_compute.py --factor KDJ_K MA5 STR_20d
  python batch_compute.py --factor KDJ_K MA5 --insert from 2020-01-01 to 2021-12-31
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
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

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
        for fname in args.factor:
            info = _find_factor(m, fname)
            if info:
                factors.append(info)
            else:
                print(f"  [警告] 未找到因子: {fname}")
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
        for s in m.list_strategies():
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
    """在因子库中查找单个因子（包括 dictionary 文件）。"""
    for s in m.list_strategies():
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


def _inject_yaml_meta(store, factors):
    """从 YAML 因子库提取语义元数据并注入到 warehouse。"""
    m = FactorLibraryManager()
    for f in factors:
        name = f["name"]
        for s in m.list_strategies():
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
    """显示仓库状态：读取 parquet 数据，计算详细统计，输出 JSON。"""
    factors = store.list_warehouse_factors()
    if not factors:
        print("\n  仓库为空，请先运行 --all 批量计算。")
        # 仍然输出空 JSON
        out_path = store.warehouse_dir.parent / "warehouse_status.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print(f"\n  仓库状态已保存至: {out_path}")
        return

    result = {}
    print(f"\n  {'因子名':28s} {'起止日期':28s} {'总数':>10s} {'空值':>10s} {'零值':>10s} {'有效值':>10s}")
    print(f"  {'-'*28} {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name in factors:
        years = store.get_warehouse_years(name)
        dfs = []
        for y in years:
            df = store._load_warehouse_year(name, y)
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            result[name] = {
                "status": "empty",
                "years": years,
                "start_date": "",
                "end_date": "",
                "total_records": 0,
                "null_count": 0,
                "zero_count": 0,
                "valid_count": 0,
            }
            print(f"  {name:28s} {'(空)':28s} {'0':>10} {'0':>10} {'0':>10} {'0':>10}")
            continue

        all_data = pd.concat(dfs).sort_index()
        # 动态取数值列（因子值可能存储在不同列名下）
        value_cols = [c for c in all_data.columns if c not in ("instrument", "datetime")]
        if not value_cols:
            result[name] = {
                "status": "no_value_column",
                "years": years,
                "start_date": "",
                "end_date": "",
                "total_records": 0,
                "null_count": 0,
                "zero_count": 0,
                "valid_count": 0,
            }
            print(f"  {name:28s} {'(无数据列)':28s} {'0':>10} {'0':>10} {'0':>10} {'0':>10}")
            continue

        dates = all_data.index.get_level_values("datetime")
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")

        # 支持多列因子（如 ff_factors），逐列统计
        columns_stats = {}
        total_records = 0
        total_null = 0
        total_zero = 0
        total_valid = 0
        for col in value_cols:
            v = all_data[col]
            n_null = int(v.isna().sum())
            n_zero = int((v == 0).sum())
            n_valid = int((v.notna() & (v != 0)).sum())
            columns_stats[col] = {
                "null_count": n_null,
                "zero_count": n_zero,
                "valid_count": n_valid,
            }
            total_records += len(v)
            total_null += n_null
            total_zero += n_zero
            total_valid += n_valid

        if len(value_cols) == 1:
            result[name] = {
                "status": "ok",
                "years": years,
                "start_date": start_date,
                "end_date": end_date,
                "total_records": total_records,
                "null_count": total_null,
                "zero_count": total_zero,
                "valid_count": total_valid,
            }
        else:
            result[name] = {
                "status": "ok",
                "years": years,
                "start_date": start_date,
                "end_date": end_date,
                "total_records": total_records,
                "columns": columns_stats,
            }

        date_range = f"{start_date} ~ {end_date}"
        print(f"  {name:28s} {date_range:28s} {total_records:>10,} {total_null:>10,} {total_zero:>10,} {total_valid:>10,}")

    # 汇总
    total_factors = len(result)
    total_records = sum(v["total_records"] for v in result.values())
    # 汇总（兼容单列和多列因子）
    def _get_counts(v):
        if "null_count" in v:
            return v["null_count"], v["zero_count"], v["valid_count"]
        if "columns" in v:
            n_null = sum(c["null_count"] for c in v["columns"].values())
            n_zero = sum(c["zero_count"] for c in v["columns"].values())
            n_valid = sum(c["valid_count"] for c in v["columns"].values())
            return n_null, n_zero, n_valid
        return 0, 0, 0

    total_null = sum(_get_counts(v)[0] for v in result.values())
    total_zero = sum(_get_counts(v)[1] for v in result.values())
    total_valid = sum(_get_counts(v)[2] for v in result.values())

    print(f"  {'-'*28} {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'合计':28s} {'':28s} {total_records:>10,} {total_null:>10,} {total_zero:>10,} {total_valid:>10,}")
    print(f"\n  共 {total_factors} 个因子，总计 {total_records:,} 条记录")

    # 输出 JSON
    out_path = store.warehouse_dir.parent / "warehouse_status.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  仓库状态已保存至: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="因子批量计算与增量追加")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--factor", "-f", nargs="+", help="一个或多个因子名，空格分隔，如: --factor KDJ_K MA5 STR_20d")
    g.add_argument("--category", "-c", help="分类 YAML 文件")
    g.add_argument("--all", "-a", action="store_true", help="所有因子")
    g.add_argument("--status", "-s", action="store_true", help="查看仓库状态")
    parser.add_argument("--append", action="store_true", help="增量追加（默认全量计算）")
    parser.add_argument("--batch", action="store_true", help="批量模式：一次 ClickHouse 查询计算多个因子（更快，仅限价格类因子）")
    parser.add_argument("--inject-meta", action="store_true", help="向已有 warehouse 因子注入 YAML 语义元数据")
    parser.add_argument("--insert", nargs=4, metavar=("from", "FROM_DATE", "to", "TO_DATE"),
                        help="插入/覆盖模式：针对因子下载指定起止时间的数据（会覆盖已有数据），"
                             "示例: --insert from 2020-01-01 to 2021-12-31，可配合 --factor/--category/--all/--batch 使用")
    parser.add_argument("--start", default="2010-01-01", help="开始日期")
    parser.add_argument("--end", default="2026-12-31", help="结束日期")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有数据")
    args = parser.parse_args()

    # ── 解析 --insert 参数 ──
    if args.insert is not None:
        if args.insert[0] != "from" or args.insert[2] != "to":
            print("错误：--insert 格式应为 --insert from <开始时间> to <结束时间>，"
                  "示例: --insert from 2020-01-01 to 2021-12-31")
            sys.exit(1)
        insert_start = args.insert[1]
        insert_end = args.insert[3]
        if insert_start > insert_end:
            print(f"错误：--insert 开始时间 {insert_start} 晚于结束时间 {insert_end}")
            sys.exit(1)
    else:
        insert_start = None
        insert_end = None

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

    # ── 确定计算起止时间与覆盖模式 ──
    if insert_start is not None:
        compute_start = insert_start
        compute_end = insert_end
        is_insert_mode = True
    else:
        compute_start = args.start
        compute_end = args.end
        is_insert_mode = False

    mode_str = "插入/覆盖" if is_insert_mode else (
        "增量追加" if args.append else "注入元数据" if args.inject_meta else "批量" if args.batch else "逐因子"
    )
    print(f"\n{'='*60}")
    print(f"{'因子仓库批量计算':^60}")
    print(f"{'='*60}")
    print(f"  因子数: {len(factors)}")
    print(f"  时间:   {compute_start} ~ {compute_end}")
    print(f"  模式:   {mode_str}")
    print(f"{'='*60}")

    total_ok = 0
    total_skip = 0
    total_fail = 0

    # 插入/覆盖模式强制 overwrite
    overwrite = True if is_insert_mode else args.overwrite

    # ── 注入语义元数据模式 ──
    if args.inject_meta:
        print("\n[可选] 向已有 warehouse 因子注入 YAML 语义元数据...")
        _inject_yaml_meta(store, factors)
        print("\n元数据注入完成")
        show_warehouse_status(store)
        return

    # ── 批量快速计算模式（一次 ClickHouse 查询）[推荐] ──
    if args.batch and not args.append:
        duckdb_factors = []
        qlib_converted = 0
        for f in factors:
            ds = f.get("duckdb_expr", "").strip()
            if ds:
                duckdb_factors.append((f["name"], ds))
            else:
                conv = FactorStore._qlib_to_duckdb(f.get("expr", ""))
                if conv:
                    duckdb_factors.append((f["name"], conv))
                    qlib_converted += 1
        if not duckdb_factors:
            print("  [警告] 所有因子均无 DuckDB 表达式，无法使用 --batch 模式。请去掉 --batch 用逐因子模式。")
        else:
            skipped = len(factors) - len(duckdb_factors)
            if skipped:
                if qlib_converted:
                    print(f"  [提示] {skipped} 个因子无 DuckDB 表达式（{qlib_converted} 个已通过 Qlib 转换加入 batch）")
                else:
                    print(f"  [提示] {skipped} 个因子无 DuckDB 表达式，已在 batch 中跳过（将回退到逐因子模式）")
            stats_all = store.batch_compute(
                duckdb_factors, compute_start, compute_end,
                overwrite=overwrite,
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
        # ── 逐因子计算模式（默认 / --append / --insert） ──
        for i, f in enumerate(factors, 1):
            name = f["name"]
            expr = f["expr"]
            duckdb_expr = f.get("duckdb_expr", "")
            try:
                if args.append and not is_insert_mode:
                    # append 模式传 None 让 warehouse 自动从最后日期+1天开始
                    # 避免从 2010-01-01 重算导致旧年份行数波动出现负数
                    n = store.append_to_warehouse(name, expr, start_date=None, duckdb_expr=duckdb_expr)
                    status = f"+{n}行" if n else "已最新"
                    print(f"  [{i}/{len(factors)}] {name:25s} {status}")
                    total_ok += 1
                else:
                    stats = store.compute_to_warehouse(
                        name, expr, compute_start, compute_end,
                        overwrite=overwrite,
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
