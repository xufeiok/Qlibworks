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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")

from qlworks.evaluation import FactorStore, DEFAULT_CONFIG
from qlworks.evaluation.runner import FactorEvaluator
from qlworks.factors import FactorLibraryManager
from qlworks.config import QLIB_DATA_DIR

# ── 自动发现 instruments 目录下的可用股票池 ──
_INSTRUMENTS_DIR = QLIB_DATA_DIR / "instruments"
_AVAILABLE_POOLS = sorted(f.stem for f in _INSTRUMENTS_DIR.glob("*.txt") if f.is_file())


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


def _load_pool_stocks(pool_name: str) -> set:
    """从 instruments 目录加载指定股票池的股票代码集合。

    Args:
        pool_name: 股票池名称（如 csi500, main_board, all_sh 等）

    Returns:
        有效股票代码的 set（已过滤掉过期条目），如 pool 文件不存在则返回空 set
    """
    pool_file = _INSTRUMENTS_DIR / f"{pool_name}.txt"
    if not pool_file.exists():
        available = ", ".join(_AVAILABLE_POOLS)
        logger.warning(f"股票池文件不存在: {pool_name}.txt，可用池: {available}")
        return set()

    stocks = set()
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                stocks.add(parts[0])
    logger.info(f"[股票池] {pool_name}: {len(stocks)} 只股票")
    return stocks


def _meta_val(meta: dict, key: str, default=None):
    """兼容新旧 meta 格式读取辅助函数。"""
    # 新格式在 data_range 下
    dr = meta.get("data_range", {}) if meta else {}
    if key in dr and dr[key]:
        return dr[key]
    # 旧格式在顶层
    if meta and key in meta and meta[key]:
        return meta[key]
    # statistics 嵌套 (warehouse meta.json 格式)
    stats = meta.get("statistics", {}) if meta else {}
    if key in stats and stats[key]:
        return stats[key]
    # last_date 兼容：实际 key 为 end_date
    if key == "last_date":
        return dr.get("end_date", default)
    return default


def _enrich_with_extra_fields(df, start_time, end_time, logger=None):
    """加载 mkt_cap（流通市值）和 industry（申万一级行业）并合并到 df。

    这样场景压力测试（分市值、分行业板块）和控制变量对冲（双变量分组、残差因子）
    才能获取数据，否则这些评测项目会因缺数据而使用 0.5 中性分。

    Args:
        df: DataFrame with MultiIndex (instrument, datetime), 含因子列和标签列
        start_time: 起始日期
        end_time: 结束日期

    Returns:
        合并后的 DataFrame，包含 circ_mv 和 sw_l1 列（如果数据可用）
    """
    log = logger or globals().get("logger")

    if not isinstance(df.index, pd.MultiIndex):
        if log:
            log.warning("[补充字段] DataFrame 非 MultiIndex，跳过")
        return df

    instruments = df.index.get_level_values("instrument").unique().tolist()

    # ── 路径 A：Qlib 路径 ──
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.config import QLIB_DATA_DIR
        qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN, mute_warning=True)

        fields = ["$circ_mv", "$sw_l1"]
        col_names = ["circ_mv", "sw_l1"]
        extra = D.features(instruments, fields, str(start_time)[:10], str(end_time)[:10])
        if not extra.empty:
            if isinstance(extra.columns, pd.MultiIndex):
                extra.columns = [c[0] for c in extra.columns]
            keep = [f for f in fields if f in extra.columns]
            if keep:
                extra = extra[keep].copy()

                # 处理 sw_l1 编码 — Qlib $sw_l1 可能存为数值ID
                if "$sw_l1" in extra.columns:
                    mapping_path = Path(str(QLIB_DATA_DIR)) / "sw_industry_mapping.json"
                    if mapping_path.exists():
                        try:
                            import json
                            with open(mapping_path, encoding="utf-8") as f:
                                id_map = json.load(f)
                            reverse_map = {v: k for k, v in id_map.get("l1", {}).items()}
                            if reverse_map:
                                mapped = extra["$sw_l1"].map(reverse_map)
                                # 只在新映射有足够匹配率时替换（>10%），否则保留原始数值
                                match_rate = mapped.notna().sum() / max(len(mapped), 1)
                                if match_rate > 0.1:
                                    extra["$sw_l1"] = mapped
                                else:
                                    if log:
                                        log.info(f"[补充字段] 行业编码映射匹配率仅 {match_rate:.1%}，保留原始数值")
                        except Exception:
                            pass

                if df.index.names != extra.index.names:
                    extra = extra.swaplevel()
                extra = extra.reindex(df.index)
                extra.columns = col_names

                df["circ_mv"] = extra["circ_mv"]
                df["sw_l1"] = extra["sw_l1"]

                n_valid_mc = int(df["circ_mv"].notna().sum())
                n_valid_ind = int(df["sw_l1"].notna().sum())
                mc_ok = n_valid_mc > len(df) * 0.1
                if log:
                    log.info(f"[补充字段] Qlib: circ_mv={n_valid_mc:,}行有效({mc_ok}), sw_l1={n_valid_ind:,}行有效 "
                             f"(共{len(instruments)}只股票)")

                # ── 行业分类 sw_l1 处理：先用本地 SW 映射解码 ──
                from qlworks.evaluation.sw_mapping import decode_sw_series

                ind_ok = n_valid_ind > len(df) * 0.1
                is_numeric_ind = True  # 默认视作数值编码

                if ind_ok:
                    # 尝试用本地申万行业映射解码
                    decoded = decode_sw_series(df["sw_l1"], level=1)
                    n_mapped = decoded.str.startswith("未知(").sum() if len(decoded) > 0 else len(decoded)
                    mapped_ratio = 1 - n_mapped / max(len(decoded), 1)
                    if mapped_ratio > 0.5:
                        # 大部分成功解码 → 直接用映射后的名称
                        df["sw_l1"] = decoded
                        is_numeric_ind = False
                        if log:
                            log.info(f"[补充字段] SW_L1 解码: {1-n_mapped:,}/{len(decoded)} 行成功 ({mapped_ratio:.1%})")
                    elif log:
                        log.info(f"[补充字段] SW_L1 解码覆盖率不足 ({mapped_ratio:.1%})")

                # 解码失败时的回退
                if is_numeric_ind or not ind_ok:
                    # 尝试回退 $industry
                    try:
                        alt_fields = ["$industry"]
                        alt_extra = D.features(instruments, alt_fields, str(start_time)[:10], str(end_time)[:10])
                        if not alt_extra.empty:
                            if isinstance(alt_extra.columns, pd.MultiIndex):
                                alt_extra.columns = [c[0] for c in alt_extra.columns]
                            if df.index.names != alt_extra.index.names:
                                alt_extra = alt_extra.swaplevel()
                            alt_extra = alt_extra.reindex(df.index)
                            df["sw_l1"] = alt_extra[alt_fields[0]]
                            n_valid_ind = int(df["sw_l1"].notna().sum())
                            ind_ok = n_valid_ind > len(df) * 0.1
                            is_numeric_ind = False
                            if log:
                                log.info(f"[补充字段] 回退 $industry: {n_valid_ind:,}行有效")
                    except Exception:
                        pass

                # ── circ_mv 独立处理：Qlib 数据有效则直接保留 ──
                if mc_ok:
                    if log:
                        log.info(f"[补充字段] Qlib circ_mv 有效 ({n_valid_mc:,}行)，直接保留")
                else:
                    if log:
                        log.warning(f"[补充字段] Qlib circ_mv 不足 ({n_valid_mc:,}行)，后续 ClickHouse 补充")

                # ── sw_l1 和 circ_mv 都满足条件则立即返回 ──
                if mc_ok and ind_ok and not is_numeric_ind:
                    if log:
                        log.info(f"[补充字段] Qlib 全达标: circ_mv={n_valid_mc:,}, sw_l1={n_valid_ind:,} → 直接使用")
                    return df

                if log:
                    ind_note = f"sw_l1={'数值编码' if is_numeric_ind else '不足'}({n_valid_ind:,}行)"
                    log.info(f"[补充字段] Qlib circ_mv={'有效' if mc_ok else '不足'}, {ind_note} → 仅对 sw_l1 用 ClickHouse 补充")

    except ImportError:
        if log:
            log.info("[补充字段] Qlib 未安装，切换到 ClickHouse 路径...")
    except Exception as e:
        if log:
            log.warning(f"[补充字段] Qlib 路径失败: {e}，切换到 ClickHouse 路径...")

    # ── 路径 B：ClickHouse 后备（仅补充缺失的数据） ──
    try:
        from qlworks.data import QuantDataAPI
        api = QuantDataAPI()
        # 用 ClickHouse 查询 circ_mv 和行业
        ch_sql = f"""SELECT p.ts_code AS instrument, p.trade_date AS datetime,
       CAST(i.circ_mv AS DOUBLE) AS circ_mv,
       sw.l1_name AS sw_l1
FROM daily_prices p
LEFT JOIN daily_indicators i ON p.ts_code=i.ts_code AND p.trade_date=i.trade_date
LEFT JOIN sw_industry_members sw ON p.ts_code=sw.ts_code
WHERE p.trade_date>='{start_time}' AND p.trade_date<='{end_time}'
ORDER BY p.ts_code, p.trade_date"""
        raw = api.query(ch_sql)
        if raw is not None and not raw.empty:
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw = raw.drop_duplicates(subset=["instrument", "datetime"])
            raw = raw.set_index(["instrument", "datetime"])
            # 按 df 的索引对齐
            raw = raw.reindex(df.index)

            # sw_l1：Qlib 不足或数值编码时用 ClickHouse 覆盖
            ch_has_sw = "sw_l1" in raw.columns and raw["sw_l1"].notna().any()
            if ch_has_sw:
                df["sw_l1"] = raw["sw_l1"]
                if log:
                    log.info(f"[补充字段] ClickHouse: sw_l1覆盖完成 (非空{raw['sw_l1'].notna().sum():,}行)")

            # circ_mv：仅在 Qlib 数据不足时用 ClickHouse 覆盖
            qlib_mc_ok = False
            try:
                qlib_mc_ok = mc_ok  # Qlib 路径已执行，使用其判断结果
            except NameError:
                qlib_mc_ok = False  # Qlib 路径未执行（异常/未安装）
            if not qlib_mc_ok and "circ_mv" in raw.columns:
                df["circ_mv"] = raw["circ_mv"]
                if log:
                    log.info(f"[补充字段] ClickHouse: circ_mv覆盖完成 (非空{raw['circ_mv'].notna().sum():,}行)")
            elif qlib_mc_ok and log:
                log.info(f"[补充字段] 保留 Qlib circ_mv ({n_valid_mc:,}行)，不做 ClickHouse 覆盖")

            n_valid_mc = int(df["circ_mv"].notna().sum())
            n_valid_ind = int(df["sw_l1"].notna().sum())
            if log:
                log.info(f"[补充字段] ClickHouse: circ_mv={n_valid_mc:,}行有效, sw_l1={n_valid_ind:,}行有效 "
                         f"(共{len(instruments)}只股票)")
    except Exception as e:
        if log:
            log.warning(f"[补充字段] ClickHouse 加载失败: {e}")

    return df


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
    parser.add_argument("--pool", default="csi500",
                        help=f"股票池，可选: {', '.join(_AVAILABLE_POOLS)} (default: csi500)")
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

            # ── 按 --pool 过滤股票池 ──
            pool_stocks = _load_pool_stocks(args.pool)
            if pool_stocks:
                n_before = len(df)
                df = df[df.index.get_level_values("instrument").isin(pool_stocks)]
                logger.info(f"[股票池] {args.pool} 过滤: {n_before:,} → {len(df):,} 行 ({df.index.get_level_values('instrument').nunique()} 只)")
                if df.empty:
                    logger.warning(f"{name}: 股票池 {args.pool} 过滤后无数据，跳过")
                    continue

            # 合并标签
            df_label = evaluator._load_labels(args.start, args.end)
            if df_label is not None:
                df = df.join(df_label, how="inner")

            # 补充 mkt_cap + industry（场景压力测试/控制变量对冲必需）
            df = _enrich_with_extra_fields(df, args.start, args.end, logger=logger)

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
