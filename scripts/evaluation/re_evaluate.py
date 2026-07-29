#!/usr/bin/env python3
"""
批量重评 + 生命周期审核（v3.0 仓库适配版）。

从仓库读取因子数据，执行完整评测流水线，更新注册表和等级引用。
"""
import argparse
from dataclasses import replace
import json
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
logger = logging.getLogger("re_evaluate")

import clickhouse_connect
import pandas as pd
from qlworks.config import QLIB_DATA_DIR, CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
from qlworks.evaluation import FactorStore, DEFAULT_CONFIG
from qlworks.evaluation.runner import FactorEvaluator
from qlworks.evaluation.lifecycle import LifecycleManager
from qlworks.factors import FactorLibraryManager

# ── 自动发现 instruments 目录下的可用股票池 ──
_INSTRUMENTS_DIR = QLIB_DATA_DIR / "instruments"
AVAILABLE_POOLS = sorted(f.stem for f in _INSTRUMENTS_DIR.glob("*.txt") if f.is_file())


def main():
    parser = argparse.ArgumentParser(description="批量重评 + 生命周期审核")
    parser.add_argument("--factor", help="指定因子名")
    parser.add_argument("--all", action="store_true", help="重评仓库中所有因子")
    parser.add_argument("--layer", choices=["core", "satellite", "archive"],
                        help="重评指定层级的因子")
    parser.add_argument("--pool", default="csi500", choices=AVAILABLE_POOLS,
                        help=f"股票池，可选: {', '.join(AVAILABLE_POOLS)} (default: csi500)")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--force-compute", action="store_true",
                        help="强制从 ClickHouse 重新计算（不读仓库缓存）")
    args = parser.parse_args()

    if not (args.factor or args.layer or args.all):
        parser.print_help()
        return

    config = replace(
        DEFAULT_CONFIG,
        instruments=args.pool,
        start_time=args.start,
        end_time=args.end,
    )
    ev = FactorEvaluator(config)
    store = FactorStore(config)
    lc = LifecycleManager(config.registry_dir)
    m = FactorLibraryManager()

    # ── 预加载辅助数据（流通市值 + 行业分类），用于场景分析图表 ──
    _aux_data = {"circ_mv": None, "sw_l1": None}
    _ch = None
    try:
        _ch = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT,
            user=CH_USER, password=CH_PASSWORD,
            database=CH_DATABASE, connect_timeout=10,
        )
        # 流通市值（注：daily_indicators.circ_mv 仅覆盖到 2020 年）
        _mv_df = _ch.query_df(f"""
            SELECT d.ts_code, d.trade_date, d.circ_mv
            FROM daily_indicators d
            WHERE d.trade_date >= '{args.start}' AND d.trade_date <= '{args.end}'
        """)
        if not _mv_df.empty:
            _mv_df = _mv_df.rename(columns={"ts_code": "instrument", "trade_date": "datetime"})
            _mv_df["instrument"] = _mv_df["instrument"].str.upper()
            _mv_df["datetime"] = pd.to_datetime(_mv_df["datetime"])
            _aux_mv = _mv_df.set_index(["instrument", "datetime"])["circ_mv"].astype("float64")
            # [Fix] 统一索引 dtype（warehouse=object, daily_indicators=string[python]），避免对齐失败
            _aux_mv.index = _aux_mv.index.set_levels(
                _aux_mv.index.levels[0].astype(object), level="instrument"
            )
            _aux_data["circ_mv"] = _aux_mv
            logger.info(f"  预加载流通市值: {len(_aux_mv):,} 行, "
                        f"日期: {_aux_mv.index.get_level_values('datetime').min().date()} ~ "
                        f"{_aux_mv.index.get_level_values('datetime').max().date()}")

        # 行业分类（sw_industry_members 按 instrument 去重，取最新记录）
        _ind_df = _ch.query_df("""
            SELECT ts_code, l1_code AS sw_l1
            FROM sw_industry_members
            ORDER BY ts_code, in_date DESC
            LIMIT 1 BY ts_code
        """)
        if not _ind_df.empty:
            _ind_df = _ind_df.rename(columns={"ts_code": "instrument"})
            _ind_df["instrument"] = _ind_df["instrument"].str.upper()
            _ind_df = _ind_df.drop_duplicates(subset="instrument")
            _aux_data["sw_l1"] = _ind_df.set_index("instrument")["sw_l1"]
            logger.info(f"  预加载行业分类: {len(_aux_data['sw_l1']):,} 只股票")
    except Exception as e:
        logger.warning(f"  辅助数据加载跳过（不影响核心评测）: {e}")
    finally:
        if _ch is not None:
            _ch.close()

    # ── 收集需要重评的因子 ──
    factors = []
    if args.factor:
        factors.append(args.factor)
    elif args.layer:
        ref_dir = Path(config.factors_dir) / args.layer
        if ref_dir.exists():
            for ref_file in ref_dir.glob("*.ref.json"):
                with open(ref_file) as f:
                    meta = json.load(f)
                factors.append(meta.get("factor_name", ref_file.stem))
    elif args.all:
        # 从仓库获取（如果仓库为空则从 YAML 获取）
        warehouse_factors = store.list_warehouse_factors()
        if warehouse_factors:
            factors = warehouse_factors
        else:
            logger.info("仓库为空，从 YAML 因子库收集...")
            for s in [s for s in m.list_strategies() if "dictionary" not in s]:
                try:
                    cfg = m.load_strategy_config(s)
                    for fd in cfg.get("factors", []):
                        name = fd.get("name")
                        if name:
                            factors.append(name)
                except Exception:
                    pass

    factors = sorted(set(factors))
    logger.info(f"待重评因子: {len(factors)} 个")

    results = {"core": 0, "satellite": 0, "archive": 0, "failed": 0}

    for i, name in enumerate(factors, 1):
        print(f"\n[{i}/{len(factors)}] {name}")

        # 找到因子表达式（同时提取 qlib 和 duckdb 两种格式）
        expr = ""
        duckdb_expr = ""
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            try:
                cfg = m.load_strategy_config(s)
                for fd in cfg.get("factors", []):
                    if fd.get("name") == name:
                        e = fd.get("expression", "")
                        if isinstance(e, dict):
                            expr = str(e.get("qlib", ""))
                            duckdb_expr = str(e.get("duckdb", ""))
                        else:
                            expr = str(e)
                        break
            except Exception:
                pass

        if not expr:
            # 因子可能在仓库 meta 中有表达式，但这里简化处理
            warehouse_meta = store.get_warehouse_meta(name)
            if warehouse_meta is None:
                print(f"  跳过: 无表达式且不在仓库中")
                results["failed"] += 1
                continue
            # 有仓库数据但没表达式 → 跳过评测（只更新注册表）
            print(f"  仓库有数据但无表达式，跳过评测")
            continue

        try:
            # [Bloomberg Data Pipeline] 确保数据在仓库中
            if args.force_compute or not store.get_warehouse_meta(name):
                store.compute_to_warehouse(name, expr, args.start, args.end,
                                           duckdb_expr=duckdb_expr or None)

            # 从仓库加载（仓库列名为 value，需重命名为因子名）
            df = store.load_from_warehouse(name, args.start, args.end)
            if df is None or df.empty:
                print(f"  无数据，跳过")
                results["failed"] += 1
                continue
            df = df.rename(columns={"value": name})

            # 合并标签
            df_label = ev._load_labels(args.start, args.end)
            if df_label is not None:
                df = df.join(df_label, how="inner")

            # ── 股票池过滤（真正按 --pool 指定的板块过滤） ──
            _pool_path = QLIB_DATA_DIR / "instruments" / f"{args.pool}.txt"
            if _pool_path.exists():
                _pool_codes = set()
                with open(_pool_path) as _f:
                    for _l in _f:
                        _parts = _l.strip().split("\t")
                        if _parts:
                            _pool_codes.add(_parts[0].upper())
                if _pool_codes:
                    _codes = df.index.get_level_values("instrument").str.upper()
                    _before = len(df)
                    df = df[_codes.isin(_pool_codes)]
                    if len(df) < _before:
                        logger.info(f"  股票池 ({args.pool}) 过滤: {_before} → {len(df)} 行")

            # ── 合并辅助数据（流通市值 + 行业），用于场景分析图表 ──
            if _aux_data.get("circ_mv") is not None:
                aux_mv = _aux_data["circ_mv"].astype("float64")
                # [Fix] 统一索引 dtype: warehouse 为 object, daily_indicators 为 string[python]
                # dtype 不匹配会导致 MultiIndex 对齐失败（所有值变 NaN）
                inst_dtype = df.index.get_level_values("instrument").dtype
                if aux_mv.index.get_level_values("instrument").dtype != inst_dtype:
                    aux_mv.index = aux_mv.index.set_levels(
                        aux_mv.index.levels[0].astype(inst_dtype), level="instrument"
                    )
                df["circ_mv"] = aux_mv
            if _aux_data.get("sw_l1") is not None:
                df["sw_l1"] = df.index.get_level_values("instrument").map(_aux_data["sw_l1"])

            # [AQR] 执行完整评测
            # ── 诊断：检查合并后的数据列状态 ──
            _col_status = {c: df[c].notna().sum() for c in df.columns}
            logger.info(f"  df列状态: {_col_status}")
            logger.info(f"  df行数: {len(df)}, circ_mv非空: {df['circ_mv'].notna().sum() if 'circ_mv' in df.columns else 'N/A'}, sw_l1非空: {df['sw_l1'].notna().sum() if 'sw_l1' in df.columns else 'N/A'}")
            result = ev.evaluate(name, df.reset_index(), category="satellite")
            q = result["qual_result"]
            tier = q["tier"]

            results[tier] = results.get(tier, 0) + 1
            print(f"  → tier={tier:10s} IC={result['ic_stats']['ic_mean']:.4f} ICIR={result['ic_stats']['icir']:.2f} 评分={q['composite_score']:.1f}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results["failed"] += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  批量重评完成!")
    print(f"  core: {results.get('core', 0)} | satellite: {results.get('satellite', 0)} | archive: {results.get('archive', 0)} | failed: {results.get('failed', 0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
