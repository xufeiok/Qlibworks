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

from qlworks.evaluation import FactorEvaluator, FactorStore, LifecycleManager, DEFAULT_CONFIG
from qlworks.factors import FactorLibraryManager


def main():
    parser = argparse.ArgumentParser(description="批量重评 + 生命周期审核")
    parser.add_argument("--factor", help="指定因子名")
    parser.add_argument("--all", action="store_true", help="重评仓库中所有因子")
    parser.add_argument("--layer", choices=["core", "satellite", "archive"],
                        help="重评指定层级的因子")
    parser.add_argument("--pool", default="csi500")
    parser.add_argument("--start", default="2020-01-01")
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

        # 找到因子表达式
        expr = ""
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            try:
                cfg = m.load_strategy_config(s)
                for fd in cfg.get("factors", []):
                    if fd.get("name") == name:
                        e = fd.get("expression", "")
                        expr = str(e.get("qlib", e)) if isinstance(e, dict) else str(e)
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
                store.compute_to_warehouse(name, expr, args.start, args.end)

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

            # [AQR] 执行完整评测
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
