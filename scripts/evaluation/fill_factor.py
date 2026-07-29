"""因子数据补齐脚本：增量补齐缺失时段/股票的因子值。

用法:
  python fill_factor.py --factor KDJ_K --fill-dates 2025-01-01 2025-12-31
  python fill_factor.py --factor KDJ_K --fill-stocks stocks_to_add.txt
  python fill_factor.py --factor KDJ_K --fill-stocks 000001.SZ,600000.SH
  python fill_factor.py --factor KDJ_K --rebuild
"""
import argparse, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd
from qlworks.evaluation.factor_store import FactorStore
from qlworks.evaluation.config import DEFAULT_CONFIG


def _parse_stock_spec(spec):
    """解析股票参数：可以是文件路径或逗号分隔的股票列表。"""
    if not spec: return None
    # 如果是文件且存在，按行读取
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as f:
            stocks = [line.strip() for line in f if line.strip()]
        print(f"  从文件读取 {len(stocks)} 只股票: {spec}")
        return stocks
    # 否则按逗号分隔
    stocks = [s.strip() for s in spec.split(",") if s.strip()]
    print(f"  解析股票列表: {len(stocks)} 只")
    return stocks


def _find_existing_parquet(store, factor_name):
    """从仓库查找已有因子数据。"""
    df = store.load_from_warehouse(factor_name)
    if df is None or df.empty:
        return None, None, set(), set()
    existing_stocks = set(df.index.get_level_values("instrument").unique())
    existing_dates = set(df.index.get_level_values("datetime").unique())
    return None, df, existing_stocks, existing_dates


def _merge_and_save(store, factor_name, existing_df, new_df, label):
    """合并新旧数据（仓库模式），去重后保存。"""
    if new_df is None or new_df.empty:
        print(f"  {label}: 无新增数据")
        return 0

    # 检查重复
    before = len(existing_df)
    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    added = len(combined) - before

    if added > 0:
        # 写入 warehouse
        store._save_warehouse_chunk(factor_name, combined)
        store._update_warehouse_meta(factor_name)
        print(f"  {label}: +{added} 行 (共 {len(combined)} 行)")
    else:
        print(f"  {label}: 已是最新, 无新增")

    return added


def main():
    parser = argparse.ArgumentParser(description="因子数据补齐")
    parser.add_argument("--factor", required=True, help="因子名称")
    parser.add_argument("--expr", default="", help="因子表达式（缓存未命中时需要）")
    parser.add_argument("--fill-dates", nargs=2, metavar=("START", "END"),
                        help="补齐指定日期范围的因子值")
    parser.add_argument("--fill-stocks", metavar="STOCKS",
                        help="补齐缺失的股票。可以是文件路径或逗号分隔的股票代码列表")
    parser.add_argument("--rebuild", action="store_true",
                        help="完整重算并覆盖（全量重新计算后合并）")
    args = parser.parse_args()

    if not (args.fill_dates or args.fill_stocks or args.rebuild):
        parser.print_help()
        print("\n请指定补齐方式: --fill-dates, --fill-stocks 或 --rebuild")
        return

    store = FactorStore(DEFAULT_CONFIG)
    existing_df, ex_stocks, ex_dates = None, set(), set()
    warehouse_df = store.load_from_warehouse(args.factor)
    if warehouse_df is not None and not warehouse_df.empty:
        existing_df = warehouse_df
        ex_stocks = set(existing_df.index.get_level_values("instrument").unique())
        ex_dates = set(existing_df.index.get_level_values("datetime").unique())

    total_added = 0

    # ── 完整重算 ──
    if args.rebuild:
        print(f"完整重算: {args.factor}")
        new_df = store._compute(args.factor, args.expr)
        if new_df is None or new_df.empty:
            print("  计算结果为空, 跳过")
            return
        if existing_df is None:
            print("  仓库无数据, 直接写入")
            store._save_warehouse_chunk(args.factor, new_df)
            store._update_warehouse_meta(args.factor)
            total_added = len(new_df)
        else:
            added = _merge_and_save(store, args.factor, existing_df, new_df, "rebuild")
            total_added += added

    # ── 补齐日期 ──
    if args.fill_dates:
        sd, ed = args.fill_dates
        print(f"补齐日期: {sd} ~ {ed}")
        new_df = store._compute(args.factor, args.expr, sd, ed)
        if new_df is not None:
            if existing_df is not None:
                added = _merge_and_save(store, args.factor, existing_df, new_df, "fill-dates")
                total_added += added
            else:
                store._save_warehouse_chunk(args.factor, new_df)
                store._update_warehouse_meta(args.factor)
                added = len(new_df)
                print(f"  fill-dates: +{added} 行 (新仓库)")
                total_added += added
        elif new_df is not None:
            store._save_cache(args.factor, new_df)
            print(f"  未找到已评测 parquet, 结果已缓存 ({len(new_df)} 行)")

    # ── 补齐股票 ──
    if args.fill_stocks:
        target_stocks = _parse_stock_spec(args.fill_stocks)
        if not target_stocks:
            print("  股票列表为空, 跳过")
            return

        if existing_df is None:
            print("  仓库无数据, 先计算全量")
            store.compute_to_warehouse(args.factor, args.expr)
            existing_df = store.load_from_warehouse(args.factor)
            if existing_df is None:
                return

        # 找出缺失的股票
        ex_stocks = set(existing_df.index.get_level_values("instrument").unique())
        missing = [s for s in target_stocks if s not in ex_stocks]
        if not missing:
            print("  所有指定股票均已存在")
            return

        print(f"  缺失 {len(missing)} 只股票, 计算中...")

        # 分批处理缺失股票，每批 50 只控制内存
        batch_size = 50
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            print(f"    批次 {i//batch_size + 1}/{(len(missing)-1)//batch_size + 1}: {len(batch)} 只")
            new_df = store._compute(args.factor, args.expr, stocks=batch)
            if new_df is not None and not new_df.empty:
                added = _merge_and_save(store, args.factor, existing_df, new_df, f"stocks batch")
                total_added += added
                # 更新 existing_df 以用于后续批次去重
                existing_df = pd.read_parquet(pf)

    if total_added > 0:
        print(f"\n补齐完成! 共新增 {total_added} 行数据")
    else:
        print("\n无需补齐")


if __name__ == "__main__":
    main()