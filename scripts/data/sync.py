"""
数据同步脚本 - 从 ClickHouse 同步数据到本地

功能：
- 全量同步：首次初始化 Qlib 数据（含去重验证）
- 增量同步：每日更新最新数据（智能检测 + 空缺补填 + 财务数据 + 去重验证）
- 财务数据同步：同步财务指标
- 智能同步：自动检测模式 + 按上市日期定制同步起点（一键推荐）

用法：
    # 全量同步（所有股票统一起始日期）
    python -m scripts.data.sync full --start_date 2010-01-01 --end_date 2025-12-31

    # 增量同步
    python -m scripts.data.sync incremental

    # 智能同步（自动检测全量/增量 + 按上市日期定制，推荐首次使用）
    python -m scripts.data.sync auto

    # 同步财务数据
    python -m scripts.data.sync financial
"""
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from pathlib import Path
from qlworks.config import QLIB_DATA_DIR
from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer


def _has_existing_features() -> bool:
    """检测 features 目录是否已有有效数据"""
    features_dir = Path(QLIB_DATA_DIR) / "features"
    if not features_dir.exists():
        return False
    for d in features_dir.iterdir():
        if d.is_dir() and list(d.glob("*.day.bin")):
            return True
    return False


def _build_instruments_dict(api, start_boundary: str = "2010-01-01") -> dict:
    """
    构建按上市日期定制的股票起始终止字典。

    2010-01-01 之前上市的股票 → 从 2010-01-01 开始
    2010-01-01 之后上市的股票 → 从上市日期开始
    """
    df_listing = api.query("""
        SELECT ts_code, min(trade_date) as first_date
        FROM daily_prices GROUP BY ts_code ORDER BY ts_code
    """)
    instruments_dict = {}
    for _, row in df_listing.iterrows():
        first = str(row['first_date'])[:10]
        instruments_dict[row['ts_code']] = start_boundary if first < start_boundary else first
    return instruments_dict, df_listing


def sync_full(start_date: str, end_date: str):
    """全量同步"""
    print("=" * 60)
    print(f"开始全量同步（含去重验证）：{start_date} - {end_date}")
    print("=" * 60)

    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)
        syncer.full_sync(start_date, end_date)


def sync_incremental():
    """增量同步（智能检测 + 空缺补填 + 财务数据 + 去重验证）"""
    print("=" * 60)
    print("开始增量同步 - 智能检测 + 空缺补填 + 财务数据 + 去重")
    print("=" * 60)

    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)
        syncer.incremental_sync(instruments_dict=None)


def sync_auto():
    """
    智能同步 - 自动检测模式 + 按上市日期定制同步起点

    规则：
    1. 自动检测：已有数据则增量，否则全量
    2. 2010-01-01 之前上市的股票从 2010-01-01 开始
    3. 2010-01-01 之后上市的股票从上市日期开始
    4. 截止日期从 ClickHouse 最新交易日自动获取
    """
    print("=" * 60)
    print("Qlib 智能数据构建 - 自动选择同步模式")
    print("=" * 60)

    with QuantDataAPI() as api:
        ch_end = api.query("SELECT max(trade_date) FROM daily_prices")
        end_date = str(ch_end.iloc[0, 0])[:10]
        START_BOUNDARY = "2010-01-01"
        print(f"[1] 截止日期: {end_date}, 最早边界: {START_BOUNDARY}")

        instruments_dict, df_listing = _build_instruments_dict(api, START_BOUNDARY)
        print(f"    共 {len(df_listing)} 只股票")

        stocks_before = sum(1 for v in instruments_dict.values() if v == START_BOUNDARY)
        stocks_after = len(instruments_dict) - stocks_before
        print(f"    2010年前上市(从{START_BOUNDARY}开始): {stocks_before} 只")
        print(f"    2010年后上市(从上市日开始): {stocks_after} 只")

        has_data = _has_existing_features()
        mode = "增量" if has_data else "全量"
        print(f"\n[2] 自动选择【{mode}同步】模式")

        stocks = list(instruments_dict.keys())
        syncer = QlibSynchronizer(api)
        if has_data:
            syncer.incremental_sync(instruments_dict=instruments_dict)
        else:
            syncer.full_sync(
                start_date=START_BOUNDARY,
                end_date=end_date,
                instruments=stocks,
                instruments_dict=instruments_dict,
            )

    print("\n" + "=" * 60)
    print("Qlib 智能数据构建完成！")
    print("=" * 60)


def sync_financial(start_date: str = None, end_date: str = None):
    """
    同步财务数据

    重要：财报数据使用公告日期（ann_date），而非期末日期（end_date）
    """
    print("=" * 60)
    print("开始同步财务数据")
    print("=" * 60)
    print("数据规范：强制使用公告日期（ann_date），不使用期末日期（end_date）")

    with QuantDataAPI() as api:
        df = api.get_financial_data(start_date=start_date, end_date=end_date)

        print(f"获取到 {len(df)} 条财务数据")
        print(f"日期范围：{df['ann_date'].min()} 至 {df['ann_date'].max()}")
        print(f"股票数量：{df['ts_code'].nunique()} 只")

        api.save_feature(
            df=df,
            name="financial_indicators",
            version="1.0",
            description="财务指标数据（使用公告日期 ann_date）",
            category="fundamental",
            date_column="ann_date",
        )

        print("财务数据同步完成")


def main():
    parser = argparse.ArgumentParser(description="数据同步脚本")
    subparsers = parser.add_subparsers(dest="command", help="同步类型")

    full_parser = subparsers.add_parser("full", help="全量同步（所有股票统一起始日期）")
    full_parser.add_argument("--start_date", type=str, required=True, help="开始日期 YYYY-MM-DD")
    full_parser.add_argument("--end_date", type=str, required=True, help="结束日期 YYYY-MM-DD")

    subparsers.add_parser("incremental", help="增量同步（每日更新最新数据）")
    subparsers.add_parser("auto", help="智能同步（自动检测模式 + 按上市日期定制，推荐首次使用）")
    subparsers.add_parser("financial", help="同步财务数据")

    args = parser.parse_args()

    if args.command == "full":
        sync_full(args.start_date, args.end_date)
    elif args.command == "incremental":
        sync_incremental()
    elif args.command == "auto":
        sync_auto()
    elif args.command == "financial":
        sync_financial()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()