"""
数据同步脚本 - 从 ClickHouse 同步 Qlib 内核数据到本地

同步到 Qlib bin 格式的字段（仅 8 个）：
  - OHLCV: open, high, low, close, volume, amount
  - 市值:  total_mv, circ_mv
  - 行业:  sw_l1, sw_l2, sw_l3

其他因子（财务/估值/动量/自定义）已迁移为 DuckDB + Parquet 预计算，
参见 qlworks.features.factor_cache.FactorCache。

用法：
    # 全量同步（首次初始化）
    python -m scripts.data.sync full --start_date 2010-01-01 --end_date 2025-12-31

    # 增量同步（每日更新最新数据）
    python -m scripts.data.sync incremental

    # 智能同步（自动检测全量/增量 + 按上市日期定制，推荐首次使用）
    python -m scripts.data.sync auto

    # 独立同步申万行业
    python -m scripts.data.sync industry
"""
import sys
import argparse

_project_root = str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
# qlworks 包在 src/ 目录下，需要单独加入
sys.path.insert(0, str(__import__('pathlib').Path(_project_root) / "src"))

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


def main():
    parser = argparse.ArgumentParser(description="数据同步脚本")
    subparsers = parser.add_subparsers(dest="command", help="同步类型")

    full_parser = subparsers.add_parser("full", help="全量同步（所有股票统一起始日期）")
    full_parser.add_argument("--start_date", type=str, required=True, help="开始日期 YYYY-MM-DD")
    full_parser.add_argument("--end_date", type=str, required=True, help="结束日期 YYYY-MM-DD")

    subparsers.add_parser("incremental", help="增量同步（每日更新最新数据）")
    subparsers.add_parser("auto", help="智能同步（自动检测模式 + 按上市日期定制，推荐首次使用）")
    subparsers.add_parser("instruments", help="仅刷新 instruments 文件（all.txt/csi500.txt 等退市日/成分股信息）")
    subparsers.add_parser("industry", help="同步申万行业数据（sw_l1/sw_l2/sw_l3，独立下载，先清后写+验证）")

    args = parser.parse_args()

    if args.command == "full":
        sync_full(args.start_date, args.end_date)
    elif args.command == "incremental":
        sync_incremental()
    elif args.command == "auto":
        sync_auto()
    elif args.command == "instruments":
        sync_instruments()
    elif args.command == "industry":
        sync_industry()
    else:
        parser.print_help()


def sync_industry():
    """独立同步申万行业数据（检测→清旧→下载→写入→验证）"""
    print("=" * 60)
    print("独立模式：申万行业数据同步")
    print("=" * 60)
    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)
        syncer.sync_industry(verify=True)


def sync_instruments():
    """仅刷新 instruments 文件，从 stock_basic 拉取最新上市/退市日期"""
    print("=" * 60)
    print("instruments 文件刷新（退市日期/成分股 PIT 信息）")
    print("=" * 60)
    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)
        syncer.sync_instruments_only()
    print("=" * 60)
    print("instruments 文件刷新完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()