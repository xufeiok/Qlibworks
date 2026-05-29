"""
数据同步脚本 - 从 ClickHouse 同步数据到本地

功能：
- 全量同步：首次初始化 Qlib 数据
- 增量同步：每日更新最新数据
- 财务数据同步：同步财务指标

用法：
    # 全量同步
    python -m scripts.data.sync full --start_date 2010-01-01 --end_date 2025-12-31
    
    # 增量同步
    python -m scripts.data.sync incremental
    
    # 同步财务数据
    python -m scripts.data.sync financial
"""
import sys
import argparse
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer


def sync_full(start_date: str, end_date: str):
    """全量同步"""
    print("=" * 60)
    print(f"开始全量同步：{start_date} - {end_date}")
    print("=" * 60)
    
    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)
        syncer.full_sync(start_date, end_date)


def sync_incremental():
    """增量同步"""
    print("=" * 60)
    print("开始增量同步")
    print("=" * 60)
    
    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)
        syncer.incremental_sync()


def sync_financial(start_date: str = None, end_date: str = None):
    """
    同步财务数据
    
    重要：财报数据使用公告日期（ann_date），而非期末日期（end_date）
    这是量化标准做法，确保使用实际发布日期而非财报期末日期
    
    Args:
        start_date: 开始日期（公告日期），默认从最早开始
        end_date: 结束日期（公告日期），默认到最新
    """
    print("=" * 60)
    print("开始同步财务数据")
    print("=" * 60)
    print("数据规范：强制使用公告日期（ann_date），不使用期末日期（end_date）")
    
    with QuantDataAPI() as api:
        df = api.get_financial_data(
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"获取到 {len(df)} 条财务数据")
        print(f"日期范围：{df['ann_date'].min()} 至 {df['ann_date'].max()}")
        print(f"股票数量：{df['ts_code'].nunique()} 只")
        
        # 保存为 Parquet 特征
        api.save_feature(
            df=df,
            name="financial_indicators",
            version="1.0",
            description="财务指标数据（使用公告日期 ann_date）",
            category="fundamental",
            date_column="ann_date"
        )
        
        print("财务数据同步完成")


def main():
    parser = argparse.ArgumentParser(description="数据同步脚本")
    subparsers = parser.add_subparsers(dest="command", help="同步类型")
    
    # 全量同步
    full_parser = subparsers.add_parser("full", help="全量同步")
    full_parser.add_argument("--start_date", type=str, required=True, help="开始日期 YYYY-MM-DD")
    full_parser.add_argument("--end_date", type=str, required=True, help="结束日期 YYYY-MM-DD")
    
    # 增量同步
    subparsers.add_parser("incremental", help="增量同步")
    
    # 财务数据同步
    subparsers.add_parser("financial", help="同步财务数据")
    
    args = parser.parse_args()
    
    if args.command == "full":
        sync_full(args.start_date, args.end_date)
    elif args.command == "incremental":
        sync_incremental()
    elif args.command == "financial":
        sync_financial()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
