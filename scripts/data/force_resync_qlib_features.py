# -*- coding: utf-8 -*-
"""
强制重新同步 Qlib features 中指定时间范围的数据（全覆盖模式）。

解决 Qlib 中 circ_mv/OHLCV 数据在 2025-03 之后陈旧的问题。
对所有主板股票重新执行全量写入，覆盖原有数据。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer

def main():
    print("=" * 60)
    print("Qlib 强制全量同步 — 覆盖 circ_mv/OHLCV 陈旧数据")
    print("=" * 60)

    api = QuantDataAPI()
    sync = QlibSynchronizer(api)

    # 强制获取所有主板股票（包括 Qlib 中已有的）
    stocks = sync._get_main_board_stocks()
    print(f"\n主板股票: {len(stocks)} 只")

    # 获取日历
    calendar_list, calendar_map = sync._get_calendar_list()
    print(f"交易日历: {len(calendar_list)} 天")

    # 2025-01-01 ~ 2026-07-03 全量写入（覆盖模式）
    print("\n执行覆盖同步: 2025-01-01 ~ 2026-07-03...")
    sync._sync_features(
        stocks, calendar_list, calendar_map,
        "2025-01-01", "2026-07-03",
        append=False,  # 覆盖模式
    )

    print("\n" + "=" * 60)
    print("同步完成！Qlib features 已与 ClickHouse 一致。")
    print("=" * 60)
    print("注意：")
    print("  1. day.txt 日历（3985天）与 ClickHouse 日历（8622天）不一致")
    print("  2. 建议后续用 sync.sync_instruments_only() 更新股票池文件")
    print("  3. 建议后续用 sync.sync_industry() 更新行业数据")

if __name__ == "__main__":
    main()
