# -*- coding: utf-8 -*-
"""
更新 Qlib features 数据，使与 ClickHouse 一致。

执行增量同步：只同步 Qlib 缺失的最新日期数据（OHLCV + 市值）。
"""
import sys
from pathlib import Path

# 确保项目在路径中
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer
from qlworks.config import QLIB_DATA_DIR

def main():
    print("=" * 60)
    print("Qlib 数据增量同步 — 从 ClickHouse 更新到 Qlib")
    print("=" * 60)

    # 1. 连接 API
    print("\n[1] 连接数据源...")
    api = QuantDataAPI()

    # 2. 创建同步器
    sync = QlibSynchronizer(api)

    # 3. 执行增量同步（只同步 Qlib 缺失的最新数据）
    print("\n[2] 执行增量同步...")
    try:
        sync.incremental_sync()
    except Exception as e:
        print(f"\n增量同步失败: {e}")
        print("尝试全量同步最近1年数据...")
        sync.full_sync("2025-01-01", "2026-12-31")

    # 4. 补充行业数据
    print("\n[3] 同步行业数据...")
    try:
        sync.sync_industry()
    except Exception as e:
        print(f"行业数据同步失败: {e}")

    # 5. 刷新 instruments
    print("\n[4] 刷新股票池...")
    try:
        sync.sync_instruments_only()
    except Exception as e:
        print(f"刷新股票池失败: {e}")

    print("\n" + "=" * 60)
    print("同步完成！")
    print(f"Qlib 数据目录: {QLIB_DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
