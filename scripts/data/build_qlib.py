"""
Qlib 数据构建脚本 - 按上市日期智能同步

规则：
1. 2010-01-01 之前上市的股票 → 从 2010-01-01 开始
2. 2010-01-01 之后上市的股票 → 从上市日期开始
3. 截止日期与 ClickHouse 最新交易日一致
4. 价格使用前复权（从 daily_adj_factors 计算）
5. 财务指标使用公告日期（ann_date）

用法：
    python scripts/data/build_qlib.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import clickhouse_connect
from qlworks.config import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
from qlworks.data.qlib_sync import QlibSynchronizer
from qlworks.data.api import QuantDataAPI


def build_qlib_data():
    print("=" * 60)
    print("Qlib 智能数据构建 - 按上市日期定制同步起点")
    print("=" * 60)

    # 连接 ClickHouse 获取元数据
    print("\n[1] 连接 ClickHouse 获取每只股票的上市日期...")
    ch = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        username=CH_USER, password=CH_PASSWORD,
        database=CH_DATABASE
    )

    # 获取最后交易日
    ch_end = ch.query("SELECT max(trade_date) FROM daily_prices").result_rows[0][0]
    end_date = str(ch_end)[:10]
    START_BOUNDARY = '2010-01-01'
    print(f"  截止日期: {end_date}")
    print(f"  最早边界: {START_BOUNDARY}")

    # 获取每只股票的上市日期
    df_listing = ch.query_df("""
        SELECT ts_code,
               min(trade_date) as first_date
        FROM daily_prices
        GROUP BY ts_code
        ORDER BY ts_code
    """)
    print(f"  共 {len(df_listing)} 只股票")

    # 确定每只股票的开始日期
    instruments_dict = {}
    for _, row in df_listing.iterrows():
        first = str(row['first_date'])[:10]
        if first < START_BOUNDARY:
            instruments_dict[row['ts_code']] = START_BOUNDARY
        else:
            instruments_dict[row['ts_code']] = first

    stocks_before = sum(1 for v in instruments_dict.values() if v == START_BOUNDARY)
    stocks_after = len(instruments_dict) - stocks_before
    print(f"  2010年前上市（从{START_BOUNDARY}开始）: {stocks_before} 只")
    print(f"  2010年后上市（从上市日开始）: {stocks_after} 只")

    ch.close()

    # 使用 QlibSynchronizer 执行同步
    print("\n[2] 使用 QlibSynchronizer 执行同步...")
    stocks = list(instruments_dict.keys())
    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)
        syncer.full_sync(
            start_date=START_BOUNDARY,
            end_date=end_date,
            instruments=stocks,
            instruments_dict=instruments_dict
        )

    print("\n" + "=" * 60)
    print("Qlib 数据构建完成！")
    print("=" * 60)


if __name__ == '__main__':
    build_qlib_data()
