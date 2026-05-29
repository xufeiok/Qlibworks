"""
数据全面核查与补全脚本

功能：
1. 检查ClickHouse中所有数据表的数据完整性
2. 识别2010年之前缺失的数据
3. 使用Tushare补全缺失数据
4. 输出详细的核查报告
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import tushare as ts

from qlworks.data.api import QuantDataAPI

# 设置Tushare Token
ts.set_token('18dd374714956ab83ae5c2028613bee423ce620124e490bf0c35fed2')
pro = ts.pro_api()

def check_all_tables():
    """检查所有数据表的时间范围"""
    api = QuantDataAPI()
    
    tables = {
        'daily_prices': 'trade_date',
        'daily_indicators': 'trade_date',
        'daily_adj_factors': 'trade_date',
        'financial_indicators': 'ann_date',
        'stock_universe': 'list_date',
        'index_daily': 'trade_date',
        'money_flow': 'trade_date',
        'income_statement': 'ann_date',
        'balance_sheet': 'ann_date',
        'cashflow_statement': 'ann_date'
    }
    
    print("=" * 80)
    print("【ClickHouse 数据表时间范围统计】")
    print("=" * 80)
    
    results = {}
    for table, date_col in tables.items():
        try:
            sql = f"SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date, COUNT(*) as total_rows FROM {table}"
            r = api.query(sql)
            min_date = r.iloc[0, 0]
            max_date = r.iloc[0, 1]
            total_rows = r.iloc[0, 2]
            
            results[table] = {
                'min_date': min_date,
                'max_date': max_date,
                'total_rows': total_rows
            }
            
            print(f"\n【{table}】")
            print(f"  时间范围: {min_date} ~ {max_date}")
            print(f"  数据行数: {total_rows:,}")
            
        except Exception as e:
            print(f"\n【{table}】")
            print(f"  查询失败: {e}")
            results[table] = {'error': str(e)}
    
    return results

def get_tushare_data_range():
    """获取Tushare API可获取的数据范围"""
    print("\n" + "=" * 80)
    print("【Tushare API 数据范围测试】")
    print("=" * 80)
    
    try:
        # 测试日线数据最早日期
        df = pro.daily(ts_code='600000.SH', start_date='19900101', end_date='19901231')
        if not df.empty:
            print(f"✅ 日线数据可获取: 1990年数据存在 ({len(df)}条)")
        else:
            print("⚠️ 1990年日线数据为空")
            
        # 测试2000年数据
        df = pro.daily(ts_code='600000.SH', start_date='20000101', end_date='20001231')
        if not df.empty:
            print(f"✅ 日线数据可获取: 2000年数据存在 ({len(df)}条)")
        else:
            print("⚠️ 2000年日线数据为空")
            
        # 测试2010年数据
        df = pro.daily(ts_code='600000.SH', start_date='20100101', end_date='20101231')
        if not df.empty:
            print(f"✅ 日线数据可获取: 2010年数据存在 ({len(df)}条)")
        else:
            print("⚠️ 2010年日线数据为空")
            
    except Exception as e:
        print(f"❌ Tushare API 调用失败: {e}")

def find_missing_stocks():
    """找出数据不完整的股票"""
    api = QuantDataAPI()
    
    print("\n" + "=" * 80)
    print("【查找数据不完整的股票】")
    print("=" * 80)
    
    # 获取股票列表
    stocks = api.get_stock_list()
    print(f"股票总数: {len(stocks)}")
    
    # 找出上市日期在2010年之前但数据开始于2010年之后的股票
    early_stocks = stocks[stocks['list_date'] < '2010-01-01']
    print(f"2010年前上市的股票数: {len(early_stocks)}")
    
    # 抽查部分股票
    sample = early_stocks.sample(min(5, len(early_stocks)))
    
    print("\n【2010年前上市但数据不完整的股票示例】")
    for _, row in sample.iterrows():
        ts_code = row['ts_code']
        list_date = row['list_date']
        
        r = api.query(f"SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM daily_prices WHERE ts_code = '{ts_code}'")
        min_date = r.iloc[0, 0]
        
        if min_date > list_date:
            print(f"\n股票: {ts_code}")
            print(f"  上市日期: {list_date}")
            print(f"  数据开始: {min_date}")
            print(f"  缺失天数: {pd.to_datetime(min_date) - pd.to_datetime(list_date)}")

def main():
    print("=" * 80)
    print("【数据全面核查与补全脚本】")
    print(f"执行时间: {datetime.now()}")
    print("=" * 80)
    
    # 1. 检查所有数据表
    table_stats = check_all_tables()
    
    # 2. 测试Tushare API
    get_tushare_data_range()
    
    # 3. 查找数据不完整的股票
    find_missing_stocks()
    
    print("\n" + "=" * 80)
    print("【核查完成】")
    print("=" * 80)
    
    # 分析结果
    daily_prices_min = table_stats['daily_prices']['min_date']
    if daily_prices_min > '2010-01-01':
        print(f"\n⚠️ 发现问题：daily_prices 最早日期为 {daily_prices_min}，晚于2010年")
        print("建议：需要补全2010年之前的数据")
    else:
        print(f"\n✅ daily_prices 数据完整，最早日期: {daily_prices_min}")

if __name__ == "__main__":
    main()
