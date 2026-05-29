"""
数据补全脚本 - 使用Tushare补全缺失数据

需要补全：
1. index_daily: 补充2010年前数据
2. money_flow: 补充2019年前数据
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

def fill_index_daily():
    """补全指数日线数据"""
    api = QuantDataAPI()
    
    print("【补全指数日线数据】")
    
    # 获取当前数据范围
    r = api.query("SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM index_daily")
    current_min = r.iloc[0, 0]
    current_max = r.iloc[0, 1]
    
    print(f"当前数据范围: {current_min} ~ {current_max}")
    
    # 目标日期范围（从1990年开始）
    target_start = pd.to_datetime('1990-01-01')
    current_min_date = pd.to_datetime(current_min)
    
    if current_min_date <= target_start:
        print("✅ 指数数据已经完整")
        return
    
    print(f"需要补充: 1990-01-01 ~ {current_min_date.date()}")
    
    # 获取指数列表
    df_index = api.query("SELECT ts_code, name FROM index_basic WHERE market IN ('SSE', 'SZSE')")
    print(f"待补全指数数量: {len(df_index)}")
    
    # 分批获取数据
    total_inserted = 0
    for _, row in df_index.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        try:
            # 获取该指数的历史数据
            df = pro.index_daily(
                ts_code=ts_code,
                start_date='19900101',
                end_date=str(current_min).split(' ')[0].replace('-', '')
            )
            
            if not df.empty:
                # 使用INSERT语句批量插入
                insert_sql = """
                    INSERT INTO index_daily (ts_code, trade_date, close, open, high, low, pre_close, change, pct_chg, vol, amount)
                    VALUES {}
                """
                
                # 构建VALUES子句
                values = []
                for _, row in df.iterrows():
                    trade_date = row['trade_date']  # YYYYMMDD格式字符串
                    values.append(f"('{row['ts_code']}', '{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}', "
                                  f"{row['close']}, {row['open']}, {row['high']}, {row['low']}, "
                                  f"{row['pre_close']}, {row['change']}, {row['pct_chg']}, {row['vol']}, {row['amount']})")
                
                sql = insert_sql.format(', '.join(values))
                api.query(sql, use_cache=False)
                total_inserted += len(df)
                print(f"  ✅ {ts_code} {name}: 补充 {len(df)} 条")
            else:
                print(f"  ⚠️ {ts_code} {name}: 无数据")
                
        except Exception as e:
            print(f"  ❌ {ts_code} {name}: {e}")
    
    print(f"\n【指数数据补全完成】共补充 {total_inserted} 条")

def fill_money_flow():
    """补全资金流数据"""
    api = QuantDataAPI()
    
    print("\n【补全资金流数据】")
    
    # 获取当前数据范围
    r = api.query("SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM money_flow")
    current_min = r.iloc[0, 0]
    current_max = r.iloc[0, 1]
    
    print(f"当前数据范围: {current_min} ~ {current_max}")
    
    # 资金流数据Tushare通常从2015年开始
    target_start = pd.to_datetime('2015-01-01')
    current_min_date = pd.to_datetime(current_min)
    
    if current_min_date <= target_start:
        print("✅ 资金流数据已经完整或已达到Tushare限制")
        return
    
    print(f"需要补充: 2015-01-01 ~ {current_min_date.date()}")
    
    # 获取股票列表
    stocks = api.get_stock_list(status='L')
    print(f"待补全股票数量: {len(stocks)}")
    
    # 分批获取数据（资金流数据量大，需要谨慎）
    total_inserted = 0
    batch_size = 100
    
    for i in range(0, len(stocks), batch_size):
        batch = stocks.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            ts_code = row['ts_code']
            
            try:
                # 获取资金流数据
                df = pro.moneyflow(
                    ts_code=ts_code,
                    start_date=target_start.replace('-', ''),
                    end_date=target_end.replace('-', '')
                )
                
                if not df.empty:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    api._get_ch_client().insert_df(df, 'money_flow')
                    total_inserted += len(df)
                    
            except Exception as e:
                # 资金流数据接口限制较多，跳过错误
                pass
        
        print(f"  已处理 {min(i+batch_size, len(stocks))}/{len(stocks)} 只股票...")
    
    print(f"\n【资金流数据补全完成】共补充 {total_inserted} 条")

def main():
    print("=" * 80)
    print("【数据补全脚本】")
    print(f"执行时间: {datetime.now()}")
    print("=" * 80)
    
    # 补全指数数据
    fill_index_daily()
    
    # 补全资金流数据（注释掉，因为数据量大）
    # fill_money_flow()
    
    print("\n" + "=" * 80)
    print("【数据补全完成】")
    print("=" * 80)

if __name__ == "__main__":
    main()
