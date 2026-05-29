"""数据完整性检查脚本"""
from qlworks.data.api import QuantDataAPI
import pandas as pd

def main():
    api = QuantDataAPI()
    
    # 获取股票列表
    stocks = api.get_stock_list()
    total_stocks = len(stocks)
    print(f"【股票总数】{total_stocks} 只")
    
    # 随机抽查10只股票
    print("\n【随机抽查10只股票的数据完整性】")
    sample_stocks = stocks.sample(min(10, total_stocks))['ts_code'].tolist()
    
    for ts_code in sample_stocks:
        print(f"\n=== {ts_code} ===")
        
        # 上市日期
        list_date = stocks[stocks['ts_code'] == ts_code]['list_date'].iloc[0]
        print(f"上市日期: {list_date}")
        
        # 行情数据
        r = api.query(f"SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM daily_prices WHERE ts_code = '{ts_code}'")
        print(f"行情: {r.iloc[0,0]} ~ {r.iloc[0,1]} ({r.iloc[0,2]}条)")
        
        # 指标数据
        r = api.query(f"SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM daily_indicators WHERE ts_code = '{ts_code}'")
        print(f"指标: {r.iloc[0,0]} ~ {r.iloc[0,1]} ({r.iloc[0,2]}条)")
        
        # 复权因子
        r = api.query(f"SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM daily_adj_factors WHERE ts_code = '{ts_code}'")
        print(f"复权: {r.iloc[0,0]} ~ {r.iloc[0,1]} ({r.iloc[0,2]}条)")
        
        # 财务数据
        r = api.query(f"SELECT MIN(ann_date), MAX(ann_date), COUNT(*) FROM financial_indicators WHERE ts_code = '{ts_code}'")
        print(f"财务: {r.iloc[0,0]} ~ {r.iloc[0,1]} ({r.iloc[0,2]}条)")
    
    # 覆盖率统计
    print("\n【数据覆盖率统计】")
    tables = ['daily_prices', 'daily_indicators', 'daily_adj_factors', 'financial_indicators']
    for table in tables:
        r = api.query(f"SELECT COUNT(DISTINCT ts_code) FROM {table}")
        count = r.iloc[0, 0]
        coverage = (count / total_stocks) * 100
        print(f"{table}: {count}/{total_stocks} 只股票 ({coverage:.1f}%)")

if __name__ == "__main__":
    main()
