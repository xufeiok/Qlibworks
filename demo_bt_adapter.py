import pandas as pd
import numpy as np
import sys
import os

# Ensure qlworks is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from qlworks.backtest import run_qlib_backtrader, EnhancedQlibStrategy

def create_dummy_data():
    """
    创建模拟的 Qlib 预测结果和价格数据，用于演示 Backtrader 适配器的使用。
    """
    # 模拟日期范围
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    instruments = ['000001.SZ', '600000.SH', '000002.SZ', '600001.SH', '000004.SZ', '600004.SH']
    
    # 1. 模拟价格数据
    price_df_dict = {}
    for inst in instruments:
        df = pd.DataFrame(index=dates)
        # 随机游走生成价格
        rets = np.random.normal(0.001, 0.02, len(dates))
        prices = 10 * np.exp(np.cumsum(rets))
        
        df['open'] = prices * (1 + np.random.normal(0, 0.005, len(dates)))
        df['high'] = df['open'] * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        df['low'] = df['open'] * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        df['close'] = prices
        df['volume'] = np.random.randint(1000, 100000, len(dates))
        
        # 修正高低价
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        price_df_dict[inst] = df

    # 2. 模拟 Qlib 预测分数
    # 创建 MultiIndex ['datetime', 'instrument']
    multi_index = pd.MultiIndex.from_product([dates, instruments], names=['datetime', 'instrument'])
    pred_df = pd.DataFrame(index=multi_index)
    
    # 为每天生成随机打分
    scores = np.random.normal(0, 1, len(pred_df))
    pred_df['score'] = scores

    # 创建模拟的沪深300基准数据 (就用第一个股票的数据模拟，稍微加点扰动)
    benchmark_df = price_df_dict[instruments[0]].copy()
    benchmark_df['close'] = benchmark_df['close'] * (1 + np.random.normal(0, 0.005, size=len(benchmark_df)))
    benchmark_df['open'] = benchmark_df['close'] * 0.99
    benchmark_df['high'] = benchmark_df['close'] * 1.02
    benchmark_df['low'] = benchmark_df['close'] * 0.98

    return pred_df, price_df_dict, benchmark_df

if __name__ == "__main__":
    print("正在生成模拟数据...")
    pred_df, price_df_dict, benchmark_df = create_dummy_data()
    
    print("模拟数据生成完毕，开始运行回测...")
    
    # 运行适配器，它将调用定制版 backtrader 和 SuperPlot
    output_dir = os.path.join(os.path.dirname(__file__), 'bt_output')
    
    run_qlib_backtrader(
        pred_df=pred_df,
        price_df_dict=price_df_dict,
        benchmark_df=benchmark_df,
        strategy_class=EnhancedQlibStrategy,
        strategy_params={
            'top_k': 3,
            'rebalance_days': 5,
            'buy_pct': 0.95,
            'use_risk_control': True,
            'stop_type': 'ATR',
            'atr_multiplier': 2.0,
            'trailing_stop': True,
            'trailing_start_pct': 0.05,  # 模拟数据波动小，降低门槛
            'trailing_callback_pct': 0.01,
            'take_profit_pct': 0.5       # 止盈一半
        },
        initial_cash=100000.0,
        output_dir=output_dir
    )
