import sys
import os

# Ensure qlworks is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from qlworks.backtest import run_duckdb_backtrader
import backtrader as bt

class SimpleMovingAverageStrategy(bt.Strategy):
    """
    一个简单的双均线演示策略，专门用于测试 DuckDB 直连。
    当短期均线上穿长期均线时买入，下穿时卖出。
    """
    params = dict(
        short_period=5,
        long_period=20,
        log_enabled=True,
    )

    def __init__(self):
        self.all_orders = []
        
        # 为每只加载的股票初始化均线指标和交叉信号
        self.inds = {}
        for d in self.datas:
            if d._name == 'benchmark':
                continue
            
            sma_short = bt.indicators.SMA(d.close, period=self.p.short_period)
            sma_long = bt.indicators.SMA(d.close, period=self.p.long_period)
            crossover = bt.indicators.CrossOver(sma_short, sma_long)
            
            self.inds[d] = {
                'crossover': crossover
            }

    def notify_order(self, order):
        if order.status == order.Completed:
            d = order.data
            self.all_orders.append({
                'datetime': self.datetime.datetime().isoformat(),
                'type': 'buy' if order.isbuy() else 'sell',
                'price': order.executed.price,
                'size': order.executed.size,
                'comm': order.executed.comm,
                'value': order.executed.value,
                'pos_size': self.getposition(d).size,
                'cash': self.broker.getcash(),
                'total_value': self.broker.getvalue(),
                'stock_code': d._name,
                'stock_name': d._name
            })
            
            action = '买入' if order.isbuy() else '卖出'
            self.log(f"{action}成交 | {d._name} | 价格: {order.executed.price:.4f} | 数量: {order.executed.size:.2f}")

    def next(self):
        for d in self.datas:
            if d._name == 'benchmark':
                continue
                
            crossover = self.inds[d]['crossover'][0]
            pos = self.getposition(d)
            
            # 金叉且无仓位 -> 买入
            if crossover > 0 and not pos:
                # 用 20% 的资金买入该股票
                size = (self.broker.getcash() * 0.2) / d.close[0]
                if size > 100:  # 至少买1手
                    self.buy(data=d, size=int(size/100)*100)
                    
            # 死叉且有仓位 -> 卖出平仓
            elif crossover < 0 and pos:
                self.close(data=d)

    def log(self, txt, dt=None):
        if self.p.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

if __name__ == "__main__":
    print("=== 测试 DuckDB 直连回测引擎 ===")
    
    # 填写你想测试的股票代码列表
    test_stocks = ['000001.SZ', '600000.SH', '000002.SZ']
    
    # 运行回测
    cerebro, results = run_duckdb_backtrader(
        ts_codes=test_stocks,
        duckdb_path=r'e:\Quant\Quant_Tushare\data\quant_data.duckdb',
        start_date='2020-01-01',
        end_date='2023-12-31',
        benchmark_code='000300.SH',
        strategy_class=SimpleMovingAverageStrategy,
        strategy_params={
            'short_period': 5,
            'long_period': 20
        },
        initial_cash=100000.0,
        output_dir='./bt_output'
    )
