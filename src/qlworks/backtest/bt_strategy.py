import pandas as pd
import backtrader as bt
import numpy as np

class QlibPandasData(bt.feeds.PandasData):
    """
    自定义的 PandasData 数据源，包含 Qlib 的预测分数 'score'。
    """
    lines = ('score',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('score', 'score'),  # 映射 DataFrame 中的 'score' 列到 line
    )

class BaseQlibStrategy(bt.Strategy):
    """
    基础的 Qlib 截面回测策略。
    定期（如每周）根据 score 进行排序，做多 Top K 的股票。
    """
    params = dict(
        top_k=5,             # 持仓数量
        rebalance_days=5,    # 调仓周期（天）
        buy_pct=0.95,        # 最大资金使用率
        log_enabled=True,
    )

    def __init__(self):
        self.days_since_rebalance = 0
        self.instruments = [d for d in self.datas if getattr(d, '_name', '') != 'benchmark']
        self.all_orders = []

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

    def next(self):
        self.days_since_rebalance += 1
        # 检查是否到达调仓日
        if self.days_since_rebalance < self.p.rebalance_days:
            return
            
        self.days_since_rebalance = 0

        # 获取当前所有有效股票的 score
        scores = []
        for d in self.instruments:
            if len(d) > 0 and not np.isnan(d.score[0]):
                scores.append((d._name, d, d.score[0]))
        
        # 按 score 降序排列
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # 选出 Top K
        top_k_feeds = [x[1] for x in scores[:self.p.top_k]]
        top_k_names = [x[0] for x in scores[:self.p.top_k]]
        
        # 平掉不在 Top K 中的仓位
        for d in self.instruments:
            if self.getposition(d).size != 0 and d not in top_k_feeds:
                self.close(d)
                self.log(f"调仓平仓: {d._name}")
        
        # 对 Top K 分配权重
        if top_k_feeds:
            target_weight = self.p.buy_pct / len(top_k_feeds)
            for d in top_k_feeds:
                self.order_target_percent(d, target=target_weight)
                
        self.log(f"调仓完成. 当前 Top {self.p.top_k}: {top_k_names}")

    def log(self, txt, dt=None):
        if self.p.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

class EnhancedQlibStrategy(bt.Strategy):
    """
    增强版的 Qlib 截面回测策略。
    结合了定期（如每周）根据 score 进行排序的截面多因子逻辑，
    以及个股级别的止盈止损（ATR/固定比例、移动止盈）等时间序列风控逻辑。
    """
    params = dict(
        top_k=5,             # 持仓数量
        rebalance_days=5,    # 调仓周期（天）
        buy_pct=0.95,        # 最大资金使用率
        log_enabled=True,
        
        # --- 止盈止损风控参数 ---
        use_risk_control=True,      # 是否启用风控增强模式
        stop_type='ATR',            # 止损类型: 'ATR' 或 'FIXED'
        stop_loss_pct=0.05,         # 固定止损比例 (5%)，仅在 stop_type='FIXED' 时有效
        atr_period=14,              # ATR 周期
        atr_multiplier=2.0,         # ATR 止损倍数
        
        trailing_stop=True,         # 是否启用移动止盈
        trailing_start_pct=0.10,    # 盈利超过 10% 启动移动止盈
        trailing_callback_pct=0.02, # 回撤 2% 止盈
        take_profit_pct=1.0         # 触发止盈时的平仓比例 (1.0为全平，0.5为平一半)
    )

    def __init__(self):
        self.days_since_rebalance = 0
        self.instruments = [d for d in self.datas if getattr(d, '_name', '') != 'benchmark']
        self.all_orders = []
        
        # 记录每只股票的交易状态
        self.trade_states = {}
        
        # 为每个数据源计算 ATR
        self.atrs = {}
        if self.p.use_risk_control and self.p.stop_type == 'ATR':
            for d in self.instruments:
                self.atrs[d] = bt.indicators.ATR(d, period=self.p.atr_period)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            d = order.data
            price = order.executed.price
            size = order.executed.size
            
            if order.isbuy():
                self.log(f"买入成交 | {d._name} | 价格: {price:.4f} | 数量: {size:.2f}")
                
                self.all_orders.append({
                    'datetime': self.datetime.datetime().isoformat(),
                    'type': 'buy',
                    'price': price,
                    'size': size,
                    'comm': order.executed.comm,
                    'value': order.executed.value,
                    'pos_size': self.getposition(d).size,
                    'cash': self.broker.getcash(),
                    'total_value': self.broker.getvalue(),
                    'stock_code': d._name,
                    'stock_name': d._name
                })
                
                # 初始化或更新持仓状态 (忽略加仓造成的复杂成本计算，以最新买入价为准)
                if d not in self.trade_states or self.getposition(d).size == size:
                    stop_price = 0.0
                    if self.p.use_risk_control:
                        if self.p.stop_type == 'ATR':
                            # 注意：如果历史数据不足，ATR可能为NaN
                            atr_val = self.atrs[d][0] if not np.isnan(self.atrs[d][0]) else (price * 0.05)
                            stop_dist = atr_val * self.p.atr_multiplier
                            stop_price = price - stop_dist
                        else:
                            stop_price = price * (1.0 - self.p.stop_loss_pct)
                            
                        self.trade_states[d] = {
                            'entry_price': price,
                            'max_high': price,
                            'stop_loss': stop_price,
                            'tp_triggered': False,
                        }
                        self.log(f"  [{d._name}] 建立风控追踪 | 初始止损价: {stop_price:.4f}")
                        
            elif order.issell():
                self.log(f"卖出成交 | {d._name} | 价格: {price:.4f} | 数量: {size:.2f}")
                
                self.all_orders.append({
                    'datetime': self.datetime.datetime().isoformat(),
                    'type': 'sell',
                    'price': price,
                    'size': size,
                    'comm': order.executed.comm,
                    'value': order.executed.value,
                    'pos_size': self.getposition(d).size,
                    'cash': self.broker.getcash(),
                    'total_value': self.broker.getvalue(),
                    'stock_code': d._name,
                    'stock_name': d._name
                })
                
                # 如果平仓完了，移除追踪
                if self.getposition(d).size == 0:
                    if d in self.trade_states:
                        del self.trade_states[d]
                        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单异常 | {order.data._name} | Status: {order.status}")

    def next(self):
        # 1. 每日风控检查 (止盈止损)
        if self.p.use_risk_control:
            self.check_risk_control()
            
        self.days_since_rebalance += 1
        # 2. 检查是否到达调仓日
        if self.days_since_rebalance >= self.p.rebalance_days:
            self.days_since_rebalance = 0
            self.rebalance()

    def check_risk_control(self):
        for d in self.instruments:
            pos = self.getposition(d)
            if pos.size > 0 and d in self.trade_states:
                state = self.trade_states[d]
                current_price = d.close[0]
                high_price = d.high[0]
                
                # 更新最高价
                if high_price > state['max_high']:
                    state['max_high'] = high_price
                    
                # 检查止损
                if current_price < state['stop_loss']:
                    self.log(f"!!! [{d._name}] 触发止损 !!! 当前价 {current_price:.4f} < 止损价 {state['stop_loss']:.4f}")
                    self.close(d)
                    # 标记为已处理，防止同日多次触发
                    del self.trade_states[d]
                    continue
                    
                # 检查移动止盈
                if self.p.trailing_stop:
                    current_profit_pct = (current_price - state['entry_price']) / state['entry_price']
                    current_pullback = (state['max_high'] - current_price) / state['max_high']
                    
                    if current_profit_pct > self.p.trailing_start_pct and current_pullback > self.p.trailing_callback_pct:
                        if not state['tp_triggered']:
                            sell_size = pos.size * self.p.take_profit_pct
                            if self.p.take_profit_pct >= 1.0:
                                self.close(d)
                                self.log(f"!!! [{d._name}] 触发100%止盈 !!! 当前价 {current_price:.4f} (回撤 {current_pullback*100:.1f}%)")
                                del self.trade_states[d]
                                continue
                            else:
                                self.sell(data=d, size=sell_size)
                                state['tp_triggered'] = True
                                # 重置止损为保本或更紧的ATR垫
                                if self.p.stop_type == 'ATR':
                                    atr_val = self.atrs[d][0] if not np.isnan(self.atrs[d][0]) else (current_price * 0.05)
                                    new_stop = current_price - atr_val * self.p.atr_multiplier
                                else:
                                    new_stop = state['entry_price'] # 至少保本
                                state['stop_loss'] = max(state['stop_loss'], new_stop)
                                self.log(f"!!! [{d._name}] 触发部分止盈 !!! 卖出比例 {self.p.take_profit_pct} | 新止损价重置为: {state['stop_loss']:.4f}")

    def rebalance(self):
        # 获取当前所有有效股票的 score
        scores = []
        for d in self.instruments:
            if len(d) > 0 and not np.isnan(d.score[0]):
                scores.append((d._name, d, d.score[0]))
        
        # 按 score 降序排列
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # 选出 Top K
        top_k_feeds = [x[1] for x in scores[:self.p.top_k]]
        top_k_names = [x[0] for x in scores[:self.p.top_k]]
        
        # 平掉不在 Top K 中的仓位
        for d in self.instruments:
            if self.getposition(d).size != 0 and d not in top_k_feeds:
                self.close(d)
                self.log(f"调仓平仓: {d._name}")
        
        # 对 Top K 分配权重
        if top_k_feeds:
            target_weight = self.p.buy_pct / len(top_k_feeds)
            for d in top_k_feeds:
                self.order_target_percent(d, target=target_weight)
                
        self.log(f"调仓完成. 当前 Top {self.p.top_k}: {top_k_names}")

    def log(self, txt, dt=None):
        if self.p.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')