import pandas as pd
import backtrader as bt
import numpy as np

class AShareCommission(bt.CommInfoBase):
    """
    【Virtu 交易执行引擎】A股真实手续费模型：
    - 印花税：仅卖出收取 (目前 A 股为 0.05% 或 0.1%，此处按保守的 0.1% 测算)
    - 券商佣金：双向收取 (通常为万分之三)
    """
    params = (
        ('stamp_duty', 0.001),     
        ('commission', 0.0003),    
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入只收佣金
            return size * price * self.p.commission
        elif size < 0:  # 卖出收佣金 + 印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        return 0

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
                # [Virtu 改进] 检查成交量容量
                # 预估目标持仓所需资金
                target_value = self.broker.getvalue() * target_weight
                current_value = self.getposition(d).size * d.close[0]
                delta_value = target_value - current_value
                
                # 如果是买入，检查是否超过成交量限制
                if delta_value > 0:
                    # 简化处理：假设今日成交量作为可用流动性参考 (实盘应取过去 N 日均量)
                    available_volume = d.volume[0] if not np.isnan(d.volume[0]) else 0
                    max_allowed_shares = available_volume * self.p.volume_limit_pct
                    max_allowed_value = max_allowed_shares * d.close[0]
                    
                    if delta_value > max_allowed_value:
                        self.log(f"  [容量受限] {d._name} 目标买入金额 {delta_value:.2f} 超过容量上限 {max_allowed_value:.2f}")
                        # 降级：只买入最大允许量
                        target_weight = (current_value + max_allowed_value) / self.broker.getvalue()
                
                self.order_target_percent(d, target=target_weight)
                
        self.log(f"调仓完成. 当前 Top {self.p.top_k}: {top_k_names}")

    def log(self, txt, dt=None):
        if self.p.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

class AShareStrategy(bt.Strategy):
    """
    【Man Group & Virtu 联合定制】A股专属截面策略：
    - 每日买入得分 > score_threshold 的前 top_k 只股票。
    - 如果没有 10 只满足条件，有几只买几只；0 只则空仓。
    - 次日分数 <= threshold 或跌出前 top_k，则卖出。
    - 严格 T+1 限制：当日买入的股票当日不可卖出。
    """
    params = dict(
        top_k=10,               # 最多持仓数量
        score_threshold=0.9,    # 选股得分阈值
        buy_pct=0.95,           # 最大资金使用率
        log_enabled=True,
    )

    def __init__(self):
        self.instruments = [d for d in self.datas if getattr(d, '_name', '') != 'benchmark']
        self.all_orders = []
        self.buy_dates = {}  # 记录买入日期以控制 T+1

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            d = order.data
            price = order.executed.price
            size = order.executed.size
            
            if order.isbuy():
                self.log(f"[Virtu 撮合] 买入成交 | {d._name} | 价格: {price:.4f} | 数量: {size:.0f}")
                self.buy_dates[d._name] = self.datetime.date(0)
            elif order.issell():
                self.log(f"[Virtu 撮合] 卖出成交 | {d._name} | 价格: {price:.4f} | 数量: {size:.0f}")
                if self.getposition(d).size == 0:
                    self.buy_dates.pop(d._name, None)
                    
            self.all_orders.append({
                'datetime': self.datetime.datetime().isoformat(),
                'type': 'buy' if order.isbuy() else 'sell',
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
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            status_name = "Unknown"
            if order.status == order.Margin:
                status_name = "Margin (资金不足)"
            elif order.status == order.Rejected:
                status_name = "Rejected (拒单)"
            elif order.status == order.Canceled:
                status_name = "Canceled (取消)"
            self.log(f"订单异常 | {order.data._name} | 状态: {status_name}")

    def next(self):
        current_date = self.datetime.date(0)
        
        # 1. 筛选得分 > threshold 的股票
        valid_scores = []
        for d in self.instruments:
            if len(d) > 0 and not np.isnan(d.score[0]) and d.score[0] > self.p.score_threshold:
                valid_scores.append((d._name, d, d.score[0]))
                
        # 2. 按得分降序排序并选取前 top_k
        valid_scores.sort(key=lambda x: x[2], reverse=True)
        top_k_feeds = [x[1] for x in valid_scores[:self.p.top_k]]
        top_k_names = [x[0] for x in valid_scores[:self.p.top_k]]
        
        # 3. 平仓逻辑 (结合 T+1 限制)
        for d in self.instruments:
            pos = self.getposition(d)
            if pos.size > 0:
                # 检查是否满足 T+1 (当日买入的不可当日生成卖单)
                # 由于 Backtrader 是在次日开盘撮合，所以在这里(收盘时)生成的卖单会在次日开盘执行，
                # 只要 buy_date < current_date 即可保证持仓超过1天。
                buy_date = self.buy_dates.get(d._name)
                is_t_plus_1 = (buy_date is None or buy_date < current_date)
                
                # 如果不在新的目标列表里，且满足 T+1，则全部卖出
                if d not in top_k_feeds and is_t_plus_1:
                    self.close(d)
                    self.log(f"[Man Group 风控] 调仓平仓单下达: {d._name} (跌出前 {self.p.top_k} 或得分 <= {self.p.score_threshold})")

        # 4. 开仓逻辑
        if top_k_feeds:
            # 严格控制每只股票的权重为 1/top_k，避免在标的不足时单只股票满仓导致风险失控
            # 注意：Backtrader 的 order_target_percent 是相对于总资产(Value)而言的
            # 为防止可用现金不足导致 Margin 被拒，计算目标权重时稍微保守一点
            target_weight = (self.p.buy_pct / self.p.top_k) * 0.98
            
            for d in top_k_feeds:
                pos = self.getposition(d)
                # 仅在无持仓时买入；对于已持仓的股票，避免微调引发的零碎交易手续费
                if pos.size == 0:
                    self.order_target_percent(d, target=target_weight)
                    
        self.log(f"今日选股分析完成. 明日目标持仓: {top_k_names}")

class AShareCommission(bt.CommInfoBase):
    """
    【Virtu 交易执行引擎】A股真实手续费模型：
    - 印花税：仅卖出收取 (目前 A 股为 0.05% 或 0.1%，此处按保守的 0.1% 测算)
    - 券商佣金：双向收取 (通常为万分之三)
    """
    params = (
        ('stamp_duty', 0.001),     
        ('commission', 0.0003),    
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入只收佣金
            return size * price * self.p.commission
        elif size < 0:  # 卖出收佣金 + 印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        return 0

class AShareStrategy(bt.Strategy):
    """
    【Man Group & Virtu 联合定制】A股专属截面策略：
    - 每日买入得分 > score_threshold 的前 top_k 只股票。
    - 如果没有 10 只满足条件，有几只买几只；0 只则空仓。
    - 次日分数 <= threshold 或跌出前 top_k，则卖出。
    - 严格 T+1 限制：当日买入的股票当日不可卖出。
    """
    params = dict(
        top_k=10,               # 最多持仓数量
        score_threshold=0.9,    # 选股得分阈值
        buy_pct=0.95,           # 最大资金使用率
        log_enabled=True,
    )

    def __init__(self):
        self.instruments = [d for d in self.datas if getattr(d, '_name', '') != 'benchmark']
        self.all_orders = []
        self.buy_dates = {}  # 记录买入日期以控制 T+1

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            d = order.data
            price = order.executed.price
            size = order.executed.size
            
            if order.isbuy():
                self.log(f"[Virtu 撮合] 买入成交 | {d._name} | 价格: {price:.4f} | 数量: {size:.0f}")
                self.buy_dates[d._name] = self.datetime.date(0)
            elif order.issell():
                self.log(f"[Virtu 撮合] 卖出成交 | {d._name} | 价格: {price:.4f} | 数量: {size:.0f}")
                if self.getposition(d).size == 0:
                    self.buy_dates.pop(d._name, None)
                    
            self.all_orders.append({
                'datetime': self.datetime.datetime().isoformat(),
                'type': 'buy' if order.isbuy() else 'sell',
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
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单异常 | {order.data._name} | 状态: {order.status}")

    def next(self):
        current_date = self.datetime.date(0)
        
        # 1. 筛选得分 > threshold 的股票
        valid_scores = []
        for d in self.instruments:
            if len(d) > 0 and not np.isnan(d.score[0]) and d.score[0] > self.p.score_threshold:
                valid_scores.append((d._name, d, d.score[0]))
                
        # 2. 按得分降序排序并选取前 top_k
        valid_scores.sort(key=lambda x: x[2], reverse=True)
        top_k_feeds = [x[1] for x in valid_scores[:self.p.top_k]]
        top_k_names = [x[0] for x in valid_scores[:self.p.top_k]]
        
        # 3. 平仓逻辑 (结合 T+1 限制)
        for d in self.instruments:
            pos = self.getposition(d)
            if pos.size > 0:
                # 检查是否满足 T+1 (当日买入的不可当日生成卖单)
                # 由于 Backtrader 是在次日开盘撮合，所以在这里(收盘时)生成的卖单会在次日开盘执行，
                # 只要 buy_date < current_date 即可保证持仓超过1天。
                buy_date = self.buy_dates.get(d._name)
                is_t_plus_1 = (buy_date is None or buy_date < current_date)
                
                # 如果不在新的目标列表里，且满足 T+1，则全部卖出
                if d not in top_k_feeds and is_t_plus_1:
                    self.close(d)
                    self.log(f"[Man Group 风控] 调仓平仓单下达: {d._name} (跌出前 {self.p.top_k} 或得分 <= {self.p.score_threshold})")

        # 4. 开仓逻辑
        if top_k_feeds:
            # 严格控制每只股票的权重为 1/top_k，避免在标的不足时单只股票满仓导致风险失控
            target_weight = self.p.buy_pct / self.p.top_k
            
            for d in top_k_feeds:
                pos = self.getposition(d)
                # 仅在无持仓时买入；对于已持仓的股票，避免微调引发的零碎交易手续费
                if pos.size == 0:
                    self.order_target_percent(d, target=target_weight)
                    
        self.log(f"今日选股分析完成. 明日目标持仓: {top_k_names}")

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
        take_profit_pct=1.0,        # 触发止盈时的平仓比例 (1.0为全平，0.5为平一半)
        
        # --- 市场冲击与容量限制 ---
        volume_limit_pct=0.10       # [Virtu 改进] 单笔订单不能超过当日成交量的10%，防止吃不掉流动性
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
                            
                        # [Two Sigma & Virtu 改进] 发送真实的止损委托单到交易所（撮合引擎）
                        # 确保日内极端行情下能被 Low 价格刺穿并及时成交，而不是等收盘
                        stop_order = self.sell(data=d, size=size, exectype=bt.Order.Stop, price=stop_price)
                        
                        self.trade_states[d] = {
                            'entry_price': price,
                            'max_high': price,
                            'stop_loss': stop_price,
                            'stop_order': stop_order,
                            'tp_triggered': False,
                        }
                        self.log(f"  [{d._name}] 建立风控追踪 | 真实止损单挂单价: {stop_price:.4f}")
                        
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
                    
                # 真实的止损现在由 Broker 底层撮合引擎（盘中 Low 刺穿 Stop 价格）自动执行。
                # 所以我们不再需要在 next() 的收盘后手动触发止损。
                # 但我们需要更新移动止盈/追踪止损的订单。
                    
                # 检查移动止盈
                if self.p.trailing_stop:
                    current_profit_pct = (current_price - state['entry_price']) / state['entry_price']
                    current_pullback = (state['max_high'] - current_price) / state['max_high']
                    
                    if current_profit_pct > self.p.trailing_start_pct and current_pullback > self.p.trailing_callback_pct:
                        if not state['tp_triggered']:
                            sell_size = pos.size * self.p.take_profit_pct
                            if self.p.take_profit_pct >= 1.0:
                                # 取消原来的止损单
                                if state.get('stop_order'):
                                    self.cancel(state['stop_order'])
                                self.close(d)
                                self.log(f"!!! [{d._name}] 触发100%止盈 !!! 当前价 {current_price:.4f} (回撤 {current_pullback*100:.1f}%)")
                                del self.trade_states[d]
                                continue
                            else:
                                if state.get('stop_order'):
                                    self.cancel(state['stop_order'])
                                self.sell(data=d, size=sell_size)
                                state['tp_triggered'] = True
                                # 重置止损为保本或更紧的ATR垫
                                if self.p.stop_type == 'ATR':
                                    atr_val = self.atrs[d][0] if not np.isnan(self.atrs[d][0]) else (current_price * 0.05)
                                    new_stop = current_price - atr_val * self.p.atr_multiplier
                                else:
                                    new_stop = state['entry_price'] # 至少保本
                                state['stop_loss'] = max(state['stop_loss'], new_stop)
                                
                                # 挂出新的止损单
                                new_stop_order = self.sell(data=d, size=pos.size - sell_size, exectype=bt.Order.Stop, price=state['stop_loss'])
                                state['stop_order'] = new_stop_order
                                
                                self.log(f"!!! [{d._name}] 触发部分止盈 !!! 卖出比例 {self.p.take_profit_pct} | 新止损单挂单价: {state['stop_loss']:.4f}")

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
                if d in self.trade_states and self.trade_states[d].get('stop_order'):
                    self.cancel(self.trade_states[d]['stop_order'])
                self.close(d)
                self.log(f"调仓平仓: {d._name}")
        
        # 对 Top K 分配权重
        if top_k_feeds:
            target_weight = self.p.buy_pct / len(top_k_feeds)
            for d in top_k_feeds:
                # [Virtu 改进] 检查成交量容量
                # 预估目标持仓所需资金
                target_value = self.broker.getvalue() * target_weight
                current_value = self.getposition(d).size * d.close[0]
                delta_value = target_value - current_value
                
                # 如果是买入，检查是否超过成交量限制
                if delta_value > 0:
                    # 简化处理：假设今日成交量作为可用流动性参考 (实盘应取过去 N 日均量)
                    available_volume = d.volume[0] if not np.isnan(d.volume[0]) else 0
                    max_allowed_shares = available_volume * self.p.volume_limit_pct
                    max_allowed_value = max_allowed_shares * d.close[0]
                    
                    if delta_value > max_allowed_value:
                        self.log(f"  [容量受限] {d._name} 目标买入金额 {delta_value:.2f} 超过容量上限 {max_allowed_value:.2f}")
                        # 降级：只买入最大允许量
                        adjusted_target_weight = (current_value + max_allowed_value) / self.broker.getvalue()
                        self.order_target_percent(d, target=adjusted_target_weight)
                        continue
                        
                self.order_target_percent(d, target=target_weight)
                
        self.log(f"调仓完成. 当前 Top {self.p.top_k}: {top_k_names}")

    def log(self, txt, dt=None):
        if self.p.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')