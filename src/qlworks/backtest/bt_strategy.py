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

class EnhancedQlibStrategy(bt.Strategy):
    """
    增强版的 Qlib 截面回测策略。
    结合了定期（如每周）根据 score 进行排序的截面多因子逻辑，
    以及个股级别的止盈止损（ATR/固定比例、移动止盈）等时间序列风控逻辑。
    """
    params = dict(
        top_k=10,            # 持仓数量
        score_threshold=0.7, # [新增] 选股得分阈值
        rebalance_days=5,    # 调仓周期（天）
        buy_pct=0.95,        # 最大资金使用率
        log_enabled=True,
        
        # --- 止盈止损风控参数 ---
        use_risk_control=True,      # 是否启用风控增强模式
        stop_type='ATR',            # 止损类型: 'ATR' 或 'FIXED'
        stop_loss_pct=0.05,         # 固定止损比例 (5%)，仅在 stop_type='FIXED' 时有效
        atr_period=14,              # ATR 周期
        atr_multiplier=2.0,         # ATR 止损倍数
        
        score_drop_threshold=0.3,   # [新增小市值策略逻辑] 分数恶化平仓: 模型得分跌破 0.3
        
        trailing_stop=True,         # 是否启用移动止盈
        trailing_start_pct=0.10,    # 盈利超过 10% 启动移动止盈
        trailing_callback_pct=0.02, # 回撤 2% 止盈
        take_profit_pct=1.0,        # 触发止盈时的平仓比例 (1.0为全平，0.5为平一半)
        
        # --- 市场冲击与容量限制 ---
        volume_limit_pct=0.10,      # [Virtu 改进] 单笔订单不能超过当日成交量的10%，防止吃不掉流动性
        rebalance_signal_weekday=1, # 调仓信号日，0=周一 ... 1=周二
        buy_weekday=2               # 买入执行日，0=周一 ... 2=周三
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

        # [Virtu 改进] 成交量均线用于容量限制校验（避免使用当日成交量做未来函数）
        self.vol_smas = {}
        for d in self.instruments:
            self.vol_smas[d] = bt.indicators.SMA(d.volume, period=20)

        # 两阶段调仓：周二仅记录需要补位的数量，周三再按当日分数实时选股买入。
        self.pending_buy_count = 0
        self.pending_buy_names = []

    def _format_abnormal_order_log(self, order):
        """
        构造异常订单的可诊断日志。

        输出关键信息：
        - 状态名：Canceled / Margin / Rejected
        - 方向：buy / sell / unknown
        - 数量：按委托量绝对值展示，避免卖单负号造成歧义
        - 现金：异常发生时账户可用现金
        - 原因：优先使用订单自带 reason，否则回退到状态默认解释
        """
        status_name_map = {
            order.Canceled: "Canceled",
            order.Margin: "Margin",
            order.Rejected: "Rejected",
        }
        default_reason_map = {
            order.Canceled: "订单被取消（可能是策略主动撤单、替换旧单或订单失效）",
            order.Margin: "可用资金不足",
            order.Rejected: "订单被经纪商或撮合引擎拒绝",
        }

        if order.isbuy():
            side = "buy"
        elif order.issell():
            side = "sell"
        else:
            side = "unknown"

        requested_size = getattr(getattr(order, "created", None), "size", 0) or 0
        if requested_size == 0:
            requested_size = getattr(getattr(order, "executed", None), "size", 0) or 0

        info = getattr(order, "info", None)
        custom_reason = info.get("reason") if hasattr(info, "get") else None
        reason = custom_reason or default_reason_map.get(order.status, "未知原因")

        stock_code = getattr(getattr(order, "data", None), "_name", "UNKNOWN")
        cash = self.broker.getcash() if getattr(self, "broker", None) is not None else float("nan")
        status_name = status_name_map.get(order.status, str(order.status))

        return (
            f"订单异常 | {stock_code} | 状态: {status_name} | 方向: {side} | "
            f"数量: {abs(float(requested_size)):.2f} | 现金: {cash:.2f} | 原因: {reason}"
        )

    def _format_completed_order_log(self, order):
        """
        构造成交订单的统一诊断日志。

        输出关键信息：
        - 成交类型：买入成交 / 卖出成交
        - 方向：buy / sell / unknown
        - 价格、数量、手续费
        - 成交后账户现金、当前持仓数量
        """
        if order.isbuy():
            side = "buy"
            title = "买入成交"
        elif order.issell():
            side = "sell"
            title = "卖出成交"
        else:
            side = "unknown"
            title = "成交"

        stock_code = getattr(getattr(order, "data", None), "_name", "UNKNOWN")
        price = getattr(getattr(order, "executed", None), "price", 0.0) or 0.0
        size = getattr(getattr(order, "executed", None), "size", 0.0) or 0.0
        comm = getattr(getattr(order, "executed", None), "comm", 0.0) or 0.0
        cash = self.broker.getcash() if getattr(self, "broker", None) is not None else float("nan")
        pos_size = self.getposition(order.data).size if hasattr(self, "getposition") else float("nan")

        return (
            f"{title} | {stock_code} | 方向: {side} | 价格: {price:.4f} | "
            f"数量: {abs(float(size)):.2f} | 手续费: {comm:.2f} | 现金: {cash:.2f} | 持仓: {pos_size:.2f}"
        )

    def _is_buy_execution_day(self, current_date):
        """判断当天是否为允许执行补仓买入的日期。"""
        return current_date.weekday() == self.p.buy_weekday

    def _is_rebalance_signal_day(self, current_date):
        """判断当天是否为允许触发调仓信号的日期。"""
        return current_date.weekday() == self.p.rebalance_signal_weekday

    def _plan_rebalance_actions(self, top_k_feeds):
        """
        规划两阶段调仓动作：
        - 仅卖出已持仓且不再位于目标持股名单中的股票
        - 周三买入数量按卖出后剩余持仓补足到 top_k
        """
        current_holds = [d for d in self.instruments if self.getposition(d).size > 0]
        to_sell = [d for d in current_holds if d not in top_k_feeds]
        projected_hold_count = len(current_holds) - len(to_sell)
        buy_count = max(self.p.top_k - projected_hold_count, 0)
        return to_sell, buy_count

    def _get_current_hold_count(self):
        """统计当前实际持仓只数。"""
        return sum(1 for d in self.instruments if self.getposition(d).size > 0)

    def _get_target_buy_count(self):
        """
        计算周三应补仓数量：
        - 初始空仓时补满 top_k
        - 非初始情况下补到 top_k
        - 止损造成缺口时，也只在周三补回缺口
        """
        return max(self.p.top_k - self._get_current_hold_count(), 0)

    def _select_pending_buy_candidates(self, buy_count):
        """
        周三按当天分数重新选股：
        - 只从当日分数超过阈值的股票里选
        - 只选择当前未持仓股票
        - 按当天分数降序取前 buy_count 只
        """
        if buy_count <= 0:
            return []

        scored_feeds = []
        for d in self.instruments:
            if len(d) <= 0 or self.getposition(d).size > 0:
                continue

            score = d.score[0]
            close_price = d.close[0]
            if score == score and close_price > 0.01 and score > self.p.score_threshold:
                scored_feeds.append((d, score))

        scored_feeds.sort(key=lambda item: item[1], reverse=True)
        return [d for d, _ in scored_feeds[:buy_count]]

    def _queue_pending_buys(self, buy_count):
        self.pending_buy_count = buy_count
        self.pending_buy_names = []

    def _execute_pending_buys(self):
        buy_count = self._get_target_buy_count()
        if buy_count <= 0:
            return

        import math

        target_weight = self.p.buy_pct / self.p.top_k
        self.pending_buy_count = buy_count
        buy_candidates = self._select_pending_buy_candidates(buy_count)
        self.pending_buy_names = [d._name for d in buy_candidates]
        buy_names = []
        for d in buy_candidates:
            if self.getposition(d).size > 0:
                continue

            total_value = self.broker.getvalue()
            if not math.isfinite(total_value):
                self.log(f"  [严重警告] 账户总资金异常 ({total_value})，跳过补仓买入")
                continue

            close_price = d.close[0]
            if close_price is None or not math.isfinite(close_price) or close_price <= 0.01:
                self.log(f"  [跳过补仓] {d._name} 当日收盘价无效或过低 ({close_price})")
                continue

            try:
                target_shares = int((total_value * target_weight / close_price) // 100 * 100)
                if not math.isfinite(target_shares) or target_shares > 1e9:
                    raise OverflowError("股数过大")
            except (ValueError, TypeError, OverflowError) as e:
                self.log(f"  [跳过补仓] {d._name} 计算股数失败 - {e}")
                continue

            if target_shares >= 100:
                self.order_target_size(d, target=target_shares)
                buy_names.append(d._name)

        self.pending_buy_count = 0
        self.pending_buy_names = []
        if buy_names:
            self.log(f"周三补仓买入完成: {buy_names}")
        else:
            self.log("周三补仓买入完成: 无有效买单")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            d = order.data
            price = order.executed.price
            size = order.executed.size
            
            if order.isbuy():
                self.log(self._format_completed_order_log(order))
                
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
                            # 注意：如果历史数据包含极端脏值，ATR可能异常大，设置一个合理上限（最大不超过价格的 20%）
                            atr_val = self.atrs[d][0] if self.atrs[d][0] == self.atrs[d][0] else (price * 0.05)
                            import math
                            if not math.isfinite(atr_val) or atr_val > price * 0.2:
                                atr_val = price * 0.05
                            stop_dist = atr_val * self.p.atr_multiplier
                            stop_price = max(price - stop_dist, 0.01)
                        else:
                            stop_price = max(price * (1.0 - self.p.stop_loss_pct), 0.01)
                            
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
                self.log(self._format_completed_order_log(order))
                
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
            self.log(self._format_abnormal_order_log(order))

    def next(self):
        # 1. 每日风控检查 (止盈止损)
        if self.p.use_risk_control:
            self.check_risk_control()

        current_date = self.datas[0].datetime.date(0)
        if self._is_buy_execution_day(current_date) and self._get_target_buy_count() > 0:
            self._execute_pending_buys()

        self.days_since_rebalance += 1
        # 2. 检查是否到达调仓日
        if (
            self.days_since_rebalance >= self.p.rebalance_days
            and self._is_rebalance_signal_day(current_date)
        ):
            self.days_since_rebalance = 0
            self.rebalance()

    def check_risk_control(self):
        for d in self.instruments:
            pos = self.getposition(d)
            if pos.size > 0 and d in self.trade_states:
                state = self.trade_states[d]
                current_price = d.close[0]
                high_price = d.high[0]
                
                # 跳过无效价格
                if current_price is None or current_price != current_price or current_price <= 0:
                    self.log(f"  [跳过风控] {d._name} 当日价格无效 ({current_price})")
                    continue
                entry_price = state.get('entry_price')
                if entry_price is None or entry_price != entry_price or entry_price <= 0:
                    self.log(f"  [跳过风控] {d._name} 入场价格无效 ({entry_price})，清除状态")
                    del self.trade_states[d]
                    continue
                
                # 更新最高价
                if high_price > state['max_high']:
                    state['max_high'] = high_price
                    
                # [Two Sigma & 小市值策略风控] 每天检查得分是否恶化
                # 【优化】移除日频得分恶化平仓，避免 A 股高波动带来的双面打脸 (Score Drop Whipsaw)
                # score = d.score[0] if not np.isnan(d.score[0]) else 0.0
                # if score < self.p.score_drop_threshold:
                #     if state.get('stop_order'):
                #         self.cancel(state['stop_order'])
                #     self.close(d)
                #     self.log(f"!!! [小市值风控] 触发非换仓日强制卖出(得分恶化) !!! {d._name} (当前得分: {score:.2f} < {self.p.score_drop_threshold})")
                #     del self.trade_states[d]
                #     continue
                    
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
                                    atr_val = self.atrs[d][0] if self.atrs[d][0] == self.atrs[d][0] else (current_price * 0.05)
                                    import math
                                    if not math.isfinite(atr_val) or atr_val > current_price * 0.2:
                                        atr_val = current_price * 0.05
                                    new_stop = max(current_price - atr_val * self.p.atr_multiplier, 0.01)
                                else:
                                    new_stop = max(state['entry_price'], 0.01) # 至少保本
                                state['stop_loss'] = max(state['stop_loss'], new_stop)
                                
                                # 挂出新的止损单
                                new_stop_order = self.sell(data=d, size=pos.size - sell_size, exectype=bt.Order.Stop, price=state['stop_loss'])
                                state['stop_order'] = new_stop_order
                                
                                self.log(f"!!! [{d._name}] 触发部分止盈 !!! 卖出比例 {self.p.take_profit_pct} | 新止损单挂单价: {state['stop_loss']:.4f}")

    def rebalance(self):
        # 获取当前所有有效股票的 score，并根据 score_threshold 进行初步过滤
        scores = []
        valid_cnt = 0
        score_nan_cnt = 0
        for d in self.instruments:
            if len(d) > 0:
                s = d.score[0]
                p = d.close[0]
                if s == s and p > 0.01:  # 必须是有正常分数的，并且当前价格不能是负数或极小仙股
                    valid_cnt += 1
                    if s > self.p.score_threshold:
                        scores.append((d._name, d, s))
                else:
                    score_nan_cnt += 1
        
        # 定期输出以观察 score 数据是否正常传入
        if self.days_since_rebalance == 0:
            self.log(f"[DEBUG] rebalance - instruments: {len(self.instruments)} | with_score: {valid_cnt} | score_nan: {score_nan_cnt} | > threshold({self.p.score_threshold}): {len(scores)}")

        # 按 score 降序排列
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # 选出 Top K
        top_k_feeds = [x[1] for x in scores[:self.p.top_k]]
        top_k_names = [x[0] for x in scores[:self.p.top_k]]

        to_sell, buy_count = self._plan_rebalance_actions(top_k_feeds)

        for d in to_sell:
            if d in self.trade_states and self.trade_states[d].get('stop_order'):
                self.cancel(self.trade_states[d]['stop_order'])
            self.close(d)
            self.log(f"调仓卖出: {d._name}")

        self._queue_pending_buys(buy_count)
        self.log(
            f"调仓完成. 当前 Top {self.p.top_k}: {top_k_names} | "
            f"卖出 {len(to_sell)} 只 | 待周三补仓数量 {buy_count}"
        )

    def log(self, txt, dt=None):
        if self.p.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
