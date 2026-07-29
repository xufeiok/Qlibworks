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
        volume_limit_pct=0.10,      # [Virtu 改进] 单笔订单不能超过20日均量的10%，防止吃不掉流动性
        rebalance_signal_weekday=1, # 调仓信号日，0=周一 ... 1=周二
        buy_weekday=2               # 买入执行日，0=周一 ... 2=周三
    )

    def __init__(self):
        self.days_since_rebalance = 0
        self.instruments = [d for d in self.datas if getattr(d, '_name', '') != 'benchmark']
        self.all_orders = []
        
        # 记录每只股票的交易状态
        self.trade_states = {}
        
        # 显式标记"正在平仓中"的股票（check_risk_control 下单后标记，订单完成/取消/拒绝后清除）
        # 防止同一 bar 内 check_risk_control 和 rebalance 对同一只股重复下单
        self._closing_stocks = set()

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
        self._pending_orders = {}
        self._pending_stop_adjustments = {}  # 止损价差调整队列

    def _has_open_order(self, d):
        target_name = getattr(d, '_name', None)
        try:
            for o in self.getorders_open():
                odata = getattr(o, 'data', None)
                if odata is not None and getattr(odata, '_name', None) == target_name:
                    return True
        except Exception:
            return False
        return False

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

        self.pending_buy_count = buy_count
        buy_candidates = self._select_pending_buy_candidates(buy_count)
        if not buy_candidates:
            self.pending_buy_count = 0
            self.pending_buy_names = []
            self.log("周三补仓买入完成: 无候选股")
            return

        self.pending_buy_names = [d._name for d in buy_candidates]

        # [修复 Margin] 预算按现金递减追踪，避免固定 target_weight 导致后几笔超支
        total_cash = self.broker.getcash()
        total_budget = total_cash * self.p.buy_pct
        remaining_budget = total_budget
        remaining_count = len(buy_candidates)
        buy_names = []

        for d in buy_candidates:
            if self._has_open_order(d):
                remaining_count -= 1
                continue
            if self.getposition(d).size > 0:
                remaining_count -= 1
                continue

            close_price = d.close[0]
            if close_price is None or not math.isfinite(close_price) or close_price <= 0.01:
                self.log(f"  [跳过补仓] {d._name} 当日收盘价无效或过低 ({close_price})")
                remaining_count -= 1
                continue

            # [风控] 涨停板不买入
            _prev_close = d.close[-1]
            if _prev_close is not None and math.isfinite(_prev_close) and _prev_close > 0.01:
                _chg = close_price / _prev_close - 1
                if _chg >= 0.0955:
                    self.log(f"  [风控] {d._name} 涨停({_chg*100:.1f}%)，跳过补仓")
                    remaining_count -= 1
                    continue

            # 剩余预算均分给剩余待买股票
            budget_per_stock = remaining_budget / remaining_count if remaining_count > 0 else 0
            if budget_per_stock < 100 * close_price:
                self.log(f"  [跳过补仓] {d._name} 剩余预算不足 ({budget_per_stock:.2f} < 100股×{close_price:.2f})")
                remaining_count -= 1
                continue

            try:
                target_shares = int((budget_per_stock / close_price) // 100 * 100)
                if not math.isfinite(target_shares) or target_shares > 1e9:
                    raise OverflowError("股数过大")
            except (ValueError, TypeError, OverflowError) as e:
                self.log(f"  [跳过补仓] {d._name} 计算股数失败 - {e}")
                remaining_count -= 1
                continue

            # [Virtu 改进] 成交量限制：使用 SMA20 日均量替代当日量
            # 当日成交量在回测中已知，但在实盘开盘时未知；SMA20 是更保守的流动性估计
            if d in self.vol_smas:
                sma_val = self.vol_smas[d][0]
                daily_volume = sma_val if (sma_val is not None and math.isfinite(sma_val) and sma_val > 0) else d.volume[0]
            else:
                daily_volume = d.volume[0]
            if daily_volume is not None and math.isfinite(daily_volume) and daily_volume > 0:
                max_shares_by_volume = int(daily_volume * self.p.volume_limit_pct // 100 * 100)
                if max_shares_by_volume < 100:
                    self.log(f"  [跳过补仓] {d._name} 当日成交量过小 ({daily_volume})，无法满足 100 股最小单位")
                    remaining_count -= 1
                    continue
                if target_shares > max_shares_by_volume:
                    self.log(f"  [缩容买入] {d._name} 订单量 {target_shares} 超过成交量限制 {max_shares_by_volume}，缩容至 {max_shares_by_volume}")
                    target_shares = max_shares_by_volume

            if target_shares >= 100:
                o = self.order_target_size(d, target=target_shares)
                if o is not None:
                    self._pending_orders[d] = o
                order_cost = target_shares * close_price
                remaining_budget -= order_cost
                buy_names.append(d._name)
                self.log(f"  [预算追踪] {d._name} 预计花费 {order_cost:.0f}, 剩余预算 {remaining_budget:.0f}")

            remaining_count -= 1

        self.pending_buy_count = 0
        self.pending_buy_names = []
        if buy_names:
            self.log(f"周三补仓买入完成: {buy_names} (使用预算 {total_budget - remaining_budget:.0f}/{total_budget:.0f})")
        else:
            self.log("周三补仓买入完成: 无有效买单")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        d = getattr(order, "data", None)
        if d is not None:
            existing = self._pending_orders.get(d)
            if existing is order:
                del self._pending_orders[d]
            
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
                            
                        self.trade_states[d] = {
                            'entry_price': price,
                            'max_high': price,
                            'stop_loss': stop_price,
                            'stop_order': None,
                            'tp_triggered': False,
                        }
                        self.log(f"  [{d._name}] 建立风控追踪 | 止损价: {stop_price:.4f}")
                        
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
            # 止损价差调整：如果收盘价高于止损价，扣回超额部分
            stock_name = d._name
            if stock_name in self._pending_stop_adjustments:
                adj = self._pending_stop_adjustments.pop(stock_name)
                diff = (adj['close_price'] - adj['effective_price']) * adj['size']
                if diff > 0 and order.status == order.Completed:
                    self.broker.add_cash(-diff)
                    self.log('  [止损罚金] ' + d._name + ' 扣除盘中收回价差 ' + format(diff, '.2f'))
                if self.getposition(d).size == 0:
                    if d in self.trade_states:
                        del self.trade_states[d]
                    self._closing_stocks.discard(d._name)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(self._format_abnormal_order_log(order))
            if order.issell() and d is not None:
                self._closing_stocks.discard(d._name)

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
                    
                low_price = d.low[0]
                if low_price is not None and low_price == low_price and low_price <= state['stop_loss']:
                    if self._has_open_order(d):
                        continue
                    current_pos = self.getposition(d).size
                    if current_pos <= 0:
                        del self.trade_states[d]
                        continue
                    o = self.close(d)
                    if o is not None:
                        self._pending_orders[d] = o
                        self._closing_stocks.add(d._name)
                    self.log(f"!!! [{d._name}] 触发止损 !!! low {low_price:.4f} <= {state['stop_loss']:.4f}")
                    del self.trade_states[d]
                    continue
                    
                # 检查移动止盈
                if self.p.trailing_stop:
                    current_profit_pct = (current_price - state['entry_price']) / state['entry_price']
                    current_pullback = (state['max_high'] - current_price) / state['max_high']
                    
                    if current_profit_pct > self.p.trailing_start_pct and current_pullback > self.p.trailing_callback_pct:
                        if not state['tp_triggered']:
                            sell_size = pos.size * self.p.take_profit_pct
                            current_pos = self.getposition(d).size
                            if current_pos <= 0:
                                del self.trade_states[d]
                                continue
                            if self.p.take_profit_pct >= 1.0:
                                # 取消原来的止损单
                                if state.get('stop_order'):
                                    self.cancel(state['stop_order'])
                                o = self.close(d)
                                if o is not None:
                                    self._pending_orders[d] = o
                                    self._closing_stocks.add(d._name)
                                self.log(f"!!! [{d._name}] 触发100%止盈 !!! 当前价 {current_price:.4f} (回撤 {current_pullback*100:.1f}%)")
                                del self.trade_states[d]
                                continue
                            else:
                                if self._has_open_order(d):
                                    continue
                                current_pos = self.getposition(d).size
                                if current_pos <= 0:
                                    continue
                                if state.get('stop_order'):
                                    self.cancel(state['stop_order'])
                                actual_sell = min(sell_size, current_pos)
                                o = self.sell(data=d, size=actual_sell)
                                if o is not None:
                                    self._pending_orders[d] = o
                                    self._closing_stocks.add(d._name)
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
            if d._name in self._closing_stocks:
                self.log(f"  [跳过卖出] {d._name} 正在平仓中（风控已触发），跳过 rebalance 卖单")
                if d in self.trade_states:
                    del self.trade_states[d]
                continue
            if self._has_open_order(d):
                self.log(f"  [跳过卖出] {d._name} 存在未完成委托，跳过")
                continue
            if d in self.trade_states and self.trade_states[d].get('stop_order'):
                self.cancel(self.trade_states[d]['stop_order'])
            current_pos = self.getposition(d).size
            if current_pos <= 0:
                self.log(f"  [跳过卖出] {d._name} 当前无持仓 ({current_pos})")
                if d in self.trade_states:
                    del self.trade_states[d]
                continue
            # [风控] 跌停板不卖出（无法成交）
            _chg = d.close[0] / d.close[-1] - 1 if d.close[-1] > 0 else 0
            if _chg <= -0.0955:
                self.log(f"  [风控] {d._name} 跌停({_chg*100:.1f}%)，暂缓平仓")
                continue
            o = self.close(d)
            if o is not None:
                self._pending_orders[d] = o
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
