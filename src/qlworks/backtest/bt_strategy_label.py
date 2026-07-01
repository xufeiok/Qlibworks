import math
import backtrader as bt

class LabelConsistencyStrategy(bt.Strategy):
    """
    标签一致性专用回测策略
    严格对齐模型训练标签: T+1 开盘买入，T+5 收盘卖出
    无任何止损、止盈、动态轮动等风控逻辑，仅验证最纯粹的模型预测力
    """
    params = dict(
        top_k=20,
        score_threshold=0.7,
        holding_days=5,
        buy_pct=0.95,
        log_enabled=True,
        reverse_test=False,         # 反向测试开关，True则买入得分最低的股票
        volume_limit_pct=0.10,      # 单笔订单不超过当日成交量的10%
    )

    def __init__(self):
        # 记录上一批买单是在哪一天（datas[0] 的 index）发出的
        self.last_buy_idx = -999 

    def log(self, txt, dt=None):
        if self.p.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"[{dt.isoformat()}] {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"买入成交 | {order.data._name} | 价格: {order.executed.price:.2f} | 数量: {order.executed.size}")
            else:
                self.log(f"卖出成交 | {order.data._name} | 价格: {order.executed.price:.2f} | 数量: {abs(order.executed.size)}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            reason = getattr(order.info, "reason", "未知原因")
            direction = "buy" if order.isbuy() else "sell" if order.issell() else "unknown"
            self.log(f"订单异常 | {order.data._name} | 状态: {order.getstatusname()} | 方向: {direction} | 数量: {abs(order.size)} | 现金: {self.broker.getcash():.2f} | 原因: {reason}")

    def next(self):
        # 当前所在的 bar 索引
        curr_idx = len(self)
        
        # 计算当前总持仓数
        holding_count = sum(1 for data in self.datas if self.getposition(data).size > 0)

        # 阶段1：判断是否到了卖出时间 (发出买单后的第 holding_days 个 bar)
        # [Renaissance 改进] 经验证：Order.Close 在每日数据中于下一 bar 执行，
        # 成交价使用当前 bar (发出卖单 bar) 的收盘价 (pannotated)。
        # 因此卖单在 T+holding_days 发出 → 成交价=Close(T+holding_days)，
        # 与标签 Ref($close,-5)/Ref($open,-1)-1 完全对齐。
        if self.last_buy_idx > 0 and curr_idx - self.last_buy_idx == self.p.holding_days and holding_count > 0:
            sell_count = 0
            for data in self.datas:
                pos = self.getposition(data)
                if pos.size > 0:
                    self.sell(data=data, size=pos.size, exectype=bt.Order.Close)
                    sell_count += 1
            if sell_count > 0:
                self.log(f">>> [T+{self.p.holding_days-1}] 发出 {sell_count} 笔收盘卖出指令 (Order.Close)")

        # 阶段2：如果当前完全空仓（初始状态，或者卖单已经完全成交），则可以开始新一轮买入
        elif holding_count == 0 and (self.last_buy_idx < 0 or curr_idx - self.last_buy_idx >= self.p.holding_days):
            self.last_buy_idx = curr_idx  # 更新买入基准日
            
            candidates = []
            for data in self.datas:
                if len(data) > 0 and data.volume[0] > 0:
                    score = getattr(data, 'score', None)
                    if score is not None and not math.isnan(score[0]):
                        if self.p.reverse_test:
                            # 反向测试：买入得分最低的，阈值反转 (例如原来是 >= 0.7, 现在是 <= 0.3)
                            if score[0] <= (1.0 - self.p.score_threshold):
                                candidates.append((score[0], data))
                        else:
                            # 正向测试：买入得分最高的
                            if score[0] >= self.p.score_threshold:
                                candidates.append((score[0], data))
            
            # 如果是反向测试，从小到大排序；正向测试从大到小排序
            candidates.sort(key=lambda x: x[0], reverse=not self.p.reverse_test)
            targets = candidates[:self.p.top_k]
            
            if targets:
                self.log(f">>> [T=0] 选股完成，发出买入指令，共 {len(targets)} 只 (Order.Market)")
                # 计算目标现金
                target_cash = self.broker.getvalue() * self.p.buy_pct
                per_stock_cash = target_cash / len(targets)
                
                for score, data in targets:
                    est_price = data.close[0]
                    if math.isnan(est_price) or est_price <= 0:
                        continue
                    size = int((per_stock_cash / est_price) // 100 * 100)

                    # [Virtu 改进] 成交量限制：单笔订单不超过当日成交量的 volume_limit_pct
                    daily_volume = data.volume[0]
                    if daily_volume is not None and math.isfinite(daily_volume) and daily_volume > 0:
                        max_shares_by_volume = int(daily_volume * self.p.volume_limit_pct // 100 * 100)
                        if max_shares_by_volume < 100:
                            self.log(f"  跳过 {data._name}: 当日成交量过小 ({daily_volume})")
                            continue
                        if size > max_shares_by_volume:
                            self.log(f"  缩容 {data._name}: {size} → {max_shares_by_volume} (成交量限制)")
                            size = max_shares_by_volume

                    if size > 0:
                        self.buy(data=data, size=size, exectype=bt.Order.Market)
