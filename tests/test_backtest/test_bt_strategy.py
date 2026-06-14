"""
回测策略日志测试
"""
import unittest
from datetime import date
from types import SimpleNamespace

from qlworks.backtest.bt_strategy import EnhancedQlibStrategy


class _FakeBroker:
    def __init__(self, cash: float):
        self._cash = cash

    def getcash(self):
        return self._cash

    def getvalue(self):
        return self._cash + 1000.0


class _FakeOrder:
    Submitted = 1
    Accepted = 2
    Completed = 4
    Canceled = 5
    Margin = 7
    Rejected = 8

    def __init__(self, *, status, size, side, stock_code="000001.SZ", reason=None, price=12.34, comm=1.23, value=1234.56):
        self.status = status
        self.created = SimpleNamespace(size=size, price=price)
        self.executed = SimpleNamespace(size=size, price=price, comm=comm, value=value)
        self.data = SimpleNamespace(_name=stock_code)
        self.info = {}
        if reason is not None:
            self.info["reason"] = reason
        self._side = side

    def isbuy(self):
        return self._side == "buy"

    def issell(self):
        return self._side == "sell"


class _FakeData:
    def __init__(self, name, score=0.0, close=10.0):
        self._name = name
        self.score = [score]
        self.close = [close]

    def __len__(self):
        return 1


class TestBtStrategyOrderDiagnostics(unittest.TestCase):
    def _build_strategy(self, cash: float = 123456.78):
        strategy = EnhancedQlibStrategy.__new__(EnhancedQlibStrategy)
        strategy.broker = _FakeBroker(cash)
        strategy.getposition = lambda data: SimpleNamespace(size=800)
        return strategy

    def test_format_abnormal_order_log_for_margin_sell(self):
        strategy = self._build_strategy()
        order = _FakeOrder(status=_FakeOrder.Margin, size=-1200, side="sell")

        message = strategy._format_abnormal_order_log(order)

        self.assertIn("订单异常", message)
        self.assertIn("状态: Margin", message)
        self.assertIn("方向: sell", message)
        self.assertIn("数量: 1200.00", message)
        self.assertIn("现金: 123456.78", message)
        self.assertIn("原因: 可用资金不足", message)

    def test_format_abnormal_order_log_prefers_custom_reason(self):
        strategy = self._build_strategy(cash=888.0)
        order = _FakeOrder(
            status=_FakeOrder.Rejected,
            size=500,
            side="buy",
            stock_code="600000.SH",
            reason="价格超出涨跌停限制",
        )

        message = strategy._format_abnormal_order_log(order)

        self.assertIn("600000.SH", message)
        self.assertIn("状态: Rejected", message)
        self.assertIn("方向: buy", message)
        self.assertIn("数量: 500.00", message)
        self.assertIn("现金: 888.00", message)
        self.assertIn("原因: 价格超出涨跌停限制", message)

    def test_format_completed_order_log_for_buy(self):
        strategy = self._build_strategy(cash=45678.9)
        order = _FakeOrder(status=_FakeOrder.Completed, size=1200, side="buy", price=10.01, comm=5.67, value=12012.0)

        message = strategy._format_completed_order_log(order)

        self.assertIn("买入成交", message)
        self.assertIn("000001.SZ", message)
        self.assertIn("方向: buy", message)
        self.assertIn("价格: 10.0100", message)
        self.assertIn("数量: 1200.00", message)
        self.assertIn("手续费: 5.67", message)
        self.assertIn("现金: 45678.90", message)
        self.assertIn("持仓: 800.00", message)

    def test_format_completed_order_log_for_sell_uses_abs_size(self):
        strategy = self._build_strategy(cash=99887.66)
        order = _FakeOrder(status=_FakeOrder.Completed, size=-900, side="sell", stock_code="300001.SZ", price=8.88, comm=3.21, value=7992.0)

        message = strategy._format_completed_order_log(order)

        self.assertIn("卖出成交", message)
        self.assertIn("300001.SZ", message)
        self.assertIn("方向: sell", message)
        self.assertIn("价格: 8.8800", message)
        self.assertIn("数量: 900.00", message)
        self.assertIn("手续费: 3.21", message)
        self.assertIn("现金: 99887.66", message)

    def test_plan_rebalance_actions_only_returns_tuesday_sell_list(self):
        strategy = self._build_strategy()
        data_a = _FakeData("A")
        data_b = _FakeData("B")
        data_c = _FakeData("C")
        data_d = _FakeData("D")
        data_e = _FakeData("E")
        strategy.instruments = [data_a, data_b, data_c, data_d, data_e]
        strategy.p = SimpleNamespace(top_k=4)
        pos_map = {"A": 100, "B": 100, "C": 100, "D": 0, "E": 0}
        strategy.getposition = lambda data: SimpleNamespace(size=pos_map.get(data._name, 0))

        to_sell, buy_count = strategy._plan_rebalance_actions([data_b, data_c, data_d, data_e])

        self.assertEqual([d._name for d in to_sell], ["A"])
        self.assertEqual(buy_count, 2)

    def test_select_pending_buys_uses_wednesday_scores(self):
        strategy = self._build_strategy()
        data_a = _FakeData("A", score=0.60)
        data_b = _FakeData("B", score=0.95)
        data_c = _FakeData("C", score=0.50)
        data_d = _FakeData("D", score=0.88)
        data_e = _FakeData("E", score=0.92)
        strategy.instruments = [data_a, data_b, data_c, data_d, data_e]
        strategy.p = SimpleNamespace(score_threshold=0.7, top_k=20)
        pos_map = {"A": 100, "B": 0, "C": 100, "D": 0, "E": 0}
        strategy.getposition = lambda data: SimpleNamespace(size=pos_map.get(data._name, 0))

        buy_candidates = strategy._select_pending_buy_candidates(buy_count=2)

        self.assertEqual([d._name for d in buy_candidates], ["B", "E"])

    def test_target_buy_count_builds_full_top_k_when_starting_empty(self):
        strategy = self._build_strategy()
        data_a = _FakeData("A")
        data_b = _FakeData("B")
        data_c = _FakeData("C")
        strategy.instruments = [data_a, data_b, data_c]
        strategy.p = SimpleNamespace(top_k=3)
        strategy.getposition = lambda data: SimpleNamespace(size=0)

        buy_count = strategy._get_target_buy_count()

        self.assertEqual(buy_count, 3)

    def test_target_buy_count_refills_gap_to_top_k(self):
        strategy = self._build_strategy()
        data_a = _FakeData("A")
        data_b = _FakeData("B")
        data_c = _FakeData("C")
        data_d = _FakeData("D")
        strategy.instruments = [data_a, data_b, data_c, data_d]
        strategy.p = SimpleNamespace(top_k=4)
        pos_map = {"A": 100, "B": 100, "C": 0, "D": 0}
        strategy.getposition = lambda data: SimpleNamespace(size=pos_map.get(data._name, 0))

        buy_count = strategy._get_target_buy_count()

        self.assertEqual(buy_count, 2)

    def test_is_buy_execution_day_only_matches_wednesday(self):
        strategy = self._build_strategy()
        strategy.p = SimpleNamespace(buy_weekday=2)

        self.assertTrue(strategy._is_buy_execution_day(date(2026, 6, 10)))
        self.assertFalse(strategy._is_buy_execution_day(date(2026, 6, 11)))

    def test_is_rebalance_signal_day_only_matches_tuesday(self):
        strategy = self._build_strategy()
        strategy.p = SimpleNamespace(rebalance_signal_weekday=1)

        self.assertTrue(strategy._is_rebalance_signal_day(date(2026, 6, 9)))
        self.assertFalse(strategy._is_rebalance_signal_day(date(2026, 6, 10)))


if __name__ == "__main__":
    unittest.main()
