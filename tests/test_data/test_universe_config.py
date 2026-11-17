"""统一股票池配置回归测试。"""

import unittest

from qlworks.evaluation.config import DEFAULT_CONFIG
from qlworks.pipeline_config import INSTRUMENTS, resolve_instrument_codes


class TestUniverseConfig(unittest.TestCase):
    def test_default_instruments_are_main_board(self):
        self.assertEqual(INSTRUMENTS, "main_board")
        self.assertEqual(DEFAULT_CONFIG.instruments, "main_board")
        self.assertEqual(DEFAULT_CONFIG.robustness_sub_pools, ["main_board"])

    def test_resolve_main_board_codes(self):
        codes = resolve_instrument_codes(INSTRUMENTS)
        self.assertGreater(len(codes), 3000)
        self.assertIn("000001.SZ", codes)
        self.assertTrue(all(code.endswith((".SH", ".SZ")) for code in codes))


if __name__ == "__main__":
    unittest.main()
