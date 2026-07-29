"""
通达信补数辅助逻辑测试
"""
import unittest
from decimal import Decimal

import pandas as pd

from qlworks.data.tdx_sync import build_daily_prices_frame, scale_forward_factors


class TestTdxSyncHelpers(unittest.TestCase):
    def test_scale_forward_factors_aligns_with_clickhouse_anchor(self):
        forward_factor_df = pd.DataFrame(
            {
                "600000.SH": [0.95, 0.95, 0.95],
                "000001.SZ": [1.00, 1.00, 1.00],
            },
            index=pd.to_datetime(["2026-06-23", "2026-07-10", "2026-07-14"]),
        )
        anchor_df = pd.DataFrame(
            {
                "ts_code": ["600000.SH", "000001.SZ"],
                "trade_date": pd.to_datetime(["2026-06-23", "2026-06-23"]),
                "adj_factor": [Decimal("95.0000"), Decimal("139.0080")],
            }
        )

        result = scale_forward_factors(forward_factor_df, anchor_df, start_date="2026-07-10")

        self.assertEqual(len(result), 4)
        self.assertEqual(
            result.loc[result["ts_code"] == "600000.SH", "adj_factor"].iloc[0],
            Decimal("95.0000"),
        )
        self.assertEqual(
            result.loc[result["ts_code"] == "000001.SZ", "adj_factor"].iloc[0],
            Decimal("139.0080"),
        )

    def test_build_daily_prices_frame_computes_pre_close_change_and_pct(self):
        market_data = {
            "Open": pd.DataFrame({"600000.SH": [10.0, 10.2]}, index=pd.to_datetime(["2026-07-10", "2026-07-11"])),
            "High": pd.DataFrame({"600000.SH": [10.3, 10.4]}, index=pd.to_datetime(["2026-07-10", "2026-07-11"])),
            "Low": pd.DataFrame({"600000.SH": [9.9, 10.0]}, index=pd.to_datetime(["2026-07-10", "2026-07-11"])),
            "Close": pd.DataFrame({"600000.SH": [10.1, 10.3]}, index=pd.to_datetime(["2026-07-10", "2026-07-11"])),
            "Volume": pd.DataFrame({"600000.SH": [1000.0, 1200.0]}, index=pd.to_datetime(["2026-07-10", "2026-07-11"])),
            "Amount": pd.DataFrame({"600000.SH": [100.0, 123.6]}, index=pd.to_datetime(["2026-07-10", "2026-07-11"])),
        }

        result = build_daily_prices_frame(market_data, start_date="2026-07-11")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pre_close"], Decimal("10.1000"))
        self.assertEqual(result.iloc[0]["change"], Decimal("0.2000"))
        self.assertEqual(result.iloc[0]["pct_chg"], Decimal("1.9802"))


if __name__ == "__main__":
    unittest.main()
