"""
warehouse 本地增量同步辅助逻辑测试
"""
import tempfile
import unittest
from pathlib import Path

import yaml

from qlworks.evaluation.warehouse_sync import load_factor_definitions, resolve_append_start_date


class TestWarehouseSyncHelpers(unittest.TestCase):
    def test_resolve_append_start_date_uses_default_when_meta_missing(self):
        result = resolve_append_start_date(meta=None, default_start_date="2010-01-01")
        self.assertEqual(result, "2010-01-01")

    def test_resolve_append_start_date_uses_next_day_of_last_date(self):
        meta = {"data_range": {"last_date": "2026-07-14"}}
        result = resolve_append_start_date(meta=meta, default_start_date="2010-01-01")
        self.assertEqual(result, "2026-07-15")

    def test_load_factor_definitions_reads_qlib_expression_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = Path(tmp_dir) / "demo.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "demo",
                        "factors": [
                            {
                                "name": "AlphaDemo",
                                "category": "量价综合",
                                "expression": {
                                    "qlib": "Mean($close,5)",
                                    "duckdb": "AVG(close) OVER (...)",
                                },
                                "meaning": "演示因子",
                            },
                            {
                                "name": "NoExpr",
                                "expression": {"duckdb": ""},
                            },
                        ],
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            factors = load_factor_definitions(strategy_name="demo", repo_path=tmp_dir)

        self.assertEqual(len(factors), 1)
        self.assertEqual(factors[0]["name"], "AlphaDemo")
        self.assertEqual(factors[0]["qlib_expr"], "Mean($close,5)")
        self.assertEqual(factors[0]["meta"]["category"], "量价综合")


if __name__ == "__main__":
    unittest.main()
