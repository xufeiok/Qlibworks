import os
import sys

# 动态添加 e:\Quant\backtrader_superplot\backtrader_superplot 到 PYTHONPATH 以便加载特定的 backtrader
bt_superplot_dir = os.path.abspath(r"e:\Quant\backtrader_superplot\backtrader_superplot")
custom_bt_dir = os.path.join(bt_superplot_dir, "backtrader-1.9.74.123", "backtrader-1.9.74.123")

if custom_bt_dir not in sys.path:
    sys.path.insert(0, custom_bt_dir)
if bt_superplot_dir not in sys.path:
    sys.path.insert(0, bt_superplot_dir)

from .bt_strategy import QlibPandasData, BaseQlibStrategy, EnhancedQlibStrategy
from .bt_runner import run_qlib_backtrader, run_duckdb_backtrader

__all__ = [
    'QlibPandasData',
    'BaseQlibStrategy',
    'EnhancedQlibStrategy',
    'run_qlib_backtrader',
    'run_duckdb_backtrader'
]
