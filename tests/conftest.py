"""
pytest 共享 Fixtures

提供量化测试常用的 mock 数据工厂，各子目录测试文件可直接使用。
"""

import sys
from pathlib import Path

# 确保 src 在 path 中（让 pytest 可以 import qlworks）
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from typing import List
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_instruments() -> List[str]:
    """标准 4 只股票代码池"""
    return ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]


@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    """10 个交易日序列"""
    return pd.date_range("2020-01-02", periods=10, freq="B")


@pytest.fixture
def sample_multi_index(sample_dates, sample_instruments) -> pd.MultiIndex:
    """标准量化面板 MultiIndex (datetime, instrument)"""
    return pd.MultiIndex.from_product(
        [sample_dates, sample_instruments],
        names=["datetime", "instrument"],
    )


@pytest.fixture
def sample_panel_frame(sample_multi_index) -> pd.DataFrame:
    """标准量价面板数据"""
    n = len(sample_multi_index)
    np.random.seed(42)
    return pd.DataFrame(
        {
            "open": np.random.randn(n) + 10,
            "high": np.random.randn(n) + 11,
            "low": np.random.randn(n) + 9,
            "close": np.random.randn(n) + 10,
            "volume": np.random.randint(100000, 10000000, size=n),
            "LABEL0": np.random.randn(n) * 0.02,
        },
        index=sample_multi_index,
    )
