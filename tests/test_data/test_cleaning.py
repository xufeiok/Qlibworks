"""
数据清洗模块测试
"""
import numpy as np
import pandas as pd
from qlworks.data.cleaning import fill_group_missing, winsorize_by_mad


def test_fill_group_missing():
    dates = pd.date_range('2020-01-01', periods=3)
    idx = pd.MultiIndex.from_product([dates, ['A', 'B']], names=['datetime', 'instrument'])
    df = pd.DataFrame({'close': [1.0, 4.0, np.nan, np.nan, 3.0, 6.0]}, index=idx)
    result = fill_group_missing(df)
    assert result.isna().sum().sum() == 0


def test_fill_group_missing_empty():
    result = fill_group_missing(pd.DataFrame())
    assert result.empty


def test_winsorize_by_mad():
    df = pd.DataFrame({'val': [1.0, 2.0, 3.0, 100.0, 4.0, 5.0]})
    result = winsorize_by_mad(df, n_threshold=3.0)
    assert result['val'].max() < 100.0
