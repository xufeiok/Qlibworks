"""
因子合成模块测试
"""
import numpy as np
import pandas as pd
from qlworks.factors.synthesis import calc_factor_correlation, synthesize_factors


def test_calc_factor_correlation():
    dates = pd.date_range('2020-01-01', periods=2)
    idx = pd.MultiIndex.from_product([dates, ['A', 'B']], names=['datetime', 'instrument'])
    df = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0], 'f2': [4.0, 3.0, 2.0, 1.0]}, index=idx)
    corr = calc_factor_correlation(df)
    assert not corr.empty
    # f1 和 f2 完全负相关
    assert abs(corr.loc['f1', 'f2'] + 1) < 0.01


def test_synthesize_factors_equal():
    dates = pd.date_range('2020-01-01', periods=2)
    idx = pd.MultiIndex.from_product([dates, ['A', 'B']], names=['datetime', 'instrument'])
    df = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0], 'f2': [4.0, 3.0, 2.0, 1.0]}, index=idx)
    composite = synthesize_factors(df, method='equal', collinearity_warning=False)
    assert len(composite) == len(df)
    assert not composite.isna().all()
