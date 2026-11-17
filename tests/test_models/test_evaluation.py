"""
模型评估模块测试
"""
import pandas as pd
import numpy as np
from qlworks.models.evaluation import select_top_instruments, evaluate_prediction_frame


def test_select_top_instruments():
    dates = pd.date_range('2020-01-01', periods=2)
    idx = pd.MultiIndex.from_product([dates, ['A', 'B', 'C']], names=['datetime', 'instrument'])
    s = pd.Series([0.1, 0.5, 0.3, 0.2, 0.6, 0.4], index=idx)
    top = select_top_instruments(s, top_k=2)
    assert len(top) == 2
    assert all(len(v) <= 2 for v in top.values)


def test_evaluate_prediction_frame():
    dates = pd.date_range('2020-01-01', periods=3)
    instruments = [f"{str(i).zfill(6)}.SZ" for i in range(1, 31)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=['datetime', 'instrument'])
    np.random.seed(42)
    pred_scores = np.random.randn(90)
    real_labels = pred_scores * 0.1 + np.random.randn(90) * 0.05
    pred_frame = pd.DataFrame({'pred': pred_scores, 'label': real_labels}, index=idx)
    result = evaluate_prediction_frame(pred_frame)
    assert 'ic_mean' in result
    assert 'rank_ic_mean' in result
    assert 'ic_positive_rate' in result
