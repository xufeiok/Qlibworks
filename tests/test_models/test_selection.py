"""
特征选择模块测试
"""
import pandas as pd
from qlworks.models.selection import FeatureSelectionResult, prepare_feature_selection_data


def test_FeatureSelectionResult():
    r = FeatureSelectionResult(
        method='test',
        selected_features=['a', 'b'],
        feature_scores=pd.Series([0.5, 0.3], index=['a', 'b']),
        params={},
    )
    assert len(r.selected_features) == 2
    assert r.method == 'test'


def test_prepare_feature_selection_data_missing_label():
    df = pd.DataFrame({'f1': [1.0, 2.0], 'f2': [3.0, 4.0]})
    import pytest
    with pytest.raises(ValueError, match='缺少标签列'):
        prepare_feature_selection_data(df, label_col='LABEL0')
