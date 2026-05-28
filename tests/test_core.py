import numpy as np 
import pandas as pd 
from qlworks.data.cleaning import fill_group_missing, winsorize_by_mad 
from qlworks.factors.synthesis import calc_factor_correlation 
from qlworks.models.selection import FeatureSelectionResult 
from qlworks.models.evaluation import select_top_instruments 

def test_fill_group_missing(): 
    dates = pd.date_range('2020-01-01', periods=3) 
    idx = pd.MultiIndex.from_product([dates, ['A','B']], names=['datetime','instrument']) 
    df = pd.DataFrame({'close': [1.0, 4.0, np.nan, np.nan, 3.0, 6.0]}, index=idx) 
    result = fill_group_missing(df) 
    assert result.isna().sum().sum() == 0 

def test_winsorize_by_mad(): 
    df = pd.DataFrame({'val': [1.0, 2.0, 3.0, 100.0, 4.0, 5.0]}) 
    result = winsorize_by_mad(df, n_threshold=3.0) 
    assert result['val'].max() < 100.0 

def test_calc_factor_correlation(): 
    dates = pd.date_range('2020-01-01', periods=2) 
    idx = pd.MultiIndex.from_product([dates, ['A','B']], names=['datetime','instrument']) 
    df = pd.DataFrame({'f1': [1.0,2.0,3.0,4.0], 'f2': [4.0,3.0,2.0,1.0]}, index=idx) 
    corr = calc_factor_correlation(df) 
    assert not corr.empty 

def test_FeatureSelectionResult(): 
    r = FeatureSelectionResult(method='test', selected_features=['a','b'], feature_scores=pd.Series([0.5,0.3], index=['a','b']), params={}) 
    assert len(r.selected_features) == 2 

def test_select_top_instruments(): 
    dates = pd.date_range('2020-01-01', periods=2) 
    idx = pd.MultiIndex.from_product([dates, ['A','B','C']], names=['datetime','instrument']) 
    s = pd.Series([0.1,0.5,0.3,0.2,0.6,0.4], index=idx) 
    top = select_top_instruments(s, top_k=2) 
    assert len(top) == 2 

if __name__ == '__main__': 
    test_fill_group_missing() 
    print('fill_group_missing: OK') 
    test_winsorize_by_mad() 
    print('winsorize_by_mad: OK') 
    test_calc_factor_correlation() 
    print('calc_factor_correlation: OK') 
    test_FeatureSelectionResult() 
    print('FeatureSelectionResult: OK') 
    test_select_top_instruments() 
    print('select_top_instruments: OK') 
    print('All core tests passed!') 
