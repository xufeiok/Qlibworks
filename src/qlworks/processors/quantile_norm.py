# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from qlib.data.dataset.processor import Processor, get_group_columns

class CSQuantileNorm(Processor):
    """
    Cross Sectional Quantile Normalization.
    
    Transforms cross-sectional data into uniform distribution [0, 1] using rank/quantile.
    This is highly recommended for tree-based models (XGBoost/LightGBM) as it preserves 
    the monotonic rank information while strictly bounding the features to avoid outlier impacts,
    unlike standard Z-Score which can distort cross-sectional ranks when variance is high.
    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df: pd.DataFrame):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        
        # Group by datetime and compute cross-sectional rank percentage (0 to 1)
        # pct=True returns quantile (0.0 to 1.0)
        df[cols] = df[cols].groupby(level="datetime", group_keys=False).rank(pct=True)
        
        return df
