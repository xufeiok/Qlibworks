# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from qlib.data.dataset.processor import Processor, get_group_columns

class CSQuantileNorm(Processor):
    """截面分位数标准化（CSQuantileNorm）。

    作用：对每个交易日的截面数据做秩变换，将特征映射到 [0, 1] 的分位数（rank(pct=True)）。

    设计动机：
    - 对树模型（XGBoost/LightGBM 等）更友好：树主要依赖相对大小与排序信息，分位数化可保留单调排名关系。
    - 严格有界：将特征压到 [0, 1]，可降低极端值对训练的影响。
    - 相比 Z-Score：当截面方差很大或分布重尾时，Z-Score 可能放大极端点并扰动截面排序；分位数化通常更稳。
    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df: pd.DataFrame):
        cols = get_group_columns(df, self.fields_group)

        # 按 datetime 分组做截面排名，rank(pct=True) 返回分位数（0.0~1.0）
        df[cols] = df[cols].groupby(level="datetime", group_keys=False).rank(pct=True)

        return df
