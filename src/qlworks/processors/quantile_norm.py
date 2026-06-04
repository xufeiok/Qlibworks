"""
分位数归一化处理器

用于将特征值转换到均匀分布，减少异常值影响，使不同量纲的因子具有可比性。
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class CSQuantileNorm:
    """
    Cross-Sectional Quantile Normalization 横截面分位数归一化
    
    将每个时间点的横截面特征值转换为其分位数位置，使因子值在 [0, 1] 区间内均匀分布。
    
    优点：
    - 对异常值具有鲁棒性
    - 使不同量纲的因子具有可比性
    - 适用于非线性模型和树模型
    """
    
    def __init__(self, eps: float = 1e-6, **kwargs):
        """
        参数：
        - eps: 防止除零的小值，确保输出在 (0, 1) 区间内
        - **kwargs: 其他 Qlib 传递的参数（如 fields_group）
        """
        self.eps = eps
        # 忽略其他未使用的参数
    
    def fit(self, df: pd.DataFrame):
        """拟合（本处理器不需要拟合）"""
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行横截面分位数归一化"""
        # 按行（时间维度）进行横截面归一化
        return df.apply(self._quantile_normalize_series, axis=1)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.transform(df)
    
    def _quantile_normalize_series(self, s: pd.Series) -> pd.Series:
        """对单个序列进行分位数归一化"""
        # 使用 rank 获取分位数位置
        ranks = s.rank(pct=True, na_option='keep')
        # 处理边界值，确保在 (0, 1) 区间内
        ranks = ranks.clip(self.eps, 1 - self.eps)
        return ranks
    
    def readonly(self):
        """Qlib 要求的方法：返回是否只读（不修改原始数据）"""
        return False
    
    def is_for_infer(self):
        """Qlib 要求的方法：返回是否用于推理阶段"""
        return True


class QuantileNormProcessor:
    """
    分位数归一化处理器（通用接口）
    """
    
    def __init__(self, axis: int = 0, eps: float = 1e-6):
        """
        参数：
        - axis: 归一化轴，0 表示按行（时间维度），1 表示按列（特征维度）
        - eps: 防止除零的小值
        """
        self.axis = axis
        self.eps = eps
    
    def fit(self, df: pd.DataFrame):
        """拟合（本处理器不需要拟合）"""
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行分位数归一化"""
        if self.axis == 0:
            # 按行（时间维度）进行横截面归一化
            return df.apply(self._quantile_normalize_series, axis=1)
        else:
            # 按列（特征维度）进行归一化
            return df.apply(self._quantile_normalize_series, axis=0)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.transform(df)
    
    def _quantile_normalize_series(self, s: pd.Series) -> pd.Series:
        """对单个序列进行分位数归一化"""
        # 使用 rank 获取分位数位置
        ranks = s.rank(pct=True, na_option='keep')
        # 处理边界值，确保在 (0, 1) 区间内
        ranks = ranks.clip(self.eps, 1 - self.eps)
        return ranks


def get_quantile_norm_processor(**kwargs):
    """工厂函数：创建分位数归一化处理器"""
    return QuantileNormProcessor(**kwargs)
