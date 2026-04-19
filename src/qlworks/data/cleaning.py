"""
模块说明：
此 `cleaning.py` 负责“基础数据层面”的清洗，例如：
1. 原始日线/分钟线级别的 OHLCV 缺失值向前/向后填充。
2. 基础财务数据或量价数据的逻辑异常修正（如成交量小于 0、极其离谱的错误报价裁剪）。

为什么这里没有体现“标准化（Standardization）”和“中性化（Neutralization）”？
- 职责划分原则：在 Qlib 架构中，标准化和中性化属于**特征工程（Feature Engineering）**和**模型输入预处理**范畴。
- 避免信息泄露：标准化（如 Z-Score）和中性化（如行业/市值回归）通常需要依赖横截面（Cross-Sectional）或未来滚动窗口的数据分布。如果在原始数据读取阶段就进行标准化，极易在切分训练/测试集时引入“前视偏差（Future Leakage）”。
- 正确的实现位置：标准化和中性化应当作为 `Processor`（如 `RobustZScoreNorm`、`CSZScoreNorm`）放置在 `dataset.py` 构建 `DataHandlerLP` 的流水线中，由模型在每次训练/推理切片时动态计算。
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def fill_group_missing(
    df: pd.DataFrame,
    group_level: str = "instrument",
    methods: Sequence[str] = ("ffill", "bfill"),
) -> pd.DataFrame:
    """
    功能概述：
    - 按股票维度填充缺失值，适用于面板数据清洗。
    输入：
    - df: 待清洗数据。
    - group_level: 分组索引层名称。
    - methods: 填充方法顺序。
    输出：
    - 缺失值按组处理后的 DataFrame。
    边界条件：
    - 若不存在目标分组层，则退化为全表填充。
    性能/安全注意事项：
    - 采用 Pandas 向量化分组操作，适合中大型样本。
    """
    result = df.copy()
    if not isinstance(result.index, pd.MultiIndex) or group_level not in result.index.names:
        for method in methods:
            if method == "ffill":
                result = result.ffill()
            elif method == "bfill":
                result = result.bfill()
        return result

    for method in methods:
        if method == "ffill":
            result = result.groupby(level=group_level, group_keys=False).apply(lambda x: x.ffill())
        elif method == "bfill":
            result = result.groupby(level=group_level, group_keys=False).apply(lambda x: x.bfill())
    return result


def winsorize_by_mad(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    n_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    功能概述：
    - 使用 中位数 ± N 倍 MAD (Median Absolute Deviation) 方式裁剪异常值。
    - 相比于 Z-Score (均值+标准差)，MAD 对极端值更稳健，不易被极端值本身带偏。
    """
    result = df.copy()
    cols = list(columns) if columns is not None else result.select_dtypes(include=[np.number]).columns
    for col in cols:
        series = result[col]
        median = series.median()
        # 计算 MAD (Median Absolute Deviation)
        mad = (series - median).abs().median()
        
        if pd.isna(mad) or mad == 0:
            continue
        
        # 乘以 1.4826 是为了让 MAD 和正态分布的标准差可比
        lower = median - n_threshold * 1.4826 * mad
        upper = median + n_threshold * 1.4826 * mad
        result[col] = series.clip(lower=lower, upper=upper)
    return result


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    功能概述：
    - 针对价量数据做基础清洗，包括缺失值、异常值与成交量合法性处理。
    输入：
    - df: 含 open/high/low/close/volume 或 `$open/...` 列的数据。
    输出：
    - 适合进入因子构建和数据质量评估的数据表。
    """
    result = fill_group_missing(df)
    price_cols = [c for c in result.columns if c in {"open", "high", "low", "close", "$open", "$high", "$low", "$close"}]
    # 改用更稳健的 MAD 方法去极值
    result = winsorize_by_mad(result, columns=price_cols, n_threshold=3.0)

    volume_cols = [c for c in result.columns if c in {"volume", "$volume"}]
    for col in volume_cols:
        result[col] = result[col].clip(lower=0)
    return result


if __name__ == "__main__":
    print("=== data/cleaning.py 独立调用示例 ===")
    
    # 1. 构造一份带有异常值和缺失值的模拟数据
    dates = pd.date_range("2020-01-01", periods=5)
    instruments = ["000001.SZ", "600000.SH"]
    multi_idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    
    df_mock = pd.DataFrame({
        "$close": [10.0, 15.0, np.nan, 15.5, 10.5, np.nan, 11.0, 16.0, 10000.0, 16.2],  # 包含 NaN 和极值 (10000.0)
        "$volume": [100, 200, -50, 250, 110, 210, 120, np.nan, 130, -10]              # 包含负成交量和 NaN
    }, index=multi_idx)
    
    print("\n[1] 原始脏数据:")
    print(df_mock)
    
    # 2. 执行清洗流水线
    df_cleaned = clean_ohlcv_data(df_mock)
    
    print("\n[2] 清洗后的数据:")
    print(df_cleaned)
    print("\n说明: NaN 被 ffill 填充，极值 1000.0 被 Z-Score 截断，负数成交量被 Clip 为 0。")
