"""
分位数归一化处理器

用于将特征值转换到均匀分布，减少异常值影响，使不同量纲的因子具有可比性。
"""

from __future__ import annotations

import pandas as pd

CS_QUANTILE_DATE_CHUNK_SIZE = 256
QUANTILE_NORM_OUTPUT_DTYPE = "float32"


def _clip_ranks(frame: pd.DataFrame, eps: float) -> pd.DataFrame:
    """
    对 rank 输出做边界裁剪，确保结果稳定落在 (0, 1) 区间。
    """
    return frame.clip(eps, 1 - eps)


def _finalize_ranks(frame: pd.DataFrame, eps: float) -> pd.DataFrame:
    """
    对 rank 结果做统一收尾，降低后续处理链的内存压力。
    """
    return _clip_ranks(frame, eps).astype(QUANTILE_NORM_OUTPUT_DTYPE, copy=False)


def _rank_by_datetime(frame: pd.DataFrame, eps: float) -> pd.DataFrame:
    """
    按 datetime 做截面 rank；当日期过多时按日期块分段处理，降低峰值内存。
    """
    unique_datetimes = frame.index.get_level_values("datetime").unique()
    if len(unique_datetimes) <= CS_QUANTILE_DATE_CHUNK_SIZE:
        return _finalize_ranks(
            frame.groupby(level="datetime", group_keys=False).rank(
                pct=True,
                method="average",
                na_option="keep",
            ),
            eps,
        )

    ranked_parts: list[pd.DataFrame] = []
    datetime_index = frame.index.get_level_values("datetime")
    for start in range(0, len(unique_datetimes), CS_QUANTILE_DATE_CHUNK_SIZE):
        chunk_datetimes = unique_datetimes[start:start + CS_QUANTILE_DATE_CHUNK_SIZE]
        chunk_mask = datetime_index.isin(chunk_datetimes)
        chunk_frame = frame.loc[chunk_mask]
        ranked_parts.append(
            _finalize_ranks(
                chunk_frame.groupby(level="datetime", group_keys=False).rank(
                    pct=True,
                    method="average",
                    na_option="keep",
                ),
                eps,
            )
        )
    return pd.concat(ranked_parts).reindex(frame.index)


class CSQuantileNorm:
    """
    Cross-Sectional Quantile Normalization 横截面分位数归一化

    语义:
    - 对每个 datetime 的股票截面分别做 rank(pct=True)
    - 每个因子列独立归一化
    - NaN 保持不变
    """

    def __init__(self, eps: float = 1e-6, fields_group: str | None = None, **kwargs):
        """
        参数：
        - eps: 防止除零的小值，确保输出在 (0, 1) 区间内
        - fields_group: 当输入列为 MultiIndex 时，仅处理指定组（如 feature / label）
        - **kwargs: 兼容 Qlib 额外传参
        """
        self.eps = eps
        self.fields_group = fields_group

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Qlib 处理器接口：允许对象被直接调用"""
        return self.fit_transform(df)

    def fit(self, df: pd.DataFrame):
        """拟合（本处理器不需要拟合）"""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行按日期截面的分位数归一化"""
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("CSQuantileNorm 要求输入索引包含 datetime MultiIndex。")

        if self.fields_group is not None and isinstance(df.columns, pd.MultiIndex):
            if self.fields_group not in df.columns.get_level_values(0):
                return df
            result = df.copy(deep=False)
            target = df[self.fields_group]
            ranked = _rank_by_datetime(target, self.eps)
            for column_name in target.columns:
                result[(self.fields_group, column_name)] = ranked[column_name]
            return result

        return _rank_by_datetime(df, self.eps)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.transform(df)

    def readonly(self):
        """Qlib 要求的方法：返回是否只读（不修改原始数据）"""
        return False

    def is_for_infer(self):
        """Qlib 要求的方法：返回是否用于推理阶段"""
        return True


class QuantileNormProcessor:
    """
    分位数归一化处理器（通用接口）

    兼容旧语义：
    - axis=0: 按行归一化（DataFrame.rank(axis=1)）
    - axis=1: 按列归一化（DataFrame.rank(axis=0)）
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
        rank_axis = 1 if self.axis == 0 else 0
        return _finalize_ranks(
            df.rank(axis=rank_axis, pct=True, method="average", na_option="keep"),
            self.eps,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.transform(df)


def _ranks_to_gauss(frame: pd.DataFrame) -> pd.DataFrame:
    """
    将 (0,1) 均匀分布的 rank 值映射到 N(0,1) 标准正态分布。

    变换公式:  √2 × erfinv(2x - 1)

    参数:
    - frame: 已做过 rank(pct=True) 的 DataFrame，值在 (0,1) 区间

    返回:
    - 标准正态分布 N(0,1) 的 DataFrame，NaN 保持
    """
    from scipy.special import erfinv

    result = frame.copy()
    # 暂时填充 NaN 以便做向量化变换，最后回写 NaN
    nan_mask = result.isna()
    result = result.fillna(0.5)  # 0.5 → erfinv(0)=0，中性值

    transformed = erfinv(2.0 * result.values.astype("float64") - 1.0) * 1.4142135623730951  # √2
    result[:] = transformed.astype(QUANTILE_NORM_OUTPUT_DTYPE, copy=False)
    result[nan_mask] = float("nan")
    return result


class CSQuantileGaussNorm:
    """
    Cross-Sectional Quantile Gauss Normalization 横截面分位数高斯正态化

    算法:
    1. 对每个 datetime 截面，每个因子列独立做 rank(pct=True) → 映射到 (0,1) 均匀分布
    2. 用 erfinv 逆误差函数映射到 N(0,1) 标准正态分布

    语义:
    - 等价于 scikit-learn 的 QuantileTransformer(output_distribution='normal')
    - 消除原始特征的偏态/长尾/极端值，输出完美正态分布
    - 每截面独立变换，不使用全局统计量 → 无前视偏差
    - 对线性模型最优；树模型使用 CSQuantileNorm 即可（高斯变换对树模型无增益）

    与 CSQuantileNorm 的区别：
    - CSQuantileNorm:  rank → (0,1) 均匀分布 → 适合树模型
    - CSQuantileGaussNorm: rank → N(0,1) 正态分布   → 适合线性/NN 模型
    """

    def __init__(self, eps: float = 1e-6, fields_group: str | None = None, **kwargs):
        """
        参数：
        - eps: 防止除零的小值，确保 rank 在 (eps, 1-eps) 区间，避免 erfinv 产生 ±∞
        - fields_group: 当输入列为 MultiIndex 时，仅处理指定组（如 feature）
        - **kwargs: 兼容 Qlib 额外传参
        """
        self.eps = eps
        self.fields_group = fields_group

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Qlib 处理器接口"""
        return self.fit_transform(df)

    def fit(self, df: pd.DataFrame):
        """拟合（本处理器不需要拟合，截面独立变换）"""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行按日期截面的分位数高斯正态化"""
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("CSQuantileGaussNorm 要求输入索引包含 datetime MultiIndex。")

        if self.fields_group is not None and isinstance(df.columns, pd.MultiIndex):
            if self.fields_group not in df.columns.get_level_values(0):
                return df
            result = df.copy(deep=False)
            target = df[self.fields_group]
            ranked = _rank_by_datetime(target, self.eps)
            gaussed = _ranks_to_gauss(ranked)
            for column_name in target.columns:
                result[(self.fields_group, column_name)] = gaussed[column_name]
            return result

        ranked = _rank_by_datetime(df, self.eps)
        return _ranks_to_gauss(ranked)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.transform(df)

    def readonly(self):
        """Qlib 要求的方法"""
        return False

    def is_for_infer(self):
        """Qlib 要求的方法：用于推理阶段"""
        return True


def get_quantile_norm_processor(**kwargs):
    """工厂函数：创建分位数归一化处理器"""
    return QuantileNormProcessor(**kwargs)
