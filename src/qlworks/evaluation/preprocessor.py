"""
因子预处理：去极值、标准化、中性化。

复用策略（按优先级）:
  1. qlworks.data.cleaning.winsorize_by_mad  → winsorize() 优先调用
  2. qlworks.processors.neutralize.CSNeutralize  → 只能在 Qlib DataHandler 流水线内运行，此处保留独立实现
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def winsorize(
    series: pd.Series,
    method: str = "mad",
    threshold: float = 5.0,
) -> pd.Series:
    """去极值处理 — 优先复用 qlworks.data.cleaning.winsorize_by_mad。"""
    s = series.copy()
    valid = s.notna()
    if valid.sum() < 10:
        return s

    if method == "mad":
        try:
            from qlworks.data.cleaning import winsorize_by_mad
            df = s.to_frame("value")
            trimmed = winsorize_by_mad(df, columns=["value"], n_threshold=threshold)
            return trimmed["value"]
        except (ImportError, Exception):
            pass

    vals = s[valid]
    if method == "mad":
        median = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - median))
        if pd.isna(mad) or mad < 1e-12:
            mad = np.nanstd(vals)
        if pd.isna(mad) or mad < 1e-12:
            mad = 1.0
        lower = median - threshold * 1.4826 * mad
        upper = median + threshold * 1.4826 * mad
    elif method == "zscore":
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-12:
            std = 1.0
        lower = mean - threshold * std
        upper = mean + threshold * std
    else:
        return s

    s[valid] = np.clip(s[valid], lower, upper)
    return s


def standardize(
    series: pd.Series,
    method: str = "zscore",
) -> pd.Series:
    """截面标准化 — qlworks 中无直接等价模块（标准化以 Processor 形式存在于 Qlib 流水线），保留原生实现。"""
    s = series.copy()
    valid = s.notna()
    if valid.sum() < 5:
        return s

    vals = s[valid].values
    if method == "zscore":
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-12:
            std = 1.0
        s[valid] = (vals - mean) / std
    elif method == "rank":
        s[valid] = pd.Series(vals).rank(pct=True).values
    return s


def neutralize(
    factor_series: pd.Series,
    industry_series: Optional[pd.Series],
    log_market_cap: Optional[pd.Series],
    method: str = "industry_market",
) -> pd.Series:
    """行业 / 市值中性化（截面回归取残差）。

    使用 Ridge 回归（alpha=1e-5）与 CSNeutralize 一致，
    但此处不可复用 CSNeutralize（它只能作为 Qlib DataHandler Processor 运行）。
    注意：dummy 编码使用 drop_first=False，与 CSNeutralize 保持一致，
    依靠 Ridge 的 L2 惩罚自动处理共线性。
    """
    if method == "none":
        return factor_series

    df = pd.DataFrame({"factor": factor_series})
    dummies = []

    if method in ("industry", "both") and industry_series is not None:
        ind_dummies = pd.get_dummies(industry_series.astype(str), prefix="ind", drop_first=False)
        dummies.append(ind_dummies)

    if method in ("market", "both") and log_market_cap is not None:
        df["log_mkt"] = log_market_cap
        dummies.append(df[["log_mkt"]])

    if not dummies:
        return factor_series

    X = pd.concat(dummies, axis=1)
    y = df["factor"]

    valid = y.notna() & X.notna().all(axis=1)
    if valid.sum() < 20:
        return factor_series

    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1e-5, fit_intercept=True)
    reg.fit(X[valid], y[valid])
    resid = pd.Series(np.nan, index=y.index)
    resid[valid] = y[valid] - reg.predict(X[valid])
    return resid


def preprocess_factor(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: Optional[str] = None,
    mkt_cap_col: Optional[str] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """对因子进行完整预处理（按日期截面）。

    处理顺序：去极值 → 中性化 → 标准化
    （中性化在标准化之前，因为中性化回归需要原始量纲）
    """
    if config is None:
        config = {}

    winsorize_method = config.get("winsorize_method", "mad")
    winsorize_th = config.get("winsorize_threshold", 5.0)
    standardize_method = config.get("standardize_method", "zscore")
    neutralization = config.get("neutralization", "industry_market")

    result = df.copy()
    # 统一转为 float64，避免 warehouse float32 与计算 float64 不兼容
    result[factor_col] = result[factor_col].astype("float64")
    result["_raw_factor"] = result[factor_col].copy()

    for dt, grp in result.groupby("datetime"):
        idx = grp.index

        result.loc[idx, factor_col] = winsorize(
            result.loc[idx, factor_col], winsorize_method, winsorize_th
        )

        ind_series = result.loc[idx, industry_col] if industry_col else None
        mkt_series = result.loc[idx, mkt_cap_col] if mkt_cap_col else None
        result.loc[idx, factor_col] = neutralize(
            result.loc[idx, factor_col], ind_series, mkt_series, neutralization
        )

        result.loc[idx, factor_col] = standardize(
            result.loc[idx, factor_col], standardize_method
        )

    return result
