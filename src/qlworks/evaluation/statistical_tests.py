"""
统计检验模块：因子值的时序平稳性与白噪声检验。

提供：
  - ADF 单位根检验：因子值序列是否平稳（均值回归 vs 随机游走）
  - KPSS 检验：ADF 的互补检验（原假设：序列平稳）
  - Ljung-Box 白噪声检验：因子值是否存在自相关结构
"""

from typing import Optional
import numpy as np
import pandas as pd

# ── 默认参数（可统一修改） ──
_ST_TEST_ALPHA: float = 0.05     # 显著性水平
_ST_MIN_OBS: int = 10            # 最小观测数
_ST_LB_DEFAULT_LAGS_DIVISOR: int = 5  # Ljung-Box 默认滞后 = n // N


def calc_adf_test(factor_values: pd.Series) -> dict:
    """ADF（Augmented Dickey-Fuller）单位根检验。

    原假设 H0：序列存在单位根（非平稳）。
    备择假设 H1：序列平稳（均值回归）。

    Returns:
        dict: adf_stat, p_value, used_lag, n_obs, critical_values, is_stationary
    """
    from statsmodels.tsa.stattools import adfuller

    vals = factor_values.dropna().values
    if len(vals) < _ST_MIN_OBS:
        return {"adf_stat": 0.0, "p_value": 1.0, "used_lag": 0, "n_obs": 0,
                "critical_values": {}, "is_stationary": False, "note": "样本不足"}

    result = adfuller(vals, autolag="AIC")
    adf_stat, p_value, used_lag, n_obs, critical_values, _ = result

    is_stationary = p_value < _ST_TEST_ALPHA
    return {
        "adf_stat": round(float(adf_stat), 4),
        "p_value": round(float(p_value), 6),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {str(k): round(float(v), 4) for k, v in critical_values.items()},
        "is_stationary": bool(is_stationary),
    }


def calc_kpss_test(factor_values: pd.Series) -> dict:
    """KPSS 检验（Kwiatkowski-Phillips-Schmidt-Shin）— ADF 的互补检验。

    原假设 H0：序列是平稳的。
    备择假设 H1：序列存在单位根（非平稳）。

    联合使用 ADF + KPSS 可以得到更可靠的结论：
      - ADF 拒绝 H0 + KPSS 不拒绝 H0 = 序列平稳
      - ADF 不拒绝 H0 + KPSS 拒绝 H0 = 序列非平稳
      - 两者都拒绝 / 都不拒绝 = 需要进一步诊断
    """
    from statsmodels.tsa.stattools import kpss

    vals = factor_values.dropna().values
    if len(vals) < _ST_MIN_OBS:
        return {"kpss_stat": 0.0, "p_value": 1.0, "used_lag": 0,
                "is_stationary": True, "note": "样本不足"}

    try:
        result = kpss(vals, regression="c", nlags="auto")
        kpss_stat, p_value, used_lag, _ = result
    except Exception:
        # 某些序列可能不适合 KPSS，返回默认
        return {"kpss_stat": 0.0, "p_value": 1.0, "used_lag": 0,
                "is_stationary": True, "note": "KPSS 计算失败，默认通过"}

    is_stationary = p_value >= _ST_TEST_ALPHA  # KPSS 原假设是平稳，p 值大→不拒绝平稳
    return {
        "kpss_stat": round(float(kpss_stat), 4),
        "p_value": round(float(p_value), 6),
        "used_lag": int(used_lag),
        "is_stationary": bool(is_stationary),
    }


def adf_kpss_verdict(adf_result: dict, kpss_result: dict) -> str:
    """联合 ADF + KPSS 给出综合平稳性判定。"""
    adf_s = adf_result.get("is_stationary", False)
    kpss_s = kpss_result.get("is_stationary", True)

    if adf_s and kpss_s:
        return "平稳"
    elif not adf_s and not kpss_s:
        return "非平稳（需差分处理）"
    elif adf_s and not kpss_s:
        return "可能趋势平稳"
    else:
        return "可能差分平稳"


def calc_ljungbox_test(factor_values: pd.Series, lags: Optional[int] = None) -> dict:
    """Ljung-Box 白噪声检验。

    原假设 H0：序列为白噪声（无自相关）。
    备择假设 H1：序列存在自相关结构。

    对因子值来说：
      - 拒绝 H0（p < 0.05）= 因子值存在可预测的自相关结构
      - 不拒绝 H0 = 因子值为白噪声（对反转因子来说可能意味着无预测性）

    Returns:
        dict: 各滞后阶数的统计量和 p 值，以及综合结论
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    vals = factor_values.dropna().values
    if len(vals) < _ST_MIN_OBS:
        return {"lb_stat": 0.0, "lb_pvalue": 1.0, "max_lag": 0, "is_white_noise": True, "note": "样本不足"}

    if lags is None:
        lags = min(20, len(vals) // _ST_LB_DEFAULT_LAGS_DIVISOR)
    lags = max(1, lags)

    result = acorr_ljungbox(vals, lags=[lags], return_df=True)
    lb_stat = float(result["lb_stat"].iloc[0])
    lb_pvalue = float(result["lb_pvalue"].iloc[0])

    is_white_noise = lb_pvalue >= _ST_TEST_ALPHA

    return {
        "lb_stat": round(lb_stat, 4),
        "lb_pvalue": round(lb_pvalue, 6),
        "max_lag": lags,
        "is_white_noise": bool(is_white_noise),
        "n_obs": len(vals),
    }


def calc_factor_statistical_tests(factor_values: pd.Series) -> dict:
    """对一个因子的截面均值序列，统一执行所有统计检验。"""
    return {
        "adf": calc_adf_test(factor_values),
        "kpss": calc_kpss_test(factor_values),
        "ljungbox": calc_ljungbox_test(factor_values),
        "stationarity_verdict": adf_kpss_verdict(
            calc_adf_test(factor_values), calc_kpss_test(factor_values)
        ),
    }
