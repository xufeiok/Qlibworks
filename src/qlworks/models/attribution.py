"""
因子归因分析模块

[Dimensional 学术验证] 用 Fama-French 风格的多因子模型分解策略收益，
判断超额收益是真正的 Alpha（选股能力）还是 Beta 暴露（承担了已知风险）。

用法:
    from qlworks.models.attribution import factor_attribution
    result = factor_attribution(strategy_daily_returns, factor_returns_df)
    print(f"Alpha (年化): {result['alpha_annualized']:.2%}, t-stat: {result['alpha_tstat']:.2f}")
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    import warnings as _w
    _w.warn(
        "statsmodels 未安装，因子归因将使用 numpy.lstsq 简化版（无 t-stat/p-value）。"
        "安装 statsmodels 以获取完整归因分析：pip install statsmodels",
        UserWarning,
        stacklevel=2,
    )


def factor_attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    annual_factor: int = 252,
) -> Dict[str, object]:
    """
    因子归因分析：策略收益 ~ α + Σβᵢ·Fᵢ + ε

    使用 OLS 回归将策略超额收益分解到多个风险因子上，
    截距项 α 代表剔除所有已知风险后的纯选股 Alpha。

    Args:
        strategy_returns: 策略每日收益率 Series (index=datetime)
        factor_returns: 因子每日收益率 DataFrame，每列一个因子
                       至少应有 'MKT'（市场超额收益）列
        risk_free_rate: 无风险利率（年化），默认 2%
        annual_factor: 年化因子（日频=252，周频=52）

    Returns:
        {
            "alpha_annualized": 年化 Alpha,
            "alpha_tstat": Alpha 的 t 统计量 (>2.0 为显著),
            "betas": {因子名: β} 各因子暴露,
            "beta_tstats": {因子名: t-stat},
            "p_value": 回归 F 检验 p 值,
            "r_squared": R²,
            "n_obs": 样本数,
            "method": "OLS (statsmodels)" 或 "numpy.lstsq (fallback)"
        }
    """
    if not HAS_STATSMODELS:
        return _fallback_lstsq(strategy_returns, factor_returns, risk_free_rate, annual_factor)

    # 对齐数据
    rf_daily = (1 + risk_free_rate) ** (1 / annual_factor) - 1
    excess_ret = strategy_returns - rf_daily

    merged = pd.concat([excess_ret, factor_returns], axis=1).dropna()
    if len(merged) < 30:
        return {"error": f"样本不足 (n={len(merged)})，至少需要 30 个交易日"}

    y = merged.iloc[:, 0].values
    X = merged.iloc[:, 1:].values
    X = sm.add_constant(X)  # 添加截距项

    model = sm.OLS(y, X).fit()

    betas = dict(zip(["Alpha"] + list(factor_returns.columns), model.params))
    tstats = dict(zip(["Alpha"] + list(factor_returns.columns), model.tvalues))

    return {
        "alpha_annualized": float(model.params[0]) * annual_factor,
        "alpha_tstat": float(model.tvalues[0]),
        "betas": {k: float(v) for k, v in betas.items() if k != "Alpha"},
        "alpha_beta": float(betas.get("Alpha", 0)),
        "beta_tstats": {k: float(v) for k, v in tstats.items() if k != "Alpha"},
        "p_value": float(model.f_pvalue),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
        "method": "OLS (statsmodels)",
    }


def _fallback_lstsq(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float,
    annual_factor: int,
) -> Dict[str, object]:
    """statsmodels 不可用时的 NumPy lstsq 回退。"""
    rf_daily = (1 + risk_free_rate) ** (1 / annual_factor) - 1
    excess_ret = strategy_returns - rf_daily

    merged = pd.concat([excess_ret, factor_returns], axis=1).dropna()
    if len(merged) < 30:
        return {"error": f"样本不足 (n={len(merged)})"}

    y = merged.iloc[:, 0].values
    X = np.column_stack([np.ones(len(merged)), merged.iloc[:, 1:].values])

    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    mse = np.sum(residuals**2) / (n - k) if len(residuals) > 0 else 0
    var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
    tstats = beta / np.sqrt(np.maximum(var_beta, 1e-12))

    return {
        "alpha_annualized": float(beta[0]) * annual_factor,
        "alpha_tstat": float(tstats[0]),
        "betas": {col: float(b) for col, b in zip(factor_returns.columns, beta[1:])},
        "r_squared": float(1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)) if len(residuals) > 0 else 0,
        "n_obs": int(n),
        "method": "numpy.lstsq (fallback)",
    }
