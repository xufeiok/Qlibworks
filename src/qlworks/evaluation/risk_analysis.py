"""
风险分析模块：因子组合的尾部风险与稳定性评估。

提供：
  - VaR / CVaR（多空组合的尾部风险）
  - 最大回撤分析
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── 默认参数（可统一修改） ──
_VAR_ALPHA: float = 0.05          # VaR/CVaR 置信水平（默认 95%）
_VAR_ALPHA_99: float = 0.01       # VaR/CVaR 99% 置信水平
_VAR_MIN_OBS: int = 10            # 最小观测数

def calc_var_cvar(returns: pd.Series, alpha: float = _VAR_ALPHA) -> dict:
    """计算多空组合的 VaR 和 CVaR。

    VaR（Value at Risk）：在 alpha 置信水平下的最大预期损失。
    CVaR（Conditional VaR）：超过 VaR 的损失的均值（尾部期望损失）。

    Args:
        returns: 多空组合日收益序列
        alpha: 置信水平（默认 5%，即 95% VaR）

    Returns:
        dict: var, cvar, var_99, cvar_99, max_drawdown, max_drawdown_duration
    """
    r = returns.dropna().values
    if len(r) < _VAR_MIN_OBS:
        return {"var_95": 0.0, "cvar_95": 0.0, "var_99": 0.0, "cvar_99": 0.0,
                "max_drawdown": 0.0, "max_drawdown_duration": 0, "n_obs": 0}

    # VaR 95% 和 99%
    var_95 = float(np.percentile(r, alpha * 100))
    var_99 = float(np.percentile(r, _VAR_ALPHA_99 * 100))  # 1% VaR

    # CVaR: 尾部均值
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).sum() > 0 else var_95
    cvar_99 = float(r[r <= var_99].mean()) if (r <= var_99).sum() > 0 else var_99

    # 最大回撤
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_dd = float(np.min(drawdown))

    # 回撤持续期（最长处于回撤状态的天数）
    in_drawdown = drawdown < 0
    max_duration = 0
    current = 0
    for d in in_drawdown:
        if d:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0

    return {
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "var_99": round(var_99, 6),
        "cvar_99": round(cvar_99, 6),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration": int(max_duration),
        "n_obs": len(r),
    }
