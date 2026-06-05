"""
IC 分析：信息系数、ICIR、胜率。
"""

from typing import Optional
import numpy as np
import pandas as pd


def calc_daily_ic(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    method: str = "spearman",
) -> pd.Series:
    """每日截面 IC（向量化加速版）。

    对 Spearman 方法：先 rank 再计算 Pearson 相关，等效但比逐日 apply 快 3-5 倍。
    对 Pearson 方法：直接 groupby corr。
    """
    corr_method = "spearman" if method == "spearman" else "pearson"
    tmp = df[[factor_col, label_col]].copy()
    tmp["_dt"] = df["datetime"].values

    if corr_method == "spearman":
        # Spearman = rank 后的 Pearson 相关
        tmp[[factor_col, label_col]] = tmp.groupby("_dt")[[factor_col, label_col]].rank()

    corr_mats = tmp.groupby("_dt")[[factor_col, label_col]].corr(method="pearson")
    ic = corr_mats.loc[pd.IndexSlice[:, factor_col], label_col].droplevel(1)
    ic.index.name = "datetime"
    return ic


def calc_rankic_series(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    group_col: Optional[str] = None,
) -> pd.Series:
    """分组内 Rank IC 序列（可选：按行业分组后聚合）。

    当 group_col 不为空时（如 industry），在每个行业内部计算 Rank IC，
    再按市值或等权聚合为市场整体行业中性 IC。
    返回值与 calc_daily_ic 对齐，可传入 calc_ic_stats。
    """
    tmp = df[[factor_col, label_col, group_col]].copy() if group_col else df[[factor_col, label_col]].copy()
    tmp["_dt"] = df["datetime"].values

    if group_col:
        # 行业内 Rank，然后按 date 聚合
        ranked = tmp.groupby(["_dt", group_col])[[factor_col, label_col]].rank()
        tmp[[factor_col, label_col]] = ranked
        # 每个行业-日期内的相关
        corr_mats = tmp.groupby(["_dt", group_col])[[factor_col, label_col]].corr(method="pearson")
        ic_all = corr_mats.loc[pd.IndexSlice[:, :, factor_col], label_col].droplevel(2)
        ic_all = ic_all.reset_index()
        ic_all.columns = ["_dt", "industry", "ic"]
        # 等权聚合为每日单一 IC
        daily = ic_all.groupby("_dt")["ic"].mean()
    else:
        # 全局 Rank IC
        tmp[[factor_col, label_col]] = tmp.groupby("_dt")[[factor_col, label_col]].rank()
        corr_mats = tmp.groupby("_dt")[[factor_col, label_col]].corr(method="pearson")
        daily = corr_mats.loc[pd.IndexSlice[:, factor_col], label_col].droplevel(1)

    daily.index.name = "datetime"
    return daily


def calc_ic_stats(ic_series: pd.Series, annual_factor: float = 252.0) -> dict:
    """计算 IC 统计量。"""
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 5:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "win_rate": 0.5, "t_stat": 0.0}

    ic_mean = float(ic_clean.mean())
    ic_std = float(ic_clean.std())
    icir = ic_mean / ic_std * np.sqrt(annual_factor) if ic_std > 1e-12 else 0.0
    win_rate = (ic_clean > 0).sum() / len(ic_clean) if ic_mean >= 0 else (ic_clean < 0).sum() / len(ic_clean)

    from scipy import stats
    t_stat, _ = stats.ttest_1samp(ic_clean.dropna(), 0)
    t_stat = float(t_stat)

    return {
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "icir": round(icir, 4),
        "win_rate": round(win_rate, 4),
        "t_stat": round(t_stat, 4),
        "ic_positive_ratio": round(float((ic_clean > 0).mean()), 4),
        "ic_series": ic_clean,
    }


def calc_cumulative_ic(ic_series: pd.Series) -> pd.Series:
    """累计 IC 序列（用于可视化）。"""
    return ic_series.dropna().cumsum()

def calc_decay_analysis(
    df: pd.DataFrame, factor_col: str, label_col: str,
    horizons: list = None, method: str = "spearman",
) -> pd.DataFrame:
    """因子衰减分析：计算不同预测期 N 天后的 IC。

    对于 5 日 forward return 标签（如 Ref($close,-5)/Ref($open,-1)-1），
    horizon=N 时计算 IC(factor_t, label_{t+N})，即预测 t+N 起的未来 5 日收益。

    为避免大数据集上重复复制 DataFrame，采用单次预处理 + 多列 shift。

    Returns:
        DataFrame with columns: horizon, ic_mean, icir, t_stat, n_days
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 40, 60]

    # 单次预处理：为每个 horizon 创建 shift 后的 label 列
    cols = {factor_col, label_col, "instrument", "datetime"}
    df_w = df[list(cols)].copy()
    df_w = df_w.sort_values(["instrument", "datetime"])

    shift_cols = {}
    max_shift = max(h for h in horizons if h > 1)
    if max_shift > 1:
        for n in horizons:
            if n > 1:
                col_name = f"_label_s{n}"
                df_w[col_name] = df_w.groupby("instrument")[label_col].transform(
                    lambda x: x.shift(-(n - 1))
                )
                shift_cols[n] = col_name

    rows = []
    for n in horizons:
        if n <= 1:
            ic = calc_daily_ic(df_w, factor_col, label_col, method)
        else:
            ic = calc_daily_ic(df_w, factor_col, shift_cols[n], method)
        stats = calc_ic_stats(ic)
        rows.append({
            "horizon": n,
            "ic_mean": stats["ic_mean"],
            "icir": stats["icir"],
            "t_stat": stats["t_stat"],
            "n_days": len(ic.dropna()),
        })
    return pd.DataFrame(rows)


def calc_ic_bootstrap_ci(
    ic_series: pd.Series, n_bootstrap: int = 1000, ci: float = 0.95,
) -> dict:
    """Bootstrap 计算 IC 均值的置信区间。

    非参数方法，不假设 IC 服从正态分布。
    Dimensional 视角：比 t-test 更鲁棒的显著性检验。
    """
    import numpy as np
    ic_clean = ic_series.dropna().values
    if len(ic_clean) < 10:
        return {"ci_lower": 0.0, "ci_upper": 0.0, "bootstrap_mean": 0.0}

    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(ic_clean, size=len(ic_clean), replace=True)
        means.append(float(np.mean(sample)))

    alpha = 1 - ci
    lower = float(np.percentile(means, alpha / 2 * 100))
    upper = float(np.percentile(means, (1 - alpha / 2) * 100))

    # 额外输出：IC 大于 0 的 Bootstrap 比例（P-value 备选）
    pct_positive = float(np.mean([m > 0 for m in means]))

    return {
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "bootstrap_mean": round(float(np.mean(means)), 6),
        "bootstrap_std": round(float(np.std(means)), 6),
        "pct_positive": round(pct_positive, 4),
        "n_bootstrap": n_bootstrap,
    }



# ──────────── Fama-MacBeth 回归 ────────────

def calc_fama_macbeth(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    control_cols: list = None,
) -> dict:
    """Fama-MacBeth 两步回归。

    Step 1 (截面): 每日对 factor_col 与 label_col 做 OLS 回归
    Step 2 (时序): 对回归系数序列做 Newey-West t 检验

    Returns:
        dict with keys: gamma_mean, gamma_std, t_stat, p_value, n_days
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    gammas = []
    for dt, grp in df.groupby("datetime"):
        y = grp[label_col].values
        x = grp[factor_col].values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if mask.sum() < 10:
            continue
        x_c = np.column_stack([np.ones(mask.sum()), x[mask]])
        y_c = y[mask]
        try:
            beta = np.linalg.lstsq(x_c, y_c, rcond=None)[0]
            gammas.append(beta[1])
        except Exception:
            continue

    if len(gammas) < 10:
        return {"gamma_mean": 0.0, "gamma_std": 0.0, "t_stat": 0.0, "p_value": 1.0, "n_days": len(gammas)}

    gamma_arr = np.array(gammas)
    gamma_mean = float(np.mean(gamma_arr))
    gamma_std = float(np.std(gamma_arr, ddof=1))
    t_stat = gamma_mean / (gamma_std / np.sqrt(len(gamma_arr))) if gamma_std > 1e-12 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(gamma_arr)-1))

    return {
        "gamma_mean": round(gamma_mean, 6),
        "gamma_std": round(gamma_std, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(float(p_value), 6),
        "n_days": len(gammas),
    }


# ──────────── Newey-West 标准误调整 ────────────

def calc_newey_west_tstat(
    series: pd.Series,
    lags: int = None,
) -> dict:
    """Newey-West HAC 标准误 t 统计量。

    自动选择滞后阶数（若未指定）：lags = int(4 * (n/100)**(2/9))
    """
    import numpy as np

    s = series.dropna().values
    n = len(s)
    if n < 10:
        return {"mean": 0.0, "nw_std": 0.0, "nw_tstat": 0.0, "nw_pvalue": 1.0}

    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    lags = max(1, min(lags, n // 3))

    mean = float(np.mean(s))
    resid = s - mean

    # 计算 OLS 方差
    var_ols = np.var(resid, ddof=1) / n

    # Newey-West 方差修正
    gamma0 = np.mean(resid ** 2)
    nw_var = gamma0
    for j in range(1, lags + 1):
        gamma_j = np.mean(resid[j:] * resid[:-j]) if j < n else 0
        w = 1 - j / (lags + 1)  # Bartlett kernel
        nw_var += 2 * w * gamma_j
    nw_var = max(nw_var / n, 1e-20)

    nw_std = float(np.sqrt(nw_var))
    nw_tstat = mean / nw_std if nw_std > 1e-12 else 0.0

    from scipy import stats
    nw_pvalue = 2 * (1 - stats.t.cdf(abs(nw_tstat), df=n-1))

    return {
        "mean": round(mean, 6),
        "nw_std": round(nw_std, 6),
        "nw_tstat": round(nw_tstat, 4),
        "nw_pvalue": round(float(nw_pvalue), 6),
        "nw_lags": lags,
        "n_obs": n,
    }


# ──────────── IC 自相关修正（Lo's Adjusted Sharpe） ────────────

def calc_lo_adjusted_sharpe(
    returns: pd.Series,
    annual_factor: float = 252.0,
    q: int = None,
) -> dict:
    """Lo (2002) 修正夏普比率的置信区间。

    考虑时间序列自相关对夏普比率标准误的向下偏误。
    """
    import numpy as np

    r = returns.dropna().values
    n = len(r)
    if n < 10:
        return {"sharpe": 0.0, "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1))
    sharpe = mean_r / std_r * np.sqrt(annual_factor) if std_r > 1e-12 else 0.0

    # 自相关修正的标准误
    if q is None:
        q = int(np.floor(n ** 0.25))
    q = max(1, min(q, n // 4))

    autocov = np.array([
        np.mean((r[t:] - mean_r) * (r[:-t] - mean_r)) if t > 0 else np.var(r, ddof=0)
        for t in range(q + 1)
    ])
    # 计算 Vq 修正因子
    vq = autocov[0] + 2 * np.sum((1 - np.arange(1, q+1) / (q+1)) * autocov[1:])
    vq = vq / autocov[0] if autocov[0] > 1e-12 else 1.0

    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2 / annual_factor) * vq / n)

    return {
        "sharpe": round(sharpe, 4),
        "se": round(float(se_sharpe), 4),
        "q_lags": q,
        "vq": round(float(vq), 4),
        "n_obs": n,
    }
