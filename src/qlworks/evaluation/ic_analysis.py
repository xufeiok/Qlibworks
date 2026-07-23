"""
IC 分析：信息系数、ICIR、胜率。
"""

from typing import Optional
import numpy as np
import pandas as pd

# ── 默认参数（可统一修改） ──
_IC_METHOD: str = "spearman"                 # IC 计算方法: spearman / pearson
_IC_ANNUAL_FACTOR: float = 252.0             # 日频年化因子
_IC_MIN_SAMPLES: int = 5                     # 最小有效样本数
_IC_NW_MIN_SAMPLES: int = 10                 # Newey-West 最小样本数
_IC_MAX_LAG: int = 120                       # IC 半衰期最大滞后天数
_IC_ROLLING_WINDOW: int = 252                # 滚动 IC 窗口（252 = 1 年）
_IC_DECAY_HORIZONS: list = [1, 5, 10, 20, 40, 60]  # 衰减分析预测期列表
_IC_BOOTSTRAP_N: int = 1000                  # Bootstrap 抽样次数
_IC_BOOTSTRAP_CI: float = 0.95               # Bootstrap 置信区间
_IC_NW_LAGS_DIVISOR_FACTOR: int = 4          # Newey-West 滞后公式系数
_IC_NW_LAGS_EXPONENT: float = 2 / 9          # Newey-West 滞后公式指数
_IC_NW_MAX_LAGS_DIVISOR: int = 3             # NW 最大滞后 = n // N
_IC_FM_MIN_SAMPLES: int = 10                 # Fama-MacBeth 最小样本数


def calc_daily_ic(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    method: str = _IC_METHOD,
    min_obs: int = 10,
) -> pd.Series:
    """每日截面 IC（向量化加速版）。

    对 Spearman 方法：先 rank 再计算 Pearson 相关，等效但比逐日 apply 快 3-5 倍。
    对 Pearson 方法：直接 groupby corr。

    当某天有效样本 < min_obs 时，该天 IC 置为 NaN（避免 1-2 只股票算出虚假相关）。
    """
    corr_method = "spearman" if method == "spearman" else "pearson"
    tmp = df[[factor_col, label_col]].copy()
    tmp["_dt"] = df["datetime"].values

    # 过滤样本过少的日期，避免 groupby corr 返回标量 NaN 导致索引结构变化
    obs_count = tmp.groupby("_dt").size()
    valid_dates = obs_count[obs_count >= min_obs].index
    tmp = tmp[tmp["_dt"].isin(valid_dates)]

    if tmp.empty:
        ic = pd.Series(dtype=float, name="ic")
        ic.index.name = "datetime"
        return ic

    if corr_method == "spearman":
        # Spearman = rank 后的 Pearson 相关
        tmp[[factor_col, label_col]] = tmp.groupby("_dt")[[factor_col, label_col]].rank()

    corr_mats = tmp.groupby("_dt")[[factor_col, label_col]].corr(method="pearson")
    try:
        ic = corr_mats.loc[pd.IndexSlice[:, factor_col], label_col].droplevel(1)
    except (AssertionError, KeyError):
        # 兜底：少数日期 groupby corr 后索引结构异常，逐日计算
        ic_vals = {}
        for dt, g in tmp.groupby("_dt"):
            if len(g) >= min_obs:
                cr = g[[factor_col, label_col]].corr(method=corr_method)
                if factor_col in cr.index and label_col in cr.columns:
                    ic_vals[dt] = cr.loc[factor_col, label_col]
        ic = pd.Series(ic_vals, name="ic")
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


def calc_ic_stats(ic_series: pd.Series, annual_factor: float = _IC_ANNUAL_FACTOR) -> dict:
    """计算 IC 统计量（含 Newey-West 自相关修正 ICIR）。

    5 日重叠标签导致 IC 序列存在强自相关，原始 ICIR (ic_mean/ic_std) 会被高估。
    Newey-West HAC 修正通过 Bartlett 核估计自相关稳健标准误，得到更保守的 ICIR。
    """
    ic_clean = ic_series.dropna()
    if len(ic_clean) < _IC_MIN_SAMPLES:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "win_rate": 0.5, "t_stat": 0.0}

    ic_mean = float(ic_clean.mean())
    ic_std = float(ic_clean.std())
    icir = ic_mean / ic_std * np.sqrt(annual_factor) if ic_std > 1e-12 else 0.0
    win_rate = (ic_clean > 0).sum() / len(ic_clean) if ic_mean >= 0 else (ic_clean < 0).sum() / len(ic_clean)

    from scipy import stats
    t_stat, _ = stats.ttest_1samp(ic_clean.dropna(), 0)
    t_stat = float(t_stat)

    # Newey-West 自相关修正 ICIR
    nw_result = calc_newey_west_tstat(ic_clean) if len(ic_clean) >= _IC_NW_MIN_SAMPLES else {"nw_std": ic_std, "n_obs": 0}
    nw_se = nw_result.get("nw_std", ic_std)
    n_obs = nw_result.get("n_obs", len(ic_clean))
    # nw_std 返回的是标准误(SE of mean)，需转换回序列标准差
    # NW 序列标准差 = NW 标准误 × √n
    nw_std_series = nw_se * np.sqrt(n_obs) if n_obs > 0 else ic_std
    icir_nw = ic_mean / nw_std_series * np.sqrt(annual_factor) if nw_std_series > 1e-12 else 0.0

    return {
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "icir": round(icir, 4),
        "icir_nw": round(icir_nw, 4),  # Newey-West 修正 ICIR，更保守准确
        "win_rate": round(win_rate, 4),
        "t_stat": round(t_stat, 4),
        "ic_positive_ratio": round(float((ic_clean > 0).mean()), 4),
        "ic_series": ic_clean,
    }


def calc_cumulative_ic(ic_series: pd.Series) -> pd.Series:
    """累计 IC 序列（用于可视化）。"""
    return ic_series.dropna().cumsum()


# ──────────── IC 半衰期 ────────────

def calc_ic_half_life(ic_series: pd.Series, max_lag: int = _IC_MAX_LAG) -> dict:
    """因子 IC 半衰期：IC 自相关衰减到 50% 所需的滞后天数。

    计算方法：
      1. 计算 IC 序列的滞后自相关系数（lag 1 到 max_lag）
      2. 找到自相关首次降低到 50% 以下的滞后阶数
      3. 如果找不到，用指数衰减模型外推

    半衰期越短 → 因子预测力衰减越快 → 需要高频调仓
    半衰期越长 → 因子预测力持续越久 → 适合低频换仓

    Args:
        ic_series: 日度 IC 序列
        max_lag: 最大考察滞后天数（默认 120 个交易日）

    Returns:
        dict: half_life_days（半衰期天数）, decay_rate（衰减率）,
              acf_1（lag-1 自相关）, effective_max_lag（有效最大滞后）
    """
    ic = ic_series.dropna()
    if len(ic) < 20:
        return {"half_life_days": None, "decay_rate": None, "acf_1": None,
                "note": "样本不足"}

    # 用 pandas 自相关（最多到 max_lag）
    acf = [ic.autocorr(lag=l) for l in range(1, min(max_lag, len(ic) // 4) + 1)]
    if not acf or acf[0] is None:
        return {"half_life_days": None, "decay_rate": None, "acf_1": None,
                "note": "自相关计算失败"}

    acf_1 = float(acf[0])
    half_life = None

    # 方法 1：直接查找自相关首次低于 0.5 的 lag
    for i, v in enumerate(acf):
        if v < 0.5 or v < 0:
            half_life = i + 1
            break

    # 方法 2：如果找不到（所有 lag 自相关都 > 0.5），用指数衰减模型外推
    if half_life is None and acf_1 > 0:
        # 假设指数衰减: acf(lag) = acf_1 ^ lag
        # 求解 acf_1 ^ half_life = 0.5
        import math
        half_life = math.log(0.5) / math.log(acf_1) if acf_1 > 0 and acf_1 != 1 else max_lag
        half_life = int(round(half_life))
    elif half_life is None:
        half_life = max_lag

    decay_rate = float(acf_1) if acf_1 is not None else None

    return {
        "half_life_days": int(half_life) if half_life else None,
        "decay_rate": round(decay_rate, 4) if decay_rate else None,
        "acf_1": round(acf_1, 4) if acf_1 is not None else None,
        "effective_max_lag": len(acf),
        "n_obs": len(ic),
    }


# ──────────── 滚动 IC 稳定性 ────────────

def calc_rolling_ic_stability(
    ic_series: pd.Series,
    window: int = _IC_ROLLING_WINDOW,
    annual_factor: float = _IC_ANNUAL_FACTOR,
) -> dict:
    """滚动 IC 稳定性分析。

    计算滚动窗口内的 IC 均值和 ICIR 的时序稳定性。
    稳定性指标：
      - rolling_icir_mean: 滚动 ICIR 的均值（越高越好）
      - rolling_icir_std: 滚动 ICIR 的标准差（越低越稳定）
      - rolling_icir_stability: rolling_icir_mean / rolling_icir_std
      - ic_rolling_volatility: 滚动 IC 标准差的时间序列均值（越低越稳定）

    Args:
        ic_series: 日度 IC 序列
        window: 滚动窗口大小（默认 252 = 1 年）
        annual_factor: 年化因子

    Returns:
        dict: 各类稳定性指标
    """
    ic = ic_series.dropna()
    if len(ic) < window + 20:
        return {"rolling_icir_mean": None, "rolling_icir_std": None,
                "rolling_icir_stability": None, "ic_rolling_vol": None,
                "note": f"样本不足（{len(ic)} < {window}+20）"}

    # 滚动 IC 均值
    rolling_mean = ic.rolling(window, min_periods=window // 2).mean()
    rolling_std = ic.rolling(window, min_periods=window // 2).std()

    # 滚动 ICIR
    rolling_icir = rolling_mean / rolling_std * np.sqrt(annual_factor)

    # 稳定性指标
    valid = rolling_icir.dropna()
    if len(valid) < 10:
        return {"rolling_icir_mean": None, "rolling_icir_std": None,
                "rolling_icir_stability": None, "ic_rolling_vol": None,
                "note": "有效滚动窗口不足"}

    rolling_icir_mean = float(valid.mean())
    rolling_icir_std = float(valid.std())
    stability = rolling_icir_mean / rolling_icir_std if rolling_icir_std > 1e-12 else 0.0

    # IC 波动率稳定性
    ic_rolling_vol = float(rolling_std.dropna().mean())

    return {
        "rolling_icir_mean": round(rolling_icir_mean, 4),
        "rolling_icir_std": round(rolling_icir_std, 4),
        "rolling_icir_stability": round(stability, 4),
        "ic_rolling_vol": round(ic_rolling_vol, 6),
        "window_days": window,
        "n_windows": len(valid),
    }

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
        horizons = _IC_DECAY_HORIZONS

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
    ic_series: pd.Series, n_bootstrap: int = _IC_BOOTSTRAP_N, ci: float = _IC_BOOTSTRAP_CI,
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
        if mask.sum() < _IC_FM_MIN_SAMPLES:
            continue
        x_c = np.column_stack([np.ones(mask.sum()), x[mask]])
        y_c = y[mask]
        try:
            beta = np.linalg.lstsq(x_c, y_c, rcond=None)[0]
            gammas.append(beta[1])
        except Exception:
            continue

    if len(gammas) < _IC_FM_MIN_SAMPLES:
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
        lags = int(_IC_NW_LAGS_DIVISOR_FACTOR * (n / 100) ** _IC_NW_LAGS_EXPONENT)
    lags = max(1, min(lags, n // _IC_NW_MAX_LAGS_DIVISOR))

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
    annual_factor: float = _IC_ANNUAL_FACTOR,
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
