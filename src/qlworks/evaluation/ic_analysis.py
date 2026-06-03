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

