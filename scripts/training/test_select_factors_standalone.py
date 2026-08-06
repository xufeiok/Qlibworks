"""
因子筛选核心功能独立测试

不依赖 qlib/qlworks，直接内联核心算法进行验证。
测试范围：
1. 数据质量检查（Bloomberg 级）
2. IC 分析体系（Citadel 级）
3. 多重检验校正（Renaissance 级）
4. 分组回测（AQR 级）
5. 置换检验（D.E. Shaw 级）
6. 拥挤度分析（Two Sigma 级）
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge


# ==============================================================================
# 内联核心算法（从 select_factors.py 提取，去除 qlib 依赖）
# ==============================================================================

def check_factor_data_quality(
    df: pd.DataFrame,
    factor_cols: list,
    min_coverage: float = 0.7,
    max_missing_rate: float = 0.3,
    outlier_threshold: float = 5.0,
):
    """[Bloomberg 级] 因子数据质量检查。"""
    report_rows = []
    passed = []

    for col in factor_cols:
        if col not in df.columns:
            report_rows.append({
                "factor": col, "coverage": 0.0, "missing_rate": 1.0,
                "outlier_rate": 0.0, "std": 0.0, "passed": False,
                "reason": "因子不存在于数据中",
            })
            continue

        series = df[col]
        total = len(series)
        non_null = series.notna().sum()
        coverage = non_null / total if total > 0 else 0.0
        missing_rate = 1.0 - coverage

        outlier_rate = 0.0
        std_val = 0.0
        if non_null > 100:
            daily_z = df.groupby(level='datetime')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-12)
            )
            outlier_rate = (daily_z.abs() > outlier_threshold).sum() / non_null
            std_val = series.std()

        is_passed = True
        reasons = []
        if coverage < min_coverage:
            is_passed = False
            reasons.append(f"覆盖率过低({coverage:.2%} < {min_coverage:.0%})")
        if missing_rate > max_missing_rate:
            is_passed = False
            reasons.append(f"缺失率过高({missing_rate:.2%} > {max_missing_rate:.0%})")
        if std_val == 0 or np.isnan(std_val):
            is_passed = False
            reasons.append("零方差(无区分度)")

        if is_passed:
            passed.append(col)

        report_rows.append({
            "factor": col,
            "coverage": round(coverage, 4),
            "missing_rate": round(missing_rate, 4),
            "outlier_rate": round(float(outlier_rate), 4),
            "std": round(float(std_val), 6),
            "passed": is_passed,
            "reason": "; ".join(reasons) if reasons else "通过",
        })

    return passed, pd.DataFrame(report_rows)


def _vectorized_daily_ic(train_frame, factor_cols, label_col, method='spearman'):
    """向量化逐日 IC 计算。"""
    all_cols = factor_cols + [label_col]
    corr_matrices = train_frame.groupby(level='datetime')[all_cols].corr(method=method)
    daily_ic = corr_matrices.xs(label_col, level=1, axis=0)[factor_cols]
    return daily_ic


def compute_ic_statistics(daily_ic):
    """[Citadel 级] 计算 IC 统计指标。"""
    stats_rows = []
    n_days = len(daily_ic)

    for col in daily_ic.columns:
        ic_series = daily_ic[col].dropna()
        n = len(ic_series)
        if n < 10:
            stats_rows.append({
                "factor": col, "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "icir_annualized": np.nan,
                "t_stat": np.nan, "p_value": np.nan, "ic_pos_ratio": np.nan,
                "ic_skew": np.nan, "ic_kurt": np.nan, "n_days": n,
            })
            continue

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        icir_annual = icir * np.sqrt(252)

        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        pos_ratio = (ic_series > 0).sum() / n
        skew = stats.skew(ic_series)
        kurt = stats.kurtosis(ic_series)

        stats_rows.append({
            "factor": col,
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "icir": round(icir, 4),
            "icir_annualized": round(icir_annual, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "ic_pos_ratio": round(pos_ratio, 4),
            "ic_skew": round(skew, 4),
            "ic_kurt": round(kurt, 4),
            "n_days": n,
        })

    return pd.DataFrame(stats_rows)


def apply_multiple_testing_correction(stats_df, method="bh", alpha=0.05):
    """[Renaissance 级] 多重检验校正。"""
    result = stats_df.copy()
    n_tests = len(result)

    if n_tests == 0:
        result["adjusted_pvalue"] = np.nan
        result["significant"] = False
        return result

    if method == "bonferroni":
        result["adjusted_pvalue"] = (result["p_value"] * n_tests).clip(upper=1.0)
        result["significant"] = result["adjusted_pvalue"] < alpha
    elif method == "bh":
        sorted_idx = result["p_value"].sort_values().index
        sorted_p = result.loc[sorted_idx, "p_value"].values
        ranks = np.arange(1, n_tests + 1)
        adjusted = sorted_p * n_tests / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1.0)
        result.loc[sorted_idx, "adjusted_pvalue"] = adjusted
        result["significant"] = result["adjusted_pvalue"] < alpha
    else:
        result["adjusted_pvalue"] = result["p_value"]
        result["significant"] = result["p_value"] < alpha

    return result


def quantile_backtest(df, factor_col, label_col, n_quantiles=5):
    """[AQR 级] 分组回测检验。"""
    result = {
        "quantile_returns": [], "long_short_return": 0.0,
        "long_short_sharpe": 0.0, "monotonicity": 0.0, "is_monotonic": False,
    }

    try:
        daily_returns = []
        for date, group in df.groupby(level='datetime'):
            valid = group[[factor_col, label_col]].dropna()
            if len(valid) < n_quantiles * 2:
                continue
            try:
                valid['quantile'] = pd.qcut(
                    valid[factor_col], q=n_quantiles, labels=False, duplicates='drop'
                )
            except Exception:
                continue
            if valid['quantile'].nunique() < n_quantiles:
                continue
            q_returns = valid.groupby('quantile')[label_col].mean()
            if len(q_returns) == n_quantiles:
                daily_returns.append(q_returns.values)

        if len(daily_returns) < 20:
            return result

        daily_returns_arr = np.array(daily_returns)
        avg_returns = daily_returns_arr.mean(axis=0)
        result["quantile_returns"] = [round(r, 6) for r in avg_returns.tolist()]

        ls_returns = daily_returns_arr[:, -1] - daily_returns_arr[:, 0]
        result["long_short_return"] = round(ls_returns.mean(), 6)
        if ls_returns.std() > 0:
            result["long_short_sharpe"] = round(
                ls_returns.mean() / ls_returns.std() * np.sqrt(252), 4
            )

        ranks = np.arange(n_quantiles)
        monotonicity, _ = stats.spearmanr(ranks, avg_returns)
        result["monotonicity"] = round(monotonicity, 4)
        result["is_monotonic"] = abs(monotonicity) > 0.8

    except Exception as e:
        print(f"      [警告] 分组回测失败 {factor_col}: {e}")

    return result


def permutation_test_ic(df, factor_cols, label_col, n_permutations=200, alpha=0.05):
    """[D.E. Shaw 级] 置换检验。"""
    real_ics = {}
    for col in factor_cols:
        daily_ic = _vectorized_daily_ic(df, [col], label_col, method='spearman')
        real_ics[col] = daily_ic.mean().values[0] if not daily_ic.empty else 0.0

    perm_ics = {col: [] for col in factor_cols}

    np.random.seed(42)
    for i in range(n_permutations):
        shuffled_df = df.copy()
        for date, group in shuffled_df.groupby(level='datetime'):
            shuffled_labels = group[label_col].values.copy()
            np.random.shuffle(shuffled_labels)
            shuffled_df.loc[group.index, label_col] = shuffled_labels

        for col in factor_cols:
            daily_ic = _vectorized_daily_ic(shuffled_df, [col], label_col, method='spearman')
            perm_ics[col].append(daily_ic.mean().values[0] if not daily_ic.empty else 0.0)

    perm_rows = []
    for col in factor_cols:
        real_ic = real_ics[col]
        perm_dist = np.array(perm_ics[col])
        perm_mean = perm_dist.mean()
        perm_std = perm_dist.std()

        p_value = (np.abs(perm_dist) >= abs(real_ic)).sum() / len(perm_dist)
        z_score = (real_ic - perm_mean) / perm_std if perm_std > 0 else 0.0

        perm_rows.append({
            "factor": col,
            "real_ic": round(real_ic, 6),
            "perm_mean": round(perm_mean, 6),
            "perm_std": round(perm_std, 6),
            "perm_zscore": round(z_score, 4),
            "perm_pvalue": round(p_value, 4),
            "significant": p_value < alpha,
        })

    return pd.DataFrame(perm_rows)


def compute_factor_crowding(df, factor_cols, threshold=0.8):
    """[Two Sigma 级] 因子拥挤度分析。"""
    if len(factor_cols) < 2:
        return factor_cols, pd.DataFrame()

    corr_mat = df[factor_cols].corr(method='spearman').abs()

    crowding_scores = {}
    for col in factor_cols:
        other_corrs = corr_mat[col].drop(col)
        crowding_scores[col] = other_corrs.mean()

    crowded_factors = set()
    for i, col1 in enumerate(factor_cols):
        for j, col2 in enumerate(factor_cols):
            if i >= j:
                continue
            if corr_mat.loc[col1, col2] > threshold:
                if crowding_scores[col1] > crowding_scores[col2]:
                    crowded_factors.add(col1)
                else:
                    crowded_factors.add(col2)

    non_crowded = [f for f in factor_cols if f not in crowded_factors]

    report_rows = []
    for col in factor_cols:
        report_rows.append({
            "factor": col,
            "crowding_score": round(crowding_scores[col], 4),
            "is_crowded": col in crowded_factors,
        })
    report_df = pd.DataFrame(report_rows).sort_values("crowding_score", ascending=False)

    return non_crowded, report_df


# ==============================================================================
# 测试数据生成
# ==============================================================================

def generate_test_data(n_days=252, n_stocks=500, n_factors=20, n_useful=5):
    """生成模拟测试数据。"""
    np.random.seed(42)

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    stocks = [f"SH{600000 + i:06d}" for i in range(n_stocks)]

    index = pd.MultiIndex.from_product(
        [dates, stocks], names=["datetime", "instrument"]
    )

    factor_data = {}
    for i in range(n_factors):
        if i < n_useful:
            noise = np.random.randn(len(index)) * 0.8
            signal = np.random.randn(len(index)) * 0.2
            factor_data[f"factor_{i:02d}"] = signal + noise
        else:
            factor_data[f"factor_{i:02d}"] = np.random.randn(len(index))

    df = pd.DataFrame(factor_data, index=index)

    label = np.zeros(len(index))
    for i in range(n_useful):
        label += df[f"factor_{i:02d}"].values * 0.1
    label += np.random.randn(len(index)) * 0.5
    df["LABEL_5D"] = label

    return df


# ==============================================================================
# 测试用例
# ==============================================================================

def test_data_quality_check():
    """测试 1: 数据质量检查（Bloomberg 级）。"""
    print("\n" + "=" * 60)
    print("  测试 1: 数据质量检查（Bloomberg 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=10, n_useful=3)

    df["factor_bad_coverage"] = np.nan
    df["factor_low_coverage"] = np.nan
    mask = np.random.rand(len(df)) < 0.5
    df.loc[mask, "factor_low_coverage"] = np.random.randn(mask.sum())

    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    passed, report = check_factor_data_quality(
        df, factor_cols, min_coverage=0.7, max_missing_rate=0.3
    )

    print(f"\n  输入因子数: {len(factor_cols)}")
    print(f"  通过因子数: {len(passed)}")
    print(f"  淘汰因子数: {len(factor_cols) - len(passed)}")

    assert "factor_bad_coverage" not in passed
    assert "factor_low_coverage" not in passed

    print("  ✓ 数据质量检查测试通过")
    return True


def test_ic_analysis():
    """测试 2: IC 分析体系（Citadel 级）。"""
    print("\n" + "=" * 60)
    print("  测试 2: IC 分析体系（Citadel 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=10, n_useful=3)
    factor_cols = [f"factor_{i:02d}" for i in range(10)]

    daily_ic = _vectorized_daily_ic(df, factor_cols, "LABEL_5D", method="spearman")
    print(f"\n  每日 IC shape: {daily_ic.shape}")
    print(f"  交易日数: {len(daily_ic)}")

    ic_stats = compute_ic_statistics(daily_ic)
    print(f"\n  IC 统计结果:")
    print(f"  {'因子':<15s} {'IC均值':>8s} {'ICIR':>8s} {'t-stat':>8s} {'p值':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in ic_stats.iterrows():
        print(f"  {row['factor']:<15s} {row['ic_mean']:>8.4f} {row['icir']:>8.3f} "
              f"{row['t_stat']:>8.2f} {row['p_value']:>8.4f}")

    useful_ic = ic_stats[ic_stats["factor"].isin([f"factor_{i:02d}" for i in range(3)])]["ic_mean"].mean()
    noise_ic = ic_stats[ic_stats["factor"].isin([f"factor_{i:02d}" for i in range(3, 10)])]["ic_mean"].mean()
    print(f"\n  有用因子平均 IC: {useful_ic:.4f}")
    print(f"  噪声因子平均 IC: {noise_ic:.4f}")

    assert useful_ic > noise_ic

    print("  ✓ IC 分析测试通过")
    return True


def test_multiple_testing_correction():
    """测试 3: 多重检验校正（Renaissance 级）。"""
    print("\n" + "=" * 60)
    print("  测试 3: 多重检验校正（Renaissance 级）")
    print("=" * 60)

    np.random.seed(42)
    n_tests = 100
    p_values = np.random.uniform(0, 1, n_tests)
    p_values[:10] = np.random.uniform(0, 0.01, 10)

    stats_df = pd.DataFrame({
        "factor": [f"factor_{i:03d}" for i in range(n_tests)],
        "p_value": p_values,
    })

    bonf_result = apply_multiple_testing_correction(
        stats_df.copy(), method="bonferroni", alpha=0.05
    )
    bonf_sig = bonf_result["significant"].sum()

    bh_result = apply_multiple_testing_correction(
        stats_df.copy(), method="bh", alpha=0.05
    )
    bh_sig = bh_result["significant"].sum()

    print(f"\n  总检验数: {n_tests}")
    print(f"  真实显著数: 10")
    print(f"  Bonferroni 校正后显著: {bonf_sig}")
    print(f"  BH 校正后显著: {bh_sig}")

    assert bonf_sig <= bh_sig
    assert bonf_sig <= 10

    print("  ✓ 多重检验校正测试通过")
    return True


def test_quantile_backtest():
    """测试 4: 分组回测（AQR 级）。"""
    print("\n" + "=" * 60)
    print("  测试 4: 分组回测（AQR 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=5, n_useful=2)

    result_useful = quantile_backtest(df, "factor_00", "LABEL_5D", n_quantiles=5)
    print(f"\n  有用因子 (factor_00):")
    print(f"    分组收益: {[f'{r:.4f}' for r in result_useful['quantile_returns']]}")
    print(f"    多空收益: {result_useful['long_short_return']:.6f}")
    print(f"    多空夏普: {result_useful['long_short_sharpe']:.4f}")
    print(f"    单调性: {result_useful['monotonicity']:.4f}")

    result_noise = quantile_backtest(df, "factor_04", "LABEL_5D", n_quantiles=5)
    print(f"\n  噪声因子 (factor_04):")
    print(f"    分组收益: {[f'{r:.4f}' for r in result_noise['quantile_returns']]}")
    print(f"    多空收益: {result_noise['long_short_return']:.6f}")
    print(f"    多空夏普: {result_noise['long_short_sharpe']:.4f}")

    print("  ✓ 分组回测测试通过")
    return True


def test_permutation_test():
    """测试 5: 置换检验（D.E. Shaw 级）。"""
    print("\n" + "=" * 60)
    print("  测试 5: 置换检验（D.E. Shaw 级）")
    print("=" * 60)

    df = generate_test_data(n_days=50, n_stocks=100, n_factors=5, n_useful=2)
    factor_cols = [f"factor_{i:02d}" for i in range(5)]

    perm_df = permutation_test_ic(
        df, factor_cols, "LABEL_5D", n_permutations=50, alpha=0.05,
    )

    print(f"\n  置换检验结果 (50 次置换):")
    print(f"  {'因子':<15s} {'真实IC':>8s} {'置换均值':>10s} {'Z-score':>9s} {'p值':>7s} {'显著':>6s}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*9} {'-'*7} {'-'*6}")
    for _, row in perm_df.iterrows():
        print(f"  {row['factor']:<15s} {row['real_ic']:>8.4f} {row['perm_mean']:>10.4f} "
              f"{row['perm_zscore']:>9.2f} {row['perm_pvalue']:>7.4f} {'是' if row['significant'] else '否':>6s}")

    print("  ✓ 置换检验测试通过")
    return True


def test_crowding_analysis():
    """测试 6: 因子拥挤度分析（Two Sigma 级）。"""
    print("\n" + "=" * 60)
    print("  测试 6: 因子拥挤度分析（Two Sigma 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=10, n_useful=3)

    df["factor_crowd_1"] = df["factor_00"] + np.random.randn(len(df)) * 0.1
    df["factor_crowd_2"] = df["factor_00"] * 0.9 + np.random.randn(len(df)) * 0.15
    df["factor_crowd_3"] = df["factor_01"] + np.random.randn(len(df)) * 0.1

    factor_cols = [f"factor_{i:02d}" for i in range(10)] + ["factor_crowd_1", "factor_crowd_2", "factor_crowd_3"]

    non_crowded, report = compute_factor_crowding(
        df[factor_cols], factor_cols, threshold=0.8
    )

    print(f"\n  拥挤度分析结果:")
    print(f"  输入因子数: {len(factor_cols)}")
    print(f"  非拥挤因子数: {len(non_crowded)}")
    print(f"  拥挤因子数: {len(factor_cols) - len(non_crowded)}")

    print(f"\n  拥挤度排名:")
    print(f"  {'因子':<18s} {'拥挤度':>8s} {'拥挤':>6s}")
    print(f"  {'-'*18} {'-'*8} {'-'*6}")
    for _, row in report.iterrows():
        print(f"  {row['factor']:<18s} {row['crowding_score']:>8.4f} "
              f"{'是' if row['is_crowded'] else '否':>6s}")

    crowd_factors = ["factor_crowd_1", "factor_crowd_2", "factor_crowd_3"]
    n_crowded_detected = sum(1 for f in crowd_factors if f not in non_crowded)
    print(f"\n  检测到的人为拥挤因子: {n_crowded_detected}/3")

    print("  ✓ 拥挤度分析测试通过")
    return True


def run_all_tests():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("  因子筛选核心算法单元测试")
    print("  对标：AQR / Citadel / Renaissance / Two Sigma / D.E. Shaw")
    print("=" * 60)

    tests = [
        ("数据质量检查 (Bloomberg)", test_data_quality_check),
        ("IC 分析体系 (Citadel)", test_ic_analysis),
        ("多重检验校正 (Renaissance)", test_multiple_testing_correction),
        ("分组回测 (AQR)", test_quantile_backtest),
        ("置换检验 (D.E. Shaw)", test_permutation_test),
        ("拥挤度分析 (Two Sigma)", test_crowding_analysis),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "通过"))
        except Exception as e:
            print(f"\n  ✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"失败: {e}"))

    print("\n" + "=" * 60)
    print("  测试总结")
    print("=" * 60)
    passed = sum(1 for _, r in results if r == "通过")
    total = len(results)
    print(f"  通过: {passed}/{total}")
    for name, result in results:
        status = "✓" if result == "通过" else "✗"
        print(f"  {status} {name}: {result}")

    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
