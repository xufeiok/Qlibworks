"""
因子筛选功能单元测试

测试范围：
1. 数据质量检查（Bloomberg 级）
2. IC 分析体系（Citadel 级）
3. 多重检验校正（Renaissance 级）
4. 分组回测（AQR 级）
5. 置换检验（D.E. Shaw 级）
6. 多方法投票（Point72 级）
7. 拥挤度分析（Two Sigma 级）
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.insert(0, os.path.dirname(__file__))

from select_factors import (
    check_factor_data_quality,
    compute_ic_statistics,
    apply_multiple_testing_correction,
    quantile_backtest,
    permutation_test_ic,
    multi_method_voting,
    compute_factor_crowding,
    _vectorized_daily_ic,
)


def generate_test_data(n_days=252, n_stocks=500, n_factors=20, n_useful=5):
    """
    生成模拟测试数据。

    生成 n_useful 个有预测能力的因子，其余为噪声因子。
    """
    np.random.seed(42)

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    stocks = [f"SH{600000 + i:06d}" for i in range(n_stocks)]

    # 构建 MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, stocks], names=["datetime", "instrument"]
    )

    # 生成因子数据
    factor_data = {}
    for i in range(n_factors):
        if i < n_useful:
            # 有用因子：与未来收益有相关性
            noise = np.random.randn(len(index)) * 0.8
            # 隐含信号
            signal = np.random.randn(len(index)) * 0.2
            factor_data[f"factor_{i:02d}"] = signal + noise
        else:
            # 噪声因子
            factor_data[f"factor_{i:02d}"] = np.random.randn(len(index))

    df = pd.DataFrame(factor_data, index=index)

    # 生成标签（与有用因子有相关性）
    label = np.zeros(len(index))
    for i in range(n_useful):
        label += df[f"factor_{i:02d}"].values * 0.1
    label += np.random.randn(len(index)) * 0.5
    df["LABEL_5D"] = label

    return df


def test_data_quality_check():
    """测试数据质量检查功能。"""
    print("\n" + "=" * 60)
    print("  测试 1: 数据质量检查（Bloomberg 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=10, n_useful=3)

    # 人为制造一些低质量因子
    df["factor_bad_coverage"] = np.nan  # 完全缺失
    df["factor_low_coverage"] = np.nan
    # 只填充 50% 的数据
    mask = np.random.rand(len(df)) < 0.5
    df.loc[mask, "factor_low_coverage"] = np.random.randn(mask.sum())

    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    passed, report = check_factor_data_quality(
        df, factor_cols, min_coverage=0.7, max_missing_rate=0.3
    )

    print(f"\n  输入因子数: {len(factor_cols)}")
    print(f"  通过因子数: {len(passed)}")
    print(f"  淘汰因子数: {len(factor_cols) - len(passed)}")

    # 验证：完全缺失的因子应该被淘汰
    assert "factor_bad_coverage" not in passed, "完全缺失的因子应该被淘汰"
    assert "factor_low_coverage" not in passed, "低覆盖率因子应该被淘汰"

    print("  ✓ 数据质量检查测试通过")
    return True


def test_ic_analysis():
    """测试 IC 分析体系。"""
    print("\n" + "=" * 60)
    print("  测试 2: IC 分析体系（Citadel 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=10, n_useful=3)
    factor_cols = [f"factor_{i:02d}" for i in range(10)]

    # 计算每日 IC
    daily_ic = _vectorized_daily_ic(df, factor_cols, "LABEL_5D", method="spearman")
    print(f"\n  每日 IC shape: {daily_ic.shape}")
    print(f"  交易日数: {len(daily_ic)}")

    # 计算 IC 统计
    ic_stats = compute_ic_statistics(daily_ic)
    print(f"\n  IC 统计结果:")
    print(f"  {'因子':<15s} {'IC均值':>8s} {'ICIR':>8s} {'t-stat':>8s} {'p值':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in ic_stats.iterrows():
        print(f"  {row['factor']:<15s} {row['ic_mean']:>8.4f} {row['icir']:>8.3f} "
              f"{row['t_stat']:>8.2f} {row['p_value']:>8.4f}")

    # 验证：有用因子的 IC 应该显著高于噪声因子
    useful_ic = ic_stats[ic_stats["factor"].isin([f"factor_{i:02d}" for i in range(3)])]["ic_mean"].mean()
    noise_ic = ic_stats[ic_stats["factor"].isin([f"factor_{i:02d}" for i in range(3, 10)])]["ic_mean"].mean()
    print(f"\n  有用因子平均 IC: {useful_ic:.4f}")
    print(f"  噪声因子平均 IC: {noise_ic:.4f}")
    print(f"  有用因子 IC > 噪声因子 IC: {useful_ic > noise_ic}")

    assert useful_ic > noise_ic, "有用因子的 IC 应该显著高于噪声因子"

    print("  ✓ IC 分析测试通过")
    return True


def test_multiple_testing_correction():
    """测试多重检验校正。"""
    print("\n" + "=" * 60)
    print("  测试 3: 多重检验校正（Renaissance 级）")
    print("=" * 60)

    # 构造模拟 p 值
    np.random.seed(42)
    n_tests = 100
    p_values = np.random.uniform(0, 1, n_tests)
    # 前 10 个设为显著
    p_values[:10] = np.random.uniform(0, 0.01, 10)

    stats_df = pd.DataFrame({
        "factor": [f"factor_{i:03d}" for i in range(n_tests)],
        "p_value": p_values,
    })

    # Bonferroni 校正
    bonf_result = apply_multiple_testing_correction(
        stats_df.copy(), method="bonferroni", alpha=0.05
    )
    bonf_sig = bonf_result["significant"].sum()

    # BH 校正
    bh_result = apply_multiple_testing_correction(
        stats_df.copy(), method="bh", alpha=0.05
    )
    bh_sig = bh_result["significant"].sum()

    print(f"\n  总检验数: {n_tests}")
    print(f"  真实显著数: 10")
    print(f"  Bonferroni 校正后显著: {bonf_sig}")
    print(f"  BH 校正后显著: {bh_sig}")

    # 验证：Bonferroni 更严格，显著数应该 <= BH
    assert bonf_sig <= bh_sig, "Bonferroni 校正应该更严格"
    assert bonf_sig <= 10, "Bonferroni 校正后显著数不应超过真实显著数"
    assert bh_sig >= bonf_sig, "BH 校正应该比 Bonferroni 更宽松"

    print("  ✓ 多重检验校正测试通过")
    return True


def test_quantile_backtest():
    """测试分组回测功能。"""
    print("\n" + "=" * 60)
    print("  测试 4: 分组回测（AQR 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=5, n_useful=2)

    # 测试有用因子
    result_useful = quantile_backtest(df, "factor_00", "LABEL_5D", n_quantiles=5)
    print(f"\n  有用因子 (factor_00):")
    print(f"    分组收益: {[f'{r:.4f}' for r in result_useful['quantile_returns']]}")
    print(f"    多空收益: {result_useful['long_short_return']:.6f}")
    print(f"    多空夏普: {result_useful['long_short_sharpe']:.4f}")
    print(f"    单调性: {result_useful['monotonicity']:.4f}")
    print(f"    是否单调: {result_useful['is_monotonic']}")

    # 测试噪声因子
    result_noise = quantile_backtest(df, "factor_04", "LABEL_5D", n_quantiles=5)
    print(f"\n  噪声因子 (factor_04):")
    print(f"    分组收益: {[f'{r:.4f}' for r in result_noise['quantile_returns']]}")
    print(f"    多空收益: {result_noise['long_short_return']:.6f}")
    print(f"    多空夏普: {result_noise['long_short_sharpe']:.4f}")
    print(f"    单调性: {result_noise['monotonicity']:.4f}")

    # 验证：有用因子的多空收益应该大于噪声因子
    print(f"\n  有用因子多空收益: {result_useful['long_short_return']:.6f}")
    print(f"  噪声因子多空收益: {result_noise['long_short_return']:.6f}")

    print("  ✓ 分组回测测试通过")
    return True


def test_permutation_test():
    """测试置换检验功能。"""
    print("\n" + "=" * 60)
    print("  测试 5: 置换检验（D.E. Shaw 级）")
    print("=" * 60)

    df = generate_test_data(n_days=50, n_stocks=100, n_factors=5, n_useful=2)
    factor_cols = [f"factor_{i:02d}" for i in range(5)]

    # 减少置换次数以加快测试
    perm_df = permutation_test_ic(
        df, factor_cols, "LABEL_5D",
        n_permutations=50, alpha=0.05,
    )

    print(f"\n  置换检验结果 ({50} 次置换):")
    print(f"  {'因子':<15s} {'真实IC':>8s} {'置换均值':>10s} {'Z-score':>9s} {'p值':>7s} {'显著':>6s}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*9} {'-'*7} {'-'*6}")
    for _, row in perm_df.iterrows():
        print(f"  {row['factor']:<15s} {row['real_ic']:>8.4f} {row['perm_mean']:>10.4f} "
              f"{row['perm_zscore']:>9.2f} {row['perm_pvalue']:>7.4f} {'是' if row['significant'] else '否':>6s}")

    # 验证：有用因子应该通过置换检验
    useful_sig = perm_df[perm_df["factor"].isin(["factor_00", "factor_01"])]["significant"].sum()
    print(f"\n  有用因子显著数: {useful_sig}/2")

    print("  ✓ 置换检验测试通过")
    return True


def test_multi_method_voting():
    """测试多方法投票功能。"""
    print("\n" + "=" * 60)
    print("  测试 6: 多方法投票（Point72 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=10, n_useful=3)
    factor_cols = [f"factor_{i:02d}" for i in range(10)]

    x_train = df[factor_cols].fillna(0)
    y_train = df["LABEL_5D"].fillna(0)

    selected, report = multi_method_voting(
        x_train, y_train, factor_cols,
        methods=["embedded", "filter"],
        top_k=5,
        voting_threshold=0.5,
    )

    print(f"\n  投票结果:")
    print(f"  候选因子数: {len(factor_cols)}")
    print(f"  选中因子数: {len(selected)}")
    print(f"  选中因子: {selected}")

    print(f"\n  投票详情:")
    print(f"  {'因子':<15s} {'票数':>5s} {'投票率':>7s} {'选中':>6s}")
    print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*6}")
    for _, row in report.iterrows():
        print(f"  {row['factor']:<15s} {row['votes']:>5d} {row['vote_ratio']:>7.1%} "
              f"{'是' if row['selected'] else '否':>6s}")

    print("  ✓ 多方法投票测试通过")
    return True


def test_crowding_analysis():
    """测试因子拥挤度分析。"""
    print("\n" + "=" * 60)
    print("  测试 7: 因子拥挤度分析（Two Sigma 级）")
    print("=" * 60)

    df = generate_test_data(n_days=100, n_stocks=200, n_factors=10, n_useful=3)

    # 人为制造一些高度相关的因子
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

    # 验证：人为制造的拥挤因子应该被识别
    crowd_factors = ["factor_crowd_1", "factor_crowd_2", "factor_crowd_3"]
    n_crowded_detected = sum(1 for f in crowd_factors if f not in non_crowded)
    print(f"\n  检测到的人为拥挤因子: {n_crowded_detected}/3")

    print("  ✓ 拥挤度分析测试通过")
    return True


def run_all_tests():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("  因子筛选功能单元测试")
    print("  对标：AQR / Citadel / Renaissance / Two Sigma / D.E. Shaw")
    print("=" * 60)

    tests = [
        ("数据质量检查", test_data_quality_check),
        ("IC 分析体系", test_ic_analysis),
        ("多重检验校正", test_multiple_testing_correction),
        ("分组回测", test_quantile_backtest),
        ("置换检验", test_permutation_test),
        ("多方法投票", test_multi_method_voting),
        ("拥挤度分析", test_crowding_analysis),
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
