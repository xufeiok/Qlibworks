"""
train_from_selected.py 新增功能单元测试

测试七层增强体系的各个功能模块：
  第一层：数据增强层 - Bloomberg 级
  第二层：模型训练层 - Point72 级
  第三层：集成增强层 - Citadel 级
  第四层：过拟合防护层 - Renaissance 级
  第五层：风险管理层 - Two Sigma 级
  第六层：后处理增强层 - AQR 级
  第七层：可解释性层 - Dimensional 级
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.insert(0, os.path.dirname(__file__))

from train_from_selected import (
    # 第一层：数据增强层
    compute_time_decay_weights,
    compute_outlier_weights,
    compute_combined_sample_weights,
    # 第二层：模型训练层
    _purge_train_samples,
    _embargo_train_samples,
    multi_seed_predict,
    # 第三层：集成增强层
    compute_multi_dim_weights,
    compute_market_regime,
    compute_regime_adaptive_weights,
    # 第四层：过拟合防护层
    parameter_stability_test,
    # 第五层：风险管理层
    compute_prediction_confidence,
    compute_risk_adjusted_score,
    extreme_event_stress_test,
    # 第六层：后处理增强层
    neutralize_prediction,
    apply_turnover_control,
    compute_dynamic_long_short_ratio,
    # 第七层：可解释性层
    compute_feature_importance_stability,
    extract_feature_importance,
)


def test_time_decay_weights():
    """测试时间衰减样本权重"""
    print("\n[测试1] 时间衰减样本权重")

    # 生成测试数据
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["stock1", "stock2"]], names=["datetime", "instrument"])

    weights = compute_time_decay_weights(idx, half_life_days=50, min_weight=0.3)

    assert len(weights) == len(idx), "权重长度不匹配"
    assert weights.min() >= 0.3, "最小权重低于下限"
    assert abs(weights.mean() - 1.0) < 0.01, "权重均值不为 1"

    # 检查近期权重 > 远期权重
    recent_weight = weights.loc[dates[-1]].mean()
    old_weight = weights.loc[dates[0]].mean()
    assert recent_weight > old_weight, "近期权重应大于远期权重"

    print(f"  ✓ 通过: 权重范围 [{weights.min():.4f}, {weights.max():.4f}], 均值={weights.mean():.4f}")
    print(f"  ✓ 通过: 近期权重={recent_weight:.4f} > 远期权重={old_weight:.4f}")
    return True


def test_outlier_weights():
    """测试异常样本降权"""
    print("\n[测试2] 异常样本降权")

    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    idx = pd.MultiIndex.from_product([dates, [f"stock{i}" for i in range(100)]],
                                     names=["datetime", "instrument"])

    # 生成正态分布标签，加入一些异常值
    labels = pd.Series(np.random.normal(0, 1, len(idx)), index=idx)
    labels.iloc[0:5] = 50.0  # 前 5 个设为异常值（足够大，确保超过阈值）

    # 手动计算第一天的 Z-score 验证
    day1 = labels.iloc[0:100]
    day1_mean = day1.mean()
    day1_std = day1.std()
    day1_zscore = (day1 - day1_mean) / day1_std
    print(f"    调试: 第一天均值={day1_mean:.4f}, 标准差={day1_std:.4f}")
    print(f"    调试: 前5个Z-score={day1_zscore.iloc[0:5].values}")

    weights = compute_outlier_weights(labels, zscore_threshold=3.0, downweight_ratio=0.5)

    assert len(weights) == len(labels), "权重长度不匹配"
    assert weights.max() == 1.0, "正常样本权重应为 1"
    assert weights.min() >= 0.1, "异常样本权重不应低于 0.1"

    # 检查异常样本是否被降权
    outlier_weight = weights.iloc[0:5].mean()
    normal_weight = weights.iloc[100:200].mean()

    print(f"    调试: 异常样本权重={outlier_weight:.4f}, 正常样本权重={normal_weight:.4f}")
    print(f"    调试: 异常样本标签均值={labels.iloc[0:5].mean():.4f}")

    assert outlier_weight < normal_weight, "异常样本权重应低于正常样本"

    print(f"  ✓ 通过: 正常样本权重={normal_weight:.4f}, 异常样本权重={outlier_weight:.4f}")
    return True


def test_purge_embargo():
    """测试 Purged K-Fold 的 purge 和 embargo 机制"""
    print("\n[测试3] Purged K-Fold purge/embargo 机制")

    # 生成测试日期
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["stock1"]], names=["datetime", "instrument"])

    # 测试 purge
    purged_idx = _purge_train_samples(idx, "2020-02-01", "2020-02-10", purge_days=5)
    purged_dates = purged_idx.get_level_values("datetime").unique()

    # 验证集前后 purge_days 天内的样本应该被删除
    valid_start = pd.Timestamp("2020-02-01")
    valid_end = pd.Timestamp("2020-02-10")
    purge_start = valid_start - pd.Timedelta(days=5)
    purge_end = valid_end + pd.Timedelta(days=5)

    for d in purged_dates:
        assert not (purge_start <= d <= purge_end), f"日期 {d} 应该被 purge 删除"

    print(f"  ✓ 通过: purge 后剩余 {len(purged_dates)} 天 (原始 {len(dates)} 天)")

    # 测试 embargo
    embargo_idx = _embargo_train_samples(idx, "2020-02-10", embargo_days=5)
    embargo_dates = embargo_idx.get_level_values("datetime").unique()

    # 验证集之后 embargo_days 天内的样本应该被删除
    embargo_end = valid_end + pd.Timedelta(days=5)
    for d in embargo_dates:
        if d > valid_end:
            assert d > embargo_end, f"日期 {d} 应该在 embargo 期之后"

    print(f"  ✓ 通过: embargo 后剩余 {len(embargo_dates)} 天")
    return True


def test_multi_dim_weights():
    """测试多维加权集成"""
    print("\n[测试4] 多维加权集成")

    # 构造模拟诊断数据
    diagnostics = [
        {"model": "lgb", "valid_ic": 0.05, "valid_icir": 1.5, "ic_decay_half": 10, "long_short_sharpe": 1.2},
        {"model": "xgb", "valid_ic": 0.04, "valid_icir": 1.2, "ic_decay_half": 8, "long_short_sharpe": 1.0},
        {"model": "cat", "valid_ic": 0.03, "valid_icir": 1.0, "ic_decay_half": 6, "long_short_sharpe": 0.8},
    ]

    ensemble_config = {
        "multi_dim_weighting": {
            "enabled": True,
            "weights": {"ic": 0.4, "icir": 0.3, "decay": 0.15, "sharpe": 0.15},
        }
    }

    weights, used_equal = compute_multi_dim_weights(diagnostics, ensemble_config)

    assert len(weights) == 3, "权重数量不匹配"
    assert abs(sum(weights) - 1.0) < 0.001, "权重之和不为 1"
    assert not used_equal, "不应该使用等权"

    # IC 最高的模型权重应该最高
    assert weights[0] > weights[1] > weights[2], "权重应该按 IC 排序"

    print(f"  ✓ 通过: 权重 = {[f'{w:.4f}' for w in weights]}")
    print(f"  ✓ 通过: LGB(IC最高)权重={weights[0]:.4f} > CatBoost(IC最低)权重={weights[2]:.4f}")
    return True


def test_market_regime():
    """测试市场状态识别"""
    print("\n[测试5] 市场状态识别")

    # 生成测试价格序列
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)  # 低波动
    prices = pd.Series(100 * np.cumprod(1 + returns))

    regime = compute_market_regime(prices, lookback_days=60, regime_type="volatility")
    assert regime in ["high_vol", "low_vol", "normal_vol", "unknown"], f"未知状态: {regime}"
    print(f"  ✓ 通过: 波动率状态 = {regime}")

    # 趋势状态
    regime_trend = compute_market_regime(prices, lookback_days=60, regime_type="trend")
    assert regime_trend in ["bull", "bear", "sideways", "unknown"], f"未知趋势状态: {regime_trend}"
    print(f"  ✓ 通过: 趋势状态 = {regime_trend}")

    return True


def test_prediction_confidence():
    """测试预测置信区间估计"""
    print("\n[测试6] 预测置信区间估计")

    # 生成模拟预测
    np.random.seed(42)
    n_samples = 100
    predictions_list = [
        np.random.normal(0.05, 0.02, n_samples),
        np.random.normal(0.05, 0.025, n_samples),
        np.random.normal(0.04, 0.02, n_samples),
    ]

    result = compute_prediction_confidence(predictions_list, confidence_level=0.95)

    assert result["enabled"], "应该启用"
    assert len(result["mean_prediction"]) == n_samples, "均值长度不匹配"
    assert len(result["std_prediction"]) == n_samples, "标准差长度不匹配"
    assert len(result["lower_bound"]) == n_samples, "下界长度不匹配"
    assert len(result["upper_bound"]) == n_samples, "上界长度不匹配"
    assert result["confidence_level"] == 0.95, "置信水平不匹配"

    # 下界应该小于均值，上界应该大于均值
    assert np.all(result["lower_bound"] <= result["mean_prediction"]), "下界应小于等于均值"
    assert np.all(result["upper_bound"] >= result["mean_prediction"]), "上界应大于等于均值"

    print(f"  ✓ 通过: 置信区间计算成功，平均标准差={result['std_prediction'].mean():.6f}")
    return True


def test_risk_adjusted_score():
    """测试风险调整打分"""
    print("\n[测试7] 风险调整打分")

    # 生成测试数据
    np.random.seed(42)
    n_stocks = 100
    predictions = pd.Series(np.random.normal(0.05, 0.02, n_stocks))
    volatility = pd.Series(np.random.uniform(0.1, 0.5, n_stocks))

    adjusted = compute_risk_adjusted_score(predictions, volatility, adjustment_strength=0.5)

    assert len(adjusted) == len(predictions), "调整后长度不匹配"

    # 高波动股票的得分应该降低
    high_vol_mask = volatility > volatility.median()
    low_vol_mask = volatility <= volatility.median()

    high_vol_change = (adjusted[high_vol_mask] - predictions[high_vol_mask]).mean()
    low_vol_change = (adjusted[low_vol_mask] - predictions[low_vol_mask]).mean()

    assert high_vol_change < 0, "高波动股票得分应该降低"
    assert low_vol_change > 0, "低波动股票得分应该提高"

    print(f"  ✓ 通过: 高波动股票平均变化={high_vol_change:.6f} (降低)")
    print(f"  ✓ 通过: 低波动股票平均变化={low_vol_change:.6f} (提高)")
    return True


def test_turnover_control():
    """测试换手率控制（EMA 平滑）"""
    print("\n[测试8] 换手率控制")

    # 生成测试数据
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["stock1", "stock2", "stock3"]],
                                     names=["datetime", "instrument"])
    np.random.seed(42)
    scores = pd.DataFrame({"score": np.random.rand(len(idx))}, index=idx)

    smoothed = apply_turnover_control(scores, ema_alpha=0.3)

    assert len(smoothed) == len(scores), "平滑后长度不匹配"
    assert "score" in smoothed.columns, "缺少 score 列"

    # 平滑后的波动率应该低于原始波动率
    orig_vol = scores["score"].groupby(level="instrument").std().mean()
    smooth_vol = smoothed["score"].groupby(level="instrument").std().mean()
    assert smooth_vol < orig_vol, "平滑后波动率应该降低"

    reduction = 1 - smooth_vol / orig_vol
    print(f"  ✓ 通过: 波动率降低 {reduction:.1%} (原始={orig_vol:.4f}, 平滑后={smooth_vol:.4f})")
    return True


def test_dynamic_long_short():
    """测试动态多空比例"""
    print("\n[测试9] 动态多空比例")

    config = {
        "high_vol_long_pct": 0.20,
        "high_vol_short_pct": 0.05,
        "low_vol_long_pct": 0.30,
        "low_vol_short_pct": 0.10,
    }

    # 高波动
    result_high = compute_dynamic_long_short_ratio(0.40, config)
    assert abs(result_high["long_pct"] - 0.20) < 0.001, "高波动多头比例错误"
    assert abs(result_high["short_pct"] - 0.05) < 0.001, "高波动空头比例错误"
    print(f"  ✓ 通过: 高波动(40%) → 多头={result_high['long_pct']:.0%}, 空头={result_high['short_pct']:.0%}")

    # 低波动
    result_low = compute_dynamic_long_short_ratio(0.10, config)
    assert abs(result_low["long_pct"] - 0.30) < 0.001, "低波动多头比例错误"
    assert abs(result_low["short_pct"] - 0.10) < 0.001, "低波动空头比例错误"
    print(f"  ✓ 通过: 低波动(10%) → 多头={result_low['long_pct']:.0%}, 空头={result_low['short_pct']:.0%}")

    # 中等波动（线性插值）
    result_mid = compute_dynamic_long_short_ratio(0.225, config)
    assert 0.20 < result_mid["long_pct"] < 0.30, "中等波动多头比例应在中间"
    print(f"  ✓ 通过: 中等波动(22.5%) → 多头={result_mid['long_pct']:.1%}, 空头={result_mid['short_pct']:.1%}")

    return True


def test_feature_importance_stability():
    """测试特征重要性稳定性分析"""
    print("\n[测试10] 特征重要性稳定性分析")

    # 构造模拟特征重要性数据
    np.random.seed(42)
    n_features = 20
    feature_names = [f"factor_{i}" for i in range(n_features)]

    fi_list = []
    for window in range(3):
        # 每个窗口的重要性略有不同
        importance = np.random.uniform(0.5, 1.5, n_features)
        # 前 10 个因子更稳定
        importance[:10] *= 0.9 + np.random.uniform(-0.1, 0.1, 10)
        # 后 10 个因子更不稳定
        importance[10:] *= 0.5 + np.random.uniform(-0.3, 0.3, 10)

        fi_dict = dict(zip(feature_names, importance))
        fi_list.append(fi_dict)

    result = compute_feature_importance_stability(fi_list)

    assert result["n_windows"] == 3, "窗口数量错误"
    assert result["n_features"] == n_features, "特征数量错误"
    assert len(result["feature_stability"]) == n_features, "稳定性字典长度错误"
    assert 0 <= result["overall_stability"] <= 1, "整体稳定性应在 0-1 之间"

    # 前 10 个因子的平均稳定性应该高于后 10 个
    stable_names = [f"factor_{i}" for i in range(10)]
    unstable_names = [f"factor_{i}" for i in range(10, 20)]

    avg_stable = np.mean([result["feature_stability"][f] for f in stable_names])
    avg_unstable = np.mean([result["feature_stability"][f] for f in unstable_names])

    assert avg_stable > avg_unstable, "稳定因子的稳定性应高于不稳定因子"

    print(f"  ✓ 通过: 整体稳定性={result['overall_stability']:.4f}")
    print(f"  ✓ 通过: 稳定因子平均稳定性={avg_stable:.4f} > 不稳定因子={avg_unstable:.4f}")
    print(f"  ✓ 通过: 稳定特征 {len(result['stable_features'])} 个, 不稳定特征 {len(result['unstable_features'])} 个")
    return True


def test_neutralize_prediction():
    """测试预测值中性化"""
    print("\n[测试11] 预测值中性化")

    # 生成测试数据
    n_stocks = 100
    predictions = pd.Series(np.random.normal(0, 1, n_stocks))
    industry = pd.Series(np.random.choice(["行业A", "行业B", "行业C"], n_stocks))
    market_cap = pd.Series(np.random.uniform(1e9, 1e12, n_stocks))

    # 行业中性化
    neutralized = neutralize_prediction(predictions, industry=industry)
    assert len(neutralized) == len(predictions), "长度不匹配"

    # 每个行业的均值应该接近 0
    for ind in industry.unique():
        mask = industry == ind
        if mask.sum() > 5:
            mean_val = neutralized[mask].mean()
            assert abs(mean_val) < 0.1, f"行业 {ind} 均值应接近 0"

    print(f"  ✓ 通过: 行业中性化后各行业均值接近 0")

    # 市值中性化
    neutralized_mc = neutralize_prediction(predictions, market_cap=market_cap)
    assert len(neutralized_mc) == len(predictions), "长度不匹配"
    print(f"  ✓ 通过: 市值中性化成功")

    return True


def test_extreme_event_stress_test():
    """测试极端行情压力测试"""
    print("\n[测试12] 极端行情压力测试")

    # 生成测试数据
    n_dates = 100
    n_stocks = 50
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    idx = pd.MultiIndex.from_product([dates, [f"stock{i}" for i in range(n_stocks)]],
                                     names=["datetime", "instrument"])

    np.random.seed(42)
    predictions_df = pd.DataFrame({"score": np.random.rand(len(idx))}, index=idx)
    actual_returns = pd.Series(np.random.normal(0, 0.02, len(idx)), index=idx)

    result = extreme_event_stress_test(predictions_df, actual_returns, threshold_pct=0.05)

    assert "normal_ic" in result, "缺少正常日 IC"
    assert "extreme_up_ic" in result, "缺少大涨日 IC"
    assert "extreme_down_ic" in result, "缺少大跌日 IC"
    assert result["n_normal_days"] > 0, "正常日数量应为正"
    assert result["n_extreme_up_days"] > 0, "大涨日数量应为正"
    assert result["n_extreme_down_days"] > 0, "大跌日数量应为正"

    print(f"  ✓ 通过: 正常日 {result['n_normal_days']} 天, 大涨 {result['n_extreme_up_days']} 天, 大跌 {result['n_extreme_down_days']} 天")
    print(f"  ✓ 通过: 正常日 IC={result['normal_ic']:.4f}")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("  train_from_selected.py 新增功能单元测试")
    print("  [七层增强体系] 全面验证")
    print("=" * 60)

    tests = [
        ("第一层：数据增强层", [
            test_time_decay_weights,
            test_outlier_weights,
        ]),
        ("第二层：模型训练层", [
            test_purge_embargo,
        ]),
        ("第三层：集成增强层", [
            test_multi_dim_weights,
            test_market_regime,
        ]),
        ("第五层：风险管理层", [
            test_prediction_confidence,
            test_risk_adjusted_score,
            test_extreme_event_stress_test,
        ]),
        ("第六层：后处理增强层", [
            test_neutralize_prediction,
            test_turnover_control,
            test_dynamic_long_short,
        ]),
        ("第七层：可解释性层", [
            test_feature_importance_stability,
        ]),
    ]

    total_passed = 0
    total_failed = 0

    for layer_name, test_funcs in tests:
        print(f"\n{'─' * 60}")
        print(f"  {layer_name}")
        print(f"{'─' * 60}")

        for test_func in test_funcs:
            try:
                test_func()
                total_passed += 1
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1

    print(f"\n{'=' * 60}")
    print(f"  测试结果: {total_passed} 通过, {total_failed} 失败")
    print(f"{'=' * 60}")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
