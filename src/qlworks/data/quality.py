from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict

import numpy as np
import pandas as pd


def _get_datetime_index(data: pd.DataFrame):
    if isinstance(data.index, pd.MultiIndex) and "datetime" in data.index.names:
        return pd.to_datetime(data.index.get_level_values("datetime"))
    return pd.to_datetime(data.index)


def _calculate_completeness(data: pd.DataFrame) -> Dict[str, object]:
    """计算数据完整性 (Completeness)：评估数据是否存在大量空值(NaN)"""
    total_elements = int(data.size)
    non_null_elements = int(data.count().sum())
    
    # completeness (完整度): 非空数据占总数据的比例，越接近 1 越好
    completeness = non_null_elements / total_elements if total_elements else 0.0
    
    # missing_stats (各列缺失统计): 每列具体缺失的行数
    missing_stats = data.isnull().sum()
    
    # missing_ratio (各列缺失比例): 每列缺失值占该列总行数的比例，超过 0.2 通常需要警惕
    missing_ratio = missing_stats / len(data) if len(data) else missing_stats * 0
    return {
        "completeness": completeness,
        "missing_stats": missing_stats,
        "missing_ratio": missing_ratio,
    }


def _calculate_consistency(data: pd.DataFrame) -> Dict[str, object]:
    """计算数据一致性 (Consistency)：评估数据是否符合金融逻辑常识（如最高价必须大于等于开/收盘价）"""
    issues = 0
    total_checks = 0
    price_alias = {
        "open": "$open" if "$open" in data.columns else "open",
        "high": "$high" if "$high" in data.columns else "high",
        "low": "$low" if "$low" in data.columns else "low",
        "close": "$close" if "$close" in data.columns else "close",
    }
    if all(col in data.columns for col in price_alias.values()):
        valid_high = (
            data[price_alias["high"]]
            >= data[[price_alias["open"], price_alias["close"]]].max(axis=1)
        ).sum()
        valid_low = (
            data[price_alias["low"]]
            <= data[[price_alias["open"], price_alias["close"]]].min(axis=1)
        ).sum()
        issues += (len(data) - int(valid_high)) + (len(data) - int(valid_low))
        total_checks += 2 * len(data)

    volume_col = "$volume" if "$volume" in data.columns else "volume"
    if volume_col in data.columns:
        # 逻辑检查 3: 成交量不能为负数
        issues += int((data[volume_col] < 0).sum())
        total_checks += len(data)

    # consistency_score (一致性得分): 符合逻辑的数据比例，1.0 表示完全没有逻辑错误
    # consistency_issues (异常数量): 违背常识的数据点总数
    score = 1.0 if total_checks == 0 else 1 - issues / total_checks
    return {"consistency_score": score, "consistency_issues": issues, "total_checks": total_checks}


def _calculate_timeliness(data: pd.DataFrame, expected_freq: str = "D") -> Dict[str, object]:
    """计算数据时效性 (Timeliness)：评估数据更新是否及时，以及时间序列是否连续无断层"""
    dt_index = _get_datetime_index(data)
    if len(dt_index) <= 1:
        return {
            "timeliness_score": 1.0,
            "freq_consistency": 1.0,
            "freshness_score": 1.0,
            "data_lag_days": 0,
        }
    time_diff = pd.Series(dt_index).diff().dropna()
    tolerance = pd.Timedelta(days=3) if expected_freq == "D" else pd.Timedelta(expected_freq) * 2
    freq_consistency = float((time_diff <= tolerance).sum() / len(time_diff))

    latest_date = dt_index.max().normalize()
    current_date = pd.Timestamp.now().normalize()
    data_lag = int((current_date - latest_date).days)
    if data_lag <= 1:
        freshness = 1.0
    elif data_lag <= 7:
        freshness = 0.8
    elif data_lag <= 30:
        freshness = 0.6
    else:
        freshness = 0.4

    # freq_consistency (频率连续性): 实际时间间隔符合预期的比例，防范数据中间出现断层
    # freshness_score (新鲜度): 距离当前时间的远近得分，越近分数越高
    # data_lag_days (延迟天数): 最新数据距离今天的天数
    return {
        "timeliness_score": (freq_consistency + freshness) / 2,
        "freq_consistency": freq_consistency,
        "freshness_score": freshness,
        "data_lag_days": data_lag,
    }


def _detect_outliers(data: pd.DataFrame) -> pd.Series:
    """使用 IQR (四分位距) 方法检测极端异常值 (Outliers)"""
    outliers = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) < 10:
            outliers[col] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outliers[col] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[col] = int(((series < lower) | (series > upper)).sum())
    return pd.Series(outliers).sort_values(ascending=False)


def generate_data_quality_report(data: pd.DataFrame, expected_freq: str = "D") -> Dict[str, object]:
    """
    功能概述：
    - 生成量化研究常用的数据质量报告，覆盖完整性、一致性、时效性和异常值四大维度。
    输入：
    - data: 原始或清洗后的价量/因子数据。
    - expected_freq: 期望频率，默认日频 "D"。
    输出：
    - 包含详细指标的字典报告。具体指标释义：
      * overall_score (综合质量得分): 完整性、一致性、时效性的平均分，越接近 1 越好。
      * completeness (完整性): 评估数据缺失情况，防止过多 NaN 导致模型丢弃大量样本。
      * consistency (一致性): 评估 OHLCV 是否违背金融常识（如最高价小于收盘价），反映底层数据源的可靠性。
      * timeliness (时效性): 评估数据是否断更，以及最后更新日期是否满足实盘/回测的最新要求。
      * outliers (异常值数量): 统计各列中偏离正常分布的极端值个数，提示是否需要进行去极值处理。
    边界条件：
    - 空表会返回较为保守的默认值，调用方应优先判空。
    性能/安全注意事项：
    - 仅执行统计分析，不修改入参数据，适合作为研究前置体检步骤。
    """
    completeness = _calculate_completeness(data)
    consistency = _calculate_consistency(data)
    timeliness = _calculate_timeliness(data, expected_freq=expected_freq)
    outliers = _detect_outliers(data)
    scores = [
        completeness["completeness"],
        consistency["consistency_score"],
        timeliness["timeliness_score"],
    ]
    overall_score = float(np.mean(scores)) if scores else 0.0
    
    # 附带在报告中返回中文指标释义，方便终端打印或前端展示
    metrics_explanation = {
        "overall_score": "综合质量得分: 完整性、一致性、时效性的平均分，越接近 1 越好。",
        "completeness": "完整性: 评估数据缺失情况，防止过多 NaN 导致模型丢弃大量样本。",
        "consistency": "一致性: 评估 OHLCV 是否违背金融常识（如最高价小于收盘价），反映底层数据源的可靠性。",
        "timeliness": "时效性: 评估数据是否断更，以及最后更新日期是否满足实盘/回测的最新要求。",
        "outliers": "异常值数量: 统计各列中偏离正常分布的极端值个数，提示是否需要进行去极值处理。"
    }

    return {
        "overall_score": overall_score,
        "completeness": completeness,
        "consistency": consistency,
        "timeliness": timeliness,
        "outliers": outliers,
        "metrics_explanation": metrics_explanation,
    }


if __name__ == "__main__":
    print("=== data/quality.py 独立调用示例 ===")
    
    # 1. 构造带有缺失和逻辑异常的模拟数据
    dates = pd.date_range("2020-01-01", periods=10)
    instruments = ["000001.SZ"]
    multi_idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    
    df_mock = pd.DataFrame({
        "$open":  [10.0, 10.2, 10.1, np.nan, 10.5, 10.4, 10.3, 10.6, 10.8, 11.0],
        "$high":  [10.5, 10.4,  9.0, 10.6,  10.8, 10.7, 10.5, 10.9, 11.2, 11.5], # 第三天最高价小于开盘价 (违背常识)
        "$low":   [ 9.8,  9.9,  9.5,  9.7,   9.8,  9.7,  9.6,  9.9, 10.1, 10.2],
        "$close": [10.2, 10.1, 10.3, 10.4,  10.4, 10.5, 10.6, 10.8, 11.0, 11.1],
        "$volume":[100,  150,  -50,  200,   180,  220,  190,  250,  300,  280]  # 第三天成交量为负数
    }, index=multi_idx)
    
    # 2. 生成质量报告
    report = generate_data_quality_report(df_mock)
    
    print("\n[1] 模拟质量评估报告 (展示核心指标):")
    print(f"- 综合得分: {report['overall_score']:.4f}")
    print(f"- 完整性得分: {report['completeness']['completeness']:.4f}")
    print(f"- 一致性得分: {report['consistency']['consistency_score']:.4f} (发现逻辑违规数: {report['consistency']['consistency_issues']})")
    print(f"- 缺失列统计:\n{report['completeness']['missing_ratio']}")
