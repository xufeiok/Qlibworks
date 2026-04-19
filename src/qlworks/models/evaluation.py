"""
模块说明：模型预测结果评估与分析 (Evaluation)

在量化投资中，模型训练完之后（输出 `pred` 分数），我们不能立刻去跑真实的回测（Backtest）。
回测非常耗时，且容易受到交易摩擦（手续费、滑点）的干扰。
我们需要在【纯预测层面】先对模型的预测能力进行“体检”。

这个文件主要计算量化界最权威的几个“预测能力”指标：
1. IC (Information Coefficient, 信息系数): 
   - 衡量模型预测分数 (`pred`) 与股票真实未来收益率 (`label`) 之间的线性相关性（皮尔逊相关系数）。
   - 范围在 [-1, 1] 之间。在 A 股，IC > 0.05 (5%) 就是一个非常有预测能力的 Alpha 模型了。
2. Rank IC (秩信息系数):
   - 衡量 `pred` 的排名和真实收益率排名之间的相关性（斯皮尔曼秩相关系数）。
   - 相比于 IC，它不受极端暴涨暴跌股票的影响，更加稳健。如果一个模型 Rank IC 很高但 IC 很低，说明它选股排序很准，但预测具体涨多少不准（在选股策略中，这就足够了）。
3. MSE (Mean Squared Error, 均方误差):
   - 回归模型的绝对误差。但在金融时间序列中，MSE 的意义不如 IC 大，因为收益率本身的方差就极大。
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """安全计算皮尔逊相关系数 (IC)。剔除 NaN 值，防止某天数据缺失导致整个结果报错。"""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() <= 1:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _safe_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """安全计算斯皮尔曼秩相关系数 (Rank IC)。"""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() <= 1:
        return float("nan")
    value, _ = spearmanr(a[mask], b[mask])
    return float(value)


def evaluate_prediction_frame(pred_frame: pd.DataFrame) -> Dict[str, object]:
    """
    功能概述：
    - 评估预测结果的 IC、RankIC、MSE 与时间序列稳定性。
    输入：
    - pred_frame: 至少包含 `pred` 与 `label` 两列，索引建议含 `datetime`。
    输出：
    - 评估结果字典，可直接用于打印、日志与回测前筛选。
    边界条件：
    - 缺少必要列时抛出异常。
    性能/安全注意事项：
    - 使用向量化和按日期分组统计，适合中等规模面板数据。
    """
    required = {"pred", "label"}
    missing = required - set(pred_frame.columns)
    if missing:
        raise ValueError(f"预测评估缺少必要列: {sorted(missing)}")

    frame = pred_frame[["pred", "label"]].dropna().copy()
    pred = frame["pred"].to_numpy(dtype=float)
    label = frame["label"].to_numpy(dtype=float)
    
    # 1. 整体视角的统计：把所有时间、所有股票的数据混在一起算一次总的 MSE 和 IC
    mse = float(np.mean((pred - label) ** 2)) if len(frame) else float("nan")
    ic = _safe_corr(pred, label)
    rank_ic = _safe_rank_corr(pred, label)

    # 2. 截面视角的统计 (更重要！)：按每天 (datetime) 分组，每天算一次 IC，最后求均值和标准差。
    ic_series = pd.Series(dtype=float)
    rank_ic_series = pd.Series(dtype=float)
    if isinstance(frame.index, pd.MultiIndex) and "datetime" in frame.index.names:
        grouped = frame.groupby(level="datetime")
        ic_series = grouped.apply(lambda x: _safe_corr(x["pred"].to_numpy(), x["label"].to_numpy()))
        rank_ic_series = grouped.apply(
            lambda x: _safe_rank_corr(x["pred"].to_numpy(), x["label"].to_numpy())
        )

    # 3. 输出评估字典：
    # - ic_mean / rank_ic_mean: 每天 IC 的均值 (衡量平均赚钱能力)
    # - ic_std / rank_ic_std: 每天 IC 的标准差 (衡量赚钱的稳定性，越小越稳)
    # - ic_positive_rate: 胜率 (有多少天的 IC 是正数，即预测方向对了的比例)
    return {
        "mse": mse,
        "ic": ic,
        "rank_ic": rank_ic,
        "ic_mean": float(ic_series.mean()) if len(ic_series) else ic,
        "ic_std": float(ic_series.std()) if len(ic_series) else float("nan"),
        "rank_ic_mean": float(rank_ic_series.mean()) if len(rank_ic_series) else rank_ic,
        "rank_ic_std": float(rank_ic_series.std()) if len(rank_ic_series) else float("nan"),
        "ic_positive_rate": float((ic_series > 0).mean()) if len(ic_series) else float("nan"),
    }


def select_top_instruments(pred: pd.Series, top_k: int = 20) -> pd.Series:
    """
    功能概述：
    - 从模型吐出的成百上千个横截面预测分数中，每天提取出排名前 TopK 的目标股票代码列表。
    - 这个函数的输出，就是直接喂给 Backtrader (或者 Qlib 内部回测引擎) 的“每日买入信号”。
    
    输入：
    - pred: MultiIndex(datetime, instrument) 的预测分数序列。
    - top_k: 每日买入的股票数量 (默认每天选最看好的 20 只股票)。
    
    输出：
    - 格式：Index 为 datetime，Value 为 list(股票代码) 的 Pandas Series。
    """
    if isinstance(pred.index, pd.MultiIndex) and "datetime" in pred.index.names:
        # 每天进行 groupby
        return pred.groupby(level="datetime").apply(
            # 每天按分数从大到小排序，取前 top_k 个，然后提取出股票代码 (instrument) 变成一个 list
            lambda x: x.sort_values(ascending=False).head(top_k).index.get_level_values("instrument").tolist()
        )
    return pd.Series(pred.sort_values(ascending=False).head(top_k).index.tolist())


if __name__ == "__main__":
    print("=== models/evaluation.py 独立调用示例 ===")
    
    # 1. 制造一份带 MultiIndex (日期，股票) 的预测结果
    dates = pd.date_range("2020-01-01", periods=3)
    instruments = [f"{str(i).zfill(6)}.SZ" for i in range(1, 31)] # 30 只股票
    multi_idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    
    # 模拟分数和真实收益率 (稍微有点正相关性)
    pred_scores = np.random.randn(90)
    real_labels = pred_scores * 0.1 + np.random.randn(90) * 0.05
    
    pred_frame = pd.DataFrame({"pred": pred_scores, "label": real_labels}, index=multi_idx)
    
    print(f"\n[1] 模型输出的预测表 (pred_frame) 前两行:\n{pred_frame.head(2)}")
    
    # 2. 调用核心评估函数
    eval_res = evaluate_prediction_frame(pred_frame)
    print("\n[2] 预测结果评估报告 (Evaluation):")
    print(f"  - 整体 MSE: {eval_res['mse']:.4f}")
    print(f"  - 每日 IC 均值 (IC Mean): {eval_res['ic_mean']:.4f} (越高越好)")
    print(f"  - 每日 Rank IC 均值 (Rank IC Mean): {eval_res['rank_ic_mean']:.4f} (越高越好)")
    print(f"  - 预测方向胜率: {eval_res['ic_positive_rate']:.2%}")
    
    # 3. 选出每天排名前 5 的目标股票
    top_stocks = select_top_instruments(pred_frame["pred"], top_k=5)
    print("\n[3] 从预测分数提取出的交易信号 (每天买入排名前5只):")
    print(top_stocks)
