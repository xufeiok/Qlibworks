import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd

try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

def optimize_portfolio(
    prices_df: pd.DataFrame,
    predictions: pd.Series,
    target_volatility: float = 0.15,
    max_weight: float = 0.10,
    min_weight: float = 0.0,
) -> pd.Series:
    """
    功能概述：
    - 基于预测收益（Alpha 分数）和历史价格协方差矩阵，进行马科维茨均值-方差组合优化（Barra 风格的简化版）。
    - 目标是：在给定的目标波动率约束下，最大化投资组合的预期收益。

    输入：
    - prices_df: 历史价格数据，DataFrame，Index 为日期，Columns 为股票代码。
    - predictions: 机器学习模型的预测得分（作为预期收益），Series，Index 为股票代码。
    - target_volatility: 投资组合的年化目标波动率上限（默认 15%）。
    - max_weight: 单只股票的最大持仓权重限制（默认 10%，防个股黑天鹅）。
    - min_weight: 单只股票的最小持仓权重限制（默认 0%，只做多）。

    输出：
    - pd.Series，索引为股票代码，值为优化后的持仓权重（和为1）。

    边界条件：
    - 如果未安装 PyPortfolioOpt，将抛出异常。
    - 预测池和价格池必须对齐，底层会自动求交集。
    - 如果优化器无解（比如给定的目标波动率太低无法达成），将退化为市值等权或 Top-K 等权策略。
    """
    if not HAS_PYPFOPT:
        raise ImportError("请先安装 PyPortfolioOpt 库: pip install PyPortfolioOpt")

    if prices_df.empty or predictions.empty:
        raise ValueError("输入的价格数据或预测得分不能为空")

    # 1. 对齐数据 (预测的股票必须在价格数据中有历史)
    common_tickers = list(set(prices_df.columns).intersection(set(predictions.index)))
    if not common_tickers:
        raise ValueError("价格数据与预测得分没有股票交集")
    
    prices = prices_df[common_tickers]
    expected_ret = predictions[common_tickers]

    # 2. 估计协方差矩阵 (风险模型)
    # 这里使用 Ledoit-Wolf 压缩收缩法，对金融噪声数据比简单的样本协方差更稳健
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # 3. 实例化有效前界优化器
    ef = EfficientFrontier(expected_ret, S, weight_bounds=(min_weight, max_weight))
    
    # 4. 执行优化: 最大化夏普比率 或 约束目标波动率
    try:
        # A股市场个股波动率通常在 25%-40% 左右。如果 target_volatility 过低 (如 0.15)，
        # 且要求满仓 (权重和为1) + 不能做空 (min_weight=0)，优化器通常会无解 (infeasible)。
        # 因此这里我们首选尝试给定波动率，如果无解则回退到最大化夏普比率 (max_sharpe)
        try:
            weights = ef.efficient_risk(target_volatility=target_volatility)
        except Exception as risk_err:
            print(f"[Portfolio Opt] 目标波动率 {target_volatility} 无法实现 ({risk_err})，回退到最大化夏普比率 (max_sharpe)...")
            # 重新实例化，因为前一次 ef 的内部状态已被改变
            ef = EfficientFrontier(expected_ret, S, weight_bounds=(min_weight, max_weight))
            weights = ef.max_sharpe()
            
        # 清理极小权重并归一化
        cleaned_weights = ef.clean_weights()
        return pd.Series(cleaned_weights)
        
    except Exception as e:
        print(f"[Portfolio Opt] 优化器求解失败: {e}。将退化为正预测分数的加权分配。")
        # 降级方案：只买入得分为正的股票，并按得分大小加权
        positive_preds = expected_ret[expected_ret > 0]
        if positive_preds.empty:
            print("[Portfolio Opt] 所有预测分为负，将退化为全市场等权！")
            return pd.Series(1.0 / len(common_tickers), index=common_tickers)
        else:
            return positive_preds / positive_preds.sum()


if __name__ == "__main__":
    print("=== models/portfolio.py 独立测试 (模拟) ===")
    if not HAS_PYPFOPT:
        print("未安装 PyPortfolioOpt，跳过测试。")
    else:
        # 构造假价格数据
        dates = pd.date_range("2020-01-01", periods=100)
        tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
        prices = pd.DataFrame(
            np.random.normal(0.001, 0.02, size=(100, 4)), 
            index=dates, columns=tickers
        )
        # 累加模拟价格走势
        prices = (1 + prices).cumprod() * 100

        # 构造假预测分 (ML 模型输出)
        preds = pd.Series({"AAPL": 0.05, "MSFT": 0.02, "GOOG": -0.01, "TSLA": 0.08})
        
        print("优化前模型预测分:")
        print(preds)
        
        weights = optimize_portfolio(prices, preds, target_volatility=0.15)
        print("\n优化后投资组合权重 (约束个股上限 10%):")
        print(weights[weights > 0])
