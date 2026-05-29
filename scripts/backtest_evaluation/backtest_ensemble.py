"""
多模型量化回测对比脚本（擂台模式）

功能：统一加载树模型、线性模型、ICIR 基准的评分文件，
      使用完全相同的回测引擎和策略参数分别进行回测，
      并输出 4 个模型（tree / linear / icir / ensemble）的绩效对比报表。

用法：
  1. 确保 model_training/ 目录下存在 score_tree.csv / score_linear.csv / score_icir.csv
  2. 按需修改下方 CONFIG 中的回测参数
  3. python backtest_ensemble.py
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.backtest.bt_runner import run_qlib_backtrader, EnhancedQlibStrategy
from qlworks.backtest.industry import load_industry_map, apply_industry_constraint
import qlib
from qlib.data import D

warnings.filterwarnings('ignore')

# ==============================================================================
# [参数区] — 所有可调参数一目了然
# ==============================================================================
CONFIG = {

    # — 回测引擎参数 —
    "start_date": "2023-01-01",           # 回测起始日期
    "end_date": "2025-12-31",             # 回测结束日期
    "cash": 1000000.0,                    # 初始资金
    "commission": 0.0005,                 # 佣金费率（外部手续费）
    "slippage": 0.001,                    # 滑点比率
    "strategy_class": "EnhancedQlibStrategy",     # 回测策略类（EnhancedQlib = 内置风控+选股逻辑）

    # — 策略参数 —
    "top_k": 20,                          # 每日最大持仓数
    "score_threshold": 0.7,               # 选股最低分数
    "buy_pct": 0.95,                      # 资金使用率
    "rebalance_days": 5,                  # 调仓周期（交易日）

    # — 风控参数 —
    "use_risk_ctrl": True,                # 启用风控
    "stop_type": "ATR",                   # 止损类型 (ATR / FIXED)
    "stop_loss_pct": 0.05,                # 固定止损比例（仅 FIXED 生效）
    "atr_period": 14,                     # ATR 计算周期
    "atr_multiplier": 2.0,                # ATR 止损倍数
    "trailing_stop": True,                # 移动止盈
    "score_drop_threshold": 0.3,          # 得分恶化平仓阈值

    # — 行业敞口控制 —
    "industry_neutral": True,             # True=施加行业约束, False=纯信号对比
    "max_per_industry": 4,               # 单行业最大持仓数

    # — 子样本剥离（零算力防伪测试） —
    "subsample_filter": {
        "enabled": False,                 # True=仅回测指定成分股（如 csi300），False=全市场
        "instrument_alias": "csi300",     # Qlib instrument 别名（csi300 / csi500 等）
    },

    # — 预测文件路径 —
    "score_files": {
        "tree":   os.path.join(os.path.dirname(__file__), "../model_training/score_tree.csv"),
        "linear": os.path.join(os.path.dirname(__file__), "../model_training/score_linear.csv"),
        "icir":   os.path.join(os.path.dirname(__file__), "../model_training/score_icir.csv"),
    },

    # — 融合权重（权重和最好为 1） —
    "ensemble_weights": {
        "tree": 0.5,                      # 树模型权重
        "linear": 0.3,                    # 线性模型权重
        "icir": 0.2,                      # ICIR 基准权重
    }
}
# ==============================================================================

# 策略类注册表：将参数区配置的字符串映射到实际的策略类
_STRATEGY_REGISTRY = {
    "EnhancedQlibStrategy": EnhancedQlibStrategy,
}


def _get_strategy_class():
    """根据 CONFIG 中的策略名，从注册表查找并返回对应的策略类。"""
    name = CONFIG["strategy_class"]
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知的策略类: {name}，可选: {list(_STRATEGY_REGISTRY.keys())}")
    return cls


def get_price_dict(instruments, start_time, end_time):
    """从 Qlib 拉取多只个股的 OHLCV 行情数据，返回 {instrument: DataFrame} 格式的字典。"""
    qlib.init(provider_uri=r"e:\Quant\Qlibworks\qlib_data", region="cn")
    df = D.features(instruments, ['$open', '$high', '$low', '$close', '$volume'],
                    start_time=start_time, end_time=end_time)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    price_dict = {}
    for inst in instruments:
        if inst in df.index.get_level_values('instrument'):
            price_dict[inst] = df.xs(inst, level='instrument')
    return price_dict


def load_subsample_set():
    """子样本剥离：从 Qlib 加载指定成分股列表（如 csi300），返回 instrument 集合。"""
    cfg = CONFIG["subsample_filter"]
    if not cfg["enabled"]:
        return None
    from qlib.data import D
    insts = D.instruments(cfg["instrument_alias"])
    inst_set = set(insts)
    print(f"    [子样本] 加载 {cfg['instrument_alias']} 成分股: {len(inst_set)} 只")
    return inst_set


def run_backtest_for_score(pred_df, label, industry_map=None, subsample_set=None, strategy_params=None):
    """对单份评分数据执行一次完整的回测（加载行情 → 行业约束 → 子样本过滤 → 运行引擎 → 解析指标），返回绩效指标字典和输出目录。"""
    tag = f"{label}" + ("_约束" if CONFIG["industry_neutral"] else "_无约束")
    print(f"\n{'='*56}")
    print(f"    回测模型: {tag}")
    print(f"{'='*56}")

    if strategy_params is None:
        strategy_params = {
            "top_k": CONFIG["top_k"],
            "score_threshold": CONFIG["score_threshold"],
            "buy_pct": CONFIG["buy_pct"],
            "rebalance_days": CONFIG["rebalance_days"],
            "use_risk_control": CONFIG["use_risk_ctrl"],
            "stop_type": CONFIG["stop_type"],
            "stop_loss_pct": CONFIG["stop_loss_pct"],
            "atr_period": CONFIG["atr_period"],
            "atr_multiplier": CONFIG["atr_multiplier"],
            "trailing_stop": CONFIG["trailing_stop"],
            "score_drop_threshold": CONFIG["score_drop_threshold"],
            "log_enabled": True,
        }

    df = pred_df.copy()

    # 子样本过滤（零算力防伪测试）
    if subsample_set is not None:
        before_count = len(df)
        df = df[df.index.get_level_values('instrument').isin(subsample_set)]
        print(f"    [子样本] {before_count} 条 → {len(df)} 条，仅保留指定成分股")

    if CONFIG["industry_neutral"] and industry_map is not None:
        df = apply_industry_constraint(df, industry_map, top_k=CONFIG["top_k"], max_per_industry=CONFIG["max_per_industry"])

    all_instruments = df.index.get_level_values('instrument').unique().tolist()
    price_dict = get_price_dict(all_instruments, CONFIG["start_date"], CONFIG["end_date"])

    output_dir = os.path.join(os.path.dirname(__file__), f"../../drafts/bt_results_{label}")
    os.makedirs(output_dir, exist_ok=True)

    cerebro, results = run_qlib_backtrader(
        pred_df=df,
        price_df_dict=price_dict,
        strategy_class=_get_strategy_class(),
        strategy_params=strategy_params,
        initial_cash=CONFIG["cash"],
        commission=CONFIG["commission"],
        set_slippage_perc=CONFIG["slippage"],
        output_dir=output_dir,
        start_date=pd.to_datetime(CONFIG["start_date"]),
        end_date=pd.to_datetime(CONFIG["end_date"])
    )

    metrics = {'label': tag, 'final_cash': CONFIG["cash"], 'total_return': 0.0,
               'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe': 0.0,
               'sqn': 0.0, 'win_rate': 0.0, 'total_trades': 0}

    try:
        strat = results[0]
        dd = strat.analyzers.drawdown.get_analysis()
        ta = strat.analyzers.tradeanalyzer.get_analysis()
        sa = strat.analyzers.sharpe.get_analysis()
        sqn_a = strat.analyzers.sqn.get_analysis()
        ra = strat.analyzers.returns.get_analysis()

        final_value = cerebro.broker.get_value()
        total_ret = ((final_value - CONFIG["cash"]) / CONFIG["cash"]) * 100
        days = len(ra)
        annual_ret = ((1 + total_ret / 100) ** (252 / days) - 1) * 100 if days > 0 else 0.0
        max_dd = dd.get('max', {}).get('drawdown', 0)
        sharpe = sa.get('sharperatio', 0.0) or 0.0
        sqn = sqn_a.get('sqn', 0.0) or 0.0
        total_trades = ta.get('total', {}).get('closed', 0)
        won = ta.get('won', {}).get('total', 0)
        win_rate = (won / total_trades * 100) if total_trades > 0 else 0.0

        metrics = {
            'label': tag,
            'final_cash': final_value,
            'total_return': total_ret,
            'annual_return': annual_ret,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'sqn': sqn,
            'win_rate': win_rate,
            'total_trades': total_trades,
        }
    except Exception as e:
        print(f"    [警告] 解析 {label} 回测指标时出错: {e}")

    return metrics, output_dir


def print_comparison(all_metrics):
    """将多个模型的绩效指标以表格形式输出到控制台，并标注冠军模型。"""
    mode = "行业约束 ON" if CONFIG["industry_neutral"] else "行业约束 OFF"
    print(f"\n{'='*80}")
    print(f"    多模型量化回测对比报告")
    print(f"    模式: {mode}  |  调仓周期: {CONFIG['rebalance_days']}天  |  Top K: {CONFIG['top_k']}")
    print(f"{'='*80}")
    print(f"{'模型':<20}{'总收益率':<14}{'年化收益率':<14}{'最大回撤':<12}{'夏普比率':<12}{'SQN':<10}{'胜率':<10}{'交易次数':<10}")
    print(f"{'-'*20}{'-'*14}{'-'*14}{'-'*12}{'-'*12}{'-'*10}{'-'*10}{'-'*10}")

    best_sharpe = -999
    best_label = ""
    for m in all_metrics:
        sharpe_str = f"{m['sharpe']:.3f}" if m['sharpe'] != 0 else "N/A"
        print(f"{m['label']:<20}{m['total_return']:<+10.2f}%{'':4}{m['annual_return']:<+8.2f}%{'':6}"
              f"{m['max_drawdown']:<8.2f}%{'':4}{sharpe_str:<12}{m['sqn']:<8.3f}{'':2}"
              f"{m['win_rate']:<6.1f}%{'':4}{m['total_trades']:<10}")
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_label = m['label']

    print(f"{'='*80}")
    print(f"    [冠军] {best_label} (夏普 {best_sharpe:.3f})")


def run_model_comparison():
    """主入口：统一加载 3 个模型的评分 → 加载行业映射 → 分别回测 → 构建融合评分 → 输出对比报告。"""
    print("=" * 60)
    print("  多模型量化回测对比系统")
    print(f"  — 行业约束={'ON' if CONFIG['industry_neutral'] else 'OFF'}  |  调仓={CONFIG['rebalance_days']}d  |  TopK={CONFIG['top_k']} —")
    print("=" * 60)

    qlib.init(provider_uri=r"e:\Quant\Qlibworks\qlib_data", region="cn")

    all_instruments = set()
    for model_name in ['tree', 'linear', 'icir']:
        score_df = load_score(model_name)
        if score_df is not None:
            all_instruments.update(score_df.index.get_level_values('instrument').unique().tolist())
    industry_map = load_industry_map(list(all_instruments), CONFIG["start_date"])
    if all_instruments:
        coverage = sum(1 for inst in industry_map) / len(all_instruments) * 100
        print(f"    行业数据覆盖率: {len(industry_map)}/{len(all_instruments)} = {coverage:.1f}%")
    else:
        coverage = 0
        print("    行业数据覆盖率: 无可用股票数据")

    subsample_set = load_subsample_set()

    all_metrics = []
    for model_name in ['tree', 'linear', 'icir']:
        score_df = load_score(model_name)
        if score_df is None:
            continue
        metrics, _ = run_backtest_for_score(score_df, model_name, industry_map, subsample_set)
        all_metrics.append(metrics)

    print(f"\n    [Model Stacking] 融合权重: {CONFIG['ensemble_weights']}")
    ensemble_score = build_ensemble_score()
    if ensemble_score is not None:
        metrics, _ = run_backtest_for_score(ensemble_score, 'ensemble', industry_map, subsample_set)
        all_metrics.append(metrics)

    print_comparison(all_metrics)


def load_score(model_name):
    """从 CONFIG 中读取对应模型的评分 CSV 文件，返回 (datetime, instrument) 双索引的 DataFrame。"""
    path = CONFIG["score_files"][model_name]
    if not os.path.exists(path):
        print(f"[错误] {model_name} 的得分文件不存在: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.set_index(['datetime', 'instrument'], inplace=True)
    return df[['score']]


def build_ensemble_score():
    """按 CONFIG 中的融合权重，将 tree / linear / icir 三份评分加权合并，重新横截面排序后保存并返回。"""
    weights = CONFIG["ensemble_weights"]
    available = []

    for name, path in CONFIG["score_files"].items():
        if not os.path.exists(path):
            print(f"    [跳过] {name} 的得分文件不存在 ({path})，融合时将忽略该模型。")
            weights[name] = 0.0
            continue
        df = pd.read_csv(path, parse_dates=['datetime'])
        df.set_index(['datetime', 'instrument'], inplace=True)
        df.rename(columns={'score': f'{name}_score'}, inplace=True)
        available.append((name, df))

    if not available:
        print("    [错误] 没有任何可用的评分文件，无法构建融合评分。")
        return None

    total_w = sum(weights.values())
    if total_w == 0:
        print("    [错误] 融合权重全为零，无法构建融合评分。")
        return None
    for k in weights:
        weights[k] /= total_w

    merged = available[0][1]
    for name, df in available[1:]:
        merged = merged.join(df, how='outer').fillna(0.5)

    merged['score'] = sum(merged[f'{name}_score'] * weights[name] for name, _ in available)
    merged['score'] = merged.groupby(level='datetime')['score'].rank(pct=True, na_option='keep')

    out = os.path.join(os.path.dirname(__file__), "../model_training/score_ensemble.csv")
    merged[['score']].to_csv(out)
    print(f"    >>> 融合得分已保存至: {out}")
    return merged[['score']]


if __name__ == "__main__":
    run_model_comparison()
