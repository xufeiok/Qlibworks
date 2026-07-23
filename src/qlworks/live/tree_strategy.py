"""
树模型回测/模拟盘共享策略参数

职责:
- 统一回测与模拟盘使用的核心选股参数
- 为不同训练产物提供稳定的 live 策略档案
"""
from __future__ import annotations

from copy import deepcopy


DEFAULT_LIVE_STRATEGY = "selected"

LIVE_STRATEGY_PROFILES: dict[str, dict] = {
    "tree": {
        "model_name": "tree",
        "model_label": "树模型",
        "score_file": "score_tree.csv",
        "runtime_model_name": "tree",
        "top_k": 20,
        "score_threshold": 0.7,
        "buy_pct": 0.95,
        "rebalance_days": 5,
        "rebalance_signal_weekday": 1,
        "buy_weekday": 2,
        "use_risk_ctrl": True,
        "stop_type": "ATR",
        "stop_loss_pct": 0.05,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "trailing_stop": True,
        "score_drop_threshold": 0.3,
        "industry_neutral": True,
        "max_per_industry": 4,
    },
    "selected": {
        "model_name": "tree_selected",
        "model_label": "精选树模型",
        "score_file": "score_tree_selected.csv",
        # 通达信执行器当前默认读取 runtime/live/tree，精选结果沿用该目录即可直连模拟盘。
        "runtime_model_name": "tree",
        "top_k": 20,
        "score_threshold": 0.7,
        "buy_pct": 0.95,
        "rebalance_days": 5,
        "rebalance_signal_weekday": 1,
        "buy_weekday": 2,
        "use_risk_ctrl": True,
        "stop_type": "ATR",
        "stop_loss_pct": 0.05,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "trailing_stop": True,
        "score_drop_threshold": 0.3,
        # 本地 selected 训练链路已默认关闭行业依赖，这里同步关闭，减少 live 侧额外前置条件。
        "industry_neutral": False,
        "max_per_industry": 4,
    },
}


def get_live_strategy_config(strategy_name: str = DEFAULT_LIVE_STRATEGY) -> dict:
    """
    返回指定 live 策略档案。

    输入:
    - strategy_name: 策略档案名，当前支持 tree/selected

    输出:
    - dict: 供回测、目标持仓生成、模拟盘执行层复用的策略参数
    """
    normalized_name = (strategy_name or DEFAULT_LIVE_STRATEGY).strip().lower()
    if normalized_name not in LIVE_STRATEGY_PROFILES:
        supported = ", ".join(sorted(LIVE_STRATEGY_PROFILES))
        raise ValueError(f"不支持的 live 策略档案: {strategy_name}，可选: {supported}")
    return deepcopy(LIVE_STRATEGY_PROFILES[normalized_name])


_DEFAULT_CONFIG = get_live_strategy_config()

MODEL_NAME = _DEFAULT_CONFIG["model_name"]
MODEL_LABEL = _DEFAULT_CONFIG["model_label"]
SCORE_FILE = _DEFAULT_CONFIG["score_file"]

TOP_K = _DEFAULT_CONFIG["top_k"]
SCORE_THRESHOLD = _DEFAULT_CONFIG["score_threshold"]
BUY_PCT = _DEFAULT_CONFIG["buy_pct"]
REBALANCE_DAYS = _DEFAULT_CONFIG["rebalance_days"]
REBALANCE_SIGNAL_WEEKDAY = _DEFAULT_CONFIG["rebalance_signal_weekday"]
BUY_WEEKDAY = _DEFAULT_CONFIG["buy_weekday"]

USE_RISK_CTRL = _DEFAULT_CONFIG["use_risk_ctrl"]
STOP_TYPE = _DEFAULT_CONFIG["stop_type"]
STOP_LOSS_PCT = _DEFAULT_CONFIG["stop_loss_pct"]
ATR_PERIOD = _DEFAULT_CONFIG["atr_period"]
ATR_MULTIPLIER = _DEFAULT_CONFIG["atr_multiplier"]
TRAILING_STOP = _DEFAULT_CONFIG["trailing_stop"]
SCORE_DROP_THRESHOLD = _DEFAULT_CONFIG["score_drop_threshold"]

INDUSTRY_NEUTRAL = _DEFAULT_CONFIG["industry_neutral"]
MAX_PER_INDUSTRY = _DEFAULT_CONFIG["max_per_industry"]
