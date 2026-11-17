import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import qlib
from qlib.data import D
from qlworks.backtest.bt_runner import run_qlib_backtrader
from qlworks.backtest.bt_strategy import EnhancedQlibStrategy
from qlworks.backtest.industry import load_industry_maps_pit, apply_industry_constraint_pit
from qlworks.config import QLIB_DATA_DIR

# ==============================================================================
# [参数区] — 三模型统一对比口径
# ==============================================================================

MODEL_NAME = "ridge_linear"
MODEL_LABEL = "Ridge线性回归"
SCORE_FILE = "score_linear.csv"

# — 回测引擎参数（三模型统一）—
INITIAL_CASH = 1000000.0
COMMISSION = 0.0005
SLIPPAGE = 0.001
STRATEGY_CLASS = EnhancedQlibStrategy

# — 策略参数（三模型统一）—
TOP_K = 20
SCORE_THRESHOLD = 0.0       # 三模型score量纲不同，统一不设阈值
BUY_PCT = 0.95
REBALANCE_DAYS = 5

# — 风控参数（三模型统一）—
USE_RISK_CTRL = True
STOP_TYPE = "ATR"
STOP_LOSS_PCT = 0.05
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
TRAILING_STOP = True
SCORE_DROP_THRESHOLD = 0.3

# — 行业敞口控制 —
INDUSTRY_NEUTRAL = True
MAX_PER_INDUSTRY = 4

# — 准入预过滤 —
ADMISSION_BUFFER = 5

# ==============================================================================

SCORE_PATH = os.path.join(os.path.dirname(__file__), f"../training/{SCORE_FILE}")
OUTPUT_DIR = os.path.dirname(__file__)


def main():
    print("=" * 60)
    print(f"=== {MODEL_LABEL} 回测 ===")
    print("=" * 60)

    print("[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")

    print(f"\n[2] 读取预测得分 ({SCORE_FILE})...")
    if not os.path.exists(SCORE_PATH):
        raise FileNotFoundError(f"找不到预测文件: {SCORE_PATH}。请先运行对应的 train_*.py！")
    pred_df = pd.read_csv(SCORE_PATH, parse_dates=["datetime"])
    pred_df.set_index(["datetime", "instrument"], inplace=True)
    from qlworks.pipeline_config import INSTRUMENTS, resolve_instrument_codes
    _pool = {code.lower() for code in resolve_instrument_codes(INSTRUMENTS)}
    pred_df = pred_df[pred_df.index.get_level_values("instrument").isin(_pool)]
    if pred_df.empty:
        raise ValueError(f"{SCORE_FILE} 不含 {INSTRUMENTS} 股票")

    start_date = pred_df.index.get_level_values("datetime").min()
    end_date = pred_df.index.get_level_values("datetime").max()
    instruments = pred_df.index.get_level_values("instrument").unique().tolist()
    print(f"    测试集: {start_date.date()} ~ {end_date.date()}  |  股票池: {len(instruments)} 只")

    if INDUSTRY_NEUTRAL:
        print("\n[2.1] 行业约束...")
        industry_maps = load_industry_maps_pit(instruments, start_date, end_date)
        pred_df = apply_industry_constraint_pit(pred_df, industry_maps, top_k=TOP_K, max_per_industry=MAX_PER_INDUSTRY)
        instruments = pred_df.index.get_level_values("instrument").unique().tolist()

    if ADMISSION_BUFFER > 0:
        TOP_CANDIDATES = TOP_K * ADMISSION_BUFFER
        print(f"\n[2.2] 准入预过滤: 每天保留 score 前 {TOP_CANDIDATES} 只...")
        keep_mask = (
            pred_df.groupby(level='datetime')['score']
            .rank(ascending=False, na_option='bottom')
            <= TOP_CANDIDATES
        )
        pred_df = pred_df[keep_mask].copy()
        instruments = pred_df.index.get_level_values("instrument").unique().tolist()

    print("\n[3] 拉取行情数据...")
    price_data = D.features(
        instruments, ["$open", "$high", "$low", "$close", "$volume"],
        start_time=start_date, end_time=end_date,
    )
    price_data.columns = ["open", "high", "low", "close", "volume"]
    price_dict = {}
    for inst in instruments:
        if inst not in price_data.index.get_level_values("instrument"):
            continue
        df = price_data.xs(inst, level="instrument").copy()
        df.dropna(subset=["close"], inplace=True)
        if df.empty:
            continue
        valid = df[df["volume"] > 0]
        if not valid.empty:
            df = df[df.index <= valid.index[-1]]
        price_dict[inst] = df
    print(f"    已拉取 {len(price_dict)} 只股票行情。")

    print("    拉取中证500基准...")
    try:
        bm = D.features(["SH000905"], ["$open", "$high", "$low", "$close", "$volume"],
                        start_time=start_date, end_time=end_date)
        bench_df = bm.xs("SH000905", level="instrument").copy() if not bm.empty else None
        if bench_df is not None:
            bench_df.columns = ["open", "high", "low", "close", "volume"]
    except Exception:
        bench_df = None

    print("\n[4] 启动回测...")
    strategy_params = dict(
        top_k=TOP_K,
        score_threshold=SCORE_THRESHOLD,
        buy_pct=BUY_PCT,
        rebalance_days=REBALANCE_DAYS,
        use_risk_control=USE_RISK_CTRL,
        stop_type=STOP_TYPE,
        stop_loss_pct=STOP_LOSS_PCT,
        atr_period=ATR_PERIOD,
        atr_multiplier=ATR_MULTIPLIER,
        trailing_stop=TRAILING_STOP,
        score_drop_threshold=SCORE_DROP_THRESHOLD,
        log_enabled=True,
    )

    run_qlib_backtrader(
        pred_df=pred_df,
        price_df_dict=price_dict,
        benchmark_df=bench_df,
        strategy_class=STRATEGY_CLASS,
        strategy_params=strategy_params,
        initial_cash=INITIAL_CASH,
        commission=COMMISSION,
        set_slippage_perc=SLIPPAGE,
        output_dir=OUTPUT_DIR,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
