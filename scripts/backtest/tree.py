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
# [参数区] — 所有可调参数一目了然
# ==============================================================================

MODEL_NAME = "tree"                       # 模型名（用于日志和输出目录）
MODEL_LABEL = "树模型"                    # 中文标签

SCORE_FILE = "score_tree.csv"             # 训练脚本输出的预测文件名

# — 回测引擎参数 —
INITIAL_CASH = 1000000.0                  # 初始资金
COMMISSION = 0.0005                       # 佣金费率（外部手续费）
SLIPPAGE = 0.001                          # 滑点比率
STRATEGY_CLASS = EnhancedQlibStrategy     # 回测策略类（内置风控+选股逻辑）

# — 策略参数 —
TOP_K = 20                                # 每日最大持仓数
SCORE_THRESHOLD = 0.7                     # 选股最低分数
BUY_PCT = 0.95                            # 资金使用率
REBALANCE_DAYS = 5                        # 调仓周期（交易日）
REBALANCE_SIGNAL_WEEKDAY = 1              # 调仓信号日：0=周一 ... 1=周二
BUY_WEEKDAY = 2                           # 买入执行日：0=周一 ... 2=周三

# — 风控参数 —
USE_RISK_CTRL = True                      # 启用风控
STOP_TYPE = "ATR"                         # 止损类型 (ATR / FIXED)
STOP_LOSS_PCT = 0.05                      # 固定止损比例（仅 STOP_TYPE=FIXED 生效）
ATR_PERIOD = 14                           # ATR 计算周期
ATR_MULTIPLIER = 2.0                      # ATR 止损倍数
TRAILING_STOP = True                      # 移动止盈
SCORE_DROP_THRESHOLD = 0.3                # 得分恶化平仓阈值

# — 行业敞口控制 —
INDUSTRY_NEUTRAL = True                   # True=施加行业约束, False=纯信号对比
MAX_PER_INDUSTRY = 4                      # 单行业最大持仓数

# ==============================================================================

SCORE_PATH = os.path.join(os.path.dirname(__file__), f"../training/{SCORE_FILE}")
OUTPUT_DIR = os.path.dirname(__file__)


def main():
    print("=" * 60)
    print(f"=== {MODEL_LABEL} 回测（{'行业约束 ON' if INDUSTRY_NEUTRAL else '行业约束 OFF'}）===")
    print("=" * 60)

    print("[1] 初始化 Qlib 环境...")
    # 限制 Qlib 任务并发数为 1，防止触发 MemoryError 和 DLL 虚拟内存不足
    # 设置 Joblib 单线程执行
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    import joblib
    joblib.parallel_backend('sequential')
    
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")
    
    # 全局强制 joblib 后端为串行
    from qlib.config import C
    C.joblib_backend = "sequential"
    C.maxtasksperchild = 1

    print(f"\n[2] 读取预测得分 ({SCORE_FILE})...")
    if not os.path.exists(SCORE_PATH):
        raise FileNotFoundError(f"找不到预测文件: {SCORE_PATH}。请先运行对应的 train_*.py！")
    pred_df = pd.read_csv(SCORE_PATH, parse_dates=["datetime"])
    pred_df.set_index(["datetime", "instrument"], inplace=True)

    start_date = pred_df.index.get_level_values("datetime").min()
    end_date = pred_df.index.get_level_values("datetime").max()
    instruments = pred_df.index.get_level_values("instrument").unique().tolist()
    print(f"    测试集: {start_date.date()} ~ {end_date.date()}  |  股票池: {len(instruments)} 只")

    if INDUSTRY_NEUTRAL:
        print("\n[2.1] 加载行业映射(PIT)并施加行业约束...")
        industry_maps = load_industry_maps_pit(instruments, start_date, end_date)
        pred_df = apply_industry_constraint_pit(pred_df, industry_maps, top_k=TOP_K, max_per_industry=MAX_PER_INDUSTRY)
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
        # 过滤 OHLC 任何一个 <= 0.01 或异常巨大的无效行（Qlib 中停牌/缺失可能被表示为 1e-19 或 3.68e19，会导致底层崩溃）
        df = df[
            (df["close"] > 0.01) & (df["close"] < 1e6) &
            (df["open"] > 0.01) & (df["open"] < 1e6) &
            (df["high"] > 0.01) & (df["high"] < 1e6) &
            (df["low"] > 0.01) & (df["low"] < 1e6)
        ]
        if df.empty:
            continue
        valid = df[df["volume"] > 0]
        if not valid.empty:
            df = df[df.index <= valid.index[-1]]
        price_dict[inst] = df
    print(f"    已拉取 {len(price_dict)} 只股票行情。")

    print("    拉取中证 500 基准...")
    bench_df = None
    try:
        bm = D.features(["SH000905"], ["$open", "$high", "$low", "$close", "$volume"],
                        start_time=start_date, end_time=end_date)
        if not bm.empty:
            bench_df = bm.xs("SH000905", level="instrument").copy()
            bench_df.columns = ["open", "high", "low", "close", "volume"]
    except Exception:
        pass
        
    if bench_df is None or bench_df.empty:
        # 如果 Qlib 数据集中没有指数数据（通常只有个股），则直接从本地数据库的 index_daily 表拉取
        try:
            from qlworks.data.api import QuantDataAPI
            with QuantDataAPI() as api:
                query = f"SELECT trade_date as datetime, open, high, low, close, vol as volume FROM index_daily WHERE ts_code = '000905.SH' AND trade_date BETWEEN '{start_date.date()}' AND '{end_date.date()}' ORDER BY trade_date"
                df_idx = api.query(query)
                if not df_idx.empty:
                    df_idx['datetime'] = pd.to_datetime(df_idx['datetime'])
                    df_idx.set_index('datetime', inplace=True)
                    bench_df = df_idx
                    print(f"    成功从数据库拉取基准数据: {len(bench_df)} 行")
        except Exception as e:
            print("    从数据库拉取基准失败:", e)

    print("\n[4] 启动回测...")
    strategy_params = dict(
        top_k=TOP_K,
        score_threshold=SCORE_THRESHOLD,
        buy_pct=BUY_PCT,
        rebalance_days=REBALANCE_DAYS,
        rebalance_signal_weekday=REBALANCE_SIGNAL_WEEKDAY,
        buy_weekday=BUY_WEEKDAY,
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
