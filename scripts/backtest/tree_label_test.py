import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import qlib
from qlib.data import D
from qlworks.backtest.bt_runner import run_qlib_backtrader
from qlworks.backtest.bt_strategy_label import LabelConsistencyStrategy
from qlworks.backtest.industry import load_industry_maps_pit, apply_industry_constraint_pit
from qlworks.config import QLIB_DATA_DIR

# ==============================================================================
# [参数区] — 标签一致性回测专用
# ==============================================================================

MODEL_NAME = "linear_label"                 # 模型名（用于日志和输出目录）
MODEL_LABEL = "线性模型-T+1标签一致性验证"     # 中文标签

SCORE_FILE = "score_tree_selected.csv"             # 训练脚本输出的预测文件名

# — 回测引擎参数 —
INITIAL_CASH = 1000000.0                  # 初始资金
STAMP_DUTY = 0.0005                       # 印花税（卖出单向），当前 A 股万5
COMMISSION = 0.0003                       # 券商佣金（双向），当前 A 股万3
SLIPPAGE = 0.001                          # 滑点比率（0.1%，买卖双向）
STRATEGY_CLASS = LabelConsistencyStrategy # 专用标签一致性策略

# — 策略参数 —
TOP_K = 20                                # 每次买入最大持仓数
SCORE_THRESHOLD = 0.7                     # 选股最低分数
BUY_PCT = 0.95                            # 资金使用率
HOLDING_DAYS = 5                          # 严格持仓天数（对照实验：T+1开盘买，T+5收盘卖）

# — 行业敞口控制 —
INDUSTRY_NEUTRAL = True                   # True=施加行业约束, False=纯信号对比
MAX_PER_INDUSTRY = 4                      # 单行业最大持仓数

# — 反向测试控制 —
REVERSE_TEST = False                       # True=反向测试（买入得分最低的股票），False=正向测试

# ==============================================================================

SCORE_PATH = os.path.join(os.path.dirname(__file__), f"../training/{SCORE_FILE}")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_label_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Qlib 数据目录
_QLIB_INSTR_DIR = Path(QLIB_DATA_DIR) / "instruments"


def _load_csi500_pit() -> dict:
    """
    加载 csi500 成分股 PIT（时间点）映射。

    Returns:
        {stock_code_lower: [(start_date, end_date), ...]}
        每只股票在 csi500 中的时间区间列表（可能有多次进出）
    """
    _csi500_path = _QLIB_INSTR_DIR / "csi500.txt"
    pit_map = {}
    if not _csi500_path.exists():
        print("    [警告] csi500.txt 不存在，无法进行 CSI500 PIT 过滤！")
        return pit_map
    with open(_csi500_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                code, s, e = parts[0].lower(), parts[1], parts[2]
                pit_map.setdefault(code, []).append((s, e))
    print(f"    [CSI500 PIT] 加载 {len(pit_map)} 只成分股历史")
    return pit_map


def _load_delist_dates() -> dict:
    """
    加载退市日期 PIT 映射。

    Returns:
        {stock_code_lower: delist_date}
        只在退市日期的股票才在字典中
    """
    _all_path = _QLIB_INSTR_DIR / "all.txt"
    delist_map = {}
    if not _all_path.exists():
        print("    [警告] all.txt 不存在，无法进行退市 PIT 过滤！")
        return delist_map
    with open(_all_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                code, _list_d, _delist_d = parts[0].lower(), parts[1], parts[2]
                if _delist_d != '9999-12-31':
                    delist_map[code] = _delist_d
    if delist_map:
        print(f"    [退市 PIT] 加载 {len(delist_map)} 只退市股日期")
    return delist_map


def _pit_filter_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    逐日 PIT 过滤：确保每天参与回测的股票同时满足：
    1. 当天属于 csi500 成分股
    2. 当天未退市

    这是避免未来函数和幸存者偏差的核心过滤。

    Args:
        pred_df: MultiIndex (datetime, instrument) 的预测 DataFrame

    Returns:
        过滤后的 DataFrame
    """
    csi500_pit = _load_csi500_pit()
    delist_pit = _load_delist_dates()

    if not csi500_pit and not delist_pit:
        print("    [PIT 过滤] 无 PIT 数据可用，跳过过滤")
        return pred_df

    before = len(pred_df)
    filtered = []
    excluded_csi500 = 0
    excluded_delisted = 0

    for (dt, inst), row in pred_df.iterrows():
        dt_str = str(dt)[:10]
        inst_lower = inst.lower()

        # 1. 退市检查
        delist_d = delist_pit.get(inst_lower, '9999-12-31')
        if dt_str > delist_d:
            excluded_delisted += 1
            continue

        # 2. CSI500 成分股检查
        in_csi500 = False
        if inst_lower in csi500_pit:
            for s, e in csi500_pit[inst_lower]:
                if s <= dt_str <= e:
                    in_csi500 = True
                    break
        if not in_csi500:
            excluded_csi500 += 1
            continue

        filtered.append((dt, inst_lower))

    after = len(filtered)
    print(f"    [PIT 过滤] 前 {before} 行 → 后 {after} 行")
    print(f"      - 退市剔除: {excluded_delisted} 行")
    print(f"      - 非CSI500剔除: {excluded_csi500} 行")
    print(f"      - 实际剔除: {before - after} 行")

    if filtered:
        filtered_df = pred_df.loc[filtered].copy()
        filtered_df.index = filtered_df.index.set_levels(
            filtered_df.index.levels[1].str.lower(), level="instrument"
        )
        return filtered_df
    else:
        print("    [严重警告] PIT 过滤后预测数据为空！回测无法进行")
        return pred_df.iloc[0:0]


def _load_forward_adjusted_prices(api, instruments, start_date, end_date):
    """
    从 ClickHouse 获取前复权价格数据（使用 adj_factor 公式）。
    同 get_daily_data(adj=True) 逻辑，保证价格复权一致性。

    SQL 前复权公式:
        adj_price = raw_price * adj_factor / latest_adj_factor
    """
    ts_list = ", ".join(f"'{c.upper()}'" for c in instruments)
    adj_sql = f"""
        SELECT
            p.ts_code AS instrument,
            p.trade_date AS datetime,
            toFloat64(p.open * COALESCE(a.adj_factor, 1) / NULLIF(latest.adj_factor, 0)) AS open,
            toFloat64(p.high * COALESCE(a.adj_factor, 1) / NULLIF(latest.adj_factor, 0)) AS high,
            toFloat64(p.low * COALESCE(a.adj_factor, 1) / NULLIF(latest.adj_factor, 0)) AS low,
            toFloat64(p.close * COALESCE(a.adj_factor, 1) / NULLIF(latest.adj_factor, 0)) AS close,
            toFloat64(p.vol) AS volume
        FROM daily_prices p
        LEFT JOIN daily_adj_factors a ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
        LEFT JOIN (
            SELECT ts_code, argMax(toFloat64(adj_factor), trade_date) AS adj_factor
            FROM daily_adj_factors
            GROUP BY ts_code
        ) latest ON p.ts_code = latest.ts_code
        WHERE p.ts_code IN ({ts_list})
          AND p.trade_date >= '{start_date.date()}' AND p.trade_date <= '{end_date.date()}'
        ORDER BY p.ts_code, p.trade_date
    """
    df = api.query(adj_sql)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['instrument'] = df['instrument'].str.lower()
        df.set_index(['instrument', 'datetime'], inplace=True)
        print(f"    从 ClickHouse 获取前复权行情: {len(df)} 行")
    return df


def main():
    print("=" * 60)
    print(f"=== {MODEL_LABEL} 回测（{'行业约束 ON' if INDUSTRY_NEUTRAL else '行业约束 OFF'} | {'反向测试' if REVERSE_TEST else '正向测试'}）===")
    print("=" * 60)

    print("[1] 初始化 Qlib 环境...")
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    import joblib
    joblib.parallel_backend('sequential')

    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")
    from qlib.config import C
    C.joblib_backend = "sequential"
    C.maxtasksperchild = 1

    print(f"\n[2] 读取预测得分 ({SCORE_FILE})...")
    if not os.path.exists(SCORE_PATH):
        raise FileNotFoundError(f"找不到预测文件: {SCORE_PATH}。请先运行对应的 train_*.py！")
    pred_df = pd.read_csv(SCORE_PATH, parse_dates=["datetime"])
    pred_df.set_index(["datetime", "instrument"], inplace=True)

    # 统一小写
    pred_df.index = pred_df.index.set_levels(
        pred_df.index.levels[1].str.lower(), level="instrument"
    )

    start_date = pred_df.index.get_level_values("datetime").min()
    end_date = pred_df.index.get_level_values("datetime").max()
    instruments = pred_df.index.get_level_values("instrument").unique().tolist()
    print(f"    测试集: {start_date.date()} ~ {end_date.date()}  |  原始股票池: {len(instruments)} 只")

    # [PIT 过滤 — 杜绝未来函数和幸存者偏差]
    # 核心改进：逐日检查每只股票在当天是否满足：
    #   1. 属于 csi500 成分股（避免非成分股混入）
    #   2. 未退市（避免幸存者偏差）
    print("\n[2.1] PIT 过滤（CSI500 + 退市）+ 统一小写...")
    pred_df = _pit_filter_predictions(pred_df)
    instruments = pred_df.index.get_level_values("instrument").unique().tolist()
    print(f"    PIT 过滤后股票池: {len(instruments)} 只")

    if INDUSTRY_NEUTRAL:
        print("\n[2.2] 加载行业映射(PIT)并施加行业约束...")
        industry_maps = load_industry_maps_pit(instruments, start_date, end_date)
        pred_df = apply_industry_constraint_pit(pred_df, industry_maps, top_k=TOP_K,
                                                 max_per_industry=MAX_PER_INDUSTRY,
                                                 reverse_test=REVERSE_TEST)
        instruments = pred_df.index.get_level_values("instrument").unique().tolist()

    print("\n[3] 拉取行情数据（前复权）...")
    price_data = D.features(
        instruments, ["$open", "$high", "$low", "$close", "$volume"],
        start_time=start_date, end_time=end_date,
    )
    if not price_data.empty:
        price_data.columns = ["open", "high", "low", "close", "volume"]
        print(f"    从 Qlib 获取行情: {len(price_data)} 行")
    else:
        print("    *** Qlib D.features 返回空，从 ClickHouse 直接拉取前复权数据 ***")
        try:
            from qlworks.data import QuantDataAPI
            with QuantDataAPI() as api:
                price_data = _load_forward_adjusted_prices(api, instruments, start_date, end_date)
        except Exception as ch_err:
            print(f"    ClickHouse 前复权拉取失败: {ch_err}")

    price_dict = {}
    missing_stocks = []
    for inst in instruments:
        if inst not in price_data.index.get_level_values("instrument"):
            missing_stocks.append(inst)
            continue
        df = price_data.xs(inst, level="instrument").copy()
        df.dropna(subset=["close"], inplace=True)
        if df.empty:
            missing_stocks.append(inst)
            continue
        df = df[
            (df["close"] > 0.01) & (df["close"] < 1e6) &
            (df["open"] > 0.01) & (df["open"] < 1e6) &
            (df["high"] > 0.01) & (df["high"] < 1e6) &
            (df["low"] > 0.01) & (df["low"] < 1e6)
        ]
        if df.empty:
            missing_stocks.append(inst)
            continue
        valid = df[df["volume"] > 0]
        if not valid.empty:
            df = df[df.index <= valid.index[-1]]
        price_dict[inst] = df

    if missing_stocks:
        print(f"    *** 警告: {len(missing_stocks)}/{len(instruments)} 只股票无行情数据或数据非法 ***")
        print(f"        (缺失样本: {missing_stocks[:5]})")
    print(f"    实际可用: {len(price_dict)} 只股票行情。")

    print("    拉取中证 500 基准...")
    bench_df = None
    try:
        bm = D.features(["SH000905"], ["$open", "$high", "$low", "$close", "$volume"],
                        start_time=start_date, end_time=end_date)
        if not bm.empty:
            bench_df = bm.xs("SH000905", level="instrument").copy()
            bench_df.columns = ["open", "high", "low", "close", "volume"]
            print(f"    从 Qlib 获取基准: {len(bench_df)} 行")
    except Exception:
        pass

    if bench_df is None or bench_df.empty:
        try:
            from qlworks.data.api import QuantDataAPI
            with QuantDataAPI() as api:
                query = f"""
                    SELECT trade_date as datetime, open, high, low, close, vol as volume
                    FROM index_daily
                    WHERE ts_code = '000905.SH'
                      AND trade_date >= '{start_date.date()}'
                      AND trade_date <= '{end_date.date()}'
                    ORDER BY trade_date
                """
                df_idx = api.query(query)
                if not df_idx.empty:
                    df_idx['datetime'] = pd.to_datetime(df_idx['datetime'])
                    df_idx.set_index('datetime', inplace=True)
                    bench_df = df_idx
                    print(f"    从 ClickHouse 获取基准: {len(bench_df)} 行")
        except Exception as e:
            print("    从 ClickHouse 拉取基准失败:", e)

    print("\n[4] 启动回测...")
    strategy_params = dict(
        top_k=TOP_K,
        score_threshold=SCORE_THRESHOLD,
        holding_days=HOLDING_DAYS,
        buy_pct=BUY_PCT,
        log_enabled=True,
        reverse_test=REVERSE_TEST,
    )

    run_qlib_backtrader(
        pred_df=pred_df,
        price_df_dict=price_dict,
        benchmark_df=bench_df,
        strategy_class=STRATEGY_CLASS,
        strategy_params=strategy_params,
        initial_cash=INITIAL_CASH,
        commission=COMMISSION,
        stamp_duty=STAMP_DUTY,
        set_slippage_perc=SLIPPAGE,
        output_dir=OUTPUT_DIR,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
