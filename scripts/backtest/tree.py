import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# 独立运行日志（终端沙盒超时后仍可查看进度）
LOG = open(os.path.join(os.path.dirname(__file__), "_tree_run.log"), "w", buffering=1)

import qlib
from qlib.data import D
from qlib.config import C
from qlworks.backtest.bt_runner import run_qlib_backtrader
from qlworks.backtest.bt_strategy import EnhancedQlibStrategy
from qlworks.backtest.industry import load_industry_maps_pit, apply_industry_constraint_pit
from qlworks.config import QLIB_DATA_DIR, CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
from qlworks.live.tree_strategy import get_live_strategy_config

# ==============================================================================
# [参数区] — 所有可调参数一目了然
# ==============================================================================

# — 回测引擎参数 —
INITIAL_CASH = 1000000.0                  # 初始资金
STAMP_DUTY = 0.0005                       # 印花税（卖出单向），当前 A 股万5
COMMISSION = 0.0003                       # 券商佣金（双向），当前 A 股万3
SLIPPAGE = 0.001                          # 滑点比率（0.1%，买卖双向）
STRATEGY_CLASS = EnhancedQlibStrategy     # 回测策略类（内置风控+选股逻辑）

# — 过滤开关 —
SKIP_ST_FILTER = False                    # True=跳过 ST 过滤（默认开启）
ADMISSION_BUFFER = 3                      # 准入预过滤缓冲系数：每天保留 top_k × buffer 只候选股
                                          # 例: top_k=10, buffer=3 → 每天保留 30 只

# ==============================================================================

OUTPUT_DIR = os.path.dirname(__file__)
_QLIB_INSTR_DIR = Path(QLIB_DATA_DIR) / "instruments"


def _load_delist_dates() -> dict:
    """
    从 all.txt 加载退市日期 PIT 映射。

    Returns:
        {stock_code_lower: delist_date_str}
        只在退市日期的股票才在字典中
    """
    _all_path = _QLIB_INSTR_DIR / "all.txt"
    delist_map = {}
    if not _all_path.exists():
        print("    [退市过滤] all.txt 不存在，无法进行退市 PIT 过滤！")
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


def _filter_delisted_only(pred_df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    """
    向量化退市过滤：剔除已退市股票。

    分两步：
    1. 剔除整个回测期前已退市的股票（无可用行情数据）
    2. 对回测期内退市的股票，剔除退市日期之后的行

    Args:
        pred_df: MultiIndex (datetime, instrument) 的预测 DataFrame
        start_date: 回测起始日期

    Returns:
        过滤后的 DataFrame
    """
    delist_pit = _load_delist_dates()
    if not delist_pit:
        print("    [退市过滤] 无退市 PIT 数据，跳过")
        return pred_df

    before = len(pred_df)
    delist_dates = {k: pd.Timestamp(v) for k, v in delist_pit.items()}

    # ---- 步骤 1：剔除整个回测期前已退市的股票 ----
    pre_delisted = {k for k, v in delist_dates.items() if v < start_date}
    if pre_delisted:
        insts_lower = pred_df.index.get_level_values("instrument").str.lower()
        pre_mask = ~insts_lower.isin(pre_delisted)
        pred_df = pred_df[pre_mask].copy()
        print(f"    [退市过滤-期前] 剔除 {len(pre_delisted)} 只期前已退市股票 "
              f"（前 5: {sorted(pre_delisted)[:5]}）")

    # ---- 步骤 2：回测期内退市股，剔除退市日期之后的行 ----
    insts = pred_df.index.get_level_values("instrument").str.lower()
    dts = pred_df.index.get_level_values("datetime")

    mapped = insts.to_series().map(delist_dates)
    row_delist = mapped.to_numpy(dtype="datetime64[ns]")
    nat_mask = np.isnat(row_delist)

    dts64 = dts.to_numpy(dtype="datetime64[ns]")
    keep_mask = nat_mask | (dts64 <= row_delist)
    filtered = pred_df[keep_mask].copy()
    after = len(filtered)
    print(f"    [退市过滤-期内] 前 {before} 行 → 后 {after} 行（剔除 {before - after} 行退市股）")

    if filtered.empty:
        print("    [严重警告] 退市过滤后预测数据为空！")
    return filtered


def _load_st_codes() -> set:
    """
    从 ClickHouse stock_basic 表获取当前 ST/*ST 股票代码列表。

    Returns:
        set of lower-cased ts_code（如 {'600654.sh', '000004.sz'}）
    """
    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT,
            username=CH_USER, password=CH_PASSWORD,
            database=CH_DATABASE,
        )
        rows = client.query(
            "SELECT ts_code FROM stock_basic "
            "WHERE name LIKE '%ST%' OR name LIKE '%*ST%'"
        )
        st_set = {r[0].lower() for r in rows.result_rows}
        client.close()
        print(f"    [ST 数据] 从 ClickHouse 加载 {len(st_set)} 只 ST/*ST 股票")
        return st_set
    except Exception as e:
        print(f"    [ST 数据] 查询失败（{e}），跳过 ST 过滤")
        return set()


def _filter_st_stocks(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤 ST/*ST 股票。

    注：当前基于 ClickHouse 中的最新股票名称判断，非严格 PIT 过滤。
    但对于 ST 股票（通常长期戴帽），此方法能覆盖绝大多数场景。

    Args:
        pred_df: MultiIndex (datetime, instrument) 的预测 DataFrame

    Returns:
        过滤后的 DataFrame
    """
    st_codes = _load_st_codes()
    if not st_codes:
        print("    [ST 过滤] 无 ST 数据，跳过")
        return pred_df

    before = len(pred_df)
    insts = pred_df.index.get_level_values("instrument").str.lower()
    st_mask = insts.isin(st_codes)
    filtered = pred_df[~st_mask].copy()
    after = len(filtered)
    print(f"    [ST 过滤] 前 {before} 行 → 后 {after} 行（剔除 {before - after} 行 ST 股）")

    if filtered.empty:
        print("    [严重警告] ST 过滤后预测数据为空！")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="树模型回测")
    parser.add_argument("--strategy", type=str, default="tree",
                        help="策略档案名，默认 'selected'，可选 'selected' 等")
    parser.add_argument("--no-st-filter", action="store_true",
                        help="跳过 ST 股票过滤（默认开启过滤）")
    parser.add_argument("--top-n", type=int, default=0,
                        help="仅保留 score 最高的 N 只候选股票；0=不限（默认）。"
                             "沙盒环境建议设置为 100 以加快回测速度。")
    args = parser.parse_args()

    config = get_live_strategy_config(args.strategy)
    score_file = config["score_file"]
    model_label = config["model_label"]
    industry_neutral = config["industry_neutral"]
    score_threshold = config["score_threshold"]
    top_k = config["top_k"]
    buy_pct = config["buy_pct"]
    rebalance_days = config["rebalance_days"]
    rebalance_signal_weekday = config["rebalance_signal_weekday"]
    buy_weekday = config["buy_weekday"]
    use_risk_ctrl = config["use_risk_ctrl"]
    stop_type = config["stop_type"]
    stop_loss_pct = config["stop_loss_pct"]
    atr_period = config["atr_period"]
    atr_multiplier = config["atr_multiplier"]
    trailing_stop = config["trailing_stop"]
    score_drop_threshold = config["score_drop_threshold"]
    max_per_industry = config["max_per_industry"]

    print("=" * 60)
    print(f"=== {model_label} 回测（{'行业约束 ON' if industry_neutral else '行业约束 OFF'}）===")
    print("=" * 60)
    LOG.write(f"=== {model_label} 回测 ===\n")

    print("[1] 初始化 Qlib 环境...")
    # 限制 Qlib 任务并发数，防止触发 MemoryError 和子进程死锁
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    import joblib
    joblib.parallel_backend('sequential')

    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")

    # Qlib 初始化后必须重新设置 joblib 后端（init 可能重置 C 对象）
    C.joblib_backend = "sequential"
    C.maxtasksperchild = 1
    LOG.write("[1] Qlib init OK\n")

    score_path = os.path.join(os.path.dirname(__file__), f"../training/{score_file}")
    print(f"\n[2] 读取预测得分 ({score_file})...")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"找不到预测文件: {score_path}。请先运行对应的 train_*.py！")
# TODO: 在这里加 filter_codes_post 过滤 pred_df 中的次新股/ST股
    pred_df = pd.read_csv(score_path, parse_dates=["datetime"])
    pred_df.set_index(["datetime", "instrument"], inplace=True)
    # 统一小写
    pred_df.index = pred_df.index.set_levels(
        pred_df.index.levels[1].str.lower(), level="instrument"
    )

    start_date = pred_df.index.get_level_values("datetime").min()
    end_date = pred_df.index.get_level_values("datetime").max()
    instruments = pred_df.index.get_level_values("instrument").unique().tolist()
    print(f"    测试集: {start_date.date()} ~ {end_date.date()}  |  原始股票池: {len(instruments)} 只")
    LOG.write(f"[2] 预测数据: {len(pred_df)} 行, {len(instruments)} 只股票\n")
    LOG.flush()

    # ========================================================================
    # [2.1] 退市过滤：剔除退市股在退市日期之后的预测信号
    # ========================================================================
    print(f"\n[2.1] 退市过滤...")
    pred_df = _filter_delisted_only(pred_df, start_date)
    instruments = pred_df.index.get_level_values("instrument").unique().tolist()
    print(f"    退市过滤后股票池: {len(instruments)} 只")
    LOG.write(f"[2.1] 退市过滤后: {len(instruments)} 只\n"); LOG.flush()

    # ========================================================================
    # [2.2] ST 过滤：剔除 ST/*ST 股票
    # ========================================================================
    if not args.no_st_filter:
        print(f"\n[2.2] ST 股票过滤...")
        pred_df = _filter_st_stocks(pred_df)
        instruments = pred_df.index.get_level_values("instrument").unique().tolist()
        print(f"    ST 过滤后股票池: {len(instruments)} 只")
        LOG.write(f"[2.2] ST过滤后: {len(instruments)} 只\n"); LOG.flush()
    else:
        print(f"\n[2.2] ST 过滤已跳过（--no-st-filter）")

    # ========================================================================
    # [2.3] 准入预过滤：每天只保留 score 排名 top_k × buffer 的候选股
    #
    #     回测引擎需要为每只股票加载全量 OHLCV 数据并注册为 Cerebro 数据源。
    #     全市场 2000+ 只股票都传入时，Cerebro 需要为每只股票对齐时间轴、
    #     维护 data feed，严重拖慢回测速度。
    #
    #     原理：策略每天只从 score 最高的 top_k 只中选股，因此在 pred_df
    #     层面提前截断——每天只保留 top_k × ADMISSION_BUFFER 只候选股的评分，
    #     其余股票评分置空后，后续行情加载时不会被 join 进来，
    #     自然不会被 Cerebro 加载。
    #
    #     例：top_k=10, buffer=3 → 每天保留 30 只候选股
    #     效果：2000+ 只 → 通常仅 150~400 只唯一股票进入 Cerebro
    #     回测速度提升 5-20x
    # ========================================================================
    if ADMISSION_BUFFER > 0:
        TOP_CANDIDATES = top_k * ADMISSION_BUFFER
        print(f"\n[2.3] 准入预过滤: 每天保留 score 前 {TOP_CANDIDATES} 只候选股...")
        LOG.write(f"[2.3] 准入预过滤: 每天 top {TOP_CANDIDATES}\n"); LOG.flush()
        before_inst = len(instruments)
        before_rows = len(pred_df)

        # 按天排名，取前 TOP_CANDIDATES 只
        keep_mask = (
            pred_df.groupby(level='datetime')['score']
            .rank(ascending=False, na_option='bottom')
            <= TOP_CANDIDATES
        )
        pred_df = pred_df[keep_mask].copy()
        instruments = pred_df.index.get_level_values("instrument").unique().tolist()

        print(f"    股票数: {before_inst} → {len(instruments)} 只")
        print(f"    数据行数: {before_rows} → {len(pred_df)} 行")
        LOG.write(f"    股票数: {before_inst} → {len(instruments)}\n"); LOG.flush()
    else:
        print(f"\n[2.3] 准入预过滤: 已跳过（ADMISSION_BUFFER=0）")
        LOG.write(f"[2.3] 准入预过滤跳过\n"); LOG.flush()

    # ========================================================================
    # [2.4] 股票池裁剪（可选）：保留 score 最高的 N 只（在行业约束之前执行）
    #
    #     --top-n 指定时生效；默认 0 = 不限。
    #     沙盒环境有时间/内存限制，建议用 --top-n 100。
    #     ⚠ 仅用于加速沙盒调试，正式回测应不使用此选项。
    #
    #     在行业约束之前执行，确保行业约束在裁剪后的池上正确施加。
    # ========================================================================
    if args.top_n > 0:
        print(f"\n[2.4] 股票池裁剪: 保留 score 最高的 {args.top_n} 只（沙盒加速模式）...")
        LOG.write(f"[2.4] 沙盒加速: top-{args.top_n}\n"); LOG.flush()
        scores_max = pred_df['score'].groupby(level='instrument').max()
        scores_max = scores_max.sort_values(ascending=False)
        keep_insts = scores_max.head(args.top_n).index.tolist()
        print(f"    保留: {len(keep_insts)}/{len(instruments)} 只 (max score: {scores_max.iloc[0]:.4f} ~ {scores_max.iloc[min(args.top_n-1, len(scores_max)-1)]:.4f})")
        LOG.write(f"[2.4] 保留 {len(keep_insts)}/{len(instruments)} 只\n"); LOG.flush()
        instruments = keep_insts
        pred_df = pred_df[pred_df.index.get_level_values('instrument').isin(instruments)]
        print(f"    裁剪后 pred_df: {len(pred_df)} 行")
        LOG.write(f"    裁剪后 pred_df: {len(pred_df)} 行\n"); LOG.flush()
    else:
        print(f"\n[2.4] 股票池裁剪跳过（全量回测，共 {len(instruments)} 只）")
        LOG.write(f"[2.4] 全量回测: {len(instruments)} 只\n"); LOG.flush()

    # ========================================================================
    # [2.5] 行业约束过滤（在裁剪后的池上施加）
    # ========================================================================
    if industry_neutral:
        print(f"\n[2.5] 加载行业映射(PIT)并施加行业约束...")
        LOG.write(f"[2.5] 行业约束...\n"); LOG.flush()
        industry_maps = load_industry_maps_pit(instruments, start_date, end_date)
        pred_df = apply_industry_constraint_pit(
            pred_df, industry_maps, top_k=top_k, max_per_industry=max_per_industry
        )
        instruments = pred_df.index.get_level_values("instrument").unique().tolist()
        print(f"    行业约束后股票池: {len(instruments)} 只")
        LOG.write(f"[2.5] 行业约束后: {len(instruments)} 只\n"); LOG.flush()
    else:
        print(f"\n[2.5] 行业约束跳过（--industry-neutral=OFF）")
        LOG.write(f"[2.5] 行业约束跳过\n"); LOG.flush()

    # ========================================================================
    # [2.6] 按 Qlib 数据存量过滤：剔除 features 目录中无数据文件的股票
    #
    #     检查 features/ 目录下是否有对应股票的子目录（含 .bin 文件）。
    #     避免 108 只预测数据存在但 Qlib 无行情文件的股票进入 [3] 的行情加载。
    #     这是纯文件系统检查，开销极小（listdir + set 运算）。
    # ========================================================================
    print(f"\n[2.6] 按 Qlib 数据存量过滤...")
    LOG.write(f"[2.6] 数据存量过滤...\n"); LOG.flush()
    features_dir = Path(QLIB_DATA_DIR) / "features"
    available_in_features = set()
    if features_dir.exists():
        for d in os.listdir(str(features_dir)):
            full = features_dir / d
            if full.is_dir():
                available_in_features.add(d.lower())
    else:
        print(f"    [警告] features 目录不存在: {features_dir}")
    before = len(instruments)
    instruments = [c for c in instruments if c in available_in_features]
    removed = before - len(instruments)
    if removed > 0:
        print(f"    剔除 {removed} 只无行情数据股票（features 目录无对应子目录）")
        # 同步从 pred_df 中剔除
        pred_df = pred_df[pred_df.index.get_level_values("instrument").isin(instruments)]
    else:
        print(f"    全部 {before} 只均有行情数据")
    print(f"    数据存量过滤后股票池: {len(instruments)} 只")
    LOG.write(f"[2.6] 数据存量过滤后: {len(instruments)} 只\n"); LOG.flush()

    # ========================================================================
    # [3] 拉取行情数据（含停牌过滤）
    #
    #     分批加载原因：Qlib D.features() 在 Windows 上多线程加载 3307 只股票
    #     × 5 特征时容易死锁（ParallelExt threading 后端 I/O 竞争）。
    #     实测 500 只 × 5 特征安全，这里每批 500 只分批加载后 concat。
    # ========================================================================
    print(f"\n[3] 拉取行情数据...")
    BATCH_SIZE = 500
    print(f"    共 {len(instruments)} 只股票，分批 {BATCH_SIZE} 只加载 5 特征...")
    LOG.write(f"[3] 开始分批加载 {len(instruments)} 只...\n"); LOG.flush()
    price_data_parts = []
    total_batches = (len(instruments) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_i in range(0, len(instruments), BATCH_SIZE):
        batch = instruments[batch_i:batch_i + BATCH_SIZE]
        batch_num = batch_i // BATCH_SIZE + 1
        print(f"    批次 {batch_num}/{total_batches}: {len(batch)} 只股票...", end=" ", flush=True)
        LOG.write(f"  batch {batch_num}/{total_batches} ({len(batch)} stocks)...\n"); LOG.flush()
        chunk = D.features(
            batch, ["$open", "$high", "$low", "$close", "$volume"],
            start_time=start_date, end_time=end_date,
        )
        if chunk is not None and not chunk.empty:
            price_data_parts.append(chunk)
            print(f"ok ({len(chunk)} 行)")
            LOG.write(f"  batch {batch_num} OK ({len(chunk)} rows)\n"); LOG.flush()
        else:
            print("空结果")
            LOG.write(f"  batch {batch_num} EMPTY\n"); LOG.flush()
    if price_data_parts:
        price_data = pd.concat(price_data_parts)
        price_data.columns = ["open", "high", "low", "close", "volume"]
        LOG.write(f"[3] 行情数据共 {len(price_data)} 行\n"); LOG.flush()
    # 释放分批加载的中间结果，降低内存压力
    del price_data_parts
    import gc; gc.collect()

    # 向量化构建 price_dict：groupby 替代逐个 xs() 切片
    # 注意：Qlib D.features() 返回的 instrument 是大写格式，需转小写与 pred_df 匹配
    # 预构建 score dict (O(1) 查表，避免 bt_runner 中 xs() 慢查询)
    print("    预构建 score 索引...")
    score_dict = {inst: grp.droplevel('instrument') for inst, grp in pred_df['score'].groupby(level='instrument')}
    print(f"    score_dict: {len(score_dict)} 只股票")
    price_dict = {}
    missing_stocks = []
    instrument_set = set(instruments)
    total_suspended_days = 0
    for inst, grp in price_data.groupby(level="instrument"):
        inst_lower = inst.lower()
        if inst_lower not in instrument_set:
            continue
        df = grp.droplevel("instrument").copy()
        df.dropna(subset=["close"], inplace=True)
        if df.empty:
            missing_stocks.append(inst_lower)
            continue
        # ----- 停牌过滤 1：OHLC 异常值过滤（停牌/缺失在 Qlib 中可能表示为 1e-19 或 3.68e19）-----
        mask = (
            (df["close"] > 0.01) & (df["close"] < 1e6) &
            (df["open"] > 0.01) & (df["open"] < 1e6) &
            (df["high"] > 0.01) & (df["high"] < 1e6) &
            (df["low"] > 0.01) & (df["low"] < 1e6)
        )
        df = df[mask]
        if df.empty:
            missing_stocks.append(inst_lower)
            continue
        # ----- 停牌过滤 2：成交量为 0 的停牌日过滤 -----
        valid = df[df["volume"] > 0]
        if not valid.empty:
            suspended_vol = len(df) - len(valid)
            total_suspended_days += suspended_vol
            # 截断到最后一个有成交量的日期之后的数据
            df = df[df.index <= valid.index[-1]]
        # ----- 提前 join score 列，避免 bt_runner 中每只股票 xs() 慢查询 -----
        if inst_lower in score_dict:
            score_s = score_dict[inst_lower]
            df = df.join(score_s, how='left')
        else:
            df['score'] = np.nan
        price_dict[inst_lower] = df

    print(f"    已拉取 {len(price_dict)} 只股票行情（停牌过滤剔除 {total_suspended_days} 行/日）")
    LOG.write(f"[3] price_dict: {len(price_dict)} 只股票\n"); LOG.flush()
    # 释放原始批量加载的大表，降低内存压力
    del price_data
    import gc; gc.collect()

    # 报告缺失股票
    for inst in instruments:
        if inst not in price_dict:
            if inst not in missing_stocks:
                missing_stocks.append(inst)
    if missing_stocks:
        print(f"    *** 警告: {len(missing_stocks)}/{len(instruments)} 只股票无可用行情数据 ***")
        print(f"        （缺失样本: {missing_stocks[:5]}）")

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
    LOG.write("[4] 启动回测...\n"); LOG.flush()
    strategy_params = dict(
        top_k=top_k,
        score_threshold=score_threshold,
        buy_pct=buy_pct,
        rebalance_days=rebalance_days,
        rebalance_signal_weekday=rebalance_signal_weekday,
        buy_weekday=buy_weekday,
        use_risk_control=use_risk_ctrl,
        stop_type=stop_type,
        stop_loss_pct=stop_loss_pct,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        trailing_stop=trailing_stop,
        score_drop_threshold=score_drop_threshold,
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
        stamp_duty=STAMP_DUTY,
        set_slippage_perc=SLIPPAGE,
        output_dir=OUTPUT_DIR,
        start_date=start_date,
        end_date=end_date,
    )
    LOG.write("[4] 回测完成\n"); LOG.flush()
    LOG.close()


if __name__ == "__main__":
    main()
