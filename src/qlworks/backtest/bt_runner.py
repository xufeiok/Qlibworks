import pandas as pd
import numpy as np
import os
import json
import argparse
import sys
import duckdb

# ========== Backtrader SuperPlot 路径配置 ==========
# 优先从环境变量 BT_SUPERPLOT_DIR 读取，后兼容旧硬编码路径
_bt_sp_env = os.environ.get('BT_SUPERPLOT_DIR')
if _bt_sp_env:
    bt_superplot_dir = os.path.abspath(_bt_sp_env)
else:
    bt_superplot_dir = os.path.abspath(r"e:\Quant\backtrader_superplot\backtrader_superplot")
custom_bt_dir = os.path.join(bt_superplot_dir, "backtrader-1.9.74.123", "backtrader-1.9.74.123")

if custom_bt_dir not in sys.path:
    sys.path.insert(0, custom_bt_dir)
if bt_superplot_dir not in sys.path:
    sys.path.insert(0, bt_superplot_dir)

import backtrader as bt

from qlworks.config import STAMP_DUTY, COMMISSION as CFG_COMMISSION

try:
    from backtrader.analyzers import SuperPlot
except ImportError:
    SuperPlot = None

from .bt_strategy import QlibPandasData, EnhancedQlibStrategy, AShareCommission


def _print_backtest_report(strat, initial_cash: float, final_value: float) -> None:
    """
    打印回测核心绩效报告。

    从策略的 analyzers 中提取关键指标，安全处理缺失/异常数据。

    Args:
        strat: 运行完成的 bt.Strategy 实例
        initial_cash: 初始资金
        final_value: 期末资金
    """
    try:
        trade_analyzer = strat.analyzers.tradeanalyzer.get_analysis()
        dd = strat.analyzers.drawdown.get_analysis()
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        try:
            sqn_analysis = strat.analyzers.sqn.get_analysis()
            sqn = sqn_analysis.get('sqn', 0.0)
        except (AttributeError, KeyError, TypeError):
            sqn = 0.0

        # 总收益率
        total_ret = 0.0
        if 'pnl' in trade_analyzer and 'net' in trade_analyzer['pnl'] and 'total' in trade_analyzer['pnl']:
            total_pnl = trade_analyzer['pnl']['net']['total']
            total_ret = (total_pnl / initial_cash) * 100
        else:
            total_ret = ((final_value - initial_cash) / initial_cash) * 100

        max_dd = dd.get('max', {}).get('drawdown', 0)
        sharpe_ratio = sharpe_analysis.get('sharperatio', 0.0) or 0.0

        total_trades = trade_analyzer.get('total', {}).get('closed', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0

        days = len(returns_analysis)
        annual_ret = ((1 + total_ret / 100) ** (252 / days) - 1) * 100 if days > 0 else 0.0

        # 基准表现提取
        bench_ret = 0.0
        bench_annual_ret = 0.0
        bench_max_dd = 0.0
        bench_final = initial_cash
        has_bench = False

        try:
            for d in strat.datas:
                if d._name == 'benchmark' and hasattr(d.p, 'dataname') and isinstance(d.p.dataname, pd.DataFrame):
                    df_bench = d.p.dataname
                    if not df_bench.empty and 'close' in df_bench.columns:
                        closes = df_bench['close'].replace(1.0, np.nan).dropna() # 排除填充的1.0脏数据
                        if len(closes) > 0:
                            has_bench = True
                            bench_start_price = closes.iloc[0]
                            bench_end_price = closes.iloc[-1]
                            
                            # 假设全仓买入基准（无交易费用的纯基准对比）
                            shares = initial_cash / bench_start_price
                            bench_final = shares * bench_end_price
                            bench_ret = ((bench_final - initial_cash) / initial_cash) * 100
                            
                            # 计算基准最大回撤
                            roll_max = closes.cummax()
                            drawdowns = (closes - roll_max) / roll_max * 100
                            bench_max_dd = drawdowns.min() * -1
                            
                            # 年化
                            bench_annual_ret = ((1 + bench_ret / 100) ** (252 / days) - 1) * 100 if days > 0 else 0.0
                    break
        except Exception as e:
            print(f"解析基准数据时出错: {e}")

        print("\n" + "=" * 40)
        print("【量化回测核心绩效报告 (Institutional Metrics)】")
        print("=" * 40)
        print(f"期末资金:   {final_value:.2f}")
        print(f"总收益率:   {total_ret:.2f}%")
        print(f"年化收益率: {annual_ret:.2f}%")
        print(f"最大回撤:   {max_dd:.2f}%")
        print(f"夏普比率:   {sharpe_ratio:.3f} (Sharpe Ratio)")
        print(f"系统质量:   {sqn:.3f} (SQN)")
        print(f"交易胜率:   {win_rate:.2f}% ({won_trades}/{total_trades})")
        calmar = annual_ret / max_dd if max_dd > 0 else float('inf')
        print(f"收益回撤比: {calmar:.2f} (Calmar Ratio Proxy)")
        
        if has_bench:
            print("-" * 40)
            print("【基准表现对比 (Benchmark Metrics)】")
            print(f"基准期末资金: {bench_final:.2f}")
            print(f"基准总收益率: {bench_ret:.2f}%")
            print(f"基准年化收益: {bench_annual_ret:.2f}%")
            print(f"基准最大回撤: {bench_max_dd:.2f}%")
            alpha_ret = total_ret - bench_ret
            alpha_annual = annual_ret - bench_annual_ret
            print(f"策略超额收益: {alpha_ret:.2f}% (总) / {alpha_annual:.2f}% (年化)")

        print("=" * 40 + "\n")
    except Exception as e:
        print(f"解析基础分析器结果时出错: {e}")

def run_qlib_backtrader(
    pred_df: pd.DataFrame,
    price_df_dict: dict,
    benchmark_df: pd.DataFrame = None,
    strategy_class=EnhancedQlibStrategy,
    strategy_params=None,
    initial_cash=100000.0,
    commission=CFG_COMMISSION,
    stamp_duty=STAMP_DUTY,
    set_slippage_perc=0.001,
    server_url='http://localhost:5888/api/backtest/upload',
    timeframe_label='1d',
    output_dir='./bt_output',
    start_date=None,
    end_date=None
):
    """
    运行基于 Qlib 预测结果的 Backtrader 回测，并支持上传至 SuperPlotView 前端。

    参数:
        pred_df: Qlib 预测结果，必须包含 MultiIndex ['datetime', 'instrument'] 和一列名为 'score'
        price_df_dict: 字典，键为股票代码(instrument)，值为包含 OHLCV 数据的 DataFrame
        strategy_class: 继承自 bt.Strategy 的回测策略类
        strategy_params: 策略参数字典
        initial_cash: 初始资金
        commission: 交易佣金费率
        set_slippage_perc: 滑点百分比
        server_url: SuperPlot 前端上传接口地址
        timeframe_label: 时间框标签
        output_dir: 结果及 SuperPlot JSON 本地保存目录
        start_date: 回测起始日期，若不指定则从 pred_df 推断
        end_date: 回测结束日期，若不指定则从 pred_df 推断
    """

    # 确定回测日期范围（使用预测数据中的实际交易日，避免引入非交易日的 NaN）
    full_calendar = pred_df.index.get_level_values('datetime').unique().sort_values()

    # [Filter] 预处理：剔除上市不满250日的新股
    from qlworks.factors.filter_utils import filter_codes_post
    _all_instruments = pred_df.index.get_level_values("instrument").unique().tolist()
    _filtered_instruments = filter_codes_post(_all_instruments, str(full_calendar[0].date()), filter_new_stocks=True, filter_st=False)
    _filtered_set = set(_filtered_instruments)
    pred_df = pred_df[pred_df.index.get_level_values("instrument").isin(_filtered_set)]
    # 也从 price_df_dict 中移除被过滤的股票
    for _inst in list(price_df_dict.keys()):
        if _inst not in _filtered_set:
            del price_df_dict[_inst]
    print(f"  [Filter] 新股过滤后: {len(_filtered_instruments)} / {len(_all_instruments)} 只股票")
    print(f"回测日期范围: {full_calendar.min().date()} ~ {full_calendar.max().date()} ({len(full_calendar)} 个交易日)")
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    # 禁止做空：A股不允许融券卖空
    cerebro.broker.set_shortcash(False)
    
    # [Virtu 改进] 注入 A 股真实手续费模型（区分买卖、印花税）
    comminfo = AShareCommission(stamp_duty=stamp_duty, commission=commission)
    cerebro.broker.addcommissioninfo(comminfo)
    
    if set_slippage_perc > 0:
        cerebro.broker.set_slippage_perc(set_slippage_perc)
    
    if strategy_params is None:
        strategy_params = {}
    cerebro.addstrategy(strategy_class, **strategy_params)
    
    # 优先添加基准数据，这样 SuperPlot 就可以把它识别为 benchmark_data 或 main feed
    if benchmark_df is not None:
        df_bench = benchmark_df.copy()
        if 'datetime' in df_bench.columns:
            df_bench = df_bench.set_index('datetime')
        elif df_bench.index.name != 'datetime':
            try:
                df_bench.index = pd.to_datetime(df_bench.index)
                df_bench.index.name = 'datetime'
            except:
                pass
        df_bench = df_bench[~df_bench.index.duplicated(keep='first')]
        # 手动重建索引以绕过 pandas reindex bug（同上）
        full_dates = pd.DatetimeIndex(full_calendar)
        matching_dates = full_dates.intersection(df_bench.index)
        df_new = pd.DataFrame(index=full_dates, columns=df_bench.columns, dtype="float64")
        for col in df_bench.columns:
            df_new.loc[matching_dates, col] = df_bench.loc[matching_dates, col].values.astype(np.float64)
        df_bench = df_new
        invalid_bench_mask = (
            df_bench['close'].isna() | (df_bench['close'] <= 0.01) | (df_bench['close'] > 1e6) |
            df_bench['open'].isna() | (df_bench['open'] <= 0.01) | (df_bench['open'] > 1e6) |
            df_bench['high'].isna() | (df_bench['high'] <= 0.01) | (df_bench['high'] > 1e6) |
            df_bench['low'].isna() | (df_bench['low'] <= 0.01) | (df_bench['low'] > 1e6) |
            df_bench['volume'].isna() | (df_bench['volume'] <= 0)
        )
        df_bench.loc[invalid_bench_mask, ['open', 'high', 'low', 'close']] = np.nan
        df_bench[['open', 'high', 'low', 'close']] = df_bench[['open', 'high', 'low', 'close']].ffill().fillna(1.0)
        df_bench['volume'] = df_bench['volume'].fillna(0)
        df_bench['score'] = np.nan # 补齐格式
        data_bench = QlibPandasData(dataname=df_bench, name='benchmark')
        cerebro.adddata(data_bench)
        print("已加载基准数据 (benchmark)")

    # 处理并添加数据源
    added_count = 0
    for idx, (inst, price_df) in enumerate(price_df_dict.items()):
        df = price_df.copy()

        # 将 MultiIndex 转换为单层 DatetimeIndex（Qlib 数据默认带 instrument 层级）
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.droplevel('instrument')

        # 确保 datetime 为索引
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        elif df.index.name != 'datetime':
            try:
                df.index = pd.to_datetime(df.index)
                df.index.name = 'datetime'
            except:
                pass

        # 如果 score 列已在数据中（tree.py 提前 join 的），跳过二次 join
        if 'score' not in df.columns:
            # 提取当前股票的预测分数
            if 'instrument' in pred_df.index.names:
                if inst in pred_df.index.get_level_values('instrument'):
                    inst_pred = pred_df.xs(inst, level='instrument')
                    if 'score' not in inst_pred.columns and len(inst_pred.columns) == 1:
                        inst_pred.columns = ['score']
                    df = df.join(inst_pred['score'], how='left')
                else:
                    df['score'] = np.nan
            else:
                df['score'] = np.nan

        # 移除重复索引
        df = df[~df.index.duplicated(keep='first')]

        # 调试：打印第一个股票的 score 统计
        if idx == 0:
            import tempfile
            _debug_p = output_dir if os.access(str(output_dir), os.W_OK) else tempfile.gettempdir()
            _debug_f = os.path.join(str(_debug_p), '_bt_debug_runner.txt')
            debug_f = open(_debug_f, 'w')
            score_notna = df['score'].notna().sum() if 'score' in df.columns else 0
            close_ok = (df['close'] > 0.01).sum() if 'close' in df.columns else 0
            debug_f.write(f'First stock: {inst}\n')
            debug_f.write(f'df rows: {len(df)}\n')
            debug_f.write(f'columns: {list(df.columns)}\n')
            debug_f.write(f'score_notna: {score_notna}\n')
            debug_f.write(f'close>0.01: {close_ok}\n')
            if score_notna > 0:
                debug_f.write(f'score>0.7: {(df["score"] > 0.7).sum()}\n')
                debug_f.write(f'score max: {df["score"].max()}\n')
            debug_f.write(f'index_dates[:5]: {df.index[:5].astype(str).tolist()}\n')
            debug_f.write(f'pred dates from df: {sorted(pred_df.index.get_level_values("datetime").unique())[:5]}\n')
            debug_f.close()

        # 对齐到统一日期范围（所有数据源等长）
        # 手动重建索引以绕过 "cannot include dtype 'M' in a buffer" 的 pandas 内部 bug
        full_dates = pd.DatetimeIndex(full_calendar)
        matching_dates = full_dates.intersection(df.index)
        df_aligned = pd.DataFrame(index=full_dates, columns=df.columns, dtype="float64")
        for col in df.columns:
            df_aligned.loc[matching_dates, col] = df.loc[matching_dates, col].values.astype(np.float64)
        df = df_aligned

        # 调试：reindex 后第一个股票 score 情况
        if idx == 0:
            debug_f = open(_debug_f, 'a')
            score_notna_re = df['score'].notna().sum()
            debug_f.write(f'After reindex: score_notna={score_notna_re}, total={len(df)}\n')
            debug_f.write(f'matching_dates[:5]: {matching_dates[:5].astype(str).tolist() if len(matching_dates) > 0 else "EMPTY!"}\n')
            debug_f.close()

        # 找出真正的无效天（原本就没有数据的天，或者价格<=0.01的天，防范 1e-19 极小值脏数据，或者异常大的脏数据，或者停牌日 volume<=0）
        invalid_mask = (
            df['close'].isna() | (df['close'] <= 0.01) | (df['close'] > 1e6) |
            df['open'].isna() | (df['open'] <= 0.01) | (df['open'] > 1e6) |
            df['high'].isna() | (df['high'] <= 0.01) | (df['high'] > 1e6) |
            df['low'].isna() | (df['low'] <= 0.01) | (df['low'] > 1e6) |
            df['volume'].isna() | (df['volume'] <= 0)
        )

        # 把无效天的 score 强行置为 NaN，确保它在这天不会被策略选中买入
        df.loc[invalid_mask, 'score'] = np.nan

        # 调试：invalid_mask 后 score 情况
        if idx == 0:
            debug_f = open(_debug_f, 'a')
            score_notna_after = df['score'].notna().sum()
            invalid_count = invalid_mask.sum()
            debug_f.write(f'After invalid_mask: score_notna={score_notna_after}, invalid={invalid_count}/{len(df)}\n')
            debug_f.close()

        # 修复价格数据，过滤掉任何<=0.01的坏价格
        df.loc[invalid_mask, ['open', 'high', 'low', 'close']] = np.nan
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
        
        # 兜底：如果连bfill都填不上（极小概率），填1.0防崩溃
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(1.0)
        
        df['volume'] = df['volume'].fillna(0)
        df['openinterest'] = 0

        # 现在 df 已经完美覆盖了 full_calendar，没有NaN价格，没有0价
        data = QlibPandasData(dataname=df, name=inst, timeframe=bt.TimeFrame.Days)
        cerebro.adddata(data)
        added_count += 1
            
    print(f"成功加载了 {added_count} 个股票的数据源。")
    # 诊断数据起始日期
    min_date = pd.Timestamp('2099-01-01')
    max_min_date = pd.Timestamp('1900-01-01')
    for d in cerebro.datas:
        d_min = d.p.dataname.index.min()
        min_date = min(min_date, d_min)
        max_min_date = max(max_min_date, d_min)
    print(f"[DEBUG] 数据最早起始日: {min_date}, 最晚起始日: {max_min_date}")
    has_superplot = False
    if SuperPlot is not None:
        cerebro.addanalyzer(SuperPlot, 
                            server_url=server_url,
                            autoconnect=True,
                            save_file='debug_output.json',  # 保存到文件以便调试
                            timeframe=timeframe_label,
                            max_points=0)  # 0 表示不裁剪，保留全部历史数据，方便前端懒加载
        has_superplot = True
        print("已成功挂载 SuperPlot 分析器，将尝试向前端系统上传回测结果。")
    else:
        print("警告: 未找到 SuperPlot 分析器，请确认 backtrader_superplot 路径配置正确。")
        
    # 添加其他常用分析器
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    # 注意：不添加 SQN 分析器，因其内部 Calmar 计算在净值归零时会 math.log(负数) 崩溃
        
    print(f"=== 开始回测 ===")
    print(f"初始资金: {cerebro.broker.get_value():.2f}")
    # 所有数据已经完全对齐，可以安全使用向量化模式加速
    try:
        results = cerebro.run()
    except Exception as e:
        print(f"[回测异常] {e}")
        print("回测运行过程中出现异常，将尝试输出已计算的中间结果。")
        results = None
    final_value = cerebro.broker.get_value()
    print(f"期末资金: {final_value:.2f}")
    
    if results:
        _print_backtest_report(results[0], initial_cash, final_value)

        # 保存 SuperPlot 结果
        if has_superplot:
            try:
                sp_data = results[0].analyzers.superplot.get_analysis()

                # 用我们自定义的带 stock_code 的订单列表覆盖原来的 orders_data
                if hasattr(results[0], 'all_orders') and len(results[0].all_orders) > 0:
                    if 'orders_data' not in sp_data:
                        sp_data['orders_data'] = {}
                    sp_data['orders_data']['orders'] = results[0].all_orders

                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f'qlib_bt_superplot.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(sp_data, f, ensure_ascii=False, indent=2)
                print(f"SuperPlot 渲染数据已本地保存至: {out_path}")
                print(rf"请使用 e:\Quant\backtrader_superplot\SuperPlotView 将其可视化。")
            except Exception as e:
                print(f"保存 SuperPlot 数据失败: {e}")
            
    return cerebro, results

def run_duckdb_backtrader(
    ts_codes: list,
    duckdb_path: str = None,
    start_date: str = '2020-01-01',
    end_date: str = '2025-12-31',
    benchmark_code: str = '000300.SH',
    strategy_class=bt.Strategy, 
    strategy_params=None,
    initial_cash=100000.0,
    commission=CFG_COMMISSION,
    set_slippage_perc=0.0,
    server_url='http://localhost:5888/api/backtest/upload',
    timeframe_label='1d',
    output_dir='./bt_output'
):
    """
    专为 A 股设计的常规 Backtrader 回测引擎。
    直接从本地 DuckDB 读取数据，无需经过 Qlib 预测阶段。
    
    参数:
        ts_codes: 需要回测的股票代码列表，例如 ['000001.SZ', '600000.SH']
        duckdb_path: DuckDB 数据库文件路径（默认使用 config.py 中的 DUCKDB_PATH）
        start_date: 回测起始日期 'YYYY-MM-DD'
        end_date: 回测结束日期 'YYYY-MM-DD'
        benchmark_code: 基准标的代码（默认沪深300），用于 SuperPlot 对比展示
        strategy_class: 回测策略类
        strategy_params: 策略参数字典
        ... 其他参数同 run_qlib_backtrader
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    # 禁止做空：A股不允许融券卖空
    cerebro.broker.set_shortcash(False)
    
    if set_slippage_perc > 0:
        cerebro.broker.set_slippage_perc(set_slippage_perc)
    
    if strategy_params is None:
        strategy_params = {}
    cerebro.addstrategy(strategy_class, **strategy_params)

    print(f"正在连接 ClickHouse+DuckDB...")
    try:
        from qlworks.config import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE, DUCKDB_PATH as CFG_DUCKDB_PATH
        import clickhouse_connect
        ch_client = clickhouse_connect.get_client(
            host=CH_HOST,
            port=CH_PORT,
            user=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE
        )
    except Exception as e:
        print(f"连接 ClickHouse 失败: {e}")
        return None, None

    if duckdb_path is None:
        duckdb_path = str(CFG_DUCKDB_PATH)

    def fetch_data(code):
        query = f"""
            SELECT trade_date as datetime, open, high, low, close, vol as volume
            FROM daily_prices
            WHERE ts_code = '{code}'
              AND trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_date
        """
        try:
            arrow_table = ch_client.query_arrow(query)
            conn = duckdb.connect()
            conn.register("kline_data", arrow_table)
            df = conn.execute("SELECT * FROM kline_data").df()
            conn.close()
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['openinterest'] = 0
            return df
        except Exception as e:
            print(f"查询 {code} 失败: {e}")
            return pd.DataFrame()

    # 1. 优先添加基准数据
    if benchmark_code:
        print(f"拉取基准数据: {benchmark_code}")
        df_bench = fetch_data(benchmark_code)
        if not df_bench.empty:
            data_bench = QlibPandasData(dataname=df_bench, name='benchmark', timeframe=bt.TimeFrame.Days)
            cerebro.adddata(data_bench)
            print(f"已加载基准数据 ({benchmark_code})")
        else:
            print(f"警告: 未找到基准 {benchmark_code} 的数据")

    # 2. 添加标的股票数据
    added_count = 0
    for code in ts_codes:
        df = fetch_data(code)
        if not df.empty:
            data = bt.feeds.PandasData(dataname=df, name=code, timeframe=bt.TimeFrame.Days)
            cerebro.adddata(data)
            added_count += 1
            print(f"已加载标的数据: {code} ({len(df)} 行)")
        else:
            print(f"跳过无数据标的: {code}")
    
    if added_count == 0:
        print("错误: 没有加载到任何有效标的数据，回测终止。")
        return None, None
        
    print(f"成功加载了 {added_count} 只标的股票。")

    # 尝试加载 SuperPlot 分析器
    has_superplot = False
    if SuperPlot is not None:
        cerebro.addanalyzer(SuperPlot, 
                            server_url=server_url,
                            autoconnect=True,
                            save_file='debug_output.json',
                            timeframe=timeframe_label,
                            max_points=0)
        has_superplot = True
        print("已成功挂载 SuperPlot 分析器，将尝试向前端系统上传回测结果。")
    else:
        print("警告: 未找到 SuperPlot 分析器，请确认 backtrader_superplot 路径配置正确。")
        
    # 添加其他常用分析器
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    # 注意：不添加 SQN 分析器，因其内部 Calmar 计算在净值归零时会 math.log(负数) 崩溃
        
    print(f"=== 开始回测 ===")
    print(f"初始资金: {cerebro.broker.get_value():.2f}")
    # runonce=False: 逐行处理兼容不等长数据
    try:
        results = cerebro.run(runonce=False)
    except Exception as e:
        print(f"[回测异常] {e}")
        print("回测运行过程中出现异常，将尝试输出已计算的中间结果。")
        results = None
    final_value = cerebro.broker.get_value()
    print(f"期末资金: {final_value:.2f}")
    
    if results:
        _print_backtest_report(results[0], initial_cash, final_value)

        # 保存 SuperPlot 结果
        if has_superplot:
            try:
                sp_data = results[0].analyzers.superplot.get_analysis()

                # 用自定义带 stock_code 的订单列表覆盖 (如果有)
                if hasattr(results[0], 'all_orders') and len(results[0].all_orders) > 0:
                    if 'orders_data' not in sp_data:
                        sp_data['orders_data'] = {}
                    sp_data['orders_data']['orders'] = results[0].all_orders

                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f'duckdb_bt_superplot.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(sp_data, f, ensure_ascii=False, indent=2)
                print(f"SuperPlot 渲染数据已本地保存至: {out_path}")
            except Exception as e:
                print(f"保存 SuperPlot 数据失败: {e}")

    return cerebro, results
