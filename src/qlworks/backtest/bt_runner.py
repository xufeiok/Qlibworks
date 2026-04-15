import pandas as pd
import numpy as np
import os
import json
import argparse
import sys
import duckdb

# 为了让直接调用本文件或提前导入时不报错，如果 sys.path 还没有被修改，则在这里进行保底处理
bt_superplot_dir = os.path.abspath(r"e:\Quant\backtrader_superplot\backtrader_superplot")
custom_bt_dir = os.path.join(bt_superplot_dir, "backtrader-1.9.74.123", "backtrader-1.9.74.123")

if custom_bt_dir not in sys.path:
    sys.path.insert(0, custom_bt_dir)
if bt_superplot_dir not in sys.path:
    sys.path.insert(0, bt_superplot_dir)

import backtrader as bt

try:
    from backtrader.analyzers import SuperPlot
except ImportError:
    SuperPlot = None

from .bt_strategy import QlibPandasData, BaseQlibStrategy, EnhancedQlibStrategy

def run_qlib_backtrader(
    pred_df: pd.DataFrame, 
    price_df_dict: dict, 
    benchmark_df: pd.DataFrame = None,
    strategy_class=EnhancedQlibStrategy, 
    strategy_params=None,
    initial_cash=100000.0,
    commission=0.001,
    set_slippage_perc=0.0,
    server_url='http://localhost:5888/api/backtest/upload',
    timeframe_label='1d',
    output_dir='./bt_output'
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
    """
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
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
        df_bench['score'] = np.nan # 补齐格式
        data_bench = QlibPandasData(dataname=df_bench, name='benchmark')
        cerebro.adddata(data_bench)
        print("已加载基准数据 (benchmark)")

    # 处理并添加数据源
    added_count = 0
    for inst, price_df in price_df_dict.items():
        df = price_df.copy()
        
        # 确保 datetime 为索引
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        elif df.index.name != 'datetime':
            # 如果没有 datetime 列，且索引不是 datetime，尝试转换
            try:
                df.index = pd.to_datetime(df.index)
                df.index.name = 'datetime'
            except:
                pass
        
        # 提取当前股票的预测分数
        if 'instrument' in pred_df.index.names:
            if inst in pred_df.index.get_level_values('instrument'):
                inst_pred = pred_df.xs(inst, level='instrument')
                # 如果预测结果只有一列且没有命名为 score，强制重命名
                if 'score' not in inst_pred.columns and len(inst_pred.columns) == 1:
                    inst_pred.columns = ['score']
                df = df.join(inst_pred['score'], how='left')
            else:
                df['score'] = np.nan
        else:
            df['score'] = np.nan
            
        # 丢弃没有价格数据的行
        df = df.dropna(subset=['close'])
        
        if len(df) > 0:
            data = QlibPandasData(dataname=df, name=inst, timeframe=bt.TimeFrame.Days)
            cerebro.adddata(data)
            added_count += 1
            
    print(f"成功加载了 {added_count} 个股票的数据源。")
            
    # 尝试加载 SuperPlot 分析器
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
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
    print(f"=== 开始回测 ===")
    print(f"初始资金: {cerebro.broker.get_value():.2f}")
    results = cerebro.run()
    final_value = cerebro.broker.get_value()
    print(f"期末资金: {final_value:.2f}")
    
    if results:
        strat = results[0]
        
        try:
            dd = strat.analyzers.drawdown.get_analysis()
            trade_analyzer = strat.analyzers.tradeanalyzer.get_analysis()
            
            # 安全地获取总收益率
            total_ret = 0.0
            if 'pnl' in trade_analyzer and 'net' in trade_analyzer['pnl'] and 'total' in trade_analyzer['pnl']:
                total_pnl = trade_analyzer['pnl']['net']['total']
                total_ret = (total_pnl / initial_cash) * 100
            else:
                total_ret = ((final_value - initial_cash) / initial_cash) * 100
                
            print(f"总收益率: {total_ret:.2f}%")
            print(f"最大回撤: {dd.get('max', {}).get('drawdown', 0):.2f}%")
        except Exception as e:
            print(f"解析基础分析器结果时出错: {e}")
        
        # 保存 SuperPlot 结果
        if has_superplot:
            try:
                sp_data = strat.analyzers.superplot.get_analysis()
                
                # 用我们自定义的带 stock_code 的订单列表覆盖原来的 orders_data
                if hasattr(strat, 'all_orders') and len(strat.all_orders) > 0:
                    if 'orders_data' not in sp_data:
                        sp_data['orders_data'] = {}
                    sp_data['orders_data']['orders'] = strat.all_orders

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
    duckdb_path: str = r'e:\Quant\Quant_Tushare\data\quant_data.duckdb',
    start_date: str = '2020-01-01',
    end_date: str = '2025-12-31',
    benchmark_code: str = '000300.SH',
    strategy_class=bt.Strategy, 
    strategy_params=None,
    initial_cash=100000.0,
    commission=0.001,
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
        duckdb_path: DuckDB 数据库文件路径
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
    
    if set_slippage_perc > 0:
        cerebro.broker.set_slippage_perc(set_slippage_perc)
    
    if strategy_params is None:
        strategy_params = {}
    cerebro.addstrategy(strategy_class, **strategy_params)

    print(f"正在连接 DuckDB: {duckdb_path}")
    try:
        conn = duckdb.connect(duckdb_path, read_only=True)
    except Exception as e:
        print(f"连接 DuckDB 失败: {e}")
        return None, None

    def fetch_data(code):
        query = f"""
            SELECT trade_date as datetime, open, high, low, close, vol as volume
            FROM daily_prices
            WHERE ts_code = '{code}'
              AND trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_date
        """
        try:
            df = conn.execute(query).df()
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
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
            # 使用标准的 PandasData
            data_bench = bt.feeds.PandasData(dataname=df_bench, name='benchmark', timeframe=bt.TimeFrame.Days)
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
            
    conn.close()
    
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
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
    print(f"=== 开始回测 ===")
    print(f"初始资金: {cerebro.broker.get_value():.2f}")
    results = cerebro.run()
    final_value = cerebro.broker.get_value()
    print(f"期末资金: {final_value:.2f}")
    
    if results:
        strat = results[0]
        
        try:
            dd = strat.analyzers.drawdown.get_analysis()
            trade_analyzer = strat.analyzers.tradeanalyzer.get_analysis()
            
            total_ret = 0.0
            if 'pnl' in trade_analyzer and 'net' in trade_analyzer['pnl'] and 'total' in trade_analyzer['pnl']:
                total_pnl = trade_analyzer['pnl']['net']['total']
                total_ret = (total_pnl / initial_cash) * 100
            else:
                total_ret = ((final_value - initial_cash) / initial_cash) * 100
                
            print(f"总收益率: {total_ret:.2f}%")
            print(f"最大回撤: {dd.get('max', {}).get('drawdown', 0):.2f}%")
        except Exception as e:
            print(f"解析基础分析器结果时出错: {e}")
        
        # 保存 SuperPlot 结果
        if has_superplot:
            try:
                sp_data = strat.analyzers.superplot.get_analysis()
                
                # 用自定义带 stock_code 的订单列表覆盖 (如果有)
                if hasattr(strat, 'all_orders') and len(strat.all_orders) > 0:
                    if 'orders_data' not in sp_data:
                        sp_data['orders_data'] = {}
                    sp_data['orders_data']['orders'] = strat.all_orders

                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f'duckdb_bt_superplot.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(sp_data, f, ensure_ascii=False, indent=2)
                print(f"SuperPlot 渲染数据已本地保存至: {out_path}")
            except Exception as e:
                print(f"保存 SuperPlot 数据失败: {e}")
            
    return cerebro, results
