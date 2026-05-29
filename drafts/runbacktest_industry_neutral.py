import os
import sys
import pandas as pd

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.config import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
from qlworks.data import QlibDataAccessor
from qlib.data import D
from qlworks.backtest.bt_runner import run_qlib_backtrader
from qlworks.backtest.bt_strategy import AShareStrategy # [Virtu/Man Group 改进] 使用风控增强版的 A股策略

def optimize_portfolio_scores(pred_df, top_k_total=10, max_per_industry=2):
    print("\n=== [Man Group] 组合优化与敞口控制 (行业中性化) ===")
    
    # 1. Load industry mapping
    print("    正在从 ClickHouse 加载股票行业映射数据...")
    import clickhouse_connect
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST,
            port=CH_PORT,
            user=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE
        )
        stock_info = client.query_df("SELECT ts_code, l1_name as industry FROM sw_industry_members")
    except Exception as e:
        print(f"    连接 ClickHouse 失败: {e}。跳过行业敞口约束！")
        return pred_df
    
    # Map instrument to industry (ts_code: 000001.SZ -> qlib: SZ000001)
    def to_qlib_symbol(ts_code):
        if not isinstance(ts_code, str):
            return ts_code
        parts = ts_code.split('.')
        if len(parts) == 2:
            return parts[1] + parts[0]
        return ts_code
    
    stock_info['instrument'] = stock_info['ts_code'].apply(to_qlib_symbol)
    industry_map = dict(zip(stock_info['instrument'], stock_info['industry']))
    
    # reset index to manipulate columns
    df = pred_df.reset_index()
    df['industry'] = df['instrument'].map(industry_map).fillna('Unknown')
    
    print(f"    执行行业分组约束: 总持仓不超过 {top_k_total} 只, 单一行业最多选取 {max_per_industry} 只。")
    
    # 2. Apply constraints
    optimized_records = []
    
    for dt, group in df.groupby('datetime'):
        # 降序排列得分
        group = group.sort_values('score', ascending=False)
        
        selected = []
        industry_counts = {}
        
        for _, row in group.iterrows():
            ind = row['industry']
            if len(selected) >= top_k_total:
                break
                
            if industry_counts.get(ind, 0) < max_per_industry:
                selected.append(row)
                industry_counts[ind] = industry_counts.get(ind, 0) + 1
                
        optimized_records.extend(selected)
        
    optimized_df = pd.DataFrame(optimized_records)
    # set index back
    optimized_df.set_index(['datetime', 'instrument'], inplace=True)
    # remove the temporary industry column so it doesn't break backtrader expectations
    optimized_df.drop(columns=['industry'], inplace=True)
    
    print(f"    优化完成! 原始打分记录: {len(df)} 条, 约束后过滤出目标入场记录: {len(optimized_df)} 条。")
    return optimized_df

def main():
    print("="*60)
    print("=== 第三阶段：真实交易回测与资金曲线 (含行业敞口控制) ===")
    print("="*60)
    
    # 1. 初始化 Qlib (获取价格数据需要)
    print("[1] 初始化 Qlib 环境...")
    accessor = QlibDataAccessor()
    accessor.ensure_init()
    
    # 2. 读取预测分数
    print("\n[2] 读取模型预测得分 (ml_predictions_score.csv)...")
    pred_path = os.path.join(os.path.dirname(__file__), "ml_predictions_score.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"找不到预测文件: {pred_path}。请先运行第二阶段建模！")
        
    pred_df = pd.read_csv(pred_path)
    pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
    pred_df.set_index(['datetime', 'instrument'], inplace=True)
    
    # 获取测试集的日期范围和股票池
    start_date = pred_df.index.get_level_values('datetime').min()
    end_date = pred_df.index.get_level_values('datetime').max()
    instruments = pred_df.index.get_level_values('instrument').unique().tolist()
    
    print(f"    测试集时间段: {start_date.date()} 至 {end_date.date()}")
    print(f"    股票池数量: {len(instruments)}")
    
    # [Man Group] 在这里插入行业敞口约束
    print("\n[2.1] 获取股票行业信息...")
    # 优先尝试从 Qlib 中获取行业信息
    industry_map = {}
    try:
        print("    尝试从 Qlib 中提取 $industry 字段...")
        # 随便取测试集中的一天来获取行业静态数据
        sample_date = start_date.strftime("%Y-%m-%d")
        ind_data = D.features(instruments, ['$industry'], start_time=sample_date, end_time=sample_date)
        if not ind_data.empty:
            for inst in instruments:
                if inst in ind_data.index.get_level_values('instrument'):
                    val = ind_data.xs(inst, level='instrument')['$industry'].iloc[0]
                    industry_map[inst] = val
            print(f"    成功从 Qlib 提取到 {len(industry_map)} 只股票的行业信息。")
    except Exception as e:
        print(f"    从 Qlib 提取行业信息失败 ({e})。")
        
    # 如果 Qlib 失败或不全，回退到 ClickHouse
    if len(industry_map) < len(instruments) * 0.5:  # 容错率
        print("    Qlib 行业信息不足，尝试回退到 ClickHouse...")
        pred_df = optimize_portfolio_scores(pred_df, top_k_total=10, max_per_industry=2)
    else:
        # 使用 Qlib 的行业信息进行约束
        df = pred_df.reset_index()
        df['industry'] = df['instrument'].map(industry_map).fillna('Unknown')
        
        print(f"    执行行业分组约束: 总持仓不超过 10 只, 单一行业最多选取 2 只。")
        optimized_records = []
        for dt, group in df.groupby('datetime'):
            group = group.sort_values('score', ascending=False)
            selected = []
            ind_counts = {}
            for _, row in group.iterrows():
                if len(selected) >= 10: break
                ind = row['industry']
                if ind_counts.get(ind, 0) < 2:
                    selected.append(row)
                    ind_counts[ind] = ind_counts.get(ind, 0) + 1
            optimized_records.extend(selected)
            
        pred_df = pd.DataFrame(optimized_records).set_index(['datetime', 'instrument'])
        pred_df.drop(columns=['industry'], inplace=True)
        print(f"    优化完成! 原始打分记录: {len(df)} 条, 约束后过滤出目标入场记录: {len(pred_df)} 条。")
    
    # 3. 从 Qlib 拉取对应的行情数据 (OHLCV)
    print("\n[3] 从 Qlib 拉取真实行情数据 (用于回测撮合)...")
    fields = ['$open', '$high', '$low', '$close', '$volume']
    price_data = D.features(instruments, fields, start_time=start_date, end_time=end_date)
    price_data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    price_df_dict = {}
    for inst in instruments:
        if inst in price_data.index.get_level_values('instrument'):
            df = price_data.xs(inst, level='instrument').copy()
            # [Renaissance 改进] 幸存者偏差防范。截断退市后的冗余数据
            df.dropna(subset=['close'], inplace=True)
            if not df.empty:
                valid_volume = df[df['volume'] > 0]
                if not valid_volume.empty:
                    last_valid_date = valid_volume.index[-1]
                    df = df[df.index <= last_valid_date]
                price_df_dict[inst] = df
                
    print(f"    成功拉取 {len(price_df_dict)} 只股票的行情。")

    # 拉取中证500指数作为基准 (Benchmark)
    print("    正在拉取 000905.SH (中证500) 作为超额收益对比基准...")
    try:
        bench_data = D.features(['SH000905'], fields, start_time=start_date, end_time=end_date)
        if not bench_data.empty:
            bench_df = bench_data.xs('SH000905', level='instrument').copy()
            bench_df.columns = ['open', 'high', 'low', 'close', 'volume']
            print("    成功拉取基准数据！")
        else:
            bench_df = None
            print("    警告: 无法拉取到 SH000905 数据，将不使用基准。")
    except Exception as e:
        bench_df = None
        print(f"    警告: 无法拉取到 SH000905 数据 ({e})，将不使用基准。")
    
    # 4. 运行 Backtrader 回测
    print("\n[4] 启动 Backtrader 引擎进行交易撮合...")
    strategy_params = dict(
        top_k=10,               # 最多持仓数量
        score_threshold=0.7,    # 选股得分阈值
        buy_pct=0.95,           # 最大资金使用率
        rebalance_days=5,       # [A股优化] 换仓周期，默认5个交易日(一周)
        daily_sell_enabled=True, # 开启非换仓日每天盘中风控检查
        stop_loss_pct=-0.08,     # 盘中硬止损: 入场后跌幅超过 8%
        volume_limit_pct=0.10,  # [Virtu 改进] 单笔订单不能超过当日成交量的10%，防止吃不掉流动性
        log_enabled=False,      # 关闭日志输出，加速回测
    )

    # 调用现成的 run_qlib_backtrader
    cerebro, results = run_qlib_backtrader(
        pred_df=pred_df,
        price_df_dict=price_df_dict,
        benchmark_df=bench_df,           # 加载中证500基准
        strategy_class=AShareStrategy,   # [Virtu/Man Group 改进] 使用风控增强版的 A股策略
        strategy_params=strategy_params,
        initial_cash=1000000.0,
        commission=0.0,          # 佣金已由 AShareCommission 内部处理
        set_slippage_perc=0.001, # 千分之一滑点 (模拟买卖损耗)
        output_dir=os.path.dirname(__file__),
        start_date=start_date,
        end_date=end_date
    )

if __name__ == "__main__":
    main()
