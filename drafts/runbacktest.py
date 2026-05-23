import os
import sys
import pandas as pd

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.data import QlibDataAccessor
from qlib.data import D
from qlworks.backtest.bt_runner import run_qlib_backtrader
from qlworks.backtest.bt_strategy import EnhancedQlibStrategy # [Virtu/Man Group 改进] 使用风控增强版的 A股策略

def main():
    print("="*60)
    print("=== 第三阶段：真实交易回测与资金曲线 ===")
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
        top_k=10,            # 持仓数量（分散到10只）
        score_threshold=0.7, # 仅买入分数 > 0.7 的股票
        buy_pct=0.95,        # 最大资金使用率 95% (留5%现金防滑点)
        rebalance_days=5,    # [Man Group 改进] 每 5 个交易日（一周）换一次仓
        use_risk_control=True, # 开启风控增强模式
        stop_type='ATR',       # 止损类型 (EnhancedQlibStrategy特有)
        atr_period=14,         # ATR周期
        atr_multiplier=2.0,    # ATR止损倍数
        trailing_stop=True,    # 开启移动止盈
        score_drop_threshold=0.3, # 分数恶化平仓: 模型得分跌破 0.3
        log_enabled=True,    # 开启日志以查看真实的挂单与止损执行情况
    )

    # 调用现成的 run_qlib_backtrader
    cerebro, results = run_qlib_backtrader(
        pred_df=pred_df,
        price_df_dict=price_df_dict,
        benchmark_df=bench_df,           # 加载中证500基准
        strategy_class=EnhancedQlibStrategy, # 使用风控增强版的 A股策略
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