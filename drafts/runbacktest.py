import os
import sys
import pandas as pd

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.data import QlibDataAccessor
from qlib.data import D
from qlworks.backtest.bt_runner import run_qlib_backtrader
from qlworks.backtest.bt_strategy import BaseQlibStrategy

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
            # 清理缺失值，保证能跑通回测
            df.dropna(subset=['close'], inplace=True)
            if not df.empty:
                price_df_dict[inst] = df
                
    print(f"    成功拉取 {len(price_df_dict)} 只股票的行情。")
    
    # 4. 运行 Backtrader 回测
    print("\n[4] 启动 Backtrader 引擎进行交易撮合...")
    strategy_params = dict(
        top_k=5,             # 每天买入预测分数最高的 5 只股票
        rebalance_days=1,    # 调仓周期: 1天 (每日换仓，考验因子短期预测能力)
        buy_pct=0.95,        # 最大资金使用率 95% (留5%现金防滑点)
        log_enabled=False    # 关闭详细的买卖日志以保持终端整洁
    )
    
    # 调用现成的 run_qlib_backtrader
    cerebro, results = run_qlib_backtrader(
        pred_df=pred_df,
        price_df_dict=price_df_dict,
        benchmark_df=None,               # 暂不使用对比基准
        strategy_class=BaseQlibStrategy, # 使用基础多头截面策略
        strategy_params=strategy_params,
        initial_cash=1000000.0,
        commission=0.001,      # 千分之一手续费 (含印花税)
        set_slippage_perc=0.001, # 千分之一滑点 (模拟买卖损耗)
        output_dir=os.path.dirname(__file__)
    )

if __name__ == "__main__":
    main()