from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
import qlib
from qlib.data import D

def load_csi500_instruments():
    file_path = r"e:\Quant\Qlibworks\qlib_data\instruments\csi500.txt"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=['instrument', 'start_date', 'end_date'])
        # 过滤出 2020-01-02 之前上市的股票，以保证数据完整性
        df = df[df['start_date'] <= '2020-01-02'].drop_duplicates(subset=['instrument'])
        return df['instrument'].tolist()
    return ["000001.SZ", "000002.SZ", "600000.SH"]

# ==============================================================================
# [全局配置区] 集中管理研究工作流的所有参数
# ==============================================================================
CONFIG = {
    # 1. 股票池与时间范围
    "instruments": load_csi500_instruments(),
    "start_time": "2020-01-02",
    "end_time": "2020-12-31",
    
    # 2. 数据集切分 (训练/验证/测试)
    "segments": {
        "train": ("2020-01-02", "2020-06-30"),
        "valid": ("2020-07-01", "2020-09-30"),
        "test":  ("2020-10-01", "2020-12-31"),
    },
    
    # 3. 特征选择参数
    "feature_selection": {
        "enable": True,
        "method": "embedded",   # filter / wrapper / embedded
        "algo": "random_forest", # 换用 RF，因为 lasso 容易把因子全部砍掉导致模型输入全是常数
        "threshold": 0.0001,    # 降低稀疏阈值，保留更多因子
        "label_col": "LABEL0",
    },
    
    # 4. 模型训练与调优参数
    "model_tuning": {
        "use_optuna": False,    # 是否开启贝叶斯寻优 (实盘建议 True)
        "n_trials": 5,          # 寻优迭代次数
    },
    
    # 5. Barra 组合优化参数
    "portfolio": {
        "enable": True,
        "target_volatility": 0.30, # 目标年化波动率 (A股中证500个股波动率通常在25%~40%，30%是较合理的组合优化上限)
        "max_weight": 0.10,        # 个股最大仓位占比
    },
    
    # 6. Backtrader 回测参数
    "backtest": {
        "enable": True,
        "top_k": 3,              # 每次调仓买入前 N 只股票
        "rebalance_days": 5,     # 调仓周期 (天)
        "initial_cash": 100000.0,
        "output_dir": "./bt_output",
        "commission": 0.001,     # A股佣金及印花税综合费率 (0.1%)
        "slippage": 0.001,       # A股滑点比例 (0.1%)
        "use_risk_control": True,# 是否启用风控增强模式
        "stop_type": "ATR",      # 止损类型
        "atr_multiplier": 2.0,   # ATR 止损倍数
        "trailing_stop": True,   # 是否启用移动止盈
        "trailing_start_pct": 0.10,    # 盈利超过 10% 启动移动止盈
        "trailing_callback_pct": 0.02, # 回撤 2% 止盈
        "take_profit_pct": 0.5,  # 触发止盈时的平仓比例 (0.5为平一半)
    }
}
# ==============================================================================

# 从项目库直接引用独立组件
from qlworks.data import DataFetchSpec, QlibDataAccessor, clean_ohlcv_data, generate_data_quality_report
from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.models import (
    apply_feature_selection,
    evaluate_prediction_frame,
    prepare_feature_selection_data,
    select_features,
    train_lgb_model,
    tune_lgbm_hyperparameters,
    optimize_portfolio,
)
from qlworks.backtest import run_qlib_backtrader, EnhancedQlibStrategy


def run_quant_research_workflow():
    """
    功能概述：
    - 直接串联 Qlibworks 底层各独立组件，执行量化研究全流程。
    - 摒弃厚重的类封装，流程完全透明。
    """
    print("=== 开始执行多因子量化研究全流程 ===")
    
    # 0. 初始化环境
    print("\n[0] 初始化 Qlib 环境...")
    accessor = QlibDataAccessor()
    accessor.ensure_init()
    
    # 1. 特征解析与数据拉取
    print("\n[1] 解析特征表达式与拉取原始数据...")
    # 使用新构建的五大因子库
    factor_files = [
        "style_factors", 
        "quality_factors", 
        "price_volume_factors", 
        "sentiment_factors", 
        "risk_factors"
    ]
    bundle = build_factor_library_bundle(factor_files)
    
    # 额外拉取行情数据供后续回测和组合优化使用
    ohlcv_fields = ["$open", "$high", "$low", "$close", "$volume"]
    ohlcv_names = ["open", "high", "low", "close", "volume"]
    
    bundle_fields_with_price = list(bundle.fields) + ohlcv_fields
    bundle_names_with_price = list(bundle.names) + ohlcv_names
    
    spec = DataFetchSpec(
        instruments=CONFIG["instruments"],
        fields=bundle_fields_with_price,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
    )
    
    raw_data = accessor.fetch_feature_label_frame(
        feature_spec=spec,
        label_fields=bundle.label_fields,
        label_names=bundle.label_names,
    )
    # 统一列名
    raw_data.columns = bundle_names_with_price + list(bundle.label_names)
    print(f"- 原始数据维度: {raw_data.shape}")
    
    # 2. 数据清洗与质量评估
    print("\n[2] 数据清洗与体检报告...")
    clean_data = clean_ohlcv_data(raw_data)
    quality_report = generate_data_quality_report(clean_data)
    print(f"- 综合质量得分: {quality_report['overall_score']:.4f}")
    
    # 3. 构建 DatasetH
    print("\n[3] 构建 DatasetH (特征工程/中性化/去极值)...")
    _, dataset = create_custom_dataset(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        fit_start_time=CONFIG["segments"]["train"][0],
        fit_end_time=CONFIG["segments"]["train"][1],
        segments=CONFIG["segments"]
    )
    
    # 4. 特征选择 (降维)
    selected_features = list(bundle.names)
    if CONFIG["feature_selection"]["enable"]:
        print("\n[4] 执行特征选择...")
        fs_conf = CONFIG["feature_selection"]
        
        train_frame = dataset.prepare("train")
        test_frame = dataset.prepare("test")
        
        x_train, y_train, x_test = prepare_feature_selection_data(
            train_frame=train_frame,
            test_frame=test_frame,
            label_col=fs_conf["label_col"]
        )
        
        # 执行选择算法
        fs_result = select_features(
            x_train, y_train, 
            method=fs_conf["method"], 
            algo=fs_conf["algo"], 
            threshold=fs_conf["threshold"]
        )
        
        selected_features = list(fs_result.selected_features)
        print(f"- 从 {len(bundle.names)} 个因子中挑选出 {len(selected_features)} 个有效因子:")
        print(selected_features)
        
        # 为了演示流水线连续性，在此处进行 DataFrame 切片
        # (工程最佳实践：重写 dataset handler 或重新生成 dataset)
        keep_cols = selected_features + [c for c in train_frame.columns if "LABEL" in c]
        # (后续送给模型的 dataset 虽然庞大，但在内部我们会提取 selected_features，
        # 此处展示降维名单已获取，不实际破坏 Qlib 底层 Dataset)
        
    # 5. 模型调优与训练
    print("\n[5] 模型训练与评估...")
    best_params = None
    if CONFIG["model_tuning"]["use_optuna"]:
        print("- 启动 Optuna 自动超参寻优...")
        best_params = tune_lgbm_hyperparameters(dataset, n_trials=CONFIG["model_tuning"]["n_trials"])
        
    print("- 训练 LightGBM...")
    model = train_lgb_model(dataset, **(best_params or {}))
    
    # 模型预测
    pred = model.predict(dataset, segment="test")
    test_frame = dataset.prepare("test")
    
    pred_frame = pd.DataFrame(
        {"pred": pred.values.flatten(), "label": test_frame[CONFIG["feature_selection"]["label_col"]].values},
        index=test_frame.index,
    ).dropna()
    
    # 模型体检
    evaluation = evaluate_prediction_frame(pred_frame)
    print("\n=== 模型体检报告 ===")
    print(f"- 每日 IC 均值: {evaluation['ic_mean']:.4f}")
    print(f"- 每日 Rank IC 均值: {evaluation['rank_ic_mean']:.4f}")
    
    # 6. Barra 组合优化
    if CONFIG["portfolio"]["enable"]:
        print("\n[6] Barra 均值-方差组合优化...")
        # 提取历史价格数据给组合优化用 (利用 raw_data 中的 close)
        prices_pivot = raw_data['close'].unstack('instrument')
        
        try:
            latest_date = pred_frame.index.get_level_values('datetime').max()
            latest_preds = pred_frame.xs(latest_date, level='datetime')['pred']
            
            portfolio_weights = optimize_portfolio(
                prices_df=prices_pivot,
                predictions=latest_preds,
                target_volatility=CONFIG["portfolio"]["target_volatility"],
                max_weight=CONFIG["portfolio"]["max_weight"],
            )
            print("- 最新截面持仓权重 (部分):")
            print(portfolio_weights[portfolio_weights > 0].sort_values(ascending=False).head())
        except Exception as e:
            print(f"[!] 组合优化失败: {e}")
            
    # 7. 回测与效果分析
    if CONFIG["backtest"]["enable"]:
        print("\n[7] 启动 Backtrader 历史回测...")
        pred_df = pred.to_frame(name="score")
        
        # 准备价格字典供 BT 使用
        price_df_dict = {}
        # 为了避免回测引擎在没有预测分的时期空转，提取测试集开始之后的行情数据
        bt_start_date = pd.to_datetime(CONFIG["segments"]["test"][0])
        for inst in CONFIG["instruments"]:
            if inst in raw_data.index.get_level_values('instrument'):
                df_inst = raw_data.xs(inst, level='instrument').copy()
                # 仅截取从回测起始日开始的行情，提前留一点预热期(例如提前30天)保证BT指标计算
                df_inst = df_inst[df_inst.index >= (bt_start_date - pd.Timedelta(days=30))]
                if not df_inst.empty:
                    price_df_dict[inst] = df_inst
                
        bt_conf = CONFIG["backtest"]
        cerebro, results = run_qlib_backtrader(
            pred_df=pred_df,
            price_df_dict=price_df_dict,
            strategy_class=EnhancedQlibStrategy,
            strategy_params={
                "top_k": bt_conf["top_k"], 
                "rebalance_days": bt_conf["rebalance_days"],
                "use_risk_control": bt_conf.get("use_risk_control", False),
                "stop_type": bt_conf.get("stop_type", "ATR"),
                "atr_multiplier": bt_conf.get("atr_multiplier", 2.0),
                "trailing_stop": bt_conf.get("trailing_stop", False),
                "trailing_start_pct": bt_conf.get("trailing_start_pct", 0.10),
                "trailing_callback_pct": bt_conf.get("trailing_callback_pct", 0.02),
                "take_profit_pct": bt_conf.get("take_profit_pct", 1.0)
            },
            initial_cash=bt_conf["initial_cash"],
            commission=bt_conf.get("commission", 0.001),
            set_slippage_perc=bt_conf.get("slippage", 0.001),
            output_dir=bt_conf["output_dir"]
        )
        print("- 回测完成，报告已保存。")


if __name__ == "__main__":
    run_quant_research_workflow()
