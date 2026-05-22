import os
import sys
import pandas as pd
import yaml
import gc # [Bloomberg Eng] 添加 gc 模块进行内存回收

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.data import QlibDataAccessor
from qlworks.features.builder import build_factor_library_bundle, FeatureBundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.models.training import train_lgb_model, train_xgb_model, train_catboost_model, predict_ensemble_models
from qlworks.models import prepare_feature_selection_data, select_features
import qlib

# ==============================================================================
# [全局配置区]
# ==============================================================================
def load_csi500_instruments():
    file_path = r"e:\Quant\Qlibworks\qlib_data\instruments\csi500.txt"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=['instrument', 'start_date', 'end_date'], dtype={'instrument': str})
        insts = df['instrument'].dropna().unique().tolist()
        return insts
    return ["000001.SZ", "000002.SZ", "600000.SH"]

CONFIG = {
    "instruments": load_csi500_instruments(),
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",
    # 【Renaissance 级改进】使用滚动窗口进行训练和测试，防止概念漂移和前视偏差
    # [Point] 引入 Embargo (隔离期)：训练集到验证集、验证集到测试集之间留出约10天安全垫，防止 T+N 标签导致的未来数据泄漏 (Look-ahead Bias)
    "rolling_windows": [
        {
            "name": "Test_2023",
            "train": ("2020-01-01", "2021-12-20"), # 提前结束，留出Embargo
            "valid": ("2022-01-01", "2022-12-20"), # 提前结束
            "test":  ("2023-01-01", "2023-12-31"), # 1年样本外测试
        },
        {
            "name": "Test_2024",
            "train": ("2021-01-01", "2022-12-20"),
            "valid": ("2023-01-01", "2023-12-20"),
            "test":  ("2024-01-01", "2024-12-31"),
        },
        {
            "name": "Test_2025",
            "train": ("2022-01-01", "2023-12-20"),
            "valid": ("2024-01-01", "2024-12-20"),
            "test":  ("2025-01-01", "2025-12-31"),
        }
    ],
    "top_k_factors": 20, # 我们选取筛选出来的 Top 20 因子来建模
    "feature_selection": {
        "method": "embedded",   
        "algo": "lightgbm",     
        "label_col": "LABEL0",  
        "remove_collinearity": False,
    }
}

def run_ml_pipeline():
    print("="*60)
    print("=== 第二阶段：【终极改造】动态因子筛选与多因子机器学习建模 ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=r"e:\Quant\Qlibworks\qlib_data", region="cn", joblib_backend="threading")
    accessor = QlibDataAccessor()
    accessor.ensure_init()
    
    # 2. 从 YAML 中一次性拉取因子库中所有的因子
    print("\n[2] 读取因子库 (Factor Library) 的所有因子公式...")
    factor_files = ["style_factors", "quality_factors", "price_volume_factors", "sentiment_factors", "risk_factors"]
    bundle_all = build_factor_library_bundle(factor_files)
    print(f">>> 成功加载 {len(bundle_all.fields)} 个因子候选池。")
    
    all_predictions = []
    fs_conf = CONFIG["feature_selection"]
    
    # 3. 【终极架构】遍历所有滚动窗口 (Walk-Forward Optimization with Dynamic Factor Selection)
    for window in CONFIG["rolling_windows"]:
        window_name = window["name"]
        print(f"\n{'='*60}")
        print(f"=== 正在处理滚动窗口: {window_name} ===")
        print(f"    [训练集/选因子]: {window['train'][0]} 到 {window['train'][1]}")
        print(f"    [验证集/调早停]: {window['valid'][0]} 到 {window['valid'][1]}")
        print(f"    [测试集/纯盲测]: {window['test'][0]} 到 {window['test'][1]}")
        print(f"{'='*60}")
        
        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }
        
        # 3.1 为该窗口构建包含【全量因子】的 DatasetH，用于挑选因子
        print(f"\n[3.1 - {window_name}] 构建全量因子的 DatasetH (用于特征筛选)...")
        # [Point72 改进] 仅提取当前 Window 所需时间段，避免每次循环计算全量(2020-2025)数据的巨大算力浪费
        _, dataset_full = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_bundle=bundle_all,
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type="tree",
            neutralize_features=False, # 若出现严重截面偏移，可设为 True
            neutralize_labels=True
        )
        
        # 3.2 在训练集上进行因子筛选
        print(f"\n[3.2 - {window_name}] 在当前训练集上执行动态因子筛选 (选取前 {CONFIG['top_k_factors']} 个)...")
        train_frame_full = dataset_full.prepare("train")
        x_train, y_train, _ = prepare_feature_selection_data(train_frame_full, label_col=fs_conf["label_col"])
        
        valid_idx = y_train.dropna().index
        x_train = x_train.loc[valid_idx]
        y_train = y_train.loc[valid_idx]
        
        fs_result = select_features(
            x_train, y_train, 
            method=fs_conf["method"], 
            algo=fs_conf["algo"], 
            threshold=0.0,
            model_kwargs={"max_features": CONFIG["top_k_factors"], "importance_type": "gain"},
            remove_collinearity=fs_conf["remove_collinearity"]
        )
        
        selected_factor_names = fs_result.selected_features
        print(f">>> {window_name} 动态因子筛选完成！本期入选的因子为:")
        print(f"    {selected_factor_names}")
        
        # [Bloomberg Data Pipeline 改进] 内存核弹危机解除：因子筛选完成后，全量数百个因子的庞大 DataFrame 必须立即销毁
        del dataset_full, train_frame_full, x_train, y_train, fs_result
        gc.collect()
        
        # 3.3 构建仅包含 Top K 因子的小数据集，防止过多无用特征干扰模型
        # 由于 dataset_full 的底层数据是通用的，我们可以利用它的 _data 结构，或者直接用切片，
        # 为了符合 Qlib 的原生训练模式，这里直接传入全量 dataset_full 也可以，
        # 因为在树模型中，被选出的 max_features=20 已经在算法内做了限制，
        # 但为了更清晰的隔离，我们通过重新生成一个小 dataset 来保证干净：
        
        # 从 bundle_all 中提取选中的因子公式
        expr_map = dict(zip(bundle_all.names, bundle_all.fields))
        selected_exprs = [expr_map[name] for name in selected_factor_names]
        
        bundle_sub = FeatureBundle(
            fields=selected_exprs,
            names=selected_factor_names,
            label_fields=bundle_all.label_fields,
            label_names=bundle_all.label_names
        )
        
        print(f"\n[3.3 - {window_name}] 根据选出的因子重构轻量级 DatasetH...")
        _, dataset_sub = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_bundle=bundle_sub,
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type="tree",
            neutralize_features=False,
            neutralize_labels=True
        )

        print(f"\n[3.4 - {window_name}] 开始训练机器学习模型 (LGBM + XGBoost + CatBoost) [GPU加速]...")
        print("    - 正在训练 LightGBM 模型 (GPU)...")
        lgb_model = train_lgb_model(dataset_sub)
        
        print("    - 正在训练 XGBoost 模型 (GPU)...")
        xgb_model = train_xgb_model(dataset_sub)
        
        print("    - 正在训练 CatBoost 模型 (GPU)...")
        cat_model = train_catboost_model(dataset_sub)
        print(f">>> {window_name} 所有模型训练完毕！")

        print(f"\n[3.5 - {window_name}] 在测试集上进行模型集成与预测 (生成 Alpha 预测得分)...")
        predictions = predict_ensemble_models([lgb_model, xgb_model, cat_model], dataset_sub, segment="test")
        
        # 将原始得分进行横截面百分位排序 (Cross-Sectional Ranking)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
            
        # [Two Sigma 改进] 严格处理 NaN 值（停牌、缺失等），防止影响横截面排序结果
        predictions = predictions.dropna(subset=["score"])
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")
        
        print(f">>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)
        
        # [Bloomberg Data Pipeline 改进] 当前 Window 结束，彻底释放轻量数据集与模型占用的显存/内存
        del dataset_sub, lgb_model, xgb_model, cat_model
        gc.collect()
    
    # 4. 合并所有滚动窗口的样本外预测结果
    print("\n[4] 所有滚动窗口执行完毕！正在合并预测结果...")
    final_predictions = pd.concat(all_predictions)
    # 按时间排序，确保回测顺序正确
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)
    
    print(f">>> 合并完成！总测试集跨度: {final_predictions.index.get_level_values('datetime').min().date()} 至 {final_predictions.index.get_level_values('datetime').max().date()}")
    print("    【预测排名 (Score) 抽样展示】(1.0代表当天全市场最强):")
    print(final_predictions.head(10))
    
    # 5. 保存预测结果，为 Backtrader 回测做准备
    output_path = os.path.join(os.path.dirname(__file__), "ml_predictions_score.csv")
    final_predictions.to_csv(output_path)
    print(f"\n>>> 预测得分已保存至: {output_path}")
    print("="*60)
    print("【下一步指引】")
    print("现在您已经有了每只股票每天的预测得分 (Score)。")
    print("下一步就是将这个 Score 喂给 Backtrader (src/qlworks/backtest/bt_runner.py)。")
    print("Backtrader 会模拟真实的交易环境：每天买入 Score 最高的 N 只股票，计算扣除印花税、滑点后的真实收益和换手率！")
    print("="*60)

if __name__ == "__main__":
    run_ml_pipeline()
