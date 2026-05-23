import os
import sys
import pandas as pd
import yaml

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.data import QlibDataAccessor
from qlworks.features.builder import FeatureBundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.models.training import train_lgb_model, train_xgb_model, train_catboost_model, predict_ensemble_models
import qlib

# ==============================================================================
# [全局配置区]
# ==============================================================================
CONFIG = {
    "instruments": "csi500",  # [Renaissance 改进] 使用 Qlib 动态股票池名称以杜绝前视和幸存者偏差
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",
    # 【Renaissance 级改进】使用滚动窗口进行训练和测试，防止概念漂移和前视偏差
    "rolling_windows": [
        {
            "name": "Test_2023",
            "train": ("2020-01-01", "2021-12-31"), # 2年训练
            "valid": ("2022-01-01", "2022-12-31"), # 1年验证
            "test":  ("2023-01-01", "2023-12-31"), # 1年样本外测试
        },
        {
            "name": "Test_2024",
            "train": ("2021-01-01", "2022-12-31"),
            "valid": ("2023-01-01", "2023-12-31"),
            "test":  ("2024-01-01", "2024-12-31"),
        },
        {
            "name": "Test_2025",
            "train": ("2022-01-01", "2023-12-31"),
            "valid": ("2024-01-01", "2024-12-31"),
            "test":  ("2025-01-01", "2025-12-31"),
        }
    ],
    "top_k_factors": 20 # 我们选取筛选出来的 Top 20 因子来建模
}

def load_factor_expressions(selected_factor_names, factor_files):
    """
    根据给定的因子名称列表，从 YAML 文件中提取对应的 Qlib 表达式
    """
    expr_map = {}
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../factors_repo"))
    for file_name in factor_files:
        yaml_path = os.path.join(repo_path, f"{file_name}.yaml")
        if not os.path.exists(yaml_path): continue
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if not data or 'factors' not in data: continue
        for factor in data['factors']:
            name = factor.get('name')
            expr = factor.get('expression')
            
            # 【修复】：处理 YAML 中 expression 是字典的情况（区分 qlib 和 duckdb 公式）
            if isinstance(expr, dict):
                expr = expr.get('qlib', '')
            
            if name in selected_factor_names and expr:
                expr_map[name] = expr
                
    # 按照输入的顺序重新排列
    ordered_exprs = []
    ordered_names = []
    for name in selected_factor_names:
        if name in expr_map:
            ordered_names.append(name)
            ordered_exprs.append(expr_map[name])
            
    return ordered_exprs, ordered_names

def run_ml_pipeline():
    print("="*60)
    print("=== 第二阶段：多因子机器学习建模与预测 ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=r"e:\Quant\Qlibworks\qlib_data", region="cn", joblib_backend="threading")
    accessor = QlibDataAccessor()
    accessor.ensure_init()
    
    # 2. 读取第一阶段筛选出的高质量因子 (树模型流派)
    print("\n[2] 读取第一阶段筛选出的高质量因子...")
    csv_path = os.path.join(os.path.dirname(__file__), "tree_model_selected_factors.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到因子筛选结果文件: {csv_path}。请先运行 factor_screening_tree.py！")
        
    screened_df = pd.read_csv(csv_path)
    
    # 取前 Top K 个因子
    top_factors_df = screened_df.head(CONFIG["top_k_factors"])
    selected_factor_names = top_factors_df["因子名称"].tolist()
    
    # 从 YAML 中反查对应的 Qlib 公式
    factor_files = ["style_factors", "quality_factors", "price_volume_factors", "sentiment_factors", "risk_factors"]
    selected_factor_exprs, selected_factor_names = load_factor_expressions(selected_factor_names, factor_files)
    
    print(f">>> 成功读取排名前 {len(selected_factor_names)} 的因子，并提取了其公式。")
    
    # 3. 准备 FeatureBundle
    bundle = FeatureBundle(
        fields=selected_factor_exprs,
        names=selected_factor_names,
        # [Citadel Alpha Lab 改进] 标签改为未来 5 天收益，匹配每周调仓
        label_fields=["Ref($close, -5)/$close - 1"],
        label_names=["LABEL_5D"]
    )
    
    all_predictions = []
    
    # 4. 【Renaissance 级改进】遍历所有滚动窗口 (Walk-Forward Optimization)
    for window in CONFIG["rolling_windows"]:
        window_name = window["name"]
        print(f"\n{'='*40}")
        print(f"=== 正在处理滚动窗口: {window_name} ===")
        print(f"    [训练集]: {window['train'][0]} 到 {window['train'][1]}")
        print(f"    [验证集]: {window['valid'][0]} 到 {window['valid'][1]}")
        print(f"    [测试集]: {window['test'][0]} 到 {window['test'][1]}")
        print(f"{'='*40}")
        
        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }
        
        print(f"\n[3-{window_name}] 为该窗口构建专属 DatasetH 数据集...")
        _, dataset = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_bundle=bundle,
            start_time=CONFIG["start_time"],
            end_time=CONFIG["end_time"],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type="tree",             # 集成模型全是树模型
            neutralize_features=False,     # 树模型关闭特征中性化
            neutralize_labels=True         # 仅开启标签中性化
        )
        print(f">>> {window_name} 训练/验证/测试集 切分完成！")

        print(f"\n[4-{window_name}] 开始训练机器学习模型 (LGBM + XGBoost + CatBoost) [GPU加速]...")
        print("    - 正在训练 LightGBM 模型 (GPU)...")
        lgb_model = train_lgb_model(dataset)
        
        print("    - 正在训练 XGBoost 模型 (GPU)...")
        xgb_model = train_xgb_model(dataset)
        
        print("    - 正在训练 CatBoost 模型 (GPU)...")
        cat_model = train_catboost_model(dataset)
        print(f">>> {window_name} 所有模型训练完毕！")

        print(f"\n[5-{window_name}] 在测试集上进行模型集成与预测 (生成 Alpha 预测得分)...")
        predictions = predict_ensemble_models([lgb_model, xgb_model, cat_model], dataset, segment="test")
        
        # 将原始得分进行横截面百分位排序 (Cross-Sectional Ranking)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True)
        
        print(f">>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)
    
    # 5. 合并所有滚动窗口的样本外预测结果
    print("\n[6] 所有滚动窗口执行完毕！正在合并预测结果...")
    final_predictions = pd.concat(all_predictions)
    # 按时间排序，确保回测顺序正确
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)
    
    print(f">>> 合并完成！总测试集跨度: {final_predictions.index.get_level_values('datetime').min().date()} 至 {final_predictions.index.get_level_values('datetime').max().date()}")
    print("    【预测排名 (Score) 抽样展示】(1.0代表当天全市场最强):")
    print(final_predictions.head(10))
    
    # 6. 保存预测结果，为 Backtrader 回测做准备
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