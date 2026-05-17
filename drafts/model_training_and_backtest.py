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
def load_csi500_instruments():
    file_path = r"e:\Quant\Qlibworks\qlib_data\instruments\csi500.txt"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=['instrument', 'start_date', 'end_date'], dtype={'instrument': str})
        insts = df['instrument'].dropna().unique().tolist()
        return insts
    return ["000001.SZ", "000002.SZ", "600000.SH"]

CONFIG = {
    "instruments": load_csi500_instruments(),
    "start_time": "2023-01-01",
    "end_time": "2025-12-31",
    "segments": {
        "train": ("2023-01-01", "2023-12-31"),
        "valid": ("2024-01-01", "2024-12-31"),
        "test":  ("2025-01-01", "2025-12-31"),
    },
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
    
    # 3. 重新构建专属的 DatasetH (只包含这 Top K 个好因子)
    print("\n[3] 为机器学习模型构建专属 DatasetH 数据集...")
    bundle = FeatureBundle(
        fields=selected_factor_exprs,
        names=selected_factor_names,
        label_fields=["Ref($close, -2)/Ref($close, -1) - 1"],
        label_names=["LABEL0"]
    )
    
    _, dataset = create_custom_dataset(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        fit_start_time=CONFIG["segments"]["train"][0],
        fit_end_time=CONFIG["segments"]["train"][1],
        segments=CONFIG["segments"],
        model_type="tree",             # 集成模型全是树模型
        neutralize_features=False,     # 【Point72修正】树模型关闭特征中性化
        neutralize_labels=True         # 【Point72修正】仅开启标签中性化
    )
    print(">>> 训练/验证/测试集 切分完成 (已完成标签截面中性化)！")

    # 4. 训练集成机器学习模型
    # 【量化逻辑】单个模型容易过拟合（比如只背题不懂变通）。
    # 业界标配是：LightGBM + XGBoost + CatBoost 三大树模型融合（Ensemble）。
    print("\n[4] 开始训练机器学习模型 (LGBM + XGBoost + CatBoost) [GPU加速]...")
    
    print("    - 正在训练 LightGBM 模型 (GPU)...")
    lgb_model = train_lgb_model(dataset)
    
    print("    - 正在训练 XGBoost 模型 (GPU)...")
    xgb_model = train_xgb_model(dataset)
    
    print("    - 正在训练 CatBoost 模型 (GPU)...")
    cat_model = train_catboost_model(dataset)
    
    print(">>> 所有模型训练完毕！")

    # 5. 模型集成与样本外预测
    print("\n[5] 在测试集上进行模型集成与预测 (生成 Alpha 预测得分)...")
    # 把三个模型的预测结果等权平均，得到最终的综合打分 (score)
    predictions = predict_ensemble_models([lgb_model, xgb_model, cat_model], dataset, segment="test")
    
    print(f">>> 预测完成！测试集共产生 {len(predictions)} 条预测得分。")
    print("    【预测得分 (Score) 抽样展示】(Score越高，代表模型认为该股票明天越能涨):")
    print(predictions.head(10))
    
    # 6. 保存预测结果，为 Backtrader 回测做准备
    output_path = os.path.join(os.path.dirname(__file__), "ml_predictions_score.csv")
    predictions.to_csv(output_path)
    print(f"\n>>> 预测得分已保存至: {output_path}")
    print("="*60)
    print("【下一步指引】")
    print("现在您已经有了每只股票每天的预测得分 (Score)。")
    print("下一步就是将这个 Score 喂给 Backtrader (src/qlworks/backtest/bt_runner.py)。")
    print("Backtrader 会模拟真实的交易环境：每天买入 Score 最高的 N 只股票，计算扣除印花税、滑点后的真实收益和换手率！")
    print("="*60)

if __name__ == "__main__":
    run_ml_pipeline()