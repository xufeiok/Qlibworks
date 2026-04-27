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

# ==============================================================================
# [全局配置区]
# ==============================================================================
def load_csi500_instruments():
    file_path = r"e:\Quant\Qlibworks\qlib_data\instruments\csi500.txt"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=['instrument', 'start_date', 'end_date'])
        df = df[df['start_date'] <= '2020-01-02'].drop_duplicates(subset=['instrument'])
        return df['instrument'].tolist()
    return ["000001.SZ", "000002.SZ", "600000.SH"]

CONFIG = {
    "instruments": load_csi500_instruments(),
    "start_time": "2020-01-02",
    "end_time": "2020-12-31",
    "segments": {
        "train": ("2020-01-02", "2020-06-30"),
        "valid": ("2020-07-01", "2020-09-30"),
        "test":  ("2020-10-01", "2020-12-31"),
    },
    "top_k_factors": 20 # 我们选取筛选出来的 Top 20 因子来建模
}

def run_ml_pipeline():
    print("="*60)
    print("=== 第二阶段：多因子机器学习建模与预测 ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    accessor = QlibDataAccessor()
    accessor.ensure_init()
    
    # 2. 读取第一阶段筛选出的高质量因子
    print("\n[2] 读取第一阶段筛选出的高质量因子...")
    csv_path = os.path.join(os.path.dirname(__file__), "factor_screening_results_with_meaning.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到因子筛选结果文件: {csv_path}。请先运行 factor_screening.py！")
        
    screened_df = pd.read_csv(csv_path)
    # 按综合得分降序排列，取前 Top K 个因子
    top_factors_df = screened_df.sort_values(by="综合得分 (Score)", ascending=False).head(CONFIG["top_k_factors"])
    selected_factor_names = top_factors_df["因子名称 (Factor)"].tolist()
    selected_factor_exprs = top_factors_df["计算公式 (Expression)"].tolist()
    
    print(f">>> 成功读取排名前 {CONFIG['top_k_factors']} 的因子用于机器学习建模。")
    
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
        segments=CONFIG["segments"]
    )
    print(">>> 训练/验证/测试集 切分完成！")

    # 4. 训练集成机器学习模型
    # 【量化逻辑】单个模型容易过拟合（比如只背题不懂变通）。
    # 业界标配是：LightGBM + XGBoost + CatBoost 三大树模型融合（Ensemble）。
    print("\n[4] 开始训练机器学习模型 (LGBM + XGBoost + CatBoost)...")
    print("    - 正在训练 LightGBM 模型...")
    lgb_model = train_lgb_model(dataset)
    
    print("    - 正在训练 XGBoost 模型...")
    xgb_model = train_xgb_model(dataset)
    
    print("    - 正在训练 CatBoost 模型...")
    cat_model = train_catboost_model(dataset)
    
    print(">>> 所有模型训练完毕！")

    # 5. 模型集成与样本外预测
    print("\n[5] 在测试集上进行模型集成与预测 (生成 Alpha 预测得分)...")
    # 把三个模型的预测结果等权平均，得到最终的综合打分 (score)
    predictions = predict_ensemble_models([lgb_model, xgb_model, cat_model], dataset)
    
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