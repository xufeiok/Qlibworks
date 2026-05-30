import os
import sys
import pandas as pd
import numpy as np
import gc

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.features.builder import build_factor_library_bundle, FeatureBundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.config import QLIB_DATA_DIR
import qlib

# ==============================================================================
# [全局配置区]
# ==============================================================================
CONFIG = {
    "instruments": "csi500", 
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",
    
    "model_type": "linear", 
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"], 
    "label_names": ["LABEL_5D"], 
    "factor_files": ["style_factors", "quality_factors", "price_volume_factors", "sentiment_factors", "risk_factors"],
    
    # [Citadel Alpha Lab 改进] 传统 ICIR 基准不需要对特征进行严苛的中性化和正交化
    # 它反映的是因子原始的预测能力组合。只对特征进行去极值标准化即可。
    "neutralize_features": False, 
    "neutralize_labels": True, 
    "symmetric_orthogonalization": False,

    "rolling_windows": [
        {
            "name": "Test_2023",
            "train": ("2020-01-01", "2021-12-20"), 
            "valid": ("2022-01-01", "2022-12-20"), 
            "test":  ("2023-01-01", "2023-12-31"), 
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
    "top_k_factors": 50, # 选出过去一段时间 ICIR 最高的 50 个因子进行加权
    "icir_window": 60,    # [Citadel 改进] 滚动 ICIR 计算窗口（交易日），60天约等于一个季度
}

def run_icir_baseline_pipeline():
    print("="*60)
    print("=== [基石引入] 传统多因子 ICIR 静态加权基准模型 ===")
    print("="*60)

    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="loky", maxtasksperchild=None)
    
    print("\n[2] 读取因子库的所有因子公式...")
    factor_files = CONFIG["factor_files"]
    bundle_all = build_factor_library_bundle(factor_files)
    bundle_all.label_fields = CONFIG["label_fields"]
    bundle_all.label_names = CONFIG["label_names"]
    
    print(f">>> 成功加载 {len(bundle_all.fields)} 个因子候选池，并将预测标签设为 {bundle_all.label_names[0]} (5天收益率)。")
    
    all_predictions = []
    
    for window in CONFIG["rolling_windows"]:
        window_name = window["name"]
        print(f"\n{'='*60}")
        print(f"=== 正在处理滚动窗口: {window_name} ===")
        print(f"{'='*60}")
        
        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }
        
        # 3.1 构建全量因子数据集
        print(f"\n[3.1 - {window_name}] 构建全量因子数据集 (用于计算历史 IC/IR)...")
        _, dataset_full = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_bundle=bundle_all,
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type=CONFIG["model_type"],
            neutralize_features=CONFIG["neutralize_features"], 
            neutralize_labels=CONFIG["neutralize_labels"],
            symmetric_orthogonalization=CONFIG["symmetric_orthogonalization"]
        )
        
        # 3.2 计算训练集（历史窗口）上每个因子的滚动 ICIR
        print(f"\n[3.2 - {window_name}] 计算训练集因子 IC/IR，确定滚动加权权重...")
        train_frame = dataset_full.prepare("train")
        
        # 提取特征和标签
        feature_cols = [c for c in train_frame.columns if c not in CONFIG["label_names"]]
        label_col = CONFIG["label_names"][0]
        
        # 计算每日截面 Rank IC
        print("    - 计算每日 Rank IC...")
        def calc_daily_ic(df):
            return df[feature_cols].corrwith(df[label_col], method='spearman')
            
        daily_ic = train_frame.groupby(level='datetime').apply(calc_daily_ic)
        
        # [Citadel 改进] 计算滚动 ICIR（均值/标准差），然后取最近一期的滚动 ICIR 作为权重
        # 相比全期静态 ICIR，滚动 ICIR 能捕捉到因子近期预测能力的变化
        icir_window = CONFIG["icir_window"]
        rolling_mean = daily_ic.rolling(window=icir_window, min_periods=max(icir_window // 2, 10)).mean()
        rolling_std = daily_ic.rolling(window=icir_window, min_periods=max(icir_window // 2, 10)).std()
        rolling_icir = rolling_mean / rolling_std.replace(0, np.nan)
        
        # 取最近一期的滚动 ICIR 作为因子权重
        latest_icir = rolling_icir.iloc[-1].fillna(0)
        
        # 选择 ICIR 绝对值最高的 Top K 个因子
        top_k_factors = latest_icir.abs().sort_values(ascending=False).head(CONFIG["top_k_factors"]).index.tolist()
        
        # 权重与最新滚动 ICIR 成正比
        weights = latest_icir[top_k_factors]
        weights = weights / weights.abs().sum()
        
        print(f">>> 选出 Top {CONFIG['top_k_factors']} 因子，权重抽样:")
        print(weights.head())
        
        # 3.3 在测试集上直接应用 ICIR 权重进行线性加权打分
        print(f"\n[3.3 - {window_name}] 在测试集上进行 ICIR 静态加权打分...")
        test_frame = dataset_full.prepare("test")
        
        # 提取测试集中的入选因子
        test_features = test_frame[top_k_factors]
        
        # 计算最终得分：Score = \sum (Factor_i * Weight_i)
        # 用矩阵乘法加速
        test_score = test_features.dot(weights)
        
        predictions = test_score.to_frame("score")
        
        # 横截面百分位排序 (Cross-Sectional Ranking)
        predictions = predictions.dropna(subset=["score"])
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")
        
        print(f">>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)
        
        del dataset_full, train_frame, test_frame, daily_ic
        gc.collect()
    
    # 4. 合并所有滚动窗口的样本外预测结果
    print("\n[4] 所有滚动窗口执行完毕！正在合并预测结果...")
    final_predictions = pd.concat(all_predictions)
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)
    
    print(f">>> 合并完成！总测试集跨度: {final_predictions.index.get_level_values('datetime').min().date()} 至 {final_predictions.index.get_level_values('datetime').max().date()}")
    
    # 5. 保存预测结果
    output_path = os.path.join(os.path.dirname(__file__), "score_icir.csv")
    final_predictions.to_csv(output_path)
    print(f"\n>>> ICIR 传统加权预测得分已保存至: {output_path}")
    print("="*60)

if __name__ == "__main__":
    run_icir_baseline_pipeline()
