import os
import sys
import pandas as pd
import numpy as np
import yaml

pd.options.mode.use_inf_as_na = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.data import DataFetchSpec, QlibDataAccessor, clean_ohlcv_data
from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.models import prepare_feature_selection_data, select_features
import qlib

CONFIG = {
    "instruments": "csi500",  # [Renaissance 改进] 使用 Qlib 动态股票池名称以杜绝前视和幸存者偏差
    "start_time": "2023-01-01",
    "end_time": "2025-12-31",
    "segments": {
        "train": ("2023-01-01", "2023-12-31"),
        "valid": ("2024-01-01", "2024-12-31"),
        "test":  ("2025-01-01", "2025-12-31"),
    },
    # 【树模型流派核心配置】
    # 1. 禁用严格的共线性剔除 (因为树模型不怕共线性，反而能利用细微差异)
    # 2. 使用支持 GPU 的 xgboost/lightgbm 提取特征重要性，避免 CPU 卡顿
    "feature_selection": {
        "enable": True,
        "method": "embedded",   
        "algo": "lightgbm",     # 改为 lightgbm (或者 xgboost) 以使用 GPU 加速
        "threshold": 0.0,    
        "max_features": 50,            
        "remove_collinearity": False,  # 【关键修改】：对树模型关闭共线性剔除
    },
    
    # --- 模型与标签配置 ---
    "model_type": "tree", # 机器学习模型类型 (tree / linear 等)
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"], # [Citadel Alpha Lab 改进] 预测标签公式: T+1开盘买入, T+5收盘卖出
    "label_names": ["LABEL_5D"], # 预测标签名称
    "factor_files": ["style_factors", "quality_factors", "price_volume_factors", "sentiment_factors", "risk_factors"], # 待加载的因子文件
    "neutralize_features": False, # 树模型保留特征的原始非线性分布
    "neutralize_labels": True, # 树模型只中性化标签
}

def load_factor_metadata(factor_files):
    metadata = {}
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../factors_repo"))
    for file_name in factor_files:
        yaml_path = os.path.join(repo_path, f"{file_name}.yaml")
        if not os.path.exists(yaml_path): continue
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if not data or 'factors' not in data: continue
        for factor in data['factors']:
            name = factor.get('name')
            if not name: continue
            metadata[name] = {
                '大类 (File)': file_name,
                '业务意义 (Meaning)': factor.get('meaning', '').replace('\n', ' '),
            }
    return metadata

def screen_factors_for_tree_model():
    print("="*80)
    print("=== 【Point72 流派】机器学习(树模型)专用因子筛选流水线 ===")
    print("核心逻辑：仅对 Label 中性化、不对 Feature 中性化、不过滤共线性、用 Feature Importance 评估")
    print("="*80)
    
    qlib.init(provider_uri=r"e:\Quant\Qlibworks\qlib_data", region="cn", joblib_backend="threading")
    accessor = QlibDataAccessor()
    accessor.ensure_init()
    
    print("\n[1] 数据拉取 (使用 Qlib 原生 DatasetH)...")
    factor_files = CONFIG["factor_files"]
    bundle = build_factor_library_bundle(factor_files)
    
    # [Citadel Alpha Lab 改进] 标签改为真实的T+1开盘到T+5收盘的收益率，防范日内跳空带来的前视偏差错位
    bundle.label_fields = CONFIG["label_fields"]
    bundle.label_names = CONFIG["label_names"]
    
    print("\n[2] 构建 DatasetH (树模型专属处理)...")
    _, dataset = create_custom_dataset(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        fit_start_time=CONFIG["segments"]["train"][0],
        fit_end_time=CONFIG["segments"]["train"][1],
        segments=CONFIG["segments"],
        model_type=CONFIG["model_type"],
        neutralize_labels=CONFIG["neutralize_labels"],      
        neutralize_features=CONFIG["neutralize_features"]    
    )
    
    print("\n[3] 训练树模型 (GPU加速) 提取特征重要性 (Feature Importance)...")
    fs_conf = CONFIG["feature_selection"]
    train_frame = dataset.prepare("train")
    
    x_train, y_train, _ = prepare_feature_selection_data(train_frame, label_col=CONFIG["label_names"][0])
    
    # [Point72 级数据对齐]: 树模型虽然能处理 X 中的 NaN，但不能用带有 NaN 的 y_train 去训练
    valid_idx = y_train.dropna().index
    x_train = x_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]
    
    if x_train.empty:
        raise ValueError("严重错误：标签对齐后训练集变为空！")
    
    fs_result = select_features(
        x_train, y_train, 
        method=fs_conf["method"], 
        algo=fs_conf["algo"], 
        threshold=0.0, # 【修复】对于树模型，重要性阈值应该设为 0，因为很多特征的重要度可能极小但不为0
        model_kwargs={"max_features": fs_conf["max_features"], "importance_type": "gain"},
        remove_collinearity=fs_conf["remove_collinearity"]
    )
    
    print(f"\n[4] 筛选完成！树模型根据非线性分裂增益选出了 {len(fs_result.selected_features)} 个因子。")
    
    # 树模型流派直接看 Feature Importance，不看 IC
    importance_scores = fs_result.feature_scores.head(30)
    
    print("\n" + "="*80)
    print("【全市场 树模型 Feature Importance Top 30】")
    metadata = load_factor_metadata(factor_files)
    
    records = []
    for factor, score in importance_scores.items():
        meta = metadata.get(factor, {'大类 (File)': '', '业务意义 (Meaning)': ''})
        records.append({
            '因子名称': factor,
            '重要性得分(RF Gain)': round(score, 6),
            '大类': meta['大类 (File)'],
            '业务意义': meta['业务意义 (Meaning)']
        })
        
    result_df = pd.DataFrame(records)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(result_df)
    
    try:
        out_path = os.path.join(os.path.dirname(__file__), "tree_model_selected_factors.csv")
        result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n- 树模型专属特征名单已保存至: {out_path}")
    except PermissionError:
        out_path = os.path.join(os.path.dirname(__file__), "tree_model_selected_factors_new.csv")
        result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n- 警告: 原文件被占用，结果已保存至新文件: {out_path}")
    print("="*80)

if __name__ == "__main__":
    screen_factors_for_tree_model()
