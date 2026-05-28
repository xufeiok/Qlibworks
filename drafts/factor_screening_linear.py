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
    # 【AQR/Citadel 流派核心配置】
    # 1. 强制剔除高共线性特征 (线性模型极度畏惧多重共线性)
    # 2. 标签中性化 (必须)
    # 3. 特征中性化 (必须)
    # 4. 使用纯线性的 f_regression 进行初步海选
    "feature_selection": {
        "enable": True,
        "method": "filter",   
        "algo": "f_regression", 
        "k": 50,               
        "remove_collinearity": True,   # 【关键修改】：对线性模型开启严格的共线性过滤
        "collinearity_threshold": 0.7 
    },
    
    # --- 模型与标签配置 ---
    "model_type": "linear", # 机器学习模型类型 (tree / linear 等)
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"], # [Citadel Alpha Lab 改进] 预测标签公式: T+1开盘买入, T+5收盘卖出
    "label_names": ["LABEL_5D"], # 预测标签名称
    "factor_files": ["style_factors", "quality_factors", "price_volume_factors", "sentiment_factors", "risk_factors"], # 待加载的因子文件
    "neutralize_features": True, # 线性模型必须开启特征中性化
    "neutralize_labels": True, # 线性模型必须同时中性化标签
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

def screen_factors_for_linear_model():
    print("="*80)
    print("=== 【AQR/Citadel 流派】传统多因子(线性)专用筛选流水线 ===")
    print("核心逻辑：双重中性化(特征+标签)、严格剔除共线性、依赖 IC/ICIR 进行最终评价")
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
    
    print("\n[2] 构建 DatasetH (线性模型专属处理)...")
    # 【关键修改】：线性模型必须同时中性化特征和标签，否则市值会污染所有因子
    # 1. 标签中性化 (必须)
    # 2. 特征中性化 (必须，但因为 237 个原始特征存在极高共线性会导致 OLS 矩阵奇异，我们需要修改 Qlib 底层或在此脚本手动分批中性化。目前开启，若再报错需要重构)
    # (注：具体的 Processor 顺序已在 create_custom_dataset 内部处理)
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
        neutralize_features=CONFIG["neutralize_features"]    # <--- 【恢复】：遵循量化原则，线性模型必须开启特征中性化
    )
    
    print("\n[3] 线性相关性与共线性初步过滤...")
    fs_conf = CONFIG["feature_selection"]
    train_frame = dataset.prepare("train")
    
    x_train, y_train, _ = prepare_feature_selection_data(train_frame, label_col=CONFIG["label_names"][0])
    
    # [AQR 级数据对齐]: 绝不能用 pd.concat().dropna() 暴力删除特征空值行！
    # 1. 标签 y 不能有 NaN (否则 sklearn 会报错)，必须剔除
    valid_idx = y_train.dropna().index
    x_train = x_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]
    
    # 2. 特征 X 如果存在零星 NaN，必须用 0 填补，绝不能删行
    # 理由：在经过 Z-Score 或中性化后，特征的截面均值是 0。用 0 填补代表对该因子的观点为“中性”，不浪费其他有效因子。
    x_train = x_train.fillna(0)
    
    if x_train.empty:
        raise ValueError("严重错误：训练集变为空！")
        
    print(f"    - 标签对齐并填补特征缺失值后，剩余有效训练样本数: {len(x_train)}")
    
    # 【修复结束】
    
    fs_result = select_features(
        x_train, y_train, 
        method=fs_conf["method"], 
        algo=fs_conf["algo"], 
        k=fs_conf["k"],
        remove_collinearity=fs_conf["remove_collinearity"],
        collinearity_threshold=fs_conf["collinearity_threshold"]
    )
    selected_features = list(fs_result.selected_features)
    print(f"\n[4] 过滤法粗筛完成，剩余 {len(selected_features)} 个因子进入深度 IC 评价。")
    
    print("\n[5] 深度量化评价 (计算 IC 与 ICIR)...")
    ic_dict, icir_dict = {}, {}
    for feature in selected_features:
        daily_ic = train_frame.groupby('datetime').apply(lambda x: x[feature].corr(x[CONFIG["label_names"][0]], method='spearman')).dropna()
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        ic_dict[feature] = ic_mean
        icir_dict[feature] = 0 if (pd.isna(ic_std) or ic_std < 1e-6) else (ic_mean / ic_std * np.sqrt(252))
        
    ic_df = pd.DataFrame({'IC': list(ic_dict.values()), 'ICIR': list(icir_dict.values())}, index=list(ic_dict.keys()))
    ic_df['Score'] = ic_df['IC'].abs() * ic_df['ICIR'].abs()
    
    # 线性流派最终依靠 IC * ICIR 打分排名
    top_factors = ic_df.sort_values(by='Score', ascending=False).head(30)
    
    print("\n" + "="*80)
    print("【全市场 线性模型 IC/ICIR 综合得分 Top 30】")
    metadata = load_factor_metadata(factor_files)
    
    records = []
    for factor, row in top_factors.iterrows():
        meta = metadata.get(factor, {'大类 (File)': '', '业务意义 (Meaning)': ''})
        records.append({
            '因子名称': factor,
            '综合得分': round(row['Score'], 4),
            'IC均值': round(row['IC'], 4),
            'ICIR': round(row['ICIR'], 2),
            '大类': meta['大类 (File)'],
            '业务意义': meta['业务意义 (Meaning)']
        })
        
    result_df = pd.DataFrame(records)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(result_df)
    
    out_path = os.path.join(os.path.dirname(__file__), "linear_model_selected_factors.csv")
    result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n- 线性模型专属特征名单已保存至: {out_path}")
    print("="*80)

if __name__ == "__main__":
    screen_factors_for_linear_model()
