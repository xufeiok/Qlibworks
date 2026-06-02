import os
import sys
import warnings

# Conda site-packages 优先，Roaming 放后面（解决 Roaming 路径污染）
sp = list(sys.path)
conda_sp = [p for p in sp if 'Anaconda' in p and 'site-packages' in p]
roaming_sp = [p for p in sp if 'Roaming' in p]
other_sp = [p for p in sp if p not in conda_sp and p not in roaming_sp]
sys.path = conda_sp + other_sp + roaming_sp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import gc
import pandas as pd
import numpy as np

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import build_factor_library_bundle, FeatureBundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.models.training import train_lgb_model, train_xgb_model, train_catboost_model, predict_ensemble_models
from qlworks.models import prepare_feature_selection_data, cached_select_features
from qlworks.config import QLIB_DATA_DIR
import qlib

# ==============================================================================
# [全局配置区]
# ==============================================================================
CONFIG = {
    # [Renaissance 改进] 摒弃静态的 List 股票池，直接使用 Qlib 内置的动态别名
    # 这样 Qlib 内部会根据每一天的日期，动态过滤出当天真实属于 csi500 且未退市的成分股，彻底杜绝前视偏差与幸存者偏差
    "instruments": "csi500", 
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",
    
    # --- 模型与标签配置 ---
    "model_type": "tree", # 机器学习模型类型 (tree / linear 等)
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"], # [Citadel Alpha Lab 改进] 预测标签公式: T+1开盘买入, T+5收盘卖出
    "label_names": ["LABEL_5D"], # 预测标签名称
    "factor_files": ["style_factors", "quality_factors", "price_volume_factors", "sentiment_factors", "risk_factors"], # 待加载的因子文件
    "factor_cache_names": ["ret_1d", "ma_5", "price_position_20"], # DuckDB + Parquet 预计算因子（注入为 Qlib 表达式）
    "neutralize_features": False, # 是否对特征进行横截面中性化
    "neutralize_labels": True, # 是否对标签进行横截面中性化 (防范日内跳空带来的前视偏差错位)
    
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
    "factor_redundancy_check": {
        "enabled": True,                  # [AQR 改进] 开启因子冗余检测
        "correlation_threshold": 0.95,    # 相关性超过此阈值视为冗余
    },
    "feature_selection": {
        "method": "embedded",   
        "algo": "lightgbm",     
        "label_col": "LABEL_5D",  
        "remove_collinearity": False,
    }
}

def run_ml_pipeline():
    print("="*60)
    print("=== 第二阶段：【终极改造】动态因子筛选与多因子机器学习建模 ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading", maxtasksperchild=None)
    
    # 2. 从 YAML 中一次性拉取因子库中所有的因子
    print("\n[2] 读取因子库 (Factor Library) 的所有因子公式...")
    factor_files = CONFIG["factor_files"]
    bundle_all = build_factor_library_bundle(factor_files)
    
    # 标签配置已在全局 CONFIG 中统一定义
    bundle_all.label_fields = CONFIG["label_fields"]
    bundle_all.label_names = CONFIG["label_names"]
    
    print(f">>> 成功加载 {len(bundle_all.fields)} 个因子候选池，并将预测标签设为 {bundle_all.label_names[0]} (5天收益率)。")
    
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
        # [性能优化] 特征筛选只需要训练集数据，将时间范围限制在训练期内，避免计算全量数据
        # 原始代码使用 segments["test"][1] 作为 end_time，导致计算 4 年的 237 个因子
        # 改成 segments["train"][1] 后数据量减少 50-75%，大幅加速
        _, dataset_full = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_bundle=bundle_all,
            factor_cache_names=CONFIG["factor_cache_names"],
            start_time=segments["train"][0],
            end_time=segments["train"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments={"train": segments["train"]},
            model_type=CONFIG["model_type"],
            neutralize_features=CONFIG["neutralize_features"],
            neutralize_labels=CONFIG["neutralize_labels"]
        )
        
        # 3.2 在训练集上进行因子筛选
        print(f"\n[3.2 - {window_name}] 在当前训练集上执行动态因子筛选 (选取前 {CONFIG['top_k_factors']} 个)...")
        train_frame_full = dataset_full.prepare("train")
        print(f"    >>> 训练集数据: {train_frame_full.shape[0]} 行 × {train_frame_full.shape[1]} 列 (仅训练期)")
        x_train, y_train, _ = prepare_feature_selection_data(train_frame_full, label_col=fs_conf["label_col"])
        
        valid_idx = y_train.dropna().index
        x_train = x_train.loc[valid_idx]
        y_train = y_train.loc[valid_idx]
        
        fs_result = cached_select_features(
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

        # [AQR 改进] 因子冗余检测：用 IC（与标签的相关性）取代自相关性来决定保留哪个因子
        if CONFIG["factor_redundancy_check"]["enabled"] and len(selected_factor_names) > 10:
            print(f"\n[3.2b - {window_name}] 执行因子冗余检测 (阈值={CONFIG['factor_redundancy_check']['correlation_threshold']})...")
            corr_thresh = CONFIG["factor_redundancy_check"]["correlation_threshold"]
            feat_in_data = [c for c in selected_factor_names if c in train_frame_full.columns]

            if len(feat_in_data) > 5:
                feat_data = train_frame_full[feat_in_data]
                # 极速版：用 pooled corr (不在意时间截面分组) 替代昂贵的 groupby.corr
                corr_mat = feat_data.corr(method='spearman').abs()

                redundant_pairs = []
                for i in range(len(corr_mat.columns)):
                    for j in range(i + 1, len(corr_mat.columns)):
                        c1, c2 = corr_mat.columns[i], corr_mat.columns[j]
                        if corr_mat.iloc[i, j] > corr_thresh:
                            redundant_pairs.append((c1, c2, corr_mat.iloc[i, j]))

                if redundant_pairs:
                    # 预计算每个因子与标签的 IC（信息系数），按日求均值
                    y_aligned = y_train.reindex(feat_data.index)
                    ic_map = {}
                    for col in feat_data.columns:
                        ic_series = feat_data[col].groupby(level='datetime').apply(
                            lambda s: s.corr(y_aligned.loc[s.index]) if len(s.dropna()) > 10 else 0
                        )
                        ic_map[col] = ic_series.mean()

                    to_drop = set()
                    for c1, c2, corr_val in redundant_pairs:
                        if c1 in to_drop or c2 in to_drop:
                            continue
                        drop_f = c2 if abs(ic_map[c2]) < abs(ic_map[c1]) else c1
                        keep_f = c1 if drop_f == c2 else c2
                        to_drop.add(drop_f)
                        print(f"    冗余对: {c1}(IC={ic_map.get(c1,np.nan):.4f}) vs {c2}(IC={ic_map.get(c2,np.nan):.4f}) → 保留 {keep_f}，剔除 {drop_f}")

                    selected_factor_names = [f for f in selected_factor_names if f not in to_drop]
                    print(f"    冗余检测完成: 剔除 {len(to_drop)} 个冗余因子，保留 {len(selected_factor_names)} 个")
            else:
                print(f"    特征数不足，跳过因子冗余检测")

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
            factor_cache_names=CONFIG["factor_cache_names"],
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type=CONFIG["model_type"],
            neutralize_features=CONFIG["neutralize_features"],
            neutralize_labels=CONFIG["neutralize_labels"]
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
    output_path = os.path.join(os.path.dirname(__file__), "score_tree.csv")
    final_predictions.to_csv(output_path)
    print(f"\n>>> 树模型预测得分已保存至: {output_path}")
    print("="*60)
    print("【下一步指引】")
    print("现在您已经有了每只股票每天的预测得分 (Score)。")
    print("下一步就是将这个 Score 喂给 Backtrader (src/qlworks/backtest/bt_runner.py)。")
    print("Backtrader 会模拟真实的交易环境：每天买入 Score 最高的 N 只股票，计算扣除印花税、滑点后的真实收益和换手率！")
    print("="*60)

if __name__ == "__main__":
    run_ml_pipeline()
