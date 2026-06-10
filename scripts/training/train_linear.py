import os
import sys
import argparse

# Conda site-packages 优先，Roaming 放后面（解决 Roaming 路径污染）
sp = list(sys.path)
conda_sp = [p for p in sp if 'Anaconda' in p and 'site-packages' in p]
roaming_sp = [p for p in sp if 'Roaming' in p]
other_sp = [p for p in sp if p not in conda_sp and p not in roaming_sp]
sys.path = conda_sp + other_sp + roaming_sp

import pandas as pd
import numpy as np
import yaml
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.features.builder import build_factor_library_bundle, FeatureBundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.models.training import train_ridge_model, predict_ensemble_models
from qlworks.models import prepare_feature_selection_data, cached_select_features
from qlworks.config import QLIB_DATA_DIR
import qlib

# ==============================================================================
# [配置加载] 优先从 YAML 文件加载，兼容传统内嵌 CONFIG 字典
# ==============================================================================

# 解析命令行参数
_parser = argparse.ArgumentParser(description="线性模型训练脚本")
_parser.add_argument("--config", type=str, default="linear_2025",
                     help="策略配置文件名称（不含后缀，从 scripts/configs/ 加载）")
_args, _ = _parser.parse_known_args()

# 加载 YAML 配置，失败时回退到传统 CONFIG 字典
try:
    from _config import load_config
    CONFIG = load_config(_args.config)
    if not CONFIG:
        raise ValueError("配置为空")
    print(f"  使用 YAML 配置: {_args.config}")
except Exception as e:
    print(f"  YAML 配置加载失败 ({e})，使用内嵌 CONFIG 字典")
    # ==============================================================================
    # [全局配置区] 传统 CONFIG 字典（内嵌版本）
    # ==============================================================================
    CONFIG = {
        "instruments": "csi500",
        "start_time": "2020-01-01",
        "end_time": "2025-12-31",

        "model_type": "linear",
        "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"],
        "label_names": ["LABEL_5D"],
        "factor_files": ["reversal_momentum_factors"],
        "factor_cache_names": [], # DuckDB + Parquet 预计算因子（注入为 Qlib 表达式）
        "neutralize_features": True,
        "neutralize_labels": True,
        "symmetric_orthogonalization": True,

        "rolling_windows": [
            {"name": "Test_2023", "train": ("2020-01-01", "2021-12-20"), "valid": ("2022-01-01", "2022-12-20"), "test":  ("2023-01-01", "2023-12-31")},
            {"name": "Test_2024", "train": ("2021-01-01", "2022-12-20"), "valid": ("2023-01-01", "2023-12-20"), "test":  ("2024-01-01", "2024-12-31")},
            {"name": "Test_2025", "train": ("2022-01-01", "2023-12-20"), "valid": ("2024-01-01", "2024-12-20"), "test":  ("2025-01-01", "2025-12-31")},
        ],
        "top_k_factors": 20,
        "icir_stability_check": {"enabled": True, "rolling_window": 60, "keep_ratio": 0.9},
        "feature_selection": {"method": "filter", "algo": "f_regression", "label_col": "LABEL_5D", "remove_collinearity": False},
    }
# ==============================================================================

def run_ml_pipeline():
    print("="*60)
    print("=== 第二阶段：【终极改造】动态因子筛选与多因子机器学习建模 ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    # 强制将 joblib_backend 设为 loky，以彻底解决 threading 在处理超大矩阵时的 MemoryError 
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="loky", maxtasksperchild=None)
    
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
        # 原始代码使用 segments["test"][1] 作为 end_time，导致计算 4 年的 10 个因子
        # 改成 segments["train"][1] 后数据量减少 50-75%，大幅加速
        _, dataset_full = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_bundle=bundle_all,
            factor_cache_names=CONFIG.get("factor_cache_names"),
            start_time=segments["train"][0],
            end_time=segments["train"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments={"train": segments["train"]},
            model_type=CONFIG["model_type"],
            neutralize_features=CONFIG["neutralize_features"], # 若出现严重截面偏移，可设为 True
            neutralize_labels=CONFIG["neutralize_labels"],
            # [AQR 改进]: 筛选因子时绝对不能开启正交化！因为正交化后的因子是全量因子的线性组合。
            # 必须用原始（去极值+中性化）的因子算 IC，挑出最好的 Top K，然后再对这 Top K 进小宇宙正交化！
            symmetric_orthogonalization=False
        )
        
        # 3.2 在训练集上进行因子筛选
        print(f"\n[3.2 - {window_name}] 在当前训练集上执行动态因子筛选 (选取前 {CONFIG['top_k_factors']} 个)...")
        train_frame_full = dataset_full.prepare("train")
        x_train, y_train, _ = prepare_feature_selection_data(train_frame_full, label_col=fs_conf["label_col"])
        
        valid_idx = y_train.dropna().index
        x_train = x_train.loc[valid_idx]
        y_train = y_train.loc[valid_idx]
        
        fs_result = cached_select_features(
            x_train, y_train, 
            method=fs_conf["method"], 
            algo=fs_conf["algo"], 
            k=CONFIG["top_k_factors"], # 修正参数传递：Filter 算法底层接收的是 k
            remove_collinearity=fs_conf["remove_collinearity"]
        )
        
        selected_factor_names = fs_result.selected_features
        print(f">>> {window_name} 动态因子筛选完成！本期入选的因子为:")
        print(f"    {selected_factor_names}")

        # [Citadel Alpha Lab 改进] ICIR 稳定性校验：计算每个因子的滚动 ICIR 正值天数占比
        if CONFIG["icir_stability_check"]["enabled"] and len(selected_factor_names) > CONFIG["top_k_factors"] * 0.5:
            print(f"\n[3.2b - {window_name}] 执行 ICIR 稳定性校验...")
            rolling_w = CONFIG["icir_stability_check"]["rolling_window"]
            keep_ratio = CONFIG["icir_stability_check"]["keep_ratio"]

            label_col = fs_conf["label_col"]
            train_data_for_ic = train_frame_full.copy()
            feature_cols = [c for c in selected_factor_names if c in train_data_for_ic.columns]

            if len(feature_cols) > 5 and label_col in train_data_for_ic.columns:
                def daily_ic_func(df):
                    return df[feature_cols].corrwith(df[label_col], method='spearman')
                daily_ic = train_data_for_ic.groupby(level='datetime').apply(daily_ic_func)

                rolling_mean = daily_ic.rolling(window=rolling_w, min_periods=rolling_w // 2).mean()
                rolling_std = daily_ic.rolling(window=rolling_w, min_periods=rolling_w // 2).std()
                rolling_icir = rolling_mean / rolling_std.replace(0, np.nan)

                pos_ratio = (rolling_icir > 0).sum() / rolling_icir.notna().sum()
                pos_ratio = pos_ratio.fillna(0).sort_values(ascending=False)

                keep_count = max(int(len(pos_ratio) * keep_ratio), CONFIG["top_k_factors"])
                stable_factors = pos_ratio.head(keep_count).index.tolist()
                print(f"    ICIR 正值天数占比 Top {keep_count} 个因子保留，{len(selected_factor_names) - keep_count} 个不稳定因子被剔除")
                selected_factor_names = stable_factors
            else:
                print(f"    特征数或标签列不足，跳过 ICIR 稳定性校验")

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
            factor_cache_names=CONFIG.get("factor_cache_names"),
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type=CONFIG["model_type"],
            neutralize_features=CONFIG["neutralize_features"],
            neutralize_labels=CONFIG["neutralize_labels"],
            symmetric_orthogonalization=CONFIG.get("symmetric_orthogonalization", False)
        )

        print(f"\n[3.4 - {window_name}] 开始训练机器学习模型 (Ridge with TimeSeriesCV) ...")
        print("    - 正在训练 Ridge 线性回归模型 ...")
        ridge_model = train_ridge_model(dataset_sub)
        
        print(f">>> {window_name} 线性模型训练完毕！")

        print(f"\n[3.5 - {window_name}] 在测试集上进行模型预测 (生成 Alpha 预测得分)...")
        predictions = predict_ensemble_models([ridge_model], dataset_sub, segment="test")
        
        # 将原始得分进行横截面百分位排序 (Cross-Sectional Ranking)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
            
        # [Two Sigma 改进] 严格处理 NaN 值（停牌、缺失等），防止影响横截面排序结果
        predictions = predictions.dropna(subset=["score"])
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")
        
        print(f">>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)
        
        # [Bloomberg Data Pipeline 改进] 当前 Window 结束，彻底释放轻量数据集与模型占用的显存/内存
        del dataset_sub, ridge_model
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
    output_path = os.path.join(os.path.dirname(__file__), "score_linear.csv")
    final_predictions.to_csv(output_path)
    print(f"\n>>> 线性模型预测得分已保存至: {output_path}")
    print("="*60)
    print("【下一步指引】")
    print("现在您已经有了每只股票每天的预测得分 (Score)。")
    print("下一步就是将这个 Score 喂给 Backtrader (src/qlworks/backtest/bt_runner.py)。")
    print("Backtrader 会模拟真实的交易环境：每天买入 Score 最高的 N 只股票，计算扣除印花税、滑点后的真实收益和换手率！")
    print("="*60)

if __name__ == "__main__":
    run_ml_pipeline()
