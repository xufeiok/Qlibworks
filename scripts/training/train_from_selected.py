"""
基于精选因子列表的模型训练与选股脚本。

从 select_factors.py 输出的 CSV 中读取精选因子列表，
仅用这些筛选后的因子进行 ML 模型训练 (LGB/XGB/CatBoost)，
输出每只股票每天的 Alpha 预测得分。

用法：
  修改文件顶部 LOCAL_CONFIG 字典中的参数，然后直接运行：
    python train_from_selected.py
"""

import os
import sys
import warnings
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

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
from pathlib import Path
from qlib.data.dataset.handler import DataHandlerLP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import (
    create_custom_dataset,
    build_custom_feature_cache,
    wrap_dataset_with_cached_train_frame,
)
from qlworks.models.training import (
    train_lgb_model, train_xgb_model, train_catboost_model,
    predict_ensemble_models, compute_ic, compute_ic_ewma,
)
from qlworks.config import QLIB_DATA_DIR
import qlib


def load_selected_factors(csv_path: str):
    """
    读取精选因子 CSV，返回 (source_files, factor_names) 元组。

    source_files: 去重后的因子文件列表（用于 build_factor_library_bundle）
    factor_names: 所有选中因子的名称列表（用于 selected_feature_names）
    """
    df = pd.read_csv(csv_path)
    df_selected = df[df["selected"] == True].copy()

    if len(df_selected) == 0:
        print("[错误] CSV 中没有 selected=True 的因子")
        sys.exit(1)

    source_files = sorted(df_selected["source_file"].unique().tolist())
    factor_names = df_selected["factor_name"].tolist()

    print(f"  源文件: {source_files}")
    print(f"  因子数量: {len(factor_names)}")
    print(f"  因子列表: {factor_names}")

    return source_files, factor_names


# ==============================================================================
# [全局配置区]
# ==============================================================================
LOCAL_CONFIG = {
    "instruments": "csi500",
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",

    "model_type": "tree",
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"],
    "label_names": ["LABEL_5D"],
    "factor_cache_names": [],

    "normalize_features": True,
    "neutralize_features": False,
    "renormalize_features_after_neutralize": False,
    "normalize_labels": True,
    # [AQR/Point72] 始终对标签做行业/市值中性化，确保模型学习纯 alpha 而非市场 beta
    "neutralize_labels": True,

    # [Renaissance] 各窗口间 train→valid→test 均保留 ≥12 天 embargo 防止标签泄露
    # 窗口定义保持不变（已有天然 12d 间隔）
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
    "train_models": ["lgb", "xgb", "cat"],
    "model_params": {
        "lgb": {"num_boost_round": 600, "early_stopping_rounds": 50},
        "xgb": {"num_boost_round": 600, "early_stopping_rounds": 50},
        # [Bloomberg] 移除 task_type: CPU 硬编码，由底层 training.py GPU 自动检测
        "cat": {"num_boost_round": 400, "early_stopping_rounds": 30},
    },

    # 精选因子 CSV 路径（select_factors.py 的输出文件）
    "factor_list": "selected_factors_20260628_010521_selected.csv",

    # 输出路径：None 自动生成 score_tree_selected.csv
    "output": None,
}


def main():
    CONFIG = dict(LOCAL_CONFIG)

    print("=" * 60)
    print("  基于精选因子的模型训练与选股")
    print("=" * 60)

    # 1. 加载精选因子列表
    print(f"\n[1] 加载精选因子列表: {CONFIG["factor_list"]}")
    source_files, selected_factor_names = load_selected_factors(CONFIG["factor_list"])
    CONFIG["factor_files"] = source_files

    # 2. 初始化 Qlib
    print("\n[2] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading", maxtasksperchild=None)

    # 3. 从因子库加载所有因子
    print("\n[3] 读取因子库 (Factor Library)...")
    bundle_all = build_factor_library_bundle(source_files)
    bundle_all.label_fields = CONFIG["label_fields"]
    bundle_all.label_names = CONFIG["label_names"]

    print(f"  >>> 成功加载 {len(bundle_all.fields)} 个因子，精选中 {len(selected_factor_names)} 个")

    all_predictions = []
    # [Citadel Alpha Lab] 跨窗口 IC 跟踪，用于 EWMA 加权集成
    model_ic_history: dict[str, list[float]] = {}
    # [Bloomberg] 全局特征缓存，首窗口构建后跨窗口复用
    global_feature_cache = None

    # 4. 遍历所有滚动窗口
    for window_idx, window in enumerate(CONFIG["rolling_windows"]):
        window_name = window["name"]
        print(f"\n{'='*60}")
        print(f"=== 正在处理滚动窗口: {window_name} ===")
        print(f"    [训练集]: {window['train'][0]} 到 {window['train'][1]}")
        print(f"    [验证集]: {window['valid'][0]} 到 {window['valid'][1]}")
        print(f"    [测试集]: {window['test'][0]} 到 {window['test'][1]}")
        print(f"{'='*60}")

        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }

        # ----- [4.0] 特征缓存：首窗口构建，后续复用 -----
        print(f"\n[4.0 - {window_name}] 特征缓存...")
        if global_feature_cache is None:
            global_feature_cache = build_custom_feature_cache(
                instruments=CONFIG["instruments"],
                feature_bundle=bundle_all,
                factor_cache_names=CONFIG["factor_cache_names"],
                start_time=CONFIG["start_time"],
                end_time=CONFIG["end_time"],
                freq="day",
            )
            print(f"    [全局缓存] 构建完成：覆盖 {CONFIG['start_time']} ~ {CONFIG['end_time']}")
        else:
            print(f"    [复用全局缓存] 跳过重复计算")

        # ----- [4.1] 一次性构建含 train/valid/test 的数据集 -----
        # [Point72 性能优化] 合并原 4.1+4.2 两次 create_custom_dataset 为一次
        print(f"\n[4.1 - {window_name}] 构建轻量级 DatasetH（{len(selected_factor_names)} 个精选因子，含 train/valid/test 三段）...")
        _, dataset_sub = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_cache=global_feature_cache,
            selected_feature_names=selected_factor_names,
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            segments=segments,
            model_type=CONFIG["model_type"],
            normalize_features=CONFIG["normalize_features"],
            neutralize_features=CONFIG["neutralize_features"],
            renormalize_features_after_neutralize=CONFIG["renormalize_features_after_neutralize"],
            normalize_labels=CONFIG["normalize_labels"],
            neutralize_labels=CONFIG["neutralize_labels"],
        )

        # ----- [4.2] 提取训练/验证数据并缓存 -----
        print(f"\n[4.2 - {window_name}] 提取训练/验证帧...")
        train_frame_full = dataset_sub.prepare("train")
        print(f"    >>> 训练集: {train_frame_full.shape[0]} 行 × {train_frame_full.shape[1]} 列")

        valid_frame = None
        try:
            valid_frame = dataset_sub.prepare("valid")
            print(f"    >>> 验证集: {valid_frame.shape[0]} 行")
        except Exception:
            print(f"    [警告] 验证集为空，跳过早停和 IC 加权")

        dataset_sub = wrap_dataset_with_cached_train_frame(
            dataset_sub,
            train_frame=train_frame_full,
            selected_feature_names=selected_factor_names,
            label_names=bundle_all.label_names,
            learn_data_key=DataHandlerLP.DK_L,
            infer_data_key=DataHandlerLP.DK_I,
            valid_frame=valid_frame,
        )

        del train_frame_full
        gc.collect()

        # ----- [4.3] 训练模型 -----
        selected_models = list(CONFIG.get("train_models", ["lgb", "xgb", "cat"]))
        model_params = CONFIG.get("model_params", {})
        models = []

        print(f"\n[4.3 - {window_name}] 开始训练机器学习模型 {selected_models}...")

        if "lgb" in selected_models:
            print("    - 正在训练 LightGBM 模型...")
            models.append(train_lgb_model(dataset_sub, params=model_params.get("lgb")))
        if "xgb" in selected_models:
            print("    - 正在训练 XGBoost 模型...")
            models.append(train_xgb_model(dataset_sub, params=model_params.get("xgb")))
        if "cat" in selected_models:
            # [Bloomberg 修复] 移除 task_type="CPU" 硬编码，由底层自动检测 GPU
            print("    - 正在训练 CatBoost 模型...")
            models.append(train_catboost_model(dataset_sub, params=model_params.get("cat")))

        if not models:
            raise ValueError("CONFIG['train_models'] 不能为空")
        print(f"  >>> {window_name} 所有模型训练完毕！")

        # ----- [4.4 预测前] 计算 IC 加权权重 -----
        # [Citadel Alpha Lab] 用验证集 IC 的 EWMA 作为集成权重
        model_ic_weights = []
        if valid_frame is not None and not valid_frame.empty:
            try:
                actual_label = valid_frame["label"].squeeze()
                if isinstance(actual_label, pd.DataFrame):
                    actual_label = actual_label.iloc[:, 0]
            except (KeyError, AttributeError):
                actual_label = None

            if actual_label is not None:
                for m_idx, model in enumerate(models):
                    model_key = selected_models[m_idx] if m_idx < len(selected_models) else f"model_{m_idx}"
                    try:
                        val_pred = model.predict(dataset_sub, segment="valid")
                        if isinstance(val_pred, pd.DataFrame):
                            val_pred = val_pred.iloc[:, 0]
                        aligned_actual = actual_label.reindex(val_pred.index).dropna()
                        valid_pred_aligned = val_pred.reindex(aligned_actual.index).dropna()
                        common_idx = aligned_actual.index.intersection(valid_pred_aligned.index)
                        if len(common_idx) >= 30:
                            ic_val = compute_ic(valid_pred_aligned.loc[common_idx],
                                                aligned_actual.loc[common_idx])
                            ewma_ic = compute_ic_ewma(model_ic_history, model_key, ic_val, half_life=4)
                            model_ic_weights.append(max(ewma_ic, 0.0))
                            print(f"      [{model_key}] 验证 IC={ic_val:.4f}, EWMA-IC={ewma_ic:.4f}")
                        else:
                            model_ic_weights.append(1.0)
                            print(f"      [{model_key}] 验证数据不足({len(common_idx)}条<30)，权重=1.0")
                    except Exception as e:
                        print(f"      [{model_key}] IC 计算异常: {e}，权重=1.0")
                        model_ic_weights.append(1.0)
            else:
                model_ic_weights = [1.0] * len(models)
        else:
            model_ic_weights = [1.0] * len(models)
            if valid_frame is not None:
                print(f"      [警告] valid_frame 不含 label 列，使用等权重集成")

        # 归一化权重
        weights_arr = np.array(model_ic_weights, dtype=float)
        if weights_arr.sum() > 0:
            model_ic_weights = (weights_arr / weights_arr.sum()).tolist()
        else:
            model_ic_weights = [1.0 / len(models)] * len(models)

        if any(w != 1.0 / len(models) for w in model_ic_weights):
            print(f"      >> 集成权重 (EWMA-IC, 归一化): {[f'{w:.3f}' for w in model_ic_weights]}")
        else:
            print(f"      >> 使用等权重集成")

        # ----- [4.4] 预测 -----
        print(f"\n[4.4 - {window_name}] 在测试集上进行模型集成与预测...")
        predictions = predict_ensemble_models(models, dataset_sub, segment="test",
                                              model_weights=model_ic_weights)

        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")

        predictions = predictions.dropna(subset=["score"])
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")

        print(f"  >>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)

        del dataset_sub, models, valid_frame
        gc.collect()

    # 5. 合并所有滚动窗口的预测结果
    print("\n[5] 所有滚动窗口执行完毕！正在合并预测结果...")
    final_predictions = pd.concat(all_predictions)
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)

    print(f"  >>> 合并完成！总测试集跨度: {final_predictions.index.get_level_values('datetime').min().date()} "
          f"至 {final_predictions.index.get_level_values('datetime').max().date()}")
    print(final_predictions.head(10))

    # 6. 保存预测结果
    output_path = CONFIG["output"] or os.path.join(os.path.dirname(__file__), "score_tree_selected.csv")
    final_predictions.to_csv(output_path)
    print(f"\n  >>> 预测得分已保存至: {output_path}")

    print("=" * 60)
    factor_count = len(selected_factor_names)
    print(f"  精选因子数: {factor_count}")
    print(f"  训练模型: {selected_models}")
    print(f"  总预测记录: {len(final_predictions):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
