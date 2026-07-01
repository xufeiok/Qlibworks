import os
import sys
import warnings
import argparse

# MLflow 新版禁止文件系统存储，设置环境变量启用（Qlib 默认使用文件系统）
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

# Conda site-packages 优先，Roaming 放后面（解决 Roaming 路径污染）
sp = list(sys.path)
conda_sp = [p for p in sp if 'Anaconda' in p and 'site-packages' in p]
roaming_sp = [p for p in sp if 'Roaming' in p]
other_sp = [p for p in sp if p not in conda_sp and p not in roaming_sp]
sys.path = conda_sp + other_sp + roaming_sp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import gc
from pathlib import Path
import pandas as pd
import numpy as np
from qlib.data.dataset.handler import DataHandlerLP

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import (
    create_custom_dataset,
    build_custom_feature_cache,
    wrap_dataset_with_cached_train_frame,
)
from qlworks.models.training import (
    train_lgb_model, train_xgb_model, train_catboost_model,
    predict_ensemble_models, compute_ic, compute_ic_ewma
)
from qlworks.models import prepare_feature_selection_data, cached_select_features
from qlworks.config import QLIB_DATA_DIR
import qlib
from _config import resolve_runtime_config

# ==============================================================================
# [全局配置区]
# 默认运行时优先使用这里的 CONFIG。
# 只有在命令行显式传入 `--config-source yaml` 时，才切换到 YAML 配置文件。
# ==============================================================================
DEFAULT_YAML_CONFIG_NAME = "tree_2025"
LOCAL_CONFIG = {
    # [Renaissance 改进] 使用 csi500 指数静态池 + 逐日 PIT 过滤。
    # _resolve_static_instruments 从 csi500.txt 读取 PIT 格式成分股数据
    # （code\tentry_date\texit_date），做窗口级粗过滤 + 逐日精确过滤：
    #   - 窗口级：保留窗口期内任一时刻是成分股的股票
    #   - 逐日级：利用 csi500.txt 的 entry/exit 日期，移除非成分股时期的数据行
    # 这消除了前视偏差（在入指数前就将股票纳入训练首日）和幸存者偏差（退市股正常包含在历史期间）。
    # 注意：停牌股票无需特殊过滤——NaN 标签会被 DropnaLabel 自然剔除。
    "instruments": "csi500", 
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",
    
    # --- 模型与标签配置 ---
    # 五个开关的语义必须分开理解，避免把“标准化”和“中性化”混淆：
    # 1) normalize_features: 因子标准化开关。
    #    - tree: True 时固定使用 CSQuantileNorm（特征横截面分位数化）
    #    - linear/nn: True 时固定使用 RobustZScoreNorm（稳健 ZScore 标准化）
    #    - 当前项目要求各模型都必须开启；若设为 False 会直接报错中断
    # 2) neutralize_features: 因子中性化开关。
    #    - 使用 CSNeutralize 剥离行业/市值暴露
    #    - tree 默认 False，linear 常见为 True
    # 3) renormalize_features_after_neutralize: 因子中性化后是否再标准化。
    #    - tree 默认 False，linear/nn 默认 True
    #    - tree 若开启，表示“残差因子”继续压成截面排序输入
    # 4) normalize_labels: 标签标准化开关。
    #    - 当前默认使用 CSQuantileNorm 对标签做横截面分位数化
    #    - 适合截面选股/排序型训练目标
    # 5) neutralize_labels: 标签中性化开关。
    #    - 使用 CSNeutralize 对标签做行业/市值残差化
    #    - 只有当你明确要学习“残差 alpha”时再开启
    "model_type": "tree", # 机器学习模型类型 (tree / linear 等)
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"], # [Citadel Alpha Lab 改进] 预测标签公式: T+1开盘买入, T+5收盘卖出
    "label_names": ["LABEL_5D"], # 预测标签名称
    "factor_files": ["reversal_momentum_factors"], # 待加载的因子文件
    "factor_cache_names": [], # DuckDB + Parquet 预计算因子（注入为 Qlib 表达式）
    "normalize_features": True, # 是否对特征进行标准化；tree=True 时固定采用 CSQuantileNorm
    "neutralize_features": False, # 是否对特征进行横截面中性化
    "renormalize_features_after_neutralize": False, # 特征中性化后是否再标准化；tree 默认关闭
    "normalize_labels": True, # 是否对标签进行横截面分位数化
    "neutralize_labels": False, # 是否对标签进行横截面中性化（行业/市值残差化）
    
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
    "top_k_factors": 6, # 我们选取筛选出来的 Top 6 因子来建模
    "feature_selection_date_stride": 2, # 特征筛选按日期抽样步长；2=隔天抽样，显著降低筛选耗时
    "train_models": ["lgb", "xgb", "cat"], # 默认包含 CatBoost；当前默认以 CPU 模式训练，避免 GPU OOM
    "model_params": {
        "lgb": {"num_boost_round": 150, "early_stopping_rounds": 20},
        "xgb": {"num_boost_round": 150, "early_stopping_rounds": 20},
        "cat": {"num_boost_round": 150, "early_stopping_rounds": 20, "task_type": "CPU"},
    },
    "factor_redundancy_check": {
        "enabled": True,                  # [AQR 改进] 开启因子冗余检测
        "correlation_threshold": 0.95,    # 相关性超过此阈值视为冗余
    },
    "icir_stability_check": {
        "enabled": True,                  # [Citadel Alpha Lab 改进] 开启 ICIR 稳定性校验
        "rolling_window": 60,             # 滚动窗口天数
        "keep_ratio": 0.8,                # 保留正 ICIR 占比最高的因子比例（树模型相对线性模型适当放宽）
    },
    "feature_selection": {
        "method": "embedded",   
        "algo": "lightgbm",     
        "label_col": "LABEL_5D",  
        "remove_collinearity": False,
    }
}

def run_ml_pipeline(config_source: str = "local", config_name: str | None = None):
    CONFIG = resolve_runtime_config(
        local_config=LOCAL_CONFIG,
        default_yaml_name=DEFAULT_YAML_CONFIG_NAME,
        config_source=config_source,
        config_name=config_name,
    )
    print("="*60)
    print("=== 第二阶段：【终极改造】动态因子筛选与多因子机器学习建模 ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading", maxtasksperchild=None)

    # [1b] 验证 csi500 成分股可用且非空，确保后续训练只在 csi500 上运行
    print("\n[1b] 验证 csi500 成分股配置...")
    try:
        _inst_dir = Path(QLIB_DATA_DIR) / "instruments" / "csi500.txt"
        if not _inst_dir.exists():
            print("  [警告] csi500.txt 文件不存在！请先运行数据同步脚本生成成分股文件。")
        else:
            with open(_inst_dir) as _f:
                _csi500_stocks = set()
                for _l in _f:
                    _l = _l.strip()
                    if not _l: continue
                    _parts = _l.split('\t')
                    if len(_parts) >= 3:
                        _csi500_stocks.add(_parts[0].lower())
            print(f"csi500 成分股文件: {_inst_dir} ({len(_csi500_stocks)} 只历史成分股)")
            print(f"样本: {sorted(_csi500_stocks)[:5]}")
    except Exception as e:
        print(f"  [警告] csi500 成分股验证失败: {e}")

    # [1c] 验证退市日期配置真实
    print("\n[1c] 验证退市日期配置...")
    try:
        _all_txt = Path(QLIB_DATA_DIR) / "instruments" / "all.txt"
        if _all_txt.exists():
            with open(_all_txt) as _f:
                _delisted = sum(1 for _l in _f if _l.strip() and not _l.strip().endswith('9999-12-31'))
            print(f"  all.txt 中含 {_delisted} 只退市股（退市日非 9999-12-31）")
    except Exception:
        pass

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
    
    # 3. 【终极架构】首窗口全局因子筛选 + 滚动窗口训练 (Walk-Forward Optimization)
    # =========================================================================
    # [AQR 改进] 因子集仅在第一窗口确定一次，后续窗口复用相同因子集。
    # 避免"每窗口重选因子 = 事后诸葛亮式自适应过拟合"。
    # 模型权重自然随窗口变化：各窗口独立训练，因子重要性（feature importance）
    # 会根据当前市场环境自动调整 -> "因子集固定，权重自适应"。
    # =========================================================================

    # ===== 3A: 首窗口全局因子筛选（仅执行一次）=====
    first_window = CONFIG["rolling_windows"][0]
    first_segments = {
        "train": first_window["train"],
        "valid": first_window["valid"],
        "test":  first_window["test"],
    }
    print(f"\n{'='*60}")
    print(f"=== 全局因子筛选（仅首窗口执行一次）===")
    print(f"    训练期: {first_window['train'][0]} ~ {first_window['train'][1]}")
    print(f"{'='*60}")

    print(f"\n[3A.0] 构建首窗口特征缓存...")
    first_feature_cache = build_custom_feature_cache(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle_all,
        factor_cache_names=CONFIG["factor_cache_names"],
        start_time=first_segments["train"][0],
        end_time=first_segments["test"][1],
        freq="day",
    )

    print(f"\n[3A.1] 构建全量因子数据集（仅训练期）...")
    _, first_dataset_full = create_custom_dataset(
        instruments=CONFIG["instruments"],
        feature_cache=first_feature_cache,
        start_time=first_segments["train"][0],
        end_time=first_segments["train"][1],
        fit_start_time=first_segments["train"][0],
        fit_end_time=first_segments["train"][1],
        segments={"train": first_segments["train"]},
        model_type=CONFIG["model_type"],
        normalize_features=CONFIG["normalize_features"],
        neutralize_features=CONFIG["neutralize_features"],
        renormalize_features_after_neutralize=CONFIG["renormalize_features_after_neutralize"],
        normalize_labels=CONFIG["normalize_labels"],
        neutralize_labels=CONFIG["neutralize_labels"]
    )

    print(f"\n[3A.2] 执行全局因子筛选 (选取前 {CONFIG['top_k_factors']} 个)...")
    train_frame_full_fs = first_dataset_full.prepare("train")
    print(f"    >>> 训练集数据: {train_frame_full_fs.shape[0]} 行 x {train_frame_full_fs.shape[1]} 列")
    label_col_name = fs_conf["label_col"]
    if label_col_name in train_frame_full_fs.columns:
        pass
    elif ("label", label_col_name) in train_frame_full_fs.columns:
        label_col_name = ("label", label_col_name)
    print(f"    >>> label_col 实际名: {label_col_name}")

    fs_stride = max(int(CONFIG.get("feature_selection_date_stride", 1)), 1)
    if fs_stride > 1:
        sampled_dates = train_frame_full_fs.index.get_level_values('datetime').unique()[::fs_stride]
        train_frame_fs = train_frame_full_fs.loc[train_frame_full_fs.index.get_level_values('datetime').isin(sampled_dates)]
        print(f"    >>> 特征筛选抽样: 每 {fs_stride} 个交易日取 1 个，降至 {train_frame_fs.shape[0]} 行")
    else:
        train_frame_fs = train_frame_full_fs

    x_train, y_train, _ = prepare_feature_selection_data(train_frame_fs, label_col=label_col_name)
    valid_idx = y_train.dropna().index
    x_train = x_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    global_fs_result = cached_select_features(
        x_train, y_train,
        method=fs_conf["method"],
        algo=fs_conf["algo"],
        threshold=0.0,
        model_kwargs={"max_features": CONFIG["top_k_factors"], "importance_type": "gain"},
        remove_collinearity=fs_conf["remove_collinearity"]
    )

    global_selected_factor_names = list(global_fs_result.selected_features)
    print(f"\n>>> 全局因子筛选完成！固定使用以下 {len(global_selected_factor_names)} 个因子:")
    for i, fname in enumerate(global_selected_factor_names, 1):
        print(f"    {i}. {fname}")
    print("    后续所有窗口将复用此因子集，因子权重由模型独立训练自动调整。")

    # [AQR 改进] 因子冗余检测：剔除高相关因子，保留 F-score 更高者
    factor_redun_conf = CONFIG.get("factor_redundancy_check", {})
    if factor_redun_conf.get("enabled", False) and len(global_selected_factor_names) > 5:
        corr_threshold = factor_redun_conf.get("correlation_threshold", 0.95)
        print(f"\n    - 因子冗余检测 (相关性阈值 > {corr_threshold})...")
        feat_data = train_frame_fs[global_selected_factor_names].dropna()
        if len(feat_data) > 5000:
            # 时间分层抽样：每个交易日等比例抽取，避免抽样偏向特定市场阶段
            feat_data = feat_data.groupby(level='datetime', group_keys=False).apply(
                lambda x: x.sample(max(1, int(len(x) * 0.2)), random_state=42)
            )
            if len(feat_data) > 5000:
                feat_data = feat_data.sample(5000, random_state=42)
        corr_matrix = feat_data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        fs_scores = global_fs_result.feature_scores
        # 改用 stack() 一次性提取上三角所有高相关对，避免 O(n²) 冗余遍历
        high_corr_pairs = upper_tri.stack()
        high_corr_pairs = high_corr_pairs[high_corr_pairs > corr_threshold]
        high_corr_pairs = high_corr_pairs.sort_values(ascending=False)
        to_drop = set()
        for (col1, col2), _ in high_corr_pairs.items():
            if col1 in to_drop or col2 in to_drop:
                continue
            # 因子缺失 F-score 时使用 -inf，确保不被误杀
            score1 = fs_scores.get(col1, -np.inf) if hasattr(fs_scores, 'get') else -np.inf
            score2 = fs_scores.get(col2, -np.inf) if hasattr(fs_scores, 'get') else -np.inf
            if score1 == -np.inf or score2 == -np.inf:
                print(f"      [警告] 因子 {col1}(score={score1}) 或 {col2}(score={score2}) 无 F-score，保留两者")
                continue
            if score1 >= score2:
                to_drop.add(col2)
            else:
                to_drop.add(col1)
        if to_drop:
            kept = [f for f in global_selected_factor_names if f not in to_drop]
            print(f"      剔除 {len(to_drop)} 个冗余因子: {sorted(to_drop)}")
            print(f"      剩余 {len(kept)} 个因子")
            global_selected_factor_names = kept
        else:
            print(f"      未发现冗余因子")

    # [Citadel Alpha Lab 改进] ICIR 稳定性校验：剔除近期 ICIR 频繁变号的因子
    icir_conf = CONFIG.get("icir_stability_check", {})
    if icir_conf.get("enabled", False) and len(global_selected_factor_names) > max(CONFIG["top_k_factors"] // 2, 3):
        rolling_w = icir_conf["rolling_window"]
        keep_ratio = icir_conf["keep_ratio"]
        min_keep = max(int(len(global_selected_factor_names) * keep_ratio), 3)
        print(f"\n    - ICIR 稳定性校验 (窗口={rolling_w}d, 正ICIR占比>{keep_ratio}, 最少保留{min_keep}个)...")
        ic_feature_cols = [c for c in global_selected_factor_names if c in train_frame_fs.columns]
        if len(ic_feature_cols) > 3 and label_col_name in train_frame_fs.columns:
            ic_data = train_frame_fs[ic_feature_cols + [label_col_name]].dropna()
            daily_ic = ic_data.groupby(level='datetime').apply(
                lambda df: df[ic_feature_cols].corrwith(df[label_col_name], method='spearman')
            )
            if not daily_ic.empty and len(daily_ic) > rolling_w // 2:
                rolling_mean = daily_ic.rolling(window=rolling_w, min_periods=rolling_w // 2).mean()
                rolling_std = daily_ic.rolling(window=rolling_w, min_periods=rolling_w // 2).std()
                # ICIR = mean(IC)/std(IC) * sqrt(252), 乘年度化因子让数值符合行业标准
                # 注意：此处 ICIR 用于正负占比排序，乘以常数不影响排序结果
                rolling_icir = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
                pos_ratio = (rolling_icir > 0).sum() / rolling_icir.notna().sum()
                pos_ratio = pos_ratio.fillna(0).sort_values(ascending=False)
                keep_count = max(int(len(pos_ratio) * keep_ratio), 3)
                stable_factors = pos_ratio.head(keep_count).index.tolist()
                removed_ic = [f for f in global_selected_factor_names if f not in stable_factors]
                if removed_ic:
                    print(f"      剔除 {len(removed_ic)} 个低稳定性因子: {removed_ic}")
                    print(f"      保留 {len(stable_factors)} 个稳定因子")
                    global_selected_factor_names = stable_factors
                else:
                    print(f"      所有因子均通过稳定性校验")
            else:
                print(f"      daily_ic 仅 {len(daily_ic) if not daily_ic.empty else 0} 行，不足 {rolling_w // 2}，跳过")
        else:
            print(f"      特征数不足，跳过")

    del first_dataset_full, train_frame_full_fs, train_frame_fs, x_train, y_train, global_fs_result
    gc.collect()

    # ===== 3B: 滚动窗口训练（复用固定因子集，IC 衰减加权）=====
    # [Citadel Alpha Lab 改进] 跨窗口跟踪每个模型的验证集 IC（信息系数），
    # 用 EWMA 指数平滑后作为集成加权权重。当一个因子逐渐失效时，模型 IC 下降，
    # 其在集成中的权重自然衰减 → 实现"特征重要性衰减"。
    model_ic_history: dict[str, list[float]] = {}  # {model_key: [ic_history]}
    for window in CONFIG["rolling_windows"]:
        window_name = window["name"]
        print(f"\n{'='*60}")
        print(f"=== 滚动窗口训练: {window_name} ===")
        print(f"    [训练集]: {window['train'][0]} ~ {window['train'][1]}")
        print(f"    [验证集]: {window['valid'][0]} ~ {window['valid'][1]}")
        print(f"    [测试集]: {window['test'][0]} ~ {window['test'][1]}")
        print(f"    固定因子集 ({len(global_selected_factor_names)} 个): {global_selected_factor_names}")
        print(f"{'='*60}")

        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }

        print(f"\n[3B.0 - {window_name}] 构建窗口级特征缓存...")
        # [性能优化] 首窗口的特征缓存已在 3A.0 构建过，直接复用避免重复计算
        if window_name == CONFIG["rolling_windows"][0]["name"] and "first_feature_cache" in dir():
            feature_cache = first_feature_cache
            print(f"      [复用 3A 缓存] 跳过重复计算")
        else:
            feature_cache = build_custom_feature_cache(
                instruments=CONFIG["instruments"],
                feature_bundle=bundle_all,
                factor_cache_names=CONFIG["factor_cache_names"],
                start_time=segments["train"][0],
                end_time=segments["test"][1],
                freq="day",
            )

        print(f"\n[3B.1 - {window_name}] 使用固定因子集构建轻量级 DatasetH...")
        _, dataset_sub = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_cache=feature_cache,
            selected_feature_names=global_selected_factor_names,
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
            neutralize_labels=CONFIG["neutralize_labels"]
        )

        train_frame_window = dataset_sub.prepare("train")
        print(f"    >>> 训练集数据: {train_frame_window.shape[0]} 行 x {train_frame_window.shape[1]} 列")

        valid_frame = None
        try:
            valid_frame = dataset_sub.prepare("valid")
            print(f"    >>> 验证集加载成功: {valid_frame.shape}")
        except Exception as e:
            print(f"      [警告] 验证集加载失败: {e}，将跳过早停")

        dataset_sub = wrap_dataset_with_cached_train_frame(
            dataset_sub,
            train_frame=train_frame_window,
            selected_feature_names=global_selected_factor_names,
            label_names=bundle_all.label_names,
            learn_data_key=DataHandlerLP.DK_L,
            infer_data_key=DataHandlerLP.DK_I,
            valid_frame=valid_frame,
        )

        del train_frame_window
        gc.collect()

        selected_models = list(CONFIG.get("train_models", ["lgb", "xgb", "cat"]))
        model_params = CONFIG.get("model_params", {})
        models = []
        print(f"\n[3B.2 - {window_name}] 训练模型 {selected_models}...")

        if "lgb" in selected_models:
            print("    - LightGBM...")
            models.append(train_lgb_model(dataset_sub, params=model_params.get("lgb")))
        if "xgb" in selected_models:
            print("    - XGBoost...")
            models.append(train_xgb_model(dataset_sub, params=model_params.get("xgb")))
        if "cat" in selected_models:
            print("    - CatBoost (CPU)...")
            models.append(train_catboost_model(dataset_sub, params=model_params.get("cat")))

        if not models:
            raise ValueError("CONFIG['train_models'] 不能为空")
        print(f"    >>> {window_name} 所有模型训练完毕！")

        # [Citadel Alpha Lab 改进] 计算每个模型的验证集 IC，EWMA 平滑后作为集成权重
        model_ic_weights = []
        if valid_frame is not None and "label" in valid_frame.columns.get_level_values(0):
            # 使用 squeeze() 统一处理单列 DataFrame 和 Series
            actual_label = valid_frame["label"].squeeze()
            if isinstance(actual_label, pd.DataFrame):
                actual_label = actual_label.iloc[:, 0]
            for m_idx, model in enumerate(models):
                model_key = selected_models[m_idx] if m_idx < len(selected_models) else f"model_{m_idx}"
                try:
                    val_pred = model.predict(dataset_sub, segment="valid")
                    if isinstance(val_pred, pd.DataFrame):
                        val_pred = val_pred.iloc[:, 0]
                    aligned_actual = actual_label.reindex(val_pred.index).dropna()
                    valid_pred_aligned = val_pred.reindex(aligned_actual.index).dropna()
                    common_idx = aligned_actual.index.intersection(valid_pred_aligned.index)
                    # [Renaissance 改进] 最小有效样本数从 10 提高至 30，保证 Spearman IC 统计意义
                    if len(common_idx) >= 30:
                        ic_val = compute_ic(valid_pred_aligned.loc[common_idx], aligned_actual.loc[common_idx])
                        ewma_ic = compute_ic_ewma(model_ic_history, model_key, ic_val, half_life=4)
                        model_ic_weights.append(max(ewma_ic, 0.0))  # 负 IC 权重截断为 0
                        print(f"      [{model_key}] 验证 IC={ic_val:.4f}, EWMA-IC={ewma_ic:.4f}")
                    else:
                        model_ic_weights.append(1.0)
                        print(f"      [{model_key}] 验证数据不足({len(common_idx)}条<30)，权重=1.0")
                except (ValueError, KeyError) as e:
                    # [结构性问题] 如索引不匹配、列不存在——需要告警
                    print(f"      [{model_key}] IC 计算结构错误: {e}，权重=1.0")
                    model_ic_weights.append(1.0)
                except Exception as e:
                    # [非关键异常] 回退等权，不影响流程
                    print(f"      [{model_key}] IC 计算异常: {e}，权重=1.0")
                    model_ic_weights.append(1.0)
        else:
            model_ic_weights = [1.0] * len(models)
            print(f"      [警告] 无有效验证数据，所有模型使用等权重")

        # 显式归一化：确保权重之和为 1（即使 predict_ensemble_models 内部做归一化也保持明确）
        weights_arr = np.array(model_ic_weights, dtype=float)
        if weights_arr.sum() > 0:
            model_ic_weights = (weights_arr / weights_arr.sum()).tolist()
        else:
            model_ic_weights = [1.0 / len(models)] * len(models)

        if any(w != 1.0 / len(models) for w in model_ic_weights):
            print(f"      >> 集成权重 (EWMA-IC, 归一化): {[f'{w:.3f}' for w in model_ic_weights]}")
        else:
            print(f"      >> 使用等权重集成")

        print(f"\n[3B.3 - {window_name}] 测试集预测...")
        predictions = predict_ensemble_models(models, dataset_sub, segment="test", model_weights=model_ic_weights)

        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
        predictions = predictions.dropna(subset=["score"])
        # 保留原始预测分数（未经 rank 归一化），供深度分析使用
        predictions["raw_score"] = predictions["score"]
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")

        print(f"    >>> 共产生 {len(predictions)} 条测试集预测。")
        all_predictions.append(predictions)

        del dataset_sub, models, feature_cache
        gc.collect()
    # 4. 合并所有滚动窗口的样本外预测结果
    print("\n[4] 所有滚动窗口执行完毕！正在合并预测结果...")
    final_predictions = pd.concat(all_predictions)
    # 按时间排序，确保回测顺序正确
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)
    
    # [PIT 过滤] 按每一天过滤 csi500 成分股 + 退市股，杜绝未来函数和幸存者偏差
    try:
        # 1. 加载 csi500 成分股历史（读取 csi500.txt 构建 PIT 映射）
        _inst_path = Path(QLIB_DATA_DIR) / "instruments" / "csi500.txt"
        _csi500_pit = {}  # {stock: [(start_date, end_date), ...]}
        if _inst_path.exists():
            with open(_inst_path) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l:
                        continue
                    _parts = _l.split('\t')
                    if len(_parts) >= 3:
                        _code, _s, _e = _parts[0].lower(), _parts[1], _parts[2]
                        _csi500_pit.setdefault(_code, []).append((_s, _e))

        # 2. 加载退市股日期（读取 all.txt）
        _all_path = Path(QLIB_DATA_DIR) / "instruments" / "all.txt"
        _delist_pit = {}  # {stock: delist_date}
        if _all_path.exists():
            with open(_all_path) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l:
                        continue
                    _parts = _l.split('\t')
                    if len(_parts) >= 3:
                        _code, _list_d, _delist_d = _parts[0].lower(), _parts[1], _parts[2]
                        if _delist_d != '9999-12-31':
                            _delist_pit[_code] = _delist_d

        # 3. 逐日过滤
        before = len(final_predictions)
        _filtered_rows = []
        for (_dt, _inst), _row in final_predictions.iterrows():
            _dt_str = str(_dt)[:10]
            _inst_lower = _inst.lower()

            # 3a. 检查是否已退市
            _delist_d = _delist_pit.get(_inst_lower, '9999-12-31')
            if _dt_str > _delist_d:
                continue  # 退市后剔除

            # 3b. 检查当天是否在 csi500 中
            _in_csi500 = False
            if _inst_lower in _csi500_pit:
                for _s, _e in _csi500_pit[_inst_lower]:
                    if _s <= _dt_str <= _e:
                        _in_csi500 = True
                        break
            if not _in_csi500:
                continue  # 非 csi500 成分股剔除

            _filtered_rows.append(((_dt, _inst_lower), _row))

        if _filtered_rows:
            final_predictions = pd.DataFrame(
                [r[1] for r in _filtered_rows],
                index=pd.MultiIndex.from_tuples([r[0] for r in _filtered_rows], names=["datetime", "instrument"])
            )
        else:
            final_predictions = final_predictions.iloc[0:0]

        after = len(final_predictions)
        _excluded_delisted = sum(1 for _dt, _inst in final_predictions.index for _ in [1]
                                 if _delist_pit.get(_inst.lower(), '9999-12-31') != '9999-12-31')
        print(f"\n  [PIT 过滤] 前 {before} 行 → 后 {after} 行 (剔除了 {before-after} 行非 csi500/退市)")
    except Exception as e:
        print(f"  [警告] PIT 过滤失败: {e}，跳过过滤")
    
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

def _parse_args():
    parser = argparse.ArgumentParser(description="树模型训练脚本")
    parser.add_argument(
        "--config-source",
        choices=["local", "yaml"],
        default="local",
        help="参数来源：local=脚本内 [全局配置区]，yaml=加载 scripts/training/configs/ 下的 YAML",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"YAML 配置文件名（不含后缀）；为空时默认使用 {DEFAULT_YAML_CONFIG_NAME}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ml_pipeline(config_source=args.config_source, config_name=args.config)
