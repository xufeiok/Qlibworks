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

import pandas as pd
import numpy as np
import gc
from pathlib import Path
from qlib.data.dataset.handler import DataHandlerLP

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import (
    create_custom_dataset,
    build_custom_feature_cache,
    wrap_dataset_with_cached_train_frame,
)
from qlworks.models.training import train_ridge_model, predict_ensemble_models
from qlworks.models import prepare_feature_selection_data, cached_select_features
from qlworks.config import QLIB_DATA_DIR
import qlib
from _config import resolve_runtime_config

# ==============================================================================
# [全局配置区]
# ==============================================================================
DEFAULT_YAML_CONFIG_NAME = "linear_2025"
LOCAL_CONFIG = {
    "instruments": "csi500",
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",

    # 五个开关的语义必须分开理解，避免把"标准化"和"中性化"混淆：
    # 1) normalize_features: 因子标准化开关。
    #    - linear/nn: True 时固定使用 RobustZScoreNorm（稳健 ZScore 标准化）
    #    - 当前项目要求各模型都必须开启；若设为 False 会直接报错中断
    # 2) neutralize_features: 因子中性化开关。
    #    - 使用 CSNeutralize 剥离行业/市值暴露
    #    - linear 默认 True
    # 3) renormalize_features_after_neutralize: 因子中性化后是否再标准化。
    #    - linear/nn 默认 True，更符合"标准化 -> 中性化 -> 再标准化"的研究流程
    # 4) normalize_labels: 标签标准化开关。
    #    - 当前默认使用 CSQuantileNorm 对标签做横截面分位数化
    # 5) neutralize_labels: 标签中性化开关。
    #    - 使用 CSNeutralize 对标签做行业/市值残差化
    #    - 只有当你明确要学习"残差 alpha"时再开启
    "model_type": "linear",
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"],
    "label_names": ["LABEL_5D"],
    "factor_files": ["reversal_momentum_factors"],
    "factor_cache_names": [],
    "normalize_features": True,
    "neutralize_features": True,
    "renormalize_features_after_neutralize": True,
    "normalize_labels": True,
    "neutralize_labels": False,

    # [Renaissance 改进] 线性模型不需要对称正交化（ridge自带L2正则天然抗多重共线性）
    "symmetric_orthogonalization": False,

    "rolling_windows": [
        {"name": "Test_2023", "train": ("2020-01-01", "2021-12-20"), "valid": ("2022-01-01", "2022-12-20"), "test":  ("2023-01-01", "2023-12-31")},
        {"name": "Test_2024", "train": ("2021-01-01", "2022-12-20"), "valid": ("2023-01-01", "2023-12-20"), "test":  ("2024-01-01", "2024-12-31")},
        {"name": "Test_2025", "train": ("2022-01-01", "2023-12-20"), "valid": ("2024-01-01", "2024-12-20"), "test":  ("2025-01-01", "2025-12-31")},
    ],
    "top_k_factors": 6,
    "feature_selection_date_stride": 2,   # 因子筛选按日期抽样步长；2=隔天抽样
    "icir_stability_check": {"enabled": True, "rolling_window": 60, "keep_ratio": 0.9},
    "factor_redundancy_check": {
        "enabled": True,
        "correlation_threshold": 0.95,
    },
    "feature_selection": {"method": "filter", "algo": "f_regression", "label_col": "LABEL_5D", "remove_collinearity": False},
}

def run_ml_pipeline(config_source: str = "local", config_name: str | None = None):
    CONFIG = resolve_runtime_config(
        local_config=LOCAL_CONFIG,
        default_yaml_name=DEFAULT_YAML_CONFIG_NAME,
        config_source=config_source,
        config_name=config_name,
    )
    print("="*60)
    print("=== 线性模型：首窗口全局因子筛选 + 滚动窗口 Ridge 训练 ===")
    print("="*60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="loky", maxtasksperchild=None)

    # [1b] 验证 csi500 成分股可用且非空
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

    # 2. 加载因子库
    print("\n[2] 读取因子库 (Factor Library) 的所有因子公式...")
    factor_files = CONFIG["factor_files"]
    bundle_all = build_factor_library_bundle(factor_files)
    bundle_all.label_fields = CONFIG["label_fields"]
    bundle_all.label_names = CONFIG["label_names"]
    print(f">>> 成功加载 {len(bundle_all.fields)} 个因子候选池，预测标签: {bundle_all.label_names[0]} (5天收益率)。")

    all_predictions = []
    fs_conf = CONFIG["feature_selection"]

    # ===== 3A: 首窗口全局因子筛选（仅执行一次，后续窗口复用）=====
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

    print(f"\n[3A.0] 构建首窗口特征缓存（覆盖全部窗口时间范围）...")
    first_feature_cache = build_custom_feature_cache(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle_all,
        factor_cache_names=CONFIG["factor_cache_names"],
        start_time=first_segments["train"][0],
        end_time=CONFIG["rolling_windows"][-1]["test"][1],
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
        # [Citadel Alpha Lab 改进] 筛选环境与训练环境保持一致（中性化+再标准化）
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
        k=CONFIG["top_k_factors"],
        remove_collinearity=fs_conf["remove_collinearity"]
    )

    global_selected_factor_names = list(global_fs_result.selected_features)
    print(f"\n>>> 全局因子筛选完成！固定使用以下 {len(global_selected_factor_names)} 个因子:")
    for i, fname in enumerate(global_selected_factor_names, 1):
        print(f"    {i}. {fname}")
    print("    后续所有窗口将复用此因子集，避免每窗口重选导致自适应过拟合。")

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
            # 因子缺失 F-score 时使用 -inf，确保不被误杀（原值为 0 会导致低分因子总被剔除）
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
        print(f"\n    - ICIR 稳定性校验 (窗口={rolling_w}d, 正ICIR占比>{keep_ratio})...")
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
                print(f"      daily_ic 仅 {len(daily_ic) if not daily_ic.empty else 0} 行，不足 {rolling_w // 2}，跳过")
        else:
            print(f"      特征数不足，跳过")

    del first_dataset_full, train_frame_full_fs, train_frame_fs, x_train, y_train, global_fs_result
    gc.collect()

    # ===== 3B: 滚动窗口训练（复用固定因子集）=====
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
        if window_name == CONFIG["rolling_windows"][0]["name"]:
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
            neutralize_labels=CONFIG["neutralize_labels"],
        )

        train_frame_window = dataset_sub.prepare("train")
        print(f"    >>> 训练集数据: {train_frame_window.shape[0]} 行 x {train_frame_window.shape[1]} 列")

        valid_frame = None
        try:
            valid_frame = dataset_sub.prepare("valid")
            print(f"    >>> 验证集加载成功: {valid_frame.shape} (Ridge用内部CV, valid仅供分析用)")
        except Exception as e:
            print(f"      [警告] 验证集加载失败: {e}")

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

        print(f"\n[3B.2 - {window_name}] 训练 Ridge 模型 (TimeSeriesCV=3折)...")
        ridge_model = train_ridge_model(dataset_sub, params={"cv": 3})
        print(f"    >>> {window_name} Ridge 模型训练完毕！")

        print(f"\n[3B.3 - {window_name}] 测试集预测...")
        predictions = predict_ensemble_models([ridge_model], dataset_sub, segment="test")

        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
        predictions = predictions.dropna(subset=["score"])
        # 保留原始预测分数，供深度分析使用
        predictions["raw_score"] = predictions["score"]
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")

        print(f"    >>> 共产生 {len(predictions)} 条测试集预测。")
        all_predictions.append(predictions)

        del dataset_sub, ridge_model, feature_cache
        gc.collect()

    # 4. 合并所有滚动窗口的样本外预测结果
    print("\n[4] 所有滚动窗口执行完毕！正在合并预测结果...")
    final_predictions = pd.concat(all_predictions)
    final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)

    # [PIT 过滤] 按每一天过滤 csi500 成分股 + 退市股
    try:
        _inst_path = Path(QLIB_DATA_DIR) / "instruments" / "csi500.txt"
        _csi500_pit = {}
        if _inst_path.exists():
            with open(_inst_path) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l: continue
                    _parts = _l.split('\t')
                    if len(_parts) >= 3:
                        _code, _s, _e = _parts[0].lower(), _parts[1], _parts[2]
                        _csi500_pit.setdefault(_code, []).append((_s, _e))

        _all_path = Path(QLIB_DATA_DIR) / "instruments" / "all.txt"
        _delist_pit = {}
        if _all_path.exists():
            with open(_all_path) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l: continue
                    _parts = _l.split('\t')
                    if len(_parts) >= 3:
                        _code, _list_d, _delist_d = _parts[0].lower(), _parts[1], _parts[2]
                        if _delist_d != '9999-12-31':
                            _delist_pit[_code] = _delist_d

        before = len(final_predictions)
        _filtered_rows = []
        for (_dt, _inst), _row in final_predictions.iterrows():
            _dt_str = str(_dt)[:10]
            _inst_lower = _inst.lower()
            _delist_d = _delist_pit.get(_inst_lower, '9999-12-31')
            if _dt_str > _delist_d:
                continue
            _in_csi500 = False
            if _inst_lower in _csi500_pit:
                for _s, _e in _csi500_pit[_inst_lower]:
                    if _s <= _dt_str <= _e:
                        _in_csi500 = True
                        break
            if not _in_csi500:
                continue
            _filtered_rows.append(((_dt, _inst_lower), _row))

        if _filtered_rows:
            final_predictions = pd.DataFrame(
                [r[1] for r in _filtered_rows],
                index=pd.MultiIndex.from_tuples([r[0] for r in _filtered_rows], names=["datetime", "instrument"])
            )
        else:
            final_predictions = final_predictions.iloc[0:0]

        after = len(final_predictions)
        print(f"\n  [PIT 过滤] 前 {before} 行 → 后 {after} 行 (剔除了 {before-after} 行非 csi500/退市)")
    except Exception as e:
        print(f"  [警告] PIT 过滤失败: {e}，跳过过滤")

    print(f">>> 合并完成！总测试集跨度: {final_predictions.index.get_level_values('datetime').min().date()} 至 {final_predictions.index.get_level_values('datetime').max().date()}")
    print(final_predictions.head(10))

    # 5. 保存预测结果
    output_path = os.path.join(os.path.dirname(__file__), "score_linear.csv")
    final_predictions.to_csv(output_path)
    print(f"\n>>> 线性模型预测得分已保存至: {output_path}")
    print("="*60)

def _parse_args():
    parser = argparse.ArgumentParser(description="线性模型训练脚本")
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
