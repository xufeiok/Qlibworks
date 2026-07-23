import os
import sys
import warnings
import argparse
import copy

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
from qlworks.factors.filter_utils import filter_codes_post
import qlib
from _config import resolve_runtime_config

# ==============================================================================
# [全局配置区]
# 默认运行时优先使用这里的 CONFIG。
# 只有在命令行显式传入 `--config-source yaml` 时，才切换到 YAML 配置文件。
# ==============================================================================
DEFAULT_YAML_CONFIG_NAME = "tree_2025"
LOCAL_CONFIG = {
    # 使用沪深主板股票池（main_board.txt）。
    # main_board 是沪深主板（600/601/603/000 开头）的静态池，无 PIT 窗口，
    # 只需做股票代码交集过滤。退市股在 all.txt 中标记，由退市过滤剔除。
    "instruments": "main_board", 
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
    "factor_files": ["selected_good_factors"], # 待加载的因子文件
    "factor_cache_names": [], # DuckDB + Parquet 预计算因子（注入为 Qlib 表达式）
    "normalize_features": True, # 是否对特征进行标准化；tree=True 时固定采用 CSQuantileNorm
    "neutralize_features": False, # 是否对特征进行横截面中性化
    "renormalize_features_after_neutralize": False, # 特征中性化后是否再标准化；tree 默认关闭
    "normalize_labels": True, # 是否对标签进行横截面分位数化
    "neutralize_labels": False, # 是否对标签进行横截面中性化（行业/市值残差化）
    "use_dynamic_filter": True, # 动态过滤：剔除停牌股 + 低流动性僵尸股
    "filter_new_stocks": True,  # 后置过滤：上市不足 250 日次新股
    "filter_st": True,          # 后置过滤：ST 股票
    
    # 运行期质量闸门：剔除标签/数据异常导致的坏窗口
    "window_quality_gate": {
        "enabled": True,
        "min_valid_samples": 30,
        "min_healthy_models": 2,
    },
    
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
    "top_k_factors": 20, # 从筛选结果中取 Top 20 个因子建模
    "feature_selection_date_stride": 2, # 特征筛选按日期抽样步长；2=隔天抽样，显著降低筛选耗时
    "train_models": ["lgb", "xgb", "cat"],
    "model_params": {
        "lgb": {"num_boost_round": 150, "early_stopping_rounds": 20, "device_type": "gpu", "gpu_device_id": 0},
        "xgb": {"num_boost_round": 150, "early_stopping_rounds": 20, "device": "cuda"},
        "cat": {"num_boost_round": 150, "early_stopping_rounds": 20, "task_type": "GPU"},
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
    },
}


def get_latest_qlib_calendar_date(calendar_path: str | Path | None = None) -> str:
    """
    读取本地 Qlib 日历中的最新交易日。

    输入:
    - calendar_path: Qlib 日历文件路径；为空时默认读取 qlib_data/calendars/day.txt

    输出:
    - 最新交易日，格式 YYYY-MM-DD

    边界:
    - 日历文件不存在或为空时抛出异常，避免训练误用过期时间窗口。
    """
    path = Path(calendar_path) if calendar_path else Path(QLIB_DATA_DIR) / "calendars" / "day.txt"
    if not path.exists():
        raise FileNotFoundError(f"Qlib 日历文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        dates = [line.strip() for line in f if line.strip()]

    if not dates:
        raise ValueError(f"Qlib 日历文件为空: {path}")

    return dates[-1]


def build_effective_local_config(base_config: dict | None = None, latest_date: str | None = None) -> dict:
    """
    基于本地 Qlib 最新交易日扩展训练配置。

    输入:
    - base_config: 原始本地配置；为空时使用 LOCAL_CONFIG
    - latest_date: 指定最新交易日；为空时自动从本地 Qlib 日历读取

    输出:
    - 已按最新交易日修正 end_time/rolling_windows 的新配置

    规则:
    - 保留已有历史窗口，避免影响 2023~2025 的样本外评估
    - 当本地数据已进入 2026 年时，自动补一个 Test_2026 窗口用于当前训练/打分
    """
    config = copy.deepcopy(base_config or LOCAL_CONFIG)
    resolved_latest_date = latest_date or get_latest_qlib_calendar_date()
    latest_ts = pd.Timestamp(resolved_latest_date)

    if latest_ts > pd.Timestamp(config["end_time"]):
        config["end_time"] = resolved_latest_date

    if latest_ts >= pd.Timestamp("2026-01-01"):
        test_2026_window = {
            "name": "Test_2026",
            "train": ("2023-01-01", "2024-12-20"),
            "valid": ("2025-01-01", "2025-12-20"),
            "test": ("2026-01-01", resolved_latest_date),
        }

        windows = []
        replaced = False
        for window in config.get("rolling_windows", []):
            if window.get("name") == "Test_2026":
                windows.append(test_2026_window)
                replaced = True
            else:
                windows.append(window)
        if not replaced:
            windows.append(test_2026_window)
        config["rolling_windows"] = windows

    return config


def _batch_factor_ic_selection(feature_cache, label_expr, label_name, train_start, train_end,
                                out_dir=None, batch_size=20, top_k=60, stride=2):
    """分批因子筛选：每次合并 batch_size 个因子计算 IC，避免 OOM。

    参数：
    - feature_cache: 包含 factor_series_list 的全局缓存
    - label_expr: Qlib 标签表达式（如 "Ref($close, -1) / $close - 1"）
    - label_name: 标签名称
    - train_start, train_end: 训练期时间范围
    - batch_size: 每批处理的因子数
    - top_k: 最终选出的因子数
    - stride: 日期抽样步长（=2 表示隔天计算 IC）

    返回：
    - selected_names: 按 |IC| 降序排列的 top_k 因子名列表
    """
    from qlib.data import D
    from qlworks.models.training import compute_ic
    import gc

    series_list = feature_cache.factor_series_list
    if not series_list:
        print("    [警告] factor_series_list 为空，无法进行因子筛选")
        return []

    factor_names = [s.name for s in series_list]
    print(f"    [分批筛选] 共 {len(factor_names)} 个因子，每批 {batch_size} 个")

    # 从 Qlib 直接加载标签数据
    print(f"    [分批筛选] 加载标签 {label_name} ({train_start} ~ {train_end})...")
    all_instruments = feature_cache.resolved_instruments or list(
        set(idx[0] for s in series_list for idx in s.index)
    )
    batch_size_instr = 500
    label_frames = []
    for i in range(0, len(all_instruments), batch_size_instr):
        batch_inst = all_instruments[i:i+batch_size_instr]
        try:
            _df = D.features(
                batch_inst,
                [label_expr],
                start_time=train_start,
                end_time=train_end,
                freq="day",
            )
            if _df is not None and not _df.empty:
                label_frames.append(_df)
        except Exception:
            continue
    if not label_frames:
        raise RuntimeError("无法加载标签数据")

    label_df = pd.concat(label_frames)
    label_df.index.names = ["instrument", "datetime"]
    # [对齐] 标签索引保持 (instrument, datetime) 格式，与 warehouse 因子 swaplevel 后一致
    label_series = label_df[label_df.columns[0]].sort_index()
    label_series = label_series.rename(label_name)
    label_dates = label_series.index.get_level_values("datetime").unique()
    if stride > 1:
        label_dates = label_dates[::stride]
        label_series = label_series[label_series.index.get_level_values("datetime").isin(label_dates)]
    print(f"    [分批筛选] 标签数据: {len(label_series):,} 条，{len(label_dates)} 个交易日")

    # 分批计算 IC
    all_ic_results = {}
    for batch_start in range(0, len(factor_names), batch_size):
        batch_names = factor_names[batch_start:batch_start + batch_size]
        print(f"    [分批筛选] 处理 [{batch_start+1}-{min(batch_start+batch_size, len(factor_names))}]/{len(factor_names)}: {', '.join(batch_names[:3])}{'...' if len(batch_names)>3 else ''}")

        # 按需合并当前批次的因子
        batch_df = feature_cache.get_warehouse_df(batch_names, start_time=train_start, end_time=train_end)
        if batch_df.empty:
            continue

        # 对齐标签和因子
        if isinstance(batch_df.columns, pd.MultiIndex):
            batch_df.columns = batch_df.columns.get_level_values(1)
        batch_df = batch_df.swaplevel().sort_index() if batch_df.index.names[0] == "datetime" else batch_df.sort_index()

        common_index = batch_df.index.intersection(label_series.index)
        if len(common_index) < 100:
            continue

        batch_df = batch_df.loc[common_index]
        labels = label_series.loc[common_index]

        # 日期抽样
        if stride > 1:
            _dates = batch_df.index.get_level_values("datetime").unique()[::stride]
            _mask = batch_df.index.get_level_values("datetime").isin(_dates)
            batch_df = batch_df.loc[_mask]
            labels = labels.loc[_mask]

        # 计算每个因子的 IC
        for col in batch_df.columns:
            feat = batch_df[col].dropna()
            lab = labels.reindex(feat.index).dropna()
            common = feat.index.intersection(lab.index)
            if len(common) < 50:
                continue
            try:
                ic_val = compute_ic(feat.loc[common], lab.loc[common])
                all_ic_results[col] = ic_val
            except Exception:
                all_ic_results[col] = 0.0

        del batch_df
        gc.collect()

    # 按 |IC| 排序并返回 top_k
    ic_df = pd.Series(all_ic_results).sort_values(key=abs, ascending=False)
    selected = list(ic_df.head(top_k).index)
    if selected:
        print(f"    [分批筛选] 完成！Top {len(selected)} 因子 (|IC| 范围: {abs(ic_df[selected[-1]]):.4f} ~ {abs(ic_df[selected[0]]):.4f})")
    else:
        print("    [分批筛选] 警告：未选出任何因子，请检查数据对齐")

    if out_dir is not None:
        ic_df.to_csv(Path(out_dir) / "batch_factor_ic.csv")
        pd.Series(selected).to_csv(Path(out_dir) / "batch_selected_factors.csv", index=False, header=["factor"])

    return selected


def run_ml_pipeline(config_source: str = "local", config_name: str | None = None):
    CONFIG = resolve_runtime_config(
        local_config=build_effective_local_config(),
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

    # [1b] 验证 main_board 股票池可用且非空，确保后续训练只在沪深主板上运行
    print("\n[1b] 验证 main_board 股票池配置...")
    try:
        _inst_dir = Path(QLIB_DATA_DIR) / "instruments" / "main_board.txt"
        if not _inst_dir.exists():
            print("  [警告] main_board.txt 文件不存在！请先运行数据同步脚本生成股票池文件。")
        else:
            with open(_inst_dir) as _f:
                _main_board_stocks = set()
                for _l in _f:
                    _l = _l.strip()
                    if not _l: continue
                    _parts = _l.split()
                    if _parts:
                        _main_board_stocks.add(_parts[0].lower())
            print(f"main_board 股票池文件: {_inst_dir} ({len(_main_board_stocks)} 只股票)")
            print(f"样本: {sorted(_main_board_stocks)[:5]}")
    except Exception as e:
        print(f"  [警告] main_board 验证失败: {e}")

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

    print(f"\n[3A.0] 构建全局特征缓存（覆盖 {CONFIG['start_time']} ~ {CONFIG['end_time']}，复用至所有窗口）...")
    global_feature_cache = build_custom_feature_cache(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle_all,
        factor_cache_names=CONFIG["factor_cache_names"],
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        freq="day",
        use_dynamic_filter=CONFIG["use_dynamic_filter"],
    )
    print(f"    >>> 全局缓存构建完成（动态过滤={'是' if CONFIG['use_dynamic_filter'] else '否'}）")

    print(f"\n[3A.1] 分批因子筛选（避免 OOM）...")
    label_expr = CONFIG["label_fields"][0]
    label_name = CONFIG["label_names"][0]
    global_selected_factor_names = _batch_factor_ic_selection(
        feature_cache=global_feature_cache,
        label_expr=label_expr,
        label_name=label_name,
        train_start=first_segments["train"][0],
        train_end=first_segments["train"][1],
        out_dir=CONFIG.get("output_dir", "."),
        batch_size=CONFIG.get("feature_selection_batch_size", 20),
        top_k=CONFIG["top_k_factors"],
        stride=max(int(CONFIG.get("feature_selection_date_stride", 2)), 1),
    )

    if not global_selected_factor_names:
        raise RuntimeError("分批因子筛选失败，未选出任何因子")
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

    # 保存全局最终因子列表供后续复盘与查询
    _selected_dir = Path(__file__).parent
    _selected_path = _selected_dir / "selected_factors_tree.txt"
    _selected_archive_path = _selected_dir / f"selected_factors_tree_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    with open(_selected_path, "w", encoding="utf-8") as _f:
        for _i, _fn in enumerate(global_selected_factor_names, 1):
            _f.write(f"{_i}. {_fn}\n")
    pd.DataFrame(
        {
            "rank": range(1, len(global_selected_factor_names) + 1),
            "factor_name": global_selected_factor_names,
        }
    ).to_csv(_selected_archive_path, index=False, encoding="utf-8-sig")
    print(f"\n>>> 全局因子筛选结果已保存至: {_selected_path} ({len(global_selected_factor_names)} 个)")
    print(f">>> 因子归档文件已保存至: {_selected_archive_path}")

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

        print(f"\n[3B.0 - {window_name}] 复用全局特征缓存...")
        feature_cache = global_feature_cache

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
            use_dynamic_filter=CONFIG.get("use_dynamic_filter", False),
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

        # 窗口质量闸门：验证集样本不足或健康模型过少时跳过该窗口
        gate_cfg = CONFIG.get("window_quality_gate", {})
        if gate_cfg.get("enabled", True):
            healthy_count = sum(1 for w in model_ic_weights if w > 1e-6)
            min_healthy = gate_cfg.get("min_healthy_models", 2)
            min_samples = gate_cfg.get("min_valid_samples", 30)

            valid_sample_count = 0
            if valid_frame is not None and valid_frame.shape[0] > 0:
                valid_sample_count = valid_frame.shape[0]

            if valid_sample_count < min_samples:
                print(f"    [闸门] {window_name} 验证集样本 {valid_sample_count} < {min_samples}，跳过")
                del dataset_sub, models, feature_cache
                gc.collect()
                continue
            if healthy_count < min_healthy:
                print(f"    [闸门] {window_name} 健康模型数 {healthy_count} < {min_healthy}，跳过")
                del dataset_sub, models, feature_cache
                gc.collect()
                continue
            print(f"    [闸门] {window_name} 通过: 验证样本 {valid_sample_count}，健康模型 {healthy_count}/{len(models)}")

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
    
    # [PIT 过滤] 按每一天过滤 main_board 股票池 + 退市股 + ST/次新股，杜绝未来函数和幸存者偏差
    try:
        before = len(final_predictions)

        # 1. 加载 main_board 股票池（静态池）
        _inst_path = Path(QLIB_DATA_DIR) / "instruments" / "main_board.txt"
        _main_board_stocks = set()
        if _inst_path.exists():
            with open(_inst_path) as _f:
                for _l in _f:
                    _l = _l.strip()
                    if not _l:
                        continue
                    _parts = _l.split()
                    if _parts:
                        _main_board_stocks.add(_parts[0].lower())

        # 2. 加载退市股日期
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

        # 3. 向量化过滤：逐日分组 → 主板/退市筛查 → ST/次新过滤
        _filter_new_stocks = CONFIG.get("filter_new_stocks", True)
        _filter_st = CONFIG.get("filter_st", True)
        _filtered_parts = []
        _total_st_removed = 0

        for _date, _day_df in final_predictions.groupby(level="datetime"):
            _dt_str = str(_date)[:10]

            # 3a. 主板股票池 + 退市检查（向量化）
            _day_insts = _day_df.index.get_level_values("instrument").str.lower()
            _in_board = _day_insts.isin(_main_board_stocks)
            _not_delisted = _day_insts.map(
                lambda x: _delist_pit.get(x, "9999-12-31") >= _dt_str
            )
            _day_df = _day_df[_in_board & _not_delisted]

            if _day_df.empty:
                continue

            # 3b. ST/次新过滤
            _codes = _day_df.index.get_level_values("instrument").unique().tolist()
            _filtered_codes = filter_codes_post(
                _codes, _dt_str,
                filter_new_stocks=_filter_new_stocks,
                filter_st=_filter_st,
            )
            _total_st_removed += len(_codes) - len(_filtered_codes)

            _filtered_set = set(_filtered_codes)
            _keep = _day_df.index.get_level_values("instrument").str.lower().isin(_filtered_set)
            _day_df = _day_df[_keep]
            if not _day_df.empty:
                _filtered_parts.append(_day_df)

        if _filtered_parts:
            final_predictions = pd.concat(_filtered_parts)
            final_predictions.sort_index(level=["datetime", "instrument"], inplace=True)
        else:
            final_predictions = final_predictions.iloc[0:0]

        after = len(final_predictions)
        print(f"\n  [PIT 过滤] 前 {before} → 后 {after} (主板/退市/ST/次新共剔除 {before-after} 行)")
    except Exception as e:
        print(f"  [警告] 股票池过滤失败: {e}，跳过过滤")
    
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
