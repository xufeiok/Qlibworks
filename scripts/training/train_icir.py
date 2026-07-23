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

# 将项目根目录 src 文件夹加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import create_custom_dataset, build_custom_feature_cache
from qlworks.config import QLIB_DATA_DIR
import qlib
from _config import resolve_runtime_config

# ==============================================================================
# [候选池支持：从 registry/candidate_pool.json 读取准入因子列表]
# ==============================================================================
_CANDIDATE_POOL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../factor_data/registry/candidate_pool.json"
)


def _load_candidate_pool_factor_files() -> list[str] | None:
    """
    从 candidate_pool.json 读取已准入因子所在的 YAML 源文件列表。
    若候选池为空或不存在，返回 None（回退到默认 factor_files）。
    """
    if not os.path.exists(_CANDIDATE_POOL_PATH):
        return None
    try:
        with open(_CANDIDATE_POOL_PATH, "r", encoding="utf-8") as f:
            pool = json.load(f)
        factors = pool.get("factors", [])
        if not factors:
            return None
        # 收集因子所在的源 YAML 文件（去重）
        source_files = set()
        for fd in factors:
            sf = fd.get("source_file")
            if sf:
                source_files.add(sf)
        if not source_files:
            return None
        return list(source_files)
    except Exception as e:
        print(f"  [候选池读取警告] {e}，回退到默认 factor_files")
        return None


# ==============================================================================
# [全局配置区]
# ==============================================================================
DEFAULT_YAML_CONFIG_NAME = "icir_2025"
LOCAL_CONFIG = {
    "instruments": "csi500", 
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",
    
    # 五个开关的语义必须分开理解：
    # 1) normalize_features: 因子标准化开关。
    #    - linear/nn: True 时固定使用 RobustZScoreNorm
    #    - 当前项目要求各模型都必须开启；若设为 False 会直接报错中断
    # 2) neutralize_features: 因子中性化开关。
    #    - ICIR 基线不做因子中性化（保持因子原始预测能力）
    # 3) renormalize_features_after_neutralize: 中性化后是否再标准化
    # 4) normalize_labels: 标签标准化开关（横截面分位数化）
    # 5) neutralize_labels: 标签中性化开关
    "model_type": "linear", 
    "label_fields": ["Ref($close, -5) / Ref($open, -1) - 1"], 
    "label_names": ["LABEL_5D"], 
    "factor_files": ["reversal_momentum_factors"],
    "factor_cache_names": [],
    "normalize_features": True,
    "neutralize_features": False, 
    "renormalize_features_after_neutralize": False,
    "normalize_labels": True, 
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
    "top_k_factors": 50,       # 选出 ICIR 最高的 K 个因子进行加权
    "icir_window": 60,          # 滚动 ICIR 计算窗口（交易日），约一个季度
    "feature_selection_date_stride": 2,  # ICIR 计算按日期抽样步长；2=隔天
    "factor_redundancy_check": {
        "enabled": True,
        "correlation_threshold": 0.70,   # 相关性超过此阈值视为冗余，只保留 IC 最高的那个（与评估侧对齐）
    },
}

def run_icir_baseline_pipeline(config_source: str = "local", config_name: str | None = None):
    CONFIG = resolve_runtime_config(
        local_config=LOCAL_CONFIG,
        default_yaml_name=DEFAULT_YAML_CONFIG_NAME,
        config_source=config_source,
        config_name=config_name,
    )
    print("="*60)
    print("=== [基石引入] 传统多因子 ICIR 静态加权基准模型 ===")
    print("="*60)

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
    
    print("\n[2] 读取因子库的所有因子公式...")
    # 优先从候选池读取，回退到配置中的 factor_files
    pool_factor_files = _load_candidate_pool_factor_files()
    factor_files = pool_factor_files if pool_factor_files is not None else CONFIG["factor_files"]
    if pool_factor_files is not None:
        print(f">>> 从候选池读取因子源文件: {factor_files}")
    else:
        print(f">>> 使用配置中的默认 factor_files: {factor_files}")
    bundle_all = build_factor_library_bundle(factor_files)
    bundle_all.label_fields = CONFIG["label_fields"]
    bundle_all.label_names = CONFIG["label_names"]
    
    print(f">>> 成功加载 {len(bundle_all.fields)} 个因子候选池，并将预测标签设为 {bundle_all.label_names[0]} (5天收益率)。")

    # ===== 首窗口构建特征缓存（后续窗口复用）=====
    first_window = CONFIG["rolling_windows"][0]
    first_segments = {
        "train": first_window["train"],
        "valid": first_window["valid"],
        "test":  first_window["test"],
    }
    print(f"\n >>> 构建首窗口特征缓存（供所有窗口复用）...")
    first_feature_cache = build_custom_feature_cache(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle_all,
        factor_cache_names=CONFIG["factor_cache_names"],
        start_time=first_segments["train"][0],
        end_time=first_segments["test"][1],
        freq="day",
    )
    
    all_predictions = []
    
    for window_idx, window in enumerate(CONFIG["rolling_windows"]):
        window_name = window["name"]
        print(f"\n{'='*60}")
        print(f"=== 正在处理滚动窗口: {window_name} ===")
        print(f"{'='*60}")
        
        segments = {
            "train": window["train"],
            "valid": window["valid"],
            "test":  window["test"],
        }

        # 缓存复用：首窗口复用 first_feature_cache，后续窗口按需新建
        if window_idx == 0:
            feature_cache = first_feature_cache
            print(f"  [复用首窗口缓存]")
        else:
            feature_cache = build_custom_feature_cache(
                instruments=CONFIG["instruments"],
                feature_bundle=bundle_all,
                factor_cache_names=CONFIG["factor_cache_names"],
                start_time=segments["train"][0],
                end_time=segments["test"][1],
                freq="day",
            )
        
        # 3.1 构建全量因子数据集（使用 feature_cache 模式）
        print(f"\n[3.1 - {window_name}] 构建全量因子数据集 (用于计算历史 IC/IR)...")
        _, dataset_full = create_custom_dataset(
            instruments=CONFIG["instruments"],
            feature_cache=feature_cache,
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
            symmetric_orthogonalization=CONFIG["symmetric_orthogonalization"]
        )
        
        # 3.2 计算训练集（历史窗口）上每个因子的滚动 ICIR
        print(f"\n[3.2 - {window_name}] 计算训练集因子 IC/IR，确定滚动加权权重...")
        train_frame = dataset_full.prepare("train")
        
        # 提取特征和标签
        feature_cols = [c for c in train_frame.columns if c not in CONFIG["label_names"]]
        label_col = CONFIG["label_names"][0]
        
        # 日期抽样加速 IC 计算
        icir_stride = max(int(CONFIG.get("feature_selection_date_stride", 1)), 1)
        if icir_stride > 1:
            all_dates = train_frame.index.get_level_values('datetime').unique()
            sampled_dates = all_dates[::icir_stride]
            train_ic_frame = train_frame.loc[train_frame.index.get_level_values('datetime').isin(sampled_dates)]
            print(f"    IC 计算抽样: 每 {icir_stride} 天取 1 个，{len(all_dates)} → {len(sampled_dates)} 天")
        else:
            train_ic_frame = train_frame

        # [AQR 改进] 因子冗余检测：去除高相关因子，只保留 IC 最高的代表
        if CONFIG["factor_redundancy_check"]["enabled"] and len(feature_cols) > 10:
            corr_threshold = CONFIG["factor_redundancy_check"]["correlation_threshold"]
            print(f"    - 因子冗余检测 (corr > {corr_threshold} 视为冗余)...")
            # 计算特征相关矩阵
            feat_sample = train_ic_frame[feature_cols].sample(min(5000, len(train_ic_frame)))
            corr_matrix = feat_sample.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # 找出高相关的因子对
            high_corr_pairs = [(col1, col2) for col1 in upper_tri.columns for col2 in upper_tri.index 
                               if pd.notna(upper_tri.loc[col1, col2]) and upper_tri.loc[col1, col2] > corr_threshold]
            if high_corr_pairs:
                # 保留与标签相关性更高的因子，剔除冗余代表
                label_corr = feat_sample.corrwith(
                    train_ic_frame.loc[feat_sample.index, label_col], method='spearman'
                ).abs()
                to_drop = set()
                for c1, c2 in high_corr_pairs:
                    if c1 in to_drop or c2 in to_drop:
                        continue
                    if label_corr.get(c1, 0) >= label_corr.get(c2, 0):
                        to_drop.add(c2)
                    else:
                        to_drop.add(c1)
                feature_cols = [c for c in feature_cols if c not in to_drop]
                print(f"    - 冗余检测: 发现 {len(high_corr_pairs)} 对高相关，剔除 {len(to_drop)} 个冗余因子，剩余 {len(feature_cols)} 个")
        
        # 计算每日截面 Rank IC
        print(f"    计算每日 Rank IC ({len(feature_cols)} 个因子)...")
        def calc_daily_ic(df):
            return df[feature_cols].corrwith(df[label_col], method='spearman')
            
        daily_ic = train_ic_frame.groupby(level='datetime').apply(calc_daily_ic)
        
        # 计算滚动 ICIR（均值/标准差）
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
        
        # 提取测试集中的入选因子（只取训练时选中的因子交集）
        available_k = [f for f in top_k_factors if f in test_frame.columns]
        if len(available_k) < len(top_k_factors):
            print(f"    [警告] 测试集中只有 {len(available_k)}/{len(top_k_factors)} 个因子可用")
        test_features = test_frame[available_k]
        weights_k = weights[available_k]
        weights_k = weights_k / weights_k.abs().sum()
        
        # 计算最终得分：Score = \sum (Factor_i * Weight_i)
        test_score = test_features.dot(weights_k)
        
        predictions = test_score.to_frame("score")
        
        # 计算该窗口下预测分数与真实标签的 Rank IC
        test_label = test_frame[label_col]
        test_df = pd.concat([predictions["score"], test_label], axis=1)
        test_daily_ic = test_df.groupby(level='datetime').apply(lambda df: df["score"].corr(df[label_col], method='spearman'))
        test_ic_mean = test_daily_ic.mean()
        test_ic_std = test_daily_ic.std()
        test_icir = test_ic_mean / test_ic_std if test_ic_std != 0 else np.nan
        print(f">>> [测试集表现] {window_name} 预测得分与标签 {label_col} 的 Rank IC: Mean = {test_ic_mean:.4f}, ICIR = {test_icir:.4f}")
        
        # 横截面百分位排序
        predictions = predictions.dropna(subset=["score"])
        # 保留原始分数
        predictions["raw_score"] = predictions["score"]
        predictions["score"] = predictions.groupby(level="datetime")["score"].rank(pct=True, na_option="keep")
        
        print(f">>> {window_name} 预测完成！共产生 {len(predictions)} 条测试集打分。")
        all_predictions.append(predictions)
        
        del dataset_full, train_frame, test_frame, daily_ic
        if window_idx > 0:
            del feature_cache
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
    
    # 5. 保存预测结果
    output_path = os.path.join(os.path.dirname(__file__), "score_icir.csv")
    final_predictions.to_csv(output_path)
    print(f"\n>>> ICIR 传统加权预测得分已保存至: {output_path}")
    print("="*60)

def _parse_args():
    parser = argparse.ArgumentParser(description="ICIR 基线训练脚本")
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
    run_icir_baseline_pipeline(config_source=args.config_source, config_name=args.config)
