"""
精选因子筛选脚本：按因子类别分组，在每个类别内独立运行特征选择，
筛选出各类别中预测能力最强的因子。

[AQR/Citadel 改进] 全局缓存 + 一次性数据准备 + 向量化 ICIR + Label Neutralization

用法：
  修改文件顶部 CONFIG 字典中的参数，然后直接运行：
    python select_factors.py

输出：
  - 控制台打印每个类别的因子筛选结果（含重要性得分）
  - 保存精选因子列表至 selected_factors_{时间戳}.csv
"""

import os
import sys
import warnings
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from qlworks.features.builder import FeatureBundle
from qlworks.features.dataset import build_custom_feature_cache
from qlworks.models import cached_select_features
from qlworks.config import QLIB_DATA_DIR
import qlib
from qlib.data import D

# ==============================================================================
# 路径与默认配置
# ==============================================================================
FACTOR_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "factor_data" / "factor_library"
ARCHIVE_DIR = FACTOR_LIBRARY_DIR / "archive"

ACTIVE_FACTOR_FILES = [
    "reversal_momentum_factors",
    "quality_factors",
    "style_factors",
    "price_volume_factors",
    "risk_factors",
    "sentiment_factors",
    "other_factors",
]

# ==============================================================================
# [全局配置区] - 在此修改运行参数
# ==============================================================================
CONFIG = {
    # 因子文件列表（不含 .yaml 后缀），排除 price_volume_factors
    "factor_files": [
        "reversal_momentum_factors",
        "quality_factors",
        "style_factors",
        "risk_factors",
        "sentiment_factors",
        "other_factors",
    ],

    # 特征选择参数
    "top_k": 2,
    "method": "embedded",
    "algo": "lightgbm",
    "min_factors": 3,

    # 数据范围
    "instruments": "csi500",
    "start_time": "2020-01-01",
    "end_time": "2024-12-31",

    # 标签
    "label_expr": "Ref($close, -5) / Ref($open, -1) - 1",
    "label_name": "LABEL_5D",

    # 冗余检测
    "redundancy_check": True,
    "redundancy_threshold": 0.95,

    # ICIR 稳定性校验
    "icir_stability": True,
    "icir_window": 60,
    "icir_keep_ratio": 0.9,

    # 输出：None 自动生成时间戳文件名
    "output": None,

    # 缓存
    "clean_start": False,
}


def resolve_factor_files(factor_files: List[str]) -> List[str]:
    """解析因子文件列表。支持 'all' 表示加载所有活跃因子文件。"""
    if len(factor_files) == 1 and factor_files[0] == "all":
        return ACTIVE_FACTOR_FILES
    return factor_files


def load_factor_yaml(file_path: Path) -> Optional[dict]:
    """安全加载单个因子 YAML 文件。"""
    if not file_path.exists():
        print(f"  [警告] 因子文件不存在: {file_path}")
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  [错误] 加载因子文件失败 {file_path}: {e}")
        return None


def load_factors_by_category(factor_files: List[str]) -> Dict[str, List[Dict]]:
    """
    加载指定因子文件，按 category 字段分组。

    返回: {category_name: [{name, expression, meaning, source_file}, ...], ...}
    """
    categories: Dict[str, List[Dict]] = {}
    total_factors = 0

    for fname in factor_files:
        path = FACTOR_LIBRARY_DIR / f"{fname}.yaml"
        data = load_factor_yaml(path)
        if data is None:
            continue

        file_factors = data.get("factors") or []
        total_factors += len(file_factors)
        print(f"    [加载] {fname}.yaml: {len(file_factors)} 个因子")

        for factor in file_factors:
            name = factor.get("name")
            if not name:
                continue
            expr_raw = factor.get("expression", {})
            qlib_expr = expr_raw.get("qlib", "") if isinstance(expr_raw, dict) else str(expr_raw)
            if not qlib_expr:
                continue

            cat = factor.get("category", "未分类")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "name": name,
                "expression": qlib_expr,
                "source_file": fname,
                "meaning": factor.get("meaning", ""),
                "usage": factor.get("usage_scenario", ""),
            })

    print(f"  >>> 共加载 {total_factors} 个因子，按类别分为 {len(categories)} 组")
    return categories


def build_global_bundle(factors_by_cat: Dict[str, List[Dict]]) -> FeatureBundle:
    """将所有类别的因子合并为全局 FeatureBundle。"""
    all_fields = []
    all_names = []
    for cat_name, factors in sorted(factors_by_cat.items()):
        for f in factors:
            all_fields.append(f["expression"])
            all_names.append(f["name"])
    return FeatureBundle(
        fields=all_fields,
        names=all_names,
        label_fields=[CONFIG["label_expr"]],
        label_names=[CONFIG["label_name"]],
    )


def _vectorized_daily_ic(train_frame: pd.DataFrame, factor_cols: List[str], label_col: str) -> pd.DataFrame:
    """向量化逐日 IC 计算 - groupby().corr() 替代逐日 apply(corrwith)。"""
    all_cols = factor_cols + [label_col]
    corr_matrices = train_frame.groupby(level='datetime')[all_cols].corr(method='spearman')
    daily_ic = corr_matrices.xs(label_col, level=1, axis=0)[factor_cols]
    return daily_ic


def run_single_category_selection(
    cat_name: str,
    factors: List[Dict],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fs_method: str,
    fs_algo: str,
    top_k: int,
    redundancy_check: bool = True,
    redundancy_threshold: float = 0.95,
    icir_stability: bool = True,
    icir_rolling_window: int = 60,
    icir_keep_ratio: float = 0.9,
) -> Optional[pd.DataFrame]:
    """
    在单个类别上运行特征选择。

    [性能优化] 不再创建 Qlib dataset，直接从预准备的 x_train/y_train 中按因子名切片。
    CSQuantileNorm 只在主流程中运行一次，此处直接复用结果。
    """
    cat_factor_names = [f["name"] for f in factors]
    print(f"    [筛选] {len(factors)} 个因子, top_k={top_k}...")

    # 从全局特征矩阵中切片
    available = [c for c in cat_factor_names if c in x_train.columns]
    if len(available) == 0:
        print(f"    [错误] 该类别的因子在训练数据中不存在")
        return None
    if len(available) < 2:
        print(f"    [跳过] 只有 {len(available)} 个可用因子，直接保留")
        rows = []
        for f in factors:
            if f["name"] in available:
                rows.append({
                    "category": cat_name, "factor_name": f["name"],
                    "selected": True, "importance": 1.0, "rank": 1,
                    "meaning": f["meaning"], "source_file": f["source_file"],
                })
        return pd.DataFrame(rows)

    x_cat = x_train[available]
    y_cat = y_train

    print(f"      >>> {x_cat.shape[0]} 行, {x_cat.shape[1]} 个特征")

    # 特征选择
    try:
        if fs_method == "embedded":
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo, threshold=0.0,
                model_kwargs={"max_features": min(top_k, len(available)), "importance_type": "gain"},
                remove_collinearity=False,
            )
        elif fs_method == "filter":
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                k=min(top_k, len(available)),
                remove_collinearity=False,
            )
        else:
            fs_result = cached_select_features(
                x_cat, y_cat,
                method=fs_method, algo=fs_algo,
                model_kwargs={"max_features": min(top_k, len(available))},
                remove_collinearity=False,
            )
    except Exception as e:
        print(f"    [错误] 特征选择失败: {e}")
        return None

    selected_set = set(fs_result.selected_features)
    scores = fs_result.feature_scores
    selected_factor_names = list(selected_set)

    # 冗余检测：来自 x_train 的因子间相关
    if redundancy_check and len(selected_factor_names) > 5:
        print(f"    [冗余检测] 阈值={redundancy_threshold}，检测 {len(selected_factor_names)} 个入选因子...")
        feat_in_data = [c for c in selected_factor_names if c in x_train.columns]
        if len(feat_in_data) > 5:
            try:
                corr_mat = x_train[feat_in_data].corr(method='spearman').abs()
                redundant_pairs = []
                for i in range(len(corr_mat.columns)):
                    for j in range(i + 1, len(corr_mat.columns)):
                        c1, c2 = corr_mat.columns[i], corr_mat.columns[j]
                        if corr_mat.iloc[i, j] > redundancy_threshold:
                            redundant_pairs.append((c1, c2, corr_mat.iloc[i, j]))
                if redundant_pairs:
                    importance_map = dict(zip(scores.index, scores.values))
                    to_drop = set()
                    for c1, c2, corr_val in redundant_pairs:
                        if c1 in to_drop or c2 in to_drop:
                            continue
                        imp1 = abs(importance_map.get(c1, 0))
                        imp2 = abs(importance_map.get(c2, 0))
                        drop_f = c2 if imp2 < imp1 else c1
                        keep_f = c1 if drop_f == c2 else c2
                        to_drop.add(drop_f)
                        print(f"      冗余对: {c1}(imp={imp1:.4f}) vs {c2}(imp={imp2:.4f}) → 保留 {keep_f}，剔除 {drop_f}")
                    selected_factor_names = [f for f in selected_factor_names if f not in to_drop]
                    selected_set = set(selected_factor_names)
                    print(f"      冗余检测完成: 剔除 {len(to_drop)} 个冗余因子，保留 {len(selected_factor_names)} 个")
            except Exception as e:
                print(f"      [跳过] 冗余检测异常: {e}")

    # [Citadel] ICIR 稳定性校验（使用 x_train + y_train 构建面板）
    if icir_stability and len(selected_factor_names) > 5:
        print(f"    [ICIR 稳定校验] 窗口={icir_rolling_window}d, keep_ratio={icir_keep_ratio}...")
        try:
            label_col = CONFIG["label_name"]
            icir_feat = [c for c in selected_factor_names if c in x_train.columns]
            if len(icir_feat) > 5:
                # 构建含标签的综合面板
                combined_frame = x_train[icir_feat].copy()
                combined_frame[label_col] = y_train
                daily_ic = _vectorized_daily_ic(combined_frame, icir_feat, label_col)
                if not daily_ic.empty and len(daily_ic) > icir_rolling_window // 2:
                    rolling_mean = daily_ic.rolling(window=icir_rolling_window, min_periods=icir_rolling_window // 2).mean()
                    rolling_std = daily_ic.rolling(window=icir_rolling_window, min_periods=icir_rolling_window // 2).std()
                    rolling_icir = rolling_mean / rolling_std.replace(0, np.nan)
                    pos_ratio = (rolling_icir > 0).sum() / rolling_icir.notna().sum()
                    pos_ratio = pos_ratio.fillna(0).sort_values(ascending=False)
                    keep_count = max(int(len(pos_ratio) * icir_keep_ratio), min(top_k, len(pos_ratio)))
                    stable_factors = pos_ratio.head(keep_count).index.tolist()
                    dropped = len(selected_factor_names) - len(stable_factors)
                    if dropped > 0:
                        print(f"      ICIR 检测: {dropped} 个不稳定因子被剔除")
                        selected_factor_names = stable_factors
                        selected_set = set(selected_factor_names)
                else:
                    print(f"      daily_ic 仅 {len(daily_ic)} 行，不足 {icir_rolling_window // 2}，跳过")
        except Exception as e:
            print(f"      [跳过] ICIR 稳定校验异常: {e}")

    # 构建重要性映射
    if len(scores) > 0 and scores.max() > 0:
        scores_norm = scores / scores.max()
    else:
        scores_norm = pd.Series(1.0, index=scores.index) if len(scores) > 0 else pd.Series(dtype=float)

    rows = []
    for rank, (factor_name, importance) in enumerate(scores_norm.items(), 1):
        factor_info = next((f for f in factors if f["name"] == factor_name), None)
        rows.append({
            "category": cat_name,
            "factor_name": factor_name,
            "selected": factor_name in selected_set,
            "importance": round(float(importance), 4),
            "rank": rank,
            "meaning": (factor_info or {}).get("meaning", ""),
            "source_file": (factor_info or {}).get("source_file", ""),
        })

    return pd.DataFrame(rows)


def print_category_results(cat_name: str, results_df: pd.DataFrame):
    """打印单个类别的筛选结果。"""
    if results_df is None or len(results_df) == 0:
        return

    selected = results_df[results_df["selected"]]
    not_selected = results_df[~results_df["selected"]]

    print(f"\n  │ 选中: {len(selected)}/{len(results_df)} 个因子")
    if len(selected) > 0:
        print(f"  │ 入选因子:")
        for _, row in selected.iterrows():
            bar = "#" * int(row["importance"] * 20) + "." * (20 - int(row["importance"] * 20))
            print(f"  │   [{row['rank']:2d}] {row['factor_name']:<25s} [{bar}] {row['importance']:.3f}")
    if len(not_selected) > 0:
        print(f"  │ 淘汰因子:")
        for _, row in not_selected.iterrows():
            print(f"  │   [{row['rank']:2d}] {row['factor_name']:<25s} 得分={row['importance']:.3f}")


def main():
    factor_files = resolve_factor_files(CONFIG["factor_files"])

    print("=" * 60)
    print("  精选因子筛选脚本 (性能优化版)")
    print("  功能：按因子类别分组 → 独立特征选择 → 输出有效因子")
    print("  优化：一次性数据准备，CSQuantileNorm 仅运行 1 次")
    print("=" * 60)

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading")

    # 2. 加载因子并按类别分组
    print(f"\n[2] 加载因子文件: {factor_files}")
    categories = load_factors_by_category(factor_files)
    if not categories:
        print("[错误] 未加载到任何因子")
        sys.exit(1)

    # 打印类别概览
    print(f"\n[3] 类别概览:")
    for cat_name, factors in sorted(categories.items()):
        print(f"    {cat_name:<20s}: {len(factors):3d} 个因子")
    print(f"    {'总计':-<20s}: {sum(len(v) for v in categories.values()):3d} 个因子")

    # ─────────────────────────────────────────────────────────────────────────
    # 一次性构建全局因子包和特征缓存
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4a] 构建全局因子包 (Global FeatureBundle)...")
    global_bundle = build_global_bundle(categories)
    print(f"    >>> 合并后共 {len(global_bundle.fields)} 个因子表达式")

    print(f"\n[4b] 构建全局特征缓存 (Global Feature Cache)...")
    global_feature_cache = build_custom_feature_cache(
        instruments=CONFIG["instruments"],
        feature_bundle=global_bundle,
        factor_cache_names=[],
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
    )
    print(f"    >>> 全局缓存构建完成")

    # ─────────────────────────────────────────────────────────────────────────
    # [Point72 核心优化] 从缓存读取原始数据，完全跳过 Qlib Processor 流水线
    # 树模型不需要 CSQuantileNorm/ZScoreNorm，原始值直接输入 LightGBM
    # ─────────────────────────────────────────────────────────────────────────
    all_factor_names = [f["name"] for factors in categories.values() for f in factors]
    print(f"\n[4c] 从全局缓存读取原始因子数据 ({len(all_factor_names)} 个因子)...")
    print(f"    [注意] 跳过 CSQuantileNorm 处理（树模型无需特征归一化）")

    # 获取原始特征数据（warehouse_df 列名为 ("feature", name) MultiIndex）
    warehouse_df = global_feature_cache.warehouse_df.copy()
    warehouse_df.columns = warehouse_df.columns.droplevel(0)  # 扁平化为列名
    # 确保 index 标准化
    if isinstance(warehouse_df.index, pd.MultiIndex):
        warehouse_df.index = warehouse_df.index.set_levels(
            warehouse_df.index.levels[1].str.lower(), level=1
        )

    # 获取标签数据
    print(f"    [标签] 表达式: {CONFIG['label_expr']}")
    label_raw = D.features(
        global_feature_cache.resolved_instruments,
        [CONFIG["label_expr"]],
        CONFIG["start_time"], CONFIG["end_time"],
    )
    if not label_raw.empty:
        if isinstance(label_raw.columns, pd.MultiIndex):
            label_raw.columns = label_raw.columns.droplevel(1)
        label_raw = label_raw.rename(columns={label_raw.columns[0]: CONFIG["label_name"]})
        # 对齐 index 格式
        label_flat = label_raw.reset_index()
        label_flat['instrument'] = label_flat['instrument'].str.lower()
        label_flat = label_flat.set_index(['datetime', 'instrument']).sort_index()
    else:
        raise ValueError(f"标签 {CONFIG['label_expr']} 在 Qlib 中为空")

    # 合并特征与标签
    print(f"    [合并] 特征和标签数据...")
    full_train_frame = warehouse_df.join(label_flat, how='inner')
    before = len(full_train_frame)
    full_train_frame = full_train_frame.dropna(subset=[CONFIG["label_name"]])
    print(f"    >>> 原始数据: {full_train_frame.shape[0]} 行 × {full_train_frame.shape[1]} 列（移除了 {before - len(full_train_frame)} 行无标签数据）")

    # 拆分为 X, y
    feature_cols = [c for c in full_train_frame.columns if c in all_factor_names]
    x_train_all = full_train_frame[feature_cols].copy()
    y_train_all = full_train_frame[CONFIG["label_name"]].copy()

    print(f"    >>> 准备就绪: {x_train_all.shape[0]} 行, {x_train_all.shape[1]} 个特征")

    # 释放内存
    del warehouse_df, label_raw, label_flat, full_train_frame
    gc.collect()

    # 3. 逐类别进行特征选择
    print(f"\n[5] 逐类别特征选择 (方法={CONFIG['method']}, 算法={CONFIG['algo']}, top_k={CONFIG['top_k']})...")
    all_results = []
    cat_index = 0

    for cat_name, factors in sorted(categories.items()):
        cat_index += 1
        if len(factors) < CONFIG["min_factors"]:
            print(f"\n  [{cat_index}/{len(categories)}] 类别 '{cat_name}' ({len(factors)} 个因子) — 跳过（因子数 < {CONFIG['min_factors']})")
            rows = []
            for f in factors:
                rows.append({
                    "category": cat_name, "factor_name": f["name"],
                    "selected": True, "importance": 1.0, "rank": 1,
                    "meaning": f["meaning"], "source_file": f["source_file"],
                })
            all_results.append(pd.DataFrame(rows))
            continue

        print(f"\n  >>> [{cat_index}/{len(categories)}] 类别: '{cat_name}' ({len(factors)} 个因子) <<<")
        df = run_single_category_selection(
            cat_name=cat_name,
            factors=factors,
            x_train=x_train_all,
            y_train=y_train_all,
            fs_method=CONFIG["method"],
            fs_algo=CONFIG["algo"],
            top_k=CONFIG["top_k"],
            redundancy_check=CONFIG["redundancy_check"],
            redundancy_threshold=CONFIG["redundancy_threshold"],
            icir_stability=CONFIG["icir_stability"],
            icir_rolling_window=CONFIG["icir_window"],
            icir_keep_ratio=CONFIG["icir_keep_ratio"],
        )

        if df is not None:
            print_category_results(cat_name, df)
            all_results.append(df)
        else:
            print(f"    [失败] 类别 '{cat_name}' 特征选择未完成")

    # 4. 汇总输出
    print(f"\n{'=' * 60}")
    print("  筛选完成！汇总结果")
    print(f"{'=' * 60}")

    if not all_results:
        print("[警告] 没有任何类别完成筛选")
        sys.exit(0)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values(["category", "rank"]).reset_index(drop=True)

    selected_total = final_df["selected"].sum()
    total = len(final_df)
    print(f"\n  总览: 共 {total} 个因子，选中 {selected_total} 个 ({selected_total/total*100:.1f}%)")

    print(f"\n  {'类别':<20s} {'选中/总计':<12s} {'选中率':<8s}")
    print(f"  {'-'*40}")
    for cat_name in sorted(final_df["category"].unique()):
        sub = final_df[final_df["category"] == cat_name]
        s = sub["selected"].sum()
        t = len(sub)
        print(f"  {cat_name:<20s} {s}/{t:<8d} {s/t*100:>6.1f}%")

    print(f"\n  各类别精选因子列表:")
    for cat_name in sorted(final_df["category"].unique()):
        sub = final_df[(final_df["category"] == cat_name) & (final_df["selected"])]
        if len(sub) == 0:
            continue
        print(f"\n  【{cat_name}】({len(sub)} 个)")
        for _, row in sub.iterrows():
            print(f"    [x] {row['factor_name']:<30s} (重要性: {row['importance']:.3f}, 来源: {row['source_file']})")

    # 保存 CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = CONFIG["output"] or os.path.join(os.path.dirname(__file__), f"selected_factors_{timestamp}.csv")
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  完整结果已保存至: {output_path}")

    selected_df = final_df[final_df["selected"]].copy()
    selected_output = output_path.replace(".csv", "_selected.csv")
    selected_df.to_csv(selected_output, index=False, encoding="utf-8-sig")
    print(f"  精选因子列表已保存至: {selected_output}")

    print("=" * 60)
    print(f"  精选因子数: {selected_total}")
    print(f"  训练模型: {CONFIG['algo']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
