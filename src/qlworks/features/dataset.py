from __future__ import annotations

import os
import sys
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, List, Optional, Type


from qlworks.features.builder import FeatureBundle


# DuckDB + Parquet 预计算因子到 Qlib 表达式的映射
# 训练时直接用 Qlib 引擎从 .bin 数据求值，和 Parquet 结果一致
FACTOR_CACHE_EXPRESSIONS = {
    "ret_1d": "$close / Ref($close, 1) - 1",
    "ma_5": "Mean($close, 5)",
    "price_position_20": "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))",
}


def _load_factors_from_warehouse(feature_bundle, start_time, end_time):
    """
    尝试从 warehouse 加载因子数据
    
    加载顺序：
    1. 优先从 factor_data/warehouse/{因子名}/{年份}.parquet 加载
    2. 如果 warehouse 中没有，则返回 None，让上层使用表达式构建
    
    Args:
        feature_bundle: 因子配置对象
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        (loaded_factors_dict, remaining_factors_list) 或 (None, all_factors_list)
    """
    from pathlib import Path
    from qlworks.config import WAREHOUSE_DIR
    
    warehouse_base = Path(WAREHOUSE_DIR)
    loaded_factors = {}
    remaining_factors = []
    
    # 遍历所有需要加载的因子（从 names 获取因子名）
    for field_name in feature_bundle.names:
        factor_dir = warehouse_base / field_name
        
        # 检查 warehouse 中是否有该因子的 parquet 文件
        if factor_dir.exists():
            parquet_files = sorted(factor_dir.glob("*.parquet"))
            if parquet_files:
                # 有 parquet 文件，尝试加载
                dfs = []
                for file in parquet_files:
                    # 根据文件名判断年份是否在时间范围内
                    try:
                        file_year = int(file.stem)
                        if start_time and file_year < int(start_time[:4]):
                            continue
                        if end_time and file_year > int(end_time[:4]):
                            continue
                    except ValueError:
                        pass  # 非年份文件名，直接加载
                    
                    try:
                        df = pd.read_parquet(file)
                        dfs.append(df)
                    except Exception as e:
                        print(f"    [警告] 读取 {file} 失败：{e}")
                
                if dfs:
                    # 合并所有年份数据
                    df = pd.concat(dfs, ignore_index=True)
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df.set_index(['instrument', 'datetime'])
                    
                    # 提取 value 列作为因子值
                    if 'value' in df.columns:
                        loaded_factors[field_name] = df['value']
                        print(f"    [warehouse 加载] {field_name}: {len(df):,} 条记录")
                        continue
        
        # warehouse 中没有，需要后续用表达式构建
        remaining_factors.append(field_name)
    
    return loaded_factors, remaining_factors


def _build_processors(
    model_type: str = "tree",
    neutralize_features: bool = False,
    neutralize_labels: bool = False,
    symmetric_orthogonalization: bool = False,
) -> tuple[list, list]:
    """
    工厂函数：根据模型流派和中性化需求构建 Processor 流水线。

    统一了 create_alpha158_dataset 和 create_custom_dataset 中重复的 Processor 装配逻辑。

    Args:
        model_type: 模型类型 ("tree" / "linear" / "nn")
        neutralize_features: 是否对特征进行截面中性化
        neutralize_labels: 是否对标签进行截面分位数化
        symmetric_orthogonalization: 是否开启对称正交化（仅 linear/nn）

    Returns:
        (infer_processors, learn_processors) 二元组
    """
    base_infer: list = []
    base_learn: list = [{"class": "DropnaLabel"}]

    # 1. 标签处理：转换为横截面分位数
    if neutralize_labels:
        label_rank = {
            "class": "CSQuantileNorm",
            "module_path": "qlworks.processors.quantile_norm",
            "kwargs": {"fields_group": "label"},
        }
        base_learn.append(label_rank)

    # 2. 特征标准化
    if model_type == "tree":
        norm_cfg = {
            "class": "CSQuantileNorm",
            "module_path": "qlworks.processors.quantile_norm",
            "kwargs": {"fields_group": "feature"},
        }
        base_infer.append(norm_cfg)
        base_learn.append(norm_cfg)
    else:
        norm_cfg = {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}}
        base_infer.append(norm_cfg)
        base_learn.append(norm_cfg)

    # 3. 特征中性化（线性模型推荐开启，树模型不推荐）
    if neutralize_features:
        feat_neutralize = {
            "class": "CSNeutralize",
            "module_path": "qlworks.processors.neutralize",
            "kwargs": {"fields_group": "feature"},
        }
        # 线性模型：先 Fillna 再中性化，避免 NaN 导致 OLS 崩溃
        base_infer.append({"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}})
        base_learn.append({"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}})
        base_infer.append(feat_neutralize)
        base_learn.append(feat_neutralize)
    else:
        # 非中性化流派：先 Fillna 收尾
        fill_value = 0.5 if model_type == "tree" else None
        fill_kwargs = {"fields_group": "feature"}
        if fill_value is not None:
            fill_kwargs["fill_value"] = fill_value
        base_infer.append({"class": "Fillna", "kwargs": fill_kwargs})
        base_learn.append({"class": "Fillna", "kwargs": fill_kwargs})

    # 4. 对称正交化（仅 linear/nn，消除共线性）
    if symmetric_orthogonalization and model_type != "tree":
        feat_ortho = {
            "class": "CSSymmetricOrthogonalize",
            "module_path": "qlworks.processors.orthogonalize",
            "kwargs": {"fields_group": "feature"},
        }
        base_infer.append(feat_ortho)
        base_learn.append(feat_ortho)

    return base_infer, base_learn


def create_dataset_from_handler(handler, segments):
    """
    功能概述：
    - 从 Handler 构建 DatasetH，并支持动态切换 segment。
    输入：
    - handler: Qlib DataHandler 对象。
    - segments: 数据切分配置，如 {"train": ("2020-01-01", "2020-06-30"), ...}。
    输出：
    - DatasetH 对象。
    """
    from qlib.data.dataset import DatasetH

    dataset = DatasetH(handler, segments)
    return dataset


def create_alpha158_dataset(
    instruments: str | list = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2020-06-30",
    freq: str = "day",
    infer_processors: Optional[list] = None,
    learn_processors: Optional[list] = None,
    segments: Optional[Dict[str, tuple]] = None,
    model_type: str = "tree",
    **kwargs,
):
    """
    功能概述：
    - 构建 Alpha158 标准数据集，适合树模型（如 LightGBM/XGBoost）。
    输入：
    - 股票池、时间区间、切分配置。
    输出：
    - `(handler, dataset)` 二元组。
    边界条件：
    - 如果处理器未指定，自动根据模型类型匹配最佳流水线。
    性能/安全注意事项：
    - 使用 Qlib 内置的 Alpha158 处理器，无需手动配置。
    """
    from qlib.contrib.data.handler import Alpha158

    infer_processors = infer_processors or [
        {"class": "CSQuantileNorm", "module_path": "qlworks.processors.quantile_norm", "kwargs": {"fields_group": "feature"}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
    ]
    learn_processors = learn_processors or [
        {"class": "DropnaLabel"},
        {"class": "CSQuantileNorm", "module_path": "qlworks.processors.quantile_norm", "kwargs": {"fields_group": "feature"}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
    ]
    
    segments = segments or {
        "train": (fit_start_time, fit_end_time),
        "valid": ("2020-07-01", "2020-09-30"),
        "test": ("2020-10-01", end_time),
    }
    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        freq=freq,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )
    dataset = create_dataset_from_handler(handler, segments)
    return handler, dataset


def create_custom_dataset(
    instruments: str | list = "csi300",
    feature_bundle: Optional[FeatureBundle] = None,
    custom_features: Optional[Dict[str, str]] = None,
    custom_labels: Optional[Dict[str, str]] = None,
    factor_cache_names: Optional[List[str]] = None,
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2020-06-30",
    freq: str = "day",
    model_type: str = "tree", # ["tree", "linear", "nn"]
    neutralize_features: bool = False, # 是否对特征 X 进行中性化
    neutralize_labels: bool = False,   # 是否对标签 Y 进行中性化
    infer_processors: Optional[list] = None,
    learn_processors: Optional[list] = None,
    segments: Optional[Dict[str, tuple]] = None,
    symmetric_orthogonalization: bool = False,
):
    """
    功能概述：
    - 构建完全自定义的数据集，支持传入 FeatureBundle 或 任意因子表达式字典。
    - 优先从 factor_data/warehouse 加载已有的单因子 parquet 文件
    - warehouse 中没有的因子才使用表达式构建
    
    加载顺序：
    1. warehouse/{因子名}/{年份}.parquet (已有单因子文件)
    2. factor_cache_names 中定义的表达式
    3. feature_bundle 中定义的表达式
    
    输入：
    - feature_bundle: 来自 builder.py 的 FeatureBundle 对象 (推荐)。
    - custom_features: 特征字典，格式如 `{"MA5": "Mean($close, 5)"}` (作为备选)。
    - custom_labels: 标签字典，格式如 `{"LABEL": "Ref($close, -2)/$close - 1"}`。
    输出：
    - `(handler, dataset)` 二元组。
    边界条件：
    - feature_bundle 优先级高于 custom_features/custom_labels。
    性能/安全注意事项：
    - 使用底层的 DataHandlerLP 灵活组装，完全兼容 Qlib 的 Processor 流水线。
    """
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.data.handler import check_transform_proc

    # 解析特征和标签配置
    if feature_bundle is not None:
        all_feature_exprs = list(feature_bundle.fields)
        all_feature_names = list(feature_bundle.names)
        label_exprs = list(feature_bundle.label_fields)
        label_names = list(feature_bundle.label_names)
    elif custom_features is not None and custom_labels is not None:
        all_feature_exprs = list(custom_features.values())
        all_feature_names = list(custom_features.keys())
        label_exprs = list(custom_labels.values())
        label_names = list(custom_labels.keys())
    else:
        raise ValueError("必须提供 feature_bundle，或者同时提供 custom_features 和 custom_labels。")

    # [重要改进] 优先从 warehouse 加载因子数据
    loaded_factors = {}
    final_feature_exprs = []
    final_feature_names = []
    
    if feature_bundle is not None:
        # 尝试从 warehouse 加载因子
        loaded_factors, remaining_fields = _load_factors_from_warehouse(
            feature_bundle, start_time, end_time
        )
        
        # ===== 修复：warehouse 加载的因子也要加入表达式 =====
        # loaded_factors 中的因子数据存在但无法注入 QlibDataLoader，
        # QlibDataLoader 只能通过表达式从 .bin 数据计算特征。
        # 因此即使 warehouse 有数据，也必须添加对应的 Qlib 表达式，
        # 让所有因子都能进入特征选择流程。
        for field_name in feature_bundle.names:
            try:
                idx = feature_bundle.names.index(field_name)
                expr = feature_bundle.fields[idx]
                final_feature_exprs.append(expr)
                final_feature_names.append(field_name)
                if field_name in loaded_factors:
                    print(f"    [表达式构建·已缓存] {field_name}: {len(loaded_factors[field_name]):,} 条warehouse数据可用")
                else:
                    print(f"    [表达式构建] {field_name}: {expr}")
            except ValueError:
                print(f"    [警告] 因子 {field_name} 在 names 列表中未找到")
    else:
        # 没有 feature_bundle，使用原始逻辑
        final_feature_exprs = all_feature_exprs
        final_feature_names = all_feature_names
    
    # 注入 DuckDB + Parquet 预计算因子的 Qlib 表达式（作为额外增量因子）
    if factor_cache_names:
        for name in factor_cache_names:
            # 如果已在 feature_bundle 中（含 warehouse 加载的），跳过避免重复
            if name in final_feature_names:
                print(f"    [跳过] {name} 已在 feature_bundle 中")
                continue
            
            expr = FACTOR_CACHE_EXPRESSIONS.get(name)
            if expr is None:
                raise ValueError(f"未知因子 `{name}`，可用：{list(FACTOR_CACHE_EXPRESSIONS.keys())}")
            final_feature_exprs.append(expr)
            final_feature_names.append(name)
            print(f"    [因子注入] {name}: {expr}")
    
    # Qlib 要求的特征与标签配置格式
    data_loader_config = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (final_feature_exprs, final_feature_names),
                "label": (label_exprs, label_names),
            },
            "freq": freq,
        },
    }

    if infer_processors is None and learn_processors is None:
        infer_processors, learn_processors = _build_processors(
            model_type=model_type,
            neutralize_features=neutralize_features,
            neutralize_labels=neutralize_labels,
            symmetric_orthogonalization=symmetric_orthogonalization,
        )
    
    segments = segments or {
        "train": (fit_start_time, fit_end_time),
        "valid": ("2020-07-01", "2020-09-30"),
        "test": ("2020-10-01", end_time),
    }

    # 核心：将 fit_start_time 和 fit_end_time 注入到需要它们的 Processor 中（如 RobustZScoreNorm）
    infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
    learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

    handler = DataHandlerLP(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        data_loader=data_loader_config,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )
    dataset = create_dataset_from_handler(handler, segments)
    return handler, dataset


def create_alpha360_dataset(
    instruments="csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2020-06-30",
    freq: str = "day",
    infer_processors: Optional[list] = None,
    learn_processors: Optional[list] = None,
    segments: Optional[Dict[str, tuple]] = None,
):
    """
    功能概述：
    - 构建 Alpha360 标准数据集，适合深度学习或高维表达能力更强的模型。
    输入：
    - 股票池、时间区间、切分配置。
    输出：
    - `(handler, dataset)` 二元组。
    边界条件：
    - 需要更高内存与计算资源。
    性能/安全注意事项：
    - 推荐在资源允许时使用，并优先配合批量缓存。
    - 注意：对于 Alpha360 的量价序列特征，通常不做截面中性化，
      而是保持时序原始比例，由深度学习模型（如时序网络）自行提取截面和时序模式。
    """
    from qlib.contrib.data.handler import Alpha360

    infer_processors = infer_processors or [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    learn_processors = learn_processors or [
        {"class": "DropnaLabel"},
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    
    segments = segments or {
        "train": (fit_start_time, fit_end_time),
        "valid": ("2020-07-01", "2020-09-30"),
        "test": ("2020-10-01", end_time),
    }
    handler = Alpha360(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        freq=freq,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )
    dataset = create_dataset_from_handler(handler, segments)
    return handler, dataset
 
class DebuggablePipeline: 
    def __init__(self, processors, name='pipeline'): 
        self.processors = processors 
        self.name = name 
    def __call__(self, df): 
        import sys 
        for i, proc in enumerate(self.processors): 
            before = df.shape 
            try: 
                df = proc(df) 
            except Exception as e: 
                raise RuntimeError(f'[{self.name}] step {i} failed: {e}') 
            after = df.shape 
            if before[1] != after[1]: 
                print(f'[DBG] [{self.name}] step {i}: cols {before[1]} to ', file=sys.stderr) 
        return df 


if __name__ == "__main__":
    print("=== features/dataset.py 独立调用示例 ===")

    try:
        import qlib
        from qlworks.config import QLIB_DATA_DIR
        qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")

        print("\n[1] 开始构建 Alpha158 数据集 (Handler + DatasetH)...")
        handler, dataset = create_alpha158_dataset(
            instruments=["600000.SH"],
            start_time="2020-01-02",
            end_time="2020-01-31",
            fit_start_time="2020-01-02",
            fit_end_time="2020-01-15"
        )

        print("\n[2] 数据集构建成功！")
        print("提取训练集 (Train Frame) 的前两行：")
        df_train = dataset.prepare("train")
        print(df_train.head(2))

    except Exception as e:
        print(f"\n[!] 演示跳过：{e} (需确保 pyqlib 安装且有本地数据)")
