from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, List, Optional, Type


from qlworks.features.builder import FeatureBundle


# DuckDB + Parquet 预计算因子到 Qlib 表达式的映射
# 训练时直接用 Qlib 引擎从 .bin 数据求值，无需从 Parquet 加载
FACTOR_CACHE_EXPRESSIONS = {
    "ret_1d": "$close / Ref($close, 1) - 1",
    "ma_5": "Mean($close, 5)",
    "price_position_20": "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))",
}


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


def create_dataset_from_handler(handler, segments: Dict[str, tuple]):
    """
    功能概述：
    - 基于已有 Qlib handler 创建 `DatasetH`，统一训练/验证/测试切分入口。
    输入：
    - handler: 已实例化的数据处理器。
    - segments: 时间切分字典。
    输出：
    - `DatasetH` 数据集对象。
    边界条件：
    - segments 至少应包含 `train`。
    性能/安全注意事项：
    - 只做配置装配，不主动触发全部数据计算。
    """
    from qlib.data.dataset import DatasetH

    return DatasetH(handler=handler, segments=segments)


def create_alpha158_dataset(
    instruments="csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2020-12-31",
    fit_start_time: str = "2020-01-01",
    fit_end_time: str = "2020-06-30",
    freq: str = "day",
    model_type: str = "tree", # ["tree", "linear", "nn"]
    neutralize_features: bool = False, # 是否对特征 X 进行中性化 (推荐线性模型开启)
    neutralize_labels: bool = False,   # 是否对标签 Y 进行中性化 (推荐树模型开启)
    infer_processors: Optional[list] = None,
    learn_processors: Optional[list] = None,
    segments: Optional[Dict[str, tuple]] = None,
):
    """
    功能概述：
    - 构建 Alpha158 标准数据集，适合作为传统机器学习基线。
    输入：
    - instruments/时间区间/处理器配置。
    输出：
    - `(handler, dataset)` 二元组。
    边界条件：
    - 处理器缺失时使用课程代码中常见的稳健标准化与缺失值填充。
    性能/安全注意事项：
    - Alpha158 特征量适中，适合快速验证与调参。
    - 注意：默认配置未包含市值中性化与行业中性化。若需截面中性化，
      可在 learn_processors 中引入 CSZScoreNorm 或 CSRobustZScoreNorm。
    """
    from qlib.contrib.data.handler import Alpha158

    # [Point72 ML 研究员提示]
    # 对于 XGBoost/LightGBM 等基于树的模型，截面 Z-Score 会破坏时间序列上的绝对动量信息（单调性）。
    # 推荐将 `RobustZScoreNorm` 替换为 `CSQuantileNorm` (截面分位数化)，或在树模型中直接传入原始值。
    # 仅当使用 Ridge/Lasso 线性模型或 神经网络 时，才需要保留 Z-Score 标准化。
    if infer_processors is None and learn_processors is None:
        infer_processors, learn_processors = _build_processors(
            model_type=model_type,
            neutralize_features=neutralize_features,
            neutralize_labels=neutralize_labels,
        )
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
        feature_exprs = list(feature_bundle.fields)
        feature_names = list(feature_bundle.names)
        label_exprs = list(feature_bundle.label_fields)
        label_names = list(feature_bundle.label_names)
    elif custom_features is not None and custom_labels is not None:
        feature_exprs = list(custom_features.values())
        feature_names = list(custom_features.keys())
        label_exprs = list(custom_labels.values())
        label_names = list(custom_labels.keys())
    else:
        raise ValueError("必须提供 feature_bundle，或者同时提供 custom_features 和 custom_labels。")

    # 注入 DuckDB + Parquet 预计算因子的 Qlib 表达式
    # 训练时 Qlib 直接从 .bin 文件求值，和 Parquet 结果一致
    if factor_cache_names:
        for name in factor_cache_names:
            expr = FACTOR_CACHE_EXPRESSIONS.get(name)
            if expr is None:
                raise ValueError(f"未知因子 `{name}`，可用: {list(FACTOR_CACHE_EXPRESSIONS.keys())}")
            feature_exprs.append(expr)
            feature_names.append(name)
            print(f"    [因子注入] {name}: {expr}")

    # Qlib 要求的特征与标签配置格式
    data_loader_config = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (feature_exprs, feature_names),
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
        num_workers=8,
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
        print(f"\n[!] 演示跳过: {e} (需确保 pyqlib 安装且有本地数据)")
