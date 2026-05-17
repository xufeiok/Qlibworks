from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, Optional, Type


from qlworks.features.builder import FeatureBundle

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
        base_infer = []
        base_learn = [{"class": "DropnaLabel"}]

        # 1. 标签(Label)中性化：常用于树模型，强制模型去拟合纯 Alpha 收益
        if neutralize_labels:
            label_neutralize = {
                "class": "CSNeutralize",
                "module_path": "qlworks.processors.neutralize",
                "kwargs": {"fields_group": "label"}
            }
            # Infer 阶段不需要用到 Label，所以只加到 learn_processors
            base_learn.append(label_neutralize)

        # 2. 特征标准化与特征中性化
        if model_type == "tree":
            # 树模型使用分位数标准化保留截面排名特征，避免极值影响
            # 【修复】Qlib 原生不支持 CSQuantileNorm，需指明 module_path 调用自定义实现
            base_infer.append({
                "class": "CSQuantileNorm", 
                "module_path": "qlworks.processors.quantile_norm",
                "kwargs": {"fields_group": "feature"}
            })
            base_learn.append({
                "class": "CSQuantileNorm", 
                "module_path": "qlworks.processors.quantile_norm",
                "kwargs": {"fields_group": "feature"}
            })
        else:
            # 线性模型/神经网络保留 Z-Score
            base_infer.append({"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}})
            base_learn.append({"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}})
            
        # 是否对特征进行线性中性化（树模型会破坏非线性，通常线性模型开启）
        if neutralize_features:
            feat_neutralize = {
                "class": "CSNeutralize",
                "module_path": "qlworks.processors.neutralize",
                "kwargs": {"fields_group": "feature"}
            }
            base_infer.append(feat_neutralize)
            base_learn.append(feat_neutralize)

        # 3. 缺失值填充（必须在最后）
        if model_type == "tree":
            # 经过分位数化后，特征在 [0, 1] 之间。默认填 0 会变成最差排名产生偏误，所以中性填充 0.5
            base_infer.append({"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0.5}})
            base_learn.append({"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0.5}})
        else:
            # Z-Score 标准化后，均值为 0，所以填充 0 是中性的
            base_infer.append({"class": "Fillna", "kwargs": {"fields_group": "feature"}})
            base_learn.append({"class": "Fillna", "kwargs": {"fields_group": "feature"}})
        
        infer_processors = base_infer
        learn_processors = base_learn
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


if __name__ == "__main__":
    print("=== features/dataset.py 独立调用示例 ===")
    
    # 演示: 尝试快速生成 Alpha158 数据集
    # 注意: 这一步需要本地 Qlib 已经初始化，否则会报错。我们这里仅作代码结构演示。
    try:
        import qlib
        from qlworks.config import QLIB_DATA_DIR
        qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")
        
        print("\n[1] 开始构建 Alpha158 数据集 (Handler + DatasetH)...")
        handler, dataset = create_alpha158_dataset(
            instruments=["600000.SH"], # 取少量数据测试
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


def create_custom_dataset(
    instruments: str | list = "csi300",
    feature_bundle: Optional[FeatureBundle] = None,
    custom_features: Optional[Dict[str, str]] = None,
    custom_labels: Optional[Dict[str, str]] = None,
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

    # Qlib 要求的特征与标签配置格式
    data_loader_config = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (feature_exprs, feature_names),
                "label": (label_exprs, label_names),
            },
            "filter_pipe": [
                # 过滤停牌和未上市股票（可选）
                {"class": "NameDFFilter", "kwargs": {"name_rule_re": ".*", "F_v": "True"}},
            ],
            "freq": freq,
        },
    }

    if infer_processors is None and learn_processors is None:
        base_infer = []
        base_learn = [{"class": "DropnaLabel"}]

        # 1. 标签(Label)中性化
        if neutralize_labels:
            label_neutralize = {
                "class": "CSNeutralize",
                "module_path": "qlworks.processors.neutralize",
                "kwargs": {"fields_group": "label"}
            }
            base_learn.append(label_neutralize)

        # 2. 特征标准化与特征中性化
        if model_type == "tree":
            base_infer.append({
                "class": "CSQuantileNorm", 
                "module_path": "qlworks.processors.quantile_norm",
                "kwargs": {"fields_group": "feature"}
            })
            base_learn.append({
                "class": "CSQuantileNorm", 
                "module_path": "qlworks.processors.quantile_norm",
                "kwargs": {"fields_group": "feature"}
            })
        else:
            base_infer.append({"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}})
            base_learn.append({"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}})
            
        if neutralize_features:
            # 【重要修复】：在线性模型流派中，如果先做中性化再 Fillna，
            # 会因为某些因子带有大量 NaN 导致 OLS 回归矩阵计算完全崩溃，最终把所有特征都变成 NaN！
            # 正确的做法是：必须先用 0 (截面均值) 填充缺失值，然后再进行行业和市值中性化
            feat_neutralize = {
                "class": "CSNeutralize",
                "module_path": "qlworks.processors.neutralize",
                "kwargs": {"fields_group": "feature"}
            }
            base_infer.append({"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}})
            base_learn.append({"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}})
            base_infer.append(feat_neutralize)
            base_learn.append(feat_neutralize)
        else:
            # 3. 缺失值填充 (非中性化流派)
            base_infer.append({"class": "Fillna", "kwargs": {"fields_group": "feature"}})
            base_learn.append({"class": "Fillna", "kwargs": {"fields_group": "feature"}})
        
        infer_processors = base_infer
        learn_processors = base_learn
    
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
