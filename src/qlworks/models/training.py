from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression


def _detect_gpu() -> bool:
    """
    检测当前环境是否有可用的 GPU。

    Returns:
        True 如果检测到 CUDA GPU，否则 False。
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except (ImportError, OSError):
        pass
    try:
        import cupy
        return True
    except ImportError:
        pass
    # 尝试直接检查 nvidia-smi
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except FileNotFoundError:
        pass  # nvidia-smi not in PATH
    except Exception:
        pass
    return False


_USE_GPU = _detect_gpu()


def prepare_split_frames(dataset) -> Dict[str, pd.DataFrame]:
    """
    功能概述：
    - 从 `DatasetH` 统一提取 train/valid/test 数据框。
    输入：
    - dataset: Qlib `DatasetH` 对象。
    输出：
    - `{segment: dataframe}` 字典。
    边界条件：
    - 若某段不存在则自动跳过。
    性能/安全注意事项：
    - 延迟到需要时才 prepare，减少无效计算。
    """
    frames = {}
    for segment in ("train", "valid", "test"):
        try:
            frames[segment] = dataset.prepare(segment)
        except Exception:
            continue
    return frames


def train_lgb_model(dataset, params: Dict[str, object] = None):
    """
    功能概述：
    - 使用 Qlib 的 `LGBModel` 训练多因子回归模型。
    - GPU 自动检测：有 GPU 时使用 GPU 加速，否则静默回退 CPU。
    输入：
    - dataset: Qlib 数据集对象。
    - params: 可覆盖的 LightGBM 参数。
    输出：
    - 训练完成的 Qlib 模型对象。
    """
    from qlib.contrib.model.gbdt import LGBModel

    base_params = {
        "loss": "mse",
        "colsample_bytree": 0.8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "n_estimators": 1000,
        "early_stopping_rounds": 30,
        "max_depth": 6,
        "num_leaves": 64,
        "min_child_samples": 20,
        "verbose": -1,
        "device": "gpu" if _USE_GPU else "cpu",
    }
    if params:
        base_params.update(params)
    model = LGBModel(**base_params)
    model.fit(dataset)
    return model


def train_xgb_model(dataset, params: Dict[str, object] = None):
    """
    功能概述：
    - 使用 Qlib 的 `XGBModel` 训练。XGBoost 对噪声的容忍度较高，与 LGBM 是经典的集成搭档。
    - GPU 自动检测：有 GPU 时使用 CUDA 加速，否则回退 CPU。
    """
    from qlib.contrib.model.xgboost import XGBModel

    base_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 1000,
        "early_stopping_rounds": 30,
        "n_jobs": 4,
    }
    if _USE_GPU:
        base_params.update({"tree_method": "hist", "device": "cuda"})
    if params:
        base_params.update(params)
    model = XGBModel(**base_params)
    model.fit(dataset)
    return model


def train_catboost_model(dataset, params: Dict[str, object] = None):
    """
    功能概述：
    - 使用 Qlib 的 `CatBoostModel` 训练。CatBoost 能更好地处理类别特征，且在金融数据上通常抗过拟合能力较强。
    - GPU 自动检测：有 GPU 时使用 GPU 训练，否则回退 CPU。
    """
    from qlib.contrib.model.catboost_model import CatBoostModel

    base_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.1,
        "iterations": 1000,
        "early_stopping_rounds": 30,
        "depth": 6,
        "thread_count": 4,
    }
    if _USE_GPU:
        base_params["task_type"] = "GPU"
    if params:
        base_params.update(params)
    model = CatBoostModel(**base_params)
    model.fit(dataset)
    return model


def train_lstm_model(dataset, params: Dict[str, object] = None):
    """
    功能概述：
    - 使用 Qlib 封装的 PyTorch LSTM 模型，专为时间序列特征（如 Alpha360）设计。
    输入：
    - dataset: 必须是按时间序列切片的 Dataset（如 Alpha360）。
    """
    from qlib.contrib.model.pytorch_lstm import LSTMModel

    # 自动检测特征维度
    try:
        sample = dataset.prepare("train")
        n_features = sample.shape[1] - 1  # 减去标签列
    except Exception:
        n_features = 6

    base_params = {
        "d_feat": n_features,     # 自动推断每个时间步的特征数
        "hidden_size": 64,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "n_epochs": 50,
        "lr": 1e-3,
        "early_stop": 10,
        "batch_size": 800,
        "metric": "loss",
        "loss": "mse",
        "n_jobs": -1,
        "GPU": 0 if _USE_GPU else -1,  # 自动检测 GPU
    }
    if params:
        base_params.update(params)
    model = LSTMModel(**base_params)
    model.fit(dataset)
    return model


def predict_ensemble_models(models: list, dataset, segment: str = "test") -> pd.DataFrame:
    """
    功能概述：
    - 对多个训练好的模型（如 LGBM + XGB + CatBoost）进行等权重集成预测。
    输入：
    - models: 模型实例列表。
    - dataset: 待预测的数据集 (Qlib Dataset 对象)。
    - segment: 预测的时间段名称，默认为 "test"。
    输出：
    - 包含 'score' 列的 DataFrame。
    """
    if not models:
        raise ValueError("模型列表不能为空")
        
    predictions = []
    for model in models:
        pred = model.predict(dataset, segment=segment)
        predictions.append(pred)
        
    # 等权重平均 (也可以在这里拓展为加权平均或 Stacking)
    ensemble_pred = pd.concat(predictions, axis=1).mean(axis=1)
    
    # 转换为 Qlib 要求的标准输出格式
    return pd.DataFrame(ensemble_pred, columns=["score"])


def train_ridge_model(dataset, params: Dict[str, object] = None):
    """
    功能概述：
    - 使用带时间序列交叉验证 (TimeSeriesSplit) 的 Ridge 回归训练，并加入样本时间衰减权重。
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np

    train_frame = dataset.prepare("train")
    
    # 动态识别标签列名 (支持 LABEL0, LABEL_5D 等)
    label_col = next((col for col in train_frame.columns if "LABEL" in str(col)), None)
    if label_col is None:
        raise ValueError("训练数据缺少 LABEL 标签列。")

    x_train = train_frame.drop(columns=[label_col]).fillna(0.0)
    y_train = train_frame[label_col].fillna(train_frame[label_col].median())

    # [Citadel Alpha Lab 改进] 指数时间衰减权重 (Exponential Decay Weighting)
    # 距离当前越近的样本权重越大。半衰期设定为 252 个交易日 (约1年)。
    dates = x_train.index.get_level_values('datetime')
    unique_dates = np.sort(dates.unique())
    # 映射每一天到距离最后一天的天数差
    # 修复 numpy.ndarray 没有 days 属性的问题：先转换为 pd.Series 计算 diff，再转换为天数
    dates_series = pd.Series(unique_dates)
    days_diff = (dates_series.iloc[-1] - dates_series).dt.days.values
    # 计算每一天的权重: w = 0.5 ^ (days / half_life)
    half_life = 252.0
    date_weights = np.power(0.5, days_diff / half_life)
    weight_map = dict(zip(unique_dates, date_weights))
    sample_weights = dates.map(weight_map).values

    # [Point72 改进] 使用 TimeSeriesSplit 进行时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    base_params = {
        "alphas": np.logspace(-4, 4, 20),
        "cv": tscv,
    }
    if params:
        base_params.update(params)

    model = RidgeCV(**base_params)
    model.fit(x_train.values, y_train.values, sample_weight=sample_weights)
    
    print(f"    - Ridge 最佳正则化系数 (Alpha): {model.alpha_:.4f}")
    
    # 包装为一个符合 Qlib model 接口的类
    class RidgeWrapper:
        def __init__(self, ridge_model, feature_cols, label_col):
            self.model = ridge_model
            self.feature_cols = feature_cols
            self.label_col = label_col
            
        def predict(self, dataset, segment="test"):
            test_frame = dataset.prepare(segment)
            x_test = test_frame.drop(columns=[self.label_col], errors="ignore").fillna(0.0)
            # 保证特征顺序一致
            x_test = x_test.reindex(columns=self.feature_cols, fill_value=0.0)
            preds = self.model.predict(x_test.values)
            return pd.Series(preds, index=x_test.index, name="score")

    return RidgeWrapper(model, x_train.columns.tolist(), label_col)


def train_linear_baseline(train_frame: pd.DataFrame) -> Tuple[LinearRegression, pd.DataFrame, pd.Series]:
    """
    功能概述：
    - 训练一个线性回归基线模型，用于快速验证数据与特征是否具备可学习性。
    输入：
    - train_frame: 含 `LABEL0` 列的训练数据。
    输出：
    - `(model, features, labels)`。
    边界条件：
    - `LABEL0` 缺失时抛出异常。
    性能/安全注意事项：
    - 线性基线训练极快，适合作为首轮烟雾测试。
    """
    if "LABEL0" not in train_frame.columns:
        raise ValueError("训练数据缺少 LABEL0 列，无法训练线性基线模型。")
    x_train = train_frame.drop(columns=["LABEL0"]).fillna(0.0)
    y_train = train_frame["LABEL0"].fillna(train_frame["LABEL0"].median())
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model, x_train, y_train


if __name__ == "__main__":
    print("=== models/training.py 独立调用示例 ===")
    import numpy as np
    
    # 1. 制造一份已经做完特征选择的“瘦身”数据
    df_train = pd.DataFrame({
        "MA5": np.random.rand(100),
        "MACD": np.random.rand(100),
        "RSI": np.random.rand(100),
        "LABEL0": np.random.rand(100) * 0.1 # 收益率标签
    })
    
    print(f"\n[1] 输入模型的特征矩阵: {df_train.columns.tolist()} (含 LABEL0)")
    
    # 2. 调用基线线性回归模型 (注意：这里用的是我们手写的基线，它需要手动剥离 LABEL0)
    # 如果是 LGBM，你应该传 Qlib Dataset 对象
    model, x_train, y_train = train_linear_baseline(df_train)
    
    print("\n[2] 训练完成！线性模型权重 (Coef):")
    for col, coef in zip(x_train.columns, model.coef_):
        print(f"  - {col}: {coef:.4f}")
        
    print("\n说明: 在真实的 pipeline 中，推荐传入 dataset 对象并调用 train_lgb_model。")
