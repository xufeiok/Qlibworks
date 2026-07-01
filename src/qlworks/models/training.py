from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, Tuple

import numpy as np
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


def _filter_finite_feature_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤特征/标签中的无效样本，避免 XGBoost 等模型因 NaN/inf 标签直接崩溃。
    """
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        label_df = df["label"] if "label" in df.columns.get_level_values(0) else pd.DataFrame(index=df.index)
        feature_df = df["feature"] if "feature" in df.columns.get_level_values(0) else pd.DataFrame(index=df.index)
    else:
        label_cols = [c for c in df.columns if "LABEL" in str(c)]
        feature_cols = [c for c in df.columns if c not in label_cols]
        label_df = df[label_cols]
        feature_df = df[feature_cols]

    label_mask = np.isfinite(label_df.to_numpy()).all(axis=1) if not label_df.empty else np.ones(len(df), dtype=bool)
    feature_mask = np.isfinite(feature_df.to_numpy()).all(axis=1) if not feature_df.empty else np.ones(len(df), dtype=bool)
    valid_mask = label_mask & feature_mask
    if valid_mask.all():
        return df
    return df.loc[valid_mask].copy()


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
        "max_depth": 6,
        "num_leaves": 24,
        "min_child_samples": 20,
        "verbose": -1,
        "device": "gpu" if _USE_GPU else "cpu",
    }
    fit_params = {
        "num_boost_round": 1000,
        "early_stopping_rounds": 30,
    }
    if params:
        base_params.update({k: v for k, v in params.items() if k not in fit_params})
        fit_params.update({k: v for k, v in params.items() if k in fit_params})
    model = LGBModel(**base_params)
    model.fit(dataset, **fit_params)
    return model


def train_xgb_model(dataset, params: Dict[str, object] = None):
    """
    功能概述：
    - 使用 Qlib 的 `XGBModel` 训练。XGBoost 对噪声的容忍度较高，与 LGBM 是经典的集成搭档。
    - GPU 自动检测：有 GPU 时使用 CUDA 加速，否则回退 CPU。
    """
    from qlib.contrib.model.xgboost import XGBModel
    from qlib.data.dataset.handler import DataHandlerLP
    from qlworks.features.dataset import PreparedDatasetView

    base_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 4,
    }
    fit_params = {
        "num_boost_round": 1000,
        "early_stopping_rounds": 30,
    }
    if _USE_GPU:
        base_params.update({"tree_method": "hist", "device": "cuda"})
    if params:
        # Qlib 的 XGBModel.fit 使用 num_boost_round / early_stopping_rounds，
        # 不能把这两个参数直接塞进构造器，否则会被底层 xgboost 当成无效参数。
        if "n_estimators" in params and "num_boost_round" not in params:
            params = dict(params)
            params["num_boost_round"] = params.pop("n_estimators")
        base_params.update({k: v for k, v in params.items() if k not in fit_params})
        fit_params.update({k: v for k, v in params.items() if k in fit_params})
    cached_results = {}
    for segment in ("train", "valid"):
        try:
            frame = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        except Exception:
            continue
        cleaned = _filter_finite_feature_label_frame(frame)
        dropped = len(frame) - len(cleaned)
        if dropped > 0:
            print(f"    [XGB 清洗] {segment}: 删除 {dropped} 条含 NaN/inf 的样本")
        cached_results[(segment, ("feature", "label"), DataHandlerLP.DK_L)] = cleaned
        if "feature" in cleaned.columns.get_level_values(0):
            cached_results[(segment, "feature", DataHandlerLP.DK_L)] = cleaned["feature"]

    wrapped_dataset = PreparedDatasetView(dataset, cached_prepare_results=cached_results)
    model = XGBModel(**base_params)
    model.fit(wrapped_dataset, **fit_params)
    return model


def train_catboost_model(dataset, params: Dict[str, object] = None):
    """
    功能概述：
    - 使用 Qlib 的 `CatBoostModel` 训练。CatBoost 能更好地处理类别特征，且在金融数据上通常抗过拟合能力较强。
    - GPU 自动检测：有 GPU 时使用 GPU 训练，否则回退 CPU。
    """
    from catboost import CatBoostRegressor, Pool
    from qlib.data.dataset.handler import DataHandlerLP

    base_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.1,
        "depth": 6,
        "thread_count": 4,
    }
    fit_params = {
        "num_boost_round": 1000,
        "early_stopping_rounds": 30,
    }
    task_type = "GPU" if _USE_GPU else "CPU"
    if params:
        if "iterations" in params and "num_boost_round" not in params:
            params = dict(params)
            params["num_boost_round"] = params.pop("iterations")
        if "task_type" in params:
            task_type = str(params["task_type"]).upper()
        base_params.update({k: v for k, v in params.items() if k not in fit_params and k != "task_type"})
        fit_params.update({k: v for k, v in params.items() if k in fit_params})
    if task_type not in {"CPU", "GPU"}:
        raise ValueError(f"CatBoost task_type 仅支持 CPU/GPU，当前为: {task_type}")

    df_train, df_valid = dataset.prepare(
        ["train", "valid"],
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    df_train = _filter_finite_feature_label_frame(df_train)
    df_valid = _filter_finite_feature_label_frame(df_valid)
    if df_train.empty or df_valid.empty:
        raise ValueError("Empty data from dataset, please check your dataset config.")

    x_train, y_train = df_train["feature"], df_train["label"]
    x_valid, y_valid = df_valid["feature"], df_valid["label"]

    if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
        y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
    else:
        raise ValueError("CatBoost doesn't support multi-label training")

    train_pool = Pool(data=x_train.values, label=y_train_1d)
    valid_pool = Pool(data=x_valid.values, label=y_valid_1d)

    model_params = dict(base_params)
    model_params["task_type"] = task_type
    model_params["iterations"] = fit_params["num_boost_round"]
    model_params["early_stopping_rounds"] = fit_params["early_stopping_rounds"]
    model_params.setdefault("verbose", 20)

    cat_model = CatBoostRegressor(**model_params)
    cat_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    class CatBoostWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, dataset, segment="test"):
            x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
            return pd.Series(self.model.predict(x_test.values), index=x_test.index)

        def get_feature_importance(self, *args, **kwargs) -> pd.Series:
            return pd.Series(self.model.get_feature_importance(*args, **kwargs))

    return CatBoostWrapper(cat_model)


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


def compute_ic(predicted: pd.Series, actual: pd.Series) -> float:
    """
    功能概述：
    - 计算 Spearman Rank IC（信息系数），衡量预测值与实际值的排序一致性。
    输入：
    - predicted: 预测得分 Series。
    - actual: 实际标签 Series。
    输出：
    - 浮点数 IC 值（-1~1），数据不足时返回 0.0。
    边界条件：
    - 需至少 10 个有效样本，否则返回 0.0。
    """
    from scipy.stats import spearmanr
    combined = pd.DataFrame({"pred": predicted, "actual": actual}).dropna()
    if len(combined) < 10:
        return 0.0
    return float(spearmanr(combined["pred"], combined["actual"])[0])


def compute_ic_ewma(model_ic_history: dict, model_key: str, new_ic: float, half_life: int = 4) -> float:
    """
    功能概述：
    - 对模型的 IC 序列计算指数移动加权平均（EWMA），用于平滑跨窗口的性能波动。
    - 半衰期参数控制历史衰减速度：半衰期越短，旧窗口的 IC 权重衰减越快。
    输入：
    - model_ic_history: 历史 IC 字典 {model_key: [ic_values]}。
    - model_key: 模型标识（如 "lgb", "xgb", "cat"）。
    - new_ic: 当前窗口的新 IC 值。
    - half_life: EWMA 半衰期（窗口数），默认 4 个窗口。
    输出：
    - 平滑后的 IC 值。
    """
    if model_key not in model_ic_history:
        model_ic_history[model_key] = []
    model_ic_history[model_key].append(new_ic)
    
    history = model_ic_history[model_key]
    if len(history) == 1:
        return history[0]
    
    # 指数衰减权重：最近窗口权重最高
    n = len(history)
    decay = np.log(2) / half_life  # 衰减系数
    weights = np.exp(-decay * np.arange(n - 1, -1, -1))  # 最近权重最大
    weights /= weights.sum()
    smoothed_ic = float(np.sum(np.array(history) * weights))
    return smoothed_ic


def predict_ensemble_models(
    models: list, dataset, segment: str = "test",
    model_weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    功能概述：
    - 对多个训练好的模型（如 LGBM + XGB + CatBoost）进行集成预测。
    - 支持等权重（默认）或 IC 加权平均。
    输入：
    - models: 模型实例列表。
    - dataset: 待预测的数据集 (Qlib Dataset 对象)。
    - segment: 预测的时间段名称，默认为 "test"。
    - model_weights: 各模型的权重列表，None 时使用等权重。
    输出：
    - 包含 'score' 列的 DataFrame。
    边界条件：
    - model_weights 长度需与 models 一致。
    """
    if not models:
        raise ValueError("模型列表不能为空")
        
    predictions = []
    for model in models:
        pred = model.predict(dataset, segment=segment)
        predictions.append(pred)
    
    if model_weights is None:
        # 等权重平均
        ensemble_pred = pd.concat(predictions, axis=1).mean(axis=1)
    else:
        if len(model_weights) != len(predictions):
            raise ValueError(f"model_weights 长度 ({len(model_weights)}) 需与 models 长度 ({len(predictions)}) 一致")
        total_weight = sum(model_weights)
        if total_weight <= 0:
            # 所有权重为 0 时回退等权重
            ensemble_pred = pd.concat(predictions, axis=1).mean(axis=1)
        else:
            # 加权平均
            normalized_weights = np.array(model_weights, dtype=float) / total_weight
            weighted_pred = sum(pred * w for pred, w in zip(predictions, normalized_weights))
            ensemble_pred = weighted_pred
    
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

    # [安全防护] 移除 y 中的无穷大值，否则 sklearn 会报 "Input y contains infinity"
    finite_mask = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    if not finite_mask.all():
        count = (~finite_mask).sum()
        print(f"    清理 {count} 条包含无穷大值的样本")
        x_train = x_train.loc[finite_mask]
        y_train = y_train.loc[finite_mask]

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
            from qlib.data.dataset.handler import DataHandlerLP
            test_frame = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
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
    print("=== models/training.py 独立调用示例（非正式训练入口） ===")
    import numpy as np
    
    # 1. 制造一份已经做完特征选择的“瘦身”数据
    df_train = pd.DataFrame({
        "MA5": np.random.rand(100),
        "MACD": np.random.rand(100),
        "RSI": np.random.rand(100),
        "LABEL0": np.random.rand(100) * 0.1 # 收益率标签
    })
    
    print("\n[警告] 当前是 training.py 的演示模式，只用于验证模型封装是否可调用。")
    print("真实训练请运行: e:/Quant/Qlibworks/scripts/training/train_tree.py")
    print(f"\n[1] 输入模型的特征矩阵: {df_train.columns.tolist()} (含 LABEL0)")
    
    # 2. 调用基线线性回归模型 (注意：这里用的是我们手写的基线，它需要手动剥离 LABEL0)
    # 如果是 LGBM，你应该传 Qlib Dataset 对象
    model, x_train, y_train = train_linear_baseline(df_train)
    
    print("\n[2] 训练完成！线性模型权重 (Coef):")
    for col, coef in zip(x_train.columns, model.coef_):
        print(f"  - {col}: {coef:.4f}")
        
    print("\n说明: 上述输出只是随机样本上的线性回归示例，不代表真实策略训练结果。")
    print("真实的 Qlib 训练/打分流程，请运行 scripts/training 下的训练脚本。")
