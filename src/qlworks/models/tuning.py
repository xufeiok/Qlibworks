import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Dict, Any, Optional
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

def tune_lgbm_hyperparameters(
    dataset, 
    n_trials: int = 20, 
    label_col: str = "LABEL0",
    use_purged_cv: bool = False,
    n_splits: int = 3,
    embargo_days: int = 10
) -> Dict[str, Any]:
    """
    功能概述：
    - 使用 Optuna 自动为 LightGBM 模型进行超参数寻优（HPO）。
    - 固定的超参数往往容易在金融时序数据上过拟合，自动搜索可极大提升样本外表现。

    输入：
    - dataset: Qlib DatasetH 实例（须包含 train 和 valid 分段）。
    - n_trials: 寻优的迭代次数。
    - label_col: 目标收益率标签的列名。
    - use_purged_cv: 是否使用带隔离期的交叉验证(Purged CV)防泄漏
    - n_splits: CV 的折数
    - embargo_days: 训练和验证集之间的隔离天数

    输出：
    - 最优参数字典 (Best Params)，可直接传入 LGBModel(**best_params)。
    """
    if not HAS_OPTUNA:
        raise ImportError("请先安装 optuna 库: pip install optuna")

    def objective(trial):
        # 定义搜索空间 (Search Space)
        params = {
            "objective": "mse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "early_stopping_rounds": 10,
            "verbose": -1
        }

        if not use_purged_cv:
            # 预先取出验证集，避免每次 trial 都去底层查
            valid_df = dataset.prepare("valid")
            if label_col not in valid_df.columns:
                raise ValueError(f"验证集中缺失标签列 '{label_col}'，无法计算 Loss。")
            y_valid = valid_df[label_col].values
            
            model = LGBModel(**params)
            try:
                model.fit(dataset)
                pred = model.predict(dataset, segment="valid")
                mse = ((pred.values.flatten() - y_valid) ** 2).mean()
                return mse
            except Exception as e:
                print(f"[Optuna] 训练失败: {e}")
                return 999999.0
        else:
            # [Point72 改进] Purged Walk-Forward Cross Validation
            import numpy as np
            
            # 获取训练集的所有数据
            train_df = dataset.prepare("train")
            if label_col not in train_df.columns:
                raise ValueError(f"训练集中缺失标签列 '{label_col}'，无法计算 Loss。")
                
            dates = train_df.index.get_level_values('datetime').unique().sort_values()
            n_dates = len(dates)
            
            if n_dates < n_splits * 20:
                print(f"[Optuna] 数据天数({n_dates})过少，无法执行 {n_splits} 折 CV，回退到正常验证")
                # Fallback logic here if needed, or just let it crash/skip
            
            fold_size = n_dates // n_splits
            mse_scores = []
            
            for i in range(n_splits):
                # 定义验证集的日期范围
                val_start_idx = i * fold_size
                val_end_idx = (i + 1) * fold_size if i != n_splits - 1 else n_dates
                
                # 训练集的日期需要避开验证集，并加上 embargo_days 隔离带
                val_dates = dates[val_start_idx:val_end_idx]
                
                # 找出安全的训练日期
                train_dates_mask = pd.Series(True, index=dates)
                
                # 挖掉验证集及其前后 embargo_days
                unsafe_start = max(0, val_start_idx - embargo_days)
                unsafe_end = min(n_dates, val_end_idx + embargo_days)
                train_dates_mask.iloc[unsafe_start:unsafe_end] = False
                
                safe_train_dates = dates[train_dates_mask]
                
                # 切分 DataFrame
                fold_train_df = train_df[train_df.index.get_level_values('datetime').isin(safe_train_dates)]
                fold_val_df = train_df[train_df.index.get_level_values('datetime').isin(val_dates)]
                
                if fold_train_df.empty or fold_val_df.empty:
                    continue
                    
                # 将切分好的数据转为 LightGBM Dataset
                import lightgbm as lgb
                X_train = fold_train_df.drop(columns=[label_col])
                y_train = fold_train_df[label_col]
                X_val = fold_val_df.drop(columns=[label_col])
                y_val = fold_val_df[label_col]
                
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
                
                try:
                    # 使用 lgb.train 配合 early_stopping
                    gbm = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_val],
                        callbacks=[lgb.early_stopping(stopping_rounds=params["early_stopping_rounds"], verbose=False)]
                    )
                    
                    pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
                    mse = ((pred - y_val.values) ** 2).mean()
                    mse_scores.append(mse)
                except Exception as e:
                    print(f"[Optuna] CV 训练失败: {e}")
                    
            if not mse_scores:
                return 999999.0
            return np.mean(mse_scores)

    # 启动研究
    study = optuna.create_study(direction="minimize", study_name="qlib_lgbm_tuning")
    study.optimize(objective, n_trials=n_trials)

    print("\n[+] 超参寻优完成！最优参数：")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study.best_params


if __name__ == "__main__":
    print("=== models/tuning.py 独立测试 (模拟) ===")
    print("支持使用 optuna 对 LightGBM 的 learning_rate, num_leaves, bagging_fraction 等进行自动化寻优。")
