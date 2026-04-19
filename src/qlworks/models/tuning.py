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
    label_col: str = "LABEL0"
) -> Dict[str, Any]:
    """
    功能概述：
    - 使用 Optuna 自动为 LightGBM 模型进行超参数寻优（HPO）。
    - 固定的超参数往往容易在金融时序数据上过拟合，自动搜索可极大提升样本外表现。

    输入：
    - dataset: Qlib DatasetH 实例（须包含 train 和 valid 分段）。
    - n_trials: 寻优的迭代次数。
    - label_col: 目标收益率标签的列名。

    输出：
    - 最优参数字典 (Best Params)，可直接传入 LGBModel(**best_params)。

    边界条件：
    - 若系统未安装 Optuna (pip install optuna)，将抛出异常。
    - 若验证集中找不到标签列，会直接报错并终止当前 trial。

    性能/安全注意事项：
    - 寻优过程会多次 fit 模型，耗时较长；建议使用精简的 train_df 进行搜索。
    - 寻优目标是“验证集上的均方误差(MSE)”最小化。
    """
    if not HAS_OPTUNA:
        raise ImportError("请先安装 optuna 库: pip install optuna")

    # 预先取出验证集，避免每次 trial 都去底层查
    valid_df = dataset.prepare("valid")
    if label_col not in valid_df.columns:
        raise ValueError(f"验证集中缺失标签列 '{label_col}'，无法计算 Loss。")

    y_valid = valid_df[label_col].values

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
        }

        model = LGBModel(**params)
        
        try:
            # 拟合训练集
            model.fit(dataset)
            # 在验证集上预测
            pred = model.predict(dataset, segment="valid")
            
            # 计算目标函数: MSE (Mean Squared Error)
            mse = ((pred.values.flatten() - y_valid) ** 2).mean()
            return mse
        except Exception as e:
            # 遇到非法参数组合或崩溃，直接返回极大值
            print(f"[Optuna] 训练失败: {e}")
            return 999999.0

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
