"""
模块说明：量化特征选择 (Feature Selection) 引擎

在多因子量化研究中，我们通常会挖掘出成百上千个因子（如 Alpha158、Alpha360，甚至上千个自建因子）。
如果不加筛选直接把所有因子喂给模型，会导致：
1. 过拟合 (Overfitting)：模型学到了很多毫无意义的噪声，实盘就亏钱。
2. 维度灾难 (Curse of Dimensionality)：训练极度缓慢，占用大量内存。
3. 多重共线性 (Multicollinearity)：很多因子高度相关（比如 MA5 和 MA10），互相干扰线性模型的权重。

因此，这个文件实现了业界标准的三大特征选择方法，帮你从“海量垃圾因子”中淘出“真金白银”：

1. 过滤法 (Filter) - `filter_feature_selection`
   - 【原理】：不依赖任何具体的机器学习模型，纯粹用统计学方法（如 F检验、互信息）挨个给因子打分。
   - 【应用场景】：当因子数量极其庞大（比如 > 500 个）时。
   - 【什么时候用】：作为“第一道海选关卡”。速度极快，几秒钟就能筛掉最没用的因子，但缺点是它看不出“因子之间的组合效应”。

2. 包装法 (Wrapper) - `wrapper_feature_selection`
   - 【原理】：用一个真实的模型（如线性回归）去试。每次训练后，把最没用的因子踢掉一个，然后再训练，一直递归消除（RFE），直到剩下你想要的数量。
   - 【应用场景】：当因子数量中等（比如 < 100 个），且你希望找出“完美配合的因子组合”时。
   - 【什么时候用】：作为“精细打磨关卡”。效果通常最好，但极其极其慢。

3. 嵌入法 (Embedded) - `embedded_feature_selection`
   - 【原理】：最聪明的方法。把特征选择过程“嵌在”模型训练的内部。比如 Lasso 回归在训练时会自动把没用的因子权重压成绝对的 0；随机森林在建树的过程中会自动计算每个因子的分裂贡献度（Feature Importance）。
   - 【应用场景】：现代量化最主流的做法！兼顾了过滤法的速度和包装法的准确性。
   - 【什么时候用】：通常作为主力筛选手段。用 Lasso 筛选线性因子，用 Random Forest 筛选非线性因子。

典型调用流水线：
1. 数据准备：x_train, y_train, x_test = prepare_feature_selection_data(train_frame, test_frame)
2. 特征筛选：result = select_features(x_train, y_train, method="embedded", method_kwargs={"method": "lasso"})
3. 应用筛选：x_train_clean, x_test_clean = apply_feature_selection(result, x_train, x_test)
4. 最终训练：模型拿 x_train_clean 去正式训练。
"""

from __future__ import annotations

import os
import sys


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, LinearRegression


@dataclass
class FeatureSelectionResult:
    """
    功能概述：
    - 统一封装一次特征选择的结果，便于后续训练、评估与记录。
    
    输出数据结构详情：
    - method (str): 特征选择方法名称，如 "wrapper:rfe", "embedded:lasso"。
    - selected_features (Sequence[str]): 最终被选中的特征名称列表。
        - 例如: ['MA5', 'MACD', 'Volume_Ratio']。这是你最需要的数据，用来给 DataFrame 瘦身。
    - feature_scores (pd.Series): 所有特征的得分或排名序列。
        - 索引 (Index) 是特征名，值 (Value) 是得分/排名。
        - 在嵌入法 (Lasso/RF) 中：值越大代表特征越重要 (Importance)。
        - 在包装法 (RFE) 中：值代表排名 (Rank)，1 表示最好 (被选中)，2 表示次之，以此类推。
    - params (Dict[str, object]): 本次选择使用的超参数记录（如 k值，阈值等），用于实验复盘。
    """

    method: str
    selected_features: Sequence[str]
    feature_scores: pd.Series
    params: Dict[str, object]


def prepare_feature_selection_data(
    train_frame: pd.DataFrame,
    test_frame: Optional[pd.DataFrame] = None,
    label_col: str = "LABEL0",
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    功能概述：
    - 为特征选择准备干净的训练/测试数据，统一处理缺失值与标签列。
    
    在流水线中的位置 (前后步骤)：
    - 【前置步骤】：数据从 `dataset.py` 中生成 `DatasetH`，调用 `dataset.prepare("train")` 得到带有 MultiIndex(datetime, instrument) 的原始 Pandas DataFrame。
    - 【当前步骤】：因为 sklearn 的特征选择算法 (SelectKBest, RFE, Lasso) 不认识 Qlib 的 Dataset 格式，也不能容忍任何 NaN 空值，且需要把 X(特征) 和 y(标签) 拆开。这个函数就是做这个“拆解和清洗”的桥梁。
    - 【后置步骤】：将清洗拆分好的 (x_train, y_train) 喂给下方的 `filter_feature_selection` 或 `wrapper_feature_selection` 等算法进行特征筛选。
    
    输入：
    - train_frame/test_frame: 含特征与标签的数据表。
    - label_col: 标签列名称。
    输出：
    - `(x_train, y_train, x_test)`。
    边界条件：
    - 标签列缺失时抛出异常。
    性能/安全注意事项：
    - 测试集缺失值必须使用训练集的均值填充，避免前视偏差（Data Leakage）。
    """
    if label_col not in train_frame.columns:
        raise ValueError(f"训练数据缺少标签列 {label_col}")

    x_train = train_frame.drop(columns=[label_col]).copy()
    y_train = train_frame[label_col].copy()

    train_means = x_train.mean(numeric_only=True)
    x_train = x_train.fillna(train_means).fillna(0.0)
    
    # [Quant All-Star Team 改进]: 绝对不能用中位数填补缺失的收益率标签 (Label)！
    # 如果一只股票当天没有未来收益率（例如停牌、涨跌停买不进），我们必须在训练中剔除这行样本，
    # 而不能假设它的收益率是全市场的均值/中位数，否则会引入极大的噪声和前视偏差。
    # 所以我们这里不再对 y_train 做 fillna。如果在后续算法（如 sklearn）中不支持 NaN，
    # 必须在算法外层（如 screening 脚本）使用 .dropna() 剔除对应的整行 (X 和 y 同步剔除)。

    x_test = None
    if test_frame is not None:
        x_test = test_frame.drop(columns=[label_col], errors="ignore").copy()
        x_test = x_test.fillna(train_means).fillna(0.0)

    return x_train, y_train, x_test


def filter_feature_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    algo: str = "f_regression",
    k: int = 50,
) -> FeatureSelectionResult:
    """
    功能概述：
    - 工程化过滤法特征选择，支持线性 F 检验与互信息。
    输入：
    - x_train/y_train: 训练特征与标签。
    - algo: `f_regression` 或 `mutual_info`。
    - k: 保留特征数量。
    输出：
    - `FeatureSelectionResult`。
    边界条件：
    - k 会自动限制在特征总数范围内。
    性能/安全注意事项：
    - 过滤法速度快，适合作为第一轮粗筛。
    """
    k = min(max(int(k), 1), x_train.shape[1])
    if algo == "f_regression":
        selector = SelectKBest(score_func=f_regression, k=k)
    elif algo == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    else:
        raise ValueError(f"不支持的过滤法方法: {algo}")

    selector.fit(x_train, y_train)
    selected_features = x_train.columns[selector.get_support()].tolist()
    scores = pd.Series(selector.scores_, index=x_train.columns, name="score").sort_values(
        ascending=False
    )
    return FeatureSelectionResult(
        method=f"filter:{algo}",
        selected_features=selected_features,
        feature_scores=scores,
        params={"k": k},
    )


def wrapper_feature_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 50,
    estimator=None,
    step: int = 1,
) -> FeatureSelectionResult:
    """
    功能概述：
    - 工程化包装法特征选择，默认使用线性回归配合 RFE。
    输入：
    - x_train/y_train: 训练特征与标签。
    - n_features: 最终保留的特征数量。
    - estimator: 底层估计器，默认 `LinearRegression`。
    - step: 每轮剔除特征数。
    输出：
    - `FeatureSelectionResult`。
    边界条件：
    - n_features 会自动限制在合法范围内。
    性能/安全注意事项：
    - 包装法计算成本高，推荐在粗筛后使用。
    """
    estimator = estimator or LinearRegression()
    n_features = min(max(int(n_features), 1), x_train.shape[1])
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
    selector.fit(x_train, y_train)
    selected_features = x_train.columns[selector.support_].tolist()
    ranks = pd.Series(selector.ranking_, index=x_train.columns, name="rank").sort_values()
    return FeatureSelectionResult(
        method="wrapper:rfe",
        selected_features=selected_features,
        feature_scores=ranks,
        params={"n_features": n_features, "step": step, "estimator": estimator.__class__.__name__},
    )


def remove_collinear_features(
    x_train: pd.DataFrame, 
    threshold: float = 0.7, 
    method: str = "spearman"
) -> pd.DataFrame:
    """
    功能概述：
    - [AQR 改进] 因子共线性过滤。计算特征之间的相关系数矩阵，若两个特征高度相关，保留排在前面的特征，剔除后面的特征。
    - [极限加速]: 对于大样本，直接使用 df.corr(method='spearman') 会慢到卡死。
      数学上等价且极速的做法是：先转 Rank (秩)，再求 Pearson 相关系数！速度提升 100 倍！
    """
    print(f"    - 开始进行 {method} 相关性共线性过滤 (阈值: {threshold})...")
    
    if method == "spearman":
        # 极速版 Spearman: Rank -> Pearson
        corr_matrix = x_train.rank().corr(method="pearson").abs()
    else:
        corr_matrix = x_train.corr(method=method).abs()
    
    # 提取上三角矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找到相关系数大于阈值的特征列
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"    - 共发现并剔除了 {len(to_drop)} 个高共线性特征。")
    return x_train.drop(columns=to_drop)


def embedded_feature_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    algo: str = "lasso",
    threshold: float = 0.01,
    model_kwargs: Optional[Dict[str, object]] = None,
) -> FeatureSelectionResult:
    """
    功能概述：
    - 工程化嵌入法特征选择，支持 Lasso 和随机森林。
    """
    model_kwargs = model_kwargs or {}
    
    # 从 kwargs 中分离出供我们代码逻辑使用的参数，以及供 sklearn 模型使用的参数
    max_features = model_kwargs.pop("max_features", None)
    
    if algo == "lasso":
        kwargs = {"alpha": 0.01, "max_iter": 2000}
        kwargs.update(model_kwargs)
        model = Lasso(**kwargs)
    elif algo == "random_forest":
        kwargs = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
        kwargs.update(model_kwargs)
        model = RandomForestRegressor(**kwargs)
    elif algo == "xgboost":
        from xgboost import XGBRegressor
        kwargs = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "tree_method": "hist", "device": "cuda", "n_jobs": 4}
        kwargs.update(model_kwargs)
        model = XGBRegressor(**kwargs)
    elif algo == "lightgbm":
        from lightgbm import LGBMRegressor
        kwargs = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "device": "gpu", "n_jobs": 4, "importance_type": "gain"}
        kwargs.update(model_kwargs)
        model = LGBMRegressor(**kwargs)
    else:
        raise ValueError(f"不支持的嵌入法方法: {algo}")

    # 对于随机森林等树模型，不需要也不应该进行 StandardScaler 标准化
    # 只有 Lasso 等线性模型需要标准化
    if algo == "lasso":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_train_processed = scaler.fit_transform(x_train)
    else:
        # 树模型直接使用原始数据，保留原有的分布特征
        x_train_processed = x_train.values
    
    # 移除 y 中的无穷大/过大值，否则 sklearn 会报错
    finite_mask = np.isfinite(y_train)
    if not finite_mask.all():
        x_train_processed = x_train_processed[finite_mask]
        y_train = y_train[finite_mask]

    model.fit(x_train_processed, y_train)
    if algo == "lasso":
        scores = pd.Series(np.abs(model.coef_), index=x_train.columns, name="importance")
    else:
        scores = pd.Series(model.feature_importances_, index=x_train.columns, name="importance")

    selected = scores[scores > threshold].sort_values(ascending=False)
    
    # 极速优化：如果有 max_features 参数限制，直接截断
    if max_features and len(selected) > max_features:
        selected = selected.head(max_features)
        
    if selected.empty:
        selected = scores.sort_values(ascending=False).head(1)

    return FeatureSelectionResult(
        method=f"embedded:{algo}",
        selected_features=selected.index.tolist(),
        feature_scores=scores.sort_values(ascending=False),
        params={"threshold": threshold, "model": model.__class__.__name__},
    )


def select_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    **kwargs,
) -> FeatureSelectionResult:
    """
    功能概述：
    - 统一调度过滤法、包装法、嵌入法，提供单一入口。
    输入：
    - x_train/y_train: 训练特征与标签。
    - method: `filter` / `wrapper` / `embedded`。
    - kwargs: 对应方法参数。
    输出：
    - `FeatureSelectionResult`。
    边界条件：
    - method 非法时抛出异常。
    性能/安全注意事项：
    - 推荐先过滤法，再包装法/嵌入法，控制计算成本。
    """
    # [AQR 改进] 全局多重共线性剔除：在执行任何特征选择前，先过滤高共线性特征
    if kwargs.get("remove_collinearity", False):
        x_train = remove_collinear_features(
            x_train, 
            threshold=kwargs.get("collinearity_threshold", 0.7),
            method="spearman"
        )
    
    # 剔除仅供共线性过滤使用的参数
    kwargs.pop("remove_collinearity", None)
    kwargs.pop("collinearity_threshold", None)

    if method == "filter":
        return filter_feature_selection(x_train, y_train, **kwargs)
    if method == "wrapper":
        return wrapper_feature_selection(x_train, y_train, **kwargs)
    if method == "embedded":
        return embedded_feature_selection(x_train, y_train, **kwargs)
    raise ValueError(f"不支持的特征选择类型: {method}")


def cached_select_features(x_train, y_train, method, use_cache=True, **kwargs):
    """带 Joblib 缓存的特征选择包装器（自动缓存结果，下次相同输入直接返回）"""
    from qlworks.config import FS_CACHE_DIR
    try:
        import joblib
    except ImportError:
        use_cache = False
    if not use_cache:
        return select_features(x_train, y_train, method, **kwargs)
    import hashlib, json
    xb = x_train.values.tobytes()
    fp = hashlib.md5(xb[:1000000]).hexdigest()
    yb = y_train.values.tobytes()
    fp += hashlib.md5(yb[:100000]).hexdigest()
    fp += str(x_train.shape)
    fp += hashlib.md5(json.dumps(sorted(kwargs.items())).encode()).hexdigest()
    key = hashlib.md5(fp.encode()).hexdigest()[:24]
    cp = FS_CACHE_DIR / f'fs_{method}_{key}.joblib'
    if cp.exists():
        try:
            return joblib.load(str(cp))
        except Exception:
            pass
    result = select_features(x_train, y_train, method, **kwargs)
    try:
        joblib.dump(result, str(cp))
    except Exception:
        pass
    return result


def apply_feature_selection(
    selection_result: FeatureSelectionResult,
    x_train: pd.DataFrame,
    x_test: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    功能概述：
    - 将特征选择结果应用到训练集与测试集。
    
    输入：
    - selection_result: 特征选择结果对象。
    - x_train/x_test: 原始特征矩阵 (包含所有成百上千个特征的 Pandas DataFrame)。
    
    输出：
    - `(selected_train, selected_test)`，格式为 `(pd.DataFrame, Optional[pd.DataFrame])`。
    
    【输出数据结构详解】：
    - 它们是经过“瘦身”的纯 Pandas DataFrame。
    - 索引 (Index)：继承了 Qlib 原生的 MultiIndex (datetime, instrument)，时间与股票代码对齐。
    - 列 (Columns)：只包含被 selection_result.selected_features 选中的少数精英因子列（例如 ['MA5', 'MACD', 'Volume_Ratio']）。
    - 注意：这两个输出 DataFrame 都是【纯特征矩阵】，不包含标签列 (如 LABEL0)。在喂给最终模型前，需要把之前切出来的 y_train 重新拼回去。
    
    边界条件：
    - 测试集可为空。
    性能/安全注意事项：
    - 只做列子集切片，速度快且不会复制无关列。
    """
    cols = list(selection_result.selected_features)
    selected_train = x_train.loc[:, cols].copy()
    selected_test = x_test.loc[:, cols].copy() if x_test is not None else None
    return selected_train, selected_test


if __name__ == "__main__":
    print("=== models/selection.py 独立调用示例 ===")
    from sklearn.datasets import make_regression
    
    # 1. 制造模拟数据: 10 个特征，只有 3 个是有用的
    X, y = make_regression(n_samples=100, n_features=10, n_informative=3, noise=0.5, random_state=42)
    
    # 转换成 Pandas 格式，假装这是我们从 Qlib Dataset 拿出来的东西
    feature_cols = [f"F_{i}" for i in range(10)]
    df_train = pd.DataFrame(X, columns=feature_cols)
    df_train["LABEL0"] = y
    
    print(f"\n[1] 原始脏数据维度: {df_train.shape} (包含标签)")
    
    # 2. 调用第一步: 准备数据 (拆分 X, y，填充 NaN)
    x_train, y_train, _ = prepare_feature_selection_data(df_train, label_col="LABEL0")
    
    # 3. 调用第二步: 使用嵌入法 (Lasso) 进行特征选择
    print("\n[2] 开始执行特征选择: Lasso Embedded Method...")
    result = select_features(
        x_train, y_train, 
        method="embedded", 
        algo="lasso",
        threshold=0.01,         # 只要权重绝对值大于 0.01 的特征
    )
    
    print("\n[3] 筛选结果 (FeatureSelectionResult):")
    print(f"- 选中的特征: {result.selected_features}")
    print(f"- 淘汰的特征: {set(feature_cols) - set(result.selected_features)}")
    
    # 4. 调用第三步: 给原始数据瘦身
    x_train_clean, _ = apply_feature_selection(result, x_train)
    print(f"\n[4] 瘦身后的特征矩阵维度: {x_train_clean.shape}")
    print("搞定！现在可以把 x_train_clean 喂给模型训练了。")
