import pandas as pd
import numpy as np
from typing import Dict, Literal, Optional
from sklearn.decomposition import PCA

def calc_factor_correlation(factors_df: pd.DataFrame, method: str = 'spearman') -> pd.DataFrame:
    """
    功能概述：
    - 计算截面因子相关系数矩阵。用于在合成前检查因子间的共线性（通常绝对值 > 0.3 即认为存在较高相关性）。
    
    输入：
    - factors_df: Pandas DataFrame，索引为 ['datetime', 'instrument']。
    - method: 'spearman' (默认，抗极值能力强) 或 'pearson'。
    
    输出：
    - Pandas DataFrame，返回所有因子的相关系数矩阵（均值）。
    """
    if factors_df.empty:
        return pd.DataFrame()
        
    # 在每个截面上计算相关系数，然后对时间轴取平均
    corr_matrices = factors_df.groupby(level='datetime').corr(method=method)
    mean_corr = corr_matrices.groupby(level=1).mean()
    return mean_corr


def orthogonalize_factors(
    factors_df: pd.DataFrame, 
    base_factors: list[str], 
    target_factor: str
) -> pd.Series:
    """
    功能概述：
    - 基于截面回归（OLS）的因子正交化（施密特正交的一步）。
    - 剥离 target_factor 中与 base_factors 共线的部分，返回纯净的正交残差。
    
    输入：
    - factors_df: 包含所有因子的 DataFrame (索引为 datetime, instrument)。
    - base_factors: 基底因子列名列表（需要保留的因子）。
    - target_factor: 目标因子列名（需要被正交/被剥离的因子）。
    
    输出：
    - pd.Series: 正交化后的目标因子（残差）。
    
    性能/安全注意事项：
    - 自动跳过含有 NaN 的行，以防线性代数计算报错。
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("正交化功能需要 statsmodels 库，请执行: pip install statsmodels")
    
    def _ortho_slice(sub_df):
        # 提取有效数据（无 NaN）
        valid_df = sub_df[base_factors + [target_factor]].dropna()
        if valid_df.empty or len(valid_df) < len(base_factors) + 2:
            return pd.Series(np.nan, index=sub_df.index)
            
        Y = valid_df[target_factor]
        X = valid_df[base_factors]
        X = sm.add_constant(X) # 截面回归一定要加截距项
        
        try:
            # 使用 OLS 回归
            model = sm.OLS(Y, X).fit()
            # 提取残差 (这就是正交化后的纯净信号)
            resid = model.resid
            
            # 将残差对齐回原始索引
            res_series = pd.Series(np.nan, index=sub_df.index)
            res_series.loc[valid_df.index] = resid
            return res_series
        except Exception:
            return pd.Series(np.nan, index=sub_df.index)

    print(f"[*] 正在将因子 '{target_factor}' 针对 {base_factors} 进行截面正交化...")
    result = factors_df.groupby(level='datetime', group_keys=False).apply(_ortho_slice)
    result.name = f"{target_factor}_ortho"
    return result


def synthesize_factors(
    factors_df: pd.DataFrame,
    method: Literal["equal", "ic_weight", "pca"] = "equal",
    ic_dict: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    功能概述：
    - 因子合成引擎，将多个弱有效因子融合成一个强综合因子（Composite Alpha）。
    - 机器学习模型本质上是高维非线性合成器，但传统的截面合成在构建大类因子或单一 Alpha 策略时依然高效。

    输入：
    - factors_df: Pandas DataFrame，索引应为 ['datetime', 'instrument'] 的多层索引，列为各个因子的数值。
    - method: 合成方法。支持 'equal' (等权合成), 'ic_weight' (IC加权), 'pca' (主成分分析第一主成分)。
    - ic_dict: 当 method='ic_weight' 时必填，格式为 {因子名: IC值}。

    输出：
    - pd.Series，索引同 factors_df，值为合成后的复合因子得分。

    边界条件：
    - 输入数据若包含过多 NaN，可能导致 PCA 失败；系统会自动进行 fillna(0) 处理。
    - 如果 IC_dict 中的键与 DataFrame 的列名不匹配，将仅保留交集列进行计算。

    性能/安全注意事项：
    - 截面标准化（groupby apply）在大数据量下可能较慢，建议在 Qlib Dataset 层已完成标准化后再传入。
    - 内存中计算避免对原始数据发生就地修改（inplace）。
    """
    if factors_df.empty:
        raise ValueError("输入的因子数据为空 DataFrame")

    # 1. 确保在横截面上做 Z-Score 标准化，以防量纲不一致（如果上游已做，这里算双保险）
    # 使用快速向量化运算代替慢速 apply
    mean_series = factors_df.groupby(level='datetime').transform('mean')
    std_series = factors_df.groupby(level='datetime').transform('std')
    norm_df = (factors_df - mean_series) / std_series
    
    # 将极值导致的 NaN 填为 0（均值）
    norm_df = norm_df.fillna(0)

    # 2. 等权合成
    if method == "equal":
        return norm_df.mean(axis=1)

    # 3. IC 加权合成 (历史IC越高，权重越大)
    elif method == "ic_weight":
        if not ic_dict:
            raise ValueError("使用 ic_weight 方法必须提供 ic_dict 参数")
        
        # 归一化 IC 权重 (绝对值求和为1)
        weights = pd.Series(ic_dict)
        weights = weights / weights.abs().sum()
        
        # 对齐列名
        common_cols = [c for c in norm_df.columns if c in weights.index]
        if not common_cols:
            raise ValueError("ic_dict 的键与因子列名没有任何交集")
        
        weighted_df = norm_df[common_cols] * weights[common_cols]
        return weighted_df.sum(axis=1)

    # 4. PCA 主成分合成 (提取最大方差的第一主成分作为综合因子)
    elif method == "pca":
        pca = PCA(n_components=1)
        # PCA 只能处理 2D 矩阵，不能有 NaN
        res = pca.fit_transform(norm_df.values)
        return pd.Series(res.flatten(), index=norm_df.index, name="pca_factor")

    else:
        raise ValueError(f"不支持的合成方法: {method}")


if __name__ == "__main__":
    print("=== factors/synthesis.py 独立测试 ===")
    # 构造模拟的多重索引数据 (datetime, instrument)
    dates = pd.date_range("2020-01-01", periods=2)
    insts = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    
    # 模拟三个因子 (A 和 B 高度共线，C 是独立的)
    np.random.seed(42)
    factor_a = np.random.randn(10)
    factor_b = factor_a * 2 + np.random.randn(10) * 0.1 # 高度共线
    factor_c = np.random.randn(10) * 0.5
    
    df = pd.DataFrame({
        "factor_A": factor_a,
        "factor_B": factor_b,
        "factor_C": factor_c,
    }, index=idx)
    
    print("原始数据:\n", df.head())
    
    # 1. 测试因子相关性计算
    print("\n1. 因子截面相关系数矩阵 (Spearman):")
    corr_mat = calc_factor_correlation(df, method='spearman')
    print(corr_mat)
    
    # 2. 测试施密特正交化
    print("\n2. 正交化测试: 将 factor_B 对 factor_A 正交化")
    try:
        ortho_b = orthogonalize_factors(df, base_factors=["factor_A"], target_factor="factor_B")
        df["factor_B_ortho"] = ortho_b
        print("\n正交化后的相关性矩阵 (B_ortho 和 A 的相关性应趋近于 0):")
        print(calc_factor_correlation(df[["factor_A", "factor_B_ortho"]]))
        factors_to_synthesize = ["factor_A", "factor_B_ortho", "factor_C"]
    except ImportError as e:
        print(f"\n[!] {e}")
        print("[!] 降级: 未执行正交化，直接使用原始 factor_B 演示合成。")
        factors_to_synthesize = ["factor_A", "factor_B", "factor_C"]
    
    # 3. 测试因子合成
    print("\n3. 因子合成测试:")
    print(f"参与合成的因子: {factors_to_synthesize}")
    
    # 等权合成
    equal_composite = synthesize_factors(df[factors_to_synthesize], method="equal")
    print("\n[等权合成 (Equal Weight)]:\n", equal_composite.head())
    
    # IC 加权合成
    ic_composite = synthesize_factors(
        df[factors_to_synthesize], 
        method="ic_weight", 
        ic_dict={factors_to_synthesize[0]: 0.05, factors_to_synthesize[1]: -0.02, factors_to_synthesize[2]: 0.08}
    )
    print("\n[IC加权合成 (IC Weight)]:\n", ic_composite.head())
    
    # PCA 主成分合成
    pca_composite = synthesize_factors(df[factors_to_synthesize], method="pca")
    print("\n[PCA主成分合成 (PCA 第一主成分)]:\n", pca_composite.head())
