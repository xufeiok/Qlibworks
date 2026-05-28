import pandas as pd
import numpy as np
from scipy.linalg import svd
from qlib.data.dataset.processor import Processor

class CSSymmetricOrthogonalize(Processor):
    """
    机构级对称正交化 (Symmetric Orthogonalization)
    用于线性模型前处理。通过 M^{-0.5} 将特征矩阵正交化，使得特征之间相关性为0，
    同时保持正交化后的特征与原始特征的距离最近，保留原始经济学含义。
    """

    def __init__(self, fields_group="feature", **kwargs):
        self.fields_group = fields_group

    def __call__(self, df):
        if self.fields_group not in df.columns.levels[0]:
            return df
        
        print(f"[CSSymmetricOrthogonalize] Running Symmetric Orthogonalization for group: {self.fields_group}...")
        data = df[self.fields_group].copy()

        def _sym_ortho_slice(sub_df):
            # 填补 NaN 以防 SVD 失败
            F = sub_df.fillna(0).values
            if F.shape[0] <= F.shape[1]:
                # 如果截面股票数小于特征数，正交化不稳定，直接返回
                return sub_df
                
            # 计算协方差矩阵 (F 已经标准化，所以 F^T F 就是相关矩阵的无偏估计)
            M = np.dot(F.T, F) / F.shape[0]
            
            try:
                # 使用 SVD 计算 M^{-0.5}
                U, S, Vh = svd(M)
                # 过滤掉极小的奇异值防除零
                S_inv_sqrt = np.diag(np.where(S > 1e-8, 1.0 / np.sqrt(S), 0.0))
                M_inv_sqrt = np.dot(U, np.dot(S_inv_sqrt, Vh))
                
                # 正交化后的特征
                F_orth = np.dot(F, M_inv_sqrt)
                
                res_df = sub_df.copy()
                res_df.loc[:, :] = F_orth
                return res_df
            except Exception as e:
                # 如果 SVD 不收敛，返回原矩阵
                return sub_df

        ortho_data = data.groupby(level='datetime', group_keys=False).apply(_sym_ortho_slice)
        df.loc[:, (self.fields_group, data.columns)] = ortho_data.values
        
        print(f"[CSSymmetricOrthogonalize] Orthogonalization completed.")
        return df
