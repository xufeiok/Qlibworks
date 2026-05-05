import numpy as np
import pandas as pd
from qlib.data.dataset.processor import Processor
from qlib.data import D

class CSNeutralize(Processor):
    """
    截面中性化处理器 (Cross-Sectional Neutralization)
    对指定的 fields_group 特征进行截面行业和市值中性化。
    """
    def __init__(self, fields_group="feature", industry_field="industry_code", market_cap_field="circ_mv", log_mc=True):
        self.fields_group = fields_group
        self.industry_field = industry_field
        self.market_cap_field = market_cap_field
        self.log_mc = log_mc

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        print(f"[{self.__class__.__name__}] Fetching exposures (industry, market_cap)...")
        instruments = df.index.get_level_values('instrument').unique().tolist()
        start_time = df.index.get_level_values('datetime').min()
        end_time = df.index.get_level_values('datetime').max()
        
        # Load exposure data from Qlib bin
        fields = [f"${self.industry_field}", f"${self.market_cap_field}"]
        
        exposures = D.features(
            instruments, 
            fields, 
            start_time=start_time, 
            end_time=end_time, 
            freq='day'
        )
        exposures.columns = ['industry', 'market_cap']
        
        # Align with the main df
        # D.features returns MultiIndex ['instrument', 'datetime']
        # But df usually has MultiIndex ['datetime', 'instrument'] in DatasetH
        if df.index.names != exposures.index.names:
            exposures = exposures.swaplevel()
            
        exposures = exposures.reindex(df.index)
        
        # Process market cap (log)
        # [AQR 改进] 保留原始市值用于 WLS (加权最小二乘法) 权重计算
        exposures['raw_market_cap'] = exposures['market_cap'].copy()
        if self.log_mc:
            exposures['market_cap'] = np.where(exposures['market_cap'] <= 0, np.nan, exposures['market_cap'])
            exposures['market_cap'] = np.log(exposures['market_cap'])
            
        print(f"[{self.__class__.__name__}] Running cross-sectional neutralization for group: {self.fields_group}...")
        
        # 核心修复：只提取当前需要处理的 fields_group 列
        # Qlib 的 df.columns 通常是 MultiIndex (如 ('feature', '$close'), ('label', 'LABEL0'))
        if isinstance(df.columns, pd.MultiIndex) and self.fields_group is not None:
            # 找到属于该 group 的列
            target_cols = df.columns[df.columns.get_level_values(0) == self.fields_group]
        else:
            target_cols = df.columns
            
        if len(target_cols) == 0:
            return df
            
        # Group by datetime to neutralize cross-sectionally
        def _neutralize_slice(sub_df):
            date = sub_df.index.get_level_values('datetime')[0]
            sub_exp = exposures.xs(date, level='datetime')
            
            # Find valid rows where exposures are not NaN
            valid_exp_mask = ~(sub_exp['industry'].isna() | sub_exp['market_cap'].isna())
            
            if not valid_exp_mask.any():
                return pd.DataFrame(np.nan, index=sub_df.index, columns=sub_df.columns)
                
            valid_exp = sub_exp[valid_exp_mask]
            # sub_df still has MultiIndex (instrument, datetime)
            # valid_exp_mask has Index (instrument)
            valid_instruments = valid_exp_mask[valid_exp_mask].index
            
            # Use xs to get instrument level data for valid instruments
            # Then we can just use instrument index
            if isinstance(sub_df.index, pd.MultiIndex):
                if sub_df.index.names[0] == 'datetime':
                    valid_sub_df = sub_df.loc[(date, valid_instruments), :]
                else:
                    valid_sub_df = sub_df.loc[(valid_instruments, date), :]
            else:
                valid_sub_df = sub_df.loc[valid_instruments]

            
            # Prepare X matrix
            ind_dummies = pd.get_dummies(valid_exp['industry'].astype(int).astype(str), prefix='ind', drop_first=True)
            X = pd.concat([valid_exp['market_cap'], ind_dummies], axis=1)
            X.insert(0, 'const', 1.0)
            
            X_mat = X.values.astype(float)
            
            # [AQR 改进] WLS weights (市值平方根)
            weights = np.sqrt(valid_exp['raw_market_cap'].values.astype(float))
            # 避免极端情况下的 NaN 或者 <=0 的权重
            weights = np.nan_to_num(weights, nan=1.0)
            weights = np.clip(weights, a_min=1e-8, a_max=None)
            
            # 仅提取目标列的数据
            target_sub_df = valid_sub_df[target_cols]
            
            # Result dataframe (只填充我们要修改的列，其余列原样返回)
            res_df = sub_df.copy()
            
            # [AQR 改进] 矩阵化极速求解 OLS
            # 提取 Y 矩阵 (目标因子)
            Y_mat = target_sub_df.values.astype(float)
            
            # 找出哪些行 (instrument) 在 Y 中是完全有效的（即所有因子都没有 NaN）
            # 检查是否有 NaN
            has_nan = np.isnan(Y_mat).any()
            
            if not has_nan and Y_mat.shape[0] > X_mat.shape[1]:
                # 理想情况：Y 矩阵完全无缺失值，直接进行全矩阵 WLS 计算
                # 权重广播
                X_mat_w = X_mat * weights[:, np.newaxis]
                Y_mat_w = Y_mat * weights[:, np.newaxis]
                
                # 求解所有特征的 beta
                # beta shape: (X_features, Y_features)
                beta, _, _, _ = np.linalg.lstsq(X_mat_w, Y_mat_w, rcond=None)
                
                # 计算残差
                resid_mat = Y_mat - X_mat @ beta
                
                # 映射回 DataFrame
                res_df.loc[valid_sub_df.index, target_cols] = resid_mat
            else:
                # 降级情况：存在 NaN，退回到逐列计算以保证严谨性
                for col in target_cols:
                    y = valid_sub_df[col].values.astype(float)
                    valid_y_mask = ~np.isnan(y)
                    
                    if valid_y_mask.sum() > X_mat.shape[1]:  # Need enough degrees of freedom
                        X_valid = X_mat[valid_y_mask]
                        y_valid = y[valid_y_mask]
                        
                        # WLS weights matching valid_y_mask
                        w_valid = weights[valid_y_mask]
                        X_valid_w = X_valid * w_valid[:, np.newaxis]
                        y_valid_w = y_valid * w_valid
                        
                        # np.linalg.lstsq solves: X_w * beta = y_w
                        beta, _, _, _ = np.linalg.lstsq(X_valid_w, y_valid_w, rcond=None)
                        
                        # Calculate residuals
                        resid = y_valid - X_valid @ beta
                        
                        # Map back to res_df
                        res_df.loc[valid_sub_df.index[valid_y_mask], col] = resid
                    
            return res_df
            
        result = df.groupby(level='datetime', group_keys=False).apply(_neutralize_slice)
        print(f"[{self.__class__.__name__}] Neutralization completed.")
        return result
