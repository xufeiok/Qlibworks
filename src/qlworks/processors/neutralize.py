import pandas as pd
import numpy as np
from qlib.data.dataset.processor import Processor
from sklearn.linear_model import Ridge

class CSNeutralize(Processor):
    """
    机构级稳健截面中性化 (Robust Cross-Sectional Neutralization)
    使用 Ridge 回归替代传统 OLS，彻底解决由于行业股票稀疏或共线性导致的奇异矩阵 (Singular Matrix) 和 NaN 爆炸问题。
    """

    def __init__(self, fields_group="feature", industry_field="industry_code", market_cap_field="circ_mv", log_mc=True, **kwargs):
        self.fields_group = fields_group
        self.industry_field = industry_field
        self.market_cap_field = market_cap_field
        self.log_mc = log_mc

    def __call__(self, df):
        if self.fields_group not in df.columns.levels[0]:
            return df
        
        # 1. 提取要中性化的目标数据矩阵
        data = df[self.fields_group].copy()
        
        print(f"[CSNeutralize] Fetching exposures (industry, market_cap) for robust Ridge neutralization...")
        
        # 2. 从 Qlib 拉取行业和市值数据
        from qlib.data import D
        instruments = df.index.get_level_values('instrument').unique().tolist()
        start_time = df.index.get_level_values('datetime').min()
        end_time = df.index.get_level_values('datetime').max()
        
        try:
            fields = [f"${self.industry_field}", f"${self.market_cap_field}"]
            exposures = D.features(
                instruments, 
                fields, 
                start_time=start_time, 
                end_time=end_time, 
                freq='day'
            )
            exposures.columns = ['industry', 'market_cap']
            
            if df.index.names != exposures.index.names:
                exposures = exposures.swaplevel()
            exposures = exposures.reindex(df.index)
            
            # 市值对数化处理
            if self.log_mc:
                exposures['market_cap'] = np.where(exposures['market_cap'] <= 0, np.nan, exposures['market_cap'])
                exposures['market_cap'] = np.log(exposures['market_cap'])
                
        except Exception as e:
            print(f"[CSNeutralize] Warning: Failed to fetch exposure data ({e}). Falling back to mean-centering.")
            neutralized_data = data.groupby(level="datetime").apply(lambda x: x - x.mean())
            df.loc[:, (self.fields_group, data.columns)] = neutralized_data.values
            return df

        print(f"[CSNeutralize] Running Ridge cross-sectional neutralization for group: {self.fields_group}...")

        # 3. 按日期进行截面中性化
        def _robust_ridge_neutralize_slice(sub_df):
            # 获取当前切片的日期
            # 兼容不同层级结构的 MultiIndex
            date = sub_df.index.get_level_values('datetime')[0] if 'datetime' in sub_df.index.names else sub_df.name
            
            try:
                sub_exp = exposures.xs(date, level='datetime')
            except KeyError:
                # 如果某天在暴露度数据中完全缺失，退化为中心化
                return sub_df - sub_df.mean()
            
            # 找到市值和行业都不为空的股票
            valid_exp_mask = ~(sub_exp['industry'].isna() | sub_exp['market_cap'].isna())
            if not valid_exp_mask.any():
                # 如果当天完全没有市值/行业数据，直接返回中心化的结果
                return sub_df - sub_df.mean()
                
            valid_exp = sub_exp[valid_exp_mask]
            valid_instruments = valid_exp_mask[valid_exp_mask].index
            
            # 构建解释变量矩阵 X (市值 + 行业虚拟变量)
            # 即使某些行业只有 1 只股票导致完全共线性，Ridge 回归也能完美处理
            ind_dummies = pd.get_dummies(valid_exp['industry'].astype(int).astype(str), prefix='ind', drop_first=False)
            X = pd.concat([valid_exp['market_cap'], ind_dummies], axis=1)
            # 填充 X 中的异常值，确保回归矩阵绝对安全
            X = X.fillna(0).values.astype(float)
            
            # 提取目标变量矩阵 Y (需要中性化的因子矩阵)
            if isinstance(sub_df.index, pd.MultiIndex):
                if sub_df.index.names[0] == 'datetime':
                    valid_sub_df = sub_df.loc[(date, valid_instruments), :]
                else:
                    valid_sub_df = sub_df.loc[(valid_instruments, date), :]
            else:
                valid_sub_df = sub_df.loc[valid_instruments]
                
            Y = valid_sub_df.values.astype(float)
            
            # 使用 Ridge 回归（引入微小的 L2 惩罚项 1e-5）
            # 这是 AQR 处理截面中性化防止矩阵奇异的杀手锏
            model = Ridge(alpha=1e-5, fit_intercept=True, solver='auto')
            
            # 由于部分因子可能在某些股票上是 NaN，我们需要用 0 临时填补 Y 才能送进 sklearn
            # 但我们计算出的残差，原来是 NaN 的地方我们还要保持它是 NaN
            Y_filled = np.nan_to_num(Y, nan=0.0)
            model.fit(X, Y_filled)
            
            # 残差 = 实际值 - 预测值 (剥离了市值和行业 Beta 后的纯净 Alpha)
            residuals = Y_filled - model.predict(X)
            
            # 将原本是 NaN 的位置恢复为 NaN
            residuals[np.isnan(Y)] = np.nan
            
            # 写回结果
            res_df = sub_df.copy()
            res_df.loc[valid_sub_df.index, data.columns] = residuals
            return res_df

        neutralized_data = data.groupby(level='datetime', group_keys=False).apply(_robust_ridge_neutralize_slice)
        df.loc[:, (self.fields_group, data.columns)] = neutralized_data.values
        
        print(f"[CSNeutralize] Ridge Neutralization completed.")
        return df
