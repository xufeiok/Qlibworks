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
        if self.log_mc:
            exposures['market_cap'] = np.where(exposures['market_cap'] <= 0, np.nan, exposures['market_cap'])
            exposures['market_cap'] = np.log(exposures['market_cap'])
            
        print(f"[{self.__class__.__name__}] Running cross-sectional neutralization...")
        
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
            
            # Result dataframe
            res_df = pd.DataFrame(np.nan, index=sub_df.index, columns=sub_df.columns)
            
            # We can do matrix operation for all features at once, but features might have NaNs!
            # If a feature has NaN, we need to handle it separately.
            # To optimize, we can fill NaN with 0 temporarily, or do column by column.
            # Since K (features) is typically ~100-200, column by column is acceptable if we use np.linalg.lstsq
            
            for col in valid_sub_df.columns:
                y = valid_sub_df[col].values.astype(float)
                valid_y_mask = ~np.isnan(y)
                
                if valid_y_mask.sum() > X_mat.shape[1]:  # Need enough degrees of freedom
                    X_valid = X_mat[valid_y_mask]
                    y_valid = y[valid_y_mask]
                    
                    # np.linalg.lstsq solves: X * beta = y
                    beta, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
                    
                    # Calculate residuals
                    resid = y_valid - X_valid @ beta
                    
                    # Map back to res_df
                    # valid_sub_df.index[valid_y_mask] gives the original instrument indices
                    res_df.loc[valid_sub_df.index[valid_y_mask], col] = resid
                    
            return res_df
            
        result = df.groupby(level='datetime', group_keys=False).apply(_neutralize_slice)
        print(f"[{self.__class__.__name__}] Neutralization completed.")
        return result
