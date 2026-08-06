import pandas as pd
import numpy as np
import logging
from qlib.data.dataset.processor import Processor
from qlib.data.data import ExpressionD, Cal
from sklearn.linear_model import Ridge

_logger = logging.getLogger(__name__)


def _fetch_features_direct(instruments, fields, start_time, end_time, freq='day'):
    """绕过 D.features() 的 ParallelExt 缺陷（Windows spawn 多进程下
    Operators 模块不可见导致 SyntaxError），直接用 ExpressionD.expression()
    逐 instrument 逐 field 取值。返回与 D.features() 相同格式的 DataFrame
    （MultiIndex: instrument x datetime, columns=fields 原始字符串）。

    [P0修复] ExpressionD.expression() 默认 time2idx=True，返回的是
    "全局交易日历位置"整数索引（行号）而非真实日期（见 qlib/data/data.py
    LocalExpressionProvider.expression）。若直接使用该整数索引，下游与
    parquet 因子/预测的日期索引完全失配（pd.to_datetime(int) 会得到
    1970 附近的垃圾日期），导致 join 恒为空。此处用 qlib 日历将整数
    位置映射回真实日期。
    """
    import pandas as pd

    result_parts = []
    for inst in instruments:
        inst_data = {}
        for f in fields:
            try:
                series = ExpressionD.expression(inst, f, start_time, end_time, freq)
                if series is not None and len(series) > 0:
                    # 整数行号索引 → 用 qlib 交易日历映射回真实日期
                    if not isinstance(series.index, pd.DatetimeIndex):
                        _cal = Cal.calendar(freq=freq)
                        series.index = pd.DatetimeIndex([_cal[i] for i in series.index])
                inst_data[f] = series
            except Exception:
                inst_data[f] = pd.Series(dtype=float)

        df_inst = pd.DataFrame(inst_data)
        if df_inst.empty or len(df_inst) == 0:
            continue
        df_inst.index.name = 'datetime'
        df_inst['instrument'] = inst
        df_inst = df_inst.reset_index().set_index(['instrument', 'datetime'])
        result_parts.append(df_inst)

    if not result_parts:
        return pd.DataFrame(columns=fields)

    result = pd.concat(result_parts).sort_index()
    # datetime 层统一转 Timestamp（兼容部分调用路径返回 str 索引的情况；
    # 正常路径下该层已是 datetime64，此处为幂等安全转换）
    dt_level = result.index.names.index('datetime') if 'datetime' in result.index.names else -1
    if dt_level >= 0:
        new_levels = list(result.index.levels)
        new_levels[dt_level] = pd.to_datetime(new_levels[dt_level])
        result.index = result.index.set_levels(new_levels)
    return result


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
        
        _logger.debug("Fetching exposures (industry, market_cap) for robust Ridge neutralization...")
        
        # 2. 从 Qlib 拉取行业和市值数据
        instruments = df.index.get_level_values('instrument').unique().tolist()
        start_time = df.index.get_level_values('datetime').min()
        end_time = df.index.get_level_values('datetime').max()
        
        fields = [f"${self.industry_field}", f"${self.market_cap_field}"]
        exposures = _fetch_features_direct(instruments, fields, start_time, end_time, freq='day')
        if exposures.empty:
            _logger.warning("行业/市值暴露度数据为空，回退均值中心化。")
            neutralized_data = data.groupby(level="datetime").apply(lambda x: x - x.mean())
            df.loc[:, (self.fields_group, data.columns)] = neutralized_data.astype(np.float32).values
            return df
        
        exposures.columns = ['industry', 'market_cap']
        
        if df.index.names != exposures.index.names:
            exposures = exposures.swaplevel()
        exposures = exposures.reindex(df.index)
        
        # 市值对数化处理
        if self.log_mc:
            exposures['market_cap'] = np.where(exposures['market_cap'] <= 0, np.nan, exposures['market_cap'])
            exposures['market_cap'] = np.log(exposures['market_cap'])

        _logger.debug("Running Ridge cross-sectional neutralization for group: %s...", self.fields_group)

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
            # [P0修复] 原实现用 sub_df.loc[(valid_instruments, date), :] 部分索引，
            # valid_instruments 为 pd.Index 对象时，pandas 2.x 对 tuple 中带 Index
            # 的部分索引行为不稳定，会导致 valid_sub_df 形状错乱（IndexError 根因）。
            # 改为布尔掩码 + isin，索引天然对齐，行为确定。
            if isinstance(sub_df.index, pd.MultiIndex):
                _inst_level = 0 if sub_df.index.names[0] == "instrument" else 1
                _mask = sub_df.index.get_level_values(_inst_level).isin(list(valid_instruments))
                valid_sub_df = sub_df.loc[_mask]
            else:
                valid_sub_df = sub_df.loc[sub_df.index.isin(list(valid_instruments))]

            Y = valid_sub_df.values.astype(float)
            if len(Y) == 0 or len(Y) != X.shape[0]:
                # 防御：股票子集为空或与解释变量行数不一致时，退化为按日去均值
                return sub_df - sub_df.mean()
            
            # 使用 Ridge 回归（引入微小的 L2 惩罚项 1e-5）
            # 这是 AQR 处理截面中性化防止矩阵奇异的杀手锏
            model = Ridge(alpha=1e-5, fit_intercept=True, solver='auto')
            
            # 由于部分因子可能在某些股票上是 NaN，我们需要用 0 临时填补 Y 才能送进 sklearn
            # 但我们计算出的残差，原来是 NaN 的地方我们还要保持它是 NaN
            Y_filled = np.nan_to_num(Y, nan=0.0)
            model.fit(X, Y_filled)
            
            # 残差 = 实际值 - 预测值 (剥离了市值和行业 Beta 后的纯净 Alpha)
            # [P0修复] sklearn 单目标回归 predict 返回 1D (n,)，
            # 与 2D 的 Y_filled (n, 1) 相减会触发 numpy 广播 → (n, n) 形状错乱
            # （IndexError: boolean index did not match... 根因），需显式 reshape。
            residuals = Y_filled - model.predict(X).reshape(Y_filled.shape)
            
            # 将原本是 NaN 的位置恢复为 NaN
            residuals[np.isnan(Y)] = np.nan
            
            # 写回结果
            # [dtype] QLib 原始数据为 float32，残差需先转 float32 再写回，
            # 避免 pandas 2.2+ 的"incompatible dtype" FutureWarning（未来会报错）。
            res_df = sub_df.copy()
            res_df.loc[valid_sub_df.index, data.columns] = residuals.astype(np.float32)
            return res_df

        neutralized_data = data.groupby(level='datetime', group_keys=False).apply(_robust_ridge_neutralize_slice)
        df.loc[:, (self.fields_group, data.columns)] = neutralized_data.astype(np.float32).values
        
        _logger.debug("Ridge Neutralization completed.")
        return df
