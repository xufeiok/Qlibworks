import pandas as pd
import numpy as np


def load_industry_map(instruments, ref_date):
    try:
        from qlib.data import D
        ind_data = D.features(instruments, ['$sw_l1'],
                              start_time=ref_date, end_time=ref_date)
        if ind_data.empty:
            return {}
        industry_map = {}
        for inst in instruments:
            if inst in ind_data.index.get_level_values('instrument'):
                val = ind_data.xs(inst, level='instrument')['$sw_l1'].iloc[0]
                if not (isinstance(val, float) and np.isnan(val)):
                    industry_map[inst] = val
        return industry_map
    except Exception:
        return {}


def load_industry_maps_pit(instruments, start_date, end_date):
    """
    [Renaissance 改进] Point-in-Time 行业映射加载器。

    每年加载一次行业快照，确保行业分类在回测期内与当时的实际分类一致。
    如果某只股票在某年退市或变更行业，后续年份的快照会正确反映这一变化。

    Args:
        instruments: 股票代码列表
        start_date: 回测起始日期 (str 或 Timestamp)
        end_date: 回测结束日期 (str 或 Timestamp)

    Returns:
        dict: {pd.Timestamp(年份起始): {instrument: industry}}
    """
    yearly_snapshots = pd.date_range(start=start_date, end=end_date, freq='YS')
    if len(yearly_snapshots) == 0:
        return {pd.Timestamp(start_date): load_industry_map(instruments, str(start_date)[:10])}

    maps = {}
    for snap_dt in yearly_snapshots:
        ref = snap_dt.strftime('%Y-%m-%d')
        imap = load_industry_map(instruments, ref)
        if imap:
            maps[snap_dt] = imap
            print(f"    - [PIT行业快照] {ref}: {len(imap)} 只股票行业已加载")

    # 确保至少有一个快照
    if not maps:
        fallback_date = str(start_date)[:10]
        maps[pd.Timestamp(start_date)] = load_industry_map(instruments, fallback_date)

    return maps


def _get_nearest_map(industry_maps, target_date):
    """
    找到 <= target_date 的最新行业快照。
    industry_maps 的 key 是 pd.Timestamp 格式的年份起始日。
    """
    map_dates = sorted(industry_maps.keys())
    for i in range(len(map_dates) - 1, -1, -1):
        if map_dates[i] <= target_date:
            return industry_maps[map_dates[i]]
    return industry_maps[map_dates[0]]


def apply_industry_constraint(pred_df, industry_map, top_k=20, max_per_industry=4):
    """
    行业约束：使用单一时点快照（向后兼容）。

    对于长周期回测，推荐使用 apply_industry_constraint_pit()，
    它会按年份切换行业快照。
    """
    desc = f"总持仓不超过 {top_k} 只, 单一行业最多 {max_per_industry} 只"
    print(f"    - [行业约束] {desc}")

    df = pred_df.reset_index()
    df['industry'] = df['instrument'].map(industry_map).fillna('Unknown')

    records = []
    for dt, group in df.groupby('datetime'):
        group = group.sort_values('score', ascending=False)
        selected = []
        ind_counts = {}
        for _, row in group.iterrows():
            if len(selected) >= top_k:
                break
            ind = row['industry']
            if ind_counts.get(ind, 0) < max_per_industry:
                selected.append(row)
                ind_counts[ind] = ind_counts.get(ind, 0) + 1
        records.extend(selected)

    constrained = pd.DataFrame(records)
    constrained = pd.DataFrame(records).set_index(['datetime', 'instrument']).drop(columns=['industry'])
    print(f"    - 约束前 {len(df)} 条 → 约束后 {len(constrained)} 条")
    return constrained[['score']]


def apply_industry_constraint_pit(pred_df, industry_maps, top_k=20, max_per_industry=4):
    """
    [Renaissance 改进] Point-in-Time 行业约束。

    回测期内每隔一定时期（每年）重新加载行业分类快照，
    按日期自动选择对应的行业映射，防止行业变更导致的前视偏差。

    Args:
        pred_df: MultiIndex [datetime, instrument] 的预测得分 DataFrame
        industry_maps: load_industry_maps_pit() 返回的 {快照日期: {股票: 行业}} 字典
        top_k: 最大持仓数
        max_per_industry: 单行业最大持仓数

    Returns:
        约束后的 DataFrame，结构与 pred_df 一致
    """
    desc = f"总持仓不超过 {top_k} 只, 单一行业最多 {max_per_industry} 只 (PIT)"
    print(f"    - [行业约束 PIT] {desc}")

    df = pred_df.reset_index()
    records = []
    for dt, group in df.groupby('datetime'):
        industry_map = _get_nearest_map(industry_maps, dt)
        group = group.copy()
        group['industry'] = group['instrument'].map(industry_map).fillna('Unknown')
        group = group.sort_values('score', ascending=False)

        selected = []
        ind_counts = {}
        for _, row in group.iterrows():
            if len(selected) >= top_k:
                break
            ind = row['industry']
            if ind_counts.get(ind, 0) < max_per_industry:
                selected.append(row)
                ind_counts[ind] = ind_counts.get(ind, 0) + 1
        records.extend(selected)

    constrained = pd.DataFrame(records)
    constrained = constrained.set_index(['datetime', 'instrument']).drop(columns=['industry'])
    print(f"    - 约束前 {len(df)} 条 → 约束后 {len(constrained)} 条")
    return constrained[['score']]
