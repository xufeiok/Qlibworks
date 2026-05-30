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


def apply_industry_constraint(pred_df, industry_map, top_k=20, max_per_industry=4):
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
