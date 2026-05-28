import os, json
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def dump_industry_features():
    qlib_dir = Path('e:/Quant/Qlibworks/qlib_data')
    features_dir = qlib_dir / 'features'
    db_path = r'e:\Quant\Quant_Tushare\data\quant_data.duckdb'

    print("1. Loading calendar...")
    cal_file = qlib_dir / 'calendars' / 'day.txt'
    calendars = pd.read_csv(cal_file, header=None)[0].tolist()
    calendars = pd.to_datetime(calendars)
    cal_series = pd.Series(np.arange(len(calendars)), index=calendars)

    print("2. Fetching sw_industry_members from DuckDB...")
    con = duckdb.connect(db_path)
    df = con.execute("""
        SELECT ts_code, l1_name, l2_name, l3_name, in_date, out_date
        FROM sw_industry_members
    """).df()
    con.close()

    df = df.dropna(subset=['ts_code'])
    df['in_date'] = pd.to_datetime(df['in_date'], errors='coerce')
    df['out_date'] = pd.to_datetime(df['out_date'], errors='coerce')
    print(f"    {len(df)} rows loaded from DuckDB.")

    # Build integer mapping
    print("3. Building industry name -> ID mapping...")
    mapping = {}
    for level in ['l1', 'l2', 'l3']:
        names = sorted(df[f'{level}_name'].dropna().unique())
        mapping[level] = {name: i + 1 for i, name in enumerate(names)}

    with open(qlib_dir / 'sw_industry_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    def to_feature_dir(ts_code):
        if not isinstance(ts_code, str):
            return ''
        return ts_code.lower().strip()

    df['symbol'] = df['ts_code'].apply(to_feature_dir)

    existing_dirs = set(os.listdir(str(features_dir)))
    df = df[df['symbol'].isin(existing_dirs)]
    print(f"    Matching symbols: {len(df)} rows")

    print("4. Dumping industry features to Qlib .bin format...")
    fields_config = {'sw_l1': 'l1', 'sw_l2': 'l2', 'sw_l3': 'l3'}

    grouped = df.groupby('symbol')
    for symbol, group in tqdm(grouped, total=len(grouped)):
        group = group.sort_values('in_date')

        for fname, level in fields_config.items():
            arr = np.full(len(calendars), np.nan, dtype=np.float32)
            for _, row in group.reset_index(drop=True).iterrows():
                in_date = row['in_date']
                out_date = row['out_date']
                if pd.isna(in_date):
                    continue

                if pd.isna(out_date) or out_date.year < 2000:
                    next_rows = group[group['in_date'] > in_date]
                    end_date = next_rows['in_date'].iloc[0] - pd.Timedelta(days=1) if len(next_rows) > 0 else calendars[-1]
                else:
                    end_date = out_date

                try:
                    idx_start = cal_series[cal_series.index >= in_date].iloc[0]
                    idx_end = cal_series[cal_series.index <= end_date].iloc[-1]
                    if idx_start <= idx_end:
                        val = mapping[level].get(row[f'{level}_name'], np.nan)
                        arr[idx_start:idx_end + 1] = val
                except (IndexError, KeyError):
                    continue

            valid = np.where(~np.isnan(arr))[0]
            if len(valid) == 0:
                continue
            start_idx = valid[0]
            sliced = arr[start_idx:]
            out_dir = features_dir / symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            np.hstack([[start_idx], sliced]).astype('<f').tofile(str(out_dir / f'{fname}.day.bin'))

    total = 0
    for sym in existing_dirs:
        for fname in fields_config:
            if (features_dir / sym / f'{fname}.day.bin').exists():
                total += 1
                break
    print(f"    Done! {total}/{len(existing_dirs)} instruments have industry features.")


if __name__ == '__main__':
    dump_industry_features()
