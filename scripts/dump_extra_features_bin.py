import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def dump_extra_features():
    csv_dir = Path('e:/Quant/Qlibworks/qlib_data_tmp/features_csv')
    qlib_dir = Path('e:/Quant/Qlibworks/qlib_data')
    features_dir = qlib_dir / 'features'
    
    # 1. Read calendar
    cal_file = qlib_dir / 'calendars' / 'day.txt'
    calendars = pd.read_csv(cal_file, header=None)[0].tolist()
    calendars = [pd.Timestamp(c) for c in calendars]
    cal_df = pd.DataFrame({'date': calendars})
    cal_df['date'] = cal_df['date'].astype('datetime64[ns]')
    cal_df.set_index('date', inplace=True)
    
    # 2. Process each CSV
    csv_files = list(csv_dir.glob('*.csv'))
    for f in tqdm(csv_files):
        symbol = f.stem.lower() # e.g. 600000.sh
        df = pd.read_csv(f)
        if df.empty:
            continue
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # filter to match calendar min/max
        df = df[df.index.isin(cal_df.index)]
        if df.empty:
            continue
            
        # align with calendar
        r_df = df.reindex(cal_df.index)
        start_date = df.index.min()
        if pd.isna(start_date):
            continue
            
        start_idx = calendars.index(start_date)
        
        # the data array is from start_idx to the end of r_df
        # actually, dump_bin just saves the array from start_idx to the end.
        # Let's slice r_df from start_date to the end.
        sliced_df = r_df.loc[start_date:]
        
        out_dir = features_dir / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for field in ['circ_mv', 'industry_code']:
            if field in sliced_df.columns:
                bin_path = out_dir / f"{field}.day.bin"
                # write start_idx + float32 data
                data = sliced_df[field].values
                # replace nan with nan is fine for float32
                np.hstack([[start_idx], data]).astype('<f').tofile(str(bin_path))

if __name__ == '__main__':
    dump_extra_features()
