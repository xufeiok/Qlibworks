import tushare as ts
import pandas as pd
import datetime
import os
import time

pro = ts.pro_api()

def get_index_history(index_code, start_year, end_year):
    print(f"Downloading {index_code} history...")
    all_data = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            sy = f"{year}{month:02d}01"
            ey = f"{year}{month:02d}31"
            try:
                df = pro.index_weight(index_code=index_code, start_date=sy, end_date=ey)
                if not df.empty:
                    all_data.append(df)
                time.sleep(0.4)  # Avoid rate limit
            except Exception as e:
                print(f"Error for {sy}-{ey}: {e}")
                
    if not all_data:
        return pd.DataFrame()
        
    res = pd.concat(all_data, ignore_index=True)
    return res

def process_and_save(df, filename):
    if df.empty:
        print(f"No data for {filename}")
        return
        
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values(['con_code', 'trade_date'])
    
    records = []
    for code, group in df.groupby('con_code'):
        group = group.sort_values('trade_date')
        dates = group['trade_date'].tolist()
        
        start = dates[0]
        prev = dates[0]
        
        for d in dates[1:]:
            if (d - prev).days > 45:
                records.append((code, start, prev))
                start = d
            prev = d
        records.append((code, start, prev))
        
    out_lines = []
    latest_date = df['trade_date'].max()
    for code, start, end in records:
        if end == latest_date:
            end_str = '9999-12-31'
        else:
            end_str = end.strftime('%Y-%m-%d')
            
        start_str = start.strftime('%Y-%m-%d')
        out_lines.append(f"{code}\t{start_str}\t{end_str}\n")
        
    filepath = os.path.join(r"e:\Quant\Qlibworks\qlib_data\instruments", filename)
    with open(filepath, 'w') as f:
        f.writelines(out_lines)
    print(f"Saved {filename} with {len(out_lines)} intervals.")

if __name__ == '__main__':
    df_300 = get_index_history('000300.SH', 2016, 2026) # Tushare 000300.SH index_weight typically starts around 2016
    process_and_save(df_300, 'csi300.txt')

    df_2000 = get_index_history('932000.CSI', 2023, 2026) # CSI2000 launched in 2023
    process_and_save(df_2000, 'csi2000.txt')
