import os
import sys
import duckdb
import pandas as pd
from pathlib import Path
import json

def export_features():
    db_path = 'e:/Quant/Quant_Tushare/data/quant_data.duckdb'
    out_dir = Path('e:/Quant/Qlibworks/qlib_data_tmp/features_csv')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to DuckDB: {db_path}")
    conn = duckdb.connect(db_path, read_only=True)
    
    # 1. Build Industry Mapping
    print("Building industry mapping...")
    ind_df = conn.execute("SELECT DISTINCT industry FROM stock_universe WHERE industry IS NOT NULL AND market = '主板'").df()
    industries = sorted(ind_df['industry'].tolist())
    ind_map = {ind: float(i) for i, ind in enumerate(industries)}
    
    with open('e:/Quant/Qlibworks/industry_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(ind_map, f, ensure_ascii=False, indent=4)
        
    print(f"Mapped {len(ind_map)} industries.")
    
    # 2. Get Symbols
    symbols = conn.execute("SELECT DISTINCT ts_code FROM stock_universe WHERE market = '主板'").df()['ts_code'].tolist()
    
    print(f"Exporting features for {len(symbols)} symbols...")
    
    for i, sym in enumerate(symbols):
        if i % 500 == 0:
            print(f"Exporting {i}/{len(symbols)}...")
        
        q = f"""
        SELECT 
            d.trade_date as date,
            d.circ_mv as circ_mv,
            s.industry as industry
        FROM daily_indicators d
        JOIN stock_universe s ON d.ts_code = s.ts_code
        WHERE d.ts_code = '{sym}'
        ORDER BY d.trade_date
        """
        df = conn.execute(q).df()
        if df.empty:
            continue
            
        df['industry_code'] = df['industry'].map(ind_map)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # select only needed columns
        df = df[['date', 'circ_mv', 'industry_code']]
        
        df.to_csv(out_dir / f"{sym}.csv", index=False)
        
    conn.close()
    print("Export complete.")

if __name__ == '__main__':
    export_features()
