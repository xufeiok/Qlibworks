import os
import pandas as pd
import duckdb
from datetime import datetime

# Paths
base_dir = r"e:\Quant\Qlibworks\qlib_data\instruments"
duckdb_path = r"e:\Quant\Quant_Tushare\data\quant_data.duckdb"

# 1. Fetch real IPO and Delist dates from DuckDB
print("Fetching real IPO and delisting dates from DuckDB...")
con = duckdb.connect(duckdb_path, read_only=True)
stock_df = con.execute("SELECT ts_code, list_date, delist_date FROM stock_universe").df()
con.close()

# Build mapping
date_map = {}
for _, row in stock_df.iterrows():
    ts_code = row['ts_code'] # e.g. 000001.SZ
    list_date = row['list_date'].strftime('%Y-%m-%d') if pd.notna(row['list_date']) else '2000-01-01'
    delist_date = row['delist_date'].strftime('%Y-%m-%d') if pd.notna(row['delist_date']) else '9999-12-31'
    date_map[ts_code] = (list_date, delist_date)

# 2. Fix all.txt, all_sh.txt, all_sz.txt
def fix_all_files(filename):
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        return
    print(f"Fixing {filename}...")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            inst = parts[0]
            if inst in date_map:
                start, end = date_map[inst]
            else:
                start, end = '2000-01-01', '9999-12-31'
            new_lines.append(f"{inst}\t{start}\t{end}\n")
            
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

for f in ['all.txt', 'all_sh.txt', 'all_sz.txt']:
    fix_all_files(f)

# 3. Fix index files by merging fragmented intervals
def merge_intervals(intervals, max_gap_days=730):
    if not intervals:
        return []
    # Sort by start_date
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        
        # Parse dates
        prev_end_dt = datetime.strptime('2199-12-31' if prev[1].startswith('9999') else prev[1], '%Y-%m-%d')
        curr_start_dt = datetime.strptime(current[0], '%Y-%m-%d')
        
        # If overlap or gap is small
        if (curr_start_dt - prev_end_dt).days <= max_gap_days:
            # Merge
            new_end = current[1] if current[1] > prev[1] else prev[1]
            if current[1].startswith('9999') or prev[1].startswith('9999'):
                new_end = '9999-12-31'
            merged[-1] = (prev[0], new_end)
        else:
            merged.append(current)
            
    return merged

def fix_index_files(filename):
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        return
    print(f"Fixing {filename}...")
    
    df = pd.read_csv(filepath, sep='\t', header=None, names=['inst', 'start', 'end'])
    
    new_records = []
    for inst, group in df.groupby('inst'):
        intervals = list(zip(group['start'], group['end']))
        merged = merge_intervals(intervals, max_gap_days=730)
        
        # Cap with real IPO and delist dates
        real_start, real_end = date_map.get(inst, ('2000-01-01', '9999-12-31'))
        
        for start, end in merged:
            # Ensure start >= IPO date
            if start < real_start:
                start = real_start
            # Ensure end <= Delist date
            if end > real_end and not real_end.startswith('9999'):
                end = real_end
            if start <= end or end.startswith('9999'):
                new_records.append(f"{inst}\t{start}\t{end}\n")
                
    with open(filepath, 'w') as f:
        f.writelines(new_records)

for f in ['csi300.txt', 'csi500.txt', 'csi1000.txt', 'sse50.txt']:
    fix_index_files(f)

print("Done fixing all instruments files to reflect true dates and continuous index membership.")
