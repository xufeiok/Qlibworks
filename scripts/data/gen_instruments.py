import os
import sys
import time
import duckdb
import pandas as pd
import tushare as ts

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../qlib_data/instruments'))
os.makedirs(OUT_DIR, exist_ok=True)

DB_PATH = r'e:/Quant/Quant_Tushare/data/quant_data.duckdb'
TUSHARE_TOKEN = '18dd374714956ab83ae5c2028613bee423ce620124e490bf0c35fed2'

# 需要生成的指数列表
# (文件名, 指数代码, 数据源方式: 'db'=从stock_universe字段, 'ts'=从Tushare)
INDEX_CONFIG = [
    ('csi300.txt',    '000300.SH', 'db',   'is_hs300'),
    ('csi500.txt',    '000905.SH', 'ts',   None),
    ('csi1000.txt',   '000852.SH', 'ts',   None),
    ('csi800.txt',    '000906.SH', 'ts',   None),
    ('ssi50.txt',     '000016.SH', 'ts',   None),
    ('csiA100.txt',   '000903.SH', 'ts',   None),
    ('sz100.txt',     '399330.SZ', 'ts',   None),
    ('zz1000.txt',    '000852.SH', 'ts',   None),
]

con = duckdb.connect(DB_PATH, read_only=True)


def get_stock_date_ranges(con, ts_codes: list[str]) -> dict:
    codes_tuple = tuple(ts_codes)
    if len(codes_tuple) == 1:
        codes_tuple = f"('{codes_tuple[0]}')"
    df = con.execute(f"""
        SELECT ts_code,
               CAST(MIN(trade_date) AS VARCHAR) as start_date,
               CAST(MAX(trade_date) AS VARCHAR) as end_date
        FROM daily_prices
        WHERE ts_code IN {codes_tuple}
        GROUP BY ts_code
    """).df()
    result = {}
    for _, row in df.iterrows():
        result[row['ts_code']] = (row['start_date'], row['end_date'])
    return result


def save_instruments_file(filename: str, records: list[tuple[str, str, str]]):
    filepath = os.path.join(OUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for code, start, end in records:
            f.write(f"{code}\t{start}\t{end}\n")
    print(f"  OK {filename}: {len(records)} 条记录 -> {filepath}")


def gen_from_db_field(con, filename: str, field_name: str):
    print(f"\n{'='*50}")
    print(f"从数据库字段 {field_name} 生成 {filename}")
    df = con.execute(f"""
        SELECT ts_code FROM stock_universe
        WHERE {field_name} = true AND list_status = 'L'
    """).df()
    if df.empty:
        print(f"  FAIL 数据库中无符合条件的记录")
        return False
    codes = df['ts_code'].tolist()
    date_ranges = get_stock_date_ranges(con, codes)
    records = []
    for code in codes:
        if code in date_ranges:
            start, end = date_ranges[code]
            records.append((code, start, end))
        else:
            print(f"  WARN {code} 在 daily_prices 中无交易数据，跳过")
    if records:
        save_instruments_file(filename, records)
        return True
    return False


def fetch_and_gen_from_tushare(filename: str, index_code: str, start_year=2018, end_year=2026):
    print(f"\n{'='*50}")
    print(f"从 Tushare 获取 {index_code} 成分股 -> {filename}")
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    all_dfs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            sy = f"{year}{month:02d}01"
            ey = f"{year}{month:02d}31"
            try:
                df = pro.index_weight(index_code=index_code, start_date=sy, end_date=ey)
                if df is not None and not df.empty:
                    all_dfs.append(df)
                time.sleep(0.35)
            except Exception as e:
                print(f"  WARN {sy}-{ey}: {e}")
                time.sleep(0.5)

    if not all_dfs:
        print(f"  FAIL 未能从 Tushare 获取 {index_code} 的任何数据")
        return False

    raw = pd.concat(all_dfs, ignore_index=True)
    raw['trade_date'] = pd.to_datetime(raw['trade_date'])
    raw = raw.sort_values(['con_code', 'trade_date'])

    unique_codes = raw['con_code'].unique().tolist()
    print(f"  获取到 {len(unique_codes)} 只成分股，正在查询交易日期范围...")
    date_ranges = get_stock_date_ranges(con, unique_codes)

    records = []
    latest_date = raw['trade_date'].max()
    for code, group in raw.groupby('con_code'):
        group = group.sort_values('trade_date')
        dates = group['trade_date'].tolist()

        start = dates[0]
        prev = dates[0]
        for d in dates[1:]:
            if (d - prev).days > 45:
                if code in date_ranges:
                    records.append((code, start.strftime('%Y-%m-%d'), prev.strftime('%Y-%m-%d')))
                start = d
            prev = d

        end_str = '9999-12-31' if prev == latest_date else prev.strftime('%Y-%m-%d')
        if code in date_ranges:
            records.append((code, start.strftime('%Y-%m-%d'), end_str))
        else:
            if prev == latest_date:
                records.append((code, start.strftime('%Y-%m-%d'), '9999-12-31'))

    if records:
        save_instruments_file(filename, records)
        return True
    return False


if __name__ == '__main__':
    print("=" * 60)
    print("  指数成分股 Instruments 文件生成器")
    print("  数据源: DuckDB + Tushare")
    print("=" * 60)

    for filename, index_code, source, field in INDEX_CONFIG:
        if source == 'db':
            gen_from_db_field(con, filename, field)
        elif source == 'ts':
            fetch_and_gen_from_tushare(filename, index_code)

    con.close()

    print(f"\n{'='*60}")
    print("  生成完成！生成的文件:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"    {fname}: {line_count} 行")
    print("=" * 60)
