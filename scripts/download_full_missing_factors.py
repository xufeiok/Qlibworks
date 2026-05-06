import tushare as ts
import pandas as pd
import clickhouse_connect
from datetime import datetime
import time
from tqdm import tqdm
import os

# ClickHouse 连接配置
CH_HOST = "192.168.10.102"
CH_PORT = 18123
CH_NATIVE_PORT = 9000
CH_USER = "xufei"
CH_PASSWORD = "xf1987216"
CH_DB = "quant_db"

def get_tushare_token():
    token_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Quant_Tushare', 'config', '.token')
    try:
        with open(token_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"警告: 无法读取 token 文件 {token_path}: {e}")
        # 回退到默认 token
        return '18dd374714956ab83ae5c2028613bee423ce620124e490bf0c35fed2'

def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        user=CH_USER,
        password=CH_PASSWORD,
        database=CH_DB
    )

def bulk_update_clickhouse(client, df, target_table, temp_table_name, dict_name, join_keys, update_col):
    if df.empty:
        print(f"数据为空，跳过 {update_col[0]} 的更新。")
        return
        
    # 去重处理，保证键的唯一性 (使用最后一次出现的值)
    join_key_names = [k[0] for k in join_keys]
    df = df.drop_duplicates(subset=join_key_names, keep='last')
        
    print(f"正在将 {len(df)} 条记录上传至 ClickHouse 临时表...")
    
    cols_def = ", ".join([f"{k[0]} {k[1]}" for k in join_keys]) + f", {update_col[0]} {update_col[1]}"
    client.command(f"CREATE TABLE IF NOT EXISTS {temp_table_name} ({cols_def}) ENGINE = Memory")
    client.command(f"TRUNCATE TABLE {temp_table_name}")
    
    df_insert = df[join_key_names + [update_col[0]]].dropna()
    client.insert_df(temp_table_name, df_insert)
    
    print(f"正在创建字典并执行全局 ALTER TABLE UPDATE...")
    key_names = ", ".join(join_key_names)
    
    # 修复 SOURCE 配置，使用全局变量
    client.command(f"""
        CREATE DICTIONARY IF NOT EXISTS default.{dict_name}
        (
            {cols_def}
        )
        PRIMARY KEY {key_names}
        SOURCE(CLICKHOUSE(HOST '{CH_HOST}' PORT {CH_NATIVE_PORT} USER '{CH_USER}' PASSWORD '{CH_PASSWORD}' DB '{CH_DB}' TABLE '{temp_table_name}'))
        LIFETIME(MIN 0 MAX 0)
        LAYOUT(COMPLEX_KEY_HASHED())
    """)
    
    tuple_keys = ", ".join([f"CAST({k[0]} AS String)" for k in join_keys])
    
    client.command(f"""
        ALTER TABLE {target_table} 
        UPDATE {update_col[0]} = dictGet('default.{dict_name}', '{update_col[0]}', tuple({tuple_keys}))
        WHERE dictHas('default.{dict_name}', tuple({tuple_keys}))
    """)
    
    client.command(f"DROP DICTIONARY IF EXISTS default.{dict_name}")
    client.command(f"DROP TABLE {temp_table_name}")
    print(f"[{update_col[0]}] 批次更新指令已发送（ClickHouse 后台 Mutation 中）。\n")

def main():
    print("--- 初始化 Tushare ---")
    token = get_tushare_token()
    ts.set_token(token)
    pro = ts.pro_api()

    print("--- 连接 ClickHouse ---")
    client = get_clickhouse_client()
    
    start_date = '20100101'
    end_date = '20260430'
    print(f"补全数据范围: {start_date} 到 {end_date}")
    
    cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
    trade_dates = cal['cal_date'].tolist()

    # 极速且安全的优化方案：将按日/按年循环中的数据按批次（例如按年）进行合并和 UPDATE
    # 避免 OOM，同时也显著降低了 ClickHouse 的 Mutation 压力

    # 1. 股东人数 (stk_holdernumber) -> financial_indicators
    print("开始分批下载与更新 股东人数 (stk_holdernumber)...")
    for year in range(2010, 2027):
        print(f"\n--- 处理 {year} 年 股东人数 ---")
        y_start = f"{year}0101"
        y_end = f"{year}1231" if year < 2026 else "20260430"
        try:
            df = pro.stk_holdernumber(start_date=y_start, end_date=y_end)
            if not df.empty:
                df.rename(columns={'holder_num': 'stk_holdernumber'}, inplace=True)
                bulk_update_clickhouse(client, df, "financial_indicators", f"temp_stk_{year}", f"dict_stk_{year}",
                                       [('ts_code', 'String'), ('ann_date', 'String')], ('stk_holdernumber', 'Float64'))
            time.sleep(0.3)
        except Exception as e:
            print(f"[{year}] stk_holdernumber 失败: {e}")

    # 2. 业绩预告 (eps_forecast) -> financial_indicators
    print("开始分批下载与更新 业绩预告 (eps_forecast)...")
    for year in range(2010, 2027):
        print(f"\n--- 处理 {year} 年 业绩预告 ---")
        y_start = f"{year}0101"
        y_end = f"{year}1231" if year < 2026 else "20260430"
        try:
            # 修复：Tushare 的 forecast 接口要求 ann_date 和 ts_code 至少输入一个参数。
            # 但按年拉取全市场数据时，由于没有 ts_code，可能会报错。
            # 因此，这里改用 trade_dates（按日循环拉取该年的预告）
            y_dates = [d for d in trade_dates if d.startswith(str(year))]
            fc_dfs = []
            for date in tqdm(y_dates, desc=f"{year} forecast"):
                try:
                    df = pro.forecast(ann_date=date)
                    if not df.empty and 'net_profit_min' in df.columns:
                        fc_dfs.append(df)
                    time.sleep(0.12)
                except Exception as inner_e:
                    pass
            
            if fc_dfs:
                df = pd.concat(fc_dfs, ignore_index=True)
                df.rename(columns={'net_profit_min': 'eps_forecast'}, inplace=True)
                bulk_update_clickhouse(client, df, "financial_indicators", f"temp_fc_{year}", f"dict_fc_{year}",
                                       [('ts_code', 'String'), ('ann_date', 'String')], ('eps_forecast', 'Float64'))
        except Exception as e:
            print(f"[{year}] forecast 失败: {e}")

    # 3. 每日指标 (dv_ttm) -> daily_indicators
    print("开始分批下载与更新 股息率 (dv_ttm)...")
    for year in range(2010, 2027):
        print(f"\n--- 处理 {year} 年 股息率 ---")
        y_dates = [d for d in trade_dates if d.startswith(str(year))]
        dv_dfs = []
        for date in tqdm(y_dates, desc=f"{year} dv_ttm"):
            try:
                df = pro.daily_basic(trade_date=date)
                if not df.empty and 'dv_ttm' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                    dv_dfs.append(df[['ts_code', 'trade_date', 'dv_ttm']])
                time.sleep(0.12)
            except Exception as e:
                pass
        if dv_dfs:
            dv_all = pd.concat(dv_dfs, ignore_index=True)
            # dv_ttm 按月度批次更新以缓解 TOO_MANY_MUTATIONS 错误
            # 由于 daily_indicators 数据量庞大，且有 2010 到 2026 共 17 年，如果按年也会有 17 次 mutations 积压
            # 但是之前报错是因为按日更新，现在按年理论上已经大幅度下降。
            # 为了防止 Too many unfinished mutations，我们可以加入同步等待或者采用更小的分批。
            # 这里我们在执行命令后主动 sleep 缓冲，让 ClickHouse 有时间处理后台 Mutation
            bulk_update_clickhouse(client, dv_all, "daily_indicators", f"temp_dv_{year}", f"dict_dv_{year}",
                                   [('ts_code', 'String'), ('trade_date', 'String')], ('dv_ttm', 'Float64'))
            print("等待 5 秒以缓解 ClickHouse Mutation 队列压力...")
            time.sleep(5)

    # 4. 融资融券 (rzye) -> daily_indicators
    print("开始分批下载与更新 融资余额 (rzye)...")
    for year in range(2010, 2027):
        print(f"\n--- 处理 {year} 年 融资余额 ---")
        y_dates = [d for d in trade_dates if d.startswith(str(year))]
        rz_dfs = []
        for date in tqdm(y_dates, desc=f"{year} rzye"):
            try:
                df = pro.margin_detail(trade_date=date) # Note: margin_detail has ts_code
                if not df.empty and 'rzye' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                    rz_dfs.append(df[['ts_code', 'trade_date', 'rzye']])
                time.sleep(0.12)
            except Exception as e:
                pass
        if rz_dfs:
            rz_all = pd.concat(rz_dfs, ignore_index=True)
            bulk_update_clickhouse(client, rz_all, "daily_indicators", f"temp_rz_{year}", f"dict_rz_{year}",
                                   [('ts_code', 'String'), ('trade_date', 'String')], ('rzye', 'Float64'))
            print("等待 5 秒以缓解 ClickHouse Mutation 队列压力...")
            time.sleep(5)

    # 5. 北向资金 (north_hold) -> daily_indicators
    print("开始分批下载与更新 北向持股 (north_hold)...")
    for year in range(2010, 2027):
        print(f"\n--- 处理 {year} 年 北向持股 ---")
        y_dates = [d for d in trade_dates if d.startswith(str(year))]
        hk_dfs = []
        for date in tqdm(y_dates, desc=f"{year} north_hold"):
            try:
                df = pro.hk_hold(trade_date=date)
                if not df.empty and 'vol' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                    df.rename(columns={'vol': 'north_hold'}, inplace=True)
                    hk_dfs.append(df[['ts_code', 'trade_date', 'north_hold']])
                time.sleep(0.12)
            except Exception as e:
                pass
        if hk_dfs:
            hk_all = pd.concat(hk_dfs, ignore_index=True)
            bulk_update_clickhouse(client, hk_all, "daily_indicators", f"temp_hk_{year}", f"dict_hk_{year}",
                                   [('ts_code', 'String'), ('trade_date', 'String')], ('north_hold', 'Float64'))
            print("等待 5 秒以缓解 ClickHouse Mutation 队列压力...")
            time.sleep(5)

    print("\n所有数据下载与全局更新任务已提交完毕！")

if __name__ == '__main__':
    main()
