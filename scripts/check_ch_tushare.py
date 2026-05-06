import clickhouse_connect
import tushare as ts
import pandas as pd
import json

def check_clickhouse():
    print("--- Checking ClickHouse ---")
    try:
        ch_client = clickhouse_connect.get_client(
            host="192.168.10.102",
            port=18123,
            user="xufei",
            password="xf1987216",
            database="quant_db"
        )
        
        tables = ["daily_indicators", "financial_indicators"]
        for table in tables:
            print(f"\\nColumns in {table}:")
            try:
                res = ch_client.query(f"DESCRIBE TABLE {table}")
                cols = [row[0] for row in res.result_rows]
                print(cols)
            except Exception as e:
                print(f"Error querying {table}: {e}")
                
        print("\\nOther tables in quant_db:")
        res = ch_client.query("SHOW TABLES")
        print([row[0] for row in res.result_rows])
                
    except Exception as e:
        print(f"Failed to connect to ClickHouse: {e}")

def check_tushare():
    print("\\n--- Checking Tushare Permissions ---")
    try:
        # Tushare token might be in config or environment. Let's try to find it.
        # usually in e:\Quant\tushare_token.txt or similar, let's assume it's initialized if we import something
        # or we can read it from a common place.
        import os
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            try:
                with open("e:/Quant/Qlibworks/scripts/tushare_token.txt", "r") as f:
                    token = f.read().strip()
            except:
                pass
                
        if not token:
            print("Please provide Tushare token.")
            # Trying default init if token is globally set
            pro = ts.pro_api()
        else:
            ts.set_token(token)
            pro = ts.pro_api()
            
        print("Testing daily_basic (for dv_ttm, turnover_rate etc)...")
        try:
            df = pro.daily_basic(ts_code='000001.SZ', trade_date='20230104')
            print("daily_basic success, columns:", df.columns.tolist())
        except Exception as e:
            print("daily_basic failed:", e)
            
        print("Testing stk_holdernumber (shareholder_change)...")
        try:
            df = pro.stk_holdernumber(ts_code='000001.SZ', start_date='20230101', end_date='20231231')
            print("stk_holdernumber success, columns:", df.columns.tolist())
        except Exception as e:
            print("stk_holdernumber failed:", e)
            
        print("Testing pledge_stat (pledge_ratio)...")
        try:
            df = pro.pledge_stat(ts_code='000001.SZ')
            print("pledge_stat success, columns:", df.columns.tolist())
        except Exception as e:
            print("pledge_stat failed:", e)
            
        print("Testing hk_hold (north_fund_change)...")
        try:
            df = pro.hk_hold(ts_code='000001.SZ', start_date='20230101', end_date='20230110')
            print("hk_hold success, columns:", df.columns.tolist())
        except Exception as e:
            print("hk_hold failed:", e)
            
        print("Testing margin_detail (margin_balance_change)...")
        try:
            df = pro.margin_detail(ts_code='000001.SZ', start_date='20230101', end_date='20230110')
            print("margin_detail success, columns:", df.columns.tolist())
        except Exception as e:
            print("margin_detail failed:", e)
            
        print("Testing forecast (eps_forecast_yoy)...")
        try:
            df = pro.forecast(ts_code='000001.SZ', start_date='20230101', end_date='20231231')
            print("forecast success, columns:", df.columns.tolist())
        except Exception as e:
            print("forecast failed:", e)

    except Exception as e:
        print(f"Tushare test failed: {e}")

if __name__ == '__main__':
    check_clickhouse()
    check_tushare()