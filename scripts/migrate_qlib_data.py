"""
数据迁移脚本：从飞牛OS ClickHouse下载沪深主板股票数据到Qlib

功能：
1. 备份当前featrue目录中每个股票的指标列表
2. 清空qlib_data目录
3. 从ClickHouse下载沪深主板股票数据
4. 生成Qlib格式的.bin文件

优化：按前缀分组，先查daily_prices+di（日频大表），再查fi（财务表），客户端合并
"""
import os
import shutil
import pickle
import struct
import math
import pandas as pd
import clickhouse_connect
from tqdm import tqdm
import datetime
from collections import defaultdict
from decimal import Decimal

CH_CONFIG = {
    "host": "192.168.10.102",
    "port": 18123,
    "user": "xufei",
    "password": "xf1987216",
    "database": "quant_db"
}
QLIB_DATA_DIR = r"e:\Quant\Qlibworks\qlib_data"


def get_main_board_stocks(client):
    """从ClickHouse获取沪深主板股票列表"""
    query = """
    SELECT DISTINCT ts_code FROM daily_prices
    WHERE (ts_code LIKE '600%' OR ts_code LIKE '601%' OR ts_code LIKE '603%' OR ts_code LIKE '605%'
        OR ts_code LIKE '000%' OR ts_code LIKE '001%' OR ts_code LIKE '002%' OR ts_code LIKE '003%')
      AND ts_code NOT LIKE '&%' ORDER BY ts_code
    """
    stocks = [row[0] for row in client.query(query).result_rows]
    print(f"    共找到 {len(stocks)} 只主板股票")
    return stocks


def backup_existing_features(qlib_data_dir):
    """备份当前featrue目录中每个股票的指标列表"""
    print("[2] 备份当前featrue目录中的指标列表...")
    features_dir = os.path.join(qlib_data_dir, "features")
    if not os.path.exists(features_dir):
        print("    features目录不存在")
        return {}

    stock_dirs = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
    backup_data = {}
    for stock_dir in tqdm(stock_dirs, desc="备份股票指标"):
        stock_path = os.path.join(features_dir, stock_dir)
        files = [f.replace('.day.bin', '') for f in os.listdir(stock_path) if f.endswith('.day.bin')]
        backup_data[stock_dir] = files

    backup_file = os.path.join(qlib_data_dir, "features_backup.pkl")
    with open(backup_file, 'wb') as f:
        pickle.dump(backup_data, f)
    all_indicators = set()
    for indicators in backup_data.values():
        all_indicators.update(indicators)
    print(f"    已备份 {len(backup_data)} 只股票, {len(all_indicators)} 种指标")
    return backup_data


def clear_qlib_data(qlib_data_dir):
    """清空qlib_data目录"""
    print("[3] 清空qlib_data目录...")
    for sub in ["features", "instruments"]:
        d = os.path.join(qlib_data_dir, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    os.makedirs(os.path.join(qlib_data_dir, "calendars"), exist_ok=True)
    print("    qlib_data目录已清空")


import numpy as np

def save_calendars(client, calendars_dir):
    """保存交易日历"""
    dates = [str(row[0])[:10] for row in client.query("SELECT DISTINCT trade_date FROM daily_prices ORDER BY trade_date").result_rows]
    with open(os.path.join(calendars_dir, "day.txt"), 'w') as f:
        for d in dates:
            f.write(d + "\n")
    print(f"    已保存 {len(dates)} 个交易日")
    return dates


def save_instruments(instruments_dir, stocks):
    """生成instruments文件"""
    sh = [s for s in stocks if s.endswith('.SH')]
    sz = [s for s in stocks if s.endswith('.SZ')]
    for slist, fname in [(sh+sz, "all.txt"), (sh, "all_sh.txt"), (sz, "all_sz.txt")]:
        with open(os.path.join(instruments_dir, fname), 'w') as f:
            for s in slist:
                f.write(f"{s}\t2010-01-01\t9999-99-99\n")
        print(f"    {fname}: {len(slist)} 只")


def to_float(v):
    """统一转float，兼容None/Decimal"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float('nan')
    if isinstance(v, Decimal):
        return float(v)
    return float(v)


def write_bin(filepath, start_index, values_array):
    """写Qlib .bin文件"""
    # 转换为小端 float32 并写入文件，Qlib 规范：第一个元素为 start_index
    np.hstack([start_index, values_array]).astype("<f").tofile(filepath)


def col_safe(c):
    """列名去引号/点号，用作文件名"""
    return c.replace('.', '_').replace('"', '').replace('`', '')



def fetch_and_save(client, stocks, features_dir, calendar_list, start_date="2010-01-01", end_date="2025-12-31"):
    """
    分批下载所有主板股票数据，保存为Qlib .bin格式

    策略：
    - 按前缀分组（如 600, 000）
    - 每个前缀只查一次 daily_prices + daily_indicators (LEFT JOIN)
    - 财务表 financial_indicators 单独一次性查完，用ASOF逻辑在客户端Join
    """
    cal_map = {d: i for i, d in enumerate(calendar_list)}

    # 按前缀分组
    prefix_map = defaultdict(list)
    for s in stocks:
        prefix_map[s[:3]].append(s)

    print(f"\n[4] 开始下载数据，共 {len(prefix_map)} 个前缀组...")

    # 1. 先一次性拉取所有财务数据（financial_indicators 表相对较小）
    print("    拉取 financial_indicators 数据...")
    try:
        fi_sql = """
        SELECT ts_code, ann_date, end_date,
            roe, roa, grossprofit_margin, netprofit_margin,
            debt_to_assets, current_ratio, eps, ocfps,
            netprofit_yoy, dt_netprofit_yoy, basic_eps_yoy,
            tr_yoy, or_yoy, stk_holdernumber, pledge_ratio, eps_forecast
        FROM financial_indicators
        ORDER BY ts_code, ann_date
        """
        fi_result = client.query(fi_sql)
        fi_rows = fi_result.result_rows
        fi_columns = fi_result.column_names
        print(f"      获取 {len(fi_rows)} 行财务数据")
    except Exception as e:
        print(f"      获取财务数据失败: {e}")
        fi_rows = []
        fi_columns = []

    # 按(ts_code, ann_date)组织财务数据，方便ASOF查询
    fi_by_stock = defaultdict(list)
    for row in fi_rows:
        code = str(row[0])
        fi_by_stock[code].append(row)

    unit_stats = {"success": 0, "failed": 0}

    # 2. 按前缀逐批查询
    for prefix, p_stocks in sorted(prefix_map.items()):
        print(f"\n    前缀 {prefix} ({len(p_stocks)}只)...")

        sql = f"""
        SELECT dp.ts_code, dp.trade_date,
            dp.open, dp.high, dp.low, dp.close, dp.vol, dp.amount,
            di.turnover_rate, di.turnover_rate_f, di.volume_ratio,
            di.pe, di.pe_ttm, di.pb, di.ps, di.ps_ttm,
            di.total_mv, di.circ_mv, di.dv_ttm, di.rzye, di.north_hold
        FROM daily_prices dp
        LEFT JOIN daily_indicators di ON dp.ts_code = di.ts_code AND dp.trade_date = di.trade_date
        WHERE dp.ts_code LIKE '{prefix}%'
          AND dp.ts_code NOT LIKE '&%'
          AND dp.trade_date >= '{start_date}'
          AND dp.trade_date <= '{end_date}'
        ORDER BY dp.ts_code, dp.trade_date
        """
        try:
            result = client.query(sql)
            rows = result.result_rows
            cols = result.column_names
        except Exception as e:
            print(f"      查询失败: {e}")
            unit_stats["failed"] += len(p_stocks)
            continue
        if not rows:
            unit_stats["failed"] += len(p_stocks)
            continue

        # 按股票代码分组
        stock_data = defaultdict(list)
        for row in rows:
            stock_data[str(row[0])].append(row)

        for stock_code in p_stocks:
            if stock_code not in stock_data:
                unit_stats["failed"] += 1
                continue

            raw_rows = stock_data[stock_code]
            code_short = stock_code.lower()

            # 转DataFrame
            records = [{col: row[idx] for idx, col in enumerate(cols)} for row in raw_rows]
            df = pd.DataFrame(records)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date').sort_index()

            stock_dir = os.path.join(features_dir, code_short)
            os.makedirs(stock_dir, exist_ok=True)

            written = 0

            # 基础行情 + 日频指标
            base_cols = {
                'open': 'open', 'high': 'high', 'low': 'low',
                'close': 'close', 'volume': 'vol', 'amount': 'amount',
                'circ_mv': 'circ_mv', 'total_mv': 'total_mv',
                'pe': 'pe', 'pe_ttm': 'pe_ttm', 'pb': 'pb', 'ps': 'ps', 'ps_ttm': 'ps_ttm',
                'turnover_rate': 'turnover_rate', 'turnover_rate_f': 'turnover_rate_f',
                'volume_ratio': 'volume_ratio',
                'dv_ttm': 'dv_ttm', 'rzye': 'rzye', 'north_hold': 'north_hold',
            }
            for qlib_name, ch_col in base_cols.items():
                if ch_col in df.columns:
                    s = df[ch_col].dropna()
                    if len(s) > 0:
                        s_dates = [d.strftime('%Y-%m-%d') for d in s.index]
                        valid_pairs = [(cal_map[d], to_float(v)) for d, v in zip(s_dates, s.values) if d in cal_map]
                        
                        if valid_pairs:
                            valid_pairs.sort(key=lambda x: x[0])
                            start_index = valid_pairs[0][0]
                            end_index = valid_pairs[-1][0]
                            
                            length = end_index - start_index + 1
                            values_array = np.full(length, np.nan, dtype=np.float32)
                            
                            for idx, val in valid_pairs:
                                values_array[idx - start_index] = val
                                
                            write_bin(os.path.join(stock_dir, f"{qlib_name}.day.bin"), start_index, values_array)
                            written += len(valid_pairs)

            # 财务指标（ASOF Join：取最近一期财务数据）
            fi_records = fi_by_stock.get(stock_code, [])
            if fi_records and len(df) > 0:
                fi_records.sort(key=lambda r: str(r[1]))
                fi_col_names = fi_columns

                # 对于每一行，找最近的财务数据
                fi_idx = 0
                fi_vals = []
                for trade_date_str in df.index:
                    td = datetime.datetime.strptime(str(trade_date_str)[:10], '%Y-%m-%d').date() if isinstance(trade_date_str, str) else trade_date_str
                    if hasattr(td, 'date') and not isinstance(td, datetime.date):
                        td = td.date()
                    elif isinstance(td, pd.Timestamp):
                        td = td.date()

                    while fi_idx < len(fi_records) and str(fi_records[fi_idx][1]) <= str(td):
                        fi_idx += 1
                    if fi_idx > 0:
                        fi_vals.append(fi_records[fi_idx - 1])
                    else:
                        fi_vals.append(None)

                fi_field_map = {
                    'roe': 3, 'roa': 4, 'grossprofit_margin': 5, 'netprofit_margin': 6,
                    'debt_to_assets': 7, 'current_ratio': 8, 'eps': 9, 'ocfps': 10,
                    'netprofit_yoy': 11, 'dt_netprofit_yoy': 12, 'basic_eps_yoy': 13,
                    'tr_yoy': 14, 'or_yoy': 15, 'stk_holdernumber': 16, 'pledge_ratio': 17,
                    'eps_forecast': 18,
                }
                for qlib_name, fi_idx_col in fi_field_map.items():
                    valid_pairs = []
                    for di, fi_row in zip(df.index, fi_vals):
                        if fi_row is not None and fi_idx_col < len(fi_row) and fi_row[fi_idx_col] is not None:
                            v = to_float(fi_row[fi_idx_col])
                            if not math.isnan(v):
                                d_str = di.strftime('%Y-%m-%d')
                                if d_str in cal_map:
                                    valid_pairs.append((cal_map[d_str], v))
                                    
                    if valid_pairs:
                        valid_pairs.sort(key=lambda x: x[0])
                        start_index = valid_pairs[0][0]
                        end_index = valid_pairs[-1][0]
                        
                        length = end_index - start_index + 1
                        values_array = np.full(length, np.nan, dtype=np.float32)
                        
                        for idx, val in valid_pairs:
                            values_array[idx - start_index] = val
                            
                        write_bin(os.path.join(stock_dir, f"{qlib_name}.day.bin"), start_index, values_array)
                        written += len(valid_pairs)

            if written > 0:
                unit_stats["success"] += 1
            else:
                unit_stats["failed"] += 1
                shutil.rmtree(stock_dir, ignore_errors=True)

    return unit_stats["success"], unit_stats["failed"]


def main():
    print("=" * 60)
    print("Qlib数据迁移：从飞牛OS ClickHouse下载沪深主板数据")
    print("=" * 60)

    fd = os.path.join(QLIB_DATA_DIR, "features")
    insd = os.path.join(QLIB_DATA_DIR, "instruments")
    cald = os.path.join(QLIB_DATA_DIR, "calendars")

    print("\n[0] 连接飞牛OS ClickHouse...")
    client = clickhouse_connect.get_client(**CH_CONFIG)
    print("    ClickHouse连接成功！")

    stocks = get_main_board_stocks(client)
    if not stocks:
        print("    未获取到股票列表，退出")
        return

    backup_existing_features(QLIB_DATA_DIR)
    calendar_list = save_calendars(client, cald)
    clear_qlib_data(QLIB_DATA_DIR)

    success, failed = fetch_and_save(client, stocks, fd, calendar_list)
    save_instruments(insd, stocks)

    print("\n" + "=" * 60)
    print(f"迁移完成！ 成功: {success} 只 | 失败: {failed} 只 | 总计: {len(stocks)} 只")
    print("=" * 60)


if __name__ == "__main__":
    main()
