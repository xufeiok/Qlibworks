#!/usr/bin/env python
"""
从 Tushare 下载数据，补全 ClickHouse 中的空表。

支持断点续传：每批数据成功写入 ClickHouse 后记录进度，
中断后重新运行会自动跳过已完成的日期。

支持的表：
  - daily_indicators  (daily_basic + margin_detail + hk_hold)
  - money_flow        (moneyflow)
  - stock_universe    (stock_basic)

用法:
  # 全量下载（断点续传）
  python scripts/download_empty_tables.py

  # 只补某张表
  python scripts/download_empty_tables.py --skip money_flow stock_universe

  # 重置某张表的进度重新下载
  python scripts/download_empty_tables.py --reset daily_indicators
"""

import json
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import tushare as ts
import clickhouse_connect
import datetime as dt_mod
from datetime import datetime


def clean_df_for_clickhouse(df):
    """清理 DataFrame 使数据能安全写入 ClickHouse：
       - None 在非 Nullable String 列 -> ''
       - NaN/NaT 在数值或日期列 -> None
    """
    df = df.copy()
    for col in df.columns:
        na_mask = df[col].isna()
        if not na_mask.any():
            continue
        if na_mask.all():
            continue
        try:
            sample = df[col].dropna().iloc[0]
        except (IndexError, KeyError):
            continue
        # 字符串列: None -> ''
        if isinstance(sample, str):
            df[col] = df[col].fillna('')
        # 日期列: NaT/None -> None (clickhouse_connect 识别 None 为 NULL)
        elif isinstance(sample, (datetime, dt_mod.date)):
            series = df[col].astype(object)
            series.loc[na_mask] = None
            df[col] = series
        # 数值列: NaN -> None
        elif isinstance(sample, (int, float, np.floating, np.integer)):
            series = df[col].astype(object)
            series.loc[na_mask] = None
            df[col] = series
        # 其他未知类型，尝试直接替换
        else:
            series = df[col].astype(object)
            series.loc[na_mask] = None
            df[col] = series
    return df

# =========================== 配置 ===========================
TUSHARE_TOKEN = "18dd374714956ab83ae5c2028613bee423ce620124e490bf0c35fed2"

CH_CONFIG = {
    "host": "10.100.0.205",
    "port": 18123,
    "user": "xufei",
    "password": "xf1987216",
    "database": "quant_db",
}

# 默认数据起止
START_DATE = "20100101"
END_DATE = "20260430"

# API 调用间隔
API_INTERVAL = 0.35

# 每批处理的天数
BATCH_SIZE = 20

# 进度文件
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), ".download_progress.json")


def parse_args():
    parser = argparse.ArgumentParser(description='Tushare -> ClickHouse 数据补全（支持断点续传）')
    parser.add_argument('--start', default=START_DATE, help='起始日期 YYYYMMDD')
    parser.add_argument('--end', default=END_DATE, help='截止日期 YYYYMMDD')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='每批天数')
    parser.add_argument('--skip', nargs='*', default=[], help='跳过的表')
    parser.add_argument('--reset', nargs='*', default=[], help='重置进度重新下载某张表')
    return parser.parse_args()


def load_progress():
    """加载进度文件"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(progress):
    """保存进度文件"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_trade_dates(pro, start_date, end_date):
    cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
    return sorted(cal['cal_date'].tolist())


def fmt_progress(current, total):
    return f"{current}/{total} ({current/total*100:.1f}%)"


def api_call(call_fn, *args, max_retries=3, **kwargs):
    """带重试的 Tushare API 调用，遇频率限制自动等待后重试"""
    for attempt in range(max_retries):
        try:
            result = call_fn(*args, **kwargs)
            return result
        except Exception as e:
            err = str(e)
            if '频率' in err or '频次' in err or '200' in err:
                wait = 5 * (attempt + 1)
                print(f"        [RATE_LIMIT] 等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise e
    # 最后一次重试
    return call_fn(*args, **kwargs)


def download_stock_universe(pro, client, progress, reset=False):
    """下载股票基本信息"""
    print("\n" + "=" * 60)
    print("[1/3] stock_universe - 股票基本信息")
    print("=" * 60)

    tbl = 'stock_universe'
    if progress.get(tbl, {}).get('done') and not reset:
        print("    [SKIP] 已完成，跳过（用 --reset stock_universe 重新下载）")
        return

    try:
        dfs = []
        for status in ['L', 'D', 'P']:
            fields_list = ['ts_code', 'symbol', 'name', 'industry', 'area', 'market',
                           'list_date', 'delist_date', 'is_hs300', 'list_status']
            df = pro.stock_basic(exchange='', list_status=status, fields=','.join(fields_list))
            if df is not None and not df.empty:
                # list_date: 转字符串后统一解析为 date
                df['list_date'] = df['list_date'].astype(str).replace('nan', '20000101').replace('None', '20000101')
                df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
                df['list_date'] = df['list_date'].dt.date
                # delist_date: 先转字符串，NaN/None -> 99991231
                df['delist_date'] = df['delist_date'].astype(str).replace('nan', '99991231').replace('None', '99991231').replace('<NA>', '99991231')
                df['delist_date'] = pd.to_datetime(df['delist_date'], format='%Y%m%d', errors='coerce')
                df['delist_date'] = df['delist_date'].dt.date
                # is_hs300 可能无权限
                if 'is_hs300' not in df.columns:
                    df['is_hs300'] = 0
                else:
                    df['is_hs300'] = df['is_hs300'].fillna(0).astype(int)
                df['is_watchlist'] = 0
                df['added_date'] = df['list_date']
                df['update_time'] = datetime.now()
                dfs.append(df)
            time.sleep(API_INTERVAL)

        if dfs:
            all_df = pd.concat(dfs, ignore_index=True)
            all_df = clean_df_for_clickhouse(all_df)
            client.command("TRUNCATE TABLE stock_universe")
            client.insert_df('stock_universe', all_df)
            verify = client.query(f"SELECT count() FROM stock_universe").result_rows[0][0]
            print(f"    [OK] 写入 {len(all_df)} 条, 当前总行数: {verify}")
            progress[tbl] = {'done': True, 'rows': verify, 'time': datetime.now().isoformat()}
            save_progress(progress)
        else:
            print("    [FAIL] 接口返回空")
    except Exception as e:
        print(f"    [FAIL] 失败: {e}")


def download_daily_table(pro, client, progress, table_name, api_func, api_fields_getter,
                         date_col, check_col, reset=False, api_name=''):
    """
    通用逐日下载函数（支持断点续传）

    参数:
      table_name: ClickHouse 表名
      api_func:   Tushare API 调用函数，参数为 trade_date
      api_fields_getter: 从 API 返回的 DataFrame 中提取需要的字段并返回 DataFrame
      date_col:   日期列名
      check_col:  用于检查该行是否有效的列名
      reset:      是否重置进度
    """
    print(f"\n{'=' * 60}")
    print(f"[{table_name}] {table_name}")
    if api_name:
        print(f"    API: {api_name}")
    print(f"{'=' * 60}")

    tbl_progress = progress.get(table_name, {})
    completed_dates = set(tbl_progress.get('completed_dates', []))

    if reset:
        completed_dates = set()
        # 清空表数据
        client.command(f"TRUNCATE TABLE {table_name}")
        print("    [RESET] 已清空表数据并重置进度")

    # 已经完成的天数
    already_done = len(completed_dates)
    if already_done > 0:
        print(f"    已有 {already_done} 天完成下载，继续未完成的部分")

    # 交易日历
    all_dates = get_trade_dates(pro, args.start, args.end)
    pending_dates = [d for d in all_dates if d not in completed_dates]
    print(f"    总计 {len(all_dates)} 个交易日, 待下载 {len(pending_dates)} 天")

    if not pending_dates:
        print("    [OK] 全部已完成!")
        return

    total_written = tbl_progress.get('total_rows', 0)
    n = len(pending_dates)

    for batch_start in range(0, n, args.batch):
        batch = pending_dates[batch_start:batch_start + args.batch]
        batch_rows = {}
        skipped_in_batch = 0

        for date in batch:
            try:
                df = api_func(trade_date=date)
                data = api_fields_getter(df, date)
                if data is not None and not data.empty:
                    batch_rows[date] = data
                else:
                    skipped_in_batch += 1
            except Exception as e:
                # 接口无权限等
                err = str(e)
                if '暂无权限' in err or '积分' in err:
                    print(f"    [FAIL] {api_name or table_name} 接口无权限: {err}")
                    return False
                print(f"        [WARN] {date}: {e}")
            time.sleep(API_INTERVAL)

        # 写入当前批次
        if batch_rows:
            try:
                batch_df = pd.concat(batch_rows.values(), ignore_index=True)
                # 按 (ts_code, date_col) 去重
                batch_df = batch_df.drop_duplicates(subset=['ts_code', date_col], keep='first')
                # trade_date 列转为 date 类型（ClickHouse Date 列要求）
                if 'trade_date' in batch_df.columns:
                    batch_df['trade_date'] = pd.to_datetime(batch_df['trade_date'], format='%Y%m%d').dt.date
                # NaN 替换为 None（ClickHouse Decimal 列不能接受 NaN）
                batch_df = clean_df_for_clickhouse(batch_df)
                client.insert_df(table_name, batch_df)
                total_written += len(batch_df)

                # 记录进度
                completed_dates.update(batch_rows.keys())
                progress[table_name] = {
                    'completed_dates': sorted(list(completed_dates)),
                    'total_rows': total_written,
                    'last_batch': batch_rows.keys(),
                    'time': datetime.now().isoformat(),
                }
                save_progress(progress)
            except Exception as e:
                print(f"        [WARN] 批次写入失败: {e}")
                # 不记录进度，下次重试会重新下载这批

        done_count = already_done + batch_start + len(batch)
        print(f"    进度: {fmt_progress(min(done_count, len(all_dates)), len(all_dates))} 交易日, "
              f"已写入 {total_written} 行")

    # 最终验证
    cnt = client.query(f"SELECT count() FROM {table_name}").result_rows[0][0]
    try:
        stocks = client.query(f"SELECT count(DISTINCT ts_code) FROM {table_name}").result_rows[0][0]
        print(f"\n    {table_name} 最终: {cnt} 行, {stocks} 只股票")
    except Exception:
        print(f"\n    {table_name} 最终: {cnt} 行")

    # 标记完成
    progress[table_name] = {
        'completed_dates': sorted(list(completed_dates)),
        'total_rows': total_written,
        'done': True,
        'time': datetime.now().isoformat(),
    }
    save_progress(progress)
    return True


def main():
    global args
    args = parse_args()
    print("=" * 60)
    print("Tushare 数据下载 - 补全 ClickHouse 空表")
    print(f"日期范围: {args.start} ~ {args.end}")
    print(f"进度文件: {PROGRESS_FILE}")
    print(f"断点续传: 支持")
    print("=" * 60)

    # 加载进度
    progress = load_progress()
    if args.reset:
        for tbl in args.reset:
            if tbl in progress:
                del progress[tbl]
                print(f"已重置 {tbl} 的进度")
        save_progress(progress)

    # 初始化
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    print("\n[连接 ClickHouse] ...")
    client = clickhouse_connect.get_client(**CH_CONFIG)
    print("    [OK]")

    # ============================================================
    # 1. stock_universe
    # ============================================================
    if 'stock_universe' not in args.skip:
        download_stock_universe(pro, client, progress, reset='stock_universe' in args.reset)
    else:
        print("\n[跳过] stock_universe")

    # ============================================================
    # 2. daily_indicators
    # ============================================================
    if 'daily_indicators' not in args.skip:
        daily_fields = (
            'ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,'
            'pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,total_mv,circ_mv,dv_ttm'
        )

        def di_api(trade_date):
            return pro.daily_basic(trade_date=trade_date, fields=daily_fields)

        def di_fields(df, date):
            if df is None or df.empty:
                return None
            result = df.copy()
            # 再查 margin_detail
            try:
                df_m = pro.margin_detail(trade_date=date)
                if df_m is not None and not df_m.empty and 'rzye' in df_m.columns:
                    result = result.merge(df_m[['ts_code', 'rzye']], on='ts_code', how='left')
            except Exception:
                pass
            time.sleep(API_INTERVAL)

            # 再查 hk_hold
            try:
                df_h = pro.hk_hold(trade_date=date)
                if df_h is not None and not df_h.empty and 'vol' in df_h.columns:
                    h = df_h[['ts_code', 'vol']].copy()
                    h.rename(columns={'vol': 'north_hold'}, inplace=True)
                    result = result.merge(h, on='ts_code', how='left')
            except Exception:
                pass
            time.sleep(API_INTERVAL)

            result['trade_date'] = date
            return result

        reset_di = 'daily_indicators' in args.reset
        download_daily_table(
            pro, client, progress,
            table_name='daily_indicators',
            api_func=di_api,
            api_fields_getter=di_fields,
            date_col='trade_date',
            check_col='pe',
            reset=reset_di,
            api_name='daily_basic + margin_detail + hk_hold'
        )
    else:
        print("\n[跳过] daily_indicators")

    # ============================================================
    # 3. money_flow
    # ============================================================
    if 'money_flow' not in args.skip:
        col_map = {
            'ts_code': 'ts_code', 'trade_date': 'trade_date',
            'net_mf_amount': 'net_mf_amount', 'net_mf_vol': 'net_mf_vol',
            'buy_sm_amount': 'buy_sm_amount', 'buy_sm_vol': 'buy_sm_vol',
            'sell_sm_amount': 'sell_sm_amount', 'sell_sm_vol': 'sell_sm_vol',
            'buy_md_amount': 'buy_md_amount', 'buy_md_vol': 'buy_md_vol',
            'sell_md_amount': 'sell_md_amount', 'sell_md_vol': 'sell_md_vol',
            'buy_lg_amount': 'buy_lg_amount', 'buy_lg_vol': 'buy_lg_vol',
            'sell_lg_amount': 'sell_lg_amount', 'sell_lg_vol': 'sell_lg_vol',
            'buy_elg_amount': 'buy_elg_amount', 'buy_elg_vol': 'buy_elg_vol',
            'sell_elg_amount': 'sell_elg_amount', 'sell_elg_vol': 'sell_elg_vol',
        }

        def mf_api(trade_date):
            return pro.moneyflow(trade_date=trade_date)

        def mf_fields(df, date):
            if df is None or df.empty:
                return None
            avail = {k: v for k, v in col_map.items() if k in df.columns}
            if not avail:
                return None
            result = df[list(avail.keys())].copy()
            result.rename(columns=avail, inplace=True)
            result['trade_date'] = date
            return result

        reset_mf = 'money_flow' in args.reset
        result = download_daily_table(
            pro, client, progress,
            table_name='money_flow',
            api_func=mf_api,
            api_fields_getter=mf_fields,
            date_col='trade_date',
            check_col='net_mf_amount',
            reset=reset_mf,
            api_name='moneyflow'
        )

        if result is False:
            print("\n    [INFO] money_flow 可能无接口权限，跳过")
    else:
        print("\n[跳过] money_flow")

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("最终数据统计：")
    print("=" * 60)
    for tbl in ['stock_universe', 'daily_indicators', 'money_flow']:
        try:
            cnt = client.query(f"SELECT count() FROM {tbl}").result_rows[0][0]
            try:
                stocks = client.query(f"SELECT count(DISTINCT ts_code) FROM {tbl}").result_rows[0][0]
                print(f"  {tbl:25s}: {cnt:>10,d} 行, {stocks} 只股票")
            except Exception:
                print(f"  {tbl:25s}: {cnt:>10,d} 行")
        except Exception:
            print(f"  {tbl:25s}: 查询失败")
    client.close()
    print("\n[OK] 完成！")


if __name__ == '__main__':
    main()
