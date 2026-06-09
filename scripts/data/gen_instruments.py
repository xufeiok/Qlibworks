"""
指数成分股 Instruments 文件生成器

功能：
  - 从 Tushare 获取主要指数（沪深300/中证500/上证50等）的成分股
  - 基于 all.txt 全库校验，正确填写每只股票在指数中的起止时间
  - 生成 Qlib 可直接加载的 instruments 文件

格式：
  code  start_date  end_date  (tab分隔)
  - 上市日期：取 all.txt 中的挂牌日
  - 退市日期：尚未退市记为 2099-12-31
  - 成分股期间：取该股票在指数中的纳入期

用法：
  python scripts/data/gen_instruments.py
"""

import os
import sys
import time
import pandas as pd
import tushare as ts

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ALL_TXT = os.path.join(ROOT, 'qlib_data/instruments/all.txt')
OUT_DIR = os.path.join(ROOT, 'qlib_data/instruments')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tushare 配置：token 及缓存目录
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
dotenv_path = os.path.join(ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN') or '18dd374714956ab83ae5c20124e490bf0c35fed2'

# Tushare 会在用户目录写 tk.csv，重定向到项目目录
TUSHARE_CACHE = os.path.join(ROOT, '.tushare_cache')
os.makedirs(TUSHARE_CACHE, exist_ok=True)
os.environ['TUSHARE_CACHE_DIR'] = TUSHARE_CACHE

# ---------------------------------------------------------------------------
# 指数列表（文件名, 指数代码, 中文名）
# ---------------------------------------------------------------------------
INDEX_CONFIG = [
    ('csi300.txt',  '000300.SH', '沪深300'),
    ('csi500.txt',  '000905.SH', '中证500'),
    ('csi1000.txt', '000852.SH', '中证1000'),
    ('ssi50.txt',   '000016.SH', '上证50'),
    ('csi800.txt',  '000906.SH', '中证800'),
]

# 年份范围（Tushare index_weight 最早数据约在 2010 年前后）
START_YEAR = 2010
END_YEAR = 2027

REQUEST_INTERVAL = 0.35  # Tushare 请求间隔（秒）


def load_all_stocks() -> dict[str, tuple[str, str]]:
    """
    读取 all.txt，返回 {code: (list_date, end_date)} 字典。
    用作每只股票上市日期与退市日期的权威来源。
    """
    stocks = {}
    with open(ALL_TXT, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                code, start, end = parts[0], parts[1], parts[2]
                stocks[code] = (start, end)
    print(f"  [基表] all.txt: {len(stocks)} 只股票")
    return stocks


def fetch_index_constituents_tushare(index_code: str) -> pd.DataFrame:
    """
    从 Tushare 拉取某指数全部历史的成分股变更记录。
    
    遍历 2005~2026 每季度，获取 index_weight 记录。
    返回字段: con_code, trade_date
    """
    pro = ts.pro_api(TUSHARE_TOKEN)

    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in [1, 4, 7, 10]:
            start_date = f"{year}{month:02d}01"
            end_date   = f"{year}{month:02d}28"
            try:
                df = pro.index_weight(index_code=index_code, start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    frames.append(df[['con_code', 'trade_date']])
                time.sleep(REQUEST_INTERVAL)
            except Exception as e:
                print(f"    WARN {start_date}~{end_date}: {e}")
                time.sleep(REQUEST_INTERVAL * 2)

    if not frames:
        raise RuntimeError(f"未能从 Tushare 获取 {index_code} 的任何成分股数据")

    raw = pd.concat(frames, ignore_index=True)
    raw['trade_date'] = pd.to_datetime(raw['trade_date'])
    raw = raw.drop_duplicates().sort_values(['con_code', 'trade_date']).reset_index(drop=True)
    return raw


def build_hold_periods(raw: pd.DataFrame, all_stocks: dict) -> list[tuple[str, str, str]]:
    """
    将 Tushare 的快照数据转为每只股票在指数中的连续期间。
    
    对每只股票取其首次出现在快照的日期作为 start_date，
    最后一次出现的日期作为 end_date，同时以 all.txt 中的挂牌日为下限。
    在指数中的股票 end_date 标为 2099-12-31。
    """
    records = []
    latest_overall_date = raw['trade_date'].max()

    for code, group in raw.groupby('con_code'):
        code_std = code
        list_info = all_stocks.get(code_std)
        if list_info is None:
            print(f"    SKIP {code_std}: 不在 all.txt 中，跳过")
            continue

        list_date = pd.Timestamp(list_info[0])
        dates = group['trade_date'].tolist()
        first_appear = min(dates)
        last_appear = max(dates)

        # 纳入起始日 = max(挂牌日, 首次出现在快照)
        start = max(first_appear, list_date)

        # 如果最后一次出现是最近期，标为仍在指数中
        if last_appear == latest_overall_date:
            end = '2099-12-31'
        else:
            end = last_appear.strftime('%Y-%m-%d')

        records.append((code_std, start.strftime('%Y-%m-%d'), end))

    # 去重：同一只股票只保留一条记录
    seen = set()
    dedup = []
    for r in records:
        key = r[0]
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    return dedup


def save_instruments_file(filename: str, records: list[tuple[str, str, str]], index_name: str):
    filepath = os.path.join(OUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for code, start, end in records:
            f.write(f"{code}\t{start}\t{end}\n")
    print(f"  [完成] {index_name} -> {filename}: {len(records)} 只成分股", flush=True)


def main():
    print("=" * 60)
    print("  指数成分股 Instruments 文件生成器")
    print(f"  输出目录: {OUT_DIR}")
    print("=" * 60)

    print("\n[1] 加载 A 股全量股票基表...")
    all_stocks = load_all_stocks()

    print("\n[2] 逐个获取指数成分股...")
    for filename, index_code, index_name in INDEX_CONFIG:
        print(f"\n  {'─'*50}")
        print(f"  [{index_name}] {index_code} → {filename}", flush=True)
        try:
            raw = fetch_index_constituents_tushare(index_code)
            print(f"  获取 {len(raw['con_code'].unique())} 只成分股，{len(raw)} 条快照记录", flush=True)
            records = build_hold_periods(raw, all_stocks)
            if records:
                save_instruments_file(filename, records, index_name)
            else:
                print(f"  FAIL 未生成任何有效记录")
        except Exception as e:
            print(f"  FAIL {e}")

    print(f"\n{'='*60}")
    print("  生成完成！文件清单:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"    {fname}: {line_count} 行")
    print("=" * 60)


if __name__ == '__main__':
    main()
