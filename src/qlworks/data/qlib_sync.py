"""
Qlib 数据同步模块

功能概述：
- 从 ClickHouse 同步数据到 Qlib 格式
- 支持全量同步和增量同步
- 自动生成 Qlib 所需的 calendars、instruments、features 文件
"""
from __future__ import annotations

import os
import shutil
import struct
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from qlworks.config import QLIB_DATA_DIR, FORCE_ADJUSTED_PRICES, FINANCIAL_USE_ANNOUNCEMENT_DATE
import warnings as _warnings


class QlibSynchronizer:
    """
    Qlib 数据同步器
    
    将 ClickHouse 中的数据同步为 Qlib 格式：
    - calendars/day.txt: 交易日历
    - instruments/all.txt: 股票池
    - features/<stock>/<field>.day.bin: 特征数据
    """
    
    def __init__(self, api):
        """
        初始化同步器
        
        Args:
            api: QuantDataAPI 实例
        """
        self.api = api
        self.qlib_dir = Path(QLIB_DATA_DIR)
        self.features_dir = self.qlib_dir / "features"
        self.instruments_dir = self.qlib_dir / "instruments"
        self.calendars_dir = self.qlib_dir / "calendars"
        
        self._print_data_specs()
        
        # Qlib 字段映射 (ClickHouse 字段 -> Qlib 字段)
        # ClickHouse 表结构：
        #   daily_prices: ts_code, trade_date, open, high, low, close, vol, amount
        #   daily_indicators: ts_code, trade_date, pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv, dv_ttm
        #   daily_adj_factors: ts_code, trade_date, adj_factor
        #   financial_indicators: ts_code, ann_date, end_date, roe, roa, grossprofit_margin, etc.
        #
        # 注意：api.py 的 get_daily_data 方法已在 SQL 层面计算前复权价格
        # 当 FORCE_ADJUSTED_PRICES=True 时，返回的 open/high/low/close 已经是前复权价格
        self.field_mapping = {
            # 基础行情（从 get_daily_data 获取，已处理前复权）
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "vol",
            "amount": "amount",
            # 市值指标（从 daily_indicators 表）
            "total_mv": "total_mv",
            "circ_mv": "circ_mv",
            # 估值（从 daily_indicators 表）
            "pe": "pe",
            "pe_ttm": "pe_ttm",
            "pb": "pb",
            "ps": "ps",
            "ps_ttm": "ps_ttm",
            # 动量
            "dv_ttm": "dv_ttm",
            # 财务（从 financial_indicators 表，使用 ann_date）
            "roe": "roe",
            "roa": "roa",
            "grossprofit_margin": "grossprofit_margin",
            "netprofit_margin": "netprofit_margin",
            "debt_to_assets": "debt_to_assets",
            "current_ratio": "current_ratio",
            "eps": "eps",
            "ocfps": "ocfps",
            "netprofit_yoy": "netprofit_yoy",
            "tr_yoy": "tr_yoy",
        }
    
    def _print_data_specs(self):
        """打印数据规范信息"""
        print("\n" + "=" * 60)
        print("数据规范配置：")
        print(f"  - 价格复权类型：{'前复权 (qfq)' if FORCE_ADJUSTED_PRICES else '不复权'}")
        print(f"  - 财报日期类型：{'公告日期 (ann_date)' if FINANCIAL_USE_ANNOUNCEMENT_DATE else '期末日期 (end_date)'}")
        print("=" * 60)
    
    def _ensure_dirs(self):
        """确保目录存在"""
        for d in [self.features_dir, self.instruments_dir, self.calendars_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _to_float(self, v):
        """统一转 float，兼容 None/Decimal"""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float('nan')
        if isinstance(v, (pd.Timestamp, datetime)):
            return float('nan')
        return float(v)
    
    def _write_bin(self, filepath: Path, start_index: int, values_array: np.ndarray):
        """
        写 Qlib .bin 文件
        
        Qlib 规范：第一个元素为 start_index (int32), 后续为 float32 数据
        """
        data = np.hstack([np.array([start_index], dtype='<i4'), values_array.astype('<f4')])
        data.tofile(str(filepath))
    
    def _get_calendar_list(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[List[str], Dict[str, int]]:
        """
        获取交易日历列表（过滤到指定日期范围）

        Args:
            start_date: 开始日期 YYYY-MM-DD（可选，不提供则取全部）
            end_date: 结束日期 YYYY-MM-DD（可选，不提供则取全部）

        Returns:
            (calendar_list, calendar_map)
        """
        cal_df = self.api.get_calendar(start_date=start_date, end_date=end_date)
        calendar_list = [str(d)[:10] for d in cal_df["trade_date"]]
        calendar_map = {d: i for i, d in enumerate(calendar_list)}
        return calendar_list, calendar_map
    
    def _save_calendars(self, calendar_list: List[str]):
        """保存交易日历"""
        cal_file = self.calendars_dir / "day.txt"
        with open(cal_file, 'w') as f:
            for d in calendar_list:
                f.write(d + "\n")
        print(f"    已保存 {len(calendar_list)} 个交易日")
    
    def _save_instruments(self, stocks: List[str]):
        """生成 instruments 文件（使用实际上市日期）"""
        sh = [s for s in stocks if s.endswith('.SH')]
        sz = [s for s in stocks if s.endswith('.SZ')]
        
        # 获取股票实际上市日期
        df_listed = self.api.get_stock_list()
        listed_dict = df_listed.set_index('ts_code')['list_date'].to_dict()
        
        for slist, fname in [(sh + sz, "all.txt"), (sh, "all_sh.txt"), (sz, "all_sz.txt")]:
            file_path = self.instruments_dir / fname
            with open(file_path, 'w') as f:
                for s in slist:
                    list_date = listed_dict.get(s, '2010-01-01')
                    # 确保日期格式正确（兼容 YYYYMMDD 和 datetime 格式）
                    date_str = str(list_date)
                    if len(date_str) == 8:  # YYYYMMDD 格式
                        list_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    elif ' ' in date_str:  # datetime 格式，如 "1999-11-10 00:00:00"
                        list_date = date_str.split(' ')[0]
                    f.write(f"{s}\t{list_date}\t9999-99-99\n")
            print(f"    {fname}: {len(slist)} 只")
    
    def _get_main_board_stocks(self) -> List[str]:
        """从 ClickHouse 获取沪深主板股票列表"""
        df = self.api.get_stock_list(market='主板', status='L')
        stocks = df["ts_code"].tolist()
        print(f"    共找到 {len(stocks)} 只主板股票")
        return stocks
    
    def full_sync(self, start_date: str, end_date: str, instruments: Optional[List[str]] = None,
                  instruments_dict: Optional[Dict[str, str]] = None):
        """
        全量同步 Qlib 数据

        Args:
            start_date: 开始日期 YYYY-MM-DD（当 instruments_dict 未提供时使用）
            end_date: 结束日期 YYYY-MM-DD
            instruments: 股票列表，None 表示全部主板股票
            instruments_dict: 每只股票的自定义开始日期字典 {stock_code: start_date}，
                             提供后覆盖统一的 start_date（按上市日期定制同步起点）
        """
        print("=" * 60)
        print(f"Qlib 全量同步：{start_date} - {end_date}")
        if instruments_dict:
            print(f"  使用按股票定制的开始日期 ({len(instruments_dict)} 只)")
        print("=" * 60)

        self._ensure_dirs()

        # 1. 获取股票列表
        print("\n[1] 获取股票列表...")
        if instruments is None:
            stocks = self._get_main_board_stocks()
        else:
            stocks = instruments

        if not stocks:
            print("    未获取到股票列表，退出")
            return

        # 2. 获取交易日历
        print("\n[2] 获取交易日历...")
        calendar_list, calendar_map = self._get_calendar_list(start_date=start_date, end_date=end_date)
        self._save_calendars(calendar_list)

        # 3. 保存 instruments 文件
        print("\n[3] 生成 instruments 文件...")
        self._save_instruments(stocks)

        # 4. 确保 features 目录为空
        print("\n[4] 准备 features 目录...")
        if self.features_dir.exists():
            try:
                shutil.rmtree(self.features_dir)
            except Exception as e:
                print(f"    清理目录失败（可能已不存在）: {e}")
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # 5. 下载并保存数据
        print("\n[5] 下载并保存数据...")
        self._sync_features(stocks, calendar_list, calendar_map, start_date, end_date, instruments_dict=instruments_dict)

        print("\n" + "=" * 60)
        print("Qlib 全量同步完成！")
        print("=" * 60)
    
    def incremental_sync(self):
        """
        增量同步 Qlib 数据
        
        只同步 Qlib 中缺失的最新数据
        """
        print("=" * 60)
        print("Qlib 增量同步")
        print("=" * 60)
        
        try:
            import qlib
            qlib.init(provider_uri=str(self.qlib_dir))
            from qlib.data import D
            
            # 获取 Qlib 最新日期
            qlib_cal = D.calendar()
            if len(qlib_cal) == 0:
                print("    Qlib 数据为空，请执行全量同步")
                return
            
            qlib_latest = qlib_cal[-1].strftime("%Y-%m-%d")
            
            # 获取 ClickHouse 最新日期
            ch_latest_df = self.api.query("SELECT MAX(trade_date) as latest FROM daily_prices")
            ch_latest = str(ch_latest_df["latest"].iloc[0])[:10]
            
            if qlib_latest >= ch_latest:
                print(f"    Qlib 数据已是最新 (Qlib: {qlib_latest}, CH: {ch_latest})")
                return
            
            print(f"    Qlib 最新：{qlib_latest}, ClickHouse 最新：{ch_latest}")
            print(f"    将同步：{qlib_latest} - {ch_latest}")
            
            # 获取股票列表
            stocks = self._get_main_board_stocks()
            calendar_list, calendar_map = self._get_calendar_list()
            
            # 只同步新数据
            self._sync_features(stocks, calendar_list, calendar_map, qlib_latest, ch_latest, append=True)
            
            print("\n" + "=" * 60)
            print("Qlib 增量同步完成！")
            print("=" * 60)
            
        except Exception as e:
            print(f"增量同步失败：{e}")
            raise
    
    def _sync_features(
        self,
        stocks: List[str],
        calendar_list: List[str],
        calendar_map: Dict[str, int],
        start_date: str,
        end_date: str,
        append: bool = False,
        instruments_dict: Optional[Dict[str, str]] = None
    ):
        """
        同步特征数据

        Args:
            stocks: 股票列表
            calendar_list: 日历列表
            calendar_map: 日历映射
            start_date: 开始日期（instruments_dict 未覆盖时的默认值）
            end_date: 结束日期
            append: 是否追加模式
            instruments_dict: 每只股票的自定义开始日期 {stock_code: start_date}
        """
        success_count = 0
        failed_count = 0

        # [Goldman Sachs 架构师] 强制从本地 day.txt 读取日历，确保 100% 对齐
        local_cal_path = self.calendars_dir / "day.txt"
        if local_cal_path.exists():
            with open(local_cal_path, 'r') as f:
                local_cal = [d.strip() for d in f.read().splitlines() if d.strip()]
            if len(local_cal) != len(calendar_list):
                print(f"  [INFO] calendar_map ({len(calendar_list)}) 与 day.txt ({len(local_cal)}) 不一致，" + 
                      f"强制从 day.txt 重建日历映射")
            else:
                print(f"  [INFO] 使用 day.txt 日历映射: {len(local_cal)} 天")
            # 始终从 day.txt 读取，确保与已写入的日历年完美对齐
            calendar_list = local_cal
            calendar_map = {d: i for i, d in enumerate(calendar_list)}

        for stock_code in tqdm(stocks, desc="同步股票数据"):
            try:
                # [Bloomberg Eng] 按股票定制开始日期
                stock_start = instruments_dict.get(stock_code, start_date) if instruments_dict else start_date

                # 获取日线数据（前复权）
                df = self.api.get_daily_data(
                    ts_codes=[stock_code],
                    start_date=stock_start,
                    end_date=end_date,
                    adj=True
                )

                if df.empty:
                    failed_count += 1
                    continue

                # 转换列名（兼容大小写）
                df.columns = df.columns.str.lower()
                df = df.rename(columns={
                    "ts_code": "symbol",
                    "trade_date": "date",
                })
                if "date" not in df.columns:
                    print(f"    同步 {stock_code} 失败：缺少 date 字段")
                    failed_count += 1
                    continue
                df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')

                # 获取财务数据（使用公告日期，向后填充避免未来数据泄露）
                financial_fields = ['roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
                                   'debt_to_assets', 'current_ratio', 'eps', 'ocfps',
                                   'netprofit_yoy', 'tr_yoy']
                df_financial = self.api.get_financial_data(
                    ts_codes=[stock_code],
                    start_date=stock_start,
                    end_date=end_date,
                    fields=financial_fields
                )
                
                if not df_financial.empty:
                    df_financial.columns = df_financial.columns.str.lower()
                    df_financial['date'] = pd.to_datetime(df_financial['ann_date']).dt.strftime('%Y-%m-%d')
                    # 按日期排序并向后填充
                    df_financial = df_financial.sort_values('date').set_index('date')[financial_fields]
                    # 合并到主数据，使用向后填充
                    df = df.set_index('date')
                    df = df.join(df_financial, how='left').ffill().reset_index()
                else:
                    df = df.set_index('date').reset_index()
                
                # 保存数据
                code_short = stock_code.lower()
                stock_dir = self.features_dir / code_short
                stock_dir.mkdir(parents=True, exist_ok=True)
                
                # 写入每个字段
                for qlib_field, ch_field in self.field_mapping.items():
                    if ch_field not in df.columns:
                        continue
                    
                    series = df.set_index("date")[ch_field]
                    valid_pairs = []
                    
                    for date_str, value in series.items():
                        if date_str in calendar_map and pd.notna(value):
                            valid_pairs.append((calendar_map[date_str], self._to_float(value)))
                    
                    if valid_pairs:
                        valid_pairs.sort(key=lambda x: x[0])
                        start_index = valid_pairs[0][0]
                        end_index = valid_pairs[-1][0]
                        
                        length = end_index - start_index + 1
                        values_array = np.full(length, np.nan, dtype=np.float32)
                        
                        for idx, val in valid_pairs:
                            values_array[idx - start_index] = val
                        
                        bin_path = stock_dir / f"{qlib_field}.day.bin"
                        self._write_bin(bin_path, start_index, values_array)
                
                success_count += 1
                
            except Exception as e:
                print(f"\n    同步 {stock_code} 失败：{e}")
                failed_count += 1
        
        print(f"\n    成功：{success_count} 只 | 失败：{failed_count} 只")
