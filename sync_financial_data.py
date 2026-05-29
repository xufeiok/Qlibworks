"""
同步财报因子数据到 Qlib features 文件夹

包含：
1. 财报因子数据（使用公告日期）
2. 申万一级行业分类
3. 市值指标（total_mv, circ_mv）
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

from qlworks.data.api import QuantDataAPI
from qlworks.config import QLIB_DATA_DIR, FORCE_ADJUSTED_PRICES, FINANCIAL_USE_ANNOUNCEMENT_DATE


class FinancialDataSynchronizer:
    """
    财报因子数据同步器
    
    将 ClickHouse 中的财报数据同步为 Qlib 格式：
    - features/<stock>/<field>.day.bin: 特征数据
    """
    
    def __init__(self, api):
        self.api = api
        self.qlib_dir = Path(QLIB_DATA_DIR)
        self.features_dir = self.qlib_dir / "features"
        self.instruments_dir = self.qlib_dir / "instruments"
        self.calendars_dir = self.qlib_dir / "calendars"
        
        # 申万一级行业映射（从二级行业映射到一级）
        self.sw_industry_map = {
            # 金融
            '银行': '金融',
            '证券': '金融',
            '保险': '金融',
            '多元金融': '金融',
            '信托': '金融',
            # 地产
            '全国地产': '地产',
            '区域地产': '地产',
            '房产服务': '地产',
            # 消费
            '白酒': '消费',
            '啤酒': '消费',
            '葡萄酒': '消费',
            '食品综合': '消费',
            '食品加工': '消费',
            '肉制品': '消费',
            '乳业': '消费',
            '调味发酵品': '消费',
            '休闲食品': '消费',
            '保健品': '消费',
            '纺织制造': '消费',
            '服装家纺': '消费',
            '家居用品': '消费',
            '小家电': '消费',
            '厨卫电器': '消费',
            '白色家电': '消费',
            '汽车零部件': '消费',
            '乘用车': '消费',
            '汽车服务': '消费',
            '其他商业': '消费',
            '一般零售': '消费',
            '专业零售': '消费',
            '互联网电商': '消费',
            # 医药
            '化学制药': '医药',
            '中药': '医药',
            '生物制品': '医药',
            '医疗器械': '医药',
            '医疗服务': '医药',
            '医美': '医药',
            # 科技
            '半导体': '科技',
            '元件': '科技',
            '集成电路': '科技',
            '软件服务': '科技',
            '互联网服务': '科技',
            '通信服务': '科技',
            '计算机设备': '科技',
            '电子制造': '科技',
            '消费电子': '科技',
            '光学光电子': '科技',
            '移动互联网': '科技',
            # 周期
            '煤炭开采': '周期',
            '钢铁': '周期',
            '有色金属': '周期',
            '化工原料': '周期',
            '化工合成材料': '周期',
            '塑料': '周期',
            '橡胶': '周期',
            '造纸': '周期',
            '玻璃': '周期',
            '水泥': '周期',
            '建材': '周期',
            '建筑装饰': '周期',
            '建筑工程': '周期',
            '工程机械': '周期',
            # 公用事业
            '电力': '公用事业',
            '水务': '公用事业',
            '燃气': '公用事业',
            '环保工程': '公用事业',
            # 交通运输
            '航空运输': '交通运输',
            '铁路运输': '交通运输',
            '公路运输': '交通运输',
            '港口': '交通运输',
            '航运': '交通运输',
            '物流': '交通运输',
            '运输设备': '交通运输',
            # 能源
            '油气开采': '能源',
            '油品贸易': '能源',
            '电力设备': '能源',
            '电气设备': '能源',
            '光伏设备': '能源',
            '风电设备': '能源',
            '电池': '能源',
            # 其他
            '综合': '其他',
            '综合服务': '其他',
            '文化传媒': '其他',
            '教育': '其他',
            '酒店餐饮': '其他',
            '旅游及景区': '其他',
        }
        
        # 字段映射（Qlib 字段名 -> ClickHouse 字段名）
        self.field_mapping = {
            # 基础行情
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "vol",
            "amount": "amount",
            # 市值指标
            "total_mv": "total_mv",
            "circ_mv": "circ_mv",
            # 估值指标
            "pe": "pe",
            "pe_ttm": "pe_ttm",
            "pb": "pb",
            "ps": "ps",
            "ps_ttm": "ps_ttm",
            "dv_ttm": "dv_ttm",
            # 财务因子（从 financial_indicators 表）
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
            "dt_netprofit_yoy": "dt_netprofit_yoy",
            "basic_eps_yoy": "basic_eps_yoy",
            "stk_holdernumber": "stk_holdernumber",
            "pledge_ratio": "pledge_ratio",
            # 申万一级行业（数值编码）
            "sw_industry": "sw_industry_code",
        }
        
        # 行业名称到数值的映射
        self.industry_code_map = {
            '金融': 1,
            '地产': 2,
            '消费': 3,
            '医药': 4,
            '科技': 5,
            '周期': 6,
            '公用事业': 7,
            '交通运输': 8,
            '能源': 9,
            '其他': 10,
        }
    
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
    
    def _adjust_per_share_data(self, df_financial: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        """
        对每股指标进行追溯调整（考虑拆股、配股等股本变动）
        
        原理：使用复权因子计算调整系数，将历史每股数据调整到当前股本基础上
        调整系数 = 最新复权因子 / 历史复权因子
        
        Args:
            df_financial: 财务数据 DataFrame，必须包含 ann_date 列
            ts_code: 股票代码
        
        Returns:
            调整后的财务数据
        """
        # 需要追溯调整的每股指标（扩展列表）
        per_share_fields = [
            'eps', 'ocfps', 'eps_forecast',
            'revenue_per_share', 'profit_per_share',
            'cash_flow_per_share', 'dividend_per_share'
        ]
        
        try:
            # 获取复权因子数据（需要覆盖财报日期范围之前的数据）
            # 使用一个更早的日期来获取完整的复权因子历史
            adj_df = self.api.query(
                f"SELECT trade_date, adj_factor FROM daily_adj_factors WHERE ts_code = '{ts_code}' ORDER BY trade_date"
            )
            
            if adj_df.empty:
                return df_financial
            
            # 获取最新复权因子（转换为float）
            latest_adj_factor = float(adj_df['adj_factor'].iloc[-1])
            
            # 将财报数据的 ann_date 与复权因子的 trade_date 对齐
            # 找到每个财报公告日期之前最近的复权因子
            df_financial['ann_date_dt'] = pd.to_datetime(df_financial['ann_date'])
            adj_df['trade_date_dt'] = pd.to_datetime(adj_df['trade_date'])
            
            # 将复权因子转换为float
            adj_df['adj_factor'] = adj_df['adj_factor'].astype(float)
            
            # 创建合并键
            df_financial = df_financial.sort_values('ann_date_dt')
            adj_df = adj_df.sort_values('trade_date_dt')
            
            # 使用 merge_asof 找到每个财报日期对应的复权因子
            merged = pd.merge_asof(
                df_financial,
                adj_df[['trade_date_dt', 'adj_factor']],
                left_on='ann_date_dt',
                right_on='trade_date_dt',
                direction='backward'  # 使用公告日期之前最近的复权因子
            )
            
            # 计算调整系数并调整每股指标
            if 'adj_factor' in merged.columns and latest_adj_factor > 0:
                merged['adjust_ratio'] = latest_adj_factor / merged['adj_factor']
                
                for field in per_share_fields:
                    if field in merged.columns:
                        # 只对非空值进行调整
                        merged[field] = merged[field].astype(float)
                        mask = pd.notna(merged[field]) & pd.notna(merged['adjust_ratio'])
                        merged.loc[mask, field] = merged.loc[mask, field] * merged.loc[mask, 'adjust_ratio']
            
            # 移除临时列
            merged = merged.drop(columns=['ann_date_dt', 'trade_date_dt', 'adj_factor', 'adjust_ratio'], errors='ignore')
            
            return merged
            
        except Exception as e:
            # 如果获取复权因子失败，返回原始数据
            print(f"    警告：股票 {ts_code} 追溯调整失败：{e}，使用原始数据")
            return df_financial
    
    def _write_bin(self, filepath: Path, start_index: int, values_array: np.ndarray):
        """写 Qlib .bin 文件"""
        data = np.hstack([np.array([start_index], dtype='<i4'), values_array.astype('<f4')])
        data.tofile(str(filepath))
    
    def _get_calendar_list(self) -> Tuple[List[str], Dict[str, int]]:
        """获取交易日历列表和映射"""
        df = self.api.get_calendar()
        calendar_list = [str(d)[:10] for d in df['trade_date'].tolist()]
        calendar_map = {date_str: idx for idx, date_str in enumerate(calendar_list)}
        return calendar_list, calendar_map
    
    def _save_calendars(self, calendar_list: List[str]):
        """保存交易日历"""
        calendar_path = self.calendars_dir / "day.txt"
        with open(calendar_path, 'w') as f:
            for d in calendar_list:
                f.write(d + "\n")
        print(f"    已保存 {len(calendar_list)} 个交易日")
    
    def _save_instruments(self, stocks: List[str]):
        """生成 instruments 文件"""
        sh = [s for s in stocks if s.endswith('.SH')]
        sz = [s for s in stocks if s.endswith('.SZ')]
        
        df_listed = self.api.get_stock_list()
        listed_dict = df_listed.set_index('ts_code')['list_date'].to_dict()
        
        for slist, fname in [(sh + sz, "all.txt"), (sh, "all_sh.txt"), (sz, "all_sz.txt")]:
            file_path = self.instruments_dir / fname
            with open(file_path, 'w') as f:
                for s in slist:
                    list_date = listed_dict.get(s, '2010-01-01')
                    date_str = str(list_date)
                    if len(date_str) == 8:
                        list_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    elif ' ' in date_str:
                        list_date = date_str.split(' ')[0]
                    f.write(f"{s}\t{list_date}\t9999-99-99\n")
            print(f"    {fname}: {len(slist)} 只")
    
    def sync(self, start_date: str, end_date: str, instruments: Optional[List[str]] = None):
        """
        同步财报因子数据到 Qlib features
        
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            instruments: 股票列表，None 表示全部主板股票
        """
        print("=" * 60)
        print(f"财报因子数据同步：{start_date} - {end_date}")
        print(f"数据规范：公告日期 = {FINANCIAL_USE_ANNOUNCEMENT_DATE}")
        print("=" * 60)
        
        self._ensure_dirs()
        
        # 1. 获取股票列表
        print("\n[1] 获取股票列表...")
        if instruments is None:
            df_stocks = self.api.get_stock_list(status='L')
            stocks = df_stocks["ts_code"].tolist()
        else:
            stocks = instruments
        
        if not stocks:
            print("    未获取到股票列表，退出")
            return
        print(f"    共 {len(stocks)} 只股票")
        
        # 2. 获取交易日历
        print("\n[2] 获取交易日历...")
        calendar_list, calendar_map = self._get_calendar_list()
        self._save_calendars(calendar_list)
        
        # 3. 保存 instruments 文件
        print("\n[3] 生成 instruments 文件...")
        self._save_instruments(stocks)
        
        # 4. 获取行业映射
        print("\n[4] 获取行业分类...")
        df_stocks = self.api.get_stock_list()
        industry_dict = df_stocks.set_index('ts_code')['industry'].to_dict()
        
        # 5. 同步特征数据
        print("\n[5] 同步特征数据...")
        self._sync_features(stocks, calendar_list, calendar_map, start_date, end_date, industry_dict)
        
        print("\n" + "=" * 60)
        print("财报因子数据同步完成！")
        print("=" * 60)
    
    def _sync_features(
        self,
        stocks: List[str],
        calendar_list: List[str],
        calendar_map: Dict[str, int],
        start_date: str,
        end_date: str,
        industry_dict: Dict[str, str]
    ):
        """同步特征数据"""
        success_count = 0
        failed_count = 0
        
        for stock_code in tqdm(stocks, desc="同步股票数据"):
            try:
                # 获取日线数据（包含市值指标）
                df = self.api.get_daily_data(
                    ts_codes=[stock_code],
                    start_date=start_date,
                    end_date=end_date,
                    adj=True
                )
                
                if df.empty:
                    failed_count += 1
                    continue
                
                # 转换列名
                df.columns = df.columns.str.lower()
                df = df.rename(columns={
                    "ts_code": "symbol",
                    "trade_date": "date",
                })
                
                df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
                
                # 获取财务数据（使用公告日期）
                financial_fields = [
                    'roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
                    'debt_to_assets', 'current_ratio', 'eps', 'ocfps',
                    'netprofit_yoy', 'tr_yoy', 'dt_netprofit_yoy', 
                    'basic_eps_yoy', 'stk_holdernumber', 'pledge_ratio'
                ]
                df_financial = self.api.get_financial_data(
                    ts_codes=[stock_code],
                    start_date=start_date,
                    end_date=end_date,
                    fields=financial_fields
                )
                
                if not df_financial.empty:
                    df_financial.columns = df_financial.columns.str.lower()
                    # 确保 ann_date 存在
                    if 'ann_date' in df_financial.columns:
                        # 对每股指标进行追溯调整（考虑拆股、配股等股本变动）
                        df_financial = self._adjust_per_share_data(df_financial, stock_code)
                        
                        df_financial['date'] = pd.to_datetime(df_financial['ann_date']).dt.strftime('%Y-%m-%d')
                        df_financial = df_financial.sort_values('date').set_index('date')[financial_fields]
                        df = df.set_index('date')
                        df = df.join(df_financial, how='left').ffill().reset_index()
                    else:
                        # 如果没有 ann_date，跳过财务数据合并
                        df = df.set_index('date').reset_index()
                else:
                    df = df.set_index('date').reset_index()
                
                # 添加申万一级行业编码
                industry = industry_dict.get(stock_code, '其他')
                sw_industry = self.sw_industry_map.get(industry, '其他')
                sw_industry_code = self.industry_code_map.get(sw_industry, 10)
                df['sw_industry_code'] = sw_industry_code
                
                # 保存数据
                code_short = stock_code.lower()
                stock_dir = self.features_dir / code_short
                stock_dir.mkdir(parents=True, exist_ok=True)
                
                # 写入每个字段
                df_indexed = df.set_index("date")
                for qlib_field, ch_field in self.field_mapping.items():
                    if ch_field not in df_indexed.columns:
                        continue
                    
                    series = df_indexed[ch_field]
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


if __name__ == "__main__":
    api = QuantDataAPI()
    syncer = FinancialDataSynchronizer(api)
    
    # 同步最近5年的数据
    syncer.sync(start_date='2020-01-01', end_date='2024-12-31')