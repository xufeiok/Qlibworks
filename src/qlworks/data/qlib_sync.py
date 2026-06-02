"""
Qlib 数据同步模块

核心公开方法（共 3 个，职责分明）：
  1. full_sync()         — 全量首次下载：写入 OHLCV + 市值 + 申万行业到 Qlib bin 格式
  2. incremental_sync()  — 增量更新已有 Qlib 数据的最新交易日
  3. sync_fields()       — 通用方法：指定 SQL 查询下载到 Qlib bin 格式

其他指标（财务/估值/动量/自定义因子）已迁移为 DuckDB + Parquet 预计算，
参见 qlworks.features.factor_cache.FactorCache。
"""
from __future__ import annotations

import os
import shutil
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from qlworks.config import QLIB_DATA_DIR, FORCE_ADJUSTED_PRICES


class QlibSynchronizer:
    """
    Qlib 数据同步器

    写入到 Qlib bin 格式的内核字段（仅 8 个）：
      - OHCLV: open, high, low, close, volume, amount
      - 市值:  total_mv, circ_mv
      - 行业:  sw_l1, sw_l2, sw_l3（通过 sync_industry）

    其余因子全部通过 FactorCache（DuckDB + Parquet）预计算。
    """

    def __init__(self, api):
        self.api = api
        self.qlib_dir = Path(QLIB_DATA_DIR)
        self.features_dir = self.qlib_dir / "features"
        self.instruments_dir = self.qlib_dir / "instruments"
        self.calendars_dir = self.qlib_dir / "calendars"

        self._print_data_specs()

        # Qlib 内核字段映射 —— 只写 OHLCV + 市值
        # get_daily_data(adj=True) 已 JOIN daily_indicators 表，
        # 因此 total_mv / circ_mv 天然在返回结果中。
        self.field_mapping = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "vol",
            "amount": "amount",
            "total_mv": "total_mv",
            "circ_mv": "circ_mv",
        }

        # 行业代码名称→数值ID映射（用于 Qlib $sw_l1/$sw_l2/$sw_l3 特性）
        self._industry_id_map: Dict[str, Dict[str, int]] = {}
    
    def _print_data_specs(self):
        print("\n" + "=" * 60)
        print("数据规范配置：")
        print(f"  - 价格复权类型：{'前复权 (qfq)' if FORCE_ADJUSTED_PRICES else '不复权'}")
        print("=" * 60)
    
    def _ensure_dirs(self):
        """确保目录存在"""
        for d in [self.features_dir, self.instruments_dir, self.calendars_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _build_industry_mapping(self):
        """
        从 ClickHouse sw_industry_members 表构建申万行业映射并保存为 JSON。

        映射格式：（与 Qlib $sw_l1/$sw_l2/$sw_l3 兼容）
        {
            "l1": {"行业名称": 数值ID, ...},  # 一级行业（31个）
            "l2": {"行业名称": 数值ID, ...},  # 二级行业（100+）
            "l3": {"行业名称": 数值ID, ...},  # 三级行业（300+）
        }
        """
        try:
            df = self.api.query("""
                SELECT DISTINCT l1_code, l1_name, l2_code, l2_name, l3_code, l3_name
                FROM sw_industry_members
            """)
            if df.empty:
                print("    [WARN] 未获取到行业数据，跳过行业映射")
                return

            mapping = {"l1": {}, "l2": {}, "l3": {}}
            # 按一/二/三级行业名称分配唯一数值ID（按名称排序，保证稳定性）
            for level, name_col, code_col in [
                ("l1", "l1_name", "l1_code"),
                ("l2", "l2_name", "l2_code"),
                ("l3", "l3_name", "l3_code"),
            ]:
                unique_names = sorted(df[name_col].dropna().unique())
                # 排除空字符串
                unique_names = [n for n in unique_names if str(n).strip()]
                mapping[level] = {n: i + 1 for i, n in enumerate(unique_names)}

            # 保存到 qlib_data 目录
            out_path = self.qlib_dir / "sw_industry_mapping.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            print(f"    申万行业映射已保存: {out_path}")
            for level in ["l1", "l2", "l3"]:
                print(f"      {level}: {len(mapping[level])} 个行业")

            self._industry_id_map = mapping

        except Exception as e:
            print(f"    [WARN] 构建行业映射失败: {e}")
            # 如果已存在映射文件，尝试加载
            existing = self.qlib_dir / "sw_industry_mapping.json"
            if existing.exists():
                try:
                    with open(existing, 'r', encoding='utf-8') as f:
                        self._industry_id_map = json.load(f)
                    print(f"    使用已有的行业映射文件: {existing}")
                except Exception:
                    pass
    
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
        注意：使用字节拼接而非 np.hstack，避免 int32+float32 被 numpy 升级为 float64
        """
        header = np.array([start_index], dtype='<i4').tobytes()
        data = values_array.astype('<f4').tobytes()
        with open(str(filepath), 'wb') as f:
            f.write(header + data)
    
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

        # 0. 构建并保存申万行业映射
        print("\n[0] 构建申万行业映射...")
        self._build_industry_mapping()

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

        # 6. 补充申万行业数据（独立方法，避免与主同步的日历逻辑冲突）
        print("\n[6] 补充申万行业数据...")
        self.sync_industry(stocks=stocks, verify=True)

        print("\n" + "=" * 60)
        print("Qlib 全量同步完成！")
        print("=" * 60)
    
    def incremental_sync(self, instruments_dict=None):
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
            self._sync_features(stocks, calendar_list, calendar_map, qlib_latest, ch_latest, append=True, instruments_dict=instruments_dict)
            
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


        BATCH_SIZE = 50
        for i in tqdm(range(0, len(stocks), BATCH_SIZE), desc="同步股票数据"):
            batch = stocks[i:i + BATCH_SIZE]
            try:
                batch_start = start_date
                if instruments_dict:
                    batch_starts = [instruments_dict.get(c, start_date) for c in batch]
                    batch_start = min(batch_starts)

                df = self.api.get_daily_data(
                    ts_codes=batch,
                    start_date=batch_start,
                    end_date=end_date,
                    adj=True
                )

                if df.empty:
                    failed_count += len(batch)
                    continue

                df.columns = df.columns.str.lower()
                df = df.rename(columns={
                    "ts_code": "symbol",
                    "trade_date": "date",
                })
                if "date" not in df.columns:
                    print(f"    批次 {i}-{i+len(batch)} 失败：缺少 date 字段")
                    failed_count += len(batch)
                    continue
                df["date"] = df["date"].apply(lambda x: str(x)[:10])

                for stock_code, grp in df.groupby("symbol"):
                    try:
                        stock_start = instruments_dict.get(stock_code, start_date) if instruments_dict else start_date
                        grp = grp[grp["date"] >= stock_start].copy()
                        if grp.empty:
                            failed_count += 1
                            continue

                        before_dedup = len(grp)
                        grp = grp.drop_duplicates(subset=['symbol', 'date'], keep='last').reset_index(drop=True)
                        deduped = before_dedup - len(grp)
                        if deduped > 0:
                            tqdm.write(f"    {stock_code}: 去重 {deduped} 行")

                        code_short = stock_code.lower()
                        stock_dir = self.features_dir / code_short
                        stock_dir.mkdir(parents=True, exist_ok=True)

                        for qlib_field, ch_field in self.field_mapping.items():
                            if ch_field not in grp.columns:
                                continue
                            series = grp.set_index("date")[ch_field]
                            _valid = [(calendar_map[d], self._to_float(v)) for d, v in series.items()
                                      if d in calendar_map and pd.notna(v)]
                            if _valid:
                                _valid.sort(key=lambda x: x[0])
                                _arr = np.full(_valid[-1][0] - _valid[0][0] + 1, np.nan, dtype=np.float32)
                                for idx, val in _valid:
                                    _arr[idx - _valid[0][0]] = val
                                self._write_bin(stock_dir / f"{qlib_field}.day.bin", _valid[0][0], _arr)

                        success_count += 1
                    except Exception as e:
                        print(f"\n    同步 {stock_code} 失败：{e}")
                        failed_count += 1

            except Exception as e:
                print(f"\n    批次 {i}-{i+len(batch)} 失败：{e}")
                failed_count += len(batch)

        print(f"\n    成功：{success_count} 只 | 失败：{failed_count} 只")

    def sync_fields(
        self,
        field_spec: dict,
        stocks: Optional[List[str]] = None,
        verify: bool = True,
    ):
        """
        通用字段同步方法：将指定指标数据下载到 Qlib features 目录。

        支持 data_type="time_series"（时间序列型）和 data_type="time_range"（时间区间型）。

        Args:
            field_spec: 字段规格字典
            stocks: 股票列表，None 表示全部
            verify: 是否抽样验证
        """
        label = field_spec.get("label", "未知字段")
        bin_pattern = field_spec.get("bin_pattern", "*.day.bin")
        bins = field_spec["bins"]
        data_type = field_spec["data_type"]
        value_cols = field_spec["value_cols"]
        parse_val = field_spec.get("parse_value", lambda v: float(v) if v is not None else float("nan"))
        verify_col = field_spec.get("verify_col", bins[0])
        verify_min = field_spec.get("verify_min", None)
        n_bins = len(bins)

        print("=" * 60)
        print(f"同步字段: {label}")
        print("=" * 60)

        # 1. 清理旧文件
        existing = 0; deleted = 0
        for d in self.features_dir.iterdir():
            if not d.is_dir(): continue
            old = list(d.glob(bin_pattern))
            if old:
                existing += 1
                for b in old: b.unlink(); deleted += 1
        print(f"  清理: {existing} 只股票, {deleted} 个文件")

        # 2. 股票列表
        if stocks is None:
            stocks = sorted([d.name for d in self.features_dir.iterdir() if d.is_dir()])
        print(f"  股票: {len(stocks)} 只")

        # 3. 日历
        with open(self.calendars_dir / "day.txt") as f:
            cal_list = [l.strip() for l in f if l.strip()]
        print(f"  日历: {cal_list[0]} ~ {cal_list[-1]} ({len(cal_list)} 天)")

        # 4. 查询
        print(f"  查询 ClickHouse...")
        try:
            df_all = self.api.query(field_spec["query"])
        except Exception as e:
            print(f"  [ERR] 查询失败: {e}"); return
        if df_all.empty:
            print(f"  [ERR] 空结果"); return
        print(f"  记录: {len(df_all)} 条")

        # 5. 按股票分组写入
        code_col = field_spec.get("code_col", "ts_code")
        grouped = df_all.groupby(code_col)
        success = 0; has_data = 0

        for stock in tqdm(stocks, desc=f"写入{label}"):
            try:
                code_up = stock.upper()
                if code_up not in grouped.groups: continue
                has_data += 1
                sub = grouped.get_group(code_up)

                if data_type == "time_series":
                    date_col = field_spec["date_col"]
                    sub = sub.copy()
                    sub["_dt"] = pd.to_datetime(sub[date_col]).dt.strftime("%Y-%m-%d")
                    val_map = {}
                    for _, row in sub.iterrows():
                        d = row["_dt"]
                        if d not in val_map: val_map[d] = {}
                        for bn in bins:
                            raw = row.get(value_cols[bn])
                            if raw is not None and pd.notna(raw):
                                val_map[d][bn] = parse_val(raw)
                    arrs = {bn: np.full(len(cal_list), float("nan"), dtype=np.float32) for bn in bins}
                    for i, cal_d in enumerate(cal_list):
                        if cal_d in val_map:
                            for bn in bins:
                                v = val_map[cal_d].get(bn)
                                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                    arrs[bn][i] = v

                elif data_type == "time_range":
                    in_col = field_spec["in_date_col"]
                    out_col = field_spec["out_date_col"]
                    arrs = {bn: np.full(len(cal_list), float("nan"), dtype=np.float32) for bn in bins}
                    for _, row in sub.iterrows():
                        in_d = str(row[in_col])[:10] if pd.notna(row.get(in_col)) else "1970-01-01"
                        out_d = str(row[out_col])[:10] if pd.notna(row.get(out_col)) else "2100-12-31"
                        if out_d == "1970-01-01": out_d = "2100-12-31"
                        try: si = next(i for i, d in enumerate(cal_list) if d >= in_d)
                        except StopIteration: si = 0
                        try: ei = next(i for i, d in enumerate(cal_list) if d > out_d)
                        except StopIteration: ei = len(cal_list)
                        for bn in bins:
                            raw = row.get(value_cols[bn])
                            if raw is not None and pd.notna(raw):
                                v = parse_val(raw)
                                if not (isinstance(v, float) and np.isnan(v)):
                                    arrs[bn][si:ei] = v
                else:
                    print(f"  [ERR] 不支持类型: {data_type}"); return

                sd = self.features_dir / stock.lower()
                sd.mkdir(parents=True, exist_ok=True)
                for bn in bins: self._write_bin(sd / f"{bn}.day.bin", 0, arrs[bn])
                success += 1
            except Exception as e:
                tqdm.write(f"  [WARN] {stock}: {e}")

        print(f"写入: {success} / {len(stocks)} 只")

        # 6. 验证
        if verify and success > 0:
            print(f"  验证 (min(100, {success}) 只)...")
            ok = bad = 0
            vs = [s.lower() for s in stocks[:100] if s.upper() in grouped.groups]
            for cl in vs:
                fp = self.features_dir / cl / f"{verify_col}.day.bin"
                if not fp.exists(): continue
                try:
                    data = np.fromfile(fp, dtype=np.float32)[1:]
                    valid = data[~np.isnan(data)]
                    if len(valid) == 0: bad += 1; continue
                    if verify_min is None or valid.min() >= verify_min: ok += 1
                    else: bad += 1; tqdm.write(f"    [WARN] {cl}: 异常 {valid[:5]}")
                except: bad += 1
            print(f"  {ok} 正确" + (f" [WARN] {bad} 异常" if bad else ""))

        # 7. 完整性
        pref = bins[0][:3]
        complete = sum(1 for s in [x.lower() for x in stocks if x.upper() in grouped.groups]
                       if len(list((self.features_dir / s).glob(f"{pref}*.day.bin"))) == n_bins)
        print(f"  完整: {complete}/{has_data} 只")
        print(f"{'=' * 60}")

        print(f"  {label} 同步完成")
        print(f"  {'=' * 60}")

    def sync_industry(self, stocks: Optional[List[str]] = None, verify: bool = True):
        """申万行业数据独立同步。使用 sync_fields 实现。"""
        self.sync_fields(
            field_spec={
                "label": "申万行业",
                "bin_pattern": "sw_l*.day.bin",
                "bins": ["sw_l1", "sw_l2", "sw_l3"],
                "data_type": "time_range",
                "query": "SELECT ts_code, l1_code, l2_code, l3_code, in_date, out_date FROM sw_industry_members",
                "code_col": "ts_code",
                "in_date_col": "in_date",
                "out_date_col": "out_date",
                "value_cols": {"sw_l1": "l1_code", "sw_l2": "l2_code", "sw_l3": "l3_code"},
                "parse_value": lambda v: float(str(v).split(".")[0]) if "." in str(v) else float(v),
                "verify_col": "sw_l1",
                "verify_min": 100,
            },
            stocks=stocks,
            verify=verify,
        )

