#!/usr/bin/env python3
"""
反转因子和动量因子计算脚本
从ClickHouse读取前复权价格数据，计算因子并保存为Parquet文件
格式: factor_data/warehouse/{因子名}/{年份}.parquet
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# 添加项目路径
src_dir = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_dir))

from qlworks.config import QLIB_DATA_DIR, CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
import clickhouse_connect


def load_factor_config():
    """加载因子配置文件"""
    config_path = Path(__file__).resolve().parents[2] / "factor_data" / "factor_library" / "reversal_momentum_factors.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['factors']


def get_clickhouse_data():
    """从ClickHouse获取前复权价格数据"""
    print("[1/4] 连接ClickHouse获取前复权价格数据...")
    
    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        user=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE
    )
    
    # 查询前复权价格数据（使用daily_adj_factors计算前复权）
    adj_divisor = "COALESCE(NULLIF(latest.adj_factor, 0), 1)"
    query = f"""
    SELECT
        p.ts_code AS ts_code,
        p.trade_date AS trade_date,
        p.close * COALESCE(a.adj_factor, 1) / {adj_divisor} AS close,
        p.vol AS volume
    FROM daily_prices p
    LEFT JOIN daily_adj_factors a ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
    LEFT JOIN (
        SELECT ts_code, adj_factor
        FROM daily_adj_factors
        WHERE (ts_code, trade_date) IN (
            SELECT ts_code, MAX(trade_date) 
            FROM daily_adj_factors 
            GROUP BY ts_code
        )
    ) latest ON p.ts_code = latest.ts_code
    WHERE p.trade_date >= '2010-01-01'
    ORDER BY p.ts_code, p.trade_date
    """
    
    df = client.query_df(query)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['ts_code'] = df['ts_code'].astype(str)
    # 转换close为float，避免Decimal除法错误
    df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0).astype(float)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(float)
    # 过滤价格为0的记录
    df = df[df['close'] > 0]
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    print(f"    获取到 {len(df):,} 条记录，{df['ts_code'].nunique()} 只股票")
    
    # 关闭连接
    client.close()
    
    return df


def get_sw_industry_data():
    """获取申万行业数据"""
    print("[2/4] 获取申万行业数据...")
    
    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        user=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE
    )
    
    # 查询最新行业分类
    query = """
    SELECT DISTINCT
        ts_code,
        industry as sw_l1
    FROM sw_industry_members
    WHERE industry IS NOT NULL
    """
    
    try:
        df = client.query_df(query)
        df['ts_code'] = df['ts_code'].astype(str)
        print(f"    获取到 {len(df)} 只股票的行业分类")
        client.close()
        return df
    except Exception as e:
        print(f"    行业数据获取失败: {e}")
        client.close()
        return None


def calculate_returns(df):
    """计算各种周期收益率"""
    print("[3/4] 计算收益率...")
    
    # 按股票分组计算收益率
    result_dfs = []
    grouped = df.groupby('ts_code')
    
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 计算各种收益率
        group['ret_1d'] = group['close'].pct_change(1)
        group['ret_5d'] = group['close'].pct_change(5)
        group['ret_20d'] = group['close'].pct_change(20)
        group['ret_63d'] = group['close'].pct_change(63)
        group['ret_126d'] = group['close'].pct_change(126)
        group['ret_232d'] = group['close'].pct_change(232)
        group['ret_252d'] = group['close'].pct_change(252)
        
        result_dfs.append(group)
    
    result = pd.concat(result_dfs, ignore_index=True)
    print(f"    收益率计算完成")
    
    return result


def calculate_factors(price_df, factors):
    """计算所有因子"""
    print("[4/4] 计算因子...")
    
    # 计算收益率
    df = calculate_returns(price_df)
    
    # 因子计算结果
    factor_results = {}
    
    for factor in factors:
        factor_name = factor['name']
        category = factor['category']
        
        print(f"    计算因子: {factor_name} ({category})")
        
        try:
            if factor_name == 'STR_20d':
                # 短期反转20日
                df['value'] = -1 * df.groupby('ts_code')['ret_20d'].shift(1)
                
            elif factor_name == 'STR_5d':
                # 短期反转5日
                df['value'] = -1 * df.groupby('ts_code')['ret_5d'].shift(1)
                
            elif factor_name == 'LTR_12m':
                # 长期反转1年(跳过20天)
                df['value'] = -1 * df.groupby('ts_code')['ret_252d'].shift(20)
                
            elif factor_name == 'VOL_REV':
                # 波动率反转 - 使用收益率的标准差
                df['value'] = -1 * df.groupby('ts_code')['ret_1d'].transform(
                    lambda x: x.rolling(60).std()
                ).shift(1)
                
            elif factor_name == 'EXTREME_REV':
                # 极端收益反转
                df['value'] = df.apply(
                    lambda row: -row['ret_1d'] if abs(row['ret_1d']) > 0.095 else 0, 
                    axis=1
                )
                
            elif factor_name == 'MOM_6m':
                # 6个月动量
                df['value'] = df['ret_126d']
                
            elif factor_name == 'MOM_12m':
                # 12个月动量
                df['value'] = df['ret_252d']
                
            elif factor_name == 'MOM_skip1m':
                # 跳过1个月动量
                df['value'] = df['ret_232d']
                
            elif factor_name == 'MOM_ACCEL':
                # 动量加速度
                df['value'] = df['ret_63d'] - df['ret_126d']
                
            elif factor_name == 'IND_NEUT_MOM':
                # 行业中性动量
                df['value'] = df['ret_126d']
                
            else:
                print(f"    未知因子: {factor_name}")
                continue
            
            # 提取因子值
            factor_df = df[['ts_code', 'trade_date', 'value']].copy()
            
            # 验证数据
            total = len(factor_df)
            if total == 0:
                print(f"    警告: {factor_name} 无数据")
                continue
            
            # 检查是否全是0或NaN
            valid = factor_df['value'].notna().sum()
            non_zero = (factor_df['value'].fillna(0) != 0).sum()
            
            if valid == 0:
                print(f"    警告: {factor_name} 数据全为NaN，跳过")
                continue
                
            if non_zero == 0:
                print(f"    警告: {factor_name} 数据全为0，跳过")
                continue
            
            # 去除重复
            factor_df = factor_df.drop_duplicates(subset=['ts_code', 'trade_date'])
            
            # 过滤异常值（收益率超过1000%的设为NaN）
            factor_df.loc[factor_df['value'].abs() > 10, 'value'] = np.nan
            
            factor_results[factor_name] = factor_df
            print(f"    {factor_name}: {len(factor_df):,} 条记录，有效率 {non_zero/total*100:.1f}%")
            
            # 清理临时列
            df.drop(columns=['value'], inplace=True, errors='ignore')
            
        except Exception as e:
            print(f"    计算 {factor_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    return factor_results


def save_factors_to_warehouse(factor_results, warehouse_dir, factors_config):
    """保存因子到数据仓库（按年分文件）"""
    print("\n保存因子到数据仓库...")
    
    warehouse_dir = Path(warehouse_dir)
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    
    saved_info = {}
    
    for factor_name, df in factor_results.items():
        # 创建因子文件夹
        factor_dir = warehouse_dir / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换日期格式
        df = df.copy()
        df['datetime'] = df['trade_date'].dt.strftime('%Y-%m-%d')
        df = df[['ts_code', 'datetime', 'value']].rename(columns={'ts_code': 'instrument'})
        
        # 获取精确到日的时间范围
        min_date = df['datetime'].min()
        max_date = df['datetime'].max()
        
        # 按年分组保存
        years = sorted(df['datetime'].str[:4].unique())
        total_records = 0
        
        for year in years:
            year_df = df[df['datetime'].str.startswith(year)].copy()
            if len(year_df) == 0:
                continue
            
            # 去除重复
            year_df = year_df.drop_duplicates(subset=['instrument', 'datetime'])
            
            # 设置 MultiIndex [instrument, datetime]
            year_df['datetime'] = pd.to_datetime(year_df['datetime'])
            year_df = year_df.set_index(['instrument', 'datetime'])
            
            # 保存年度文件
            year_file = factor_dir / f"{year}.parquet"
            year_df.to_parquet(year_file)
            total_records += len(year_df)
        
        # 获取因子配置信息
        factor_config = next((f for f in factors_config if f['name'] == factor_name), None)
        
        # 保存meta.json（详细格式）
        meta = {
            "factor_name": factor_name,
            "version": factor_config.get("version", "1.0") if factor_config else "1.0",
            "category": factor_config.get("category", "unknown") if factor_config else "unknown",
            "sub_category": factor_config.get("sub_category", "unknown") if factor_config else "unknown",
            "expression": {
                "qlib": factor_config.get("expression", {}).get("qlib", "") if factor_config else "",
                "duckdb": factor_config.get("expression", {}).get("duckdb", "") if factor_config else ""
            },
            "function_description": factor_config.get("meaning", "") if factor_config else "",
            "theory_background": factor_config.get("logic", {}).get("theory", "") if factor_config else "",
            "applicable_conditions": {
                "market": "中国A股市场",
                "frequency": factor_config.get("parameters", {}).get("freq", "daily") if factor_config else "daily",
                "lookback_period": f"{factor_config.get('parameters', {}).get('lookback', 0)} 交易日" if factor_config else "",
                "expected_direction": factor_config.get("logic", {}).get("expected_direction", "") if factor_config else "",
                "data_requirement": "需要前复权收盘价数据"
            },
            "reference": factor_config.get("ref", "") if factor_config else "",
            "lifecycle_stage": factor_config.get("lifecycle_stage", "exploration") if factor_config else "exploration",
            "data_range": {
                "start_date": min_date,
                "end_date": max_date,
                "years": years
            },
            "statistics": {
                "total_records": int(total_records),
                "unique_stocks": int(df['instrument'].nunique()),
                "data_years": len(years)
            },
            "source": "ClickHouse daily_prices with forward adjustment",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(factor_dir / "meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"    {factor_name}: {total_records:,} 条记录，{len(years)} 年 ({min_date} ~ {max_date})")
        saved_info[factor_name] = meta
    
    return saved_info


def validate_warehouse(warehouse_dir):
    """验证数据仓库"""
    print("\n验证数据仓库...")
    
    warehouse_dir = Path(warehouse_dir)
    print("=" * 85)
    print(f"{'因子名称':<15} {'时间范围':<20} {'记录数':>12} {'文件数':>8} {'股票数':>8} {'非零率':>10}")
    print("=" * 85)
    
    for factor_dir in sorted(warehouse_dir.iterdir()):
        if not factor_dir.is_dir():
            continue
        
        meta_file = factor_dir / "meta.json"
        if not meta_file.exists():
            continue
        
        with open(meta_file, encoding='utf-8') as f:
            meta = json.load(f)
        
        # 读取一个样本文件验证
        sample_files = list(factor_dir.glob("*.parquet"))
        if sample_files:
            df = pd.read_parquet(sample_files[0])
            non_zero_pct = (df['value'].fillna(0) != 0).mean() * 100
        else:
            non_zero_pct = 0
        
        # 获取时间范围
        data_range = meta.get('data_range', {})
        date_str = f"{data_range.get('start_date', '')} ~ {data_range.get('end_date', '')}"
        
        print(f"{factor_dir.name:<15} {date_str:<20} {meta.get('statistics', {}).get('total_records', 0):>12,} {len(sample_files):>8} {meta.get('statistics', {}).get('unique_stocks', 0):>8,} {non_zero_pct:>9.1f}%")
    
    print("=" * 85)


def main():
    print("=" * 60)
    print("反转因子与动量因子计算")
    print("数据来源: ClickHouse (前复权价格)")
    print("=" * 60)
    
    # 加载因子配置
    factors = load_factor_config()
    print(f"加载了 {len(factors)} 个因子")
    
    # 获取价格数据
    price_df = get_clickhouse_data()
    
    # 获取行业数据(暂不使用)
    industry_df = get_sw_industry_data()
    
    # 计算因子
    factor_results = calculate_factors(price_df, factors)
    
    if not factor_results:
        print("没有计算出任何因子，请检查数据源和计算逻辑")
        return []
    
    # 保存因子到数据仓库
    warehouse_dir = Path(__file__).resolve().parents[2] / "factor_data" / "warehouse"
    saved_info = save_factors_to_warehouse(factor_results, warehouse_dir, factors)
    
    print(f"\n完成! 保存了 {len(saved_info)} 个因子到:\n  {warehouse_dir}")
    
    # 验证数据
    validate_warehouse(warehouse_dir)
    
    return saved_info


if __name__ == "__main__":
    main()
