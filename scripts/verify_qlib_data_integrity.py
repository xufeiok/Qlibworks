"""
验证 Qlib 数据完整性
检查 qlib_data 目录中的数据是否与 build_qlib_from_duckdb.py 的预期一致
"""
import os
from pathlib import Path
from collections import defaultdict

QLIB_DATA_DIR = Path(r"e:\Quant\Qlibworks\qlib_data")
INSTRUMENTS_FILE = QLIB_DATA_DIR / "instruments" / "all.txt"
FEATURES_DIR = QLIB_DATA_DIR / "features"
CALENDARS_FILE = QLIB_DATA_DIR / "calendars" / "day.txt"

# 根据 build_qlib_from_duckdb.py 中定义的字段
EXPECTED_FIELDS = [
    "open", "close", "high", "low", "volume", "amount", 
    "vwap", "pe", "pe_ttm", "pb", "ps", "ps_ttm", 
    "total_mv", "circ_mv", "turnover_rate", "dv_ttm", 
    "rzye", "north_hold", "roe_ttm", "roa", 
    "grossprofit_margin", "netprofit_margin", "netprofit_yoy", 
    "tr_yoy", "basic_eps_yoy", "debt_to_assets", 
    "current_ratio", "inv_turn", "ocfps", "eps", 
    "dt_netprofit_yoy", "stk_holdernumber", "pledge_ratio", 
    "eps_forecast", "factor"
]

def check_instruments():
    """检查 instruments 文件"""
    print("=" * 80)
    print("1. 检查 instruments 文件")
    print("=" * 80)
    
    if not INSTRUMENTS_FILE.exists():
        print(f"❌ instruments 文件不存在: {INSTRUMENTS_FILE}")
        return []
    
    with open(INSTRUMENTS_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    instruments = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 3:
            code, start, end = parts
            instruments.append({
                'code': code,
                'start': start,
                'end': end
            })
    
    print(f"✓ instruments 文件包含 {len(instruments)} 只股票")
    print(f"  示例: {instruments[0]}")
    print(f"  示例: {instruments[-1]}")
    
    # 检查是否有重复
    codes = [inst['code'] for inst in instruments]
    if len(codes) != len(set(codes)):
        print(f"⚠️  发现重复的股票代码!")
    else:
        print(f"✓ 无重复股票代码")
    
    return instruments

def check_calendars():
    """检查日历文件"""
    print("\n" + "=" * 80)
    print("2. 检查 calendars 文件")
    print("=" * 80)
    
    if not CALENDARS_FILE.exists():
        print(f"❌ calendars 文件不存在: {CALENDARS_FILE}")
        return []
    
    with open(CALENDARS_FILE, 'r', encoding='utf-8') as f:
        dates = [line.strip() for line in f if line.strip()]
    
    print(f"✓ 日历文件包含 {len(dates)} 个交易日")
    print(f"  起始日期: {dates[0]}")
    print(f"  结束日期: {dates[-1]}")
    
    return dates

def check_features(instruments):
    """检查特征数据文件"""
    print("\n" + "=" * 80)
    print("3. 检查 features 目录")
    print("=" * 80)
    
    if not FEATURES_DIR.exists():
        print(f"❌ features 目录不存在: {FEATURES_DIR}")
        return
    
    # 获取所有股票目录
    feature_dirs = [d for d in FEATURES_DIR.iterdir() if d.is_dir()]
    print(f"✓ features 目录包含 {len(feature_dirs)} 个股票文件夹")
    
    # 检查是否有缺失的股票
    instrument_codes = set(inst['code'].lower() for inst in instruments)
    feature_codes = set(d.name.lower() for d in feature_dirs)
    
    missing_in_features = instrument_codes - feature_codes
    extra_in_features = feature_codes - instrument_codes
    
    if missing_in_features:
        print(f"❌ instruments 中有但 features 中缺失的股票 ({len(missing_in_features)} 只):")
        for code in sorted(list(missing_in_features)[:10]):
            print(f"  - {code}")
        if len(missing_in_features) > 10:
            print(f"  ... 还有 {len(missing_in_features) - 10} 只")
    else:
        print(f"✓ 所有 instrument 股票在 features 中都有对应目录")
    
    if extra_in_features:
        print(f"⚠️  features 中有多余的股票 ({len(extra_in_features)} 只):")
        for code in sorted(list(extra_in_features)[:10]):
            print(f"  - {code}")
    
    # 检查每个股票的字段文件
    print(f"\n检查字段完整性 (抽查前10只股票)...")
    sample_stocks = sorted(list(feature_codes))[:10]
    
    # Qlib 的字段文件格式为: {field}.{freq}.bin (例如: open.day.bin)
    expected_field_files = [f"{field}.day.bin" for field in EXPECTED_FIELDS]
    
    for stock_code in sample_stocks:
        stock_dir = FEATURES_DIR / stock_code
        if not stock_dir.exists():
            continue
        
        bin_files = list(stock_dir.glob("*.bin"))
        field_file_names = [f.name for f in bin_files]
        
        missing_files = set(expected_field_files) - set(field_file_names)
        extra_files = set(field_file_names) - set(expected_field_files)
        
        if missing_files:
            missing_fields = [f.replace('.day.bin', '') for f in missing_files]
            print(f"❌ {stock_code} 缺失字段: {missing_fields}")
        if extra_files:
            extra_fields = [f.replace('.day.bin', '') for f in extra_files]
            print(f"⚠️  {stock_code} 多余字段: {extra_fields}")
    
    print(f"✓ 字段完整性检查完成")
    print(f"  期望的字段数: {len(EXPECTED_FIELDS)}")
    # 打印最后一个抽样股票的字段数作为参考
    if sample_stocks:
        last_stock = sample_stocks[-1]
        last_stock_dir = FEATURES_DIR / last_stock
        if last_stock_dir.exists():
            last_bin_files = list(last_stock_dir.glob("*.bin"))
            print(f"  实际找到的字段数 (抽样 {last_stock}): {len(last_bin_files)}")

def check_data_consistency():
    """检查数据一致性"""
    print("\n" + "=" * 80)
    print("4. 数据一致性检查")
    print("=" * 80)
    
    # 随机抽查几只股票的数据文件
    import random
    
    if FEATURES_DIR.exists():
        all_stocks = [d.name for d in FEATURES_DIR.iterdir() if d.is_dir()]
        if all_stocks:
            sample_stocks = random.sample(all_stocks, min(5, len(all_stocks)))
            print(f"随机抽查股票: {sample_stocks}")
            
            for stock in sample_stocks:
                stock_dir = FEATURES_DIR / stock
                bin_files = list(stock_dir.glob("*.bin"))
                
                if bin_files:
                    # 检查文件大小是否合理
                    sizes = [f.stat().st_size for f in bin_files]
                    avg_size = sum(sizes) / len(sizes)
                    print(f"  {stock}: {len(bin_files)} 个字段, 平均文件大小 {avg_size:.0f} bytes")

def main():
    print(f"开始验证 Qlib 数据完整性...")
    print(f"数据目录: {QLIB_DATA_DIR}\n")
    
    # 1. 检查 instruments
    instruments = check_instruments()
    
    # 2. 检查 calendars
    calendars = check_calendars()
    
    # 3. 检查 features
    if instruments:
        check_features(instruments)
    
    # 4. 数据一致性
    check_data_consistency()
    
    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
