"""
验证 Qlib 数据完整性
检查 qlib_data 目录中的数据是否与 build_qlib_from_duckdb.py 的预期一致
"""
import os
import struct
import datetime
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


def read_bin_dates(bin_path):
    """
    读取 .bin 文件中所有有数据的日期

    .bin 格式：每16字节一组（8字节日期double + 8字节值double）
    日期编码：YYYYMMDD (如 20100104.0)
    """
    if not bin_path.exists():
        return set()
    try:
        data = bin_path.read_bytes()
        dates = set()
        for i in range(0, len(data), 16):
            if i + 16 > len(data):
                break
            date_val = struct.unpack_from('<d', data, i)[0]
            val = struct.unpack_from('<d', data, i + 8)[0]
            if val is not None and not (isinstance(val, float) and val != val):
                dates.add(int(date_val))
        return dates
    except Exception:
        return set()


def parse_date_from_int(date_int):
    """将 YYYYMMDD int 转换为日期字符串"""
    y = date_int // 10000
    m = (date_int % 10000) // 100
    d = date_int % 100
    return f"{y:04d}-{m:02d}-{d:02d}"


def format_gap_summary(missing_dates):
    """将缺失日期列表压缩为范围描述"""
    if not missing_dates:
        return "无缺失"
    sorted_dates = sorted(missing_dates)
    ranges = []
    start = sorted_dates[0]
    end = sorted_dates[0]
    for d in sorted_dates[1:]:
        if d == end + 1:
            end = d
        else:
            ranges.append((start, end))
            start = d
            end = d
    ranges.append((start, end))

    parts = []
    for r in ranges:
        if r[0] == r[1]:
            parts.append(parse_date_from_int(r[0]))
        else:
            parts.append(f"{parse_date_from_int(r[0])} ~ {parse_date_from_int(r[1])}")

    if len(parts) > 5:
        return f"{', '.join(parts[:5])} ... 共 {len(missing_dates)} 天"
    return f"{', '.join(parts)} （共 {len(missing_dates)} 天）"

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

def check_calendar_coverage(calendars, sample_size=100, compare_fields=None):
    """
    检查 .bin 文件中实际数据的日期是否与交易日历一致

    逻辑：
    - 对每只股票的每个字段，读取 .bin 中的实际日期
    - 与全量日历比较，找出缺失的日期
    - 对于行情字段（open/high/low/close/vol），只在股票上市期间内校验

    Args:
        calendars: 交易日历列表（字符串格式 'YYYY-MM-DD'）
        sample_size: 抽样检查的股票数
        compare_fields: 要检查的字段列表，默认只检查关键字段
    """
    print("\n" + "=" * 80)
    print("5. 日历覆盖检查（检查 .bin 文件中是否有遗漏的日期）")
    print("=" * 80)

    calendar_set = set()
    for c in calendars:
        parts = c.split('-')
        calendar_set.add(int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2]))
    sorted_calendar = sorted(calendar_set)

    if compare_fields is None:
        compare_fields = ["close", "volume", "amount", "pe", "circ_mv", "turnover_rate"]

    all_stocks = sorted([d.name for d in FEATURES_DIR.iterdir() if d.is_dir()])
    if not all_stocks:
        print("❌ features 目录为空")
        return

    sample_stocks = all_stocks[:sample_size]
    print(f"抽样检查 {len(sample_stocks)} 只股票，对比字段: {compare_fields}")
    print()

    total_gaps = 0
    total_check_dates = 0
    stock_gap_summary = []

    for stock_code in sample_stocks:
        stock_dir = FEATURES_DIR / stock_code
        field_gaps = {}

        for field in compare_fields:
            bin_path = stock_dir / f"{field}.day.bin"
            if not bin_path.exists():
                field_gaps[field] = f"文件缺失"
                continue

            dates_in_bin = read_bin_dates(bin_path)
            if not dates_in_bin:
                field_gaps[field] = "文件中无有效数据"
                continue

            min_date = min(dates_in_bin)
            max_date = max(dates_in_bin)

            expected_dates = {d for d in sorted_calendar if min_date <= d <= max_date}
            missing = sorted(expected_dates - dates_in_bin)

            total_check_dates += len(expected_dates)
            if missing:
                total_gaps += len(missing)
                field_gaps[field] = format_gap_summary(missing)
            else:
                field_gaps[field] = f"完整 ({parse_date_from_int(min_date)} ~ {parse_date_from_int(max_date)}, {len(dates_in_bin)}天)"

        if field_gaps:
            stock_gap_summary.append((stock_code, field_gaps))

    print(f"\n扫描结果：共扫描 {len(sample_stocks)} 只股票，检查 {total_check_dates} 个日期点")
    print(f"发现数据缺口 {total_gaps} 天\n")

    # 按缺口严重程度排序输出
    if stock_gap_summary:
        has_issues = [(s, f) for s, f in stock_gap_summary
                       if any("缺失" in v or "无有效" in v or "共 " in v for v in f.values())]
        clean = [(s, f) for s, f in stock_gap_summary if s not in {x[0] for x in has_issues}]

        if has_issues:
            print("--- 存在问题的股票 ---")
            for stock_code, gaps in has_issues[:20]:
                print(f"\n  [{stock_code}]")
                for field, desc in gaps.items():
                    if "完整" in desc:
                        print(f"    ✅ {field}: {desc}")
                    else:
                        print(f"    ❌ {field}: {desc}")
            if len(has_issues) > 20:
                print(f"  ... 还有 {len(has_issues) - 20} 只股票未显示")

        print(f"\n--- 汇总 ---")
        print(f"  完整无缺: {len(clean)} 只")
        print(f"  存在缺口: {len(has_issues)} 只")
    else:
        print("✅ 所有抽样股票数据完整，无日期缺失")


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

    # 5. 日历覆盖检查（新增）
    if calendars:
        check_calendar_coverage(calendars)

    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
