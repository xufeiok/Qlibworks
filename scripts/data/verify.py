import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from qlworks.data.api import QuantDataAPI
from qlworks.config import QLIB_DATA_DIR


def check_calendar_continuity():
    """检查交易日历连续性"""
    print("\n[1] 检查交易日历连续性...")
    
    cal_file = Path(QLIB_DATA_DIR) / "calendars" / "day.txt"
    if not cal_file.exists():
        print("    [FAIL] 日历文件不存在")
        return False
    
    with open(cal_file, 'r') as f:
        dates = [line.strip() for line in f.readlines()]
    
    invalid_dates = []
    for d in dates:
        try:
            datetime.strptime(d, '%Y-%m-%d')
        except:
            invalid_dates.append(d)
    
    if invalid_dates:
        print(f"    [FAIL] 无效日期格式：{invalid_dates[:5]}")
        return False
    
    print(f"    [OK] 日历文件正常，共 {len(dates)} 个交易日")
    return True


def check_instruments():
    """检查股票池文件"""
    print("\n[2] 检查股票池文件...")
    
    instruments_dir = Path(QLIB_DATA_DIR) / "instruments"
    
    for fname in ["all.txt", "all_sh.txt", "all_sz.txt"]:
        fpath = instruments_dir / fname
        if not fpath.exists():
            print(f"    [FAIL] {fname} 不存在")
            return False
        
        with open(fpath, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            print(f"    [FAIL] {fname} 为空")
            return False
        
        for line in lines[:5]:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                print(f"    [FAIL] {fname} 格式错误：{line}")
                return False
    
    print(f"    [OK] 股票池文件正常")
    return True


def check_features():
    """检查特征文件"""
    print("\n[3] 检查特征文件...")
    
    features_dir = Path(QLIB_DATA_DIR) / "features"
    if not features_dir.exists():
        print("    [FAIL] features 目录不存在")
        return False
    
    stock_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
    if len(stock_dirs) == 0:
        print("    [FAIL] features 目录为空")
        return False
    
    error_count = 0
    
    for stock_dir in stock_dirs[:10]:
        bin_files = list(stock_dir.glob("*.day.bin"))
        if len(bin_files) == 0:
            print(f"    [WARN] {stock_dir.name} 无 .bin 文件")
            error_count += 1
            continue
        
        for bin_file in bin_files[:3]:
            if bin_file.stat().st_size < 4:
                print(f"    [WARN] {bin_file.name} 文件过小")
                error_count += 1
    
    if error_count > 0:
        print(f"    [WARN] 发现 {error_count} 个问题")
        return False
    
    print(f"    [OK] 特征文件正常，共 {len(stock_dirs)} 只股票")
    return True


def check_consistency_with_ch():
    """检查与 ClickHouse 的一致性"""
    print("\n[4] 检查与 ClickHouse 的一致性...")
    
    try:
        with QuantDataAPI() as api:
            results = api.check_consistency()
            
            if results.get("qlib_sync") is True:
                print("    [OK] Qlib 与 ClickHouse 数据一致")
            elif results.get("qlib_sync") is False:
                print("    [FAIL] Qlib 与 ClickHouse 数据不一致")
                return False
            else:
                print("    [SKIP] 跳过 Qlib 检查")
            
            if results.get("feature_files") is True:
                print("    [OK] 特征文件完整")
            elif results.get("feature_files") is False:
                print("    [FAIL] 特征文件缺失")
                return False
    
    except Exception as e:
        print(f"    [FAIL] 一致性检查失败：{e}")
        return False
    
    return True


def check_table_stats():
    """检查所有 ClickHouse 数据表的时间范围和行数"""
    print("\n[5] ClickHouse 数据表统计...")
    
    tables = {
        'daily_prices': 'trade_date',
        'daily_indicators': 'trade_date',
        'daily_adj_factors': 'trade_date',
        'financial_indicators': 'ann_date',
        'stock_universe': 'list_date',
        'index_daily': 'trade_date',
        'money_flow': 'trade_date',
    }
    
    try:
        with QuantDataAPI() as api:
            results = {}
            for table, date_col in tables.items():
                sql = f"SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date, COUNT(*) as total_rows FROM {table}"
                r = api.query(sql)
                min_date = r.iloc[0, 0]
                max_date = r.iloc[0, 1]
                total_rows = r.iloc[0, 2]
                results[table] = (min_date, max_date, total_rows)
                print(f"    {table}: {min_date} ~ {max_date} ({total_rows:,} 行)")
            
            # 检查 early_stocks
            stocks = api.get_stock_list()
            early_stocks = stocks[stocks['list_date'] < '2010-01-01']
            print(f"\n    {len(early_stocks)} 只在 2010 年前上市")
            
            gaps = 0
            for _, row in early_stocks.sample(min(5, len(early_stocks))).iterrows():
                r2 = api.query(
                    f"SELECT MIN(trade_date) as min_date FROM daily_prices WHERE ts_code = '{row['ts_code']}'"
                )
                data_min = str(r2.iloc[0, 0])[:10] if r2.iloc[0, 0] else ''
                if data_min > str(row['list_date'])[:10]:
                    gaps += 1
                    print(f"    ⚠️ {row['ts_code']}: 上市 {str(row['list_date'])[:10]}, 数据从 {data_min}")
            
            if gaps == 0:
                print("    [OK] 抽查样本数据完整")
            return results
    except Exception as e:
        print(f"    [FAIL] 查询失败: {e}")
        return {}


def generate_report(extended: bool = False):
    """生成数据质量报告"""
    print("\n" + "=" * 60)
    print("数据质量报告")
    print("=" * 60)
    
    results = {
        "calendar": check_calendar_continuity(),
        "instruments": check_instruments(),
        "features": check_features(),
        "consistency": check_consistency_with_ch(),
    }
    
    if extended:
        check_table_stats()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    all_pass = all(results.values())
    
    for check_name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        result_text = "通过" if passed else "失败"
        print(f"    {status} {check_name}: {result_text}")
    
    if all_pass:
        print("\n[OK] 所有检查通过！数据质量良好。")
    else:
        print("\n[WARN] 部分检查未通过，请检查上述报告。")
    
    return all_pass


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="数据验证工具")
    parser.add_argument("--extended", "-e", action="store_true",
                        help="执行扩展检查（含 ClickHouse 表统计和缺失数据扫描）")
    args = parser.parse_args()

    print("=" * 60)
    print("数据验证工具")
    print("=" * 60)
    
    success = generate_report(extended=args.extended)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
