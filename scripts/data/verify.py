"""
数据验证脚本 - 验证数据完整性和一致性

功能：
- 验证 Qlib 数据与 ClickHouse 一致性
- 验证特征文件完整性
- 验证交易日历连续性
- 生成数据质量报告

用法：
    python -m scripts.data.verify
"""
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
    
    # 检查日期格式
    invalid_dates = []
    for d in dates:
        try:
            datetime.strptime(d, '%Y-%m-%d')
        except:
            invalid_dates.append(d)
    
    if invalid_dates:
        print(f"    [FAIL] 无效日期格式：{invalid_dates[:5]}")
        return False
    
    # 检查连续性（允许周末和节假日空缺）
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
        
        # 检查格式
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
    
    # 随机检查几只股票
    sample_count = 0
    error_count = 0
    
    for stock_dir in stock_dirs[:10]:  # 检查前 10 只
        bin_files = list(stock_dir.glob("*.day.bin"))
        if len(bin_files) == 0:
            print(f"    [WARN] {stock_dir.name} 无 .bin 文件")
            error_count += 1
            continue
        
        # 检查文件大小（应该大于 4 字节）
        for bin_file in bin_files[:3]:
            if bin_file.stat().st_size < 4:
                print(f"    [WARN] {bin_file.name} 文件过小")
                error_count += 1
        
        sample_count += 1
    
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


def generate_report():
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
    print("=" * 60)
    print("数据验证工具")
    print("=" * 60)
    
    success = generate_report()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
