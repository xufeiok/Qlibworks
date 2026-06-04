"""
反转因子单因子评测脚本

从 warehouse 加载反转因子数据，进行 IC 分析、分层回测和稳健性检验。
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from qlworks.evaluation.runner import FactorEvaluator
from qlworks.evaluation.config import DEFAULT_CONFIG


def load_factor_from_warehouse(factor_name, warehouse_dir):
    """从 warehouse 加载因子数据"""
    factor_dir = Path(warehouse_dir) / factor_name
    
    if not factor_dir.exists():
        raise FileNotFoundError(f"因子目录不存在: {factor_dir}")
    
    # 读取所有年度文件
    parquet_files = sorted(factor_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {factor_dir}")
    
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    print(f"  加载 {factor_name}: {len(df):,} 条记录, {df['instrument'].nunique()} 只股票")
    return df


def main():
    print("=" * 60)
    print("反转因子单因子评测")
    print("=" * 60)
    
    # 配置
    warehouse_dir = Path(__file__).resolve().parents[2] / "factor_data" / "warehouse"
    print(f"因子仓库: {warehouse_dir}")
    
    # 定义反转因子列表
    reversal_factors = [
        "STR_5d",
        "STR_20d", 
        "LTR_12m",
        "VOL_REV",
        "EXTREME_REV"
    ]
    
    print(f"\n待评测反转因子: {reversal_factors}")
    
    # 创建评测器
    evaluator = FactorEvaluator()
    
    # 批量评测
    results = []
    for factor_name in reversal_factors:
        print(f"\n--- 评测 {factor_name} ---")
        
        try:
            # 加载因子数据
            factor_df = load_factor_from_warehouse(factor_name, warehouse_dir)
            
            # 评测
            result = evaluator.evaluate(factor_name, factor_df)
            
            # 打印关键指标
            ic_stats = result["ic_stats"]
            ls_stats = result["ls_stats"]
            qual = result["qual_result"]
            
            print(f"\n【{factor_name} 评测结果】")
            print(f"├── IC 均值: {ic_stats['ic_mean']:.4f}")
            print(f"├── ICIR: {ic_stats['icir']:.2f}")
            print(f"├── 胜率: {ic_stats['win_rate']:.2%}")
            print(f"├── T统计量: {ic_stats['t_stat']:.2f}")
            print(f"├── 单调性: {ic_stats['monotonicity']:.4f}")
            print(f"├── 多空年化收益: {ls_stats.get('annual_return', 0):.2%}")
            print(f"├── 多空夏普比率: {ls_stats.get('sharpe_ratio', 0):.2f}")
            print(f"└── 等级评定: {qual['tier']} (综合评分: {qual['composite_score']:.1f})")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ 评测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总报告
    print("\n" + "=" * 60)
    print("反转因子评测汇总")
    print("=" * 60)
    print(f"{'因子名称':<12} {'IC均值':<10} {'ICIR':<8} {'胜率':<8} {'多空收益':<12} {'等级'}")
    print("-" * 60)
    
    for result in results:
        if "error" in result:
            continue
            
        name = result["factor_name"]
        ic = result["ic_stats"]["ic_mean"]
        icir = result["ic_stats"]["icir"]
        win_rate = result["ic_stats"]["win_rate"]
        ls_ret = result["ls_stats"].get("annual_return", 0)
        tier = result["qual_result"]["tier"]
        
        print(f"{name:<12} {ic:<10.4f} {icir:<8.2f} {win_rate:<8.1%} {ls_ret:<12.2%} {tier}")
    
    print("\n报告已保存到: reports/evaluation/")


if __name__ == "__main__":
    main()