#!/usr/bin/env python3
"""
因子库管理系统使用演示 (Demo Script)
展示如何加载 YAML 因子配置，并生成 Qlib 可以使用的特征表达式。
"""

import sys
import os

# 将 qlworks 源码目录加入环境变量，方便导入
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

from qlworks.factors import FactorLibraryManager
from tabulate import tabulate

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
    
    # 1. 实例化因子库管理器 (默认会自动寻找 factors_repo 文件夹)
    manager = FactorLibraryManager()
    
    # 2. 列出当前已配置的策略组合
    print("\n[INFO] 当前因子库中包含以下组合：")
    strategies = manager.list_strategies()
    print(strategies)
    
    if not strategies:
        print("未找到任何策略配置文件，请检查 factors_repo 目录。")
        return
        
    # 获取我们刚刚配置的 master_factor_dictionary.yaml
    # 注意，这里的 strategy_name 要与文件名（不含.yaml后缀）一致
    strategy_name = 'alpha158_factor_dictionary'
    if strategy_name not in strategies:
        strategy_name = strategies[0]
        
    # 3. 打印完整的策略分析报告 (展示知识库的价值)
    manager.print_strategy_report(strategy_name)
    
    # 4. 获取可以在 Qlib 中直接使用的特征表达式
    # 这是将其接入 Qlib 模型训练的核心步骤
    expressions, names = manager.get_qlib_expressions(strategy_name)
    
    print("💻 生成的 Qlib FeatureRecord 结构如下 (可直接用于 DataHandlerLP 实例化):")
    feature_record = [expressions, names]
    print("expressions:", expressions)
    print("names:", names)
    
    # 5. 获取可以在 DuckDB 中直接使用的特征表达式
    duckdb_expressions, _ = manager.get_duckdb_expressions(strategy_name)
    print("\n🦆 生成的 DuckDB 表达式如下 (可用于构建 SQL Select 子句):")
    for name, expr in zip(names, duckdb_expressions):
        print(f"  - {name}: {expr}")
        
    print("\n你可以像这样在 Qlib 中使用它:")
    print("handler = DataHandlerLP(")
    print("    instruments='csi300',")
    print("    start_time='2020-01-01',")
    print("    end_time='2020-12-31',")
    print("    infer_processors=[...],")
    print("    learn_processors=[...],")
    print("    # 用自定义配置替换 Alpha158")
    print(f"    data_loader={{'class': 'QlibDataLoader', 'kwargs': {{'config': {feature_record}}}}}")
    print(")")

if __name__ == "__main__":
    main()
