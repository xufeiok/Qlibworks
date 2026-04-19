from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataclasses import dataclass
from typing import List, Sequence

from qlworks.factors.manager import FactorLibraryManager


@dataclass
class FeatureBundle:
    """
    功能概述：
    - 统一描述一组研究特征与监督标签，便于数据集构建和模型训练复用。
    输入：
    - fields/names: 特征表达式与显示名称。
    - label_fields/label_names: 标签表达式与名称。
    输出：
    - 轻量配置对象。
    边界条件：
    - names 与 fields 数量应保持一致。
    性能/安全注意事项：
    - 仅保存配置，不直接计算任何特征。
    """

    fields: Sequence[str]
    names: Sequence[str]
    label_fields: Sequence[str]
    label_names: Sequence[str]


def build_alpha_feature_bundle() -> FeatureBundle:
    """
    功能概述：
    - 构建一组适合作为快速研究原型的手工特征，映射课程中的特征工程演示。
    输入：
    - 无。
    输出：
    - 可直接用于 `D.features` 的表达式集合。
    边界条件：
    - 表达式需由当前 Qlib 版本支持。
    性能/安全注意事项：
    - 特征数量控制在较小规模，便于快速试验和调参。
    """
    fields = [
        "$open",
        "$high",
        "$low",
        "$close",
        "$volume",
        "$close / Ref($close, 1) - 1",
        "$close / Ref($close, 5) - 1",
        "Mean($close, 5)",
        "Mean($close, 20)",
        "Std($close, 20)",
        "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))",
        "$volume / Mean($volume, 20)",
        "Corr($close, $volume, 10)",
        "Mean($close > Ref($close, 1), 10)",
    ]
    names = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1d",
        "ret_5d",
        "ma_5",
        "ma_20",
        "std_20",
        "price_position_20",
        "volume_ratio_20",
        "price_volume_corr_10",
        "up_prob_10",
    ]
    return FeatureBundle(
        fields=fields,
        names=names,
        label_fields=["Ref($close, -1) / $close - 1"],
        label_names=["LABEL0"],
    )


def build_factor_library_bundle(strategy_names: str | list[str], repo_path: str = None) -> FeatureBundle:
    """
    功能概述：
    - 从 YAML 因子库加载策略因子并转成统一特征包。
    - 支持加载单个因子文件，或合并多个因子文件。
    输入：
    - strategy_names: 因子组合名称（或名称列表），如 "alpha158_factor_dictionary" 或 ["alpha158_factor_dictionary", "gtja191_factor_dictionary"]。
    - repo_path: 因子库目录，可为空使用默认目录。
    输出：
    - 统一的特征与标签配置对象。
    边界条件：
    - 若策略不存在将抛出文件错误。遇到同名因子会自动跳过并警告。
    性能/安全注意事项：
    - 仅解析 YAML，不执行 SQL 或 Qlib 表达式，安全风险低。
    """
    manager = FactorLibraryManager(repo_path=repo_path)
    fields, names = manager.get_qlib_expressions(strategy_names)
    return FeatureBundle(
        fields=fields,
        names=names,
        label_fields=["Ref($close, -1) / $close - 1"],
        label_names=["LABEL0"],
    )

if __name__ == "__main__":
    print("=== features/builder.py 独立调用示例 ===")
    
    # 1. 构建默认的手工特征包 (Alpha 演示特征)
    print("\n[1] 构建手工特征包 (build_alpha_feature_bundle):")
    manual_bundle = build_alpha_feature_bundle()
    print(f"- 特征数量: {len(manual_bundle.fields)}")
    print(f"- 特征名称 (前5个): {manual_bundle.names[:5]}")
    print(f"- 标签表达式: {manual_bundle.label_fields}")
    
    # 2. 尝试从因子库 (YAML) 构建单文件特征包
    print("\n[2] 尝试从 Factor Library 构建单文件特征包 (build_factor_library_bundle):")
    try:
        library_bundle = build_factor_library_bundle(strategy_names="weekly_reversal_v1")
        print(f"- 成功！从因子库读取到了 {len(library_bundle.fields)} 个因子表达式。")
        print(f"- 因子名称: {library_bundle.names[:5]}")
    except Exception as e:
        print(f"- 因子库读取失败: {e}")
        
    # 3. 尝试从因子库合并多个文件特征包
    print("\n[3] 尝试合并多个 Factor Library 文件 (多因子合成):")
    try:
        # 同时加载 weekly_reversal 和 master 两个字典的因子
        multi_bundle = build_factor_library_bundle(strategy_names=["weekly_reversal_v1", "master_factor_dictionary"])
        print(f"- 成功！合并后共读取到了 {len(multi_bundle.fields)} 个因子表达式。")
        print(f"- 因子名称 (前5个): {multi_bundle.names[:5]}")
    except Exception as e:
        print(f"- 因子库读取失败: {e}")
