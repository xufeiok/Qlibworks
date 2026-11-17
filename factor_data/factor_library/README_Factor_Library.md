# 量化因子库五大类家族管理指南 (Factor Library Guide)

## 概述

本因子库于 2026-10-08 完成**五大类家族归档**，依据《量化因子家族管理规范（头部机构标准版·适配A股周频多因子）》
落地为 **5 个家族文件 + 1 个 other 暂存文件**，每个文件头部带统计注释块（因子总数/子类构成/因子名清单/
warehouse 缓存对应情况），因子按 category 分组、组内按名称排序，条目间空 1 行，便于快速查找。

设计要点：
- **文件 = 因子家族**（物理组织与质检单元），与 warehouse parquet 缓存按因子名解耦；
- **归档因子与 warehouse 一一对应**（239/239），无缓存因子（KUP2/KLOW2）暂存 other 待计算；
- 子类 category 字段保留（质量/成长/价值/动量与反转/量价形态/滚动统计/波动与风险/规模与流动性），
  支持家族文件内按子集加载。

---

## 五大家族归档总览

| # | 家族 | 文件 | 因子数 | 子类构成 | 家族口径 |
|---|---|---|---|---|---|
| 1 | **Quality 质量基本面** | `quality_factors.yaml` | 11 | 质量×8、成长×3 | 季度更新、策略底仓、半衰期 6-12 月 |
| 2 | **Value 价值估值** | `value_factors.yaml` | 4 | 价值×4 | 均值回归、需控行业市值暴露 |
| 3 | **Momentum&Price 量价趋势** | `momentum_price_factors.yaml` | 215 | 动量与反转×13、量价形态×57、滚动统计×145 | 周频核心 Alpha、占比红线 ≤35% |
| 4 | **Sentiment 预期情绪** | `sentiment_factors.yaml` | 0 | （骨架待接入数据） | 事件驱动 |
| 5 | **Risk 风险流动性** | `risk_factors.yaml` | 9 | 波动与风险×4、规模与流动性×5 | 仅做风险约束，不参与 Alpha 比拼 |
| — | 暂存区 | `other_factors.yaml` | 2 | KUP2、KLOW2（warehouse 无缓存） | 计算缓存后按 category 归档至家族 |
| | **合计** | | **241** | | |

合并来源：`momentum_price_factors` ← 原 `reversal_momentum_factors` + `price_volume_factors` + `rolling_factors`
（备份于 `archive/family5_backup_20261008/`）；`value_factors` ← 原 `style_factors` 估值子类独立成档。

---

## 使用方法

```python
import sys; sys.path.insert(0, r"E:\Quant\Qlibworks\src")
from qlworks.features import build_factor_library_bundle

# 1. 加载单一家族（如质量基本面选股）
quality_bundle = build_factor_library_bundle("quality_factors")

# 2. 混合加载多家族（送机器学习模型）
ml_bundle = build_factor_library_bundle([
    "quality_factors", "value_factors",
    "momentum_price_factors", "risk_factors",
])

# 3. 家族内取子类（如只要「动量与反转」，不取形态/滚动统计）
mom_bundle = build_factor_library_bundle(
    "momentum_price_factors", categories=["动量与反转"]
)  # → 13 个因子
```

训练/筛选/评测脚本默认加载列表已全部同步（`select_factors.py`、`screen_factors.py`、
`train_tree-doubao.py`、`train_tree.py`、`train_icir.py`、`train_linear.py`、`_config.py`、
`rebuild_warehouse_staged.py`、`eval_selected_library.py`、`compute_all_missing.py`、
`update_warehouse_from_local_qlib.py`、`sync_factors.py`、`factor_def.py`）。

单子库实验脚本（`train_icir`/`train_linear`/`_config`/`train_tree`）通过 CONFIG 中的
`factor_categories: ["动量与反转"]` 保持原口径不变。

---

## 归档纪律（机构铁律）

1. **五大家族文件内的因子必须与 warehouse 缓存一一对应**；无缓存新因子一律先入 `other_factors.yaml`，
   计算缓存就绪后按 category 归档至对应家族；
2. **挖掘不限数量、入池严控配额**（参考规范三：质量 32-36%、价值 20-24%、量价 28-32%、
   预期 8-12%、风险 4-8%；量价总占比红线 ≤35%）；
3. **家族内 |r| ≥ 0.7 去冗余，全局跨家族兜底去冗余**（保留优先级：基本面 > 预期 > 量价 > 风险）；
4. **禁止在主库新建副本集**（原 `selected_good_factors.yaml` 160 个副本已归档至
   `archive/reorg_backup_20261008/`；实盘精选池由 `select_factors.py` 流程输出）。

---

## 注意事项

1. **因子正负向**：树模型可自动捕捉单调性，无需手动翻转；Barra 线性模型需在预处理时翻转负向因子。
2. **避免未来函数**：财务类因子确保 Point-in-Time 对齐。
3. **存档目录**：`archive/` 不参与日常加载（`family5_backup_20261008/` 为本次五大类归档前备份，
   `reorg_backup_20261008/` 为两级体系重组前备份，`gtja191_factor_dictionary.yaml` 为 GTJA191 旧字典）。
4. **warehouse 孤儿缓存**：`warehouse/` 中有 12 个 GTJA191 Alpha 因子目录（Alpha2/6/13 等）在
   主库 YAML 中无定义，来自归档字典，可在接入 GTJA191 全集时一并归档至 `momentum_price_factors`。
