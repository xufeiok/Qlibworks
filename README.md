# Qlibworks

A 股多因子量化研究工作区：以 Qlib 为数据/回测内核，DuckDB + Parquet 承载因子仓库，
覆盖 **因子挖掘 → 家族化筛选评测 → 模型训练 → 回测 → 实盘目标生成** 的完整流水线。

## 技术栈

- 语言 / 环境：Python 3.10+（本项目在 `Qlib_env` 虚拟环境运行）
- 数据内核：pyqlib（bin 格式）、ClickHouse（源库）、DuckDB + Parquet（因子仓库/缓存）
- 模型：LightGBM / XGBoost / CatBoost（集成树模型）、线性（ICIR 加权）
- 回测：Qlib Backtest + 自研 Backtrader 适配层
- 前端演示：Vue 3 + Vite（`frontend/`，独立子系统）

## 目录结构

```
Qlibworks/
├── src/qlworks/          # 库代码（src-layout，可复用逻辑）
│   ├── evaluation/       #   因子评测核心（factor_stats/deep_eval/runner/selector/...）
│   ├── features/         #   特征构建与因子缓存（builder/dataset/factor_cache）
│   ├── factors/          #   因子库管理、合成、注册表视图（registry）
│   ├── models/           #   训练/选择/调参/归因
│   ├── backtest/         #   回测引擎与策略（bt_runner/bt_strategy/industry）
│   ├── data/             #   数据同步（ClickHouse/TDX → Qlib，api）
│   ├── live/             #   实盘目标生成与策略配置
│   ├── processors/       #   因子处理（中性化/正交化/分位标准化）
│   └── core/             #   因子注册表查询（FactorRegistry）
├── scripts/              # 命令行入口（薄层）
│   ├── training/         #   筛选与训练：factor_screen.py（唯一评测入口）、train_* 系列
│   ├── evaluation/       #   批量编排 / 回填 / 生命周期审计
│   ├── backtest/         #   回测脚本（tree/linear/icir/ensemble）
│   ├── data/             #   数据同步与修复
│   ├── live/             #   实盘目标生成
│   ├── registry/         #   元数据升级
│   └── _archive/         #   已废弃脚本归档（保留历史，不参与运行）
├── tests/                # 单元测试（镜像 src 分层）
├── tools/                # 一次性运维工具（数据恢复 / 因子管理）
├── docs/                 # 设计文档、方法论、核查报告
├── outputs/              # 训练/回测成果输出（outputs/<脚本名>/<模式>/，已 gitignore，可重新生成）
├── factor_data/          # 因子数据（factor_library / factor_sets / raw；仓库/缓存已 gitignore）
├── demos/                # 组件用法示例
└── frontend/             # Vue 前端子系统
```

## 环境准备

```powershell
# 安装为可编辑包（库代码位于 src/）
pip install -e .
# 或使用项目指定的解释器
E:\Conda_env\Qlib_env\python.exe -m pip install -e .
```

需要配置的环境变量（切勿写入仓库）：`TUSHARE_TOKEN`、`CH_HOST` / `CH_PORT` / `CH_USER` / `CH_PASSWORD` / `CH_DATABASE` 等，统一在 `.env` 中维护（已被 `.gitignore` 忽略）。

## 核心运行入口

| 环节 | 入口 | 说明 |
|---|---|---|
| 数据同步 | `python -m scripts.data.sync auto` | ClickHouse → Qlib 内核数据 |
| 因子筛选 / 深评 / 准入 | `python scripts/training/factor_screen.py [--deep-eval] [--admit]` | 唯一评测入口，家族配置驱动 |
| 树模型训练 | `python scripts/training/train_tree-doubao.py` | LGB + XGB + CatBoost 集成 |
| 线性训练 | `python scripts/training/train_linear.py` / `train_icir.py` | ICIR 加权线性 |
| 批量评测 | `python scripts/evaluation/batch_eval_4.py` | 多因子批量驱动 |
| 回测 | `python scripts/backtest/tree.py`（另有 `linear.py` / `icir` / `ensemble`） | Qlib Backtest |
| 实盘目标 | `python scripts/live/generate_tree_targets.py` | 生成每日目标持仓 |

## 测试

```powershell
python -m pytest tests -q
```

（测试依赖 `pytest`；若目标环境未安装，可先 `pip install pytest`。）

## 数据与运行时产物

- 版本库只保存**代码、配置与因子定义（yaml/json）**；
- 因子仓库（`factor_data/warehouse|cache|registry|reports|qualified_factors`）、
  Qlib 数据（`qlib_data/`）、运行产物（`runtime/`、`mlruns/`、`bt_output/`）均已在
  `.gitignore` 中排除，可本地重建，不入版本库。
