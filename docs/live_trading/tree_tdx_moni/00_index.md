# tree_tdx_moni 文档索引

## 目录

- `01_architecture.md`
  - 本地研究层、通达信执行层、归档审计层的职责划分
- `02_file_contracts.md`
  - 目标持仓文件、状态文件、日志文件字段约定
- `03_strategy_rules.md`
  - 选股、调仓、执行口径
- `04_risk_controls.md`
  - 模拟盘首版风控与熔断规则
- `05_runbook.md`
  - 每日运行步骤与排障要点

## 代码入口

- 本地目标持仓生成:
  - `scripts/live/generate_tree_targets.py`
- 回测基线:
  - `scripts/backtest/tree.py`
- 通达信模拟盘执行:
  - `D:\chenxu\TDX_MONI\PYPlugins\user\tree_live_executor.py`

## 运行产物目录

- `runtime/live/tree/signals/daily`
- `runtime/live/tree/signals/archive`
- `runtime/live/tree/state`
- `D:\chenxu\TDX_MONI\PYPlugins\user\logs`
