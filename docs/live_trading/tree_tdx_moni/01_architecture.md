# 架构说明

## 总体目标

将现有树模型流程拆成三层：

1. 本地研究层
2. 通达信执行层
3. 归档审计层

这样做的原因是：

- Qlib 训练、因子筛选、历史分数生成更适合留在本地 `Qlib_env`
- 通达信更适合负责账户、持仓、委托、成交
- 两边通过标准文件交互，便于复盘和替换

## 分层职责

### 1. 本地研究层

位置：

- `scripts/training/train_tree.py`
- `scripts/live/generate_tree_targets.py`

职责：

- 训练模型
- 保存筛选后的因子列表
- 基于 `score_tree.csv` 生成最新目标持仓

### 2. 通达信执行层

位置：

- `D:\chenxu\TDX_MONI\PYPlugins\user\tree_live_executor.py`
- `D:\chenxu\TDX_MONI\PYPlugins\user\tdx_trade_adapter.py`

职责：

- 获取账户句柄
- 查询总资产和当前持仓
- 读取目标持仓文件
- 计算差额单
- 先卖后买执行

### 3. 归档审计层

位置：

- `runtime/live/tree`
- `D:\chenxu\TDX_MONI\PYPlugins\user\logs`

职责：

- 保存目标文件
- 保存最新状态
- 保存执行日志
- 为后续对账和复盘留痕

## 数据流

`train_tree.py` -> `score_tree.csv` -> `generate_tree_targets.py` -> `target_positions_YYYYMMDD.csv` -> `tree_live_executor.py` -> 通达信模拟盘委托/成交日志
