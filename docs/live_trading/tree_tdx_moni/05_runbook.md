# 每日运行手册

## 收盘后

1. 运行 `scripts/training/train_tree.py`（按你的训练节奏决定是否当天重训）
2. 运行 `scripts/live/generate_tree_targets.py`
3. 检查 `runtime/live/tree/signals/daily` 是否生成目标持仓文件

## 次日开盘前

1. 确认通达信客户端已登录
2. 确认 `live_config.py` 中账户和路径正确
3. 确认 `AUTO_TRADE` 开关状态符合预期

## 次日执行

1. 在通达信模拟量化中运行 `tree_live_executor.py`
2. 先看日志是否成功读取目标文件
3. 再看是否成功获取账户资产和持仓
4. 最后检查是否生成委托结果

## 常见问题

### 1. 找不到目标持仓文件

- 检查 `runtime/live/tree/signals/daily`
- 检查 `generate_tree_targets.py` 是否执行成功

### 2. 获取不到账户句柄

- 检查 `ACCOUNT`
- 检查客户端是否已登录
- 检查是否为正确的模拟交易账户

### 3. 获取不到有效价格

- 检查客户端行情刷新状态
- 检查证券代码格式是否为 `600000.SH`

### 4. 下单无反应

- 检查 `AUTO_TRADE` 是否关闭
- 检查日志中的账户、价格、数量是否合理
