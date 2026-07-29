# 文件字段契约

## 1. 目标持仓文件

文件名：

- `target_positions_YYYYMMDD.csv`

目录：

- `runtime/live/tree/signals/daily`

字段：

- `trade_date`：目标交易日
- `instrument`：Qlib 格式证券代码，如 `600000.sh`
- `score`：截面排序分数
- `raw_score`：模型原始分数
- `target_weight`：目标权重
- `rank`：当天排序名次

## 2. 最新状态文件

文件名：

- `latest_target.json`

目录：

- `runtime/live/tree/state`

字段：

- `model_name`
- `trade_date`
- `signal_file`
- `signal_count`
- `industry_neutral`
- `top_k`
- `score_threshold`
- `buy_pct`

## 3. 通达信执行日志

文件名：

- `tree_live_executor_YYYYMMDD.log`

目录：

- `D:\chenxu\TDX_MONI\PYPlugins\user\logs`

内容：

- 加载的目标文件
- 自动交易开关状态
- 资产快照
- 计划单条数
- 每笔下单结果
