# 量化全明星团队 (Quant All-Star Team) 知识库

## 核心原则与教训 (Lessons Learned)

本知识库记录了在 A 股机器学习量化实战中，由不同流派量化专家总结出的核心经验。这些经验特别针对 A 股市场（T+1、高波动、长熊市）以及底层回测框架（Qlib + Backtrader）进行了优化。

### 1. [Renaissance Backtest] 消除幸存者偏差与前视偏差
- **动态股票池 vs 静态列表**：绝不使用静态的股票代码列表（如固定的 `load_csi500_instruments`），这会将未来才纳入的股票或已经退市的股票混入历史训练中。**解决方案**：在 Qlib 中必须使用内置的动态别名（如 `"csi500"`），框架会自动在每一天过滤出合法的股票。
- **Label 错位防范**：如果你是“每周一开盘买入，下周一收盘/开盘卖出”，那么预测标签绝对不能是简单的 `Ref($close, -5) / $close - 1`，因为你买不到上一个交易日的收盘价，会吃到巨大的日内跳空假收益。**正确写法**：`Ref($close, -5) / Ref($open, -1) - 1`（昨天收盘预测，今天开盘买入，持有5天）。

### 2. [Citadel Alpha Lab] 横截面相对强弱 (Cross-Sectional Ranking)
- **绝对收益在熊市的失效**：在 A 股的大级别熊市（如 2023-2024），大盘泥沙俱下，所有股票的绝对收益率都是负数。如果模型预测绝对收益，将无法区分“谁跌得少”。
- **解决方案**：必须将 Label 转换为截面百分位排名（CSRank）。在 Qlib 的 `learn_processors` 中加入 `CSQuantileNorm`（针对 label），迫使树模型去学习个股在同一天的“相对强弱”。

### 3. [Two Sigma Risk] 高频风控与低频建仓的解耦
- **双面打脸陷阱 (Whipsaw)**：如果是“周频换仓”策略，千万**不要**在每天去检查模型的“得分是否恶化”并强制平仓。因为模型是对 5 天后预测，单日特征的波动（噪音）会导致得分骤降被洗盘洗出局，随后股票反弹，而你的资金却因为“每周只建仓一次”而闲置（Cash Drag）。
- **解耦设计**：
  - **建仓/换仓**：完全交由模型得分（低频）。
  - **日内风控**：完全交由价格行为（如 ATR 追踪止损、单票固定 8% 止损）。不要让高频的预测噪音干扰长周期的持仓。

### 4. [Man Group Portfolio] 多头 Beta 敞口与择时
- **死多头 (Long-only) 的局限**：即便选股（Alpha）能力再强，如果仓位长期保持 95%，在熊市中也难逃 30% 以上的回撤。
- **下一步破局点**：必须引入 **Alpha-Beta 分离** 或 **大盘择时模块 (Market Timing)**。当宏观风控指标（如沪深300趋势）走坏时，强制降低 `buy_pct`（总仓位）至 20% 甚至空仓。

### 5. [Bloomberg Data Pipeline] 内存与计算性能优化
- **Walk-Forward 内存溢出 (OOM)**：在滚动回测中，每次循环都会生成庞大的 DataFrame（包含几百个特征），如果不清理，极易导致 RAM 或 VRAM 溢出。
- **规范写法**：在每次循环的末尾，必须显式调用：
  ```python
  del dataset, train_frame, models
  import gc
  gc.collect()
  ```

### 6. [Virtu Execution] 交易摩擦与 A 股合规
- **滑点与印花税**：A 股目前印花税为单边（卖出）千分之一，佣金双边万分之三。回测必须将此计入。
- **整手限制**：必须强制 `size` 为 100 的整数倍。使用 `target_shares = int((target_value / close_price) // 100 * 100)` 而非 Backtrader 自带的百分比下单，防止产生畸形持仓。

### 7. [Infrastructure] 基础设施
- **GitHub 网络阻断**：国内直连 GitHub 经常遇到端口 443 Reset。**规范做法**：使用 SSH 协议 (`git@github.com:User/Repo.git`) 绕过 HTTPS 干扰。
