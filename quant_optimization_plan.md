# Qlibworks 机构级量化流水线深度整改计划 (基于 Quant All-Star Team)

基于 **Quant All-Star Team (量化明星团队)** 的 15 家顶尖机构的行事范式，我们对当前 `e:\Quant\Qlibworks` 项目的架构、代码逻辑及流水线（Workflow）进行了全面而深度的审查。

当前项目已经具备了非常优秀的**模块化雏形**（Data -> Factor -> ML -> Portfolio -> Backtest），但如果以华尔街顶级机构的标准来衡量，在“统计严谨性”、“业务逻辑映射”以及“微观执行”上仍存在几处致命的漏洞与可优化的空间。

以下是详细的深度整改计划。

---

## 1. [AQR 因子模型构建器] 因子中性化与共线性处理
**当前问题：**
- 在 `processors/neutralize.py` 中，你使用了普通最小二乘法（OLS, `np.linalg.lstsq`）进行行业和市值的截面中性化。
- **机构视角**：A 股市场存在大量微盘股，其财务数据和量价特征存在极大的噪音。使用等权的 OLS 会导致微盘股的极端异常值主导回归超平面，导致中性化后的残差失真。同时，未进行多重共线性（VIF/相关性）剔除就直接将大量因子送入 ML 模型。

**整改计划：**
- **改用 WLS（加权最小二乘法）**：在截面回归时，应以股票市值的平方根（$\sqrt{Market Cap}$）作为权重进行 WLS 回归，从而让中大盘股在确定行业基准时拥有更多的话语权。
- **因子共线性过滤**：在 `workflow.py` 的特征选择阶段，除了基于树模型的特征重要性过滤外，必须加入 Spearman 秩相关系数或 VIF 的过滤，剔除相关性 > 0.7 的同质化因子。

## 2. [Point72 机器学习研究员] 交叉验证与标签泄漏
**当前问题：**
- `workflow.py` 中使用了静态的切分（Train: 1-6月, Valid: 7-9月, Test: 10-12月）。
- **机构视角**：金融时间序列是非平稳的（Non-stationary）。静态切分极易导致模型在特定市场环境（如2020年疫情水牛）中过拟合。且如果在特征工程中使用了长周期的 Rolling 算子，Train 和 Valid 之间存在严重的时间重叠（数据泄漏）。

**整改计划：**
- **引入 Purged Walk-Forward Cross Validation（带净化的滚动交叉验证）**：采用类似 Marcos Lopez de Prado 提出的 Purged K-Fold，在每次切分时强制在训练集和测试集之间留出一段 Embargo（隔离期），防止自相关性导致的标签泄漏。

## 3. [Man Group 组合优化专家] 预期收益的数学映射（致命错误）
**当前问题（严重）：**
- 在 `models/portfolio.py` 中，你直接将机器学习模型的预测分 `predictions` 作为 `expected_ret` 喂给了 PyPortfolioOpt 的 `EfficientFrontier`。
- **机构视角**：这是一个严重的数学错误。ML 模型（如 LightGBM 回归）在 Qlib 默认的截面标准化体系下，输出的 `score` 往往是截面 Z-score 或者相对排序分（甚至可能在 -0.05 到 0.05 之间），它**绝对不是年化预期收益率（Annualized Expected Returns）**！将 Z-score 直接扔进马科维茨均值-方差优化器，会导致优化器得出极其荒谬的权重，或者频繁提示 "infeasible"。

**整改计划：**
- **映射到预期收益**：在将 ML Score 传入组合优化前，必须进行数学映射。常用的机构做法是使用 Grinold 的 Alpha 公式：
  $E(R) = Volatility \times IC \times Score$
  将截面标准化后的 Score，乘以个股的历史波动率，再乘以模型预估的 IC 值，转化为真实的预期超额收益。
- **或者退化为秩权重**：如果不进行均值-方差优化，应该按 Rank 排序打分，转化为纯粹的截面多空权重。

## 4. [Two Sigma / Virtu 交易执行] 微观风控与滑点模拟
**当前问题：**
- 在 `backtest/bt_strategy.py` 的 `EnhancedQlibStrategy` 中，你的止损逻辑写在 `next()` 函数里，通过 `current_price < state['stop_loss']` 来判断。
- **机构视角**：Backtrader 的 `next()` 是在**每日收盘后（Close）**触发的！这意味着如果某只股票盘中暴跌 20%，你的风控系统在盘中根本不会触发，而是只能眼睁睁看着它以收盘价结算，第二天才发出卖单。这完全失去了“止损”的意义。
- 此外，未限制成交量比例。

**整改计划：**
- **使用真正的 Stop Order**：修改 BT 策略，在建仓时同步向 Broker 发送真实的 `bt.Order.Stop`（止损单）。这样 BT 的底层撮合引擎会自动使用盘中的 `Low`（最低价）去刺穿止损价，并在盘中准确成交，真实还原市场环境。
- **成交量限制（Volume Limit）**：在发送订单时，引入 `VWAP` 或 `Volume` 限制，单次报单不得超过该股票过去 5 日日均成交量的 10%，否则由于市场冲击成本（Market Impact），回测收益在实盘中根本吃不到。

## 5. [Renaissance 回测引擎] 幸存者偏差防范
**当前问题：**
- `workflow.py` 在获取宇宙时，过滤了 `df['start_date'] <= '2020-01-02'`，但未动态剔除退市股票。
- **机构视角**：如果一只股票在 2020 年 11 月退市，而我们在回测中默认持有它，可能导致计算出错或虚高的收益。

**整改计划：**
- 需要在 Qlib 的 Dataset 提取和 BT 的数据灌入时，动态检查 `Volume == 0` 或 `NaN`，在退市或长期停牌前强制清仓。

---

## 即刻执行的修复行动 (Immediate Action)

为了确保流水线在正确的逻辑下运行，我将优先修复以下**最关键的数学与执行逻辑错误**：

1. **修复 `models/portfolio.py`**：加入 ML Score 到年化预期收益（Expected Returns）的映射逻辑（Grinold Alpha 公式转换），防止优化器崩溃或输出荒谬权重。
2. **修复 `backtest/bt_strategy.py`**：将伪止损（收盘后检查）重构为真实的 `bt.Order.Stop` 订单委托，确保能在日内被精确触发。
3. **修复 `workflow.py`**：调整参数传递以适配上述更改。