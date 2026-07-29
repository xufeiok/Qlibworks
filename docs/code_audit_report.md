# 代码审核报告

> 审核时间：2026-06-18
> 审核范围：`train_tree.py`, `tree_label_test.py`, `bt_runner.py`, `bt_strategy_label.py`, `industry.py`, `sync_qlib_direct.py`

---

## 目录

- [P0 - 严重问题](#p0---严重问题)
  - [Bug 1: `instruments: "csi500"` 完全无效](#bug-1-instruments-csi500-完全无效)
  - [Bug 2: 回测中 ~200 只股票被静默跳过](#bug-2-回测中-200-只股票被静默跳过)
- [P1 - 重要问题](#p1---重要问题)
  - [Bug 3: 行业约束 PIT 映射是「伪 PIT」](#bug-3-行业约束-pit-映射是伪-pit)
  - [Bug 4: 前复权价格 + 前向填充 = 潜在的虚假收益](#bug-4-前复权价格--前向填充--潜在的虚假收益)
  - [Bug 5: try-except 吞掉验证集错误](#bug-5-try-except-吞掉验证集错误)
- [P2 - 次要问题](#p2---次要问题)
  - [Issue 6: 所有股票退市日标记为 9999-12-31](#issue-6-所有股票退市日标记为-9999-12-31)
  - [Issue 7: Order.Close 与实际标签公式不完全匹配](#issue-7-orderclose-与实际标签公式不完全匹配)
  - [Issue 8: reverse_test 排序逻辑有歧义](#issue-8-reverse_test-排序逻辑有歧义)
- [修复建议汇总](#修复建议汇总)

---

## P0 - 严重问题

### Bug 1: `instruments: "csi500"` 完全无效

**位置：** [`train_tree.py#L48`](file:///c:/xfworks/Qlibworks/scripts/training/train_tree.py#L48)

```python
"instruments": "csi500",
```

**现象：**

- `qlib.data.D.instruments("csi500")` 返回 **2 只** 股票（而非 CSI500 的 ~500 只成分股）
- 训练实际使用的股票池是全量数据（**每天 700+ 只**）
- 实际 score_tree.csv 中共有 **883 只唯一股票**，远超 CSI500 的范围

**根因：**

`qlib_data/instruments/csi500.txt` 格式遵循 Qlib 标准（每行 `code\tentry_date\texit_date`），但当前 Qlib 的 `DataService` 未正确解析该动态池格式。默认只识别 `all.txt` 这种静态池。

**影响：**

配置语义与实际情况完全脱节。虽不影响训练结果（模型仍然在全量数据上工作），但：
- 如果有意限制为 CSI500 成分股进行回测，当前的 883 只选股池远大于期望范围
- 新人阅读代码会产生严重误导
- 无法复现「CSI500 内的选股」场景

**修复：**

```python
# 方案一：与实际情况一致
"instruments": "all",

# 方案二：修复 csi500.txt 解析（需修改 instrument provider）
```

---

### Bug 2: 回测中 ~200 只股票被静默跳过

**位置：** [`tree_label_test.py#L105-L116`](file:///c:/xfworks/Qlibworks/scripts/backtest/tree_label_test.py#L105-L116)

```python
price_data = D.features(instruments, [...], ...)
price_dict = {}
for inst in instruments:
    if inst not in price_data.index.get_level_values("instrument"):
        continue  # ← 静默跳过，无日志
```

**现象：**

- 行业约束后 702 只候选股票
- `D.features()` 仅返回 **489 只**（缺失 213 只，占 30%）
- 没有任何警告或日志说明哪些股票缺失及缺失原因

**根因：**

缺失原因可能有多种，但代码完全放弃了异常处理和信息提示：

1. **大小写不匹配**：Qlib features 目录是小写（`000001.sz`），而 score 中是大写（`000001.SZ`）
2. **数据未同步**：部分股票在 qlib_data 中无对应 `.day.bin` 文件
3. **后上市股票**：2023 年之后才上市的股票，在 D.features 中受限

**影响：**

回测实际可用股票池比预期小 30%。如果缺失的 213 只股票中包含较多低收益股，回测结果会被**高估**（天然过滤了烂票）；反之如果高收益股被跳过，结果会被**低估**。无论哪种情况，结果都不可信任。

**修复：**

```python
avail = set(price_data.index.get_level_values("instrument"))
missing_count = len(instruments) - len(avail)
if missing_count > 0:
    missing_list = [s for s in instruments if s not in avail]
    print(f"  *** 警告: {missing_count}/{len(instruments)} 只股票无行情数据！***")
    print(f"  *** 缺失股票样本: {missing_list[:10]}")
```

---

## P1 - 重要问题

### Bug 3: 行业约束 PIT 映射是「伪 PIT」

**位置：** [`industry.py#L91-L115`](file:///c:/xfworks/Qlibworks/src/qlworks/backtest/industry.py#L91-L115)、[`sync_qlib_direct.py`](file:///c:/xfworks/Qlibworks/scripts/data/sync_qlib_direct.py) 行业写入部分

```python
# industry.py - 看似每年取不同快照
for snap_dt in yearly_snapshots:
    ref = snap_dt.strftime('%Y-%m-%d')
    imap = load_industry_map(instruments, ref)  # 但从 bin 文件读，所有年份值相同

# sync_qlib_direct.py - 行业写入时，全时间序列用同一个值
arr = np.full(len(calendar_list), float(val), dtype=np.float32)
write_bin(stock_dir / f"sw_{level}.day.bin", 0, arr)
```

**现象：**

代码`load_industry_maps_pit()` 每年初加载一次行业快照，表面上实现了 PIT 行业分类。但底层 `sw_l1/sw_l2/sw_l3` bin 文件是在同步时用**当前快照**一次性写入的——2020 年和 2025 年读到的行业分类完全相同。

**根因：**

`sync_qlib_direct.py` 从 ClickHouse 查询 `sw_industry_members` 表时，只取了当前最新的行业分类，没有按年份存储历史行业分类。

```python
ind_df = client.query_df("""
    SELECT DISTINCT l1_code, l1_name, l2_code, l2_name, l3_code, l3_name
    FROM sw_industry_members   -- 没有时间过滤，只返回当前行业
""")
```

**影响：**

如果一只股票在回测期内换过行业（如 2022 年从「房地产」改为「综合」），2020~2021 年的行业约束会错误地使用新行业，导致行业过度集中约束出错。不过 A 股换行业不频繁，实际影响有限。

**修复：**

```python
# ClickHouse 中如果 sw_industry_members 有历史版本，按年份分批查询写入
# 或者使用股票范围限定，只取当前年份的行业映射
for year in range(2020, 2027):
    df_year = client.query_df(f"""
        SELECT ... FROM sw_industry_members
        WHERE ... AND start_date <= '{year}-01-01' 
          AND (end_date >= '{year}-01-01' OR end_date IS NULL)
    """)
```

---

### Bug 4: 前复权价格 + 前向填充 = 潜在的虚假收益

**位置：** [`bt_runner.py#L168-L175`](file:///c:/xfworks/Qlibworks/src/qlworks/backtest/bt_runner.py#L168-L175)

```python
df.loc[invalid_mask, ['open', 'high', 'low', 'close']] = np.nan
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill().bfill()
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(1.0)
```

**现象：**

当股票停牌时：
1. volume=0 触发 `invalid_mask`
2. 当日价格被置为 NaN
3. `ffill()` 将前一日价格填充到停牌日
4. 第三行 `fillna(1.0)` 兜底，极少触发

**问题在于前复权机制：**

当前数据源 `FORCE_ADJUSTED_PRICES=true` 强制使用前复权价格。前复权会修改历史价格来反映分红送转——这意味着**历史价格会随未来事件改变**。例如：

```
T+1: 买入，价格 10.00
T+3: 除权除息（10 送 10）
T+5: 前复权后，T+1 的价格被改为 5.00
```

如果用前复权价格计算持仓收益：

```
(T+5 收盘 5.50) / (T+1 复权后开盘 5.00) - 1 = +10%
但实际现金 + 送股的真实收益是 -5% + 送股 = 正收益但不同
```

在 5 天短周期下偏差较小，但频繁分红（如每年 >5% 股息率的银行股）会累积系统性的虚假收益。

**修复：**

```python
# 方案一：使用后复权或不复权数据
# 方案二：记录实际买入价，计算真实 P&L
# 方案三：在回测中自行处理复权因子，不依赖 Qlib 的前复权
```

---

### Bug 5: try-except 吞掉验证集错误

**位置：** [`train_tree.py#L298-L300`](file:///c:/xfworks/Qlibworks/scripts/training/train_tree.py#L298-L300)

```python
valid_frame = None
try:
    valid_frame = dataset_sub.prepare("valid")
except Exception:
    pass  # ← 任何错误被静默吞掉
```

**现象：**

如果 `prepare("valid")` 失败（数据缺失、内存不足、日期范围不匹配等任何原因），脚本完全不知情，继续往下执行。这意味着模型可能在**没有验证集**的情况下训练。

**影响：**

- LightGBM/XGBoost 的 `early_stopping_rounds` 参数没有验证集可用，模型无法早停，可能过拟合
- 验证集指标数据丢失，无法判断模型的泛化能力
- 训练过程不会报错，隐蔽性极强

**修复：**

```python
import logging
logger = logging.getLogger(__name__)

try:
    valid_frame = dataset_sub.prepare("valid")
    logger.info(f"验证集加载成功: {valid_frame.shape}")
except Exception as e:
    logger.warning(f"验证集加载失败: {e}，将跳过早停")
    valid_frame = None
```

---

## P2 - 次要问题

### Issue 6: 所有股票退市日标记为 9999-12-31

**位置：** [`sync_qlib_direct.py`](file:///c:/xfworks/Qlibworks/scripts/data/sync_qlib_direct.py) instruments 写入

```python
f.write(f"{s}\t{ld_str}\t9999-12-31\n")
```

**影响：**

已退市股票在 `all.txt` 中仍然标记为可交易。回测时这些股票如果被策略选中，Qlib 允许在退市日期之后仍返回价格数据（前复权保留历史价），但实际上这些股票在退市后已无法交易。

### Issue 7: Order.Close 与实际标签公式不完全匹配

**位置：** [`bt_strategy_label.py#L43`](file:///c:/xfworks/Qlibworks/scripts/backtest/bt_strategy_label.py#L43)

```python
self.sell(data=data, size=pos.size, exectype=bt.Order.Close)
```

训练标签是 `Ref($close, -5) / Ref($open, -1) - 1`：
- 买入价：T+1 的 **开盘价**
- 卖出价：T+5 的 **收盘价**

但策略用 `bt.Order.Close` 卖出，这会在当日**收盘时**以**收盘价**成交。如果 T+5 当日的开盘价和收盘价不同（日内波动），回测收益与标签收益就会产生偏差。

### Issue 8: reverse_test 排序逻辑有歧义

**位置：** [`bt_strategy_label.py#L67-L72`](file:///c:/xfworks/Qlibworks/scripts/backtest/bt_strategy_label.py#L67-L72)

```python
candidates.sort(key=lambda x: x[0], reverse=not self.p.reverse_test)
```

`industry.py` 中也有反向排序逻辑：

```python
group = group.sort_values('score', ascending=reverse_test)
```

行业约束和策略本身各有一套排序逻辑。如果两者不同步（一人打开 `reverse_test`，另一人忘记），会导致排序矛盾。

---

## 修复建议汇总

| 优先级 | 问题 | 修复内容 | 难度 | 对结果影响 |
|--------|------|----------|------|-----------|
| **P0** | `instruments: "csi500"` | 改为 `"all"` | 低 | 语义正确性 |
| **P0** | 回测缺失股票静默跳过 | 加缺失日志 | 低 | 结果可信度 |
| **P1** | 行业 PIT 伪 PIT | ClickHouse 分批查询历史行业 | 中 | 行业约束准确性 |
| **P1** | 前复权 + ffill 偏差 | 改用不复权/自行处理复权 | 中 | 收益率大小 |
| **P1** | try-catch 吞验证错误 | 加 logging/警告 | 低 | 训练鲁棒性 |
| **P2** | 退市日 9999-12-31 | 从数据库同步真实退市日 | 低 | 可交易池准确性 |
| **P2** | Order.Close vs 标签 | 策略改用 Market 订单 | 低 | 收益计算匹配度 |
| **P2** | reverse_test 歧义 | 统一排序到 bt_runner | 低 | 反向测试正确性 |

---

## 影响评估总结

### 回测结果偏高因素

1. **前复权 + ffill**（P1，中等影响）：短周期下影响较小，方向上**高估**收益
2. **退市日 9999-12-31**（P2，低影响）：退市股仍可交易，但分数低，极少被选中

### 回测结果偏低因素

1. **~200 只股票静默跳过**（P0，重大影响）：回测池从 702 缩减到 489，如果跳过的高分股无法买入，**低估**收益
2. **Order.Close vs 标签**（P2，低影响）：日内波动造成的偏差可正可负

### 当前回测 2932% 的可信度

综合考虑上述问题，在修复前该结果应被视为**受控实验下的上界估计**，而非预期收益率。主要价值在于验证了**模型在受约束条件下有正向预测力**，而非实际交易中能获得等额回报。
