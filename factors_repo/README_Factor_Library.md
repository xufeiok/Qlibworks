# 量化因子库系统化分类说明 (Factor Library Guide)

## 概述

为了更好地支持机器学习多因子量化研究，本因子库按照**大类划分**的思想进行了系统化升级，所有的 YAML 因子配置文件被分为 6 个独立的文件。这种分类方式不仅能够解决因子同质化、共线性的问题，也极大地方便了我们在构建模型（如 LightGBM、CatBoost、LSTM）时按需加载不同的特征组合。

因子来源结合了经典的 Alpha158、GTJA191 以及经过市场实证的各类风格、质量、风险因子。所有因子的计算公式主要使用 `Qlib` 表达式实现，部分支持 `DuckDB` 表达式。

---

## 因子大类划分

### 1. 风格因子 (`style_factors.yaml`)
反映股票在长期内相对稳定的风格偏好，主要用于定义策略风格、做风格中性或控制暴露不直接参与打分，只做框架。
*   **市值因子 (Size)**：如总市值、流通市值（越小越好 / 负向）。
*   **价值因子 (Value)**：如市盈率 (PE TTM)、市净率 (PB LF)、市销率 (PS TTM)（越低越便宜 / 负向）。
*   **红利因子 (Dividend)**：如股息率 (DV TTM)（越高越好 / 正向）。
*   **低波动因子 (Low Volatility)**：如过去20日收益率标准差（波动越小越好 / 负向）。

### 2. 质量因子 (`quality_factors.yaml`)
反映公司的基本面财务健康状况和盈利能力，用于排雷、过滤垃圾股、提升策略夏普比率，通常适合小市值策略必备，完全不与风格重复。
*   **盈利能力**：ROE (TTM)、ROA (TTM)、毛利率 (Gross Profit Margin)、净利率 (Net Profit Margin)（正向）。
*   **现金流质量**：盈利现金流含量 (OCF / Net Profit)（正向）。
*   **成长能力**：营收同比 (TR YoY)、净利润同比 (Net Profit YoY)、扣非净利润同比（正向）。
*   **资产负债结构**：资产负债率 (Debt to Assets，负向)、流动比率 (Current Ratio，正向)。

### 3. 价量因子 (`price_volume_factors.yaml`)
包含市场最常用的量价博弈信号，主要用于抓趋势、抓反转、抓资金。此分类融合了 **Alpha158** 和 **国泰君安191 (GTJA191)** 的核心特征。
*   **反转与动量**：1月反转（跌越多越容易反弹 / 负向）、短期动量。
*   **流动性与换手**：日均换手率（Turnover Rate）、非流动性 Amihud 指标。
*   **技术面偏离**：均线偏离 (MA Cross)、OBV 能量潮、K线特征 (KLEN, KMID 等)。
*   **量价相关性**：GTJA191 因子中的各类短周期相关系数。

### 4. 情绪因子 (`sentiment_factors.yaml`)
基于分析师预期、外资或杠杆资金行为构建的特征，用于抓预期差、抓拐点，通常能提前于财报发现价格异动。
*   **一致预期**：预期净利润增速（正向）。
*   **资金面流向**：北向资金持股变动（正向）。
*   **杠杆资金**：融资余额变化（正向）。

### 5. 风险因子 (`risk_factors.yaml`)
主要用于策略的风控、止损、过滤庄股以及防暴雷，往往在截面打分或者股票初筛中作为硬性阈值条件。
*   **波动率风险**：Beta、20日波动率。
*   **微观筹码风险**：股东人数变化（环比减少代表筹码集中）。
*   **事件性风险**：解禁压力（未来30日解禁市值/总市值）、质押比例（Pledge Ratio）。

### 6. 其他因子 (`other_factors.yaml`)
尚未被归入以上 5 大类的补充性因子或者定制化另类数据因子。

---

## 在 Qlibworks 框架中的使用方法

你可以直接在 `features/builder.py` 或在策略的 `workflow.py` 中传入单个或多个类别的文件名来合并特征。

```python
from qlworks.features import build_factor_library_bundle

# 1. 仅加载质量因子（做基本面选股）
quality_bundle = build_factor_library_bundle("quality_factors")

# 2. 混合加载（量价 + 质量 + 情绪）送给机器学习模型
ml_bundle = build_factor_library_bundle([
    "price_volume_factors",
    "quality_factors",
    "sentiment_factors"
])

# 打印生成的特征名
print(ml_bundle.names)
```

## 注意事项

1.  **因子正负向**：部分因子（如市值、估值、反转）在传统多因子打分中为负向因子。当输入到基于树的机器学习模型（如 LightGBM）时，模型能自动捕捉非线性与单调性，因此无需手动取倒数或加负号；但在 Barra 线性模型中，需要注意在预处理时翻转其方向。
2.  **避免未来函数**：财务类因子（质量因子）请确保底层的 Qlib 特征数据已做滞后处理（例如使用 TTM 或 Point-in-Time 数据），切勿直接使用财报发布当期的绝对日期对齐。
3.  **存档目录**：旧版的杂乱因子字典（如 `master_factor_dictionary.yaml`、`gtja191_factor_dictionary.yaml`）已经被移动至 `factors_repo/archive/` 目录进行备份，以防加载冲突。
