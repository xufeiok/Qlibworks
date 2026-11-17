# QlibWorks Quant Studio - 前端系统设计规范

## 概述
基于 Vue 3 + Naive UI + TradingView lightweight-charts 的一站式量化研究前端系统。覆盖数据、因子、策略、回测、市场五大管理域，各自独立为 Vue Router 子页面。

## 技术栈
- **前端框架**: Vue 3.5 + Vite 6 + TypeScript
- **UI 组件库**: Naive UI (原生暗色主题)
- **K线图表**: TradingView lightweight-charts
- **统计图表**: ECharts + Vue-ECharts
- **路由**: Vue Router 4 (Lazy Load + KeepAlive)
- **状态管理**: Pinia + TanStack Query
- **后端**: FastAPI + WebSocket
- **测试**: Vitest + Playwright

## 路由设计
| 路径 | 页面 | 说明 |
|---|---|---|
| / | Dashboard | 首页看板 |
| /data | 数据管理 | 数据源/质量/导入 |
| /data/sources | 数据源管理 | DuckDB/CH/CSV |
| /data/quality | 质量报告 | 数据质量仪表盘 |
| /factors | 因子管理 | 因子库/IC/合成 |
| /factors/editor/:id | 因子编辑器 | Monaco Editor |
| /factors/ic-analysis | IC 分析 | ECharts 面板 |
| /factors/synthesis | 因子合成 | 合成工作台 |
| /strategies | 策略管理 | 训练/调优/评估 |
| /strategies/train | 模型训练 | 训练工作台 |
| /strategies/tune | Optuna 调优 | 调优面板 |
| /strategies/evaluate | 模型评估 | 对比面板 |
| /backtest | 回测管理 | 运行/结果/对比 |
| /backtest/new | 新建回测 | 配置页 |
| /backtest/:id | 回测结果 | SuperPlot 视图 |
| /backtest/compare | 多策略对比 | 对比面板 |
| /backtest/portfolio | 组合优化 | PyPortfolioOpt |
| /market | 市场行情 | K线/板块/选股 |
| /market/stock/:code | 个股分析 | 深度分析 |
| /market/screener | 选股器 | 多因子筛选 |
| /market/sentiment | 情绪仪表盘 | 市场情绪 |
| /market/ai-report | AI 研报 | 日报系统 |

## 实施路线图
### Phase 1: 脚手架 + 主题 + 路由 + /data 页面
- Vite 6 + Vue 3 + TypeScript 初始化
- Naive UI 暗色主题配置
- Vue Router 4 路由框架搭建
- Pinia store 基础结构
- TopNav + NavSide 布局组件
- /data 页面: 数据总览 + 数据源管理 + 质量报告
- Axios 封装 + FastAPI 数据 API 对接

### Phase 2: /factors 因子管理
- YAML 因子库浏览器
- Monaco Editor 因子编辑器
- ECharts IC 分析面板
- 因子合成工作台

### Phase 3: /strategies 策略管理
- 模型训练工作台
- WebSocket 训练进度
- Optuna 调优面板
- 模型评估对比

### Phase 4: /backtest 回测管理
- 回测运行配置器
- SuperPlot 指标仪表盘
- 多策略对比
- 参数扫描 + Walk-Forward
- 组合优化

### Phase 5: /market 市场行情
- lightweight-charts K线组件
- 板块分析
- 选股器
- 情绪仪表盘
- AI 研报集成

## 设计原则
1. 每个模块独立为 Vue Router 子页面，页面级 KeepAlive
2. 页面内子路由作为 Tab 切换
3. Pinia store 跨页面共享，模块内状态由页面自身管理
4. WebSocket 统一通过 composable useWebSocket 管理
5. Naive UI 暗色主题作为全局设计语言基础
