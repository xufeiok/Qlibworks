# Phase 1: Vue 3 脚手架 + 主题系统 + 路由框架 + /data 页面

> REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 搭建 Vue 3 + Vite + Naive UI 前端脚手架，完成全局布局（TopNav + NavSide）和 /data 数据管理页。

**Tech Stack:** Vue 3.5 + Vite 6 + TypeScript + Naive UI + Vue Router 4 + Pinia + lightweight-charts + ECharts + FastAPI

---

## Task 1: Vite + Vue 3 项目初始化

**Create:** E:/Quant/Qlibworks/frontend/

- [ ] Step 1: 用 Vite 创建 Vue 3 + TypeScript 项目
- [ ] Step 2: 安装依赖（Naive UI, Vue Router, Pinia, ECharts, lightweight-charts 等）
- [ ] Step 3: 配置 tsconfig.json, vite.config.ts
- [ ] Step 4: 验证项目能正常 dev 启动

## Task 2: 全局布局组件

**Create:** src/layout/AppLayout.vue, src/layout/TopNav.vue, src/layout/NavSide.vue
**Create:** src/router/index.ts
**Create:** src/stores/

- [ ] Step 1: 创建 Naive UI 暗色主题配置文件
- [ ] Step 2: 实现 Vue Router 4 路由表（所有顶层路由 + Lazy Load）
- [ ] Step 3: 实现 TopNav.vue（品牌 + 导航链接 + 搜索）
- [ ] Step 4: 实现 NavSide.vue（可折叠侧边栏菜单）
- [ ] Step 5: 实现 AppLayout.vue（组合布局 + RouterView）
- [ ] Step 6: 创建 Pinia store 基础结构
- [ ] Step 7: 实现 /dashboard 占位页

## Task 3: API 对接层

**Create:** src/api/index.ts, src/api/data.ts, src/api/factors.ts, src/api/strategies.ts, src/api/backtest.ts, src/api/market.ts
**Create:** src/composables/useWebSocket.ts

- [ ] Step 1: Axios 封装（baseURL, 拦截器, 错误处理）
- [ ] Step 2: FastAPI 健康检查接口对接
- [ ] Step 3: WebSocket composable 封装
- [ ] Step 4: 数据 API 接口定义

## Task 4: /data 数据管理页面

**Create:** src/views/data/DataOverview.vue, src/views/data/DataSourceTable.vue, src/views/data/QualityReport.vue, src/views/data/ImportWizard.vue
**Create:** src/stores/dataStore.ts

- [ ] Step 1: 创建 dataStore Pinia（数据源列表、质量指标、更新状态）
- [ ] Step 2: 实现 DataSourceTable.vue（Naive n-data-table 展示）
- [ ] Step 3: 实现 QualityReport.vue（质量评分卡片 + ECharts 图表）
- [ ] Step 4: 实现 ImportWizard.vue（导入配置步骤向导）
- [ ] Step 5: 实现 DataOverview.vue（组合上述组件）
- [ ] Step 6: 对接数据 API

## Task 5: 后端 FastAPI 数据接口

**Modify:** src/qlworks/（新增前端 API 适配层）

- [ ] Step 1: 创建 FastAPI 数据接口（overview/sources/quality/import）
- [ ] Step 2: 创建 CORS 配置，允许前端跨域访问
- [ ] Step 3: 实现数据概览 API
- [ ] Step 4: 实现数据源状态 API
- [ ] Step 5: 实现数据质量报告 API

