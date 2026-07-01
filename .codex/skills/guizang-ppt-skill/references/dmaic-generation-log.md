# DMAIC 循环 PPT 生成 · 工作范例

**源文件**: `/vol1/1000/KnowledgeBase/营销管理知识库/DMAIC循环.md`（97行）
**输出**: `/vol1/1000/Mydraft/DMAIC循环培训/index.html`（593行，12页）
**主题**: 🖋 墨水经典

---

## 内容结构 → 页面规划

源文件包含以下模块：
- 关键词（5个）
- 核心定义（D → M → A → I → C 五阶段概述 + 注意事项 + 适用场景）
- 案例1：制造业降低屏幕不良率（D/M/A/I/C 各一段）
- 案例2：服务业缩短排队时间（D/M/A/I/C 各一段）
- 价值意义（3点）
- 学习及实践方式步骤（5个阶段 × 工具 + 行动）
- 主要工具（4个）
- 延展书籍/网站（4个）

**映射到12页 PPT 的结构**：

| # | 布局 | 对应源内容 |
|---|------|-----------|
| 01 | hero dark · 封面 | 标题 DMAIC + 副标题 |
| 02 | light · 数据大字报 | 核心定义页 → 6个 stat-card（D/M/A/I/C + ∞循环） |
| 03 | hero light · 章节幕封 | 五阶段概览 |
| 04 | light · 左文右图 | Define → SIPOC/VOC/项目章程 |
| 05 | dark · 左文右图 | Measure → MSA/流程图/CpCpk |
| 06 | light · 左文右图 | Analyze → 鱼骨图/假设检验/帕累托图 |
| 07 | dark · 左文右图 | Improve → DOE/FMEA/成本效益分析 |
| 08 | light · 左文右图 | Control → SPC/标准化作业/防错设计 |
| 09 | dark · Pipeline | 案例1 → 5步横排 pipeline + 成果数字 |
| 10 | light · Pipeline | 案例2 → 5步横排 pipeline + 成果数字 |
| 11 | hero dark · 大字报 | 三大核心价值 |
| 12 | hero dark · 收束 | 行动建议 |

---

## 主题色选择逻辑

**规则**: 5套预设只能选一，不允许自定义。

本次选 🖋 墨水经典（`--ink:#0a0a0b; --paper:#f1efea;`），理由：
- 培训课件属于**通用/商业发布**场景
- 墨水经典是最安全的默认选择

---

## 预检类清单（对照 template.html 验证）

生成前必须确认以下类在 `assets/template.html` 的 `<style>` 中存在：

**字体/布局类**：
- `h-hero`, `h-xl`, `h-sub`, `h-md`
- `lead`, `kicker`, `meta-row`
- `frame`, `col`

**数字卡片**：
- `stat-card`, `stat-label`, `stat-nb`, `stat-unit`, `stat-note`

**流水线**：
- `pipeline-section`, `pipeline-label`, `pipeline`, `step`, `step-nb`, `step-title`, `step-desc`

**网格**：
- `grid-2-7-5`, `grid-2-6-6`, `grid-2-8-4`, `grid-3-3`, `grid-4`, `grid-6`, `grid-3`

**组件**：
- `callout`, `callout-src`, `pillar`, `.t`, `.d`
- `frame-img`, `.r-16x10`, `.r-4x3`, `.h-22`, `.h-26`
- `img-cap`, `chrome`, `foot`

**动效**：
- `data-anim`, `data-anim="left"`, `data-anim="right"`, `data-anim="step"`
- `data-animate="pipeline"`

---

## 页面主题节奏

```
01 hero dark  ← 封面
02 light       ← 数据大字报
03 hero light  ← 章节幕封
04 light       ← Define
05 dark        ← Measure
06 light       ← Analyze
07 dark        ← Improve
08 light       ← Control
09 dark        ← 案例1
10 light       ← 案例2
11 hero dark   ← 价值意义
12 hero dark   ← 收束
```

**检查**：
- ✅ 连续3页以上无重复主题
- ✅ 8页以上有 ≥1 hero dark + ≥1 hero light
- ✅ 有 dark 正文页（05/07/09）穿插在 light 页之间

---

## 生成的教训/注意点

1. **数字太长要缩写**：例如"8%"写成`0.5<span class="stat-unit">%</span>`，避免 stat-nb 溢出
2. **Pipeline 每页只放一个案例**：案例1和案例2分两页，每页5步横排
3. **案例页用 pipeline layout**：比左文右图更清晰地展示 D→M→A→I→C 五步顺序
4. **收束页用 hero dark**：留白多、放一行行动建议即可

---

## 快速复制模板命令

```bash
mkdir -p "/vol1/1000/Mydraft/项目名/images"
cp "/vol1/1000/Github/skills/guizang-ppt-skill/assets/template.html" "/vol1/1000/Mydraft/项目名/index.html"
# 然后在 index.html 的 <div id="deck"> 里替换 <!-- SLIDES_HERE -->
```
