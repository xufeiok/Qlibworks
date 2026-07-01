---
name: xiaohongshu-creator
description: 小红书（Xiaohongshu/RED）全能创作者技能。搜索笔记、AI创意文案、AI生图（封面+配图）、选图发布、定时发布、违禁词审核、多账号管理。支持从 FnOS/夸克网盘提取图片，发布前必须经用户确认。
category: social-media
tags: [xiaohongshu, social-media, content-creation, mcp, image-generation, automation]
---

# 小红书创作者技能（xiaohongshu-creator）

## 概述

全能小红书运营技能，覆盖从内容创作到发布的全流程：

```
创意文案 → AI生图/选图 → 违禁词审核 → 用户确认 → 发布/定时发布
```

## 依赖组件

| 组件 | 用途 | 安装方式 |
|------|------|----------|
| **xpzouying/xiaohongshu-mcp** (13.6K★) | MCP Server：登录、搜索、发帖、评论、点赞 | Docker 或 Go 编译 |
| **xhs-cli** (npm) | CLI 工具：多账号会话、人机确认发帖 | `npm install -g xhs-cli` |
| **ComfyUI** (可选) | AI 生图 | 已有 |
| **Hermes image_generate** | 生图兜底 | 内置 |

> 首次使用先运行 `/vol1/1000/Github/skills/xiaohongshu-creator/scripts/setup.sh`

## 核心工作流程

### 流程总览

```
用户指令
  │
  ├─ "搜/查XX" → 搜索笔记（MCP search）
  │
  ├─ "发一篇XX" →
  │    1. content_creator.py  → AI生成标题+正文+标签+生图提示词
  │    2. 违禁词审核 (内置库+LLM)
  │    3. 生图/选图：
  │       ├─ 有图需求 → image_generator.py (ComfyUI/image_gen工具) 生封面+配图
  │       └─ 无图需求  → 纯文字笔记
  │    4. 呈现给用户确认
  │    5. 确认后 → publisher.py 发布
  │
  └─ "定时发XX" →
      1. 同发布流程生成内容
      2. 确认后 → scheduler.py + Hermes cron 设定定时
```

### 三步工作法

**第一步：AI 创意**
```bash
python scripts/content_creator.py --topic "成都楼市2026趋势" --template realestate
```
输出：标题、正文、标签、生图提示词、违禁词审核结果

**第二步：生图 / 选图（可选）**
```bash
# AI生图
python scripts/image_generator.py --prompt "生成的提示词" --style cover

# 从本地目录选图
python scripts/image_picker.py --source /vol1/1000/KnowledgeBase/lib-地产/raw/images --count 3

# 纯文字则跳过此步
```

**第三步：确认 + 发布**
```bash
# 预览模式（默认，只填表不发布）
xhs post --title "标题" --content "正文" --image /path/to/img

# 确认后发布
xhs post --title "标题" --content "正文" --image /path/to/img --publish
```

## 可用工具 / 命令

### 1. 搜索
```bash
# 通过 MCP 搜索（需 MCP 已配置）
# Hermes 加载 MCP 后直接调用 search_content tool

# 或通过 CLI
python scripts/content_creator.py --search "关键词" --limit 10
```

### 2. 创意内容生成
```bash
python scripts/content_creator.py \
  --topic "创作主题" \
  --template lifestyle \
  --style 专业/通俗/种草 \
  --count 3
```

### 3. AI 生图
```bash
# ComfyUI 生图（优先）
python scripts/image_generator.py \
  --provider comfyui \
  --prompt "提示词" \
  --type cover \
  --style 小红书封面 \
  --count 1

# image_generate 工具兜底
python scripts/image_generator.py \
  --provider hermes \
  --prompt "提示词" \
  --type content
```

### 4. 从 FnOS/夸克网盘选图
```bash
python scripts/image_picker.py \
  --source /vol1/1000/KnowledgeBase/lib-地产/raw/images \
  --keyword "成都" \
  --count 3 \
  --random
```

### 5. 发布
```bash
# 预览（只填表，不自动发布）
python scripts/publisher.py --preview \
  --title "标题" \
  --content "正文" \
  --images /path/img1.jpg /path/img2.jpg

# 直接发布
python scripts/publisher.py \
  --title "标题" \
  --content "正文" \
  --images /path/img1.jpg

# 纯文字发布
python scripts/publisher.py \
  --title "标题" \
  --content "正文"
```

### 6. 定时发布
```bash
# 创建定时任务（配合 Hermes cron）
python scripts/scheduler.py \
  --action create \
  --topic "主题" \
  --schedule "0 10 * * 1" \
  --template lifestyle
```

### 7. 账号管理
```bash
# 添加账号
xhs account add my-xhs-account
xhs account use my-xhs-account
xhs login

# 检查登录状态
python scripts/login_manager.py --check

# 列出账号
xhs account list
```

## 内容模板（templates/）

| 模板 | 用途 | 风格 |
|------|------|------|
| `default.yaml` | 通用内容 | 默认 |
| `lifestyle.yaml` | 生活方式/种草 | 亲切口语 |
| `realestate.yaml` | 地产营销 | 专业数据+煽动 |
| `knowledge.yaml` | 知识分享 | 结构化 |
| `cover_templates.yaml` | 封面风格 | 配色+排版配置 |

## 违禁词审核

双通道审核：
1. **规则库** — `references/forbidden-words.txt` 内置违禁词匹配
2. **LLM 审核** — AI 判断内容合规性

审核结果包含：风险等级（高/中/低）、违禁词列表、修改建议。

## 目录说明

```
xiaohongshu-creator/
├── SKILL.md               ← 本文件
├── scripts/
│   ├── setup.sh           ← 一键安装所有依赖
│   ├── setup_mcp.sh       ← 单独部署 xiaohongshu-mcp
│   ├── setup_xhscli.sh    ← 单独安装 xhs-cli
│   ├── login_manager.py   ← 账号登录/状态/切换
│   ├── content_creator.py ← AI创意+违禁词审核
│   ├── image_picker.py    ← 本地/网盘选图
│   ├── image_generator.py ← AI生图编排
│   ├── publisher.py       ← 发布编排
│   └── scheduler.py       ← 定时发布
├── templates/
│   ├── default.yaml / lifestyle.yaml
│   ├── realestate.yaml / knowledge.yaml
│   └── cover_templates.yaml
├── references/
│   ├── xhs-cli-commands.md
│   ├── mcp-tools.md
│   ├── forbidden-words.txt
│   └── feishu-publish.md       ← 飞书发布配置参考
└── config/
    └── accounts.json.example
```

## MCP 集成配置

在 Hermes 的 `config.yaml` 中添加 MCP server：

```yaml
mcp_servers:
  xiaohongshu:
    command: docker
    args:
      - run
      - --rm
      - -i
      - -v
      - /tmp/xiaohongshu-mcp:/app/configs
      - ghcr.io/xpzouying/xiaohongshu-mcp:latest
```

或使用 Go 二进制：

```yaml
mcp_servers:
  xiaohongshu:
    command: /usr/local/bin/xiaohongshu-mcp
    args: []
```

## 相关技能
- **social-content**: 社交媒体内容策略与创意（不含自动化发布）
- **copywriting**: 文案写作（为小红书内容提供文案素材）
- **realestate-marketing-kb**: 地产营销知识库（含小红书作为渠道）
- **feishu-lark-cli**: 飞书 CLI 工具（用于发布文案到飞书公开文档）

---

## 图片规格检查（自动校验）

发布图片前自动检查是否符合小红书要求：

| 项目 | 要求 | 超标处理 |
|------|------|----------|
| 单张大小 | ≤ 20MB | 自动跳过，提示用户 |
| 格式 | JPG / PNG / WebP / GIF | 自动跳过不支持格式 |
| 数量 | 1 ~ 18 张 | 超出仅保留前 18 张 |
| 比例 | 推荐 3:4 竖版 | 提示但不会阻止 |

跳过检查：`publisher.py` 加 `--skip-image-check`

## 飞书公开文档发布（可选）

定时任务或发布时可同步将文案发布到飞书公开文档（有链接的人可阅读）。

### 配置方式

**环境变量**（推荐）：
```bash
export FEISHU_APP_ID="cli_a96305a3b97a1cd3"
export FEISHU_APP_SECRET="你的飞书AppSecret"
```

**飞书 CLI 配置文件**（自动识别，但 FnOS 有 DNS 问题，推荐用 Python API）：
`~/.lark-cli/hermes/config.json`

### 命令

```bash
# 单独发布到飞书
python scripts/scheduler.py --action feishu-publish --topic "标题" --content "正文"

# 创建定时任务并同步飞书
python scripts/scheduler.py --action create --topic "主题" --schedule "0 22 * * 1" --feishu

# 发布时同步飞书
python scripts/publisher.py --title "标题" --content "正文" --images img.jpg --feishu-doc
```

## 每周文案工作流（西悦云庭）

每周四 22:30 自动执行，为下周（周五~下周四）生成7篇置业顾问转发用的小红书文案。

### 工作流
```
周四 22:30 → 自动生成7篇文案草稿
           → 通过飞书发送预览消息给我确认
           → 我在飞书回复「确认发布」
           → 执行 publish 推送到飞书公开文档
```

### 项目价值角度（11个）
国企、现房、公园、双园、醇熟配套、学校汇聚、低密、全能套四、横厅、健康森活、青年平墅

### 知识库
项目详细信息存于 `/vol1/1000/KnowledgeBase/lib-地产/wiki/entities/西悦云庭.md`

### 手动操作
```bash
# 生成预览并飞书发我确认（草稿存于 ~/.hermes/scripts/drafts/）
python3 ~/.hermes/scripts/weekly_xhs_plan.py --action preview

# 我确认后，发布到飞书文档
python3 ~/.hermes/scripts/weekly_xhs_plan.py --action publish
```

1. **发布前必须经用户确认** — 默认预览模式，从不自动发布
2. **小红书记者中心同一账号不能多端登录** — MCP 登录后不要再开网页版
3. **标题 ≤ 20字，正文 ≤ 1000字** — 小红书硬限制
4. **每日发帖上限约 50 篇** — 超过有风控风险
5. **新号需实名认证** — 否则会频繁弹出认证提醒
6. **封面图建议 3:4 竖版** — 小红书最佳展示比例
7. **避免引流/搬运** — 官方重点打击对象
8. **图片为可选项** — 纯文字笔记同样支持发布
