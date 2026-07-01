---
name: skills-sh-manual-install
description: 从 skills.sh (npx skills) 安装第三方技能到 Hermes 的方法——绕过交互式安装器，直接从 GitHub 下载技能文件并按 Hermes 目录结构存放。适用于 npx skills add 卡在"Which agents do you want to install to?"交互界面、或技能安装器不支持 Hermes 的情况。
version: 1.0
author: Hermes Agent
license: MIT
---

# Skills.sh 手动安装指南

## 问题

`npx skills add <owner/repo> --skill <skill-name>` 会卡在交互式 agent 选择界面：

```
◇  Found 1 skill
●  Selected 1 skill: xxx
◇  54 agents
◆  Which agents do you want to install to?
│  ── Universal (.agents/skills) ── always included ────────────
│    • Amp
│    • Antigravity
│    • Cline
│    • Codex
│    ...
```

原因是 skills.sh 的安装器面向 Cursor/Claude Code/OpenClaw 等 Agent，**不支持 Hermes**。

## 解决方案

手动从 GitHub 下载技能文件，按 Hermes 技能目录格式存放。

## 操作步骤

### Step 1：定位技能文件

技能文件托管在 `https://github.com/<owner>/<repo>/tree/main/` 或 `tree/master/`。

典型结构：
```
<repo>/
├── SKILL.md
├── assets/
│   └── template.html（可选）
├── references/
│   ├── xxx.md（可选）
│   └── yyy.md（可选）
└── scripts/
    └── xxx.py（可选）
```

### Step 2：确定安装路径

Hermes 技能目录格式：`/opt/data/skills/<category>/<skill-name>/`

常用 category：
- `creative` — 创意/设计/视觉类
- `productivity` — 效率工具类
- `software-development` — 开发类
- `research` — 研究类
- `mlops` — 机器学习运维类
- `github` — GitHub 相关类
- `media` — 媒体类

### Step 3：确定技能名称

SKILL.md 的 YAML frontmatter 中的 `name` 字段即为技能名（如 `guizang-ppt-skill`）。

### Step 4：下载并安装

```python
import urllib.request
import os

owner = "op7418"
repo = "guizang-ppt-skill"
skill_name = "guizang-ppt-skill"
category = "creative"

# 1. 获取仓库文件列表
api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as r:
    items = json.loads(r.read())

# 2. 递归下载所有文件
def download_dir(path, base_url):
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    req = urllib.request.Request(api, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        children = json.loads(r.read())
    for item in children:
        if item['type'] == 'file':
            url = item['download_url']
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as r:
                content = r.read().decode('utf-8')
            local_path = f"/opt/data/skills/{category}/{skill_name}/{item['path'].replace(f'{path}', '').lstrip('/')}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"OK: {local_path}")
        elif item['type'] == 'dir':
            download_dir(item['path'], base_url)

download_dir("", "")
```

### Step 5：验证安装

```bash
# 检查文件是否存在
find /opt/data/skills/<category>/<skill-name> -type f | sort

# 验证技能能被 Hermes 识别
# 用 skills_list() 或 skill_view(name) 查看
```

## 关键判断

### 技能是否适合 Hermes？

检查 SKILL.md 的 YAML frontmatter：
- 有 `name` 和 `description` 字段 → 格式兼容
- 只依赖通用工具（terminal, write_file, browser 等）→ 可用
- 依赖特定 Agent 的专有 API（如 `ask question`, Claude Code tools）→ 部分可用，需要读 SKILL.md 判断

### 技能是否已预装？

```bash
find /opt/data/skills -name "SKILL.md" | xargs grep "^name:" | grep -i "<keyword>"
```

### 分类目录

如果技能目录下 `SKILL.md` 能被 `skills_list` 识别，说明分类正确。不需要额外注册——Hermes 自动扫描 `/opt/data/skills/` 下的子目录。

## 已知限制

- 如果技能附带了 `scripts/*.py` 或 `scripts/*.sh`，需确认依赖是否在 Hermes 环境中存在
- 如果 SKILL.md 大量引用了 Agent 专有工具（Cursor 的 `read_task_memory`, OpenClaw 的 `claw_*` 命令），功能会受限
- skills.sh 安装器会在克隆后自动运行 `ctx-skill-add`，Hermes 不需要这一步——只需确保文件在正确位置

## 相关技能

- `find-skills` — 从 skills.sh 搜索可用技能
- `skill-creator` — 创建新技能的方法
