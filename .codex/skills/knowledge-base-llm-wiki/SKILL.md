# knowledge-base-llm-wiki

> Karpathy llm-wiki 联邦多库知识库管理（v3.2）

## 触发条件

当用户提到以下场景时，**必须**加载本技能：
- 「知识库」「加入知识库」「查询知识库」
- 「建立知识库」「新领域」「新的知识库」
- 「检查知识库」「健康检查」「重建索引」
- 「这个知识库怎么用」「知识库结构」
- 管理/维护现有 `/vol1/1000/KnowledgeBase/` 知识库
- 把网页/链接加入知识库、URL转markdown、网页剪藏

## 知识库路径

```
/vol1/1000/KnowledgeBase/          # 联邦知识库根目录
```

## 架构版本

当前版本：**v3.2**（hub/ 中枢 + 网页剪藏 ingest）

## 顶层结构

```
KnowledgeBase/
├── hub/                       # 中枢（唯一入口）
│   ├── index.md             # 全局总索引
│   ├── contracts/
│   │   ├── GLOBAL-CONTRACT.md   # 全局宪法（五条铁律）
│   │   └── DOMAIN-RULES.md      # 领域路由规则
│   ├── dashboard/
│   │   └── 健康面板.md          # 全局健康一览
│   ├── connections/
│   │   └── 跨域链接.md          # 跨领域关联记录
│   └── scripts/              # 中枢自动化脚本
│
├── lib-{领域}/               # 领域库（N个，结构完全统一）
│   ├── raw/                  # 原始资料（人类唯一可写，LLM只读）
│   ├── wiki/                 # LLM生成（唯一可写区）
│   ├── outputs/              # LLM输出物
│   ├── LLM-CONTRACT.md      # 本领域合约（继承全局）
│   ├── README.md
│   └── .gitignore
│
├── scripts/                  # 全局脚本（位于根目录）
│   ├── global-lint.py        # 健康检查
│   ├── global-reindex.py     # 重建索引
│   └── cross-lib-search.py  # 跨库搜索
│
├── CLAUDE.md                # 中枢Schema（根入口）
├── index.md                 # 根总索引（指向hub/）
├── log.md                   # 全局日志
└── .gitignore
```

## 核心规则（五条铁律）

来自 `hub/contracts/GLOBAL-CONTRACT.md`：

1. **人类只写 raw，LLM 只读 raw，绝不修改 raw**
2. **LLM 唯一可写区：wiki/、outputs/**
3. **一个领域 = 一个上下文，禁止跨域污染**
4. **LLM 不得访问 lib-private/、**/private/、**/99_private_raw/**
5. **原文件永不修改、永不重命名**

## 三级权限

| 路径 | LLM 权限 |
|------|---------|
| wiki/, outputs/ | 可读可写 |
| raw/ | 可读不可写 |
| **/private/, **/99_private_raw/ | **完全禁止** |

## raw/ 格式分类

| 目录 | 内容 |
|------|------|
| doc_word/ | Word 文档 + 文本文件（.doc/.docx/.txt）|
| doc_ppt/ | PowerPoint 演示文稿（.ppt/.pptx）|
| doc_excel/ | Excel 表格（.xls/.xlsx）|
| pdf/ | PDF 文档 |
| clippings/ | **网页剪藏**（从URL抓取的markdown原文）|
| notes_md/ | Markdown 笔记 |
| images/ | 图片资源 |
| other_files/ | 其他格式文件 |
| private/ | **绝对私密（LLM禁止访问）** |

## wiki/ 页面分类

| 目录 | 内容 | 命名 |
|------|------|------|
| sources/ | 来源摘要（对应raw/文件） | source-{原文件名}.md |
| concepts/ | 概念/模型/框架/理论 | 理论名称.md |
| entities/ | 人物/工具/项目/品牌 | 实体名称.md |
| synthesis/ | 综述/多来源综合分析 | 主题综述.md |
| templates/ | 可复用模板 | {模板名}.md |

**固定文件**（每个wiki根目录必须有）：
- index.md — 内容索引
- log.md — 操作日志（仅追加）
- overview.md — 领域概览
- QUESTIONS.md — 待研究问题

## 标准工作流

### 工作流1：加入本地文件（Ingest）

```
用户：把这份资料加入知识库 / 上传了文件
↓
1. 确认属于哪个领域库（路由规则见 hub/contracts/DOMAIN-RULES.md）
2. 将文件放入 lib-{领域}/raw/ 对应格式目录
   - .doc/.docx/.txt → raw/doc_word/
   - .ppt/.pptx → raw/doc_ppt/
   - .xls/.xlsx → raw/doc_excel/
   - .pdf → raw/pdf/
   - .png/.jpg/.gif → raw/images/
   - 其他格式 → raw/other_files/
3. 读取文件内容
4. 生成摘要 → wiki/sources/source-{文件名}.md
5. 如有必要，生成/更新 entities/ 实体页
6. 更新 wiki/index.md（内容索引）
7. 追加 wiki/log.md（记录本次 ingest）
8. 运行 global-reindex.py（可选）
```

### 工作流2：从URL加入网页内容（Web Ingest）

```
用户：这个链接加入知识库 / 把这个网页保存到知识库
↓
1. 确认属于哪个领域库（hub/contracts/DOMAIN-RULES.md）
2. 抓取网页内容
   a. 优先使用 defuddle 技能（如果已加载）：干净markdown输出
   b. 备选（terminal工具）：curl + 降噪正则提取
   c. JS动态页面（Doubao等）：使用 browser 工具截图/提取
3. 保存为 .md 文件 → lib-{领域}/raw/clippings/
   文件名格式：YYYY-MM-DD_来源站点_标题摘要.md
   例：2026-05-03_doubao_Karpathy-LLM-Wiki联邦搭建指南.md
4. 读取markdown内容，生成摘要 → wiki/sources/source-{文件名}.md
5. 如有必要，生成/更新 entities/ 或 concepts/ 实体页
6. 更新 wiki/index.md 和 log.md
7. 追加到 hub/connections/跨域链接.md（如有跨域内容）
```

### 工作流3：查询知识

```
用户：查询 X / 知识库里有没有关于X的内容
↓
1. 读取 hub/contracts/DOMAIN-RULES.md 判断属于哪个领域
2. 进入对应 lib-{领域}/ 读取相关内容
3. 综合回答，结论写回主领域（如果需要生成新页面）
```

### 工作流4：新建领域库

```
用户：我想新建一个知识库
↓
1. 创建 lib-{新领域}/ 目录结构
2. 创建 raw/ 的所有子目录
3. 创建 wiki/ 的所有子目录 + index.md/log.md/overview.md/QUESTIONS.md
4. 创建 outputs/
5. 创建 LLM-CONTRACT.md（继承全局契约）
6. 创建 README.md
7. 在 hub/index.md 注册新库
8. 在 hub/contracts/DOMAIN-RULES.md 添加路由规则
9. 运行 global-reindex.py
```

### 工作流5：健康检查

```
python3 scripts/global-lint.py
python3 scripts/global-reindex.py
```

## 全局脚本

| 脚本 | 命令 | 用途 |
|------|------|------|
| global-lint.py | `python3 scripts/global-lint.py` | 检查所有库的结构完整性 |
| global-reindex.py | `python3 scripts/global-reindex.py` | 重建全库 index.md，同步 hub |
| cross-lib-search.py | `python3 scripts/cross-lib-search.py <关键词> [lib名...]` | 跨库全文搜索 |

## 文件命名规范

```
YYYY-MM-DD_类型_名称.后缀
例：2026-05-03_营销方案_五一活动.docx
    2026-05-03_量化策略_均线因子.xlsx
    2026-05-03_写作素材_散文灵感.md
    2026-05-03_doubao_Karpathy联邦搭建指南.md  ← 网页剪藏
```

## 网页抓取详细方法

### 方法A：defuddle 技能（推荐）

如果 `defuddle` 技能已加载，直接使用：
```
skill_view(name='defuddle')
```
它会返回干净的 markdown 内容，直接保存即可。

### 方法B：terminal + curl（备选）

```bash
# 基础抓取（静态页面）
curl -s --max-time 20 \
  -H "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  "https://example.com/page" -o page.html

# 提取正文（Python正则降噪）
python3 -c "
import re, sys
html = open('page.html', encoding='utf-8', errors='ignore').read()
# 移除 script/style
html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
# 提取正文文本
text = re.sub(r'<[^>]+>', ' ', html)
text = re.sub(r'\s+', ' ', text).strip()
print(text[:3000])
"
```

### 方法C：browser 工具（JS动态页面）

对于 Doubao/微博等 JS 渲染页面，使用 browser MCP 工具：
```
使用 browser 工具访问 {url}，提取页面主要内容
```

### 常见JS动态页面特征

| 平台 | 特征 | 方案 |
|------|------|------|
| Doubao（豆包） | JS渲染，curl返回空HTML | browser 工具 |
| 微博 | JS加载内容 | browser 工具 |
| 知乎 | 部分JS，需要登录 | browser 工具 |
| CSDN / 简书 / 知乎文章 | 大多静态，直接curl | curl 方法 |

## 跨域引用

引用其他领域库时使用相对路径：

```
[[../../lib-营销管理/wiki/concepts/定位理论]]
[[../../lib-地产/wiki/entities/西悦云庭]]
```

**规则**：只允许跨库阅读，不允许跨库写入。结论必须写回「用户指定的主领域」。

## Obsidian Vault

`lib-{领域}/wiki/` 是 Obsidian vault 目录，直接用 Obsidian 打开即可：
- 图谱视图（Graph View）
- 反向链接（Backlinks）
- 双向链接（Wikilinks）

`.obsidian/` 配置已保存在各库的 wiki/ 目录下。

## 已知领域库

| 领域 | 路径 | 内容 |
|------|------|------|
| 营销管理 | lib-营销管理/ | 69个营销管理理论笔记 |
| 地产 | lib-地产/ | 西悦云庭项目原始文档+摘要 |
| 量化交易 | lib-量化交易/ | （空库，待建） |
| 写作 | lib-写作/ | （空库，待建） |

## 常见陷阱

### 陷阱1：误修改 raw/ 文件
raw/ 中的原文件永远不要修改或重命名。如果需要「修改」，应该：
- 重新 ingest 一个新版本到 raw/（保留原文件）
- 在 wiki/ 中生成新版本摘要

### 陷阱2：跨域写入
错误：在 lib-地产 中直接写 lib-营销管理 的内容
正确：结论写回 lib-地产/wiki/，引用理论时使用跨库路径

### 陷阱3：删除原始文件
删除操作只能删除 LLM 生成的内容（wiki/），raw/ 中的文件只能由人类操作。

### 陷阱4：忘记追加 log.md
每次 ingest 或修改 wiki/ 内容后，必须追加 log.md 记录操作。

### 陷阱5：Doubao 等 JS 动态页面用 curl 白费力气
- `curl` 访问 Doubao 返回空框架 HTML（JS动态渲染）
- 第1次 curl 失败后立即换 browser 工具，不要连续重试
- 预防：先用 curl 测试，有内容再处理，无内容立刻换方案

## 架构演进历史

- v1.0: LLM Wiki 单库（raw/ + wiki/ 平铺）
- v2.0: 联邦多库（domains/ 下按领域独立）
- v3.0: lib-{领域} 标准化（raw/分格式 + wiki/分类型）
- v3.1: hub/ 中枢（contracts/ + dashboard/ + connections/）
- **v3.2**: Web Ingest（URL → raw/clippings/ → wiki/sources/）

## 参考资料

- Karpathy LLM-Wiki 理论
- [[hub/contracts/GLOBAL-CONTRACT]]
- [[hub/contracts/DOMAIN-RULES]]
- [[defuddle]] — 网页 → 干净markdown（优先使用）
