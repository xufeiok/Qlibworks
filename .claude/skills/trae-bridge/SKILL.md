---
name: trae-bridge
description: Trae技能桥接 — 自动发现和加载 Trae 的 60 个技能。当需要数据库操作、Docker管理、GitHub CLI、文档处理、量化交易、微信发布等能力时，自动从 Trae 技能库查找并应用。
---

# Trae 技能桥接

本技能作为 Claude Code 和 Trae 技能系统之间的桥梁。当需要某项能力时，先检查下方索引找到对应的 Trae 技能，然后读取其 SKILL.md 获取详细指令。

## 使用方法

1. 根据用户需求，在下方技能索引中匹配最相关的技能
2. 读取 `E:/Quant/.trae/skills/<技能名>/SKILL.md` 获取完整指令
3. 严格按照 Trae 技能中的指导执行

## Trae 技能完整索引

### 数据库与存储
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `database-operations` | PostgreSQL/MySQL/SQLite/MongoDB 连接、查询、导出、备份 | 连接数据库、执行SQL、导出数据、查看表结构 |
| `docker-database-operations` | Docker容器内操作其他容器中的数据库（ClickHouse/DuckDB/MySQL） | 连Docker里的数据库、扫端口找数据库 |

### Docker与容器
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `docker-management` | Docker容器和镜像管理，docker-compose操作 | 查看/启停容器、容器日志、清理镜像 |

### 开发工具
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `gh-cli` | GitHub CLI 全面参考（PR/Issue/Actions/Release/Gist） | GitHub操作、PR管理、Issue查询 |
| `git-commit` | 智能Git提交（分析diff、生成conventional commit消息） | 提交代码、生成commit消息 |
| `github-actions` | GitHub Actions CI/CD 工作流 | 配置CI/CD、自动化构建部署 |
| `debugging` | Python调试技术（pdb、IDE工具） | Python调试、断点、调用栈分析 |
| `systematic-debugging` | 系统化调试方法论 | 遇到bug、测试失败、异常行为 |
| `test-driven-development` | TDD开发流程 | 实现功能或修复bug之前 |
| `pytest-testing` | Pytest测试框架（fixtures/mocking/CI集成） | 编写Python测试 |
| `mcp-builder` | MCP服务器构建指南（Python FastMCP / Node SDK） | 构建MCP服务器、集成外部API |
| `using-git-worktrees` | Git worktree隔离工作空间 | 需要隔离开发环境时 |

### 文档处理
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `docx` | Word文档创建/编辑/审阅（Track Changes/批注/格式保留） | 处理.docx文件 |
| `xlsx` | Excel电子表格创建/编辑/分析（公式/格式化/数据可视化） | 处理电子表格 |
| `pdf` | PDF操作（提取文本表格/创建/合并拆分/表单填写） | 处理PDF文档 |
| `pdf-extraction` | PDF解析（IBM docling，支持OCR） | 从PDF提取结构化内容 |
| `doc-parser` | 高级文档解析（IBM docling：PDF/Word/PPT/图片/HTML） | 从复杂文档提取结构化内容 |
| `somark-document-parser` | Somark文档解析（PDF/DOCX/图片） | 文档转结构化数据 |
| `xparse-parser` | xparse-cli文档解析（支持加密PDF/OFD/HTML） | 文档转markdown/JSON |

### 设计与前端
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `frontend-design` | 高质量前端界面设计（避免AI美学同质化） | 构建网页/组件/仪表盘/landing page |
| `canvas-design` | 静态视觉设计（海报/艺术/PNG/PDF） | 创建海报、设计、静态视觉 |
| `theme-factory` | 主题样式工具包（10个预设主题） | 给制品应用主题风格 |
| `huashu-design` | 花叔Design——HTML高保真原型/交互Demo/动画/设计探索 | 做原型、设计Demo、动画Demo |
| `algorithmic-art` | p5.js算法艺术（流场/粒子系统） | 生成艺术、创意编码 |
| `brand-guidelines` | Anthropic品牌色和字体应用 | 需要品牌风格时 |
| `web-artifacts-builder` | 复杂多组件HTML制品（React/Tailwind/shadcn） | 需要复杂前端制品 |

### 内容与社交
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `social-content` | 社交媒体内容创建/排期/优化（LinkedIn/Twitter/Instagram等） | 社交媒体发文、内容日历 |
| `copywriting` | 营销文案撰写（首页/落地页/定价页/功能页） | 写营销文案 |
| `publish-to-wechat` | 自动发布Markdown到微信公众号草稿箱 | 发布微信公众号文章 |
| `guizang-ppt-skill` | 电子杂志风格网页PPT生成 | 制作杂志风PPT |
| `visual-story-designer` | 文章转视觉故事板（小红书/抖音/Instagram） | 文章转图文、社媒视觉 |
| `news-topic-generator` | 新闻选题生成（5W1H+发散思维） | 提供新闻需要写作创意 |
| `trending-news-scanner` | 全网热搜扫描（百度/微博 Top10） | 查看热门新闻 |
| `changelog-generator` | 从git提交自动生成用户友好的更新日志 | 生成changelog |
| `competitive-ads-extractor` | 竞品广告分析（Facebook/LinkedIn广告库） | 分析竞品广告 |

### 元技能与工具
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `skill-creator` | 创建/更新Claude技能的指南 | 创建新技能 |
| `skill-from-github` | 从GitHub优质项目学习并创建技能 | 需要从开源项目学习 |
| `skill-from-masters` | 从领域专家方法论创建技能 | 创建技能前先学习最佳实践 |
| `skill-lookup` | 发现和安装Agent Skills | 查找可用技能 |
| `skill-router` | 仓库感知的技能管理（扫描技术栈，加载相关技能） | 切换项目、管理技能 |
| `skill-share` | 创建技能并自动通过Slack分享 | 团队分享技能 |
| `search-skill` | 从可信市场搜索和推荐Claude Code技能 | 按需求查找技能 |
| `find-skills` | 从Skills.sh目录发现可安装技能 | 查找/添加技能 |
| `skills-sh-manual-install` | 从skills.sh手动安装技能到Hermes | skills.sh安装卡住时 |
| `toolbox` | 开发前后工具箱——加载技能/代理组合 | 开发开始或结束时 |
| `workflow-orchestrator` | 技能发现+自定义工作流+编排执行 | 设计复杂工作流 |
| `writing-skills` | 创建/编辑/验证技能的技能 | 编写技能文件 |
| `requesting-code-review` | 代码审查请求（完成任务、实现功能后） | 提交前验证 |
| `prompt-lookup` | 提示词查找 | 需要查找提示词模板 |

### 业务专项
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `quant-all-star-team` | 量化交易策略/因子工程/回测/风控/算法执行 | 量化交易相关问题 |
| `realestate-marketing-kb` | 地产营销知识库（项目信息+文案方法论） | 地产营销决策和文案 |
| `hermes-nas-integration` | 飞牛OS NAS存储与Hermes容器集成 | NAS存储/容器文件共享配置 |
| `knowledge-base-llm-wiki` | LLM Wiki知识库 | 知识库相关操作 |
| `karpathy-guidelines` | Karpathy的LLM编码最佳实践 | 编码方法论参考 |
| `edit-protected-files` | Hermes环境中编辑受保护文件（.env等） | 编辑凭证文件 |
| `repo-stats-autoupdate` | README统计自动更新（技能/代理/测试数量） | 发布前更新README |

### 其他
| 技能名 | 描述 | 触发场景 |
|--------|------|---------|
| `artifacts-builder` | 复杂多组件HTML制品构建工具 | 需要状态管理/路由的制品 |
| `deckify` | Deckify相关 | - |
| `docling` | IBM Docling文档解析 | 文档解析 |
| `mlops` | MLOps相关 | 机器学习运维 |
| `xparse-parser` | 通用文档解析 | 文件转结构化数据 |

## 注意事项

- **优先使用 Claude Code 原生能力**：ECC 技能系统和官方插件已覆盖大量场景，Trae 技能作为补充
- **重叠领域**：python-testing(ECC) 与 pytest-testing(Trae) 重叠时，优先用 ECC 版本
- **环境限定**：hermes-nas-integration、edit-protected-files 等技能仅在 Hermes 容器环境适用
- **读取路径**：所有Trae技能位于 `E:/Quant/.trae/skills/<技能名>/SKILL.md`
