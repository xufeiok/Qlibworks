# 小红书技能设计执行经验总结

> 从西悦云庭、樾江峰荟/亿澜峰荟三个项目的每周文案自动化任务中提炼

## 一、架构设计得失

### ✅ 做对的

1. **双脚本并行架构**
   - 一个项目一个脚本（`weekly_xhs_plan.py` + `weekly_yjyl_plan.py`），各自独立
   - 共用同一个 `xiaohongshu-creator` 技能
   - 错开执行时间（22:00 + 22:30）避免资源竞争

2. **两步工作流（预览→确认→发布）**
   - `--action preview`: 生成草稿 + 飞书消息预览（用户确认前什么都不发）
   - `--action publish`: 用户确认后才执行，写入飞书文档
   - 符合用户「先确认再执行」的铁律

3. **草稿持久化**
   - 草稿存 `~/.hermes/scripts/drafts/latest_xxx_draft.json`
   - 发布前可反复修改草稿内容
   - 发布后标记 `status: published` 防止重复发布

4. **知识库驱动内容**
   - 项目价值角度从知识库中提取（`lib-地产/wiki/sources/`）
   - 文案生成时引用知识库，保证内容准确

### ❌ 做错的

1. **飞书 API block type 搞错**
   - 最初用了 `block_type: 3=text, 4=heading1, 22=divider`
   - 正确是：`2=text, 3=heading1, 4=heading2, 5=heading3, 27=divider`
   - heading 块的 key 要用 `heading1`/`heading2`/`heading3`，不是 `text`
   - divider 块用 `"divider": {}` 空对象，不是 text
   - 结果：前三次发布都是空白文档（写了但API拒绝）

2. **lark CLI DNS 不可靠**
   - FnOS 的 DNS 服务器（192.168.1.1）对 open.feishu.cn 解析不稳定
   - 即使设了 `GODEBUG=netdns=go` 和 hosts 文件，Go 的 DNS 解析仍间歇性失败
   - 而 Python requests 和 curl 始终正常——因为 Go 用纯 Go DNS 解析器，不走系统 /etc/hosts
   - 最终改用 Python requests API 逐块写入（block-by-block），稳定可靠

3. **Cookie 跨平台传输不可行**
   - 已确认：Chromium 148 及以上版本的 Cookie 用平台级加密（Windows DPAPI / Linux OSCrypt）
   - 跨平台复制 Cookie 文件 → 解密失败 → xhs-cli 报"未登录"
   - 解决方案：在 Windows 上用 xhs-cli 发布，服务器只做内容生成

4. **飞书消息确认流程的误区**
   - 最初设计：飞书回复「确认发布」→ 自动触发 publish
   - 实际：Hermes gateway 不理解自定义命令
   - 修正：用户在终端回复确认，我手动执行 `--action publish`

## 二、地产内容规则（重要经验）

从用户多次纠正中总结的硬规则：

| 规则 | 说明 | 触发场景 |
|------|------|---------|
| 数字前加"约" | 所有数字前加"约"字 | 地产营销行业规范 |
| 外部配套定性描述 | 不写具体距离、学校名、地铁站名 | 避免承诺不确定的外部条件 |
| 产品名规范 | 建筑面积约133平米青年平墅（非"洋房"） | 西悦云庭专属 |
| 公区精装（非五重精装） | 户型内精装口碑不好，只说公区 | 樾江峰荟/亿澜峰荟 |
| 高知社区（非智慧社区） | 业主多为政企中高层 | 樾江峰荟/亿澜峰荟 |
| 园林贝尔高林 | 五重园林标准，樾江峰荟贝尔高林操刀 | 大邑项目核心卖点 |
| 不写已过时的配套 | 如泡桐树小学已无 | 需定期检查知识库 |

## 三、改进建议

1. **知识库定期更新**：用户提出的修正应同步更新知识库，确保所有脚本调用的数据一致
2. **发布仍需 Windows**：服务器无GUI+Chromium加密，真正发布到小红书还是需要用户在Windows执行
3. **飞书文件夹权限**：机器人只能写共享空间，不能写用户个人"我的文件夹"
