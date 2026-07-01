# xhs-cli 命令参考
# 来源: https://github.com/joohw/xhs-cli

## 账号管理
xhs account add <name>          # 添加账号
xhs account use <name>          # 设置当前账号
xhs account current             # 显示当前账号
xhs account list                # 列出所有账号
xhs account show <name>         # 查看单账号配置

## 登录
xhs login                       # 登录当前账号（扫码）
xhs login --account <name>      # 登录指定账号

## 数据
xhs metrics                     # 运营数据摘要
xhs recent --limit 20           # 已发笔记列表
xhs detail <noteId>             # 单篇笔记详情
xhs posted                      # 本地发帖归档

## 发帖
xhs post \
  --title "标题" \
  --content "正文" \
  --image ./img1.jpg \
  --image ./img2.jpg

  # --publish      # 加此参数自动点击发布按钮（默认只填表预览）
  # --account xxx  # 临时使用其他账号
  # --content-file path  # 从文件读正文

## 其他
xhs help                        # 帮助

## 数据目录
~/.xhs-cli/.cache/
├── accounts/registry.json      # 账号注册表
├── accounts/<name>/browser-data/  # 浏览器会话
└── published/                  # 发帖归档

## 注意
- 标题 ≤ 20字
- 正文 ≤ 1000字
- 图片 1-18张（可选）
- --publish 才自动点击发布
