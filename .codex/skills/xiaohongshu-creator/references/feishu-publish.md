# 飞书公开文档发布配置

## 飞书应用凭证获取

1. 打开 https://open.feishu.cn/app
2. 创建企业自建应用（或使用已有应用）
3. 在「凭证与基础信息」中获取 `App ID` 和 `App Secret`
4. 在「权限管理」中添加权限：
   - `docx:document`（文档读写）
   - `drive:drive`（云文档管理）
5. 发布应用

## 环境变量配置

```bash
export FEISHU_APP_ID="cli_xxxxxxxxxxxxx"
export FEISHU_APP_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxx"
```

建议写入 `~/.bashrc` 或 `~/.hermes/.env` 中持久化。

## 飞书 CLI 安装（可选）

```bash
# 安装 lark CLI
curl -sL https://github.com/larksuite/cli/releases/download/v1.0.23/lark-cli-1.0.23-linux-amd64.tar.gz | tar -xz
cp lark-cli /vol1/1000/HermesAgent/venv/bin/lark
chmod +x /vol1/1000/HermesAgent/venv/bin/lark
```

## 测试配置

```bash
python scripts/scheduler.py --action feishu-publish --topic "测试" --content "这是一条测试文案"
```

如果返回文档 URL，说明配置成功。
