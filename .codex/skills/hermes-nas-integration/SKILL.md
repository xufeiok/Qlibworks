---
name: hermes-nas-integration
description: 飞牛OS(FNS) NAS存储与Hermes容器的集成方法 — 文件共享协议配置与容器内读写限制的解决方案
category: productivity
version: 1.1.0
---

# 飞牛OS NAS 与 Hermes 容器集成

## 用户环境摘要

| 属性 | 值 |
|---|---|
| NAS 系统 | 飞牛OS (FnOS) + 1Panel |
| 宿主机网关 | 172.19.0.1 |
| Docker 桥接 | 172.17.0.1 |
| 容器内 Python | /opt/data/venv/bin/python3 |
| pip 路径 | /opt/data/venv/bin/pip |

## 存储路径映射（核心）

**重要规则：容器内写文件必须用装载路径（/opt/xxx），绝对不能用 /vol1/...（那是底层路径，在 overlay 下只读）**

| 装载路径（容器内） | NAS 原始路径 | 用途 |
|---|---|---|
| /opt/knowledgebase | /vol1/1000/KnowledgeBase | 知识库根目录 |
| /opt/knowledge/地产营销 | /vol1/1000/KnowledgeBase/地产营销 | 地产营销知识库 |
| /opt/database | /vol1/1000/MyDatabase | 数据库文件 |
| /opt/github | /vol1/1000/Github | GitHub 仓库本地副本 |

## GitHub 凭证（已持久化）

无需重新配置，直接使用：
- **用户名**: xufeiok
- **Token**: [已删除 — 旧token已废弃]
- **gh CLI**: v2.46.0（已安装）
- **协议**: HTTPS
- **已验证仓库**: xufeiok/Obisidion_base, xufeiok/Quant_Tushare, xufeiok/backtrader_superplot

```bash
# 查看 gh 认证状态
gh auth status

# 克隆私有仓库（无需再输凭证）
git clone https://github.com/xufeiok/Obisidion_base.git /opt/github/Obisidion_base
```

## 已知限制

### NFS/CIFS root_squash 问题（仅限网络挂载）
当容器通过 **NFS 或 CIFS/SMB** 协议挂载飞牛 NAS 共享目录时，NAS 默认开启 **root_squash**：
- 容器内即使是 root 用户，写入时也会被映射为 NAS 上的普通用户
- 实际表现为：容器内无法写入文件（Operation not permitted）
- 属主显示为 NAS 上的实际用户（如 xufei）

**现象：**
```bash
# 可以创建目录
mkdir /vol1/1000/地产营销/西悦云庭  # 成功

# 但无法写入文件
echo 'test' > /vol1/1000/地产营销/test.txt  # Operation not permitted
```

**验证方法：**
```bash
mount | grep vol1
# 如果是 cifs/nfs 类型，则存在 root_squash 限制

df -h /vol1/1000/
# Filesystem 显示为 cifs/nfs 网络卷
```

---

### btrfs 子卷直接挂载：无限制 ✅

当前容器使用 **btrfs 子卷直接挂载**（通过 device mapper），不是 NFS/CIFS。飞牛OS 1Panel 装载配置：

| 装载路径（容器内） | NAS存储路径 | 权限 |
|---|---|---|
| /opt/knowledgebase | /vol1/1000/KnowledgeBase | 读写 |
| /opt/database | /vol1/1000/MyDatabase | 读写 |
| /opt/github | /vol1/1000/Github | 读写 |

**重要：写文件必须用装载路径（如 /opt/knowledgebase）。**

容器内 `/vol1/...` 路径在 overlay 文件系统下表现为只读（Operation not permitted），即使底层存储是读写的。写入操作必须通过装载路径 `/opt/xxx` 进行。

## 解决方案

### 方案1：飞牛开启 WebDAV 共享（推荐）

在飞牛OS控制面板 → 共享文件夹 → 地产营销 → 创建WebDAV共享

优点：容器内可以通过 HTTP 直接读写文件

容器内访问方式：
```bash
# 挂载 WebDAV（如果有 curl 支持）
curl -u user:pass https://nas-ip/webdav/地产营销/项目名/

# 或用 Python requests
```

### 方案2：用户直接发送内容

最可靠的方式——用户通过聊天窗口直接发送文件内容：
- 文字直接粘贴
- 图片截图发送
- Word/PDF 文件上传到聊天

Hermes 读取后处理，结果由用户手动同步回 NAS

### 方案3：确认飞牛NAS的局域网IP

如果知道飞牛NAS的局域网IP（如 192.168.x.x），容器内可以尝试通过 SMB/WebDAV 协议访问。

## 工作流程建议

对于地产项目知识库场景：

```
1. 用户通过飞牛Web界面上传文件到 NAS
2. 用户在 NAS 上开启 WebDAV 共享
3. Hermes 通过 WebDAV URL 读取文件并解析
4. 处理结果存入 /opt/data/skills/ 目录（容器内有写权限）
5. 如果需要回写 NAS，用户通过 NAS 管理界面手动同步
```

## GitHub 私有仓库操作

GitHub token 存储在 `~/.config/gh/hosts.yml`（gh CLI 未安装，但 token 可用）。

克隆私有仓库：
```bash
TOKEN=$(cat ~/.config/gh/hosts.yml | grep "oauth_token:" | head -1 | awk '{print $2}')
git clone https://x:${TOKEN}@github.com/owner/repo.git /tmp/repo
```

## 快速检查命令

```bash
# 检查存储挂载情况
mount | grep -E "vol1|cifs|nfs|opt"

# 检查写入权限（用装载路径）
touch /opt/knowledgebase/test 2>&1

# 查看 GitHub token
cat ~/.config/gh/hosts.yml | grep oauth_token
```
