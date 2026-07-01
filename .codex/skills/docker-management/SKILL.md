---
name: docker-management
description: Docker 容器和镜像管理——查看/启动/停止/删除容器，管理镜像，docker-compose 操作，容器日志。触发：帮我看看有哪些容器在跑、帮我重启下 XX 容器、查看容器日志、清理无用镜像、docker-compose up/down。
---

# Docker 管理

## 环境限制

当前 Hermes 运行在**容器内**，Docker Socket 未挂载时无法直接执行 docker 命令。

### 检查 Docker 是否可用

```bash
# 方式1：检查 socket
ls -la /var/run/docker.sock 2>/dev/null && echo "socket可用" || echo "socket不可用"

# 方式2：检查 docker CLI
docker ps 2>&1 | head -5

# 方式3：通过 Docker API (socket 未挂载时的替代)
curl -s --unix-socket /var/run/docker.sock http://localhost/containers/json 2>/dev/null | python3 -c "import sys,json; [print(c['Names'], c['State']) for c in json.load(sys.stdin)]" 2>/dev/null
```

### Socket 不可用时的备选方案

若 Docker Socket 挂载在宿主机，尝试以下路径：

```bash
# 检查宿主机 Docker Socket
ls -la /var/run/docker.sock 2>/dev/null || \
ls -la /host/var/run/docker.sock 2>/dev/null || \
ls -la ~/docker.sock 2>/dev/null

# 尝试通过 nfs/共享挂载访问
ls -la /mnt/docker.sock 2>/dev/null
ls -la /opt/data/docker.sock 2>/dev/null
```

---

## 查看容器

```bash
# 列出运行中的容器
docker ps

# 列出所有容器（包括停止的）
docker ps -a

# 查看特定镜像的容器
docker ps -a --filter "ancestor=nginx"

# 查看容器详情（含 IP）
docker inspect <container_name_or_id> | python3 -c "
import sys,json
d=json.load(sys.stdin)
c=d[0]
print(f'Name: {c[\"Name\"]}')
print(f'State: {c[\"State\"][\"Status\"]}')
print(f'IP: {c[\"NetworkSettings\"][\"IPAddress\"]}')
print(f'Image: {c[\"Config\"][\"Image\"]}')
"
```

---

## 启动/停止/重启

```bash
# 启动
docker start <container>

# 停止
docker stop <container>

# 重启
docker restart <container>

# 强制停止（杀进程）
docker kill <container>

# 批量操作示例：停止所有运行中的容器
docker ps -q | xargs docker stop
```

---

## 容器日志

```bash
# 实时跟踪日志
docker logs -f <container>

# 查看最近 100 行
docker logs --tail 100 <container>

# 按时间过滤（过去 1 小时）
docker logs --since 1h <container>

# 带时间戳
docker logs -t <container> | tail -50

# 搜索错误关键字
docker logs <container> 2>&1 | grep -i error | tail -20
```

---

## 镜像管理

```bash
# 列出所有镜像
docker images

# 删除悬空镜像（无 tag）
docker image prune -a

# 删除特定镜像
docker rmi <image_id>

# 批量删除未使用的镜像
docker image prune -a -f

# 查看镜像大小（排序）
docker images --format "table {{.Repository}}\t{{.Size}}" | sort -k2 -hr | head -20
```

---

## docker-compose

```bash
# 启动（后台）
docker-compose up -d

# 停止并删除
docker-compose down

# 只停止服务（保留网络/卷）
docker-compose stop

# 重启
docker-compose restart

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f [service_name]

# 重新构建并启动
docker-compose up -d --build

# 强制重新创建容器
docker-compose up -d --force-recreate
```

---

## 清理

```bash
# 清理停止的容器、未使用的网络、悬空镜像
docker system prune -f

# 清理全部未使用资源（容器、网络、镜像、构建缓存）
docker system prune -a -f --volumes  # 慎用，会删卷

# 检查磁盘占用
docker system df
```

---

## 实用脚本

### 批量查看容器状态

```bash
#!/bin/bash
echo "=== 容器状态 ==="
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
echo ""
echo "=== 资源占用 ==="
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
```

### 快速重启问题容器

```bash
#!/bin/bash
CONTAINER=$1
docker logs --tail 50 $CONTAINER 2>&1 | grep -iE "(error|exception|fatal)" && \
echo "检测到错误，重启中..." && docker restart $CONTAINER && \
echo "已重启" || echo "无明显错误"
```

---

## 已知问题

1. **Socket 不可用** — 若 /var/run/docker.sock 未挂载，需通过 docker API 或告诉用户从宿主机操作
2. **权限问题** — 非 root 用户可能需要 sudo，执行前加 `sudo` 或将用户加入 docker 组
3. **容器内操作** — 当前 Hermes 在容器内，docker 命令只能在宿主机执行
