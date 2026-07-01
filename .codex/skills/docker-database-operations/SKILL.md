---
name: docker-database-operations
description: 在 Docker 容器内操作其他 Docker 容器中的数据库（ClickHouse、DuckDB、MySQL等）— 网络发现、驱动安装、连接方法。触发：连 Docker 里的数据库、查 ClickHouse、在 Docker 里扫端口找数据库。
category: productivity
version: 1.1
---

# Docker 容器内操作数据库

## 环境限制（当前 Hermes 容器）

### 已确认的环境状态

| 检查项 | 状态 | 说明 |
|-------|------|------|
| Docker Socket | ❌ 不可用 | `/var/run/docker.sock` 未挂载 |
| Docker CLI | ❌ 不可用 | 容器内无 docker 命令 |
| 系统 Python | ⚠️ 无 pip | `python3` 可用但无 pip 模块 |
| venv Python | ✅ 可用 | `/opt/data/venv/bin/python3` 配套 pip |
| 容器网络 | 172.19.0.x | 需要扫描发现数据库地址 |

### pip 安装路径（关键！）

```bash
# 错误 - pip 不在系统路径
pip install xxx          # command not found
python3 -m pip install xxx  # No module named pip

# 正确 - 使用 venv 的 pip
/opt/data/venv/bin/pip install xxx
/opt/data/venv/bin/python3 -c "import xxx"
```

### 安装常用数据库驱动

```bash
/opt/data/venv/bin/pip install duckdb clickhouse-connect pymysql redis psycopg2-binary
```

---

## 数据库发现：端口扫描

容器内无法使用 `docker ps`，需要通过 socket 扫描发现数据库：

```python
import socket

def scan_ports(hosts, ports):
    """扫描指定主机的端口开放情况"""
    results = []
    for host in hosts:
        for port, name in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((host, port))
                status = '✅ 开放' if result == 0 else '❌ 关闭'
                print(f'{host}:{port} ({name}) - {status}')
                sock.close()
            except Exception as e:
                print(f'{host}:{port} ({name}) - 错误: {e}')
    return results

# 常见数据库端口
db_ports = [
    (8123, 'ClickHouse HTTP'),
    (9000, 'ClickHouse TCP'),
    (5432, 'PostgreSQL'),
    (3306, 'MySQL/MariaDB'),
    (27017, 'MongoDB'),
    (6379, 'Redis'),
    (7687, 'Neo4j'),
]

# 需要扫描的主机
hosts = ['172.19.0.1', '172.17.0.1', 'host.docker.internal', '127.0.0.1']

scan_ports(hosts, db_ports)
```

---

## 连接示例

### ClickHouse（1Panel 环境）

```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host='172.19.0.2',   # 扫描得到的 IP，需完整扫描网段
    port=8123,            # HTTP 接口（不是 9000）
    username='admin',     # 1Panel 安装默认用户是 admin，不是 default
    password='your_password'
)

# 查询
result = client.query('SHOW DATABASES')
print([r[0] for r in result.result_rows])

# 写入
client.insert('table_name', data, column_names=['col1', 'col2'])
```

### DuckDB（文件型）

DuckDB 是嵌入式数据库，通过**共享卷**访问文件：

```python
import duckdb

# 挂载共享卷后直接访问
conn = duckdb.connect('/data/my_database.duckdb')
df = conn.execute('SELECT * FROM table').fetchdf()

# DuckDB TCP 服务器模式（如果启动时加了 --tcp）
conn = duckdb.connect('http://<duckdb-host>:7867')
```

### MySQL/MariaDB

```python
import pymysql

conn = pymysql.connect(
    host='<目标IP>',
    port=3306,
    user='root',
    password='xxx',
    database='dbname'
)
cursor = conn.cursor()
cursor.execute('SELECT * FROM table LIMIT 10')
```

---

## Docker Compose 网络配置（推荐）

让 Hermes 和数据库在同一网络：

```yaml
services:
  hermes:
    networks:
      - db_network
    volumes:
      - /opt/data:/opt/data

  clickhouse:
    networks:
      - db_network
    ports:
      - "8123:8123"
      - "9000:9000"

  duckdb:
    networks:
      - db_network
    volumes:
      - ./data:/data  # 共享卷

networks:
  db_network:
    driver: bridge
```

---

## 快速检查命令

```bash
# 检查网络
cat /etc/hosts | grep 172

# 扫描端口（Python）
/opt/data/venv/bin/python3 -c "
import socket
for port in [8123, 9000, 5432, 3306]:
    try:
        s = socket.socket()
        s.settimeout(1)
        r = s.connect_ex(('172.19.0.1', port))
        print(f'Port {port}: {'✅' if r==0 else '❌'}')
        s.close()
    except: pass
"

# 验证写入权限（btrfs 挂载卷）
touch /opt/knowledge/test_write 2>&1 && echo "写入OK" && rm /opt/knowledge/test_write

# 检查 pip 路径
which pip  # 通常不存在
/opt/data/venv/bin/pip --version  # 正确路径
```

---

## 1Panel + 飞牛OS 实战发现（重要）

### 1Panel 创建的 Docker 网络

1Panel 通常创建两个网络：
- `1panel-network`: 172.19.0.0/16, 网关 172.19.0.1（Hermes 容器在此网络）
- `bridge`: 172.17.0.0/16, 网关 172.17.0.1（默认 Docker bridge）

### ClickHouse 容器多网络行为

通过 1Panel 安装的 ClickHouse 容器通常同时加入了 `1panel-network` 和 `bridge` 两个网络。
**关键发现：ClickHouse 在 1panel-network 上的 IP 不是网关地址，而是另一个地址（如 172.19.0.2），
需要扫描整个 172.19.0.x 网段才能找到，不能只扫 172.19.0.1。**

### 扫描整个网段找 ClickHouse

```python
import socket

# 扫描 172.19.0.x 整个网段（排除网关和广播）
for host_suffix in range(2, 255):
    host = f'172.19.0.{host_suffix}'
    for port in [8123, 9000]:
        try:
            sock = socket.socket()
            sock.settimeout(0.1)
            r = sock.connect_ex((host, port))
            if r == 0:
                print(f"✅ {host}:{port} OPEN")
            sock.close()
        except:
            pass
```

### 1Panel ClickHouse 认证信息

| 项目 | 值 |
|------|-----|
| 用户名 | `admin`（不是 `default`） |
| 密码 | 与 1Panel 安装时设置的管理密码相同 |
| HTTP 端口 | 8123 |
| TCP 端口 | 9000（原生协议，非 HTTP） |
| 连接库 | `clickhouse_connect`（HTTP 接口） |

```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host='172.19.0.2',   # 扫描得到的 IP，不是网关
    port=8123,            # HTTP 接口
    username='admin',     # 1Panel 安装的 ClickHouse 用户是 admin
    password='your_password',
    connect_timeout=10
)
print(client.server_version)  # 查询版本验证连接
```

### ClickHouse vs DuckDB vs MySQL 在此环境的连接

| 数据库 | 发现方式 | IP | 端口 | 用户 |
|--------|----------|-----|------|------|
| MySQL | 扫 172.19.0.1 | 172.19.0.1 | 3306 | root |
| ClickHouse | 扫 172.19.0.x | 172.19.0.2 | 8123 | admin |
| DuckDB | 文件路径访问 | - | - | - |

---

## 已知问题

1. **Docker Socket 未挂载** — 无法执行 `docker exec` 进入数据库容器
2. **无网络扫描工具** — 需要用 Python socket 自己写扫描脚本
3. **pip 在系统 Python 不可用** — 必须用 `/opt/data/venv/bin/pip`
4. **/opt/data/venv/bin/pip 安装的包只在 venv 中可用** — 脚本要用 `/opt/data/venv/bin/python3` 运行
5. **ClickHouse 在多网络环境下** — 不能只扫网关地址，要扫整个 172.19.0.x 网段
6. **ClickHouse 端口 9000 是 TCP 原生协议** — `clickhouse_connect` 用 HTTP 8123，Python clickhouse_driver 用 TCP 9000
