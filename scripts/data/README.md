# 数据管理脚本使用说明

## 概述

本目录包含 Qlibworks 项目的数据管理脚本，基于 **ClickHouse + DuckDB + Parquet + Qlib** 架构设计。

## 架构设计

### 核心原则

1. **唯一真实数据源 (SSOT)**: 远程 ClickHouse 是唯一数据源
2. **工具专精**: 每个工具只做自己擅长的事
3. **统一访问**: 通过 `QuantDataAPI` 统一访问数据
4. **分层缓存**: DuckDB 作为智能缓存层
5. **自动化**: 所有重复操作自动化

### 数据流向

```
ClickHouse (远程) → DuckDB (本地缓存) → Qlib/Parquet (专用格式)
```

## 配置说明

### 环境变量

在项目根目录创建 `.env` 文件（参考 `.env.example`）：

```bash
# ClickHouse 连接配置（使用 reader 用户）
CH_HOST=10.100.0.205
CH_PORT=18123
CH_USER=reader
CH_PASSWORD=
CH_DATABASE=quant_db

# DuckDB 路径
DUCKDB_PATH=./quant_cache.duckdb

# Qlib 数据目录
QLIB_DATA_DIR=./qlib_data

# 缓存过期时间（秒）
CACHE_TTL=86400
```

## 使用方法

### 1. 数据同步

#### 全量同步（首次初始化）

```bash
# 同步 2010-2025 年的数据
python -m scripts.data.sync full --start_date 2010-01-01 --end_date 2025-12-31
```

#### 增量同步（每日更新）

```bash
# 自动检测并同步最新数据
python -m scripts.data.sync incremental
```

#### 同步财务数据

```bash
# 同步财务指标到 Parquet 特征库
python -m scripts.data.sync financial
```

### 2. 数据验证

```bash
# 运行数据完整性检查
python -m scripts.data.verify
```

验证内容：
- ✅ 交易日历连续性
- ✅ 股票池文件完整性
- ✅ 特征文件完整性
- ✅ 与 ClickHouse 数据一致性

### 3. 编程接口

#### 使用 QuantDataAPI

```python
from qlworks.data.api import QuantDataAPI

# 使用上下文管理器（自动关闭连接）
with QuantDataAPI() as api:
    # 查询 ClickHouse
    df = api.query("SELECT * FROM daily_prices LIMIT 100")
    
    # 获取日线数据（默认强制前复权）
    daily = api.get_daily_data(
        ts_codes=['600000.SH', '000001.SZ'],
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    # 获取交易日历
    calendar = api.get_calendar(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    # 获取股票列表
    stocks = api.get_stock_list(market='主板', status='L')
    
    # 保存特征
    api.save_feature(
        df=df,
        name='momentum_factors',
        version='1.0',
        description='动量因子',
        category='technical'
    )
    
    # 加载特征
    factors = api.load_feature('momentum_factors')
    
    # 同步 Qlib 数据
    api.sync_qlib_full('2010-01-01', '2025-12-31')
    
    # 检查一致性
    results = api.check_consistency()
```

#### 使用 QlibSynchronizer

```python
from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer

with QuantDataAPI() as api:
    syncer = QlibSynchronizer(api)
    
    # 全量同步
    syncer.full_sync('2010-01-01', '2025-12-31')
    
    # 增量同步
    syncer.incremental_sync()
```

## 目录结构

```
scripts/data/
├── __init__.py          # 包初始化
├── sync.py              # 数据同步脚本
└── verify.py            # 数据验证脚本

src/qlworks/data/
├── __init__.py          # 导出 API
├── api.py               # QuantDataAPI 统一接口
├── qlib_sync.py         # Qlib 数据同步器
├── access.py            # Qlib 数据访问（已有）
├── cleaning.py          # 数据清洗（已有）
└── quality.py           # 数据质量（已有）
```

## 最佳实践

### 1. 每日自动化流程

创建定时任务（如 Windows Task Scheduler）：

```bash
# 每日 18:30 执行增量更新
python -m scripts.data.sync incremental

# 验证数据一致性
python -m scripts.data.verify
```

### 2. 特征管理

```python
# 保存新版本特征
api.save_feature(df, 'value_factors', '2.0', category='fundamental')

# 加载最新版本
factors = api.load_feature('value_factors')

# 加载指定版本
factors_v1 = api.load_feature('value_factors', version='1.0')

# 列出所有特征
all_features = api.list_features()
```

### 3. 缓存管理

```python
# 清理过期缓存
api.clear_cache()

# 清理指定 TTL 的缓存
api.clear_cache(ttl=3600)  # 清理 1 小时前的缓存
```

## 故障排查

### 问题 1: ClickHouse 连接失败

**症状**: `Connection refused` 或 `Authentication failed`

**解决**:
1. 检查 `.env` 文件中的配置是否正确
2. 确认 ClickHouse 服务是否运行
3. 确认 `reader` 用户权限

### 问题 2: DuckDB 路径错误

**症状**: `Cannot open file ... quant_data.duckdb`

**解决**:
1. 检查 `DUCKDB_PATH` 配置
2. 确保目录存在
3. 首次运行会自动创建

### 问题 3: Qlib 数据不一致

**症状**: 验证报告显示 `Qlib 与 ClickHouse 数据不一致`

**解决**:
```bash
# 执行增量同步
python -m scripts.data.sync incremental
```

## 版本历史

- **v2.0** (2026-05-29): 重构为统一 API 架构，使用 reader 用户
  - 新增 `QuantDataAPI` 统一接口
  - 新增 `QlibSynchronizer` 同步器
  - 整合重复脚本
  - 移除硬编码配置

- **v1.0**: 初始版本（已废弃）
  - `migrate_qlib_data.py`
  - `download_missing_factors.py`
  - 多个独立脚本

## 注意事项

1. **不要直接使用旧脚本**: `migrate_qlib_data.py` 等已废弃
2. **使用 reader 用户**: 只读查询，保证数据安全
3. **定期清理缓存**: 避免 DuckDB 文件过大
4. **备份重要数据**: 全量同步前自动备份
