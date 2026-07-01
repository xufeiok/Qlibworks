---
name: database-operations
description: 通用数据库操作——PostgreSQL、MySQL、SQLite、MongoDB 的连接、查询、数据导出、备份。触发：连接数据库、查询数据、导出表、查看表结构、执行 SQL、数据库备份。
---

# 通用数据库操作

## 连接配置

### PostgreSQL

```python
import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=5432,
    dbname='mydb',
    user='postgres',
    password='xxx'
)
cursor = conn.cursor()
cursor.execute('SELECT * FROM table LIMIT 10')
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
```

### MySQL

```python
import pymysql

conn = pymysql.connect(
    host='localhost',
    port=3306,
    dbname='mydb',
    user='root',
    password='xxx'
)
cursor = conn.cursor()
cursor.execute('SELECT * FROM table LIMIT 10')
rows = cursor.fetchall()
conn.close()
```

### SQLite

```python
import sqlite3

conn = sqlite3.connect('/path/to/db.sqlite')
cursor = conn.cursor()
cursor.execute('SELECT * FROM table LIMIT 10')
rows = cursor.fetchall()
conn.close()
```

### MongoDB

```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['mydb']
collection = db['mycollection']
for doc in collection.find().limit(10):
    print(doc)
```

---

## 常用操作

### 查看表/集合列表

```python
# PostgreSQL / MySQL
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'  # public for PG, 'database_name' for MySQL
""")
print([r[0] for r in cursor.fetchall()])

# SQLite
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print([r[0] for r in cursor.fetchall()])

# MongoDB
print(db.list_collection_names())
```

### 查看表结构

```python
# PostgreSQL
cursor.execute("""
    SELECT column_name, data_type, is_nullable 
    FROM information_schema.columns 
    WHERE table_name = 'your_table'
    ORDER BY ordinal_position
""")

# MySQL
cursor.execute('DESCRIBE your_table')

# SQLite
cursor.execute('PRAGMA table_info(your_table)')
```

### 导出数据到 CSV

```python
import csv

cursor.execute('SELECT * FROM your_table')
rows = cursor.fetchall()
headers = [desc[0] for desc in cursor.description]

with open('export.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)
print(f'已导出 {len(rows)} 行到 export.csv')
```

### 备份数据库

```bash
# PostgreSQL
pg_dump -h localhost -U postgres -d mydb -f backup.sql

# MySQL
mysqldump -h localhost -u root -p mydb > backup.sql

# SQLite (直接复制文件)
cp /path/to/db.sqlite /path/to/db.sqlite.backup
```

### 批量插入

```python
data = [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]

# PostgreSQL
from psycopg2.extras import execute_values
cursor.executemany('INSERT INTO users (id, name) VALUES (%s, %s)', data)
conn.commit()

# MySQL
cursor.executemany('INSERT INTO users (id, name) VALUES (%s, %s)', data)
conn.commit()
```

---

## 环境注意

- Python 驱动安装：`/opt/data/venv/bin/pip install psycopg2-binary pymysql pymongo`
- 连接超时：生产环境务必加 `connect_timeout=10`
- 密码别硬编码，用环境变量或配置文件
- 执行写操作后记得 `conn.commit()`
