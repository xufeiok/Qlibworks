"""测试 YAML DuckDB 表达式 CTE 分解"""
import sys
sys.path.insert(0, r"e:\Quant\Qlibworks\src")
sys.path.insert(0, r"e:\Quant\Qlibworks\scripts\evaluation")

import compute_all_missing as cam

# Alpha1 YAML DuckDB 表达式
alpha1 = """(-1 * CORR(PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY (LN(volume)
- LAG(LN(volume), 1) OVER (PARTITION BY ts_code ORDER BY trade_date))), PERCENT_RANK()
OVER (PARTITION BY trade_date ORDER BY ((close-open)/open))) OVER (PARTITION
BY ts_code ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND CURRENT ROW))"""

print("=== Alpha1 flatten ===")
result = cam._flatten_duckdb_expr(alpha1)
print(result[:600])
print("...")
print()

# Alpha5
alpha5 = """(-1*MAX(CORR((RANK() OVER (PARTITION BY ts_code ORDER BY high ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)),
(RANK() OVER (PARTITION BY ts_code ORDER BY volume ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)))
OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW))
OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))"""

print("=== Alpha5 flatten ===")
result = cam._flatten_duckdb_expr(alpha5)
print(result[:600])
print("...")
print()

# Alpha6
alpha6 = """(PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY SIGN((((open*0.85)+(high*0.15))-LAG(((open*0.85)+(high*0.15)),4) 
OVER (PARTITION BY ts_code ORDER BY trade_date))))*-1)"""

print("=== Alpha6 flatten ===")
result = cam._flatten_duckdb_expr(alpha6)
print(result[:600])
print("...")
print()
