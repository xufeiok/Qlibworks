"""验证 Qlib 表达式的替代方案"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"e:\Quant\Qlibworks\src")

import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR
from pathlib import Path

qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN,
          joblib_backend="threading", maxtasksperchild=1)
from qlib.config import C as _QC
_QC.dataloader_workers = 0

pool = [p.strip().split()[0] for p in open(str(QLIB_DATA_DIR)+"/instruments/all.txt") if p.strip()]
batch = pool[:50]
print(f"股票池: {len(pool)} 只, 测试前 50 只")

# 测试 Rank(X,1) 替代 Rank(X)
expr1 = "Rank(Delta(Log($volume),1), 1)"
print(f"\n1. 测试 Rank(X,1): {expr1}")
try:
    df = D.features(batch, [expr1], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
    print(df.head(3))
except Exception as e:
    print(f"   失败: {e}")

# 测试 Corr Rank(X,1) 
expr2 = "Corr(Rank(Delta(Log($volume),1),1), Rank((($close-$open)/$open),1), 6)"
print(f"\n2. Corr(Rank(,1), Rank(,1), 6): {expr2}")
try:
    df = D.features(batch, [expr2], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
    print(df.head(3))
except Exception as e:
    print(f"   失败: {e}")

# 测试 Max(X,N) 替代 Ts_Max(X,N)
expr3 = "Max($close, 5)"
print(f"\n3. Max(X,N) 替代 Ts_Max: {expr3}")
try:
    df = D.features(batch, [expr3], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
    print(df.head(3))
except Exception as e:
    print(f"   失败: {e}")

# 测试完整的 Alpha5 用替代写法
# 原: (-1*Ts_Max(Corr(Ts_Rank($volume,5),Ts_Rank($high,5),5),3))
# 改: (-1*Max(Corr(Rank($volume,5),Rank($high,5),5),3))
expr4 = "(-1*Max(Corr(Rank($volume,5),Rank($high,5),5),3))"
print(f"\n4. Alpha5 替代: {expr4}")
try:
    df = D.features(batch, [expr4], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
    print(df.head(3))
except Exception as e:
    print(f"   失败: {e}")

# 测试 Alpha1 用 Rank(,1)
expr5 = "(-1 * Corr(Rank(Delta(Log($volume),1),1),Rank(($close-$open)/$open,1),6))"
print(f"\n5. Alpha1 替代: {expr5}")
try:
    df = D.features(batch, [expr5], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
    print(df.head(3))
except Exception as e:
    print(f"   失败: {e}")
