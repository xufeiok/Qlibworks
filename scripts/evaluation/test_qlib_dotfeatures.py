"""验证 Qlib D.features() 能否计算复杂 Alpha 因子"""
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
print(f"股票池: {len(pool)} 只, 测试前 50 只")

batch = pool[:50]

# 简单因子测试
expr0 = "$close - Ref($close, 5)"
print(f"\n1. 测试简单因子: {expr0}")
try:
    df = D.features(batch, [expr0], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行, 列: {list(df.columns)}")
    print(df.head(3))
except Exception as e:
    print(f"   失败: {e}")

# Alpha with Ts_Rank
expr1 = "Rank(Delta(Log($volume),1))"
print(f"\n2. 测试 Rank(Delta...): {expr1}")
try:
    df = D.features(batch, [expr1], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
except Exception as e:
    print(f"   失败: {e}")

# Simple Corr
expr2 = "Corr($close, $volume, 5)"
print(f"\n3. 测试 Corr: {expr2}")
try:
    df = D.features(batch, [expr2], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
except Exception as e:
    print(f"   失败: {e}")

# Alpha1 (complex)
expr3 = "(-1 * Corr(Rank(Delta(Log($volume),1)),Rank((($close-$open)/$open)),6))"
print(f"\n4. 测试 Alpha1: {expr3}")
try:
    df = D.features(batch, [expr3], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
except Exception as e:
    print(f"   失败: {e}")

# Alpha with Ts_Max
expr4 = "Ts_Max($close, 5)"
print(f"\n5. 测试 Ts_Max: {expr4}")
try:
    df = D.features(batch, [expr4], "2020-01-01", "2020-01-31")
    print(f"   结果: {len(df)} 行")
except Exception as e:
    print(f"   失败: {e}")
