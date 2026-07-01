"""快速验证 Qlib 能否计算复杂 Alpha 因子"""
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

# 股票池
pool = [p.strip().split()[0] for p in open(str(QLIB_DATA_DIR)+"/instruments/all.txt") if p.strip()]
print(f"股票池: {len(pool)} 只")

# 测试 Alpha1
expr1 = '(-1 * Corr(Rank(Delta(Log($volume),1)),Rank((($close-$open)/$open)),6))'
try:
    df = D.features(pool[:100], [expr1], "2020-01-01", "2020-01-31")
    print(f"Alpha1: {len(df)} 行")
    if not df.empty:
        print(df.head(5))
except Exception as e:
    print(f"Alpha1 失败: {e}")

# 测试 Alpha5
expr5 = '(-1*Ts_Max(Corr(Ts_Rank($volume,5),Ts_Rank($high,5),5),3))'
try:
    df = D.features(pool[:100], [expr5], "2020-01-01", "2020-01-31")
    print(f"Alpha5: {len(df)} 行")
    if not df.empty:
        print(df.head(5))
except Exception as e:
    print(f"Alpha5 失败: {e}")

# 测试 Alpha191 (常见简单因子)
expr191 = '((-1*Corr(Rank($close),Rank($volume),2)*Rank((($close-$open)/$open)))*Corr($close,Mean($volume,50),10))'
try:
    df = D.features(pool[:100], [expr191], "2020-01-01", "2020-01-31")
    print(f"Alpha191: {len(df)} 行")
    if not df.empty:
        print(df.head(5))
except Exception as e:
    print(f"Alpha191 失败: {e}")
