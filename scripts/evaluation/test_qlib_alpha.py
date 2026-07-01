"""测试 Qlib 能否计算 Alpha1 因子"""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, r"e:\Quant\Qlibworks\src")
os.environ["FORCE_QLIB"] = "true"

import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR

qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN,
          joblib_backend="threading", maxtasksperchild=1)
from qlib.config import C as _QC
_QC.dataloader_workers = 0

# 读取股票池
pool = []
ifile = str(QLIB_DATA_DIR) + "/instruments/all.txt"
with open(ifile) as f:
    for line in f:
        p = line.strip().split()
        if p:
            pool.append(p[0])
print(f"股票池: {len(pool)} 只")

# 测试 Alpha1 - 使用 YAML 中的原始表达式（无转义）
expr = "(-1 * Corr(Rank(Delta(Log($volume),1)),Rank((($close-$open)/$open)),6))"
print(f"表达式: {expr}")

try:
    df = D.features(pool[:50], [expr], "2020-01-01", "2020-01-31")
    print(f"Alpha1 结果: {len(df)} 行")
    print(df.head(10))
except Exception as e:
    print(f"Alpha1 失败: {e}")

# 测试一个简单因子
expr2 = "$close-Ref($close,5)"
print(f"\n表达式2: {expr2}")
try:
    df2 = D.features(pool[:50], [expr2], "2020-01-01", "2020-01-31")
    print(f"结果: {len(df2)} 行")
    print(df2.head(10))
except Exception as e:
    print(f"失败: {e}")
