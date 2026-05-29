import qlib
import sys, os
import pandas as pd
pd.options.mode.use_inf_as_na = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from qlib.data import D

qlib.init(provider_uri=r'E:/Quant/Qlibworks/qlib_data', joblib_backend="threading", maxtasksperchild=None)

inst = "000001.SZ"
fields = ["$close"]
df = D.features([inst], fields, start_time='2023-01-01', end_time='2023-03-31')
print("000001.SZ features loaded successfully:")
print(df.head())

