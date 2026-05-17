import qlib
import sys, os
import pandas as pd
pd.options.mode.use_inf_as_na = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from qlib.data import D
import qlib.data.storage.file_storage as fs

original_getitem = fs.FeatureStorage.__getitem__
def new_getitem(self, key):
    try:
        return original_getitem(self, key)
    except Exception as e:
        print(f"Error in __getitem__: key={key}, self.start_index={self.start_index}, self.end_index={self.end_index}")
        raise

fs.FeatureStorage.__getitem__ = new_getitem

qlib.init(provider_uri=r'E:/Quant/Qlibworks/qlib_data', joblib_backend="threading", maxtasksperchild=None)

inst = "000001.SZ"
fields = ["$close", "$open", "$high", "$low", "$volume"]
for f in fields:
    try:
        df = D.features([inst], [f], start_time='2023-01-01', end_time='2023-03-31')
        print(f"Field {f} loaded successfully for {inst}. Length: {len(df)}")
    except Exception as e:
        print(f"Error loading field {f}: {e}")

