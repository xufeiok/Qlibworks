import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
# note: moved from scripts/ to tests/ — '../src' now goes through project root correctly
import qlib
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR

if __name__ == '__main__':
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region='cn')
    df = D.features(['SH600519', 'SZ000001'], ['$sw_l1', '$sw_l2', '$sw_l3'], start_time='2024-01-02', end_time='2024-01-05')
    print(df)
