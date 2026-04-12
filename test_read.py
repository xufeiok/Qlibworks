import qlib
from qlib.data import D

qlib.init(provider_uri='e:/Quant/Qlibworks/qlib_data')
df = D.features(['000001.SZ'], ['$circ_mv', '$industry_code'], start_time='2026-01-01', end_time='2026-02-01')
print(df)
