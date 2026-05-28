import qlib
from qlib.data import D

if __name__ == '__main__':
    qlib.init(provider_uri='e:/Quant/Qlibworks/qlib_data', region='cn')
    df = D.features(['SH600519', 'SZ000001'], ['$sw_l1', '$sw_l2', '$sw_l3'], start_time='2024-01-02', end_time='2024-01-05')
    print(df)
