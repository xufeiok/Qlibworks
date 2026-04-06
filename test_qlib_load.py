import qlib
from qlib.data import D

if __name__ == '__main__':
    qlib.init(provider_uri='E:/Quant/Qlibworks/qlib_data', region='cn')
    data = D.features(['000001.sz', '000002.sz'], ['$close', '$volume', '$open', '$high', '$low'])
    print('Test loaded rows:', len(data))
    print(data.head())
    print(data.tail())
