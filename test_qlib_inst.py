import qlib
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from qlib.data import D
from qlworks.features.builder import build_factor_library_bundle

qlib.init(provider_uri=r'E:/Quant/Qlibworks/qlib_data')

if __name__ == '__main__':
    insts = ['000001.SZ', '000002.SZ', '600000.SH']
    bundle = build_factor_library_bundle(["style_factors", "quality_factors"])
    print("Fields:", len(bundle.fields))
    df = D.features(insts, bundle.fields, start_time='2023-01-01', end_time='2023-12-31')
    print("df size:", len(df))
    print(df.head())
