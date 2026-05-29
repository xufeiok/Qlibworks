import qlib
import sys, os
import pandas as pd
pd.options.mode.use_inf_as_na = True

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from qlib.data import D
from qlworks.features.builder import build_factor_library_bundle

qlib.init(provider_uri=r'E:/Quant/Qlibworks/qlib_data', joblib_backend="threading", maxtasksperchild=None)

from drafts.factor_screening import load_csi500_instruments

if __name__ == '__main__':
    insts = load_csi500_instruments()
    bundle = build_factor_library_bundle(["style_factors", "quality_factors", "price_volume_factors", "sentiment_factors", "risk_factors"])
    
    print("Checking each field for 000017.SZ...")
    bad_fields = []
    for field in bundle.fields:
        try:
            df = D.features(["000017.SZ"], [field], start_time='2023-01-01', end_time='2023-03-31')
        except Exception as e:
            print(f"Error with field {field}: {e}")
            bad_fields.append(field)
            
    print(f"Done! Found {len(bad_fields)} bad fields.")
    for f in bad_fields:
        print(f)
