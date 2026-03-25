import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import qlib
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR


def main() -> int:
    qlib.init(provider_uri=str(QLIB_DATA_DIR))
    ins = D.instruments("all", start_time="2007-01-01", end_time="2026-01-01")
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(
        instruments=ins,
        fields=fields,
        start_time="2012-01-01",
        end_time="2024-12-31",
    )
    df = df.dropna()
    piv = df.reset_index().rename(columns={"instrument": "symbol"})
    piv = piv.sort_values(["symbol", "datetime"])
    piv["ret1d"] = piv.groupby("symbol")["$close"].shift(-1) / piv["$close"] - 1.0
    piv = piv.dropna()
    mask = (piv["datetime"] < pd.Timestamp("2022-01-01")) & (piv["datetime"] >= pd.Timestamp("2013-01-01"))
    train = piv[mask]
    X = train[fields].to_numpy()
    y = train["ret1d"].to_numpy()
    if len(train) < 1000:
        print("insufficient training data")
        return 2
    model = LinearRegression()
    model.fit(X, y)
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "linret.joblib")
    print(str(out_dir / "linret.joblib"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
