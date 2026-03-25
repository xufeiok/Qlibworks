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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qlib
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR


def main() -> int:
    qlib.init(provider_uri=str(QLIB_DATA_DIR))
    model_path = ROOT / "artifacts" / "linret.joblib"
    if not model_path.exists():
        print("model missing")
        return 2
    model = joblib.load(model_path)
    ins = D.instruments("all", start_time="2012-01-01", end_time="2026-01-01")
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    start = "2022-01-01"
    end = "2024-12-31"
    df = D.features(instruments=ins, fields=fields, start_time=start, end_time=end)
    df = df.dropna()
    piv = df.reset_index().rename(columns={"instrument": "symbol"})
    piv = piv.sort_values(["symbol", "datetime"])
    piv["ret1d_fwd"] = piv.groupby("symbol")["$close"].shift(-1) / piv["$close"] - 1.0
    piv = piv.dropna()
    X = piv[fields].to_numpy()
    piv["pred"] = model.predict(X)
    topn = 50
    rets = []
    for dt, g in piv.groupby("datetime"):
        gg = g.sort_values("pred", ascending=False).head(topn)
        r = float(gg["ret1d_fwd"].mean())
        rets.append((dt, r))
    perf = pd.DataFrame(rets, columns=["date", "ret"]).set_index("date").sort_index()
    perf["cum"] = (1 + perf["ret"]).cumprod()
    n = len(perf)
    if n == 0:
        print("no backtest data")
        return 3
    ann = (1.0 + perf["ret"]).prod() ** (252.0 / n) - 1.0
    vol = float(perf["ret"].std()) * np.sqrt(252.0)
    shp = float(perf["ret"].mean()) / (perf["ret"].std() + 1e-12) * np.sqrt(252.0)
    roll_max = perf["cum"].cummax()
    dd = perf["cum"] / roll_max - 1.0
    mdd = float(dd.min())
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "backtest.csv"
    perf.to_csv(out_csv)
    plt.figure(figsize=(10, 4))
    plt.plot(perf.index, perf["cum"], label="cum")
    plt.title("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "backtest.png"
    plt.savefig(out_png)
    print(str(out_csv))
    print(str(out_png))
    print(f"annualized={ann:.6f}, vol={vol:.6f}, sharpe={shp:.6f}, max_dd={mdd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
