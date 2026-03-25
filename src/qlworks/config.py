from pathlib import Path


DUCKDB_PATH = Path(r"c:\xfworks\Quant_Tushare\data\quant_data.duckdb")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CSV_DIR = DATA_DIR / "csv"
QLIB_DATA_DIR = PROJECT_ROOT / "qlib_data"
