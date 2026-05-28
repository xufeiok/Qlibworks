import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _p = Path(__file__).resolve().parent.parent.parent / '.env'
    if _p.exists():
        load_dotenv(dotenv_path=str(_p), override=True)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _resolve_path(env_key, default_val):
    val = os.environ.get(env_key)
    if val:
        p = Path(val)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    p_def = Path(default_val)
    return p_def if p_def.is_absolute() else (PROJECT_ROOT / p_def).resolve()

DUCKDB_PATH = _resolve_path('DUCKDB_PATH', 'e:/Quant/Quant_Tushare/data/quant_data.duckdb')
DATA_DIR = PROJECT_ROOT / 'data'
CSV_DIR = DATA_DIR / 'csv'
QLIB_DATA_DIR = _resolve_path('QLIB_DATA_DIR', 'qlib_data')

CH_HOST = os.environ.get('CH_HOST', '192.168.10.102')
CH_PORT = int(os.environ.get('CH_PORT', '18123'))
CH_USER = os.environ.get('CH_USER', 'xufei')
CH_PASSWORD = os.environ.get('CH_PASSWORD', 'xf1987216')
CH_DATABASE = os.environ.get('CH_DATABASE', 'quant_db')

FS_CACHE_DIR = _resolve_path('FS_CACHE_DIR', '.fs_cache')
