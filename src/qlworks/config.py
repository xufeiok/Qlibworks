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

DUCKDB_PATH = _resolve_path('DUCKDB_PATH', './factor_data/cache/quant_cache.duckdb')
QLIB_DATA_DIR = _resolve_path('QLIB_DATA_DIR', 'qlib_data')

CH_HOST = os.environ.get('CH_HOST', '192.168.10.102')
CH_PORT = int(os.environ.get('CH_PORT', '18123'))
CH_USER = os.environ.get('CH_USER', 'reader')
CH_PASSWORD = os.environ.get('CH_PASSWORD', '')
CH_DATABASE = os.environ.get('CH_DATABASE', 'quant_db')

FS_CACHE_DIR = _resolve_path('FS_CACHE_DIR', 'factor_data/cache')
FACTOR_CACHE_DIR = FS_CACHE_DIR / 'factors'

# ==================== 鏁版嵁瑙勮寖閰嶇疆 ====================
FORCE_ADJUSTED_PRICES = os.environ.get('FORCE_ADJUSTED_PRICES', 'true').lower() == 'true'

FINANCIAL_USE_ANNOUNCEMENT_DATE = os.environ.get('FINANCIAL_USE_ANNOUNCEMENT_DATE', 'true').lower() == 'true'

FINANCIAL_DATE_COLUMNS = {
    'ann_date': 'ann_date',
    'end_date': 'end_date',
}

ADJUSTED_PRICE_TYPE = os.environ.get('ADJUSTED_PRICE_TYPE', 'qfq')

# ==================== 浜ゆ槗璐圭巼閰嶇疆 ====================
STAMP_DUTY = float(os.environ.get('STAMP_DUTY', '0.0005'))  # 鍗拌姳绋庯紙鍗栧嚭鍗曞悜锛夛紝褰撳墠A鑲′竾鍒嗕箣浜?
COMMISSION = float(os.environ.get('COMMISSION', '0.0003'))  # 鍒稿晢浣ｉ噾锛堝弻鍚戯級锛岄粯璁や竾鍒嗕箣涓?

