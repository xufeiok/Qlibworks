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

DUCKDB_PATH = _resolve_path('DUCKDB_PATH', './data/cache/quant_cache.duckdb')
DATA_DIR = PROJECT_ROOT / 'data'
CSV_DIR = DATA_DIR / 'csv'
QLIB_DATA_DIR = _resolve_path('QLIB_DATA_DIR', 'qlib_data')

CH_HOST = os.environ.get('CH_HOST', '10.100.0.205')
CH_PORT = int(os.environ.get('CH_PORT', '18123'))
CH_USER = os.environ.get('CH_USER', 'reader')
CH_PASSWORD = os.environ.get('CH_PASSWORD', '')
CH_DATABASE = os.environ.get('CH_DATABASE', 'quant_db')

FS_CACHE_DIR = _resolve_path('FS_CACHE_DIR', '.fs_cache')

# ==================== 数据规范配置 ====================
# 强制使用前复权价格（不复权数据会导致因子失效）
FORCE_ADJUSTED_PRICES = os.environ.get('FORCE_ADJUSTED_PRICES', 'true').lower() == 'true'

# 财报数据使用公告日期（ann_date），而非期末日期（end_date）
# 公告日期是数据正式发布日期，更适合量化因子计算
FINANCIAL_USE_ANNOUNCEMENT_DATE = os.environ.get('FINANCIAL_USE_ANNOUNCEMENT_DATE', 'true').lower() == 'true'

# 财报日期字段映射（ClickHouse 表中的列名）
FINANCIAL_DATE_COLUMNS = {
    'ann_date': 'ann_date',      # 公告日期（优先使用）
    'end_date': 'end_date',      # 期末日期（不使用）
}

# 价格复权类型：'qfq'=前复权, 'hfq'=后复权, None=不复权
# 前复权是量化标准做法，确保价格连续性和可比性
ADJUSTED_PRICE_TYPE = os.environ.get('ADJUSTED_PRICE_TYPE', 'qfq')
