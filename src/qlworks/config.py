import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _p = Path(__file__).resolve().parents[2] / '.env'
    if not hasattr(load_dotenv, '_already_loaded'):
        if _p.exists():
            load_dotenv(dotenv_path=str(_p), override=True)
        load_dotenv._already_loaded = True
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

# ==================== 凭证配置 ====================
# 统一从环境变量 / 项目根 .env 读取，禁止在代码中硬编码密钥
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')

FS_CACHE_DIR = _resolve_path('FS_CACHE_DIR', 'factor_data/cache')
FACTOR_CACHE_DIR = FS_CACHE_DIR / 'factors'

# ==================== 因子仓库配置 ====================
WAREHOUSE_DIR = _resolve_path('WAREHOUSE_DIR', 'factor_data/warehouse')

# ==================== 成果输出目录 ====================
# 训练等脚本的成果文件（预测得分、因子清单等）统一收纳到 outputs/<脚本名>/<模式>/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_output_dir(script_name: str, mode: str) -> Path:
    """
    返回并自动创建成果输出目录 outputs/<脚本名>/<模式>/。

    入参:
        script_name: 脚本标识，通常取脚本文件名主干（如 "train_tree-doubao"）
        mode: 运行模式（如 "backtest" / "forecast"）
    返回:
        Path: 成果目录，不存在时自动创建
    异常:
        目录创建失败抛出 OSError，由调用方处理
    注意:
        仅用于收纳成果文件，不含模型缓存/中间产物
    """
    out_dir = OUTPUTS_DIR / script_name / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def output_file(script_name: str, mode: str, filename: str) -> Path:
    """
    返回成果文件路径 outputs/<脚本名>/<模式>/<文件名>（只拼接，不创建目录）。

    入参:
        script_name: 产出该成果的脚本名（如 "train_tree-doubao"）
        mode: 运行模式（如 "backtest" / "forecast"）
        filename: 成果文件名（如 "score_tree.csv"）
    返回:
        Path: 成果文件的完整路径
    注意:
        供回测/live 等下游读取使用；目录由产出脚本负责创建
    """
    return OUTPUTS_DIR / script_name / mode / filename

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

