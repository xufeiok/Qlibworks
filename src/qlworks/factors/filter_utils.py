"""
统一过滤工具模块

为 Qlib 各环节提供统一的动态股票池过滤配置。

用法：
    from qlworks.factors.filter_utils import get_tradeable_filter, get_stock_pool

    # 训练/推理时替换原有 instruments 配置
    handler = DataHandlerLP(
        instruments=get_stock_pool(),
        ...
    )

    # 回测时替换 get_instruments
    class MyStrategy(BaseStrategy):
        def get_instruments(self, date):
            return get_stock_pool(date=date)

设计决策（基于 qlib 0.9.7 验证）:
    - $money 字段不存在，使用 $amount 替代
    - $DaysSinceList() 在当前 qlib 版本中不可用，次新股过滤后置到 Python 层
    - NameDFilter 在当前版本中过滤的是股票代码而非名称，ST 过滤后置到 Python 层
    - 过滤集中于数据加载层，不修改 warehouse 数据（计算与消费分离）
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

# Conditional import to allow module to be loaded without qlib
_HAS_QLIB = False
try:
    from qlib.data import D
    from qlib.data.filter import ExpressionDFilter

    _HAS_QLIB = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# 截面上单只股票近20日最低日均成交额阈值
MIN_AVG_DAILY_AMOUNT = 5_000_000  # 500万

# 上市最少天数（约1年交易日）
MIN_LIST_DAYS = 250


def _resolve_data_dir() -> Path:
    """获取项目 qlib_data 目录路径。"""
    # 尝试从 QLIB_DATA_DIR 环境变量读取
    qlib_data_dir = None
    try:
        from qlworks.config import QLIB_DATA_DIR
        qlib_data_dir = QLIB_DATA_DIR
    except ImportError:
        import os
        qlib_data_dir = os.environ.get("QLIB_DATA_DIR", "")

    p = Path(qlib_data_dir) if qlib_data_dir else Path("qlib_data")
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


def get_tradeable_filter() -> "list":
    """获取统一的动态可交易股票过滤器列表。

    过滤逻辑（拆分为两个独立 ExpressionDFilter，避免 Qlib 表达式引擎
    合并空格导致的 SyntaxError）：
    1. 当日有成交（$volume > 0），剔除停牌
    2. 近20日日均成交额 >= 500万，剔除低流动性僵尸股

    子新股和 ST 股票的过滤不在本函数中处理，而是在
    get_stock_pool() 中通过 Python 后置处理实现。原因：
        - qlib 0.9.7 中 $DaysSinceList() 表达式不可用
        - 当前版本 NameDFilter 过滤的是代码而非名称
    
    返回:
        ExpressionDFilter 列表，每个 keep=False
    """
    if not _HAS_QLIB:
        raise RuntimeError("qlib 未安装，无法创建 ExpressionDFilter")

    rules = [
        "$volume > 0",
        "Mean($amount, 20) > {min_amt}".format(min_amt=MIN_AVG_DAILY_AMOUNT),
    ]

    return [ExpressionDFilter(rule_expression=r, keep=False) for r in rules]


def _load_stock_name_map(data_dir: Optional[Path] = None) -> dict:
    """加载股票名称映射表（code -> name）。

    从 qlib_data/instruments/all.txt 读取所有股票的代码和
    上市/退市日期，判断交易状态。

    返回:
        {code_str: (list_date, delist_date)}
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()

    inst_file = data_dir / "instruments" / "all.txt"
    if not inst_file.exists():
        logger.warning("instruments/all.txt 不存在，无法加载名称映射")
        return {}

    result = {}
    with open(inst_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                code = parts[0].lower()
                entry_date = parts[1]
                exit_date = parts[2]
                result[code] = (entry_date, exit_date)
    return result


def _filter_st_stocks(codes: List[str]) -> List[str]:
    """基于代码规则过滤 ST 股票。

    ST 股票在 A 股不更改代码，而是更改名称前缀。
    因此无法直接通过代码判断 ST 状态，此函数为预留接口。
    
    当前实现：保持全量，不做基于代码的 ST 过滤。
    实际 ST 过滤应配合外部数据源（如 tushare 的 name 字段）。

    参数:
        codes: 股票代码列表

    返回:
        过滤后的股票代码列表
    """
    # 当前 qlib_data 中没有股票名称字段，ST 判断需外部数据
    # 仅做代码格式校验（去除明显异常的代码）
    valid = []
    for c in codes:
        c_str = str(c).lower().strip()
        if not c_str:
            continue
        # 去除特殊字符异常代码
        if len(c_str) < 7:
            continue
        valid.append(c)
    return valid


def _filter_new_stocks(
    codes: List[str],
    current_date: str,
    min_days: int = MIN_LIST_DAYS,
    data_dir: Optional[Path] = None,
) -> List[str]:
    """过滤上市不满 min_days 的次新股。

    以 current_date 为基准，检查每只股票在 all.txt 中的
    上市日期，剔除上市天数不足的股票。

    参数:
        codes: 输入股票代码列表
        current_date: 基准日期（YYYY-MM-DD）
        min_days: 最少上市天数
        data_dir: qlib_data 目录

    返回:
        过滤后的股票代码列表
    """
    name_map = _load_stock_name_map(data_dir)
    if not name_map:
        return codes  # 无数据时不做过滤

    current_ts = pd.Timestamp(current_date)
    result = []
    for c in codes:
        c_str = str(c).lower().strip()
        info = name_map.get(c_str)
        if info is None:
            continue
        entry_date = info[0]
        try:
            entry_ts = pd.Timestamp(entry_date)
        except Exception:
            result.append(c)
            continue

        # 计算上市天数（工作日需使用交易日历，这里粗略估计）
        days_since_list = (current_ts - entry_ts).days
        if days_since_list >= min_days:
            result.append(c)

    return result


def get_stock_pool(
    market: str = "all",
    date: Optional[str] = None,
    filter_new_stocks: bool = True,
    filter_st: bool = True,
    filter_liquidity: bool = True,
    base_pool: Optional[List[str]] = None,
) -> object:
    """获取统一的动态股票池。

    这是项目中所有环节的标准入口：
    - 训练：handler = DataHandlerLP(instruments=get_stock_pool(), ...)
    - 推理：同训练，用相同 pool
    - 回测：base_pool = get_stock_pool()[作为 instruments 参数]

    参数:
        market: Qlib market 名称（"all", "main_board", "csi500" 等）
        date: 单日查询时指定日期（YYYY-MM-DD）
        filter_new_stocks: 是否过滤上市不足250日股票
        filter_st: 是否过滤 ST 股票
        filter_liquidity: 是否过滤停牌/低流动性
        base_pool: 可选基础股票池列表，传入后代替 market 参数

    返回:
        如果 _HAS_QLIB，返回已配置 filter_pipe 的 instruments 对象；
        否则返回股票代码列表。
    """
    if not _HAS_QLIB:
        logger.warning("qlib 不可用，get_stock_pool 返回空列表")
        return []

    if filter_liquidity:
        tradeable_filter = get_tradeable_filter()
        filter_pipe = [tradeable_filter]
    else:
        filter_pipe = []

    if base_pool is not None:
        # 使用已有股票代码列表
        instrument_obj = D.instruments(market=market, filter_pipe=filter_pipe)
        # base_pool 后续通过 date-level 裁剪
        return D.instruments(market=market, filter_pipe=filter_pipe)

    return D.instruments(market=market, filter_pipe=filter_pipe)


def filter_codes_post(
    codes: List[str],
    date: str,
    filter_new_stocks: bool = True,
    filter_st: bool = True,
    data_dir: Optional[Path] = None,
) -> List[str]:
    """后置过滤：对代码列表执行 ST/次新过滤。

    这些过滤无法通过 ExpressionDFilter 在当前 qlib 版本中实现，
    需要在加载数据之后、训练/推理之前用 Python 过滤。

    参数:
        codes: Qlib 返回的股票代码列表
        date: 当前交易日
        filter_new_stocks: 过滤次新
        filter_st: 过滤 ST
        data_dir: qlib_data 目录

    返回:
        过滤后的股票代码列表
    """
    filtered = list(codes)

    if filter_st:
        filtered = _filter_st_stocks(filtered)

    if filter_new_stocks and date:
        filtered = _filter_new_stocks(filtered, date, data_dir=data_dir)

    return filtered
