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

import functools
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


@functools.lru_cache(maxsize=4)
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

@functools.lru_cache(maxsize=1)
def _load_st_periods(data_dir: Optional[Path] = None) -> dict:
    """Load ST stock periods {code: [(start_date, end_date), ...]} from local cache or tushare."""
    if data_dir is None:
        data_dir = _resolve_data_dir()

    cache_path = data_dir / "instruments" / "st_periods.csv"
    if cache_path.exists():
        import csv
        result = {}
        with open(cache_path, "r", encoding="utf-8") as f_st:
            reader = csv.reader(f_st)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    code, start, end = row[0].strip(), row[1].strip(), row[2].strip()
                    result.setdefault(code, []).append((start, end))
        if result:
            return result

    try:
        import tushare as ts
        import os
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            token_path = Path.home() / ".tushare" / "token.conf"
            if token_path.exists():
                token = token_path.read_text().strip()
        if not token:
            return {}
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.namechange(fields="ts_code,name,start_date,end_date,change_reason")
        if df is None or df.empty:
            return {}
        st_mask = df["name"].str.contains("ST", na=False)
        st_df = df[st_mask].copy()
        if st_df.empty:
            return {}
        result = {}
        for _, row in st_df.iterrows():
            code = row["ts_code"].split(".")[0].lower()
            start = str(row["start_date"]) if pd.notna(row["start_date"]) else "19000101"
            end = str(row["end_date"]) if pd.notna(row["end_date"]) else "20991231"
            result.setdefault(code, []).append((start, end))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8", newline="") as f_st:
            import csv
            writer = csv.writer(f_st)
            writer.writerow(["code", "start_date", "end_date"])
            for code, periods in result.items():
                for s, e in periods:
                    writer.writerow([code, s, e])
        return result
    except Exception:
        return {}


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


def _filter_st_stocks_by_date(codes: List[str], date: str, data_dir: Optional[Path] = None) -> List[str]:
    """Filter stocks that are in ST status on the given date."""
    st_periods = _load_st_periods(data_dir)
    if not st_periods:
        return _filter_st_stocks(codes)

    date_int = int(date.replace("-", ""))
    result = []
    for c in codes:
        c_str = str(c).lower().strip().replace(".sh", "").replace(".sz", "")
        if not c_str or len(c_str) < 6:
            continue
        periods = st_periods.get(c_str, [])
        is_st = False
        for s, e in periods:
            if int(s) <= date_int <= int(e):
                is_st = True
                break
        if not is_st:
            result.append(c)
    return result

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
        filtered = _filter_st_stocks_by_date(filtered, date, data_dir=data_dir)

    if filter_new_stocks and date:
        filtered = _filter_new_stocks(filtered, date, data_dir=data_dir)

    return filtered


# ==============================================================================
# 标签可交易性过滤
# ==============================================================================

def _check_qlib_available():
    """检查 qlib 是否可用。"""
    if not _HAS_QLIB:
        raise RuntimeError("qlib 未安装，无法执行标签可交易性过滤")


def filter_untradeable_labels(label_df, instruments, start_time, end_time):
    """剔除无法买入的标签样本（涨跌停 / 一字板 / 持仓期停牌）。

    根据策略 T+1 开盘买入、T+5 收盘卖出的执行规则，检查：
    1. T+1 开盘时是否涨跌停无法成交或出现一字板跳空
    2. T+5 收盘时是否停牌（成交量=0，收盘价可能为停牌前价格）

    检测逻辑：
    1. 跳空检测：T+1 开盘相对 T 收盘涨跌幅 ≥ ±9%
    2. 一字板检测：T+1 全天 high==low（无成交/无对手盘）
    3. 持仓期末停牌检测：T+5 成交量=0（收盘价不可靠）

    参数：
    - label_df: 标签 DataFrame，MultiIndex(datetime, instrument) 或 plain index
    - instruments: Qlib 股票列表
    - start_time: 起始时间 YYYY-MM-DD
    - end_time: 结束时间 YYYY-MM-DD

    返回:
    - 清洗后的标签 DataFrame（不可交易样本标签为 NaN）
    """
    _check_qlib_available()

    label_df_clean = label_df.copy()
    if label_df_clean.empty:
        return label_df_clean

    label_name = label_df_clean.columns[0]

    # 加载 T+1 开盘价与 T 收盘价
    entry_gap_expr = "Ref($open, -1) / $close - 1"
    price_df = D.features(
        instruments, [entry_gap_expr],
        start_time=start_time, end_time=end_time, freq="day",
    )

    if price_df is None or price_df.empty:
        return label_df_clean

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.droplevel(1)

    gap_col = price_df.columns[0]

    # 对齐 MultiIndex
    if isinstance(label_df_clean.index, pd.MultiIndex):
        price_df = price_df.reindex(label_df_clean.index)
        untradeable_mask = label_df_clean.index.isin(price_df.index)
    else:
        price_df = price_df.reindex(label_df_clean.index)
        untradeable_mask = pd.Series(True, index=label_df_clean.index)

    # 跳空检测：T+1 open / T close - 1 ≥ ±9%
    gap_val = price_df[gap_col]
    untradeable = (gap_val >= 0.09) | (gap_val <= -0.09)

    # 一字板检测：T+1 全天一字板时 open≈close，gap 可能 < 9%，
    # 但实际无成交/无对手盘，无法买入。用 Ref($high,-1)==Ref($low,-1) 取 T+1 日数据。
    try:
        one_liner = D.features(
            instruments, ["Ref($high, -1) == Ref($low, -1)"],
            start_time=start_time, end_time=end_time, freq="day",
        )
        if one_liner is not None and not one_liner.empty:
            if isinstance(one_liner.columns, pd.MultiIndex):
                one_liner.columns = one_liner.columns.droplevel(1)
            one_liner = one_liner.reindex(price_df.index, fill_value=False)
            untradeable = untradeable | (one_liner[one_liner.columns[0]] == True)
    except Exception:
        pass  # 一字板检测失败不影响主流程

    # 持仓期末停牌检测：T+5 成交量=0 时收盘价为停牌前价格，标签不可靠
    # Ref($volume, -5) 取 T+5 日成交量
    try:
        vol_t5 = D.features(
            instruments, ["Ref($volume, -5) == 0"],
            start_time=start_time, end_time=end_time, freq="day",
        )
        if vol_t5 is not None and not vol_t5.empty:
            if isinstance(vol_t5.columns, pd.MultiIndex):
                vol_t5.columns = vol_t5.columns.droplevel(1)
            vol_t5 = vol_t5.reindex(price_df.index, fill_value=False)
            t5_suspended = vol_t5[vol_t5.columns[0]] == True
            untradeable = untradeable | t5_suspended
            _susp_n = t5_suspended.sum()
            if _susp_n > 0:
                print(f"      [停牌检测] 剔除 {_susp_n} 个 T+5 停牌样本 "
                      f"({start_time} ~ {end_time})")
    except Exception:
        pass  # 停牌检测失败不影响主流程

    # 标注不可交易
    label_df_clean.loc[untradeable[untradeable].index, label_name] = float("nan")
    removed = untradeable.sum()
    if removed > 0:
        print(f"      [标签过滤] 剔除 {removed} 个不可交易样本 "
              f"({start_time} ~ {end_time}, 总 {len(label_df_clean)})")

    return label_df_clean


def apply_label_filter(frame: pd.DataFrame, instruments, start_time, end_time,
                       label_names) -> pd.DataFrame:
    """对已 prepared 的训练/验证帧应用标签可交易性过滤。

    从 MultiIndex 列结构中提取标签列 → 过滤 → 写回帧。
    涨跌停股的特征列保持不动，仅标签标 NaN。

    参数：
    - frame: Qlib DataHandlerLP.prepare() 输出的 DataFrame
    - instruments: Qlib 股票列表
    - start_time, end_time: 时间范围
    - label_names: 标签列名列表，如 ["LABEL_5D"]

    返回：
    - 过滤后的 DataFrame
    """
    if frame.empty:
        return frame

    result = frame.copy()
    for label_name in label_names:
        # 兼容 MultiIndex 列和 flat 列
        if isinstance(result.columns, pd.MultiIndex):
            col_key = ("label", label_name)
            if col_key not in result.columns:
                continue
            label_series = result[col_key]
        else:
            col_key = label_name
            if col_key not in result.columns:
                continue
            label_series = result[col_key]

        label_df = label_series.to_frame(label_name)
        label_df_clean = filter_untradeable_labels(
            label_df, instruments, start_time, end_time
        )
        result[col_key] = label_df_clean[label_name]

    return result
