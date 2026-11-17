# -*- coding: utf-8 -*-
"""
公共模块：生产预测模式（forecast）的滚动窗口切分工具。

背景：训练脚本（train_tree-doubao / train_linear / train_icir）默认使用
"年度滚动窗口"（train 2年 + valid 1年 + test 1年，test 为历史区间）做回测评测。
生产周度选股需要的是"截至最近数据日（上周五）的最新截面打分"，
因此提供 build_forecast_config() 动态构建单窗口配置：

    as_of        = 最近交易日（默认 Qlib 日历最新日，即上周五）
    label_anchor = as_of 往前 label_days 个交易日（该日 LABEL_5D 已完整，
                   作为测试段起点，且其真实收益可事后验证）
    train_end    = label_anchor 往前 embargo_days 个交易日（防前视偏差）
    train        = [as_of - train_years 年, train_end]    （用尽最新数据）
    valid        = [train_end 后一交易日, label_anchor]   （最近数据做验证）
    test         = [label_anchor, as_of]                  （最近一周信号窗，
                   最后一天 as_of 即"本周选股"信号截面）

使用方式（各训练脚本）：
    from _forecast_utils import build_forecast_config
    _fc_cfg, _fc_window = build_forecast_config(
        base_config=LOCAL_CONFIG, as_of_date=..., train_years=3, valid_months=6)
    CONFIG = resolve_runtime_config(local_config=_fc_cfg, ...)
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

# 日历文件默认路径（Qlib 标准布局）
DEFAULT_CALENDAR_REL = Path("qlib_data") / "calendars" / "day.txt"


def load_qlib_calendar_dates(calendar_path: str | Path | None = None) -> list[str]:
    """
    读取 Qlib 交易日历的全部日期（升序）。

    参数:
    - calendar_path: Qlib 日历文件路径，默认为 qlib_data/calendars/day.txt

    返回:
    - 升序日期字符串列表 ["YYYY-MM-DD", ...]
    """
    path = Path(calendar_path) if calendar_path else DEFAULT_CALENDAR_REL
    if not path.exists():
        raise FileNotFoundError(f"Qlib 日历文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        dates = [line.strip() for line in f if line.strip()]
    if not dates:
        raise ValueError(f"Qlib 日历文件为空: {path}")
    dates = sorted(dates)
    return dates


def calendar_shift(dates: list[str], anchor: str, n_back: int) -> str:
    """从 anchor 日期往前数 n_back 个交易日，返回对应日期字符串（不足则取最早日）。"""
    anchor_idx = None
    for i, d in enumerate(dates):
        if d >= anchor:
            anchor_idx = i
            break
    if anchor_idx is None:
        anchor_idx = len(dates) - 1
    target = max(anchor_idx - n_back, 0)
    return dates[target]


def calendar_back_days(dates: list[str], start: str, end: str) -> list[str]:
    """返回 [start, end] 区间内的交易日列表（含两端）。"""
    return [d for d in dates if start <= d <= end]


def build_forecast_config(
    base_config: dict | None = None,
    as_of_date: str | None = None,
    train_years: int = 3,
    valid_months: int = 6,
    label_days: int = 5,
    embargo_days: int = 10,
    calendar_path: str | Path | None = None,
) -> tuple[dict, dict]:
    """
    构建【生产预测模式】的单窗口配置（每周选股场景）。

    返回:
    - (deepcopy 后的配置字典, 单窗口 dict)，rolling_windows 已替换为
      单窗口 {"name": "Forecast_<YYYYMMDD>", "train": ..., "valid": ..., "test": ...}
    """
    config = copy.deepcopy(base_config or {})
    dates = load_qlib_calendar_dates(calendar_path)
    as_of = as_of_date or dates[-1]
    if as_of not in dates:
        # 允许传入非交易日：对齐到最近一个交易日（<= as_of）
        _prior = [d for d in dates if d <= as_of]
        if not _prior:
            raise ValueError(f"as_of_date {as_of} 早于日历最早日 {dates[0]}")
        as_of = _prior[-1]

    # 关键锚点计算（全部按交易日偏移）
    label_anchor = calendar_shift(dates, as_of, label_days)
    train_end = calendar_shift(dates, label_anchor, embargo_days)
    valid_start = calendar_shift(dates, train_end, -1)  # train_end 后一交易日
    if valid_start <= train_end:
        valid_start = calendar_shift(dates, label_anchor, embargo_days + 1)

    # 训练起点：as_of 往前 train_years 个自然年，再对齐到交易日
    as_of_ts = pd.Timestamp(as_of)
    train_start_raw = (as_of_ts - pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")
    train_start = calendar_shift(dates, train_start_raw, 0)
    if train_start > train_end:
        raise ValueError(f"训练窗口异常: train_start({train_start}) > train_end({train_end})")

    # 验证段：train_end 之后 ~valid_months 个月（不超过 label_anchor）
    valid_end_raw = (pd.Timestamp(train_end) + pd.DateOffset(months=valid_months)).strftime("%Y-%m-%d")
    if valid_end_raw > label_anchor:
        valid_end = label_anchor
    else:
        valid_end = calendar_shift(dates, valid_end_raw, 0)

    window = {
        "name": f"Forecast_{as_of.replace('-', '')}",
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (label_anchor, as_of),
    }
    config["rolling_windows"] = [window]
    config["end_time"] = as_of
    return config, window
