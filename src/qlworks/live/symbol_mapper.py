"""
证券代码格式映射工具

输入:
- Qlib 风格: 600000.sh / 000001.sz
- 通达信风格: 600000.SH / 000001.SZ

输出:
- 在两种格式之间稳定转换

边界:
- 仅支持 6 位 A 股代码 + SH/SZ/BJ 后缀
- 非法格式直接抛出 ValueError，避免把错误代码送进交易层
"""
from __future__ import annotations

import re


_SYMBOL_PATTERN = re.compile(r"^(?P<code>\d{6})\.(?P<market>sh|sz|bj)$", re.IGNORECASE)


def _split_symbol(symbol: str) -> tuple[str, str]:
    if not isinstance(symbol, str):
        raise ValueError("symbol 必须为字符串")

    match = _SYMBOL_PATTERN.match(symbol.strip())
    if not match:
        raise ValueError(f"非法证券代码格式: {symbol}")

    return match.group("code"), match.group("market")


def normalize_symbol_to_tdx(symbol: str) -> str:
    """
    将证券代码规范化为通达信格式。

    例如:
    - 600000.sh -> 600000.SH
    """
    code, market = _split_symbol(symbol)
    return f"{code}.{market.upper()}"


def normalize_symbol_to_qlib(symbol: str) -> str:
    """
    将证券代码规范化为 Qlib/本地研究格式。

    例如:
    - 600000.SH -> 600000.sh
    """
    code, market = _split_symbol(symbol)
    return f"{code}.{market.lower()}"
