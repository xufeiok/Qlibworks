# -*- coding: utf-8 -*-
"""GTJA191 因子字典 → DuckDB 表达式生成器（分阶段 CTE + 函数宏版）。

背景
----
旧实现把 qlib 表达式「单行内联」翻译成 DuckDB，窗口函数直接套窗口函数，
在 DuckDB 上必然触发：
  - window function calls cannot be nested
  - window functions are not allowed in window definitions
  - aggregate function calls cannot contain window function calls
导致 191 个因子中 139 个无法计算。

本实现
------
1. 递归下降解析 qlib 表达式为 AST（支持隐式乘法、`^`、`&`/`|`、比较运算）；
2. 任何一个**窗口语义**函数（Ref/Delta/Sum/Mean/Std/Corr/Cov/Rank/Ts_Rank/
   Sma/Wma/Decaylinear/RegBeta/SumAc/SumIf/Count/Prod/HighDay/LowDay/Sequence）
   都落成一个独立的 CTE 阶段：
       sN AS (SELECT *, (<窗口表达式>) AS fN FROM sN-1)
   阶段之间只引用上一阶段产出的**标量列**，从根本上消除窗口嵌套；
3. 复合窗口函数用 DuckDB 原生能力实现，避免近似：
   - Sma        → 指数平滑的截断加权和（list_transform + list_sum）
   - Wma/Decaylinear → 线性加权和（list_transform + list_sum）
   - Ts_Rank    → list_filter 计数 / d（精确时序排名，非 (x-min)/(max-min) 近似）
   - HighDay/LowDay → list_position(list_reverse(...), list_max/min(...)) - 1
   - RegBeta    → COVAR_SAMP / VAR_SAMP
   - SumIf/Count → 窗口聚合 + FILTER
   - Prod       → EXP(SUM(LN(x)))
4. 输出恒以 `WITH` 开头，且最终列固定为 ts_code / trade_date / value，
   以匹配 FactorStore._compute / batch_compute 的 CTE 分支。

用法
----
    python tools/factor_management/gen_gtja191_duckdb.py           # 写回 YAML
    python tools/factor_management/gen_gtja191_duckdb.py --dry-run # 仅体检不落盘

边界与假设
----------
- 价格/成交量字段来自 FactorStore._build_adj_sql 的 `_raw`（前复权口径）；
- Alpha172/186 使用的中间变量 LD/HD/TR 按研报附录定义展开（见 _DERIVED_VARS）；
- Alpha30(MKT/SMB/HML 缺 Fama-French 数据)、Alpha69(DTM/DBM 研报未给定义)
  仍标记为 unsupported，不写 duckdb 表达式（保持现有值不动）；
- Alpha143(SELF 递归)、Alpha149/181/182(基准指数 BANCHMARKINDEX*) 已通过
  _SQL_OVERRIDES 手写 CTE SQL 实现，基准列 benchmark_close/open 由
  FactorStore._build_adj_sql 从 index_daily(000300.SH) 挂接。
"""
from __future__ import annotations

import argparse
import math
import re
from typing import Dict, List, Tuple

import yaml

YAML_PATH = r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml"

# ─────────────────────────── 1. 词法 / 语法解析 ───────────────────────────


class Unsupported(Exception):
    """表达式使用了无法映射到 DuckDB/_raw 的语法或字段。"""


_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<num>\d+\.\d*|\.\d+|\d+)"
    r"|(?P<var>\$[A-Za-z_]\w*)"
    r"|(?P<ident>[A-Za-z_]\w*)"
    r"|(?P<op><=|>=|==|!=|&&|\|\||[+\-*/^<>()&|,;]))"
)


def _tokenize(text: str) -> List[Tuple[str, str]]:
    """把表达式切成 (类型, 文本) 序列；`OR`/`AND` 归一为运算符。"""
    out: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            if text[pos].isspace():
                pos += 1
                continue
            raise Unsupported(f"无法识别字符 {text[pos]!r} @ {pos}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind == "ident" and val.upper() in ("OR", "AND"):
            out.append(("op", "||" if val.upper() == "OR" else "&&"))
        elif kind == "op" and val in ("&", "|"):
            # 源公式混用单/双符号（Alpha4/55/98/137/172/182/186），统一为逻辑运算
            out.append(("op", "&&" if val == "&" else "||"))
        else:
            out.append((kind, val))
    out.append(("eof", ""))
    return out


class Parser:
    """递归下降解析器，产出 tuple AST。

    节点形态：
      ("num", float) / ("var", "close") / ("bin", op, l, r) / ("un", op, x)
      ("call", name, [args])
    """

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.t = tokens
        self.i = 0

    # -- 基础操作 --
    def peek(self) -> Tuple[str, str]:
        return self.t[self.i]

    def take(self, kind: str = None, val: str = None) -> Tuple[str, str]:
        tok = self.t[self.i]
        if kind and tok[0] != kind:
            raise Unsupported(f"期望 {kind}，实际 {tok}")
        if val and tok[1] != val:
            raise Unsupported(f"期望 {val}，实际 {tok}")
        self.i += 1
        return tok

    def accept(self, kind: str, val: str = None) -> bool:
        tok = self.peek()
        if tok[0] == kind and (val is None or tok[1] == val):
            self.i += 1
            return True
        return False

    # -- 文法 --
    def parse(self):
        node = self.parse_or()
        if self.peek()[0] != "eof":
            raise Unsupported(f"表达式未被完整消费，剩余 {self.t[self.i:self.i + 3]}")
        return node

    def parse_or(self):
        node = self.parse_and()
        while self.peek() == ("op", "||"):
            self.take()
            node = ("bin", "||", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_cmp()
        while self.peek() == ("op", "&&"):
            self.take()
            node = ("bin", "&&", node, self.parse_cmp())
        return node

    def parse_cmp(self):
        node = self.parse_add()
        while self.peek()[0] == "op" and self.peek()[1] in ("==", "!=", ">", "<", ">=", "<="):
            op = self.take()[1]
            node = ("bin", op, node, self.parse_add())
        return node

    def parse_add(self):
        node = self.parse_mul()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            node = ("bin", op, node, self.parse_mul())
        return node

    def parse_mul(self):
        """乘除 + 隐式乘法（如 `(a)(b)`、`(a)$volume`）。"""
        node = self.parse_unary()
        while True:
            tok = self.peek()
            if tok[0] == "op" and tok[1] in ("*", "/"):
                self.take()
                node = ("bin", tok[1], node, self.parse_unary())
            elif _starts_primary(tok):
                # 隐式乘法：把紧邻的下一个 primary 当作乘数
                node = ("bin", "*", node, self.parse_unary())
            else:
                return node

    def parse_unary(self):
        tok = self.peek()
        if tok[0] == "op" and tok[1] in ("+", "-"):
            self.take()
            return ("un", tok[1], self.parse_unary())
        return self.parse_pow()

    def parse_pow(self):
        node = self.parse_primary()
        if self.peek() == ("op", "^"):
            self.take()
            node = ("bin", "^", node, self.parse_unary())  # 右结合
        return node

    def parse_primary(self):
        tok = self.peek()
        if tok[0] == "num":
            self.take()
            return ("num", float(tok[1]))
        if tok[0] == "var":
            self.take()
            return ("var", tok[1][1:])
        if tok[0] == "ident":
            self.take()
            name = tok[1]
            if self.accept("op", "("):
                args = []
                if not self.accept("op", ")"):
                    args.append(self.parse_or())
                    while self.accept("op", ","):
                        args.append(self.parse_or())
                    self.take("op", ")")
                return ("call", name, args)
            return ("var", name)  # 裸标识符（SEQUENCE / LD / HD / TR ...）
        if tok == ("op", "("):
            self.take()
            node = self.parse_or()
            self.take("op", ")")
            return node
        raise Unsupported(f"无法作为 primary 解析：{tok}")


def _starts_primary(tok: Tuple[str, str]) -> bool:
    if tok[0] in ("num", "var"):
        return True
    if tok[0] == "ident":
        return True
    return tok == ("op", "(")


def _key(node) -> tuple:
    """AST 的结构化指纹：让「同形子表达式」共享同一阶段，避免 CTE 爆炸。"""
    kind = node[0]
    if kind in ("num", "var"):
        return (kind, node[1])
    if kind == "un":
        return ("un", node[1], _key(node[2]))
    if kind == "bin":
        return ("bin", node[1], _key(node[2]), _key(node[3]))
    if kind == "call":
        return ("call", node[1].upper(), tuple(_key(a) for a in node[2]))
    return ("raw", repr(node))


# ─────────────────────────── 2. 文本规整 ───────────────────────────

# 源字典中已确认的拼写/断词错误（仅修正语法，不改变公式语义）
_TYPO_FIXES = [
    (re.compile(r"\bHGIH\b", re.I), "$high"),
    (re.compile(r"\bCL\s+OSE\b", re.I), "$close"),
    (re.compile(r"\bCLOS\s+E\b", re.I), "$close"),
    (re.compile(r"\bD\s+ELAY\s*\(", re.I), "Delay("),
    (re.compile(r"\bDELAT\b", re.I), "Delta"),
]

# 研报附录给出的中间变量（Alpha172/186 使用）
_DERIVED_VARS = {
    "HD": "($high - Ref($high,1))",
    "LD": "(Ref($low,1) - $low)",
    "TR": "Max(Max($high-$low, Abs($high-Ref($close,1))), Abs($low-Ref($close,1)))",
}

# 裸标识符别名
_VAR_ALIASES = {
    "VOL": "volume",
    # 研报中的基准指数（_raw 由 FactorStore._build_adj_sql 从 index_daily 挂接）
    "BANCHMARKINDEXCLOSE": "benchmark_close",
    "BANCHMARKINDEXOPEN": "benchmark_open",
}

# 源字典转录错误（括号错位/漏乘号/漏参数），按因子名整体替换后重新生成。
# 仅修正语法层面，不改动研报语义；YAML 的 expression.qlib 保持原样以备追溯。
_EXPR_OVERRIDES = {
    # 原式多出一层未闭合括号：delta((((close-low)-(high-close))/((high-low)),1))
    "Alpha2": "-1 * Delta(((($close-$low)-($high-$close))/($high-$low)),1)",
    # 原式把 ^2 写在 MAX 之外、窗口 5 悬空：RANK(MAX(RET<0?STD(RET,20):CLOSE)^2),5)
    "Alpha10": "Rank(Max((If(($close/Ref($close,1)-1)<0, Std($close/Ref($close,1)-1,20), $close))^2,5))",
    # 原式 Corr 缺窗口参数、尾部多一个 ,2
    "Alpha36": "Rank(Sum(Corr(Rank($volume),Rank($vwap),6),6))",
    # 原式 (20-2)(SUM(...)) 缺乘号，且 SUM 内漏减 1：SUM((CLOSE/DELAY(CLOSE,1),20)^2,20)
    "Alpha166": ("-20*(20-1)^1.5*Sum($close/Ref($close,1)-1-Mean($close/Ref($close,1)-1,20),20)"
                 "/((20-1)*(20-2)*(Sum(($close/Ref($close,1)-1)^2,20))^1.5)"),
    # 原式分母 Sum(...) 漏写窗口，按同式 20 日窗口补齐
    "Alpha181": ("Sum((($close/Ref($close,1)-1)-Mean(($close/Ref($close,1)-1),20))"
                 "-(BANCHMARKINDEXCLOSE-Mean(BANCHMARKINDEXCLOSE,20))^2,20)"
                 "/Sum((BANCHMARKINDEXCLOSE-Mean(BANCHMARKINDEXCLOSE,20))^3,20)"),
}

# RANK 作底数再取幂的因子：1..N 的秩直接做指数会溢出浮点范围（Alpha115 高达 3400^3400），
# 改用当日截面百分位秩 [0,1]，保证 X^Y 有界可算；其余因子的 RANK 仍保持研报 1..N 口径。
_PERCENT_RANK_FACTORS = {"Alpha115"}

# 无法用 qlib 语法表达（递归自引用 / 条件回归）的因子，直接给出 DuckDB CTE SQL。
_SQL_OVERRIDES = {
    # Alpha143：SELF 为递归自引用。原式等价于「只对上涨日累乘 (close/prev_close)」，
    # 即 EXP(Σ_{ret>0} ln(1+ret))，用累积对数收益实现，初值 1（首日）。
    "Alpha143": """WITH s1 AS (
    SELECT *, (close / NULLIF(LAG(close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date), 0) - 1) AS ret_s
    FROM _raw),
s2 AS (SELECT *, CASE WHEN ret_s > 0 THEN LN(1 + ret_s) ELSE 0 END AS lg FROM s1),
s3 AS (SELECT *, SUM(lg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum
    FROM s2)
SELECT ts_code, trade_date, CAST(EXP(cum) AS DOUBLE) AS value
FROM s3
WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL""",
    # Alpha149：REGBETA(FILTER(RET, INDEX_RET<0), FILTER(INDEX_RET, INDEX_RET<0), 252)
    # 用滑动窗口内的条件回归实现：252 个交易日内指数下跌日的样本做 OLS 斜率。
    "Alpha149": """WITH s1 AS (
    SELECT *,
           (close / NULLIF(LAG(close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date), 0) - 1) AS rs,
           (benchmark_close / NULLIF(LAG(benchmark_close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date), 0) - 1) AS rb
    FROM _raw),
s2 AS (
    SELECT *,
           COUNT(*) FILTER (WHERE rb < 0) OVER w AS n,
           SUM(rs)  FILTER (WHERE rb < 0) OVER w AS sx,
           SUM(rb)  FILTER (WHERE rb < 0) OVER w AS sy,
           SUM(rs * rb) FILTER (WHERE rb < 0) OVER w AS sxy,
           SUM(rb * rb) FILTER (WHERE rb < 0) OVER w AS syy
    FROM s1
    WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW))
SELECT ts_code, trade_date,
       CAST((sxy - sx * sy / NULLIF(n, 0)) / NULLIF(syy - sy * sy / NULLIF(n, 0), 0) AS DOUBLE) AS value
FROM s2
WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL""",
}


def normalize(expr: str) -> str:
    """规整公式文本：取修正行、去分号、修拼写、展开中间变量。"""
    s = (expr or "").strip()
    if not s or s.startswith("TODO:"):
        raise Unsupported("空表达式或 TODO")
    if "\n" in s:
        # Alpha190 等：多行时最后一行是「有调整」的可用版本
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        s = lines[-1]
    s = s.replace("(公式有部分缺失 有调整)", "").strip().rstrip(";").strip()
    for pat, rep in _TYPO_FIXES:
        s = pat.sub(rep, s)
    s = s.replace("./", "/").replace(".*", "*")
    for name, definition in _DERIVED_VARS.items():
        s = re.sub(rf"(?<![A-Za-z_0-9]){name}(?![A-Za-z_0-9])", definition, s)
    return s


# ─────────────────────────── 3. SQL 生成 ───────────────────────────

_RAW_COLUMNS = {
    "open", "high", "low", "close", "volume", "vol", "amount", "vwap",
    "pre_close", "change_pct", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
    "total_mv", "circ_mv", "dv_ttm", "turnover_rate", "industry_code",
    "benchmark_close", "benchmark_open",
}

_PART = "PARTITION BY ts_code ORDER BY trade_date"

# 产生布尔值的运算符（参与算术时需转 0/1）
_LOGIC_OPS = {"==", "!=", ">", "<", ">=", "<=", "&&", "||"}


def _win(n: int) -> str:
    """时序窗口定义：最近 n 期（含当前行）。"""
    return f"OVER ({_PART} ROWS BETWEEN {n - 1} PRECEDING AND CURRENT ROW)"


class SqlGen:
    """把 AST 编译成分阶段 CTE SQL。"""

    def __init__(self):
        self.stages: List[Tuple[str, str, str]] = []  # (cte名, 列名, 表达式)
        self._memo: Dict[tuple, str] = {}

    # -- 阶段管理 --
    def _stage(self, sql: str) -> str:
        i = len(self.stages) + 1
        cte, col = f"s{i}", f"f{i}"
        self.stages.append((cte, col, sql))
        return col

    @staticmethod
    def _bare(sql: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_]\w*", sql))

    def _argcol(self, node) -> str:
        """把子表达式落成列（避免长表达式在窗口内重复出现）。"""
        sql = self.emit(node)
        return sql if self._bare(sql) else self._stage(sql)

    # -- 编译 --
    def emit(self, node) -> str:
        key = _key(node)
        if key in self._memo:
            return self._memo[key]
        kind = node[0]
        if kind == "num":
            sql = self._num(node[1])
        elif kind == "var":
            sql = self._var(node[1])
        elif kind == "un":
            sql = f"(-{self._arith(node[2])})"
        elif kind == "bin":
            sql = self._bin(node[1], node[2], node[3])
        elif kind == "call":
            sql = self._call(node[1], node[2])
        else:
            raise Unsupported(f"未知节点 {node}")
        self._memo[key] = sql
        return sql

    @staticmethod
    def _num(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else repr(v)

    def _var(self, name: str) -> str:
        low = name.lower()
        low = _VAR_ALIASES.get(name.upper(), low)
        if low in _RAW_COLUMNS:
            return low
        if name.upper() == "SEQUENCE":
            return self._call("Sequence", [])
        raise Unsupported(f"_raw 中不存在的字段/变量：{name}")

    def _arith(self, node) -> str:
        """算术上下文中把布尔子表达式转成 0/1（Alpha148 等把比较结果直接参与乘法）。"""
        sql = self.emit(node)
        if node[0] == "bin" and node[1] in _LOGIC_OPS:
            return f"CAST({sql} AS INTEGER)"
        return sql

    def _bin(self, op: str, lnode, rnode) -> str:
        if op in ("+", "-", "*", "/", "^"):
            left, right = self._arith(lnode), self._arith(rnode)
        else:
            left, right = self.emit(lnode), self.emit(rnode)
        if op in ("+", "-", "*", "&&", "||"):
            sql_op = {"&&": "AND", "||": "OR"}.get(op, op)
            return f"({left} {sql_op} {right})"
        if op == "/":
            return f"({left} / NULLIF({right}, 0))"
        if op == "^":
            return f"POWER({left}, {right})"
        if op in ("==", "!="):
            return f"({left} {'=' if op == '==' else '<>'} {right})"
        return f"({left} {op} {right})"

    # -- 函数分派 --
    def _int_arg(self, node, fname: str) -> int:
        if node[0] == "num" and float(node[1]).is_integer():
            return int(node[1])
        raise Unsupported(f"{fname} 的窗口长度必须是整数字面量，实际 {node}")

    def _call(self, name: str, args: list) -> str:
        fn = name.upper()
        n = len(args)

        # —— 标量函数：直接内联，不落阶段 ——
        if fn == "ABS" and n == 1:
            return f"ABS({self.emit(args[0])})"
        if fn == "SIGN" and n == 1:
            return f"SIGN({self.emit(args[0])})"
        if fn == "LOG" and n == 1:
            return f"LN(NULLIF({self.emit(args[0])}, 0))"
        if fn == "SQRT" and n == 1:
            return f"SQRT({self.emit(args[0])})"
        if fn in ("LESS", "MINIMUM") and n == 2:
            return f"LEAST({self.emit(args[0])}, {self.emit(args[1])})"
        if fn in ("GREATER", "MAXIMUM") and n == 2:
            return f"GREATEST({self.emit(args[0])}, {self.emit(args[1])})"
        if fn == "IF" and n == 3:
            return (f"CASE WHEN {self.emit(args[0])} THEN {self.emit(args[1])} "
                    f"ELSE {self.emit(args[2])} END")

        # —— 时序窗口函数：每个落成一个 CTE 阶段 ——
        if fn in ("REF", "DELAY") and n == 2:
            x = self._argcol(args[0])
            d = self._int_arg(args[1], fn)
            if d < 0:  # 未来数据（仅作公式完整性支持，实际不使用）
                return self._stage(f"LEAD({x}, {-d}) OVER ({_PART})")
            return self._stage(f"LAG({x}, {d}) OVER ({_PART})")

        if fn == "DELTA" and n == 2:
            x = self._argcol(args[0])
            d = self._int_arg(args[1], fn)
            return self._stage(f"({x} - LAG({x}, {d}) OVER ({_PART}))")

        if fn == "SUM" and n == 2:
            x, d = self._argcol(args[0]), self._int_arg(args[1], fn)
            return self._stage(f"SUM({x}) {_win(d)}")

        if fn in ("MEAN", "MA") and n == 2:
            x, d = self._argcol(args[0]), self._int_arg(args[1], fn)
            return self._stage(f"AVG({x}) {_win(d)}")

        if fn in ("MEAN", "MA") and n == 1:
            # Alpha127：未给窗口 → 全历史扩展均值
            x = self._argcol(args[0])
            return self._stage(f"AVG({x}) OVER ({_PART} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)")

        if fn == "STD" and n == 2:
            x, d = self._argcol(args[0]), self._int_arg(args[1], fn)
            return self._stage(f"STDDEV_SAMP({x}) {_win(d)}")

        if fn == "STD" and n == 1:
            # Alpha54：未给窗口 → 全历史扩展标准差
            x = self._argcol(args[0])
            return self._stage(
                f"STDDEV_SAMP({x}) OVER ({_PART} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)")

        if fn == "TS_MAX" and n == 2:
            x, d = self._argcol(args[0]), self._int_arg(args[1], fn)
            return self._stage(f"MAX({x}) {_win(d)}")

        if fn == "TS_MIN" and n == 2:
            x, d = self._argcol(args[0]), self._int_arg(args[1], fn)
            return self._stage(f"MIN({x}) {_win(d)}")

        if fn in ("MAX", "MIN"):
            agg = "MAX" if fn == "MAX" else "MIN"
            if n == 1:  # 全历史扩展窗口
                return self._stage(
                    f"{agg}({self._argcol(args[0])}) OVER (PARTITION BY ts_code)")
            if n == 2:
                # 二元 Max(A,B) 为逐元素取大；Max(X,N)（N 为窗口长度）为时序取大。
                # 判据：第二参数是 >=2 的整数字面量，且第一参数不是字面量。
                second_int = (args[1][0] == "num" and float(args[1][1]).is_integer()
                              and int(args[1][1]) >= 2)
                first_num = args[0][0] == "num"
                if second_int and not first_num:
                    d = int(args[1][1])
                    return self._stage(f"{agg}({self._argcol(args[0])}) {_win(d)}")
                scalar = "GREATEST" if fn == "MAX" else "LEAST"
                return f"{scalar}({self.emit(args[0])}, {self.emit(args[1])})"
            raise Unsupported(f"{fn} 参数个数 {n} 不支持")

        if fn == "CORR" and n == 3:
            a, b = self._argcol(args[0]), self._argcol(args[1])
            d = self._int_arg(args[2], fn)
            return self._stage(f"CORR({a}, {b}) {_win(d)}")

        if fn == "COV" and n == 3:
            a, b = self._argcol(args[0]), self._argcol(args[1])
            d = self._int_arg(args[2], fn)
            return self._stage(f"COVAR_SAMP({a}, {b}) {_win(d)}")

        if fn == "REGBETA" and n == 2:
            # Alpha21/147：REGBETA(Y, SEQUENCE(N)) —— 以时间序号为自变量的 N 期回归斜率
            seq = args[1]
            if not (seq[0] == "call" and seq[1].upper() == "SEQUENCE" and len(seq[2]) == 1):
                raise Unsupported(f"REGBETA 第二参仅支持 SEQUENCE(N)，实际 {seq}")
            y = self._argcol(args[0])
            d = self._int_arg(seq[2][0], fn)
            idx = self._stage(f"ROW_NUMBER() OVER ({_PART})")
            return self._stage(
                f"(COVAR_SAMP({y}, CAST({idx} AS DOUBLE)) {_win(d)}"
                f" / NULLIF(VAR_SAMP(CAST({idx} AS DOUBLE)) {_win(d)}, 0))")

        if fn == "REGBETA" and n == 3:
            y, x = self._argcol(args[0]), self._argcol(args[1])
            d = self._int_arg(args[2], fn)
            return self._stage(
                f"(COVAR_SAMP({y}, {x}) {_win(d)} / NULLIF(VAR_SAMP({x}) {_win(d)}, 0))")

        if fn == "RANK" and n == 1:
            # 研报口径：横截面排名 1..N；入库/评测阶段再做当日截面百分位
            return self._stage(
                f"RANK() OVER (PARTITION BY trade_date ORDER BY {self.emit(args[0])})")

        if fn == "TS_RANK" and n == 2:
            x = self._argcol(args[0])
            d = self._int_arg(args[1], fn)
            lst = f"list_filter(list({x}) {_win(d)}, v -> v IS NOT NULL AND v <= {x})"
            return self._stage(f"(LEN({lst})::DOUBLE / {d})")

        if fn in ("HIGHDAY", "LOWDAY") and n == 2:
            x = self._argcol(args[0])
            d = self._int_arg(args[1], fn)
            lst = f"list_filter(list({x}) {_win(d)}, v -> v IS NOT NULL)"
            g = "list_max" if fn == "HIGHDAY" else "list_min"
            return self._stage(f"(list_position(list_reverse({lst}), {g}({lst})) - 1)")

        if fn in ("DECAYLINEAR", "WMA") and n == 2:
            x = self._argcol(args[0])
            d = self._int_arg(args[1], fn)
            # 权重 1..d（越近权重越大），分母固定 d(d+1)/2
            lst = f"list_filter(list({x}) {_win(d)}, v -> v IS NOT NULL)"
            return self._stage(
                f"(list_sum(list_transform({lst}, (v,i) -> v * (i + 1))) / ({d} * ({d} + 1) / 2.0))")

        if fn in ("SMA", "SMEAN", "EMA") and n == 2:
            # Alpha146 分母 SMA(X,60)：研报只定义 SMA(X,N,M)，2 参按简单移动平均处理
            x, d = self._argcol(args[0]), self._int_arg(args[1], fn)
            return self._stage(f"AVG({x}) {_win(d)}")

        if fn in ("SMA", "SMEAN", "EMA") and n == 3:
            x = self._argcol(args[0])
            nn, mm = self._int_arg(args[1], fn), self._int_arg(args[2], fn)
            if nn <= 0 or mm <= 0 or mm > nn:
                raise Unsupported(f"Sma({nn},{mm}) 参数非法")
            alpha = mm / nn
            q = 1.0 - alpha
            # 截断长度：q^K < 1e-6，保证与递归 EMA 的偏差可忽略
            k = max(2, min(4000, int(math.ceil(math.log(1e-6) / math.log(q)))))
            frame = (f"OVER ({_PART} ROWS BETWEEN {k - 1} PRECEDING AND CURRENT ROW)")
            lst = f"list_filter(list({x}) {frame}, v -> v IS NOT NULL)"
            return self._stage(
                f"(({mm}.0 / {nn}) * list_sum(list_transform({lst}, "
                f"(v,i) -> v * POWER(1.0 - {mm}.0 / {nn}, COUNT({x}) {frame} - 1 - i))))")

        if fn in ("SUMIF", "SUM_IF") and n == 3:
            x = self._argcol(args[0])
            d = self._int_arg(args[1], fn)
            cond = self.emit(args[2])
            return self._stage(f"SUM({x}) FILTER (WHERE {cond}) {_win(d)}")

        if fn == "COUNT" and n == 2:
            d = self._int_arg(args[1], fn)
            cond = self.emit(args[0])
            return self._stage(f"COUNT(*) FILTER (WHERE {cond}) {_win(d)}")

        if fn in ("PROD",) and n == 2:
            x = self._argcol(args[0])
            d = self._int_arg(args[1], fn)
            return self._stage(f"EXP(SUM(LN(NULLIF({x}, 0))) {_win(d)})")

        if fn in ("SUMAC", "SUM_AC") and n == 1:
            x = self._argcol(args[0])
            return self._stage(
                f"SUM({x}) OVER ({_PART} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)")

        if fn in ("SEQUENCE",) and n in (0, 1):
            return self._stage(f"ROW_NUMBER() OVER ({_PART})")

        raise Unsupported(f"不支持的函数 {name}({n} 个参数)")

    # -- 组装 --
    def build(self, ast) -> str:
        final = self.emit(ast)
        if not self.stages:
            return (f"WITH _base AS (SELECT * FROM _raw)\n"
                    f"SELECT ts_code, trade_date, CAST({final} AS DOUBLE) AS value\n"
                    f"FROM _base\nWHERE ts_code IS NOT NULL AND trade_date IS NOT NULL")
        ctes, prev = [], "_raw"
        for cte, col, sql in self.stages:
            ctes.append(f"{cte} AS (SELECT *, CAST({sql} AS DOUBLE) AS {col} FROM {prev})")
            prev = cte
        return ("WITH " + ",\n".join(ctes) +
                f"\nSELECT ts_code, trade_date, CAST({final} AS DOUBLE) AS value\n"
                f"FROM {prev}\nWHERE ts_code IS NOT NULL AND trade_date IS NOT NULL")


def qlib_to_duckdb(expr: str, name: str = "") -> str:
    """qlib 表达式 → 分阶段 CTE 的 DuckDB SQL（以 WITH 开头）。"""
    if name in _SQL_OVERRIDES:
        return _SQL_OVERRIDES[name]
    expr = _EXPR_OVERRIDES.get(name, expr)
    ast = Parser(_tokenize(normalize(expr))).parse()
    sql = SqlGen().build(ast)
    if name in _PERCENT_RANK_FACTORS:
        # 当日截面排名由 1..N 改为 [0,1] 百分位
        sql = sql.replace("RANK() OVER (PARTITION BY trade_date",
                          "PERCENT_RANK() OVER (PARTITION BY trade_date")
    return sql


# ─────────────────────────── 4. 批量写回字典 ───────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只体检，不写回 YAML")
    args = ap.parse_args()

    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    ok, bad = [], []
    for factor in data.get("factors", []):
        qlib_expr = (factor.get("expression") or {}).get("qlib", "")
        try:
            factor["expression"]["duckdb"] = qlib_to_duckdb(qlib_expr, factor["name"])
            ok.append(factor["name"])
        except Unsupported as exc:
            bad.append((factor["name"], str(exc)))

    print(f"生成成功 {len(ok)} / 失败 {len(bad)}")
    for name, why in bad:
        print(f"  [跳过] {name}: {why}")

    if args.dry_run:
        return

    class _Dumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    def _str_presenter(dumper, value):
        if "\n" in value:
            return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", value)

    yaml.add_representer(str, _str_presenter)
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False,
                  Dumper=_Dumper, default_flow_style=False, width=4096)
    print(f"已写回 {YAML_PATH}")


if __name__ == "__main__":
    main()
