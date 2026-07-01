#!/usr/bin/env python3
"""
计算剩余复杂 Alpha 因子：基于 YAML DuckDB 表达式做 CTE 分解。

方法：对 YAML DuckDB 表达式中的嵌套窗口函数，
将内层窗口函数提取到独立 CTE 中，避免 DuckDB 的嵌套窗口限制。

使用方式：
  cd e:\Quant\Qlibworks
  E:\Conda_env\Qlib_env\python.exe scripts\evaluation\compute_all_missing.py
"""
import sys, logging, warnings, re, os
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compute_missing")

import yaml
from qlworks.evaluation import FactorStore, DEFAULT_CONFIG

YAMLS = [
    Path(r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml"),
    Path(r"e:\Quant\Qlibworks\factor_data\factor_library\price_volume_factors.yaml"),
]
START = "2010-01-01"
END = "2026-12-31"


def _extract_window_fn(sql: str):
    """
    从 DuckDB SQL 中提取最内层的 OVER 子句。
    
    最内层定义：函数参数和 OVER 子句中都 不 包含其他 OVER。
    即: FUNC(args) OVER (clause) 中 args 不含 OVER 且 clause 不含 OVER。
    
    返回: [(start, end, full_match), ...]
    """
    results = []
    for m in re.finditer(r'\bOVER\s*\(', sql, re.IGNORECASE):
        over_start = m.start()
        paren_depth = 1
        i = m.end()
        while i < len(sql) and paren_depth > 0:
            if sql[i] == '(':
                paren_depth += 1
            elif sql[i] == ')':
                paren_depth -= 1
            i += 1
        if paren_depth != 0:
            continue
        over_end = i

        # 从 over_start 往前找对应的 FUNC(...)
        j = over_start - 1
        while j >= 0 and sql[j] in ' \t\n\r':
            j -= 1
        if j < 0:
            continue
        if sql[j] == ')':
            pd = 1
            j -= 1
            while j >= 0 and pd > 0:
                if sql[j] == ')':
                    pd += 1
                elif sql[j] == '(':
                    pd -= 1
                j -= 1
        while j >= 0 and sql[j] in ' \t\n\r':
            j -= 1
        func_end = j + 1
        while j >= 0 and (sql[j].isalnum() or sql[j] == '_'):
            j -= 1
        func_start = j + 1

        fn_name = sql[func_start:func_end]
        full_fn = sql[func_start:over_end]

        # 找函数参数部分: FUNC(args) 的 args
        args_start = func_end + 1  # 跳过函数名后的 (
        pd = 1
        k = args_start
        while k < over_start and pd > 0:
            if sql[k] == '(': pd += 1
            elif sql[k] == ')': pd -= 1
            k += 1
        func_args = sql[args_start:k - 1]  # 去掉外层 ()
        over_clause = sql[m.end():over_end - 1]

        if 'OVER' not in func_args.upper() and 'OVER' not in over_clause.upper():
            results.append((func_start, over_end, full_fn))

    return results


def _flatten_duckdb_expr(duckdb_sql: str) -> str:
    """
    将含嵌套窗口函数的 DuckDB SQL 展开为 CTE SQL。
    
    方法：从内到外逐层提取窗口函数到独立 CTE，
    每个 CTE JOIN 其依赖的 CTE，正确解析列引用。
    """
    sql = duckdb_sql.strip()
    
    # 提取所有最内层窗口函数到 CTE
    all_wfs = []  # [{name, ph, fn, deps}, ...]
    while True:
        wfs = _extract_window_fn(sql)
        if not wfs:
            break
        
        for start, end, full_fn in wfs:
            idx = len(all_wfs)
            cte_name = f"_wf{idx}"
            placeholder = f"__wf{idx}__"
            
            # 检查 full_fn 中是否包含已有占位符（依赖）
            deps = []
            for existing in all_wfs:
                if existing['ph'] in full_fn:
                    deps.append(existing)
            
            all_wfs.append({
                'name': cte_name,
                'ph': placeholder,
                'fn': full_fn,
                'deps': deps,
            })
            
            # 替换这个窗口函数为占位符
            sql = sql[:start] + placeholder + sql[end:]
            break  # 一次只提取一个，下次循环再取新的最内层
    
    if not all_wfs:
        return sql
    
    # 构建 CTE 定义
    cte_defs = []
    for wf in all_wfs:
        # 解析依赖：替换占位符为 CTE 列引用
        wf_fn = wf['fn']
        # 构建 JOIN 子句
        joins = ""
        for dep in wf['deps']:
            wf_fn = wf_fn.replace(dep['ph'], f"{dep['name']}.{dep['name']}")
            joins += f" LEFT JOIN {dep['name']} ON _raw.ts_code={dep['name']}.ts_code AND _raw.trade_date={dep['name']}.trade_date"
        
        cte_defs.append(
            f"{wf['name']} AS ("
            f"SELECT _raw.ts_code, _raw.trade_date, {wf_fn} AS {wf['name']} "
            f"FROM _raw{joins})"
        )
    
    # 构建最终 SELECT（用标量子查询）
    select_expr = sql
    for wf in all_wfs:
        select_expr = select_expr.replace(
            wf['ph'],
            f"(SELECT {wf['name']} FROM {wf['name']} "
            f"WHERE _raw.ts_code={wf['name']}.ts_code AND _raw.trade_date={wf['name']}.trade_date)"
        )
    
    return "WITH " + ", ".join(cte_defs) + \
           f" SELECT _raw.ts_code, _raw.trade_date, {select_expr} AS value FROM _raw"


def _simplify_duckdb_expr(duckdb_sql: str) -> str:
    """
    简化复杂 DuckDB SQL 表达式，用 CTE 分解嵌套窗口函数。
    """
    # 移除空白行
    sql = re.sub(r'\s+', ' ', duckdb_sql).strip()
    
    # 使用两层策略：
    # 策略A: 如果已经是简单 SQL（无嵌套 OVER），直接返回
    # 策略B: 通过 CTE 分解嵌套
    
    # 统计 OVER 数量
    over_count = len(re.findall(r'\bOVER\s*\(', sql, re.IGNORECASE))
    if over_count <= 1:
        return sql  # 无嵌套，直接使用
    
    return _flatten_duckdb_expr(sql)


def is_missing_factor(name, wdir):
    fdir = wdir / name
    if not fdir.is_dir():
        return True
    return not bool(list(fdir.glob("*.parquet")))


def main():
    wdir = Path(r"e:\Quant\Qlibworks\factor_data\warehouse")
    store = FactorStore(DEFAULT_CONFIG)

    # 收集缺失因子
    missing = []
    for yp in YAMLS:
        if not yp.exists():
            continue
        with open(yp, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for fd in data.get("factors", []):
            name = fd.get("name", "")
            if not name or not is_missing_factor(name, wdir):
                continue
            expr = fd.get("expression", {})
            missing.append({
                "name": name,
                "qlib": str(expr.get("qlib", "")),
                "duckdb": str(expr.get("duckdb", "")),
            })

    logger.info(f"共 {len(missing)} 个缺失因子")

    succeeded = 0
    failed = 0

    for i, m in enumerate(missing, 1):
        name = m["name"]
        
        # 策略1: 直接用 YAML DuckDB 表达式
        de = m["duckdb"]
        if de and '暂未实现' not in de and de != 'None':
            logger.info(f"[{i}/{len(missing)}] {name} -> YAML")
            try:
                stats = store.compute_to_warehouse(
                    name=name, expr=m["qlib"],
                    start_date=START, end_date=END,
                    overwrite=False, duckdb_expr=de,
                )
                total_rows = sum(stats.values())
                if total_rows > 0:
                    logger.info(f"  OK {total_rows:,} rows")
                    succeeded += 1
                    continue
            except Exception as e:
                msg = str(e)[:80].replace('\n', ' ')
                logger.warning(f"  YAML fail: {msg}")
        
        # 策略2: CTE 分解
        if de and '暂未实现' not in de and de != 'None':
            try:
                cte_sql = _simplify_duckdb_expr(de)
                logger.info(f"[{i}/{len(missing)}] {name} -> CTE")
                try:
                    stats = store.compute_to_warehouse(
                        name=name, expr=m["qlib"],
                        start_date=START, end_date=END,
                        overwrite=False, duckdb_expr=cte_sql,
                    )
                    total_rows = sum(stats.values())
                    if total_rows > 0:
                        logger.info(f"  OK {total_rows:,} rows")
                        succeeded += 1
                        continue
                except Exception as e:
                    msg = str(e)[:80].replace('\n', ' ')
                    logger.warning(f"  CTE fail: {msg}")
            except Exception as e:
                msg = str(e)[:80].replace('\n', ' ')
                logger.warning(f"  CTE build fail: {msg}")
        
        # 策略3: Qlib兜底（duckdb_expr=None 让 factor_store 自动走 Qlib 路径）
        logger.info(f"[{i}/{len(missing)}] {name} -> Qlib")
        try:
            stats = store.compute_to_warehouse(
                name=name, expr=m["qlib"],
                start_date=START, end_date=END,
                overwrite=False, duckdb_expr=None,
            )
            total_rows = sum(stats.values())
            if total_rows > 0:
                logger.info(f"  OK {total_rows:,} rows")
                succeeded += 1
                continue
        except Exception as e:
            msg = str(e)[:80].replace('\n', ' ')
            logger.warning(f"  Qlib fail: {msg}")
        
        logger.error(f"  FAIL {name}")
        failed += 1

    logger.info(f"\nDone -> OK: {succeeded}, FAIL: {failed}")


if __name__ == "__main__":
    main()
