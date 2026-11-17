# -*- coding: utf-8 -*-
"""
主板优先的因子仓库分阶段重建脚本。

功能:
- 复制现有仓库到独立目录，保留未参与重建的因子。
- 对指定策略的因子执行主板股票池覆盖重算。
- 重算后检查文件、无穷值、异常哨兵值和元数据覆盖。

边界条件:
- 不直接覆盖 factor_data/warehouse。
- 部分因子失败时保留现场并返回非零状态，不自动切换目录。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlworks.evaluation import DEFAULT_CONFIG, FactorStore
from qlworks.evaluation.factor_store import _F32_SENTINEL, _sanitize_values, _WARMUP_YEARS
from qlworks.evaluation.warehouse_sync import load_factor_definitions

ACTIVE_STRATEGIES = [
    # [五大类归档 2026-10-08] 因子库归档为 5 大类 + other：
    #   momentum_price 合并原 price_volume/reversal_momentum/rolling，
    #   value 替代原 style（估值子类独立成档）。
    "quality_factors",
    "value_factors",
    "momentum_price_factors",
    "risk_factors",
    "sentiment_factors",
    "other_factors",
]
GTJA191_ARCHIVE = (
    PROJECT_ROOT
    / "factor_data"
    / "factor_library"
    / "archive"
    / "gtja191_factor_dictionary.yaml"
)


_SUPPORTED_FUNCTIONS = {
    "Abs",
    "Delta",
    "Greater",
    "If",
    "Less",
    "Log",
    "Max",
    "Mean",
    "Min",
    "Quantile",
    "Ref",
    "Std",
    "Sum",
}


def _split_args(text: str) -> list[str]:
    """按顶层逗号切分函数参数。"""
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    args.append(text[start:].strip())
    return args


def _window_clause(period: int, ordered: bool = True) -> str:
    """生成每日截面下的滚动窗口 SQL。"""
    if period <= 0:
        raise ValueError(f"非法窗口长度: {period}")
    order = "ORDER BY trade_date" if ordered else ""
    return (
        f"OVER (PARTITION BY ts_code {order} "
        f"ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW)"
    )


def _convert_function(name: str, args: list[str]) -> str:
    """将单个 Qlib 函数转换为 DuckDB 表达式。"""
    if name not in _SUPPORTED_FUNCTIONS:
        raise ValueError(f"不支持的 Qlib 函数: {name}")

    if name == "Ref":
        offset = int(args[1])
        function = "LEAD" if offset < 0 else "LAG"
        return (
            f"{function}({args[0]}, {abs(offset)}) "
            f"OVER (PARTITION BY ts_code ORDER BY trade_date)"
        )
    if name == "Delta":
        offset = abs(int(args[1]))
        return (
            f"{args[0]} - LAG({args[0]}, {offset}) "
            f"OVER (PARTITION BY ts_code ORDER BY trade_date)"
        )
    if name in {"Mean", "Sum", "Std", "Min", "Max"}:
        sql_function = {
            "Mean": "AVG",
            "Sum": "SUM",
            "Std": "STDDEV",
            "Min": "MIN",
            "Max": "MAX",
        }[name]
        return f"{sql_function}({args[0]}) {_window_clause(int(args[1]))}"
    if name == "Quantile":
        return (
            f"quantile_cont({args[0]}, {args[2]}) "
            f"{_window_clause(int(args[1]))}"
        )
    if name == "Abs":
        return f"ABS({args[0]})"
    if name == "Log":
        return f"LN(NULLIF({args[0]}, 0))"
    if name == "Greater":
        return f"CASE WHEN {args[0]} > {args[1]} THEN 1 ELSE 0 END"
    if name == "Less":
        return f"CASE WHEN {args[0]} < {args[1]} THEN 1 ELSE 0 END"
    if name == "If":
        return f"CASE WHEN {args[0]} THEN {args[1]} ELSE {args[2]} END"
    raise ValueError(f"未实现的 Qlib 函数: {name}")


def _convert_qlib_expression(expression: str) -> str | None:
    """递归转换 Qlib 表达式，失败时返回 None 以走 Qlib 回退。"""
    text = re.sub(r"\$(\w+)", r"\1", expression.strip())

    def convert(text_part: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(text_part):
            match = re.match(r"([A-Za-z_]\w*)\s*\(", text_part[index:])
            if not match:
                output.append(text_part[index])
                index += 1
                continue

            name = match.group(1)
            open_index = index + match.end() - 1
            depth = 1
            close_index = open_index + 1
            while close_index < len(text_part) and depth:
                if text_part[close_index] == "(":
                    depth += 1
                elif text_part[close_index] == ")":
                    depth -= 1
                close_index += 1
            if depth:
                raise ValueError(f"括号不匹配: {text_part}")

            raw_args = _split_args(text_part[open_index + 1:close_index - 1])
            converted_args = [convert(arg) for arg in raw_args]
            output.append(_convert_function(name, converted_args))
            index = close_index
        return "".join(output)

    try:
        return convert(text)
    except Exception:
        return None


def _can_compile_duckdb_expression(expression: str) -> bool:
    """用单行空样例验证 DuckDB 表达式的解析和绑定。"""
    try:
        conn = duckdb.connect()
        raw = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [pd.Timestamp("2026-09-11")],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
                "vol": [1.0],
                "amount": [1.0],
                "vwap": [1.0],
                "pre_close": [1.0],
                "change_pct": [0.0],
                "pe": [1.0],
                "pe_ttm": [1.0],
                "pb": [1.0],
                "ps": [1.0],
                "ps_ttm": [1.0],
                "total_mv": [1.0],
                "circ_mv": [1.0],
                "dv_ttm": [0.0],
                "turnover_rate": [1.0],
                "industry": ["银行"],
                "industry_code": ["801780"],
            }
        )
        conn.register("_raw", raw)
        if expression.strip().upper().startswith("WITH"):
            conn.execute(f"SELECT * FROM ({expression}) AS _tmp LIMIT 0")
        else:
            conn.execute(f"SELECT {expression} AS value FROM _raw LIMIT 0")
        return True
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _read_stock_codes(path: Path) -> list[str]:
    """读取 Qlib instruments 文件中的股票代码。"""
    codes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            codes.append(parts[0].upper())
    return sorted(set(codes))


def _remove_factor_dir(root: Path, name: str) -> None:
    """删除输出目录中的单个因子目录，路径必须位于目标根目录内。"""
    target = (root / name).resolve()
    root_resolved = root.resolve()
    if target.parent != root_resolved:
        raise RuntimeError(f"拒绝删除非目标因子目录: {target}")
    if target.exists():
        shutil.rmtree(target)


def _prepare_factor_expression(item: dict) -> str | None:
    """清理占位表达式，返回可交给 FactorStore 的 DuckDB 表达式。"""
    duckdb_expr = str(item.get("duckdb_expr") or "").strip()
    if not duckdb_expr or "暂未实现" in duckdb_expr:
        return None
    return duckdb_expr


def _load_archive_definitions(path: Path) -> list[dict]:
    """加载归档 GTJA191 因子定义，转换为仓库重建统一结构。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items: list[dict] = []
    for factor in data.get("factors", []):
        name = factor.get("name")
        expression = factor.get("expression", {})
        qlib_expr = (
            str(expression.get("qlib", ""))
            if isinstance(expression, dict)
            else str(expression or "")
        )
        duckdb_expr = (
            str(expression.get("duckdb", ""))
            if isinstance(expression, dict)
            else ""
        )
        if not name or not qlib_expr:
            continue
        items.append(
            {
                "name": name,
                "qlib_expr": qlib_expr,
                "duckdb_expr": duckdb_expr,
                "meta": factor,
            }
        )
    return items


def _factor_has_data(root: Path, name: str) -> bool:
    """判断因子目录是否至少存在一个非空 Parquet 文件。"""
    factor_dir = root / name
    if not factor_dir.exists():
        return False
    for path in factor_dir.glob("*.parquet"):
        if pq.ParquetFile(path).metadata.num_rows > 0:
            return True
    return False


def _run_batch_groups(
    store: FactorStore,
    factors: list[tuple[str, str]],
    start_date: str,
    end_date: str,
    stocks: list[str],
    output_root: Path,
) -> list[str]:
    """
    分组执行批量计算。

    单组失败时只回退该组因子，避免一个坏表达式拖累全部批次。
    """
    simple = [(name, expr) for name, expr in factors if not expr.strip().upper().startswith("WITH")]
    cte = [(name, expr) for name, expr in factors if expr.strip().upper().startswith("WITH")]
    groups: list[list[tuple[str, str]]] = [[item] for item in cte]
    groups.extend(simple[offset:offset + 20] for offset in range(0, len(simple), 20))

    fallback: list[str] = []
    for index, group in enumerate(groups, 1):
        names = [name for name, _ in group]
        print(f"  [批次 {index}/{len(groups)}] {len(group)} 个因子")
        try:
            stats = store.batch_compute(
                group,
                start_date,
                end_date,
                stocks=stocks,
                overwrite=True,
            )
        except Exception as exc:
            print(f"    [警告] 批次异常，转入逐因子: {exc}")
            stats = {}
        for name in names:
            if name not in stats or not _factor_has_data(output_root, name):
                fallback.append(name)
    return fallback


def _compute_qlib_fallback_batch(
    store: FactorStore,
    items: list[dict],
    start_date: str,
    end_date: str,
    stocks: list[str],
    qlib_dir: Path,
) -> list[str]:
    """
    使用 Qlib 多表达式批量计算回退因子。

    返回仍未成功计算、需要逐因子处理的因子名。
    """
    failed: list[str] = []
    groups = [items[offset:offset + 20] for offset in range(0, len(items), 20)]
    # 预热：向前回退若干年，保证 Qlib 滚动窗口在跨年处可读取上一年数据。
    compute_start = (
        pd.Timestamp(start_date) - pd.DateOffset(years=_WARMUP_YEARS)
    ).strftime("%Y-%m-%d")
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D

        qlib.init(
            provider_uri=str(qlib_dir),
            region=REG_CN,
            joblib_backend="threading",
            maxtasksperchild=1,
        )
    except Exception as exc:
        print(f"  [警告] Qlib 初始化失败，保留逐因子回退: {exc}")
        return [item["name"] for item in items]

    for group_index, group in enumerate(groups, 1):
        expressions = [item["qlib_expr"] for item in group]
        print(
            f"  [Qlib批次 {group_index}/{len(groups)}] "
            f"{len(group)} 个因子"
        )
        group_failed: set[str] = set()
        parts_by_factor: dict[str, list[pd.DataFrame]] = {
            item["name"]: [] for item in group
        }
        # 按股票分批，但一次性传入完整区间，避免滚动窗口在分块边界断窗。
        for stock_offset in range(0, len(stocks), 500):
            stock_batch = stocks[stock_offset:stock_offset + 500]
            try:
                frame = D.features(
                    stock_batch,
                    expressions,
                    compute_start,
                    end_date,
                    freq="day",
                )
            except Exception:
                group_failed.update(item["name"] for item in group)
                break
            if frame.empty:
                continue
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = [column[0] for column in frame.columns]

            for item in group:
                expression = item["qlib_expr"]
                name = item["name"]
                if expression not in frame.columns:
                    group_failed.add(name)
                    continue
                factor = frame[[expression]].rename(columns={expression: "value"})
                factor = factor.dropna(subset=["value"])
                if factor.empty:
                    continue
                parts_by_factor[name].append(factor)

        if group_failed:
            failed.extend(item["name"] for item in group)
            continue

        for item in group:
            name = item["name"]
            parts = parts_by_factor[name]
            if not parts:
                group_failed.add(name)
                continue
            factor = pd.concat(parts).sort_index()
            factor = factor[~factor.index.duplicated(keep="last")]
            factor = _sanitize_values(factor, "value")
            factor = factor.astype({"value": "float32"})
            # 丢弃预热区间，按单文件合并写回
            factor = factor[
                factor.index.get_level_values("datetime") >= pd.Timestamp(start_date)
            ]
            if factor.empty:
                group_failed.add(name)
                continue
            years = {int(y) for y in factor.index.get_level_values("datetime").year}
            store.save_warehouse_frame(name, factor, replace_years=years)

        if group_failed:
            continue

        for item in group:
            name = item["name"]
            if not _factor_has_data(store.warehouse_dir, name):
                failed.append(name)
            else:
                store._update_warehouse_meta(name)
    return sorted(set(failed))


def _inject_metadata(store: FactorStore, items: list[dict]) -> None:
    """将 YAML 语义字段写入重建仓库 meta。"""
    fields = {
        "version",
        "category",
        "sub_category",
        "expression",
        "function_description",
        "theory_background",
        "applicable_conditions",
        "lifecycle_stage",
        "meaning",
        "usage_scenario",
        "strategy_hint",
    }
    for item in items:
        meta = item.get("meta") or {}
        semantic = {key: meta.get(key) for key in fields if meta.get(key)}
        if semantic:
            store.inject_warehouse_meta(item["name"], semantic)


def _validate_factor(root: Path, name: str) -> list[str]:
    """校验单因子结果中的非有限值、哨兵值和元数据日期。"""
    issues: list[str] = []
    factor_dir = root / name
    files = sorted(factor_dir.glob("*.parquet"))
    if not files:
        return ["无 Parquet 文件"]

    last_dates: list[pd.Timestamp] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows == 0:
            issues.append(f"{path.stem} 为空")
            continue

        value_stats = parquet.metadata.row_group(0).column(0).statistics
        if value_stats and value_stats.min is not None and value_stats.max is not None:
            minimum = float(value_stats.min)
            maximum = float(value_stats.max)
            if not np.isfinite(minimum) or not np.isfinite(maximum):
                issues.append(f"{path.stem} 含无穷值")
            if np.isclose(abs(minimum), _F32_SENTINEL, rtol=1e-7, atol=0.0):
                issues.append(f"{path.stem} 含异常哨兵值")
            if np.isclose(abs(maximum), _F32_SENTINEL, rtol=1e-7, atol=0.0):
                issues.append(f"{path.stem} 含异常哨兵值")

        date_stats = parquet.metadata.row_group(0).column(
            parquet.metadata.num_columns - 1
        ).statistics
        if date_stats and date_stats.max is not None:
            last_dates.append(pd.Timestamp(date_stats.max))

    meta_path = factor_dir / "meta.json"
    if not meta_path.exists():
        issues.append("缺少 meta.json")
        return issues

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_last = meta.get("data_range", {}).get("last_date", "")
    actual_last = max(last_dates).strftime("%Y-%m-%d") if last_dates else ""
    if meta_last != actual_last:
        issues.append(f"meta末日不一致: meta={meta_last}, actual={actual_last}")
    return issues


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="主板优先的因子仓库分阶段重建")
    # [五大类归档 2026-10-08] 默认策略改为 momentum_price_factors（原 price_volume 已合并入档）
    parser.add_argument("--strategy", default="momentum_price_factors", help="YAML 策略名")
    parser.add_argument("--all-active", action="store_true", help="合并加载全部活跃分类")
    parser.add_argument("--all-sources", action="store_true", help="合并加载全部活跃分类 + GTJA191 归档")
    parser.add_argument("--archive", action="store_true", help="加载 GTJA191 归档定义")
    parser.add_argument("--force-qlib", action="store_true", help="全部使用 Qlib 表达式计算")
    parser.add_argument("--start-date", default="2020-01-01", help="重建起始日期")
    parser.add_argument("--end-date", default="2026-09-11", help="重建结束日期")
    parser.add_argument("--qlib-dir", required=True, help="已重建的 Qlib 目录")
    parser.add_argument("--source-warehouse", default=None, help="现有仓库目录")
    parser.add_argument("--output-root", default=None, help="重建输出目录")
    parser.add_argument("--factor", nargs="*", default=None, help="仅重建指定因子")
    parser.add_argument("--no-copy-existing", action="store_true", help="不复制现有仓库")
    parser.add_argument("--only-existing", action="store_true", help="仅重建现有仓库中已存在的因子")
    parser.add_argument("--resume-existing", action="store_true", help="在已有重建目录上续算指定因子")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    qlib_dir = Path(args.qlib_dir).resolve()
    if not (qlib_dir / "calendars" / "day.txt").exists():
        raise FileNotFoundError(f"重建 Qlib 目录无效: {qlib_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else PROJECT_ROOT / f"factor_data/warehouse_rebuild_main_board_{timestamp}"
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.resume_existing:
        raise RuntimeError(f"输出目录非空，拒绝覆盖: {output_root}")

    source_warehouse = (
        Path(args.source_warehouse).resolve()
        if args.source_warehouse
        else PROJECT_ROOT / "factor_data" / "warehouse"
    )
    if args.resume_existing:
        if not output_root.exists():
            raise FileNotFoundError(f"续算目录不存在: {output_root}")
        print(f"[1] 在已有重建目录续算: {output_root}")
    elif not args.no_copy_existing:
        if not source_warehouse.exists():
            raise FileNotFoundError(f"现有仓库不存在: {source_warehouse}")
        shutil.copytree(source_warehouse, output_root)
        print(f"[1] 已复制现有仓库: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"[1] 已创建空仓库: {output_root}")

    import qlworks.config as qlworks_config

    qlworks_config.QLIB_DATA_DIR = qlib_dir

    config = replace(
        DEFAULT_CONFIG,
        warehouse_dir=str(output_root),
        cache_dir=str(output_root.parent / "cache_rebuild_main_board"),
        factors_dir=str(output_root.parent / "qualified_factors_rebuild_main_board"),
    )
    store = FactorStore(config)

    repo_path = str(PROJECT_ROOT / "factor_data" / "factor_library")
    if args.all_sources:
        items = []
        seen_names: set[str] = set()
        for strategy in ACTIVE_STRATEGIES:
            for item in load_factor_definitions(strategy, repo_path=repo_path):
                if item["name"] not in seen_names:
                    items.append(item)
                    seen_names.add(item["name"])
        for item in _load_archive_definitions(GTJA191_ARCHIVE):
            if item["name"] not in seen_names:
                items.append(item)
                seen_names.add(item["name"])
    elif args.archive:
        items = _load_archive_definitions(GTJA191_ARCHIVE)
    elif args.all_active:
        items = []
        seen_names: set[str] = set()
        for strategy in ACTIVE_STRATEGIES:
            for item in load_factor_definitions(strategy, repo_path=repo_path):
                if item["name"] not in seen_names:
                    items.append(item)
                    seen_names.add(item["name"])
    else:
        items = load_factor_definitions(
            strategy_name=args.strategy,
            repo_path=repo_path,
        )
    if args.only_existing:
        items = [
            item for item in items
            if (source_warehouse / item["name"] / "meta.json").exists()
            or (source_warehouse / item["name"]).is_dir()
        ]
    if args.factor:
        requested = set(args.factor)
        items = [item for item in items if item["name"] in requested]
    if not items:
        raise RuntimeError("未找到待重建因子")

    stock_path = qlib_dir / "instruments" / "main_board.txt"
    stocks = _read_stock_codes(stock_path)
    strategy_label = "all_active" if args.all_active else args.strategy
    if args.archive:
        strategy_label = "gtja191_archive"
    if args.all_sources:
        strategy_label = "all_sources"
    print(f"[2] 策略={strategy_label} 因子={len(items)} 主板股票={len(stocks)}")
    print(f"[3] 时间范围: {args.start_date} ~ {args.end_date}")

    batch_factors: list[tuple[str, str]] = []
    fallback_names: list[str] = []
    financial_names: list[str] = []
    for item in items:
        _remove_factor_dir(output_root, item["name"])
        if args.force_qlib:
            fallback_names.append(item["name"])
            continue
        duckdb_expr = _prepare_factor_expression(item)
        if duckdb_expr:
            # YAML 原生 DuckDB 表达式可能依赖 _sw_industry 等联表，
            # 单行样例预检会误判为不可编译，交由 batch_compute 真实执行判定
            converted = duckdb_expr
        else:
            converted = _convert_qlib_expression(item["qlib_expr"])
            if converted and not _can_compile_duckdb_expression(converted):
                converted = None
        if converted:
            batch_factors.append((item["name"], converted))
        elif FactorStore._is_financial_factor(
            duckdb_expr or item["qlib_expr"]
        ):
            financial_names.append(item["name"])
        else:
            fallback_names.append(item["name"])

    failures: dict[str, str] = {}
    if batch_factors:
        print(f"[4] 批量计算: {len(batch_factors)} 个因子")
        fallback_names.extend(
            _run_batch_groups(
                store,
                batch_factors,
                args.start_date,
                args.end_date,
                stocks,
                output_root,
            )
        )

    fallback_names = sorted(set(fallback_names))
    print(f"[5] Qlib 批量回退: {len(fallback_names)} 个因子")
    item_map = {item["name"]: item for item in items}
    fallback_items = [item_map[name] for name in fallback_names]
    remaining_names = _compute_qlib_fallback_batch(
        store,
        fallback_items,
        args.start_date,
        args.end_date,
        stocks,
        qlib_dir,
    )
    remaining_names = sorted(set(remaining_names) | set(financial_names))

    print(f"[6] 剩余逐因子回退: {len(remaining_names)} 个因子")
    for index, name in enumerate(remaining_names, 1):
        item = item_map[name]
        print(f"  [{index}/{len(remaining_names)}] {name}")
        try:
            store.compute_to_warehouse(
                name=name,
                expr=item["qlib_expr"],
                start_date=args.start_date,
                end_date=args.end_date,
                stocks=stocks,
                overwrite=True,
                duckdb_expr=_prepare_factor_expression(item),
            )
        except Exception as exc:
            failures[name] = str(exc)

    _inject_metadata(store, items)

    print("[7] 校验重建因子...")
    validation: dict[str, list[str]] = {}
    for item in items:
        name = item["name"]
        issues = _validate_factor(output_root, name)
        if issues:
            validation[name] = issues

    report = {
        "strategy": strategy_label,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "qlib_dir": str(qlib_dir),
        "output_root": str(output_root),
        "factor_count": len(items),
        "stock_count": len(stocks),
        "batch_count": len(batch_factors),
        "fallback_count": len(fallback_names),
        "failures": failures,
        "validation_issues": validation,
    }
    report_path = output_root / "rebuild_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures or validation:
        raise RuntimeError(f"因子仓库重建未通过，详见: {report_path}")
    print(f"\n因子仓库重建校验通过: {report_path}")


if __name__ == "__main__":
    main()
