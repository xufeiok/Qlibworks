"""
2024 标签体检脚本。

功能：
- 拉取主板股票在 2024 年的原始 open/close、标签组成项与最终标签
- 自动识别异常股票、异常日期、异常标签样本
- 一次性输出明细与汇总 CSV/JSON，便于定位数据污染来源

默认标签：
    Ref($close, -5) / Ref($open, -1) - 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import qlib
from qlib.data import D

from qlworks.config import PROJECT_ROOT, QLIB_DATA_DIR
from qlworks.features.dataset import _resolve_static_instruments


DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_UNIVERSE = "main_board"
DEFAULT_LABEL_EXPR = "Ref($close, -5) / Ref($open, -1) - 1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runtime" / "diagnostics" / "label_health_2024"


def _ensure_abs_label_column(frame: pd.DataFrame) -> pd.DataFrame:
    """
    保证 DataFrame 中存在 abs_label_5d 列，便于汇总函数复用。
    """
    if "abs_label_5d" in frame.columns:
        return frame
    result = frame.copy()
    result["abs_label_5d"] = result["label_5d"].abs()
    return result


def fetch_label_health_frame(
    start_time: str = DEFAULT_START_DATE,
    end_time: str = DEFAULT_END_DATE,
    instruments: str = DEFAULT_UNIVERSE,
) -> pd.DataFrame:
    """
    拉取原始价格、标签组成项与标签值。

    输入：
    - start_time/end_time: 体检区间
    - instruments: 股票池名称

    输出：
    - 包含 datetime/instrument/open/close/next_open/future_close_5d/label_5d 的 DataFrame
    """
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")
    resolved = _resolve_static_instruments(instruments, start_time=start_time, end_time=end_time, verbose=False)

    fields = [
        "$open",
        "$close",
        "Ref($open, -1)",
        "Ref($close, -5)",
        DEFAULT_LABEL_EXPR,
    ]
    raw = D.features(resolved, fields, start_time=start_time, end_time=end_time)
    frame = raw.reset_index()

    index_names = list(raw.index.names)
    rename_map = {
        index_names[0]: index_names[0],
        index_names[1]: index_names[1],
        "$open": "open",
        "$close": "close",
        "Ref($open, -1)": "next_open",
        "Ref($close, -5)": "future_close_5d",
        DEFAULT_LABEL_EXPR: "label_5d",
    }
    frame = frame.rename(columns=rename_map)

    if "datetime" not in frame.columns or "instrument" not in frame.columns:
        raise ValueError(f"Qlib 返回列缺少 datetime/instrument，当前列为: {frame.columns.tolist()}")

    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["instrument"] = frame["instrument"].astype(str).str.lower()
    return frame.sort_values(["datetime", "instrument"], kind="mergesort").reset_index(drop=True)


def flag_label_anomalies(
    frame: pd.DataFrame,
    abs_label_threshold: float = 5.0,
    min_price_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    给标签样本打异常标记。

    判定规则：
    - next_open/future_close_5d 非正或过小
    - label 非有限值
    - |label| 超过阈值
    """
    result = frame.copy()

    price_issue = (
        result["open"].isna()
        | result["close"].isna()
        | result["next_open"].isna()
        | result["future_close_5d"].isna()
        | (result["open"] <= 0)
        | (result["close"] <= 0)
        | (result["next_open"] <= min_price_threshold)
        | (result["future_close_5d"] <= min_price_threshold)
    )
    label_issue = (
        ~pd.to_numeric(result["label_5d"], errors="coerce").notna()
        | (result["label_5d"].abs() > abs_label_threshold)
    )

    result["price_issue"] = price_issue.astype(bool)
    result["label_issue"] = label_issue.astype(bool)
    result["is_anomaly"] = (result["price_issue"] | result["label_issue"]).astype(bool)
    result["abs_label_5d"] = result["label_5d"].abs()
    return result


def summarize_anomalies_by_stock(frame: pd.DataFrame) -> pd.DataFrame:
    """
    按股票汇总异常标签分布。
    """
    frame = _ensure_abs_label_column(frame)
    anomalies = frame.loc[frame["is_anomaly"]].copy()
    if anomalies.empty:
        return pd.DataFrame(columns=["instrument", "anomaly_count", "label_issue_count", "price_issue_count", "max_abs_label_5d"])

    summary = (
        anomalies.groupby("instrument", as_index=False)
        .agg(
            anomaly_count=("is_anomaly", "sum"),
            label_issue_count=("label_issue", "sum"),
            price_issue_count=("price_issue", "sum"),
            max_abs_label_5d=("abs_label_5d", "max"),
        )
        .sort_values(["anomaly_count", "max_abs_label_5d", "instrument"], ascending=[False, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return summary


def summarize_anomalies_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    """
    按日期汇总异常标签分布。
    """
    frame = _ensure_abs_label_column(frame)
    anomalies = frame.loc[frame["is_anomaly"]].copy()
    if anomalies.empty:
        return pd.DataFrame(columns=["datetime", "anomaly_count", "label_issue_count", "price_issue_count", "max_abs_label_5d"])

    summary = (
        anomalies.groupby("datetime", as_index=False)
        .agg(
            anomaly_count=("is_anomaly", "sum"),
            label_issue_count=("label_issue", "sum"),
            price_issue_count=("price_issue", "sum"),
            max_abs_label_5d=("abs_label_5d", "max"),
        )
        .sort_values(["anomaly_count", "max_abs_label_5d", "datetime"], ascending=[False, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return summary


def build_overview(frame: pd.DataFrame) -> dict:
    """
    构建体检摘要。
    """
    frame = _ensure_abs_label_column(frame)
    anomaly_rows = frame.loc[frame["is_anomaly"]]
    return {
        "total_rows": int(len(frame)),
        "anomaly_rows": int(frame["is_anomaly"].sum()),
        "label_issue_rows": int(frame["label_issue"].sum()),
        "price_issue_rows": int(frame["price_issue"].sum()),
        "unique_anomaly_stocks": int(anomaly_rows["instrument"].nunique()) if not anomaly_rows.empty else 0,
        "unique_anomaly_dates": int(anomaly_rows["datetime"].nunique()) if not anomaly_rows.empty else 0,
        "max_abs_label_5d": float(frame["abs_label_5d"].max()) if not frame.empty else None,
        "label_mean": float(frame["label_5d"].mean()) if not frame.empty else None,
        "label_std": float(frame["label_5d"].std()) if not frame.empty else None,
        "label_min": float(frame["label_5d"].min()) if not frame.empty else None,
        "label_max": float(frame["label_5d"].max()) if not frame.empty else None,
    }


def save_reports(
    frame: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 200,
) -> dict[str, Path]:
    """
    将异常明细与汇总保存到目录。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anomalies = frame.loc[frame["is_anomaly"]].copy()
    anomalies = anomalies.sort_values(
        ["abs_label_5d", "datetime", "instrument"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    by_stock = summarize_anomalies_by_stock(frame)
    by_date = summarize_anomalies_by_date(frame)
    overview = build_overview(frame)

    all_rows_path = output_path / "label_health_all_rows.csv"
    anomaly_rows_path = output_path / "label_health_anomalies.csv"
    top_rows_path = output_path / "label_health_anomalies_top.csv"
    by_stock_path = output_path / "label_health_by_stock.csv"
    by_date_path = output_path / "label_health_by_date.csv"
    overview_path = output_path / "label_health_overview.json"

    frame.to_csv(all_rows_path, index=False, encoding="utf-8-sig")
    anomalies.to_csv(anomaly_rows_path, index=False, encoding="utf-8-sig")
    anomalies.head(top_n).to_csv(top_rows_path, index=False, encoding="utf-8-sig")
    by_stock.to_csv(by_stock_path, index=False, encoding="utf-8-sig")
    by_date.to_csv(by_date_path, index=False, encoding="utf-8-sig")
    overview_path.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": output_path,
        "all_rows": all_rows_path,
        "anomalies": anomaly_rows_path,
        "anomalies_top": top_rows_path,
        "by_stock": by_stock_path,
        "by_date": by_date_path,
        "overview": overview_path,
    }


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。
    """
    parser = argparse.ArgumentParser(description="体检 2024 年标签与原始价格数据。")
    parser.add_argument("--start-time", default=DEFAULT_START_DATE, help="开始日期，默认 2024-01-01")
    parser.add_argument("--end-time", default=DEFAULT_END_DATE, help="结束日期，默认 2024-12-31")
    parser.add_argument("--instruments", default=DEFAULT_UNIVERSE, help="股票池名称，默认 main_board")
    parser.add_argument("--abs-label-threshold", type=float, default=5.0, help="标签绝对值异常阈值，默认 5.0")
    parser.add_argument("--min-price-threshold", type=float, default=0.05, help="价格过小阈值，默认 0.05")
    parser.add_argument("--top-n", type=int, default=200, help="导出最严重异常样本数，默认 200")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认 runtime/diagnostics/label_health_2024",
    )
    return parser


def main() -> None:
    """
    脚本入口。
    """
    args = build_parser().parse_args()

    print("=" * 68)
    print("  2024 标签体检：异常股票 / 异常日期 / 原始 open-close 联查")
    print("=" * 68)
    print(f"[1] 拉取样本: universe={args.instruments}, {args.start_time} ~ {args.end_time}")
    frame = fetch_label_health_frame(
        start_time=args.start_time,
        end_time=args.end_time,
        instruments=args.instruments,
    )
    print(f"    >>> 原始样本数: {len(frame):,}")

    print("[2] 执行异常判定...")
    diagnosed = flag_label_anomalies(
        frame,
        abs_label_threshold=args.abs_label_threshold,
        min_price_threshold=args.min_price_threshold,
    )
    overview = build_overview(diagnosed)
    print(f"    >>> 异常样本数: {overview['anomaly_rows']:,}")
    print(f"    >>> 标签异常: {overview['label_issue_rows']:,}")
    print(f"    >>> 价格异常: {overview['price_issue_rows']:,}")
    print(f"    >>> 异常股票数: {overview['unique_anomaly_stocks']:,}")
    print(f"    >>> 异常日期数: {overview['unique_anomaly_dates']:,}")
    print(f"    >>> 最大 |label_5d|: {overview['max_abs_label_5d']}")

    print("[3] 写出体检报告...")
    outputs = save_reports(diagnosed, output_dir=args.output_dir, top_n=args.top_n)
    print(f"    >>> 输出目录: {outputs['output_dir']}")
    print(f"    >>> 异常明细: {outputs['anomalies']}")
    print(f"    >>> 股票汇总: {outputs['by_stock']}")
    print(f"    >>> 日期汇总: {outputs['by_date']}")
    print(f"    >>> 概览摘要: {outputs['overview']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
