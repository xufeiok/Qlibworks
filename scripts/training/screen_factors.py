"""
多因子批量粗筛脚本（Screening）— 基于 qlworks.evaluation.selector 的 5 道快检门。

[定位]
  在 selector.py 之上提供可执行入口：对因子库全量因子做机构级粗筛。
  与 select_factors.py（embedded 特征选择 / 深筛）、train_tree.py（模型训练）分工：
      Screening 粗筛（本脚本）→ Vetting 深评 → 模型选股

[5 道快检门]（selector.screening_pipeline）
  ① 数据质量门   覆盖率 / 常数因子剔除
  ② IC 统计      全样本 Spearman IC / ICIR / 正 IC 占比 win_rate
  ③ 稳定性       滚动窗口 ICIR 正向占比
  ④ 冗余         高相关剔除（保留 IC 强者）
  ⑤ 多重检验     BH（FDR）校正，控制挑选因子时的选择偏差

[两阶段执行（防 OOM，对标 train_tree 分批模式）]
  阶段一：分批加载因子 → 逐批计算 Spearman IC → |IC| 粗筛取 top_k
  阶段二：对 top_k 因子全量加载 → screening_pipeline 跑 5 道门 → 粗筛卡

[用法]
  python screen_factors.py                        # 使用脚本内 CONFIG
  python screen_factors.py --factor-files all --top-k 60
  python screen_factors.py --factor-files style_factors --start-time 2023-01-01 \
      --end-time 2023-12-31 --top-k 10            # 快速试跑

[输出]（默认 scripts/training/ 下）
  - screen_card_{ts}.csv      全因子粗筛卡（每因子一行，含 5 道门判定）
  - screened_factors_{ts}.txt 最终候选因子名单
"""

import os
import sys
import gc
import warnings
import argparse
from datetime import datetime
from pathlib import Path

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

sp = list(sys.path)
conda_sp = [p for p in sp if 'Anaconda' in p and 'site-packages' in p]
roaming_sp = [p for p in sp if 'Roaming' in p]
other_sp = [p for p in sp if p not in conda_sp and p not in roaming_sp]
sys.path = conda_sp + other_sp + roaming_sp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qlworks.evaluation.selector import (
    compute_factor_ics, compute_daily_ic_frame, select_top_by_abs, screening_pipeline,
)
from qlworks.features.dataset import build_custom_feature_cache
from qlworks.config import QLIB_DATA_DIR
from select_factors import load_factors_by_category, build_global_bundle
import qlib
from qlib.data import D

# ==============================================================================
# [全局配置区] - 在此修改运行参数（也可用命令行参数覆盖）
# ==============================================================================
CONFIG = {
    # 因子文件列表（'all' 表示加载所有活跃因子文件）
    "factor_files": [
        "reversal_momentum_factors", "quality_factors", "style_factors",
        "risk_factors", "sentiment_factors", "other_factors",
    ],

    # 股票池与时间范围
    "instruments": "main_board",
    "start_time": "2020-01-01",
    "end_time": "2025-12-31",

    # 标签（与 select_factors.py / train_tree.py 对齐）
    "label_expr": "Ref($close, -5) / Ref($open, -1) - 1",
    "label_name": "LABEL_5D",

    # 阶段一：分批 IC 粗筛
    "batch_size": 20,       # 每批因子数（防 OOM）
    "top_k": 60,            # 进入阶段二深筛的因子数（按 |IC| 降序）
    "min_samples": 50,      # IC 计算有效样本下限

    # 阶段二：5 道快检门参数（selector.DEFAULT_SCREENING_CONFIG 同键）
    "min_coverage": 0.8,         # ① 覆盖率门槛
    "min_nunique": 50,           # ① 常数因子门槛
    "icir_window": 60,           # ③ ICIR 滚动窗口
    "icir_keep_ratio": 0.8,      # ③ ICIR 正向占比保留比例
    "icir_min_keep": 3,          # ③ ICIR 至少保留数
    "redundancy_threshold": 0.90,    # ④ 冗余相关系数阈值
    "redundancy_method": "spearman", # ④ 冗余相关方法
    "correction_alpha": 0.05,    # ⑤ 多重检验显著性水平
    "correction_method": "bh",   # ⑤ bh / holm

    # 分年度 IC 报告（时间切片一致性，报告用、不做硬性门槛）
    "subperiod_ic_threshold": 0.02,  # 年度 |IC| ≥ 此值计为"显著年度"

    # 输出
    "output_dir": os.path.dirname(os.path.abspath(__file__)),
}


def _load_label_series(feature_cache, label_expr: str, label_name: str,
                       start_time: str, end_time: str) -> pd.Series:
    """加载标签并统一为 (instrument, datetime) MultiIndex，与因子数据对齐。"""
    label_raw = D.features(
        feature_cache.resolved_instruments,
        [label_expr],
        start_time, end_time,
    )
    if label_raw.empty:
        raise ValueError(f"标签 {label_expr} 在 {start_time}~{end_time} 为空")
    if isinstance(label_raw.columns, pd.MultiIndex):
        label_raw.columns = label_raw.columns.droplevel(1)
    label_raw = label_raw.rename(columns={label_raw.columns[0]: label_name})
    label_flat = label_raw.reset_index()
    label_flat["instrument"] = label_flat["instrument"].str.lower()
    label_flat = label_flat.set_index(["instrument", "datetime"]).sort_index()
    return label_flat[label_name]


def _normalize_batch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列降为单层（get_warehouse_df 返回 MultiIndex ['feature', name]）。"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    return df


def _load_factor_batch(cache, name_expr_map: dict, batch_names: list,
                       start_time: str, end_time: str) -> pd.DataFrame:
    """加载一批因子数据：优先 parquet 缓存（get_warehouse_df），
    无缓存时回退 Qlib 表达式实时计算（D.features），保证因子覆盖完整。

    name_expr_map: {因子名: Qlib 表达式}（来自因子 YAML 的 expression 字段）
    """
    # 1) parquet 缓存路径（快）
    df = cache.get_warehouse_df(batch_names, start_time=start_time, end_time=end_time)
    if not df.empty:
        return _normalize_batch_columns(df).sort_index()

    # 2) Qlib 表达式实时计算路径（覆盖全部因子，含无 parquet 缓存者）
    exprs = [name_expr_map[n] for n in batch_names if n in name_expr_map]
    if not exprs:
        return pd.DataFrame()
    raw = D.features(cache.resolved_instruments, exprs, start_time, end_time)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(0)
    # 列名（表达式）映射回因子名；instrument 统一小写与标签对齐
    expr_to_name = {name_expr_map[n]: n for n in batch_names if n in name_expr_map}
    raw = raw.rename(columns=lambda c: expr_to_name.get(c, c))
    raw.index = raw.index.set_levels(
        [lv.str.lower() if lv.name == "instrument" else lv for lv in raw.index.levels],
    )
    return raw.sort_index()


def _batch_ic_selection(cache, factor_names, name_expr_map, label_series,
                        start_time: str, end_time: str,
                        batch_size: int = 20, top_k: int = 60,
                        min_samples: int = 50):
    """阶段一：分批加载因子并逐批计算 Spearman IC，按 |IC| 粗筛取 top_k。"""
    all_ic: dict = {}
    n = len(factor_names)
    for i in range(0, n, batch_size):
        batch_names = factor_names[i:i + batch_size]
        print(f"    [分批] 处理 [{i+1}-{min(i+batch_size, n)}]/{n}: "
              f"{', '.join(batch_names[:3])}{'...' if len(batch_names) > 3 else ''}")

        batch_df = _load_factor_batch(cache, name_expr_map, batch_names, start_time, end_time)
        if batch_df.empty:
            continue

        common_index = batch_df.index.intersection(label_series.index)
        if len(common_index) < 100:
            continue
        ics = compute_factor_ics(batch_df.loc[common_index],
                                 label_series.loc[common_index],
                                 min_samples=min_samples)
        all_ic.update(ics.to_dict())

        del batch_df
        gc.collect()

    ic_series = pd.Series(all_ic, dtype=float)
    if ic_series.empty:
        return ic_series, []
    selected = select_top_by_abs(ic_series, top_k)
    print(f"    [粗筛完成] {len(ic_series)} 个因子计算 IC，"
          f"按 |IC| 取 top {len(selected)}: {selected[:5]}{'...' if len(selected) > 5 else ''}")
    return ic_series, selected


def _load_factor_frame(cache, factor_names, name_expr_map, label_series,
                       start_time: str, end_time: str) -> tuple:
    """全量加载指定因子（阶段二用，因子数已收敛到 top_k），返回 (factor_frame, labels)。"""
    df = _load_factor_batch(cache, name_expr_map, factor_names, start_time, end_time)
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    common_index = df.index.intersection(label_series.index)
    if len(common_index) < 100:
        return pd.DataFrame(), pd.Series(dtype=float)
    return df.loc[common_index], label_series.loc[common_index]


def _screening_config_from(cfg: dict) -> dict:
    """提取 selector.screening_pipeline 认识的配置键。"""
    keys = ["min_coverage", "min_nunique", "min_samples",
            "icir_window", "icir_keep_ratio", "icir_min_keep",
            "redundancy_threshold", "redundancy_method",
            "correction_alpha", "correction_method"]
    return {k: cfg[k] for k in keys if k in cfg}


class _Tee:
    """同时输出到终端与日志文件（Windows 终端 stdout 捕获不可靠，日志文件为准）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()          # 逐行落盘，避免缓冲丢失
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ─────────────────────────── 分年度 IC 报告 ───────────────────────────

def _annual_ic_report(daily_ic: pd.DataFrame, ic_threshold: float = 0.02) -> pd.DataFrame:
    """分年度 IC 报告（时间切片一致性）：每因子每年 IC + 正 IC 占比 + 显著年度统计。

    基于 selector.compute_daily_ic_frame 的逐日截面 IC 面板（date × factor）按年度聚合。

    Args:
        daily_ic: 逐日截面 IC 面板（index=datetime，columns=因子）
        ic_threshold: 年度 |IC| ≥ 此值计为"显著年度"

    Returns:
        DataFrame，index=因子名，列为：
          - ic_{YYYY}: 各年度 IC
          - pos_{YYYY}: 各年度正 IC 占比
          - n_years / n_ic_years: 总年度数 / 显著年度数
          - ic_consistency: 显著年度占比（n_ic_years / n_years，0~1）
    """
    if daily_ic is None or daily_ic.empty:
        return pd.DataFrame()
    years = daily_ic.index.year

    annual_ic = daily_ic.groupby(years).mean().T          # 因子 × 年度 IC
    annual_pos = (daily_ic > 0).groupby(years).mean().T   # 因子 × 年度正 IC 占比

    annual_ic.columns = [f"ic_{y}" for y in annual_ic.columns]
    annual_pos.columns = [f"pos_{y}" for y in annual_pos.columns]

    ic_cols = list(annual_ic.columns)
    n_years = len(ic_cols)
    significant = annual_ic.abs() >= ic_threshold
    report = pd.concat([annual_ic, annual_pos], axis=1)
    report["n_years"] = n_years
    report["n_ic_years"] = significant.sum(axis=1)
    report["ic_consistency"] = report["n_ic_years"] / n_years
    return report


def _print_annual_report(report: pd.DataFrame, top_n: int = 60):
    """打印分年度 IC 报告：因子 × 年度 IC 矩阵 + 显著年度占比。"""
    # 仅匹配年度列（ic_{YYYY}），排除 ic_consistency 等汇总列
    ic_cols = [c for c in report.columns if c.startswith("ic_") and c[3:].isdigit()]
    years = sorted({c[3:] for c in ic_cols})
    print(f"\n  分年度 IC 报告（{len(years)} 个年度，|IC|≥阈值计为显著；仅报告，不作筛选门槛）")
    print("  " + "-" * 80)
    display = report[ic_cols + ["n_ic_years", "ic_consistency"]].copy()
    display = display.sort_values("ic_consistency", ascending=False).head(top_n)
    display["ic_consistency"] = display["ic_consistency"].map("{:.0%}".format)
    display[ic_cols] = display[ic_cols].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    display = display.rename(columns={c: f"IC{c[3:]}" for c in ic_cols})
    print(display.to_string())


def _print_summary(result: dict, ic_all: pd.Series):
    card = result["screen_card"]
    candidates = result["candidates"]
    print(f"\n{'=' * 60}")
    print("  粗筛结果总览")
    print(f"{'=' * 60}")

    n_total = len(card)
    n_qc = int(card["quality_ok"].sum()) if n_total else 0
    n_ic = int(ic_all.notna().sum())
    n_stable = len(result["stable_factors"])
    n_kept = len(result["screen_card"][result["screen_card"]["redundant"] == False]) \
        if n_total and "redundant" in card else 0
    n_sig = int(card["significant"].sum()) if n_total else 0
    print(f"  总因子数        : {n_total}")
    print(f"  ① 通过质量门    : {n_qc}/{n_total}  (覆盖率≥{CONFIG['min_coverage']}, nunique≥{CONFIG['min_nunique']})")
    print(f"  ② 计算到 IC     : {n_ic}/{n_total}")
    print(f"  ③ 通过 ICIR 稳定: {n_stable}")
    print(f"  ④ 冗余剔除后    : {n_kept}")
    print(f"  ⑤ 多重检验显著  : {n_sig}")

    if candidates:
        print(f"\n  最终候选因子 ({len(candidates)} 个):")
        for i, f in enumerate(candidates, 1):
            row = card[card["factor_name"] == f].iloc[0] if n_total else None
            ic_str = f", IC={row['ic']:.4f}" if row is not None and pd.notna(row["ic"]) else ""
            print(f"    [{i:2d}] {f}{ic_str}")
    else:
        print("\n  [注意] 无因子通过全部快检门，请放宽 screening 参数")


def main(argv=None):
    parser = argparse.ArgumentParser(description="多因子批量粗筛（selector 5 道快检门）")
    parser.add_argument("--factor-files", help="因子文件列表，逗号分隔；'all' 表示全部活跃因子")
    parser.add_argument("--instruments", default=CONFIG["instruments"])
    parser.add_argument("--start-time", default=CONFIG["start_time"])
    parser.add_argument("--end-time", default=CONFIG["end_time"])
    parser.add_argument("--top-k", type=int, default=CONFIG["top_k"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--min-coverage", type=float, default=CONFIG["min_coverage"])
    parser.add_argument("--icir-window", type=int, default=CONFIG["icir_window"])
    parser.add_argument("--keep-ratio", type=float, default=CONFIG["icir_keep_ratio"])
    parser.add_argument("--redundancy-threshold", type=float, default=CONFIG["redundancy_threshold"])
    parser.add_argument("--correction-method", default=CONFIG["correction_method"])
    parser.add_argument("--subperiod-ic-threshold", type=float, default=CONFIG["subperiod_ic_threshold"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    args = parser.parse_args(argv)

    cfg = dict(CONFIG)
    cfg["instruments"] = args.instruments
    cfg["start_time"] = args.start_time
    cfg["end_time"] = args.end_time
    cfg["top_k"] = args.top_k
    cfg["batch_size"] = args.batch_size
    cfg["min_coverage"] = args.min_coverage
    cfg["icir_window"] = args.icir_window
    cfg["icir_keep_ratio"] = args.keep_ratio
    cfg["redundancy_threshold"] = args.redundancy_threshold
    cfg["correction_method"] = args.correction_method
    cfg["subperiod_ic_threshold"] = args.subperiod_ic_threshold
    cfg["output_dir"] = args.output_dir
    if args.factor_files:
        cfg["factor_files"] = ["all"] if args.factor_files.strip() == "all" \
            else [f.strip() for f in args.factor_files.split(",")]

    # 终端输出同时写入日志文件（Windows 下 stdout 捕获不可靠，以日志为准）
    log_path = os.path.join(cfg["output_dir"], "screen_factors.log")
    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)

    print("=" * 60)
    print("  多因子批量粗筛脚本 (selector.screening_pipeline)")
    print("  5 道快检门：质量门 / IC统计 / ICIR稳定 / 冗余 / 多重检验")
    print("=" * 60)
    print(f"  股票池: {cfg['instruments']}   时间: {cfg['start_time']} ~ {cfg['end_time']}")
    print(f"  因子文件: {cfg['factor_files']}")

    # 1. 初始化 Qlib
    print("\n[1] 初始化 Qlib 环境...")
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn", joblib_backend="threading")

    # 2. 加载因子并按类别分组
    print("\n[2] 加载因子文件...")
    categories = load_factors_by_category(cfg["factor_files"])
    if not categories:
        print("[错误] 未加载到任何因子")
        sys.exit(1)
    for cat_name, factors in sorted(categories.items()):
        print(f"      {cat_name:<22s}: {len(factors):3d} 个因子")
    all_factor_names = [f["name"] for factors in categories.values() for f in factors]
    print(f"      {'总计':<20s}: {len(all_factor_names)} 个因子")

    # 因子名 → Qlib 表达式映射（Qlib 回退加载路径使用）
    name_expr_map = {f["name"]: f["expression"]
                     for factors in categories.values() for f in factors}
    if len(name_expr_map) != len(all_factor_names):
        print("      [警告] 存在重名因子，后加载的表达式覆盖前者")

    # 3. 构建全局因子包与特征缓存
    print(f"\n[3] 构建全局特征缓存 (覆盖 {cfg['start_time']} ~ {cfg['end_time']})...")
    global_bundle = build_global_bundle(categories, cfg["label_expr"], cfg["label_name"])
    feature_cache = build_custom_feature_cache(
        instruments=cfg["instruments"],
        feature_bundle=global_bundle,
        factor_cache_names=[],
        start_time=cfg["start_time"],
        end_time=cfg["end_time"],
        use_dynamic_filter=False,
    )

    # 4. 加载标签（一次全量，与因子数据对齐）
    print("\n[4] 加载标签...")
    label_series = _load_label_series(feature_cache, cfg["label_expr"], cfg["label_name"],
                                      cfg["start_time"], cfg["end_time"])
    print(f"      >>> 标签 {len(label_series):,} 条, "
          f"{label_series.index.get_level_values('datetime').nunique()} 个交易日")

    # 5. 阶段一：分批 IC 粗筛
    print(f"\n[5] 阶段一：分批 IC 粗筛 (batch_size={cfg['batch_size']}, top_k={cfg['top_k']})...")
    ic_all, top_names = _batch_ic_selection(
        feature_cache, all_factor_names, name_expr_map, label_series,
        cfg["start_time"], cfg["end_time"],
        batch_size=cfg["batch_size"], top_k=cfg["top_k"], min_samples=cfg["min_samples"],
    )
    if not top_names:
        print("[错误] 无因子通过 IC 粗筛")
        sys.exit(1)

    # 提示无有效数据的因子（qlib 数据中字段缺失，如 $pe_ttm/$pb 等）
    missing = [f for f in all_factor_names if f not in ic_all.index]
    if missing:
        print(f"      [提示] {len(missing)}/{len(all_factor_names)} 个因子在数据中无有效值被跳过: "
              f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")

    # 6. 阶段二：全量加载 top_k 因子 → 5 道快检门
    print(f"\n[6] 阶段二：对 top {len(top_names)} 因子执行 5 道快检门...")
    factor_frame, labels = _load_factor_frame(feature_cache, top_names, name_expr_map, label_series,
                                              cfg["start_time"], cfg["end_time"])
    if factor_frame.empty:
        print("[错误] 阶段二因子数据为空")
        sys.exit(1)
    print(f"      >>> 因子面板 {factor_frame.shape[0]:,} 行 × {factor_frame.shape[1]} 列")

    result = screening_pipeline(factor_frame, labels, config=_screening_config_from(cfg))

    # 6.5 分年度 IC 报告（时间切片一致性，仅报告、不参与筛选门槛）
    daily_ic = compute_daily_ic_frame(factor_frame, labels)
    annual_report = _annual_ic_report(daily_ic, cfg["subperiod_ic_threshold"])
    if not annual_report.empty:
        _print_annual_report(annual_report)

    # 7. 汇总输出
    _print_summary(result, ic_all)
    card = result["screen_card"]
    if not card.empty:
        # 分年度 IC 列合并进粗筛卡（ic_YYYY / pos_YYYY / ic_consistency）
        if not annual_report.empty:
            card = card.merge(annual_report, left_on="factor_name", right_index=True, how="left")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        card_path = os.path.join(cfg["output_dir"], f"screen_card_{timestamp}.csv")
        card.to_csv(card_path, index=False, encoding="utf-8-sig")
        print(f"\n  完整粗筛卡已保存至: {card_path}")

        candidates = result["candidates"]
        txt_path = os.path.join(cfg["output_dir"], f"screened_factors_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(candidates))
        print(f"  候选因子名单已保存至: {txt_path}")


if __name__ == "__main__":
    main()
