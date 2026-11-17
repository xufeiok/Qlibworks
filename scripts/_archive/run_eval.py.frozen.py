#!/usr/bin/env python3
"""
单因子评测入口脚本 — 仓库优先（v3.0）

数据流：
  1. 首次：batch_compute.py --all  → 将因子写入 warehouse/
  2. 每周：batch_compute.py --all --append → 增量追加新数据
  3. 评测：run_eval.py --factor KDJ_K → 从 warehouse 读取，只做评测

用法：
  # 评测单个因子（从 warehouse 读取数据）
  python run_eval.py --factor KDJ_K

  # 评测某个分类
  python run_eval.py --category price_volume_factors

  # 评测仓库中所有因子
  python run_eval.py --all

  # 自定义参数
  python run_eval.py --factor KDJ_K --pool main_board --start 2020-01-01

  # 仅查看因子列表（不评测）
  python run_eval.py --status

  # 强制从 ClickHouse 实时计算并写入仓库（不读取缓存）
  python run_eval.py --factor KDJ_K --recompute
"""
import argparse
from dataclasses import replace
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")

from qlworks.evaluation import FactorStore, DEFAULT_CONFIG
from qlworks.evaluation.runner import FactorEvaluator
from qlworks.factors import FactorLibraryManager
from qlworks.config import QLIB_DATA_DIR
from qlworks.pipeline_config import (
    FILTER_LIMIT_UPDOWN, FILTER_SUSPENDED, INSTRUMENTS,
)

# ── 自动发现 instruments 目录下的可用股票池 ──
_INSTRUMENTS_DIR = QLIB_DATA_DIR / "instruments"
_AVAILABLE_POOLS = [INSTRUMENTS] if (_INSTRUMENTS_DIR / f"{INSTRUMENTS}.txt").exists() else []


# ==============================================================================
# [Phase 2 入库决策]（2026-10-12 由 admit_to_multifactor.py 合并而来）
# 单因子评测（Phase 1）完成后，在同一脚本内直接执行"三关准入 + 写候选池"，
# 与机构范式对齐：WW BRAIN 模拟器→提交校验一气呵成、DE Shaw OOS+评审即入池。
# candidate_pool.json 格式逐字节保持（factors[].status），训练端零改动。
# ==============================================================================
import json as _json
import numpy as _np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FACTOR_LIBRARY_DIR = _PROJECT_ROOT / "factor_data" / "factor_library"
_QUALIFIED_DIR = _PROJECT_ROOT / "factor_data" / "qualified_factors"
_REGISTRY_DIR = _PROJECT_ROOT / "factor_data" / "registry"
_REPORTS_DIR = _PROJECT_ROOT / "factor_data" / "reports"
_CANDIDATE_POOL_PATH = _REGISTRY_DIR / "candidate_pool.json"
_TIER_DIRS = ["core", "satellite", "archive"]

from qlworks.pipeline_config import (  # noqa: E402  单一事实源共享配置
    LABEL_EXPR as _ADMIT_LABEL_EXPR, LABEL_NAME as _ADMIT_LABEL_NAME, INSTRUMENTS as _ADMIT_INSTRUMENTS,
    CORR_THRESHOLD, ICIR_IMPROVE_MIN,
    ADMISSION_CUTOFF_DATE, ADMISSION_WINDOW_YEARS,
)

_ADMIT_THRESHOLDS = {
    "max_correlation_existing": CORR_THRESHOLD,  # 与已有池因子相关性上限（pipeline_config）
    "max_correlation_barra": 0.50,           # 与 Barra 风格因子相关性上限
    "min_oos_icir": 0.5,                     # 样本外 ICIR 下限
    "min_recent_3y_ic_positive_ratio": 0.60, # 近 3 年 IC 正向比例下限
    "direction_consistency_required": True,  # 所有子时段 IC 是否必须同号
}

# ── 准入窗口（动态：截止日前 ADMISSION_WINDOW_YEARS 年，年度切片） ──
_ADMISSION_WINDOW = None  # 惰性初始化: {"sub_periods":[...], "corr_start":..., "corr_end":...}


def _build_admission_window() -> dict:
    """准入窗口严格早于 ADMISSION_CUTOFF_DATE（避免测试期泄漏）。"""
    cutoff = pd.Timestamp(ADMISSION_CUTOFF_DATE)
    end_dt = cutoff - pd.Timedelta(days=1)
    start_dt = end_dt - pd.DateOffset(years=ADMISSION_WINDOW_YEARS) + pd.Timedelta(days=1)
    sub = []
    for y in range(start_dt.year, end_dt.year + 1):
        ys = f"{y}-01-01"
        ye = f"{y}-12-31"
        if y == end_dt.year:
            ye = end_dt.strftime("%Y-%m-%d")
        sub.append((str(y), ys, ye))
    return {
        "sub_periods": sub,
        "corr_start": start_dt.strftime("%Y-%m-%d"),
        "corr_end": end_dt.strftime("%Y-%m-%d"),
    }


def _admission_window():
    global _ADMISSION_WINDOW
    if _ADMISSION_WINDOW is None:
        _ADMISSION_WINDOW = _build_admission_window()
    return _ADMISSION_WINDOW


def _load_admit_label_frame(start: str, end: str) -> pd.Series | None:
    """准入窗口标签（与 admit 原实现一致：DK 标签 + 可交易过滤）。"""
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.config import QLIB_DATA_DIR as _QLIB_DATA_DIR
        from qlworks.factors.filter_utils import filter_untradeable_labels

        qlib.init(provider_uri=str(_QLIB_DATA_DIR), region=REG_CN)
        ins_file = Path(str(_QLIB_DATA_DIR)) / "instruments" / f"{_ADMIT_INSTRUMENTS}.txt"
        if not ins_file.exists():
            return None
        instruments = []
        with open(ins_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    instruments.append(parts[0])
        if not instruments:
            return None
        raw = D.features(instruments, [_ADMIT_LABEL_EXPR], start, end)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw = raw.rename(columns={raw.columns[0]: _ADMIT_LABEL_NAME})
        raw = filter_untradeable_labels(raw, instruments, start, end)
        if raw is None or raw.empty:
            return None
        flat = raw.reset_index()
        flat["instrument"] = flat["instrument"].str.lower()
        return flat.set_index(["instrument", "datetime"]).sort_index()[_ADMIT_LABEL_NAME]
    except Exception as e:
        logger.warning(f"[admit] 标签加载失败（{e}），增量检验回退近似路径")
        return None


def _load_warehouse_series(name: str, start: str, end: str) -> pd.Series | None:
    """从 warehouse 读取因子在准入窗口内的序列（单文件 data.parquet，instrument 统一小写）。

    [2026-10-12 修复] warehouse 实际存储为 {name}/data.parquet 单文件
    （MultiIndex=(instrument,datetime)，列名 value，含全年份）；
    旧实现按 {year}.parquet 分片读取，导致准入窗口数据永远缺失、三关检验退化为默认通过。
    """
    fdir = _PROJECT_ROOT / "factor_data" / "warehouse" / name
    pf = fdir / "data.parquet"
    if not pf.exists():
        return None
    try:
        df = pd.read_parquet(pf)
    except Exception as e:
        logger.warning(f"[admit] {name} data.parquet 读取失败: {e}")
        return None
    if df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex):
        dts = df.index.get_level_values("datetime")
        df = df.loc[(dts >= start) & (dts <= end)]
        series = df.iloc[:, 0]
        if series.index.nlevels == 2:
            series = series.rename_axis(index=["instrument", "datetime"])
            series.index = series.index.set_levels(
                [lv.str.lower() if lv.name == "instrument" else lv for lv in series.index.levels],
            )
    else:
        series = df.iloc[:, 0]
    series = series.replace([_np.inf, -_np.inf], _np.nan).dropna()
    if series.empty:
        return None
    return series


def _slice_period(series: pd.Series, start: str, end: str) -> pd.Series:
    dts = series.index.get_level_values("datetime")
    return series[(dts >= start) & (dts <= end)]


def _daily_ic(factor_s: pd.Series, label_s: pd.Series, min_cross: int = 20) -> pd.Series:
    """逐日截面 Spearman IC（与 admit 原实现一致；无区分度截面记 0）。"""
    frame = pd.concat({"f": factor_s, "y": label_s}, axis=1).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    daily = {}
    for dt, g in frame.groupby(level="datetime"):
        if len(g) < min_cross:
            continue
        f_rank = g["f"].rank()
        y_rank = g["y"].rank()
        if f_rank.nunique() < 2 or y_rank.nunique() < 2:
            daily[dt] = 0.0
            continue
        ic = f_rank.corr(y_rank)
        if pd.notna(ic):
            daily[dt] = ic
    return pd.Series(daily, dtype=float)


def _nw_icir(ic_series: pd.Series, annual_factor: float = 252.0) -> float:
    """Newey-West 自相关修正 ICIR（与全链路口径一致）。"""
    ic_clean = ic_series.dropna()
    n = len(ic_clean)
    if n < 10:
        return 0.0
    mean = float(ic_clean.mean())
    std = float(ic_clean.std())
    if std <= 1e-12 or abs(mean) <= 1e-12:
        return 0.0
    from qlworks.evaluation.ic_analysis import calc_newey_west_tstat
    nw = calc_newey_west_tstat(ic_clean)
    nw_se = nw.get("nw_std") or 0.0
    n_obs = nw.get("n_obs") or n
    nw_std_series = nw_se * _np.sqrt(n_obs) if n_obs > 0 else std
    if nw_std_series <= 1e-12:
        return 0.0
    return mean / nw_std_series * _np.sqrt(annual_factor)


def load_candidate_pool() -> dict:
    """读取候选池；v1 旧格式自动迁移为 v2（与 admit 原实现一致）。"""
    if _CANDIDATE_POOL_PATH.exists():
        with open(_CANDIDATE_POOL_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
        if "factors" not in data:
            legacy = data.get("candidate_pool") or []
            return {
                "_meta": {"version": "2.0", "set_version": "v1",
                          "description": "多因子准入候选池（Alpha Book）— 由 run_eval.py --admit 三关检验写入，下游 train_tree-doubao.py 从此读取因子名单",
                          "pipeline_note": "v1 旧格式自动迁移，遗留候选仅登记 rejected",
                          "updated_at": pd.Timestamp.now().isoformat()[:19],
                          "admit_thresholds": _ADMIT_THRESHOLDS},
                "factors": [], "rejected": legacy,
                "stats": {"total_candidates": len(legacy), "admitted": 0,
                          "rejected_corr": 0, "rejected_marginal": 0, "rejected_direction": 0},
            }
        return data
    return {
        "_meta": {"version": "2.0", "set_version": "v1",
                  "description": "多因子准入候选池（Alpha Book）— 由 run_eval.py --admit 三关检验写入，下游 train_tree-doubao.py 从此读取因子名单",
                  "updated_at": pd.Timestamp.now().isoformat()[:19],
                  "admit_thresholds": _ADMIT_THRESHOLDS},
        "factors": [], "rejected": [],
        "stats": {"total_candidates": 0, "admitted": 0,
                  "rejected_corr": 0, "rejected_marginal": 0, "rejected_direction": 0},
    }


def save_candidate_pool(pool: dict):
    _REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CANDIDATE_POOL_PATH, "w", encoding="utf-8") as f:
        _json.dump(pool, f, ensure_ascii=False, indent=2)
    logger.info(f"[admit] 候选池已更新: {_CANDIDATE_POOL_PATH}（factors={len(pool['factors'])}, rejected={len(pool['rejected'])}）")


def compute_correlation_matrix(factor_names: list[str], start: str, end: str) -> pd.DataFrame:
    """从 warehouse 读取因子值，计算两两 RankIC 相关矩阵（单文件 data.parquet 结构）。"""
    factor_data = {}
    missing = []
    for name in factor_names:
        fdir = _PROJECT_ROOT / "factor_data" / "warehouse" / name
        pf = fdir / "data.parquet"
        if not pf.exists():
            missing.append(name)
            continue
        try:
            df = pd.read_parquet(pf)
        except Exception:
            missing.append(name)
            continue
        if isinstance(df.index, pd.MultiIndex):
            dts = df.index.get_level_values("datetime")
            df = df.loc[(dts >= start) & (dts <= end)]
        factor_data[name] = df

    if missing:
        logger.warning(f"[admit] warehouse 无 {start[:4]}~{end[:4]} 数据: {missing}")
        if not factor_data:
            return pd.DataFrame()

    merged = pd.concat(factor_data.values(), axis=1, keys=factor_data.keys())
    merged.columns = merged.columns.get_level_values(0)
    merged = merged.replace([_np.inf, -_np.inf], _np.nan)

    daily_corrs = []
    for dt, group in merged.groupby(level="datetime"):
        valid = group.dropna(how="all")
        if len(valid) < 50:
            continue
        ranked = valid.rank(pct=True)
        corr = ranked.corr(method="spearman")
        daily_corrs.append(corr)
    if not daily_corrs:
        return pd.DataFrame()
    corr_sum = sum(c.reindex(merged.columns, axis=0).reindex(merged.columns, axis=1).fillna(0) for c in daily_corrs)
    return corr_sum / len(daily_corrs)


def check_correlation(new_factor: str, existing_factors: list[dict],
                      corr_matrix: pd.DataFrame,
                      threshold: float = 0.70) -> tuple[bool, list[str]]:
    """第一关：与候选池已有因子相关性检验。"""
    if not existing_factors:
        return True, []
    if corr_matrix is None or corr_matrix.empty or new_factor not in corr_matrix.index:
        return True, []
    high_corr = []
    for e in existing_factors:
        ename = e.get("name")
        if ename and ename in corr_matrix.columns:
            r = abs(corr_matrix.loc[new_factor, ename])
            if pd.notna(r) and r > threshold:
                high_corr.append(f"{ename}({r:.2f})")
    return (len(high_corr) == 0), high_corr


def check_marginal_contribution(new_factor: str, existing_factors: list[dict],
                                label_frame: pd.Series | None = None,
                                corr_matrix: pd.DataFrame | None = None,
                                start: str = "", end: str = "") -> tuple[bool, str]:
    """第二关：增量 ICIR 边际贡献（等权组合信号对比，NW 修正，分年度）。"""
    if not existing_factors:
        return True, "候选池为空，无需边际检验"
    existing_names = [f["name"] for f in existing_factors]
    if label_frame is not None:
        factor_frames = {}
        missing = []
        for name in existing_names + [new_factor]:
            series = _load_warehouse_series(name, start, end)
            if series is None:
                missing.append(name)
            else:
                factor_frames[name] = series
        have_existing = [n for n in existing_names if n in factor_frames]
        if new_factor in factor_frames and have_existing:
            base = pd.DataFrame({n: factor_frames[n] for n in have_existing})
            base_rank = base.groupby(level="datetime").rank(pct=True)
            s0 = base_rank.mean(axis=1)
            joined = pd.concat({"s0": s0, "new": factor_frames[new_factor]}, axis=1)
            joined_rank = joined.groupby(level="datetime").rank(pct=True)
            s1 = (joined_rank["s0"] + joined_rank["new"]) / 2
            details = []
            deltas = []
            for label, ys, ye in _admission_window()["sub_periods"]:
                ic0 = _daily_ic(_slice_period(s0, ys, ye), _slice_period(label_frame, ys, ye))
                ic1 = _daily_ic(_slice_period(s1, ys, ye), _slice_period(label_frame, ys, ye))
                if len(ic0) < 10 or len(ic1) < 10:
                    continue
                icir0 = _nw_icir(ic0)
                icir1 = _nw_icir(ic1)
                deltas.append(icir1 - icir0)
                details.append(f"{label} ICIR_NW {icir0:.3f}→{icir1:.3f} (Δ{icir1 - icir0:+.3f})")
            if deltas:
                avg_delta = float(_np.mean(deltas))
                if avg_delta < ICIR_IMPROVE_MIN:
                    return False, f"增量 ICIR 边际贡献为负: {avg_delta:+.3f} ({'; '.join(details)})"
                return True, f"增量 ICIR 边际贡献: {avg_delta:+.3f} ({'; '.join(details)})"
        logger.warning(f"[admit] {new_factor}: 增量 ICIR 检验数据不足，回退平均相关近似")
    if corr_matrix is None or new_factor not in corr_matrix.index:
        return True, "新因子不在相关矩阵中，默认通过"
    vals = []
    for ename in existing_names:
        if ename in corr_matrix.columns:
            v = abs(corr_matrix.loc[new_factor, ename])
            vals.append(v)
    if not vals:
        return True, "与已有池因子无交集"
    avg_corr = _np.mean(vals)
    if avg_corr > 0.40:
        return False, f"冗余度偏高: 与已有池平均相关 {avg_corr:.2f} > 0.40（标签缺失近似）"
    return True, f"通过（与已有池平均相关 {avg_corr:.2f}，标签缺失近似）"


def check_direction_consistency(new_factor: str, label_frame: pd.Series | None = None,
                                start: str = "", end: str = "") -> tuple[bool, str]:
    """第三关：分年度 IC 方向一致性（子时段 IC 同号）。"""
    series = _load_warehouse_series(new_factor, start, end)
    if series is None:
        return True, "因子无 warehouse 数据，默认通过"
    if len(series) < 1000:
        return True, "数据量不足"
    if label_frame is not None:
        ic_means = []
        details = []
        for label, ys, ye in _admission_window()["sub_periods"]:
            ic_series = _daily_ic(_slice_period(series, ys, ye), _slice_period(label_frame, ys, ye))
            if ic_series.empty:
                continue
            ic_means.append(ic_series.mean())
            details.append(f"{label} IC={ic_series.mean():.4f}")
        if len(ic_means) >= 2:
            if all(v > 0 for v in ic_means) or all(v < 0 for v in ic_means):
                return True, "IC 方向一致: " + "; ".join(details)
            return False, "IC 方向不一致: " + "; ".join(details)
        logger.warning(f"[admit] {new_factor}: 标签子时段不足，回退因子值符号校验")
    signs = []
    details = []
    for label, ys, ye in _admission_window()["sub_periods"]:
        subset = _slice_period(series, ys, ye)
        if len(subset) < 100:
            continue
        mean_val = subset.mean()
        signs.append(mean_val)
        details.append(f"{label}={mean_val:.4f}")
    if not signs:
        return True, "子时段无有效数据"
    direction = _np.sign(signs)
    if len(set(direction)) > 1:
        return False, f"方向不一致: {'; '.join(details)}"
    return True, f"方向一致: {'; '.join(details)}"


def _bump_set_version(pool: dict) -> str:
    meta = pool.setdefault("_meta", {})
    cur = meta.get("set_version", "v1")
    try:
        n = int(str(cur).lstrip("v")) + 1
    except (ValueError, TypeError):
        n = 2
    meta["set_version"] = f"v{n}"
    return meta["set_version"]


def _admit_entry(factor_name: str, factor_info: dict, result: dict, tier: str,
                 ic_mean: float, icir_nw: float) -> dict | None:
    """构造候选池 Alpha Book 标准条目（direction 由真实 IC 符号推导）。"""
    from datetime import datetime
    if ic_mean is None:
        print(f"[admit] {factor_name}: 评测无 IC 指标，判 pending_data，不入池")
        return None
    direction = "positive" if float(ic_mean) > 0 else "negative"
    now = datetime.now().isoformat(timespec="seconds")
    result["latest_icir"] = icir_nw
    return {
        "name": factor_name,
        "tier": tier,
        "direction": direction,
        "category": factor_info.get("category", ""),
        "sub_category": factor_info.get("sub_category", ""),
        "meaning": factor_info.get("meaning", ""),
        "source_file": factor_info.get("_source_file", ""),
        "latest_icir": float(icir_nw or 0),
        "admitted_at": now,
        "eval_date": now,
        "rolling_ic": {},
        "tier_history": [{"tier": tier, "at": now}],
        "status": "admitted",
        "_metrics": {"ic_mean": ic_mean, "icir_nw": icir_nw},
        "_screening": result,
    }


def admit_factor(new_factor_name: str, new_factor_info: dict,
                 existing_factors: list[dict], corr_matrix: pd.DataFrame,
                 label_frame: pd.Series | None = None) -> dict:
    """对新因子执行三关准入检验（与 admit 原实现一致）。"""
    result = {
        "name": new_factor_name,
        "tier": new_factor_info.get("_tier", "satellite"),
        "category": new_factor_info.get("category", ""),
        "sub_category": new_factor_info.get("sub_category", ""),
        "meaning": new_factor_info.get("meaning", ""),
        "_source_file": new_factor_info.get("_source_file", ""),
        "admitted": False,
        "reasons": [],
        "corr_check": {"passed": False, "high_corr_factors": []},
        "marginal_check": {"passed": False, "detail": ""},
        "direction_check": {"passed": False, "detail": ""},
    }
    corr_passed, high_corr = check_correlation(
        new_factor_name, existing_factors, corr_matrix,
        threshold=_ADMIT_THRESHOLDS["max_correlation_existing"])
    result["corr_check"]["passed"] = corr_passed
    result["corr_check"]["high_corr_factors"] = high_corr
    if not corr_passed:
        result["reasons"].append(f"相关性检验未通过: 与 {', '.join(high_corr)} 高度相关")

    win = _admission_window()
    marginal_passed, marginal_detail = check_marginal_contribution(
        new_factor_name, existing_factors, label_frame, corr_matrix,
        start=win["corr_start"], end=win["corr_end"])
    result["marginal_check"]["passed"] = marginal_passed
    result["marginal_check"]["detail"] = marginal_detail
    if not marginal_passed:
        result["reasons"].append(f"边际贡献检验未通过: {marginal_detail}")

    dir_passed, dir_detail = check_direction_consistency(
        new_factor_name, label_frame, start=win["corr_start"], end=win["corr_end"])
    result["direction_check"]["passed"] = dir_passed
    result["direction_check"]["detail"] = dir_detail
    if not dir_passed:
        result["reasons"].append(f"方向一致性未通过: {dir_detail}")

    result["admitted"] = corr_passed and marginal_passed and dir_passed
    return result


def run_admit_decision(factor_name: str, factor_info: dict, tier: str,
                       ic_mean: float, icir_nw: float,
                       label_frame: pd.Series | None = None) -> bool:
    """[Phase 2] 单因子评测完成后立即执行准入决策并写候选池。返回是否 admitted。"""
    pool = load_candidate_pool()
    existing = list(pool.get("factors", []))
    if any(f.get("name") == factor_name for f in existing):
        print(f"  [admit] {factor_name} 已在候选池中，跳过")
        return True
    win = _admission_window()
    corr_matrix = compute_correlation_matrix(
        [factor_name] + [f["name"] for f in existing],
        win["corr_start"], win["corr_end"])
    result = admit_factor(factor_name, factor_info, existing, corr_matrix, label_frame)
    print(f"   [准入] 相关:{'✓' if result['corr_check']['passed'] else '✗'} "
          f"边际:{'✓' if result['marginal_check']['passed'] else '✗'} "
          f"方向:{'✓' if result['direction_check']['passed'] else '✗'}")
    if result["admitted"]:
        entry = _admit_entry(factor_name, factor_info, result, tier, ic_mean, icir_nw)
        if entry is None:
            return False
        pool["factors"].append(entry)
        print(f"   [准入] {factor_name} (tier={tier}, direction={entry['direction']}, icir={entry['latest_icir']:.2f})")
    else:
        pool["rejected"].append(result)
        print(f"   [拒绝] {factor_name} 原因: {'; '.join(result['reasons'])}")
    pool["_meta"]["updated_at"] = pd.Timestamp.now().isoformat()[:19]
    _bump_set_version(pool)
    pool["stats"] = {
        "total_candidates": len(pool["factors"]) + len(pool["rejected"]),
        "admitted": len(pool["factors"]),
        "rejected_corr": sum(1 for r in pool["rejected"] if not r.get("corr_check", {}).get("passed", False)),
        "rejected_marginal": sum(1 for r in pool["rejected"] if not r.get("marginal_check", {}).get("passed", False)),
        "rejected_direction": sum(1 for r in pool["rejected"] if not r.get("direction_check", {}).get("passed", False)),
    }
    save_candidate_pool(pool)
    return result["admitted"]


def _collect_factors(args):
    """从 YAML 因子库收集因子列表（含 duckdb_expr 用于自动计算）。

    [P1-4] --factor-list：直接读取 screen_factors.py 粗筛输出名单
    （screened_factors_*.txt，每行一个因子名），实现粗筛 → 深评漏斗衔接。
    """
    m = FactorLibraryManager()
    factors = []

    if args.factor_list:
        names = _read_factor_list(args.factor_list)
        for name in names:
            info = _find_factor(m, name)
            if info:
                factors.append(info)
            else:
                logger.warning(f"{name}: 因子库中未找到定义，跳过")
        return factors

    if args.factor:
        info = _find_factor(m, args.factor)
        if info:
            factors.append(info)
    elif args.category:
        cfg = m.load_strategy_config(args.category)
        for fd in cfg.get("factors", []):
            name = fd.get("name")
            if not name:
                continue
            expr = fd.get("expression", "")
            duckdb_expr = ""
            if isinstance(expr, dict):
                duckdb_expr = str(expr.get("duckdb", ""))
                expr = str(expr.get("qlib", str(expr)))
            factors.append({"name": name, "expr": str(expr),
                           "duckdb_expr": duckdb_expr, "category": args.category,
                           "sub_category": fd.get("sub_category", ""),
                           "meaning": fd.get("meaning", ""),
                           "_source_file": fd.get("_source_file", "")})
    elif args.all:
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            try:
                cfg = m.load_strategy_config(s)
                for fd in cfg.get("factors", []):
                    name = fd.get("name")
                    if not name:
                        continue
                    expr = fd.get("expression", "")
                    duckdb_expr = ""
                    if isinstance(expr, dict):
                        duckdb_expr = str(expr.get("duckdb", ""))
                        expr = str(expr.get("qlib", str(expr)))
                    factors.append({"name": name, "expr": str(expr),
                                   "duckdb_expr": duckdb_expr, "category": s,
                                   "sub_category": fd.get("sub_category", ""),
                                   "meaning": fd.get("meaning", ""),
                                   "_source_file": fd.get("_source_file", "")})
            except Exception:
                pass

    return factors


def _find_factor(m, name):
    for s in [s for s in m.list_strategies() if "dictionary" not in s]:
        try:
            cfg = m.load_strategy_config(s)
            for fd in cfg.get("factors", []):
                if fd.get("name") == name:
                    expr = fd.get("expression", "")
                    duckdb_expr = ""
                    if isinstance(expr, dict):
                        duckdb_expr = str(expr.get("duckdb", ""))
                        expr = str(expr.get("qlib", str(expr)))
                    return {"name": name, "expr": str(expr),
                            "duckdb_expr": duckdb_expr, "category": s,
                            "sub_category": fd.get("sub_category", ""),
                            "meaning": fd.get("meaning", ""),
                            "_source_file": fd.get("_source_file", "")}
        except Exception:
            pass
    return None


def _load_pool_stocks(pool_name: str) -> set:
    """从 instruments 目录加载指定股票池的股票代码集合。

    Args:
        pool_name: 股票池名称（当前生产统一使用 main_board）

    Returns:
        有效股票代码的 set（已过滤掉过期条目），如 pool 文件不存在则返回空 set
    """
    pool_file = _INSTRUMENTS_DIR / f"{pool_name}.txt"
    if not pool_file.exists():
        available = ", ".join(_AVAILABLE_POOLS)
        logger.warning(f"股票池文件不存在: {pool_name}.txt，可用池: {available}")
        return set()

    stocks = set()
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                stocks.add(parts[0])
    logger.info(f"[股票池] {pool_name}: {len(stocks)} 只股票")
    return stocks


def _meta_val(meta: dict, key: str, default=None):
    """兼容新旧 meta 格式读取辅助函数。"""
    # 新格式在 data_range 下
    dr = meta.get("data_range", {}) if meta else {}
    if key in dr and dr[key]:
        return dr[key]
    # 旧格式在顶层
    if meta and key in meta and meta[key]:
        return meta[key]
    # statistics 嵌套 (warehouse meta.json 格式)
    stats = meta.get("statistics", {}) if meta else {}
    if key in stats and stats[key]:
        return stats[key]
    # last_date 兼容：实际 key 为 end_date
    if key == "last_date":
        return dr.get("end_date", default)
    return default


def _enrich_with_extra_fields(df, start_time, end_time, logger=None):
    """加载 mkt_cap（流通市值）和 industry（申万一级行业）并合并到 df。

    这样场景压力测试（分市值、分行业板块）和控制变量对冲（双变量分组、残差因子）
    才能获取数据，否则这些评测项目会因缺数据而使用 0.5 中性分。

    Args:
        df: DataFrame with MultiIndex (instrument, datetime), 含因子列和标签列
        start_time: 起始日期
        end_time: 结束日期

    Returns:
        合并后的 DataFrame，包含 circ_mv 和 sw_l1 列（如果数据可用）
    """
    log = logger or globals().get("logger")

    if not isinstance(df.index, pd.MultiIndex):
        if log:
            log.warning("[补充字段] DataFrame 非 MultiIndex，跳过")
        return df

    instruments = df.index.get_level_values("instrument").unique().tolist()

    # ── 路径 A：Qlib 路径 ──
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.config import QLIB_DATA_DIR
        qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN, mute_warning=True)

        fields = ["$circ_mv", "$sw_l1"]
        col_names = ["circ_mv", "sw_l1"]
        extra = D.features(instruments, fields, str(start_time)[:10], str(end_time)[:10])
        if not extra.empty:
            if isinstance(extra.columns, pd.MultiIndex):
                extra.columns = [c[0] for c in extra.columns]
            keep = [f for f in fields if f in extra.columns]
            if keep:
                extra = extra[keep].copy()

                # 处理 sw_l1 编码 — Qlib $sw_l1 可能存为数值ID
                if "$sw_l1" in extra.columns:
                    mapping_path = Path(str(QLIB_DATA_DIR)) / "sw_industry_mapping.json"
                    if mapping_path.exists():
                        try:
                            import json
                            with open(mapping_path, encoding="utf-8") as f:
                                id_map = json.load(f)
                            reverse_map = {v: k for k, v in id_map.get("l1", {}).items()}
                            if reverse_map:
                                mapped = extra["$sw_l1"].map(reverse_map)
                                # 只在新映射有足够匹配率时替换（>10%），否则保留原始数值
                                match_rate = mapped.notna().sum() / max(len(mapped), 1)
                                if match_rate > 0.1:
                                    extra["$sw_l1"] = mapped
                                else:
                                    if log:
                                        log.info(f"[补充字段] 行业编码映射匹配率仅 {match_rate:.1%}，保留原始数值")
                        except Exception:
                            pass

                if df.index.names != extra.index.names:
                    extra = extra.swaplevel()
                extra = extra.reindex(df.index)
                extra.columns = col_names

                df["circ_mv"] = extra["circ_mv"]
                df["sw_l1"] = extra["sw_l1"]

                n_valid_mc = int(df["circ_mv"].notna().sum())
                n_valid_ind = int(df["sw_l1"].notna().sum())
                mc_ok = n_valid_mc > len(df) * 0.1
                if log:
                    log.info(f"[补充字段] Qlib: circ_mv={n_valid_mc:,}行有效({mc_ok}), sw_l1={n_valid_ind:,}行有效 "
                             f"(共{len(instruments)}只股票)")

                # ── 行业分类 sw_l1 处理：先用本地 SW 映射解码 ──
                from qlworks.evaluation.sw_mapping import decode_sw_series

                ind_ok = n_valid_ind > len(df) * 0.1
                is_numeric_ind = True  # 默认视作数值编码

                if ind_ok:
                    # 尝试用本地申万行业映射解码
                    decoded = decode_sw_series(df["sw_l1"], level=1)
                    n_mapped = decoded.str.startswith("未知(").sum() if len(decoded) > 0 else len(decoded)
                    mapped_ratio = 1 - n_mapped / max(len(decoded), 1)
                    if mapped_ratio > 0.5:
                        # 大部分成功解码 → 直接用映射后的名称
                        df["sw_l1"] = decoded
                        is_numeric_ind = False
                        if log:
                            log.info(f"[补充字段] SW_L1 解码: {1-n_mapped:,}/{len(decoded)} 行成功 ({mapped_ratio:.1%})")
                    elif log:
                        log.info(f"[补充字段] SW_L1 解码覆盖率不足 ({mapped_ratio:.1%})")

                # 解码失败时的回退
                if is_numeric_ind or not ind_ok:
                    # 尝试回退 $industry
                    try:
                        alt_fields = ["$industry"]
                        alt_extra = D.features(instruments, alt_fields, str(start_time)[:10], str(end_time)[:10])
                        if not alt_extra.empty:
                            if isinstance(alt_extra.columns, pd.MultiIndex):
                                alt_extra.columns = [c[0] for c in alt_extra.columns]
                            if df.index.names != alt_extra.index.names:
                                alt_extra = alt_extra.swaplevel()
                            alt_extra = alt_extra.reindex(df.index)
                            df["sw_l1"] = alt_extra[alt_fields[0]]
                            n_valid_ind = int(df["sw_l1"].notna().sum())
                            ind_ok = n_valid_ind > len(df) * 0.1
                            is_numeric_ind = False
                            if log:
                                log.info(f"[补充字段] 回退 $industry: {n_valid_ind:,}行有效")
                    except Exception:
                        pass

                # ── circ_mv 独立处理：Qlib 数据有效则直接保留 ──
                if mc_ok:
                    if log:
                        log.info(f"[补充字段] Qlib circ_mv 有效 ({n_valid_mc:,}行)，直接保留")
                else:
                    if log:
                        log.warning(f"[补充字段] Qlib circ_mv 不足 ({n_valid_mc:,}行)，后续 ClickHouse 补充")

                # ── sw_l1 和 circ_mv 都满足条件则立即返回 ──
                if mc_ok and ind_ok and not is_numeric_ind:
                    if log:
                        log.info(f"[补充字段] Qlib 全达标: circ_mv={n_valid_mc:,}, sw_l1={n_valid_ind:,} → 直接使用")
                    return df

                if log:
                    ind_note = f"sw_l1={'数值编码' if is_numeric_ind else '不足'}({n_valid_ind:,}行)"
                    log.info(f"[补充字段] Qlib circ_mv={'有效' if mc_ok else '不足'}, {ind_note} → 仅对 sw_l1 用 ClickHouse 补充")

    except ImportError:
        if log:
            log.info("[补充字段] Qlib 未安装，切换到 ClickHouse 路径...")
    except Exception as e:
        if log:
            log.warning(f"[补充字段] Qlib 路径失败: {e}，切换到 ClickHouse 路径...")

    # ── 路径 B：ClickHouse 后备（仅补充缺失的数据） ──
    try:
        from qlworks.data import QuantDataAPI
        api = QuantDataAPI()
        # 用 ClickHouse 查询 circ_mv 和行业
        ch_sql = f"""SELECT p.ts_code AS instrument, p.trade_date AS datetime,
       CAST(i.circ_mv AS DOUBLE) AS circ_mv,
       sw.l1_name AS sw_l1
FROM daily_prices p
LEFT JOIN daily_indicators i ON p.ts_code=i.ts_code AND p.trade_date=i.trade_date
LEFT JOIN sw_industry_members sw ON p.ts_code=sw.ts_code
WHERE p.trade_date>='{start_time}' AND p.trade_date<='{end_time}'
ORDER BY p.ts_code, p.trade_date"""
        raw = api.query(ch_sql)
        if raw is not None and not raw.empty:
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw = raw.drop_duplicates(subset=["instrument", "datetime"])
            raw = raw.set_index(["instrument", "datetime"])
            # 按 df 的索引对齐
            raw = raw.reindex(df.index)

            # sw_l1：Qlib 不足或数值编码时用 ClickHouse 覆盖
            ch_has_sw = "sw_l1" in raw.columns and raw["sw_l1"].notna().any()
            if ch_has_sw:
                df["sw_l1"] = raw["sw_l1"]
                if log:
                    log.info(f"[补充字段] ClickHouse: sw_l1覆盖完成 (非空{raw['sw_l1'].notna().sum():,}行)")

            # circ_mv：仅在 Qlib 数据不足时用 ClickHouse 覆盖
            qlib_mc_ok = False
            try:
                qlib_mc_ok = mc_ok  # Qlib 路径已执行，使用其判断结果
            except NameError:
                qlib_mc_ok = False  # Qlib 路径未执行（异常/未安装）
            if not qlib_mc_ok and "circ_mv" in raw.columns:
                df["circ_mv"] = raw["circ_mv"]
                if log:
                    log.info(f"[补充字段] ClickHouse: circ_mv覆盖完成 (非空{raw['circ_mv'].notna().sum():,}行)")
            elif qlib_mc_ok and log:
                log.info(f"[补充字段] 保留 Qlib circ_mv ({n_valid_mc:,}行)，不做 ClickHouse 覆盖")

            n_valid_mc = int(df["circ_mv"].notna().sum())
            n_valid_ind = int(df["sw_l1"].notna().sum())
            if log:
                log.info(f"[补充字段] ClickHouse: circ_mv={n_valid_mc:,}行有效, sw_l1={n_valid_ind:,}行有效 "
                         f"(共{len(instruments)}只股票)")
    except Exception as e:
        if log:
            log.warning(f"[补充字段] ClickHouse 加载失败: {e}")

    return df


def show_status(store):
    """显示仓库中所有因子的状态。"""
    factors = store.list_warehouse_factors()
    print(f"\n  仓库中有 {len(factors)} 个因子:")
    print(f"  {'因子名':30s} {'年份':20s} {'总行数':>10s}  {'最后日期':14s}  {'等级':12s}")
    print(f"  {'-'*30} {'-'*20} {'-'*10}  {'-'*14}  {'-'*12}")
    for name in factors:
        meta = store.get_warehouse_meta(name)
        eval_meta = store.get_evaluated_meta(name)
        tier = eval_meta.get("tier", "未评测") if eval_meta else "未评测"
        if meta:
            years_list = _meta_val(meta, "years", [])
            years = ",".join(str(y) for y in (years_list or []))
            total = _meta_val(meta, "total_records", 0) or _meta_val(meta, "total_rows", 0)
            last = _meta_val(meta, "last_date", "")
            print(f"  {name:30s} {years:20s} {total:>10,}  {last:14s}  {tier:12s}")
        else:
            print(f"  {name:30s} {'无元数据':20s}")
    print()


def _read_factor_list(path: str) -> list:
    """读取粗筛因子名单文件（txt 每行一个因子名；CSV 优先取 factor_name 列，否则第一列）。"""
    p = Path(path)
    if not p.exists():
        logger.error(f"因子名单文件不存在: {path}")
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    names = []
    if p.suffix.lower() == ".csv":
        import csv
        import io
        rows = list(csv.reader(io.StringIO(text)))
        if rows and rows[0] and "factor_name" in rows[0]:
            idx = rows[0].index("factor_name")
        else:
            idx = 0
        for r in rows[1:]:
            if r and r[idx].strip():
                names.append(r[idx].strip())
    else:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "//")):
                names.append(line)
    return names


def main():
    parser = argparse.ArgumentParser(description="单因子评测系统（仓库优先 v3.0）")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--factor", "-f", help="指定单因子名称")
    g.add_argument("--category", "-c", help="指定分类 YAML 文件")
    g.add_argument("--all", "-a", action="store_true", help="评测仓库中所有因子")
    g.add_argument("--status", "-s", action="store_true", help="查看仓库状态")
    g.add_argument("--demo", action="store_true", help="演示模式")
    parser.add_argument("--factor-list", type=str, default=None,
                        help="粗筛输出名单（screen_factors.py 生成的 screened_factors_*.txt，每行一个因子名），评测该名单内因子")
    parser.add_argument("--pool", default=INSTRUMENTS,
                        help=f"股票池，可选: {', '.join(_AVAILABLE_POOLS)} (default: main_board，与训练端 train_tree-doubao.py 对齐)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--recompute", action="store_true",
                        help="强制从 ClickHouse 实时计算并写入仓库（跳过仓库缓存）")
    parser.add_argument("--walk-forward", action="store_true", help="启用 Walk-Forward 滚动外推验证")
    parser.add_argument("--dry-run", action="store_true", help="仅列出因子，不评测")
    # ── [Phase 2 入库决策]（admit 合并） ──
    parser.add_argument("--admit", action="store_true",
                        help="评测完成后立即执行三关准入并写候选池（评测即入库，机构范式）")
    parser.add_argument("--admit-only", action="store_true",
                        help="只对已评测报告执行准入决策（扫描 reports 目录，替代旧 admit_to_multifactor.py）")
    parser.add_argument("--tier", type=str, default=None,
                        choices=["core", "satellite", "archive"],
                        help="因子档位（默认从评测结果/报告路径自动推断）")
    args = parser.parse_args()

    if not any([args.factor, args.category, args.all, args.demo, args.status, args.factor_list, args.admit_only]):
        parser.print_help()
        return

    store = FactorStore()
    config = replace(
        DEFAULT_CONFIG,
        instruments=args.pool,
        start_time=args.start,
        end_time=args.end,
    )

    # ── 状态模式 ──
    if args.status:
        show_status(store)
        return

    # ── [Phase 2] --admit-only：只对已评测报告执行准入（替代旧 admit --build-all / --factor） ──
    if args.admit_only:
        win = _admission_window()
        label_frame = _load_admit_label_frame(win["corr_start"], win["corr_end"])
        pool = load_candidate_pool()
        existing = list(pool.get("factors", []))
        existing_names = {f.get("name") for f in existing}

        target = args.factor or args.factor_list
        if target:
            # 单因子/名单模式：从因子库解析定义
            m = FactorLibraryManager()
            names = [target] if args.factor else _read_factor_list(args.factor_list)
            candidates = []
            for name in names:
                info = _find_factor(m, name)
                if info:
                    info["_tier"] = args.tier or "satellite"
                    candidates.append(info)
        else:
            # 全量模式：扫描 reports/{core,satellite}/ 已评测报告
            candidates = []
            for tier in ("core", "satellite"):
                tier_dir = _REPORTS_DIR / tier
                if not tier_dir.exists():
                    continue
                for rf in tier_dir.iterdir():
                    if rf.suffix == ".html":
                        name = rf.stem.split("_")[0]
                        info = _find_factor(FactorLibraryManager(), name)
                        if info:
                            info["_tier"] = args.tier or tier
                            candidates.append(info)

        print(f"[admit-only] 待准入因子: {len(candidates)} 个 | 已有池: {len(existing)} 个")
        for cand in candidates:
            name = cand["name"]
            if name in existing_names:
                print(f"  [跳过] {name} 已在候选池中")
                continue
            entry = None
            ic_mean = icir_nw = None
            # 从 registry.json 取该因子评测指标（--admit-only 兼容路径）
            # 注意：registry.json factors 为 list 结构且通常无 ic_mean 字段——
            # 评测指标由 run_eval 评测时直接产生，建议用 --admit（评测即入库）。
            reg = {}
            reg_path = _REGISTRY_DIR / "registry.json"
            if reg_path.exists():
                try:
                    with open(reg_path, "r", encoding="utf-8") as f:
                        reg = _json.load(f)
                except Exception:
                    reg = {}
            reg_factors = reg.get("factors") or []
            rentry = None
            if isinstance(reg_factors, dict):
                rentry = reg_factors.get(name) or {}
            elif isinstance(reg_factors, list):
                rentry = next((e for e in reg_factors if e.get("name") == name), {}) or {}
            ic_mean = rentry.get("ic_mean")
            icir_nw = rentry.get("icir_nw") or rentry.get("icir")
            if ic_mean is None:
                print(f"  [跳过] {name}: registry 无 IC 指标——请用 `run_eval.py --factor {name} --admit` 评测即入库（机构范式主路径）")
                continue
            run_admit_decision(name, cand, cand.get("_tier", "satellite"),
                               ic_mean, icir_nw, label_frame)
        return

    # ── 收集因子 ──
    factors = _collect_factors(args)

    if args.dry_run:
        print(f"共 {len(factors)} 个因子:")
        for f in factors:
            print(f"  {f['name']:30s} {f['expr'][:60]}")
        return

    # ── 实例化评测器 ──
    evaluator = FactorEvaluator(config)

    # ── 执行评测 ──
    for i, f in enumerate(factors, 1):
        name = f["name"]
        expr = f["expr"]
        print(f"\n[{i}/{len(factors)}] {name}")

        try:
            # [Citadel Alpha Lab] 仓库优先：先确保数据在仓库中
            if args.recompute or not store.get_warehouse_meta(name):
                duckdb_expr = f.get("duckdb_expr", "")
                logger.info(f"{name} 仓库无数据或强制重算，从 ClickHouse 计算...")
                store.compute_to_warehouse(name, expr, args.start, args.end,
                                           duckdb_expr=duckdb_expr)
            else:
                warehouse_meta = store.get_warehouse_meta(name)
                wh_start = _meta_val(warehouse_meta, "first_date", "")
                wh_end = _meta_val(warehouse_meta, "last_date", "")
                total_rows_val = _meta_val(warehouse_meta, "total_records", 0) or _meta_val(warehouse_meta, "total_rows", 0)
                logger.info(f"{name} 从仓库读取 ({wh_start} ~ {wh_end}, {total_rows_val:,} 行)")

            # 从仓库加载数据（仓库列名为 value，需重命名为因子名）
            df = store.load_from_warehouse(name, args.start, args.end)
            if df is None or df.empty:
                logger.warning(f"{name} 无数据，跳过")
                continue
            df = df.rename(columns={"value": name})

            # ── 按 --pool 过滤股票池 ──
            pool_stocks = _load_pool_stocks(args.pool)
            if pool_stocks:
                n_before = len(df)
                df = df[df.index.get_level_values("instrument").isin(pool_stocks)]
                logger.info(f"[股票池] {args.pool} 过滤: {n_before:,} → {len(df):,} 行 ({df.index.get_level_values('instrument').nunique()} 只)")
                if df.empty:
                    logger.warning(f"{name}: 股票池 {args.pool} 过滤后无数据，跳过")
                    continue

            # 合并标签
            df_label = evaluator._load_labels(args.start, args.end)
            if df_label is not None:
                df = df.join(df_label, how="inner")

            # [数据质量] 标签可交易性过滤（与粗筛/精选/训练端一致）：
            # 涨跌停/一字板/持仓期停牌样本标签置 NaN，避免评测统计来自不可交易样本
            if FILTER_LIMIT_UPDOWN:
                from qlworks.factors.filter_utils import filter_untradeable_labels
                label_col = evaluator.config.label_name
                if label_col in df.columns:
                    inst_list = df.index.get_level_values("instrument").unique().tolist()
                    label_clean = filter_untradeable_labels(
                        df[[label_col]].copy(), inst_list,
                        str(args.start)[:10], str(args.end)[:10],
                    )
                    if label_clean is not None:
                        df[label_col] = label_clean[label_col]

            # [数据质量] 补充 volume / change_pct 列：
            # 使评测端 filter_ashare_constraints（涨跌停/停牌过滤）真正生效——
            # 原实现因 df 缺这两列而静默失效（if 条件恒不满足，一条数据都没过滤）
            if FILTER_SUSPENDED or evaluator.config.limit_up_pct is not None:
                try:
                    import qlib as _q
                    from qlib.config import REG_CN as _REG_CN
                    from qlib.data import D as _D
                    if not evaluator._qinited:
                        _q.init(provider_uri=str(QLIB_DATA_DIR), region=_REG_CN)
                        evaluator._qinited = True
                    inst_list = df.index.get_level_values("instrument").unique().tolist()
                    _extra = _D.features(
                        inst_list,
                        ["$volume", "Ref($close, 1) / $close - 1"],
                        str(args.start)[:10], str(args.end)[:10],
                    )
                    if _extra is not None and not _extra.empty:
                        if isinstance(_extra.columns, pd.MultiIndex):
                            _extra.columns = _extra.columns.droplevel(1)
                        _extra = _extra.rename(columns={
                            "$volume": "volume",
                            "Ref($close, 1) / $close - 1": "change_pct",
                        })
                        if df.index.names != _extra.index.names:
                            _extra = _extra.swaplevel()
                        df["volume"] = _extra["volume"]
                        df["change_pct"] = _extra["change_pct"]
                except Exception as _e:
                    logger.warning(f"[数据质量] volume/change_pct 加载失败（{_e}），涨跌停/停牌过滤保持不生效")

            # 补充 mkt_cap + industry（场景压力测试/控制变量对冲必需）
            df = _enrich_with_extra_fields(df, args.start, args.end, logger=logger)

            # [AQR] 完整评测流水线
            if args.walk_forward:
                result = evaluator.walk_forward_evaluate(
                    name, df.reset_index(),
                    config_override={"category": f.get("category", "satellite")},
                )
            else:
                result = evaluator.evaluate(
                    name, df.reset_index(),
                    category=f.get("category", "satellite"),
                )
            q = result["qual_result"]
            icon = {"core": "[OK]", "satellite": "[~]", "archive": "[ ]"}.get(q["tier"], "[ ]")
            # 读取最新 meta 信息（兼容新旧格式）
            cur_meta = store.get_warehouse_meta(name) or {}
            total_str = _meta_val(cur_meta, "total_records", 0) or _meta_val(cur_meta, "total_rows", 0)
            if total_str:
                total_str = f"{total_str:,}"
            last_dt = _meta_val(cur_meta, "last_date", "?")
            last_rows = f"({total_str}行, 至{last_dt})" if not args.recompute else ""
            print(f"  {icon} 等级={q['tier']:10s} IC={result['ic_stats']['ic_mean']:.4f} ICIR={result['ic_stats']['icir']:.2f} 评分={q['composite_score']:.1f} {last_rows}")

            # [Phase 2 入库决策]（--admit：评测即入库，机构范式——WW 模拟器→提交校验一气呵成）
            if args.admit:
                tier = args.tier or q["tier"]
                if tier not in ("core", "satellite"):
                    print(f"  [admit] {name}: 评测档位={tier}（非 core/satellite），不入候选池——先调因子再评测（机构范式：reject 不进 Alpha Book）")
                else:
                    win = _admission_window()
                    label_frame = _load_admit_label_frame(win["corr_start"], win["corr_end"])
                    ic_mean = result["ic_stats"].get("ic_mean")
                    icir_nw = result["ic_stats"].get("icir_nw") or result["ic_stats"].get("icir")
                    run_admit_decision(name, f, tier, ic_mean, icir_nw, label_frame)

        except Exception as e:
            print(f"  [X] {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n完成! 报告: {config.report_dir}")


if __name__ == "__main__":
    main()
