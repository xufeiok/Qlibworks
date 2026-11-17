# -*- coding: utf-8 -*-
"""
factor_stats.py — 单因子统计核心（机构单一事实源，2026-10-12 抽取）

统一全链路的单因子统计口径，消除"筛选端 / 深评端 / 准入端各算一遍"的分裂：
  - 逐日截面 Spearman IC
  - Newey-West 自相关修正 ICIR
  - 因子间 RankIC 相关矩阵
  - 三关准入（低相关 / 增量 ICIR 边际贡献 / IC 方向一致性）
  - 候选池（Alpha Book v2）读写与准入编排

所有函数无全局副作用：路径 / 阈值 / 准入窗口均由调用方传入（机构"薄脚本 + 厚核心库"）。
下游消费方：factor_screen.py（唯一评测入口脚本）、（历史）run_eval.py。
候选池格式 v2 逐字节保持（factors[].status），train_tree-doubao.py 只读 status，训练端零改动。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("factor_stats")


# ══════════════════════════════════════════════════════════════════════
# 因子家族口径（全链路唯一事实源：筛查端 / 候选池 / 训练端类别名称前后统一）
# ══════════════════════════════════════════════════════════════════════

# 五大家族；其余归入 other
FACTOR_FAMILIES = ("quality", "value", "momentum_price", "sentiment", "risk")

# 关键词用于把"家族名 / 策略(文件 stem)名 / YAML 中文子类"统一归一化到家族口径
_FAMILY_KEYWORDS = (
    ("momentum_price", ("momentum", "price", "量价", "动量")),
    ("sentiment", ("sentiment", "情绪", "资金")),
    ("quality", ("quality", "质量", "成长", "财务", "基本面")),
    ("value", ("value", "价值", "估值")),
    ("risk", ("risk", "风险", "波动")),
)


def normalize_factor_family(category) -> str:
    """将任意类别标识归一化为五大家族口径（其余归入 other）。

    输入可能是家族名（momentum_price）、策略/文件 stem（momentum_price_factors）
    或 YAML 中文子类（量价形态 / 动量与反转 / 质量…），统一映射为同一家族名，
    保证筛查端 → 候选池 → 训练端类别名称前后一致。
    """
    c = str(category or "").lower()
    for fam, keys in _FAMILY_KEYWORDS:
        if any(k in c for k in keys):
            return fam
    return "other"


# ══════════════════════════════════════════════════════════════════════
# 准入窗口（严格早于 ADMISSION_CUTOFF_DATE，避免测试期泄漏）
# ══════════════════════════════════════════════════════════════════════

def build_admission_window(cutoff_date: str, years: int) -> dict:
    """准入窗口 = 截止日前 N 年，按年切片（机构：样本外评测切分先于准入检验）。"""
    cutoff = pd.Timestamp(cutoff_date)
    end_dt = cutoff - pd.Timedelta(days=1)
    start_dt = end_dt - pd.DateOffset(years=years) + pd.Timedelta(days=1)
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


# ══════════════════════════════════════════════════════════════════════
# 标签与因子序列加载（warehouse 单文件 data.parquet，instrument 统一小写）
# ══════════════════════════════════════════════════════════════════════

def load_label_frame(label_expr: str, label_name: str, instruments: str,
                     qlib_data_dir: Path, start: str, end: str) -> Optional[pd.Series]:
    """准入窗口标签（DK 标签 + 可交易过滤）。qlib.init 幂等。"""
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.factors.filter_utils import filter_untradeable_labels

        qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN)
        ins_file = qlib_data_dir / "instruments" / f"{instruments}.txt"
        if not ins_file.exists():
            return None
        insts = []
        with open(ins_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    insts.append(parts[0])
        if not insts:
            return None
        raw = D.features(insts, [label_expr], start, end)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw = raw.rename(columns={raw.columns[0]: label_name})
        raw = filter_untradeable_labels(raw, insts, start, end)
        if raw is None or raw.empty:
            return None
        flat = raw.reset_index()
        flat["instrument"] = flat["instrument"].str.lower()
        return flat.set_index(["instrument", "datetime"]).sort_index()[label_name]
    except Exception as e:
        logger.warning(f"[factor_stats] 标签加载失败（{e}），增量检验回退近似路径")
        return None


def load_warehouse_series(warehouse_root: Path, name: str, start: str, end: str) -> Optional[pd.Series]:
    """从 warehouse 读取因子在窗口内序列（单文件 data.parquet，MultiIndex，instrument 小写化）。"""
    pf = warehouse_root / name / "data.parquet"
    if not pf.exists():
        return None
    try:
        df = pd.read_parquet(pf)
    except Exception as e:
        logger.warning(f"[factor_stats] {name} data.parquet 读取失败: {e}")
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
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return None
    return series


def slice_period(series: pd.Series, start: str, end: str) -> pd.Series:
    dts = series.index.get_level_values("datetime")
    return series[(dts >= start) & (dts <= end)]


# ══════════════════════════════════════════════════════════════════════
# 逐日 IC / NW-ICIR（唯一口径：全链路统一）
# ══════════════════════════════════════════════════════════════════════

def daily_ic_series(factor_s: pd.Series, label_s: pd.Series, min_cross: int = 20) -> pd.Series:
    """逐日截面 Spearman IC（无区分度截面记 0；与筛选端 compute_daily_ic_frame 同源口径）。"""
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


def nw_icir(ic_series: pd.Series, annual_factor: float = 252.0) -> float:
    """Newey-West 自相关修正 ICIR（5 日重叠标签 → 原生 mean/std 虚高 √5 量级，必须修正）。"""
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
    nw_std_series = nw_se * np.sqrt(n_obs) if n_obs > 0 else std
    if nw_std_series <= 1e-12:
        return 0.0
    return mean / nw_std_series * np.sqrt(annual_factor)


# ══════════════════════════════════════════════════════════════════════
# 相关矩阵 / 三关准入（机构：低相关 + 增量贡献 + 方向一致）
# ══════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(warehouse_root: Path, factor_names: list[str],
                               start: str, end: str) -> pd.DataFrame:
    """两两 RankIC 相关矩阵（单文件 data.parquet 结构，按日截面均值）。"""
    factor_data = {}
    missing = []
    for name in factor_names:
        pf = warehouse_root / name / "data.parquet"
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
        logger.warning(f"[factor_stats] warehouse 无 {start[:4]}~{end[:4]} 数据: {missing}")
        if not factor_data:
            return pd.DataFrame()

    merged = pd.concat(factor_data.values(), axis=1, keys=factor_data.keys())
    merged.columns = merged.columns.get_level_values(0)
    merged = merged.replace([np.inf, -np.inf], np.nan)

    daily_corrs = []
    for _, group in merged.groupby(level="datetime"):
        valid = group.dropna(how="all")
        if len(valid) < 50:
            continue
        ranked = valid.rank(pct=True)
        corr = ranked.corr(method="spearman")
        daily_corrs.append(corr)
    if not daily_corrs:
        return pd.DataFrame()
    corr_sum = sum(c.reindex(merged.columns, axis=0).reindex(merged.columns, axis=1).fillna(0)
                   for c in daily_corrs)
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
                                label_frame: Optional[pd.Series],
                                corr_matrix: Optional[pd.DataFrame],
                                start: str, end: str,
                                sub_periods: list[tuple[str, str, str]],
                                improve_min: float = 0.0,
                                warehouse_root: Optional[Path] = None) -> tuple[bool, str]:
    """第二关：增量 ICIR 边际贡献（等权组合信号对比，NW 修正，分年度）。"""
    if not existing_factors:
        return True, "候选池为空，无需边际检验"
    existing_names = [f["name"] for f in existing_factors]
    if label_frame is not None and warehouse_root is not None:
        factor_frames = {}
        missing = []
        for name in existing_names + [new_factor]:
            series = load_warehouse_series(warehouse_root, name, start, end)
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
            for label, ys, ye in sub_periods:
                ic0 = daily_ic_series(slice_period(s0, ys, ye), slice_period(label_frame, ys, ye))
                ic1 = daily_ic_series(slice_period(s1, ys, ye), slice_period(label_frame, ys, ye))
                if len(ic0) < 10 or len(ic1) < 10:
                    continue
                icir0 = nw_icir(ic0)
                icir1 = nw_icir(ic1)
                deltas.append(icir1 - icir0)
                details.append(f"{label} ICIR_NW {icir0:.3f}→{icir1:.3f} (Δ{icir1 - icir0:+.3f})")
            if deltas:
                avg_delta = float(np.mean(deltas))
                if avg_delta < improve_min:
                    return False, f"增量 ICIR 边际贡献为负: {avg_delta:+.3f} ({'; '.join(details)})"
                return True, f"增量 ICIR 边际贡献: {avg_delta:+.3f} ({'; '.join(details)})"
        logger.warning(f"[factor_stats] {new_factor}: 增量 ICIR 检验数据不足，回退平均相关近似")
    if corr_matrix is None or new_factor not in corr_matrix.index:
        return True, "新因子不在相关矩阵中，默认通过"
    vals = []
    for ename in existing_names:
        if ename in corr_matrix.columns:
            v = abs(corr_matrix.loc[new_factor, ename])
            vals.append(v)
    if not vals:
        return True, "与已有池因子无交集"
    avg_corr = float(np.mean(vals))
    if avg_corr > 0.40:
        return False, f"冗余度偏高: 与已有池平均相关 {avg_corr:.2f} > 0.40（标签缺失近似）"
    return True, f"通过（与已有池平均相关 {avg_corr:.2f}，标签缺失近似）"


def check_direction_consistency(new_factor: str, label_frame: Optional[pd.Series],
                                start: str, end: str,
                                sub_periods: list[tuple[str, str, str]],
                                warehouse_root: Optional[Path] = None) -> tuple[bool, str]:
    """第三关：分年度 IC 方向一致性（子时段 IC 同号）。"""
    if warehouse_root is None:
        return True, "无 warehouse 根路径，默认通过"
    series = load_warehouse_series(warehouse_root, new_factor, start, end)
    if series is None:
        return True, "因子无 warehouse 数据，默认通过"
    if len(series) < 1000:
        return True, "数据量不足"
    if label_frame is not None:
        ic_means = []
        details = []
        for label, ys, ye in sub_periods:
            ic_series = daily_ic_series(slice_period(series, ys, ye), slice_period(label_frame, ys, ye))
            if ic_series.empty:
                continue
            ic_means.append(ic_series.mean())
            details.append(f"{label} IC={ic_series.mean():.4f}")
        if len(ic_means) >= 2:
            if all(v > 0 for v in ic_means) or all(v < 0 for v in ic_means):
                return True, "IC 方向一致: " + "; ".join(details)
            return False, "IC 方向不一致: " + "; ".join(details)
        logger.warning(f"[factor_stats] {new_factor}: 标签子时段不足，回退因子值符号校验")
    signs = []
    details = []
    for label, ys, ye in sub_periods:
        subset = slice_period(series, ys, ye)
        if len(subset) < 100:
            continue
        mean_val = subset.mean()
        signs.append(mean_val)
        details.append(f"{label}={mean_val:.4f}")
    if not signs:
        return True, "子时段无有效数据"
    direction = np.sign(signs)
    if len(set(direction)) > 1:
        return False, f"方向不一致: {'; '.join(details)}"
    return True, f"方向一致: {'; '.join(details)}"


def admit_factor(new_factor_name: str, new_factor_info: dict,
                 existing_factors: list[dict], corr_matrix: pd.DataFrame,
                 label_frame: Optional[pd.Series],
                 thresholds: dict,
                 sub_periods: list[tuple[str, str, str]],
                 corr_start: str, corr_end: str,
                 warehouse_root: Optional[Path] = None) -> dict:
    """三关准入检验（机构：WW self-corr<0.7 + 增量贡献 + 方向一致）。"""
    result = {
        "name": new_factor_name,
        "tier": new_factor_info.get("_tier", "satellite"),
        "category": normalize_factor_family(new_factor_info.get("category", "")),
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
        threshold=thresholds.get("max_correlation_existing", 0.70))
    result["corr_check"]["passed"] = corr_passed
    result["corr_check"]["high_corr_factors"] = high_corr
    if not corr_passed:
        result["reasons"].append(f"相关性检验未通过: 与 {', '.join(high_corr)} 高度相关")

    marginal_passed, marginal_detail = check_marginal_contribution(
        new_factor_name, existing_factors, label_frame, corr_matrix,
        start=corr_start, end=corr_end, sub_periods=sub_periods,
        improve_min=thresholds.get("icir_improve_min", 0.0),
        warehouse_root=warehouse_root)
    result["marginal_check"]["passed"] = marginal_passed
    result["marginal_check"]["detail"] = marginal_detail
    if not marginal_passed:
        result["reasons"].append(f"边际贡献检验未通过: {marginal_detail}")

    dir_passed, dir_detail = check_direction_consistency(
        new_factor_name, label_frame, start=corr_start, end=corr_end,
        sub_periods=sub_periods, warehouse_root=warehouse_root)
    result["direction_check"]["passed"] = dir_passed
    result["direction_check"]["detail"] = dir_detail
    if not dir_passed:
        result["reasons"].append(f"方向一致性未通过: {dir_detail}")

    result["admitted"] = corr_passed and marginal_passed and dir_passed
    return result


# ══════════════════════════════════════════════════════════════════════
# 候选池（Alpha Book v2）读写 / 准入编排
# ══════════════════════════════════════════════════════════════════════

def load_candidate_pool(pool_path: Path, thresholds: dict) -> dict:
    """读取候选池；v1 旧格式自动迁移为 v2。"""
    if pool_path.exists():
        with open(pool_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "factors" not in data:
            legacy = data.get("candidate_pool") or []
            return {
                "_meta": {"version": "2.0", "set_version": "v1",
                          "description": "多因子准入候选池（Alpha Book）— 由唯一评测脚本三关检验写入，下游 train_tree-doubao.py 从此读取因子名单",
                          "pipeline_note": "v1 旧格式自动迁移，遗留候选仅登记 rejected",
                          "updated_at": pd.Timestamp.now().isoformat()[:19],
                          "admit_thresholds": thresholds},
                "factors": [], "rejected": legacy,
                "stats": {"total_candidates": len(legacy), "admitted": 0,
                          "rejected_corr": 0, "rejected_marginal": 0, "rejected_direction": 0},
            }
        data.setdefault("watchlist", [])
        data.setdefault("stats", {})
        return data
    return {
        "_meta": {"version": "2.0", "set_version": "v1",
                  "description": "多因子准入候选池（Alpha Book）— 由唯一评测脚本三关检验写入，下游 train_tree-doubao.py 从此读取因子名单",
                  "updated_at": pd.Timestamp.now().isoformat()[:19],
                  "admit_thresholds": thresholds},
        "factors": [], "rejected": [], "watchlist": [],
        "stats": {"total_candidates": 0, "admitted": 0,
                  "rejected_corr": 0, "rejected_marginal": 0, "rejected_direction": 0,
                  "watchlisted": 0},
    }


def save_candidate_pool(pool: dict, pool_path: Path):
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    logger.info(f"[factor_stats] 候选池已更新: {pool_path}（factors={len(pool['factors'])}, watchlist={len(pool.get('watchlist', []))}, rejected={len(pool['rejected'])}）")


def bump_set_version(pool: dict) -> str:
    meta = pool.setdefault("_meta", {})
    cur = meta.get("set_version", "v1")
    try:
        n = int(str(cur).lstrip("v")) + 1
    except (ValueError, TypeError):
        n = 2
    meta["set_version"] = f"v{n}"
    return meta["set_version"]


def admit_entry(factor_name: str, factor_info: dict, result: dict, tier: str,
                ic_mean: float, icir_nw: float) -> Optional[dict]:
    """构造候选池 Alpha Book 标准条目（direction 由真实 IC 符号推导）。"""
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
        "category": normalize_factor_family(factor_info.get("category", "")),
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


def run_admit_decision(pool_path: Path, thresholds: dict,
                       warehouse_root: Path,
                       factor_name: str, factor_info: dict, tier: str,
                       ic_mean: float, icir_nw: float,
                       label_frame: Optional[pd.Series] = None) -> bool:
    """单因子评测完成后立即执行准入决策并写候选池。返回是否 admitted。"""
    cutoff = thresholds.get("admission_cutoff_date")
    years = thresholds.get("admission_window_years", 3)
    if not cutoff:
        raise ValueError("[factor_stats] thresholds 缺少 admission_cutoff_date")
    win = build_admission_window(cutoff, years)
    sub_periods = win["sub_periods"]
    corr_start, corr_end = win["corr_start"], win["corr_end"]

    pool = load_candidate_pool(pool_path, thresholds)
    existing = list(pool.get("factors", []))
    if any(f.get("name") == factor_name for f in existing):
        print(f"  [admit] {factor_name} 已在候选池中，跳过")
        return True
    corr_matrix = compute_correlation_matrix(
        warehouse_root,
        [factor_name] + [f["name"] for f in existing],
        corr_start, corr_end)
    result = admit_factor(factor_name, factor_info, existing, corr_matrix,
                          label_frame, thresholds, sub_periods,
                          corr_start, corr_end, warehouse_root)
    print(f"   [准入] 相关:{'✓' if result['corr_check']['passed'] else '✗'} "
          f"边际:{'✓' if result['marginal_check']['passed'] else '✗'} "
          f"方向:{'✓' if result['direction_check']['passed'] else '✗'}")
    if result["admitted"]:
        entry = admit_entry(factor_name, factor_info, result, tier, ic_mean, icir_nw)
        if entry is None:
            return False
        pool["factors"].append(entry)
        print(f"   [准入] {factor_name} (tier={tier}, direction={entry['direction']}, icir={entry['latest_icir']:.2f})")
    else:
        # 观察池分流：相关✓ + 方向✓ + 仅边际✗ + 信号本身达 satellite/core（≥70 分）
        # → 进 watchlist 替补池（池内同方向因子衰退时可重启用），而非冷宫 rejected
        corr_ok = result['corr_check']['passed']
        dir_ok = result['direction_check']['passed']
        marg_ok = result['marginal_check']['passed']
        signal_ok = tier in ("core", "satellite")
        if corr_ok and dir_ok and not marg_ok and signal_ok:
            result["watch_reason"] = "信号有效但与池内共线（边际增量为负），池内同方向因子衰退时可替补"
            result["added_to_watchlist_at"] = pd.Timestamp.now().isoformat()[:19]
            pool.setdefault("watchlist", []).append(result)
            print(f"   [观察池] {factor_name} (tier={tier}) 信号有效但边际共线 → watchlist 替补池")
        else:
            pool["rejected"].append(result)
            print(f"   [拒绝] {factor_name} 原因: {'; '.join(result['reasons'])}")
    # [类别归组排序] 候选池按类别分组，同类别内按因子有效性 |ICIR_NW| 降序，
    # 便于下游（train_tree-doubao.py）按类别读取并施加类别配额约束。
    pool["factors"].sort(
        key=lambda e: (str(e.get("category", "")),
                       -abs(float(e.get("latest_icir", 0) or 0))))
    pool["_meta"]["updated_at"] = pd.Timestamp.now().isoformat()[:19]
    bump_set_version(pool)
    pool["stats"] = {
        "total_candidates": len(pool["factors"]) + len(pool["rejected"]) + len(pool.get("watchlist", [])),
        "admitted": len(pool["factors"]),
        "rejected_corr": sum(1 for r in pool["rejected"] if not r.get("corr_check", {}).get("passed", False)),
        "rejected_marginal": sum(1 for r in pool["rejected"] if not r.get("marginal_check", {}).get("passed", False)),
        "rejected_direction": sum(1 for r in pool["rejected"] if not r.get("direction_check", {}).get("passed", False)),
        "watchlisted": len(pool.get("watchlist", [])),
    }
    save_candidate_pool(pool, pool_path)
    return result["admitted"]
