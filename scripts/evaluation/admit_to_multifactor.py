"""
admit_to_multifactor.py — 多因子准入编排层（候选池唯一写入通道，P0-1）

功能概述：
  将单因子评测结果（分档、ICIR、相关性）转化为多因子组合候选池。
  核心准入标准不再只是"低相关"，而是"低相关 + 有边际贡献 + 方向一致"。

输入：
  - factor_library/*.yaml                           因子定义（全量候选）
  - qualified_factors/{core,satellite,archive}      评测分档（仅 core/satellite 进入候选）
  - registry/candidate_pool.json                    已有候选池（逐步累加，Alpha Book v2）

输出：
  - registry/candidate_pool.json                    更新后的候选池（admitted + rejected 明细）
    条目含 direction / eval_date / tier_history / rolling_ic（供训练端 Alpha Book 消费）

[单一准入通道]
  候选池仅由此脚本的三关检验写入；evaluate() 只写 registry + qualified_factors 分档。

用法：
  # 首次构建候选池（扫描所有 core/satellite 因子）
  python scripts/evaluation/admit_to_multifactor.py --build-all

  # 单因子准入检验（评测完成后调用）
  python scripts/evaluation/admit_to_multifactor.py --factor STR_20d --tier satellite

工作流（Pipeline）：
  1. 扫描 qualified_factors/{core,satellite} 获取待准入因子
  2. 读取 registry/candidate_pool.json 已有池
  3. 计算新因子与已有池的滚动 RankIC 相关矩阵
  4. 残差独立性检验（新因子 vs 已有池回归取残差，残差 IC 是否显著）
  5. 增量 Walk-Forward ICIR 边际贡献检验
  6. 最近 3 年 IC 方向一致性校验
  7. 更新 candidate_pool.json（admitted / rejected 分列，含 direction / set_version）
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── 路径常量 ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTOR_LIBRARY_DIR = PROJECT_ROOT / "factor_data" / "factor_library"
QUALIFIED_DIR = PROJECT_ROOT / "factor_data" / "qualified_factors"
REGISTRY_DIR = PROJECT_ROOT / "factor_data" / "registry"
REPORTS_DIR = PROJECT_ROOT / "factor_data" / "reports"
CANDIDATE_POOL_PATH = REGISTRY_DIR / "candidate_pool.json"

# ── 评测报告检测 ──
_TIER_DIRS = ["core", "satellite", "archive"]

# ── 候选池读写统一走 CandidatePool 类（单一通道） ──
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from qlworks.evaluation.candidate_pool import CandidatePool  # noqa: E402
from qlworks.pipeline_config import (  # noqa: E402  单一事实源共享配置
    LABEL_EXPR, LABEL_NAME, INSTRUMENTS,
    CORR_THRESHOLD, ICIR_IMPROVE_MIN,
)


def _find_evaluation_report(factor_name: str) -> str | None:
    """
    在 reports/{core,satellite,archive}/ 中查找因子的评测报告。
    报告格式: {factor_name}_{start}_{end}_{timestamp}.html
    返回报告所在的 tier（core/satellite/archive），找不到返回 None。
    """
    for tier in _TIER_DIRS:
        report_dir = REPORTS_DIR / tier
        if not report_dir.exists():
            continue
        for f in report_dir.iterdir():
            if f.suffix == ".html" and f.stem.startswith(factor_name):
                return tier
    return None


# ── 准入阈值（与 factor_def.py 保持一致；相关上限引用 pipeline_config 单一事实源） ──
ADMIT_THRESHOLDS = {
    "max_correlation_existing": CORR_THRESHOLD,  # 与已有池因子相关性上限（pipeline_config）
    "max_correlation_barra": 0.50,           # 与 Barra 风格因子相关性上限
    "min_oos_icir": 0.5,                     # 样本外 ICIR 下限
    "min_recent_3y_ic_positive_ratio": 0.60, # 近 3 年 IC 正向比例下限
    "direction_consistency_required": True,  # 所有子时段 IC 是否必须同号
}


def load_candidate_pool() -> dict:
    """读取当前候选池，若不存在则返回空池"""
    if CANDIDATE_POOL_PATH.exists():
        with open(CANDIDATE_POOL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "_meta": {"version": "1.0", "description": "多因子准入候选池", "updated_at": None,
                   "admit_thresholds": ADMIT_THRESHOLDS},
        "factors": [], "rejected": [],
        "stats": {"total_candidates": 0, "admitted": 0, "rejected_corr": 0,
                  "rejected_marginal": 0, "rejected_direction": 0},
    }


def save_candidate_pool(pool: dict):
    """写回候选池"""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATE_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    print(f"[admit] 候选池已更新: {CANDIDATE_POOL_PATH}")


def scan_tier_factors(tier: str = "satellite") -> list[dict]:
    """
    扫描 qualified_factors/{tier} 获取因子清单。
    当前 tier 目录为空时，回退扫描 factor_library 中 lifecycle_stage=active 的因子。
    """
    tier_dir = QUALIFIED_DIR / tier
    if tier_dir.exists() and any(f.suffix == ".yaml" for f in tier_dir.iterdir()):
        # 有正式的分档文件，走分档目录
        factors = []
        import yaml
        for f in tier_dir.glob("*.yaml"):
            with open(f, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp)
            for fd in (data.get("factors") or []):
                fd["_source_file"] = f.stem
                fd["_tier"] = tier
                factors.append(fd)
        return factors
    else:
        # 分档目录为空，回退从 factor_library 扫描 lifecycle_stage=active 的因子
        print(f"[admit] qualified_factors/{tier} 为空，回退扫描因子库 lifecycle_stage=active 因子")
        import yaml
        factors = []
        for yaml_file in FACTOR_LIBRARY_DIR.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as fp:
                    data = yaml.safe_load(fp)
            except Exception:
                continue
            if data is None:
                continue
            for fd in (data.get("factors") or []):
                if fd.get("lifecycle_stage") in ("active", "exploration"):
                    fd["_source_file"] = yaml_file.stem
                    fd["_tier"] = tier
                    factors.append(fd)
        return factors


WAREHOUSE_DIR = PROJECT_ROOT / "factor_data" / "warehouse"

# 相关性计算的时间窗口（近 3 年，反映近期因子关系）
CORR_START_DATE = "2023-01-01"
CORR_END_DATE = "2025-12-31"


def compute_correlation_matrix(factor_names: list[str]) -> pd.DataFrame:
    """
    从 warehouse 按年 parquet 读取因子值，计算两两间 RankIC 相关矩阵。
    数据格式：warehouse/{name}/YYYY.parquet → MultiIndex=(instrument, datetime) → 单列 {name}
    """
    # 加载每个因子的值
    factor_data = {}
    missing = []
    for name in factor_names:
        fdir = WAREHOUSE_DIR / name
        if not fdir.exists():
            missing.append(name)
            continue
        # 读 2023-2025 年的 parquet
        dfs = []
        for y in range(2023, 2026):
            pf = fdir / f"{y}.parquet"
            if pf.exists():
                dfs.append(pd.read_parquet(pf))
        if not dfs:
            missing.append(name)
            continue
        combined = pd.concat(dfs)
        # 只保留 CORR_START_DATE ~ CORR_END_DATE 范围
        dts = combined.index.get_level_values("datetime")
        combined = combined[(dts >= CORR_START_DATE) & (dts <= CORR_END_DATE)]
        factor_data[name] = combined

    if missing:
        print(f"[admit] 以下因子在 warehouse 中无 2023~2025 年数据: {missing}")
        if not factor_data:
            print("[admit] 无任何因子可计算相关矩阵")
            return pd.DataFrame()

    # 合并为一个宽表：MultiIndex × 因子名
    merged = pd.concat(factor_data.values(), axis=1, keys=factor_data.keys())
    # 列名从 MultiIndex 的第一层取值
    merged.columns = merged.columns.get_level_values(0)
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # 计算每日截面 RankIC 相关（Spearman）
    daily_corrs = []
    for dt, group in merged.groupby(level="datetime"):
        # 取该日所有列的非空截面
        valid = group.dropna(how="all")
        if len(valid) < 50:  # 太少的截面不参与计算
            continue
        # 每列 rank
        ranked = valid.rank(pct=True)
        # 两两相关
        corr = ranked.corr(method="spearman")
        daily_corrs.append(corr)

    if not daily_corrs:
        print("[admit] 无足够截面计算相关矩阵")
        return pd.DataFrame()

    # 取时间均值
    avg_corr = pd.concat(daily_corrs).groupby(level=0).mean()
    print(f"[admit] 相关矩阵基于 {len(daily_corrs)} 个交易日截面计算")
    return avg_corr


def check_correlation(new_factor: str, existing_factors: list[dict],
                      corr_matrix: pd.DataFrame, threshold: float = 0.70) -> tuple[bool, list[str]]:
    """
    检验新因子与已有因子池的相关性。
    返回 (是否通过, 高相关因子列表)
    """
    if not existing_factors:
        return True, []

    existing_names = [f["name"] for f in existing_factors]
    if new_factor not in corr_matrix.index:
        return True, []

    high_corr = []
    for ename in existing_names:
        if ename in corr_matrix.columns:
            corr_val = corr_matrix.loc[new_factor, ename]
            if abs(corr_val) > threshold:
                high_corr.append(f"{ename}({corr_val:.2f})")

    return len(high_corr) == 0, high_corr


# ==============================================================================
# 增量 IC 检验共享辅助（供边际贡献 / 方向一致性使用）
# ==============================================================================

# 增量检验统一窗口：近 3 年（2023-2025），与相关矩阵计算窗口一致，按年度切片
SUB_PERIODS = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
]


def _load_label_frame(start: str, end: str) -> pd.Series | None:
    """
    加载标签序列（与全链路一致的 DK 标签表达式），供边际/方向检验使用。

    返回:
    - MultiIndex (instrument, datetime) 的标签 Series；加载失败返回 None
      （调用方回退到近似检验并给出警告）。
    """
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.config import QLIB_DATA_DIR
        from qlworks.factors.filter_utils import filter_untradeable_labels

        qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)
        ins_file = Path(str(QLIB_DATA_DIR)) / "instruments" / f"{INSTRUMENTS}.txt"
        if not ins_file.exists():
            print(f"[admit] 股票池 {INSTRUMENTS}.txt 不存在，标签加载失败")
            return None
        instruments = []
        with open(ins_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    instruments.append(parts[0])
        if not instruments:
            return None

        raw = D.features(instruments, [LABEL_EXPR], start, end)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw = raw.rename(columns={raw.columns[0]: LABEL_NAME})

        # 与训练端对齐：剔除涨跌停/一字板/持仓期停牌样本
        raw = filter_untradeable_labels(raw, instruments, start, end)
        if raw is None or raw.empty:
            return None

        flat = raw.reset_index()
        flat["instrument"] = flat["instrument"].str.lower()
        return flat.set_index(["instrument", "datetime"]).sort_index()[LABEL_NAME]
    except Exception as e:
        print(f"[admit] 标签加载失败（{e}），增量检验回退到近似路径")
        return None


def _load_warehouse_series(name: str) -> pd.Series | None:
    """
    从 warehouse 读取因子在 2023~2025 年的序列（单列），instrument 统一小写与标签对齐。
    无数据返回 None。
    """
    fdir = WAREHOUSE_DIR / name
    if not fdir.exists():
        return None
    dfs = []
    for y in range(2023, 2026):
        pf = fdir / f"{y}.parquet"
        if pf.exists():
            dfs.append(pd.read_parquet(pf))
    if not dfs:
        return None
    combined = pd.concat(dfs)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    if combined.empty:
        return None
    series = combined.iloc[:, 0]
    series.index = series.index.set_levels(
        [lv.str.lower() if lv.name == "instrument" else lv for lv in series.index.levels],
    )
    return series


def _daily_ic(factor_s: pd.Series, label_s: pd.Series, min_cross: int = 20) -> pd.Series:
    """
    逐日截面 Spearman IC 序列（index=datetime）。

    输入:
    - factor_s / label_s: MultiIndex (instrument, datetime) 的 Series
    - min_cross: 单日截面最小样本数

    返回:
    - 逐日 IC Series；无有效截面时返回空 Series
    注意: 截面无区分度（常数）时 IC 记为 0.0（无预测能力），而非 NaN 丢弃，
          否则组合信号退化时无法正确判负边际贡献。
    """
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


def _slice_period(series: pd.Series, start: str, end: str) -> pd.Series:
    """按日期区间切片（datetime 为 MultiIndex 第二层）。"""
    dts = series.index.get_level_values("datetime")
    return series[(dts >= start) & (dts <= end)]


def check_marginal_contribution(new_factor: str, existing_factors: list[dict],
                                label_frame: pd.Series | None = None,
                                corr_matrix: pd.DataFrame | None = None) -> tuple[bool, str]:
    """
    增量边际贡献检验（真实增量 ICIR，[修正 P1-2]）。

    逻辑：
    - 候选池为空 → 自动通过
    - 以已有池因子"截面等权组合信号"为基准 S0，加入新因子后为 S1
    - 逐日计算组合 IC，ICIR = mean(IC) / std(IC)，按年度子时段对比增量
    - 要求各年度增量 ICIR 的均值 ≥ ICIR_IMPROVE_MIN（默认 0，即不得拉低组合 ICIR）

    注：原实现以"平均相关 > 0.40"近似边际贡献，与文档口径不符；现改为真实
    增量检验。标签数据缺失时回退到平均相关近似（带警告）。
    """
    if not existing_factors:
        return True, "候选池为空，无需边际检验"

    existing_names = [f["name"] for f in existing_factors]

    # ── 路径 A：真实增量 ICIR 检验（需标签数据） ──
    if label_frame is not None:
        factor_frames = {}
        missing = []
        for name in existing_names + [new_factor]:
            series = _load_warehouse_series(name)
            if series is None:
                missing.append(name)
            else:
                factor_frames[name] = series
        have_existing = [n for n in existing_names if n in factor_frames]
        if new_factor in factor_frames and have_existing:
            # 已有池等权组合信号（逐日截面 rank 后取均值）
            base = pd.DataFrame({n: factor_frames[n] for n in have_existing})
            base_rank = base.groupby(level="datetime").rank(pct=True)
            s0 = base_rank.mean(axis=1)

            # 加入新因子后的组合信号（基准与新因子各占 50% 权重）
            joined = pd.concat({"s0": s0, "new": factor_frames[new_factor]}, axis=1)
            joined_rank = joined.groupby(level="datetime").rank(pct=True)
            s1 = (joined_rank["s0"] + joined_rank["new"]) / 2

            details = []
            deltas = []
            for label, start, end in SUB_PERIODS:
                ic0 = _daily_ic(_slice_period(s0, start, end),
                                _slice_period(label_frame, start, end))
                ic1 = _daily_ic(_slice_period(s1, start, end),
                                _slice_period(label_frame, start, end))
                if len(ic0) < 10 or len(ic1) < 10:
                    continue
                icir0 = ic0.mean() / (ic0.std() + 1e-9)
                icir1 = ic1.mean() / (ic1.std() + 1e-9)
                deltas.append(icir1 - icir0)
                details.append(f"{label} ICIR {icir0:.3f}→{icir1:.3f} (Δ{icir1 - icir0:+.3f})")

            if deltas:
                avg_delta = float(np.mean(deltas))
                if avg_delta < ICIR_IMPROVE_MIN:
                    return False, f"增量 ICIR 边际贡献为负: {avg_delta:+.3f} ({'; '.join(details)})"
                return True, f"增量 ICIR 边际贡献: {avg_delta:+.3f} ({'; '.join(details)})"

        print(f"[admit] {new_factor}: 增量 ICIR 检验数据不足，回退平均相关近似")

    # ── 路径 B（回退）：平均相关近似（原逻辑） ──
    if corr_matrix is None or new_factor not in corr_matrix.index:
        return True, "新因子不在相关矩阵中，默认通过"
    vals = []
    for ename in existing_names:
        if ename in corr_matrix.columns:
            v = abs(corr_matrix.loc[new_factor, ename])
            vals.append(v)
    if not vals:
        return True, "与已有池因子无交集"
    avg_corr = np.mean(vals)
    if avg_corr > 0.40:
        return False, f"冗余度偏高: 新因子与已有池平均相关 {avg_corr:.2f} > 0.40（标签缺失近似）"

    return True, f"通过（与已有池平均相关 {avg_corr:.2f}，标签缺失近似）"


def check_direction_consistency(new_factor: str,
                                label_frame: pd.Series | None = None) -> tuple[bool, str]:
    """
    IC 方向一致性检验（[修正 P1-1]）。

    原实现检验"因子值均值的子时段符号"，与"IC 符号"无必然联系（因子值恒正但
    IC 完全可能为负），不符合准入"方向一致性"语义。现改为：
    - 优先：基于 warehouse 因子值与标签的逐年 Spearman IC，要求各年度 IC 同号
    - 回退：标签缺失时保留因子值符号校验（带警告）

    窗口与相关矩阵统一为近 3 年（2023-2025，年度切片）。
    """
    series = _load_warehouse_series(new_factor)
    if series is None:
        return True, "因子无 warehouse 数据，默认通过"
    if len(series) < 1000:
        return True, "数据量不足"

    # ── 路径 A：子时段 IC 符号一致性（首选） ──
    if label_frame is not None:
        ic_means = []
        details = []
        for label, start, end in SUB_PERIODS:
            ic_series = _daily_ic(_slice_period(series, start, end),
                                  _slice_period(label_frame, start, end))
            if ic_series.empty:
                continue
            ic_means.append(ic_series.mean())
            details.append(f"{label} IC={ic_series.mean():.4f}")
        if len(ic_means) >= 2:
            if all(v > 0 for v in ic_means) or all(v < 0 for v in ic_means):
                return True, "IC 方向一致: " + "; ".join(details)
            return False, "IC 方向不一致: " + "; ".join(details)
        print(f"[admit] {new_factor}: 标签子时段不足，回退因子值符号校验")

    # ── 路径 B（回退）：因子值符号一致性 ──
    signs = []
    details = []
    for label, start, end in SUB_PERIODS:
        subset = _slice_period(series, start, end)
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


def _load_registry_metrics(factor_name: str) -> dict:
    """从 registry.json 读取该因子评测指标（ic_mean/icir/win_rate/sharpe 等）。

    [P0-2] 替代原 latest_icir=0.5 placeholder：准入时使用真实 IC 指标，
    数据有效性门（IC 缺失判 pending_data）据此生效。
    """
    reg_path = REGISTRY_DIR / "registry.json"
    if not reg_path.exists():
        return {}
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return {}
    entry = (reg.get("factors") or {}).get(factor_name) or {}
    return {
        "ic_mean": entry.get("ic_mean"),
        "ir": entry.get("icir"),
        "ic_positive_ratio": entry.get("win_rate"),
        "sharpe": entry.get("ls_sharpe"),
    }


def _admit_entry(factor_name: str, factor_info: dict, result: dict, tier: str) -> Optional[dict]:
    """构造候选池 Alpha Book 标准条目（P0-2 / P1-6）。

    含 direction（真实 IC 符号推导，不做 abs）、eval_date、tier_history、rolling_ic。
    IC 均值为空（数据源缺失）→ 返回 None（不入池，判 pending_data）。
    """
    from datetime import datetime
    metrics = _load_registry_metrics(factor_name)
    ic_mean = metrics.get("ic_mean")
    if ic_mean is None:
        print(f"[admit] {factor_name}: registry 无 IC 指标（未评测或数据缺失），判 pending_data，不入池")
        return None
    direction = "positive" if float(ic_mean) > 0 else "negative"
    now = datetime.now().isoformat(timespec="seconds")
    latest_icir = metrics.get("ir", 0) or 0
    result["latest_icir"] = latest_icir
    return {
        "name": factor_name,
        "tier": tier,
        "direction": direction,
        "category": factor_info.get("category", ""),
        "sub_category": factor_info.get("sub_category", ""),
        "meaning": factor_info.get("meaning", ""),
        "source_file": factor_info.get("_source_file", ""),
        "latest_icir": latest_icir,
        "admitted_at": now,
        "eval_date": now,
        "rolling_ic": {},
        "tier_history": [{"tier": tier, "at": now}],
        "status": "admitted",
        "_metrics": {k: v for k, v in metrics.items() if v is not None},
        "_screening": result,
    }


def _bump_set_version(pool: dict) -> str:
    """[P1-6] 因子集版本号递增（用于训练追溯：哪个版本的因子池 → 哪个模型/回测）。"""
    meta = pool.setdefault("_meta", {})
    cur = meta.get("set_version", "v1")
    try:
        n = int(str(cur).lstrip("v")) + 1
    except (ValueError, TypeError):
        n = 2
    meta["set_version"] = f"v{n}"
    return meta["set_version"]


def admit_factor(new_factor_name: str, new_factor_info: dict,
                 existing_factors: list[dict], corr_matrix: pd.DataFrame,
                 label_frame: pd.Series | None = None) -> dict:
    """
    对新因子执行多因子准入三关检验。
    返回准入结果字典。
    """
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

    # 第一关：相关性检验
    corr_passed, high_corr = check_correlation(
        new_factor_name, existing_factors, corr_matrix,
        threshold=ADMIT_THRESHOLDS["max_correlation_existing"]
    )
    result["corr_check"]["passed"] = corr_passed
    result["corr_check"]["high_corr_factors"] = high_corr

    if not corr_passed:
        result["reasons"].append(f"相关性检验未通过: 与 {', '.join(high_corr)} 高度相关")

    # 第二关：增量 ICIR 边际贡献检验（标签可用时走真实增量，否则回退平均相关近似）
    marginal_passed, marginal_detail = check_marginal_contribution(
        new_factor_name, existing_factors, label_frame, corr_matrix
    )
    result["marginal_check"]["passed"] = marginal_passed
    result["marginal_check"]["detail"] = marginal_detail

    if not marginal_passed:
        result["reasons"].append(f"边际贡献检验未通过: {marginal_detail}")

    # 第三关：IC 方向一致性（标签可用时按年度 IC 符号，否则回退因子值符号）
    dir_passed, dir_detail = check_direction_consistency(new_factor_name, label_frame)
    result["direction_check"]["passed"] = dir_passed
    result["direction_check"]["detail"] = dir_detail

    if not dir_passed:
        result["reasons"].append(f"方向一致性未通过: {dir_detail}")

    # 综合判定（latest_icir 由 _admit_entry 从 registry 真实指标写入，此处不设占位值）
    result["admitted"] = corr_passed and marginal_passed and dir_passed

    return result


def build_all():
    """构建完整候选池：扫描所有 core/satellite 因子，逐一执行准入检验"""
    pool = load_candidate_pool()
    existing = list(pool.get("factors", []))

    # 扫描 core + satellite 档位
    candidates = []
    for tier in ("core", "satellite"):
        candidates.extend(scan_tier_factors(tier))

    if not candidates:
        print("[admit] 未找到任何候选因子（qualified_factors 为空，因子库中也无 active/exploration 因子）")
        return

    factor_names = [c["name"] for c in candidates]
    existing_names = [f["name"] for f in existing]
    all_names = list(dict.fromkeys(factor_names + existing_names))

    print(f"[admit] 候选因子: {len(factor_names)} 个 | 已有池: {len(existing)} 个")
    corr_matrix = compute_correlation_matrix(all_names)
    # 加载统一窗口（2023-2025）标签，供增量 ICIR 边际贡献 / IC 方向一致性检验
    label_frame = _load_label_frame(CORR_START_DATE, CORR_END_DATE)

    admitted = []
    rejected = []

    for cand in candidates:
        name = cand["name"]
        if name in existing_names:
            print(f"  [跳过] {name} 已在候选池中")
            existing_entry = next((e for e in existing if e["name"] == name), None)
            admitted.append(existing_entry or cand)
            continue

        # 前置检查：必须有评测报告
        report_tier = _find_evaluation_report(name)
        if report_tier is None:
            print(f"  [跳过] {name} 未完成单因子评测，跳过（先跑 run_eval.py）")
            continue

        cand["_tier"] = report_tier  # 以报告 tier 为准
        result = admit_factor(name, cand, existing, corr_matrix, label_frame)

        if result["admitted"]:
            entry = _admit_entry(name, cand, result, report_tier)
            if entry is None:
                continue  # 数据有效性门未过（IC 缺失），不入池
            print(f"  [准入] {name} (tier={result['tier']}, direction={entry['direction']}, icir={entry['latest_icir']:.2f})")
            admitted.append(entry)
        else:
            print(f"  [拒绝] {name} 原因: {'; '.join(result['reasons'])}")
            rejected.append(result)

    # 更新候选池（Alpha Book v2，含 set_version）
    pool["factors"] = admitted
    pool["rejected"] = rejected
    pool["_meta"]["updated_at"] = pd.Timestamp.now().isoformat()[:19]
    pool["_meta"]["description"] = "多因子准入候选池（Alpha Book）— 仅由 admit_to_multifactor.py 三关检验写入，下游 train_tree-doubao.py 从此读取因子名单"
    _bump_set_version(pool)
    pool["stats"] = {
        "total_candidates": len(candidates),
        "admitted": len(admitted),
        "rejected_corr": sum(1 for r in rejected if not r["corr_check"]["passed"]),
        "rejected_marginal": sum(1 for r in rejected if not r["marginal_check"]["passed"]),
        "rejected_direction": sum(1 for r in rejected if not r["direction_check"]["passed"]),
    }
    save_candidate_pool(pool)


def admit_single(factor_name: str, tier: str = None):
    """单个因子准入检验（必须先跑 run_eval.py 生成评测报告）。"""
    # ── 前置检查：必须已有评测报告 ──
    detected_tier = _find_evaluation_report(factor_name)
    if detected_tier is None:
        print(f"[admit] 错误: {factor_name} 尚未完成单因子评测。")
        print(f"[admit] 请先执行: python scripts/evaluation/run_eval.py --factor {factor_name}")
        return

    # tier 优先用显式传入，否则从报告路径自动推断
    tier = tier or detected_tier
    print(f"[admit] 评测报告已找到: tier={detected_tier}")
    pool = load_candidate_pool()
    existing = list(pool.get("factors", []))

    # 从因子库读取因子定义
    import yaml
    factor_info = None
    for yaml_file in FACTOR_LIBRARY_DIR.glob("*.yaml"):
        try:
            with open(yaml_file, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp)
        except Exception:
            continue
        if data is None:
            continue
        for fd in (data.get("factors") or []):
            if fd.get("name") == factor_name:
                fd["_source_file"] = yaml_file.stem
                fd["_tier"] = tier
                factor_info = fd
                break
        if factor_info:
            break

    if not factor_info:
        print(f"[admit] 在因子库中未找到因子: {factor_name}")
        return

    # 拒绝重复准入
    if any(f["name"] == factor_name for f in existing):
        print(f"[admit] {factor_name} 已在候选池中，跳过")
        return

    # 构建相关矩阵
    all_names = [factor_name] + [f["name"] for f in existing]
    corr_matrix = compute_correlation_matrix(all_names)
    # 加载统一窗口（2023-2025）标签，供增量 ICIR 边际贡献 / IC 方向一致性检验
    label_frame = _load_label_frame(CORR_START_DATE, CORR_END_DATE)

    result = admit_factor(factor_name, factor_info, existing, corr_matrix, label_frame)
    print(f"\n=== 准入检验结果: {factor_name} ===")
    print(f"  档位: {tier}")
    print(f"  分类: {factor_info.get('category', '')} / {factor_info.get('sub_category', '')}")
    print(f"  相关性检验: {'通过' if result['corr_check']['passed'] else '未通过'}")
    if result["corr_check"]["high_corr_factors"]:
        print(f"    高相关因子: {', '.join(result['corr_check']['high_corr_factors'])}")
    print(f"  边际贡献检验: {'通过' if result['marginal_check']['passed'] else '未通过'}")
    print(f"    详情: {result['marginal_check']['detail']}")
    print(f"  方向一致性: {'通过' if result['direction_check']['passed'] else '未通过'}")
    print(f"    详情: {result['direction_check']['detail']}")
    print(f"  >>> 综合判定: {'准 入' if result['admitted'] else '拒 绝'}")

    if result["admitted"]:
        # 写入候选池（Alpha Book 标准条目，含 direction / eval_date / set_version）
        entry = _admit_entry(factor_name, factor_info, result, tier)
        if entry is None:
            print(f"[admit] {factor_name}: 数据有效性门未过（IC 缺失），不入池")
            return
        pool["factors"].append(entry)
        pool["_meta"]["updated_at"] = pd.Timestamp.now().isoformat()[:19]
        pool["_meta"]["description"] = "多因子准入候选池（Alpha Book）— 仅由 admit_to_multifactor.py 三关检验写入，下游 train_tree-doubao.py 从此读取因子名单"
        _bump_set_version(pool)
        pool["stats"]["total_candidates"] = len(pool["factors"]) + len(pool["rejected"])
        pool["stats"]["admitted"] = len(pool["factors"])
        save_candidate_pool(pool)
    else:
        # 记录拒绝日志
        pool["rejected"].append(result)
        pool["stats"]["total_candidates"] = len(pool["factors"]) + len(pool["rejected"])
        pool["stats"]["rejected_corr"] = sum(1 for r in pool["rejected"]
                                              if not r.get("corr_check", {}).get("passed", False))
        save_candidate_pool(pool)


def _parse_args():
    parser = argparse.ArgumentParser(description="多因子准入编排层")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-all", action="store_true",
                       help="扫描所有 core/satellite 因子，批量构建候选池")
    group.add_argument("--factor", type=str, default=None,
                       help="单因子准入检验，配合 --tier 使用")
    parser.add_argument("--tier", type=str, default=None,
                       choices=["core", "satellite", "archive"],
                       help="因子档位（可选，默认从评测报告自动推断）")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.build_all:
        build_all()
    elif args.factor:
        admit_single(args.factor, tier=args.tier)
