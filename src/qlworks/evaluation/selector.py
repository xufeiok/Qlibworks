"""
统一因子筛选服务（Screening）— 收敛因子筛选的重复逻辑。

[背景]
scripts/training/select_factors.py 与 scripts/training/train_tree.py 各自实现了
高度相似的筛选逻辑：
  - IC 粗筛（逐列 Spearman IC，按 |IC| 排序取 top_k）
  - 因子冗余检测（高相关因子对剔除低重要度者）
  - ICIR 稳定性校验（滚动窗口 ICIR 正向占比）
  - 跨窗口聚合（因子在多窗口的生存率）

本模块将这些逻辑收敛为通用原语，两处脚本统一调用，避免口径漂移：
  - 修一处，处处生效
  - 可独立单元测试（无需跑完整训练）

对标世界一流粗筛（Screening）阶段，提供 5 道快检门：
  ① 数据质量门（覆盖率/常数因子）  compute_coverage / check_quality_gate
  ② IC 统计（IC/ICIR/正 IC 占比）    compute_factor_ics / compute_win_rate
  ③ 稳定性（滚动 ICIR 正向占比）     check_icir_stability
  ④ 冗余（高相关剔除）               check_redundancy
  ⑤ 多重检验校正（BH/Holm）          apply_multiple_testing_correction
编排入口：screening_pipeline

[使用]
  from qlworks.evaluation.selector import (
      compute_factor_ics, screen_by_ic, check_redundancy, check_icir_stability,
      compute_coverage, check_quality_gate, compute_win_rate,
      apply_multiple_testing_correction, select_top_by_abs,
      aggregate_across_windows, screening_pipeline,
  )
"""

import numpy as np
import pandas as pd


DEFAULT_SCREENING_CONFIG = {
    "min_coverage": 0.8,        # 覆盖率门槛（有效值占比）
    "min_nunique": 50,          # 常数因子门槛（至少 50 个不同值）
    "min_samples": 50,          # IC 计算有效样本下限
    "icir_window": 60,          # ICIR 滚动窗口
    "icir_keep_ratio": 0.8,     # ICIR 正向占比保留比例
    "icir_min_keep": 3,         # ICIR 至少保留数
    "redundancy_threshold": 0.90,   # 冗余相关系数阈值
    "redundancy_method": "spearman",  # 冗余相关方法（select_factors 用 spearman）
    "correction_alpha": 0.05,   # 多重检验显著性水平
    "correction_method": "bh",  # bh / holm
}


def _spearman_ic(predicted: pd.Series, actual: pd.Series) -> float:
    """Spearman Rank IC，与 qlworks.models.training.compute_ic 逻辑一致（内联以解耦重依赖）。"""
    from scipy.stats import spearmanr
    combined = pd.DataFrame({"pred": predicted, "actual": actual}).dropna()
    if len(combined) < 10:
        return 0.0
    if combined["pred"].nunique() == 1 or combined["actual"].nunique() == 1:
        return 0.0
    return float(spearmanr(combined["pred"], combined["actual"])[0])


# ─────────────────────────── ① 数据质量门 ───────────────────────────

def compute_coverage(factor_frame: pd.DataFrame) -> pd.Series:
    """每因子的覆盖率 = 非 NaN 值占比（0~1）。"""
    if factor_frame.empty:
        return pd.Series(dtype=float)
    return factor_frame.notna().mean(axis=0)


def check_quality_gate(factor_frame: pd.DataFrame,
                       min_coverage: float = 0.8,
                       min_nunique: int = 50) -> pd.Series:
    """数据质量门：覆盖率 ≥ min_coverage 且 非常数（nunique ≥ min_nunique）。

    返回 Series(bool, index=因子名)。覆盖差/常数的因子即使 IC 虚高也应淘汰。
    """
    if factor_frame.empty:
        return pd.Series(dtype=bool)
    coverage = compute_coverage(factor_frame)
    nunique = factor_frame.nunique(dropna=True)
    return (coverage >= min_coverage) & (nunique >= min_nunique)


# ─────────────────────────── ② IC 统计 ───────────────────────────

def compute_factor_ics(factor_frame: pd.DataFrame, label_series: pd.Series,
                       min_samples: int = 50, compute_ic_fn=None) -> pd.Series:
    """逐列计算因子 IC（Spearman），返回 Series(col → ic)。

    与 train_tree._batch_factor_ic_selection 的逐列逻辑一致：
    每列独立 dropna 后与标签对齐（索引无关顺序），保证列级缺失互不干扰。

    Args:
        factor_frame: 因子数据（行=样本, 列=因子）
        label_series: 标签 Series（与 factor_frame 行对齐）
        min_samples: 有效样本数下限，不足返回空
        compute_ic_fn: IC 计算函数，默认内联 Spearman（与 training.compute_ic 逻辑一致）

    Returns:
        Series，index=因子列名，values=IC
    """
    if compute_ic_fn is None:
        compute_ic_fn = _spearman_ic
    ics = {}
    for col in factor_frame.columns:
        feat = factor_frame[col].dropna()
        lab = label_series.reindex(feat.index).dropna()
        common = feat.index.intersection(lab.index)
        if len(common) < min_samples:
            continue
        try:
            ics[col] = compute_ic_fn(feat.loc[common], lab.loc[common])
        except Exception:
            ics[col] = 0.0
    return pd.Series(ics, dtype=float)


def compute_daily_ic_frame(factor_frame: pd.DataFrame,
                           label_series: pd.Series) -> pd.DataFrame:
    """逐日截面 Spearman IC 面板：返回 DataFrame(date × factor)。

    向量化 groupby().corr() 替代逐日 apply，等价于逐日循环的截面 IC。
    因子数据需为 MultiIndex（含 'datetime' level），标签与因子行对齐。
    """
    frame = factor_frame.copy()
    frame = frame.join(label_series.rename("__label__"), how="inner")
    if len(frame) == 0:
        return pd.DataFrame()
    all_cols = list(frame.columns)
    if "__label__" not in all_cols:
        return pd.DataFrame()
    feat_cols = [c for c in all_cols if c != "__label__"]
    try:
        corr = frame.groupby(level="datetime")[all_cols].corr(method="spearman")
        daily_ic = corr.xs("__label__", level=1, axis=0)[feat_cols]
    except Exception:
        return pd.DataFrame()
    return daily_ic


def compute_win_rate(daily_ic: pd.DataFrame) -> pd.Series:
    """正 IC 占比（win rate）= 每日 IC > 0 的天数占比（0~1）。

    Args:
        daily_ic: compute_daily_ic_frame 的输出（date × factor）
    """
    if daily_ic.empty:
        return pd.Series(dtype=float)
    return (daily_ic > 0).mean(axis=0)


def screen_by_ic(factor_frame: pd.DataFrame, label_series: pd.Series,
                 top_k: int, min_samples: int = 50, compute_ic_fn=None) -> list:
    """IC 粗筛：按 |IC| 降序取 top_k 因子。

    Args:
        factor_frame: 因子数据（行=样本, 列=因子）
        label_series: 标签 Series（与 factor_frame 行对齐）
        top_k: 保留因子数
        min_samples: 有效样本数下限
        compute_ic_fn: IC 计算函数，默认 Spearman

    Returns:
        因子名列表（按 |IC| 降序，长度 ≤ top_k）
    """
    ic_series = compute_factor_ics(factor_frame, label_series, min_samples, compute_ic_fn)
    return select_top_by_abs(ic_series, top_k)


def select_top_by_abs(series: pd.Series, top_k: int) -> list:
    """按 |值| 降序取 top_k，返回 index 列表（通用 top-k 截断）。"""
    s = pd.Series(series).dropna().sort_values(key=abs, ascending=False)
    return list(s.head(top_k).index)


# ─────────────────────────── ③ 稳定性 ───────────────────────────

def check_icir_stability(factor_frame: pd.DataFrame, label_series: pd.Series,
                         factor_names: list, rolling_window: int = 60,
                         keep_ratio: float = 0.8, min_keep: int = 3) -> list:
    """ICIR 稳定性校验：滚动窗口 ICIR 正向占比，保留 top keep_ratio 的因子。

    逐日截面 Spearman IC → 滚动 mean/std → 滚动 ICIR →
    正向天数占比排序 → 保留前 keep_ratio（至少 min_keep 个）。

    Args:
        factor_frame: 因子数据（MultiIndex，需含 'datetime' level）
        label_series: 标签 Series（与 factor_frame 行对齐）
        factor_names: 待检测因子列表
        rolling_window: 滚动窗口天数
        keep_ratio: ICIR 正向占比保留比例（0~1）
        min_keep: 至少保留因子数

    Returns:
        稳定因子名列表（按 ICIR 正向占比降序；样本不足时返回原列表）
    """
    feat = [c for c in factor_names if c in factor_frame.columns]
    if not feat:
        return list(factor_names)

    daily_ic = compute_daily_ic_frame(factor_frame[feat], label_series)
    if daily_ic.empty or len(daily_ic) < rolling_window // 2:
        return list(factor_names)

    rolling_mean = daily_ic.rolling(window=rolling_window, min_periods=rolling_window // 2).mean()
    rolling_std = daily_ic.rolling(window=rolling_window, min_periods=rolling_window // 2).std()
    rolling_icir = rolling_mean / rolling_std.replace(0, np.nan)

    pos_ratio = (rolling_icir > 0).sum() / rolling_icir.notna().sum()
    pos_ratio = pos_ratio.fillna(0).sort_values(ascending=False)
    keep_count = max(int(len(pos_ratio) * keep_ratio), min_keep)
    return pos_ratio.head(keep_count).index.tolist()


# ─────────────────────────── ④ 冗余 ───────────────────────────

def check_redundancy(factor_frame: pd.DataFrame, factor_names: list,
                     threshold: float = 0.90, rank: dict = None,
                     method: str = "spearman") -> list:
    """冗余检测：剔除相关性 > threshold 的因子对中 rank 较低者。

    Args:
        factor_frame: 因子数据（行=样本, 列=因子），用于计算相关矩阵
        factor_names: 待检测因子列表（输出保持此顺序）
        threshold: 相关系数阈值（绝对值，超过即视为冗余）
        rank: {name: 数值}，值越大越优先保留；
              缺省时按 factor_names 顺序判定（靠前者保留）
        method: 相关系数方法（spearman/pearson），select_factors 用 spearman，
                train_tree 沿用 pearson 以保持原行为

    Returns:
        保留的因子名列表（保持 factor_names 原顺序）
    """
    feat_in_data = [c for c in factor_names if c in factor_frame.columns]
    if len(feat_in_data) < 2:
        return list(factor_names)
    try:
        corr_mat = factor_frame[feat_in_data].corr(method=method).abs()
    except Exception:
        return list(factor_names)

    to_drop = set()
    for i in range(len(corr_mat.columns)):
        for j in range(i + 1, len(corr_mat.columns)):
            c1, c2 = corr_mat.columns[i], corr_mat.columns[j]
            if corr_mat.iloc[i, j] <= threshold:
                continue
            if c1 in to_drop or c2 in to_drop:
                continue
            if rank and c1 in rank and c2 in rank:
                # 有重要性分：保留得分更高者
                drop_f = c2 if rank[c1] >= rank[c2] else c1
            else:
                # 无重要性分：保留列表顺序靠前者
                idx1 = factor_names.index(c1) if c1 in factor_names else 999
                idx2 = factor_names.index(c2) if c2 in factor_names else 999
                drop_f = c2 if idx1 <= idx2 else c1
            to_drop.add(drop_f)
    return [f for f in factor_names if f not in to_drop]


# ─────────────────────────── ⑤ 多重检验校正 ───────────────────────────

def apply_multiple_testing_correction(daily_ic: pd.DataFrame,
                                      n_tested: int = None,
                                      alpha: float = 0.05,
                                      method: str = "bh"):
    """IC 多重检验校正：逐日 IC → t-stat → p 值 → BH/Holm 校正。

    从 N 个因子中挑 top 必然存在选择偏差（data mining bias）：
    t-stat = mean(IC)/std(IC) * sqrt(观测天数)，校正后仅统计显著的因子通过。

    Args:
        daily_ic: compute_daily_ic_frame 的输出（date × factor）
        n_tested: 参与检验的因子总数（含被质量门淘汰者）；默认 = 列数
        alpha: 显著性水平
        method: "bh"（Benjamini-Hochberg FDR）或 "holm"

    Returns:
        (significant Series(bool), p_adjusted Series(float))，index=因子名
    """
    import math

    def _norm_cdf(x: float) -> float:
        """标准正态 CDF（math.erf 实现，解耦 scipy.stats.norm）。"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _adjust_pvalues(p_series: pd.Series, alpha: float) -> pd.Series:
        """内联 BH / Holm 多重检验校正（scipy.stats.multipletests 在新版 scipy 已移除）。

        BH:   q = p*m/rank 升序，从大到小累计最小值，q < alpha 拒绝
        Holm: adj = p*(m-rank+1)，累计最大值，adj < alpha 顺序拒绝
        """
        p = p_series.dropna()
        m = len(p)
        if m == 0:
            return p_series
        sorted_p = p.sort_values()
        ranks = np.arange(1, m + 1)
        if method == "holm":
            adj = np.minimum(1.0, sorted_p.values * (m - ranks + 1))
            adj = np.maximum.accumulate(adj)          # Holm 顺序性
        else:
            adj = np.minimum(1.0, sorted_p.values * m / ranks)
            adj = np.minimum.accumulate(adj[::-1])[::-1]  # BH 单调性
        return pd.Series(adj, index=sorted_p.index)

    daily_ic = daily_ic.dropna(axis=1, how="all")
    if daily_ic.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=float)

    mu = daily_ic.mean()
    sd = daily_ic.std()
    t = (mu / sd.replace(0, np.nan)) * np.sqrt(daily_ic.count())
    t = t.replace([np.inf, -np.inf], np.nan).dropna()
    if t.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=float)

    p = pd.Series({k: 2.0 * (1.0 - _norm_cdf(abs(v))) for k, v in t.items()})
    n = n_tested if (n_tested and n_tested > len(p)) else len(p)
    if n > len(p):
        # 实际参与检验的因子数 > 面板列数（被质量门淘汰者计入），
        # 将缺失因子的 p 值视为 1.0，参与校正以反映选择偏差
        pad = pd.Series(1.0, index=[f"__pad_{i}__" for i in range(n - len(p))])
        p = pd.concat([p, pad])

    p_adj = _adjust_pvalues(p, alpha)
    p_adj = p_adj.drop(index=[i for i in p_adj.index if str(i).startswith("__pad_")])
    significant = p_adj.lt(alpha)
    return significant, p_adj


# ─────────────────────────── 跨窗口聚合 ───────────────────────────

def aggregate_across_windows(selections: dict, min_window_ratio: float = 0.5) -> dict:
    """跨窗口聚合：因子在 >= min_window_ratio 比例的窗口中入选则最终选中。

    Args:
        selections: {factor_name: [bool, bool, ...]}，列表长度为窗口数
        min_window_ratio: 入选窗口占比门槛（0~1）

    Returns:
        {factor_name: {"selected": bool, "n_selected": int, "n_windows": int}}
    """
    n_windows = max((len(v) for v in selections.values()), default=0)
    if n_windows == 0:
        return {}
    # 向上取整保证"过半窗口"语义：n=3, ratio=0.5 → 至少 2 个窗口（而非 int(1.5)=1）
    min_wins = max(1, int(np.ceil(n_windows * min_window_ratio)))
    out = {}
    for name, sels in selections.items():
        n_sel = int(sum(bool(s) for s in sels))
        out[name] = {
            "selected": n_sel >= min_wins,
            "n_selected": n_sel,
            "n_windows": n_windows,
        }
    return out


# ─────────────────────────── 编排：screening_pipeline ───────────────────────────

def screening_pipeline(factor_frame: pd.DataFrame, label_series: pd.Series,
                       config: dict = None) -> dict:
    """批量因子粗筛流水线：5 道快检门，输出每因子的"粗筛卡"与候选清单。

    流程：
      ① check_quality_gate   数据质量门（覆盖率/常数因子）
      ② IC 统计              全样本 IC、ICIR、正 IC 占比
      ③ check_icir_stability 滚动 ICIR 稳定性
      ④ check_redundancy     冗余剔除（对稳定因子）
      ⑤ 多重检验校正         BH/Holm（对稳定因子）

    Args:
        factor_frame: 因子数据（MultiIndex，含 'datetime' level）
        label_series: 标签 Series（与 factor_frame 行对齐）
        config: 覆盖 DEFAULT_SCREENING_CONFIG 的配置 dict

    Returns:
        dict:
          - screen_card: DataFrame，每因子一行（coverage/nunique/ic/icir/win_rate/
            t_stat/p_adjusted/significant/quality_ok/stable/redundant/selected）
          - candidates: 最终通过粗筛的因子名列表
          - ic / icir / win_rate / coverage: Series
          - stable_factors / redundant_removed: list
          - significant: Series(bool)
    """
    cfg = {**DEFAULT_SCREENING_CONFIG, **(config or {})}
    if factor_frame.empty:
        empty_ser = pd.Series(dtype=float)
        return {"screen_card": pd.DataFrame(), "candidates": [],
                "ic": empty_ser, "icir": empty_ser, "win_rate": empty_ser,
                "coverage": empty_ser, "stable_factors": [], "redundant_removed": [],
                "significant": pd.Series(dtype=bool)}

    factor_frame = factor_frame.copy()

    # ① 数据质量门
    quality_ok = check_quality_gate(factor_frame, cfg["min_coverage"], cfg["min_nunique"])
    passed_qc = quality_ok[quality_ok].index.tolist()

    # ② IC 统计（对通过质量门的因子）
    ic = compute_factor_ics(factor_frame[passed_qc], label_series, cfg["min_samples"])
    daily_ic = compute_daily_ic_frame(factor_frame[passed_qc], label_series)
    if not daily_ic.empty:
        icir = (daily_ic.mean() / daily_ic.std().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        win_rate = compute_win_rate(daily_ic)
    else:
        icir, win_rate = pd.Series(dtype=float), pd.Series(dtype=float)

    # ③ 稳定性
    stable_factors = check_icir_stability(
        factor_frame, label_series, passed_qc,
        rolling_window=cfg["icir_window"], keep_ratio=cfg["icir_keep_ratio"],
        min_keep=cfg["icir_min_keep"],
    )

    # ④ 冗余（对稳定因子）
    redundant_removed = []
    kept = stable_factors
    if len(stable_factors) >= 2:
        kept = check_redundancy(factor_frame, stable_factors,
                                threshold=cfg["redundancy_threshold"],
                                method=cfg["redundancy_method"])
        redundant_removed = [f for f in stable_factors if f not in kept]

    # ⑤ 多重检验校正（对稳定因子）
    significant = pd.Series(True, index=kept)
    p_adjusted = pd.Series(dtype=float)
    if not daily_ic.empty and stable_factors:
        sig, p_adj = apply_multiple_testing_correction(
            daily_ic[stable_factors],
            n_tested=len(factor_frame.columns),
            alpha=cfg["correction_alpha"], method=cfg["correction_method"],
        )
        significant = sig.reindex(kept).fillna(True)
        p_adjusted = p_adj

    # ⑥ 组装粗筛卡
    candidates = [f for f in kept if bool(significant.get(f, True))]
    card_rows = []
    for f in factor_frame.columns:
        is_stable = f in stable_factors
        is_kept = f in kept
        is_redundant = f in redundant_removed
        card_rows.append({
            "factor_name": f,
            "coverage": float(quality_ok.get(f, False)),
            "nunique": int(factor_frame[f].nunique(dropna=True)) if f in factor_frame else 0,
            "quality_ok": bool(quality_ok.get(f, False)),
            "ic": float(ic.get(f, np.nan)),
            "icir": float(icir.get(f, np.nan)) if f in icir.index else np.nan,
            "win_rate": float(win_rate.get(f, np.nan)) if f in win_rate.index else np.nan,
            "p_adjusted": float(p_adjusted.get(f, np.nan)) if f in p_adjusted.index else np.nan,
            "significant": bool(significant.get(f, False)),
            "stable": is_stable,
            "redundant": is_redundant,
            "selected": (f in candidates),
        })
    screen_card = pd.DataFrame(card_rows)
    if not screen_card.empty:
        screen_card = screen_card.sort_values(["selected", "ic"],
                                              ascending=[False, False]).reset_index(drop=True)

    return {
        "screen_card": screen_card,
        "candidates": candidates,
        "ic": ic, "icir": icir, "win_rate": win_rate,
        "coverage": compute_coverage(factor_frame),
        "stable_factors": stable_factors,
        "redundant_removed": redundant_removed,
        "significant": significant,
    }
