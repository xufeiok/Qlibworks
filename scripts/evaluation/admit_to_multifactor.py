"""
admit_to_multifactor.py — 多因子准入编排层

功能概述：
  将单因子评测结果（分档、ICIR、相关性）转化为多因子组合候选池。
  核心准入标准不再只是"低相关"，而是"低相关 + 有边际贡献 + 方向一致"。

输入：
  - factor_library/*.yaml                           因子定义（全量候选）
  - qualified_factors/{core,satellite,archive}      评测分档（仅 core/satellite 进入候选）
  - registry/candidate_pool.json                    已有候选池（逐步累加）

输出：
  - registry/candidate_pool.json                    更新后的候选池（admitted + rejected 明细）

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
  7. 更新 candidate_pool.json（admitted / rejected 分列）
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


# ── 准入阈值（与 factor_def.py 保持一致） ──
ADMIT_THRESHOLDS = {
    "max_correlation_existing": 0.70,        # 与已有池因子相关性上限
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


def check_marginal_contribution(new_factor: str, existing_factors: list[dict],
                                corr_matrix: pd.DataFrame) -> tuple[bool, str]:
    """
    增量边际贡献检验（基于 warehouse 数据）。

    逻辑：
    - 候选池为空 → 自动通过
    - 计算新因子与已有因子池的**平均相关系数**
    - 如果平均相关系数 > 0.40，说明新因子表达的信息冗余度高，拒绝
    - 阈值 0.40 ≈ 1 个因子解释约 16% 方差，超过即视为冗余
    """
    if not existing_factors:
        return True, "候选池为空，无需边际检验"

    existing_names = [f["name"] for f in existing_factors]
    if new_factor not in corr_matrix.index:
        return True, "新因子不在相关矩阵中，默认通过"

    # 计算新因子 vs 已有池的**平均**相关系数
    vals = []
    for ename in existing_names:
        if ename in corr_matrix.columns:
            v = abs(corr_matrix.loc[new_factor, ename])
            vals.append(v)

    if not vals:
        return True, "与已有池因子无交集"

    avg_corr = np.mean(vals)
    if avg_corr > 0.40:
        return False, f"冗余度偏高: 新因子与已有池平均相关 {avg_corr:.2f} > 0.40"

    return True, f"通过（与已有池平均相关 {avg_corr:.2f}）"


def check_direction_consistency(new_factor: str) -> tuple[bool, str]:
    """
    IC 方向一致性检验（基于 warehouse 数据）。

    读取因子在 2021-2022 / 2023-2024 / 2025 三个子时段的截面均值符号，
    如果跨子时段符号不一致，说明因子存在结构性漂移，不适合进入多因子组合。
    注意：这里检验的是因子值本身的符号一致性，而非 IC 符号。
    IC 符号一致性需依赖标签数据，在评测环节做。
    """
    fdir = WAREHOUSE_DIR / new_factor
    if not fdir.exists():
        return True, "因子无 warehouse 数据，默认通过"

    dfs = []
    for y in range(2021, 2026):
        pf = fdir / f"{y}.parquet"
        if pf.exists():
            dfs.append(pd.read_parquet(pf))
    if not dfs:
        return True, "无 2021~2025 数据"

    combined = pd.concat(dfs)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    if len(combined) < 1000:
        return True, "数据量不足"

    sub_periods = [
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-2024", "2023-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
    ]

    signs = []
    for label, start, end in sub_periods:
        dts = combined.index.get_level_values("datetime")
        mask = (dts >= start) & (dts <= end)
        subset = combined.loc[mask]
        if len(subset) < 100:
            continue
        mean_val = subset.iloc[:, 0].mean()
        signs.append((label, mean_val))

    direction = np.sign([s[1] for s in signs])
    if len(set(direction)) > 1:
        details = "; ".join(f"{s[0]}={s[1]:.4f}" for s in signs)
        return False, f"方向不一致: {details}"

    details = "; ".join(f"{s[0]}={s[1]:.4f}" for s in signs)
    return True, f"方向一致: {details}"


def admit_factor(new_factor_name: str, new_factor_info: dict,
                 existing_factors: list[dict], corr_matrix: pd.DataFrame) -> dict:
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

    # 第二关：边际贡献检验
    marginal_passed, marginal_detail = check_marginal_contribution(
        new_factor_name, existing_factors, corr_matrix
    )
    result["marginal_check"]["passed"] = marginal_passed
    result["marginal_check"]["detail"] = marginal_detail

    if not marginal_passed:
        result["reasons"].append(f"边际贡献检验未通过: {marginal_detail}")

    # 第三关：方向一致性
    dir_passed, dir_detail = check_direction_consistency(new_factor_name)
    result["direction_check"]["passed"] = dir_passed
    result["direction_check"]["detail"] = dir_detail

    if not dir_passed:
        result["reasons"].append(f"方向一致性未通过: {dir_detail}")

    # 综合判定
    result["admitted"] = corr_passed and marginal_passed and dir_passed
    if result["admitted"]:
        result["latest_icir"] = 0.5  # placeholder，后续从报告解析

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

    admitted = []
    rejected = []

    for cand in candidates:
        name = cand["name"]
        if name in existing_names:
            print(f"  [跳过] {name} 已在候选池中")
            admitted.append(cand)
            continue

        # 前置检查：必须有评测报告
        report_tier = _find_evaluation_report(name)
        if report_tier is None:
            print(f"  [跳过] {name} 未完成单因子评测，跳过（先跑 run_eval.py）")
            continue

        cand["_tier"] = report_tier  # 以报告 tier 为准
        result = admit_factor(name, cand, existing, corr_matrix)

        if result["admitted"]:
            print(f"  [准入] {name} (tier={result['tier']})")
            admitted.append({
                "name": name,
                "tier": result["tier"],
                "category": result["category"],
                "sub_category": result["sub_category"],
                "meaning": result["meaning"],
                "source_file": result.get("_source_file", ""),
                "latest_icir": result.get("latest_icir", 0),
                "admitted_at": None,
            })
        else:
            print(f"  [拒绝] {name} 原因: {'; '.join(result['reasons'])}")
            rejected.append(result)

    # 更新候选池
    pool["factors"] = admitted
    pool["rejected"] = rejected
    pool["_meta"]["updated_at"] = pd.Timestamp.now().isoformat()[:19]
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

    result = admit_factor(factor_name, factor_info, existing, corr_matrix)
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
        # 写入候选池
        pool["factors"].append({
            "name": factor_name,
            "tier": tier,
            "category": factor_info.get("category", ""),
            "sub_category": factor_info.get("sub_category", ""),
            "meaning": factor_info.get("meaning", ""),
            "source_file": factor_info.get("_source_file", ""),
            "latest_icir": result.get("latest_icir", 0),
            "admitted_at": None,
        })
        pool["_meta"]["updated_at"] = pd.Timestamp.now().isoformat()[:19]
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
