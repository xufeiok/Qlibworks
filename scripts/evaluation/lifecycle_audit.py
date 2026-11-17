# -*- coding: utf-8 -*-
"""
lifecycle_audit.py — 因子生命周期自动退役（P2-2，计划文档 §5 P2-2）

输入:
  - factor_data/warehouse/<name>/data.parquet   因子值（registry 全因子 + sentiment 新因子）
  - qlib 标签（LABEL_EXPR=Ref($close,-5)/Ref($open,-1)-1，main_board 池）
  - factor_data/registry/registry.json          family_map + 当前 lifecycle_stage
  - factor_data/registry/lifecycle_log.json     生命周期事件日志（追加）

状态机（滚动 26 周 ICIR，年化口径 mean/std*√52）:
  【强度判定取 |ICIR| —— 方向无关】负向稳定因子（如短期反转 ICIR -2.3）强度高，
  不得因符号为负被误归档；方向反转由下方独立告警处理。
  - 连续 8 周  |rolling_26w_icir| < 0.35 → 降档 satellite（core→satellite，不再参与 core 训练）
  - 连续 12 周 |rolling_26w_icir| < 0.20 → 归档 archive（不再进任何训练；候选池标记 retired）
  - 连续 12 周 |rolling_26w_icir| ≥ 0.50 → 恢复 core（已降档因子验证回升）
  - 方向反转：滚动 26 周 IC 均值符号与主符号相反连续 12 周 → alert（记录，不自动翻转方向）

输出:
  - factor_data/warehouse/lifecycle_monitor.csv  全因子最新状态快照
  - registry.json factors[].lifecycle_stage      同步更新（active/satellite/archive）
  - registry/lifecycle_log.json                  事件追加（downgrade/archive/promote/direction_alert）

用法:
  python scripts/evaluation/lifecycle_audit.py [--start 2023-01-01] [--end 2026-10-09]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE_DIR = PROJECT_ROOT / "factor_data" / "warehouse"
REGISTRY_DIR = PROJECT_ROOT / "factor_data" / "registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"
LIFECYCLE_LOG_PATH = REGISTRY_DIR / "lifecycle_log.json"
MONITOR_PATH = WAREHOUSE_DIR / "lifecycle_monitor.csv"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from qlworks.pipeline_config import LABEL_EXPR, LABEL_NAME, INSTRUMENTS  # noqa: E402

# ── 状态机参数（对齐计划 §5 P2-2） ──
ICIR_WINDOW_WEEKS = 26
DOWNGRADE_ICIR = 0.35        # 连续 DOWNGRADE_WEEKS 周 < 0.35 → satellite
DOWNGRADE_WEEKS = 8
ARCHIVE_ICIR = 0.20          # 连续 ARCHIVE_WEEKS 周 < 0.20 → archive
ARCHIVE_WEEKS = 12
PROMOTE_ICIR = 0.50          # 连续 PROMOTE_WEEKS 周 >= 0.50 → 恢复 core
PROMOTE_WEEKS = 12
ALERT_WEEKS = 12             # 方向反转连续周数 → alert
MIN_CROSS = 30               # 单日截面最小样本数
MIN_WEEKS = 26               # 参与判定的最少周数


def load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(reg):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def load_lifecycle_log():
    if LIFECYCLE_LOG_PATH.exists():
        with open(LIFECYCLE_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"events": [], "last_updated": None}


def save_lifecycle_log(log):
    log["last_updated"] = datetime.now().isoformat(timespec="seconds")
    with open(LIFECYCLE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def load_label_frame(start: str, end: str) -> pd.Series | None:
    """qlib 标签（与 admit_to_multifactor 同一口径，instrument 小写）"""
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.config import QLIB_DATA_DIR
        from qlworks.factors.filter_utils import filter_untradeable_labels

        qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)
        ins_file = Path(str(QLIB_DATA_DIR)) / "instruments" / f"{INSTRUMENTS}.txt"
        if not ins_file.exists():
            print(f"[audit] 股票池 {INSTRUMENTS}.txt 不存在")
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
        raw = filter_untradeable_labels(raw, instruments, start, end)
        if raw is None or raw.empty:
            return None
        flat = raw.reset_index()
        flat["instrument"] = flat["instrument"].str.lower()
        return flat.set_index(["instrument", "datetime"]).sort_index()[LABEL_NAME]
    except Exception as e:
        print(f"[audit] 标签加载失败({e})")
        return None


def load_factor_series(name: str, start: str, end: str) -> pd.Series | None:
    """warehouse 因子值（instrument 统一小写，窗口切片）"""
    fdir = WAREHOUSE_DIR / name
    if not fdir.exists():
        return None
    df = pd.read_parquet(fdir / "data.parquet")
    df = df[~df.index.duplicated(keep="first")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return None
    s = df.iloc[:, 0]
    s.index = s.index.set_levels(
        [lv.str.lower() if lv.name == "instrument" else lv for lv in s.index.levels],
    )
    dts = s.index.get_level_values("datetime")
    s = s[(dts >= start) & (dts <= end)]
    return s if len(s) > 0 else None


def weekly_ic_series(factor_s: pd.Series, label_s: pd.Series, min_cross: int = MIN_CROSS) -> pd.Series:
    """周频 IC（每周最后一个交易日截面的 Spearman IC，与筛选端周频面板口径一致）"""
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
            continue
        ic = f_rank.corr(y_rank)
        if pd.notna(ic):
            daily[dt] = ic
    ds = pd.Series(daily, dtype=float)
    if ds.empty:
        return ds
    # 周频聚合：取每周最后交易日 IC（周内最后一截面）
    ds.index = pd.to_datetime(ds.index)
    weekly = ds.groupby(pd.Grouper(freq="W-FRI")).last().dropna()
    return weekly


def rolling_icir(weekly_ic: pd.Series, window: int = ICIR_WINDOW_WEEKS) -> pd.Series:
    """滚动 26 周 ICIR（年化：mean/std*√52）；std<eps 时记 0"""
    def _icir(x):
        if len(x) < 10:
            return np.nan
        std = x.std(ddof=1)
        if std <= 1e-12:
            return 0.0
        return x.mean() / std * np.sqrt(52.0)
    return weekly_ic.rolling(window, min_periods=10).apply(_icir, raw=True)


def consecutive_below(seq: pd.Series, threshold: float, n: int) -> bool:
    """seq 末尾是否连续 n 期低于阈值（当前周为最近一期）"""
    if len(seq) < n:
        return False
    return bool((seq.iloc[-n:] < threshold).all())


def consecutive_above(seq: pd.Series, threshold: float, n: int) -> bool:
    if len(seq) < n:
        return False
    return bool((seq.iloc[-n:] >= threshold).all())


def reset_stages():
    """重置 registry 全部 lifecycle_stage 为 active + 清空 lifecycle_log（用于修复后幂等重跑）"""
    reg = load_registry()
    for f in reg.get("factors", []):
        f["lifecycle_stage"] = "active"
        f.pop("lifecycle_updated_at", None)
    save_registry(reg)
    log = {"events": [], "last_updated": datetime.now().isoformat(timespec="seconds")}
    save_lifecycle_log(log)
    if MONITOR_PATH.exists():
        MONITOR_PATH.unlink()
    print("[audit] 已重置 registry.lifecycle_stage → active，清空 lifecycle_log 与 monitor")


def run_audit(start: str, end: str):
    print(f"[audit] 生命周期审计窗口: {start} ~ {end}")
    reg = load_registry()
    log = load_lifecycle_log()
    family_map = reg.get("family_map", {})

    # 因子全集：registry family_map + sentiment 新因子（扫描 warehouse 补充未注册因子）
    factor_names = []
    for fam, names in family_map.items():
        for n in names:
            if n not in factor_names:
                factor_names.append(n)
    # 补充未注册 warehouse 因子（sentiment 等），标记 family=unknown
    for d in sorted(WAREHOUSE_DIR.iterdir()):
        if d.is_dir() and (d / "data.parquet").exists() and d.name not in factor_names:
            factor_names.append(d.name)
    print(f"[audit] 监控因子: {len(factor_names)} 个")

    label_s = load_label_frame(start, end)
    if label_s is None:
        print("[audit] 标签不可用，终止")
        return

    # 现注册表 stage 索引
    stage_by_name = {f.get("name"): f.get("lifecycle_stage", "active") for f in reg.get("factors", [])}

    rows = []
    events = []
    for i, name in enumerate(factor_names):
        fam = next((k for k, v in family_map.items() if name in v), "unknown")
        fs = load_factor_series(name, start, end)
        if fs is None or len(fs) < MIN_WEEKS:
            continue
        wic = weekly_ic_series(fs, label_s)
        if len(wic) < MIN_WEEKS:
            continue
        r_icir = rolling_icir(wic)
        if r_icir.dropna().empty:
            continue

        cur_stage = stage_by_name.get(name, "active")
        new_stage = cur_stage
        last_icir = float(r_icir.iloc[-1]) if pd.notna(r_icir.iloc[-1]) else 0.0
        last_icir_abs = abs(last_icir)
        last_ic = float(wic.iloc[-1]) if pd.notna(wic.iloc[-1]) else 0.0
        decision = ""
        event = None

        # 主符号：最近 52 周 IC 均值符号（与 registry direction 对齐的可复核口径）
        main_sign = np.sign(wic.iloc[-52:].mean()) if len(wic) >= 52 else np.sign(last_ic)
        recent_sign = np.sign(wic.iloc[-ALERT_WEEKS:].mean())

        # 强度判定统一取 |ICIR|（负向稳定因子不被误杀）
        if cur_stage != "archive" and consecutive_below(r_icir.abs(), ARCHIVE_ICIR, ARCHIVE_WEEKS):
            if new_stage != "archive":
                new_stage = "archive"
                decision = f"连续{ARCHIVE_WEEKS}周|滚动ICIR|<{ARCHIVE_ICIR:.2f} → archive"
                event = {"type": "archive", "factor": name, "family": fam, "at": str(wic.index[-1].date()),
                         "rolling_icir": round(last_icir, 3), "detail": decision, "from_stage": cur_stage}
        elif cur_stage != "satellite" and cur_stage != "archive" and consecutive_below(r_icir.abs(), DOWNGRADE_ICIR, DOWNGRADE_WEEKS):
            new_stage = "satellite"
            decision = f"连续{DOWNGRADE_WEEKS}周|滚动ICIR|<{DOWNGRADE_ICIR:.2f} → satellite"
            event = {"type": "downgrade", "factor": name, "family": fam, "at": str(wic.index[-1].date()),
                     "rolling_icir": round(last_icir, 3), "detail": decision, "from_stage": cur_stage}
        elif cur_stage == "satellite" and consecutive_above(r_icir.abs(), PROMOTE_ICIR, PROMOTE_WEEKS):
            new_stage = "core"
            decision = f"连续{PROMOTE_WEEKS}周|滚动ICIR|>={PROMOTE_ICIR:.2f} → core（恢复）"
            event = {"type": "promote", "factor": name, "family": fam, "at": str(wic.index[-1].date()),
                     "rolling_icir": round(last_icir, 3), "detail": decision, "from_stage": cur_stage}
        elif main_sign != 0 and recent_sign != 0 and main_sign != recent_sign and len(wic) >= 52:
            decision = f"方向反转告警: 52周主符号{'+' if main_sign > 0 else '-'} vs 最近{ALERT_WEEKS}周{'+' if recent_sign > 0 else '-'}"
            event = {"type": "direction_alert", "factor": name, "family": fam, "at": str(wic.index[-1].date()),
                     "rolling_icir": round(last_icir, 3), "detail": decision}

        rows.append({
            "factor": name, "family": fam, "current_stage": new_stage,
            "latest_rolling_icir_26w": round(last_icir, 3),
            "latest_weekly_ic": round(last_ic, 4),
            "weeks_evaluated": len(wic),
            "last_decision": decision or "-", "decision_date": str(wic.index[-1].date()),
        })
        if event:
            events.append(event)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(factor_names)}")

    monitor = pd.DataFrame(rows).sort_values(["current_stage", "family", "factor"])
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    monitor.to_csv(MONITOR_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[audit] monitor 已保存: {MONITOR_PATH} ({len(monitor)} 因子)")

    # 更新 registry lifecycle_stage + 补注册未入库因子（sentiment 等）
    stage_map = dict(zip(monitor["factor"], monitor["current_stage"]))
    family_map_local = {k: set(v) for k, v in family_map.items()}
    reg_names = {f.get("name") for f in reg.get("factors", [])}
    updated = 0
    added = 0
    for f in reg.get("factors", []):
        ns = stage_map.get(f.get("name"))
        if ns and ns != f.get("lifecycle_stage"):
            f["lifecycle_stage"] = ns
            f["lifecycle_updated_at"] = datetime.now().isoformat(timespec="seconds")
            updated += 1
    # 补注册：monitor 中有但 registry 无的因子（sentiment 新因子）
    for name, ns in stage_map.items():
        if name in reg_names:
            continue
        fam = next((k for k, v in family_map_local.items() if name in v), "sentiment")
        reg.setdefault("factors", []).append({
            "name": name, "family": fam, "category": "情绪 (Sentiment)",
            "source_file": "sentiment_factors.yaml", "lifecycle_stage": ns,
            "registered_by": "lifecycle_audit", "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        family_map.setdefault(fam, []).append(name)
        added += 1
        print(f"  [补注册] {name} → {fam} ({ns})")
    save_registry(reg)
    print(f"[audit] registry.lifecycle_stage 更新: {updated} 个；补注册新因子: {added} 个")

    # 事件追加
    log["events"].extend(events)
    save_lifecycle_log(log)
    print(f"[audit] 事件: {len(events)} 条（downgrade/archive/promote/direction_alert）已追加 lifecycle_log.json")

    # 汇总
    print("\n=== 生命周期汇总 ===")
    if monitor.empty:
        print("  无因子进入审计")
        return
    print(monitor.groupby("current_stage").size().to_string())
    changed = monitor[monitor["last_decision"] != "-"]
    if len(changed):
        print("\n=== 本次触发事件因子 ===")
        print(changed[["factor", "family", "current_stage", "latest_rolling_icir_26w", "last_decision"]].to_string(index=False))
    else:
        print("  本次无状态变更")


def _parse_args():
    p = argparse.ArgumentParser(description="因子生命周期自动退役审计")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-10-09")
    p.add_argument("--reset-stages", action="store_true",
                   help="先重置 registry.lifecycle_stage 为 active 并清空日志（修复后幂等重跑）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.reset_stages:
        reset_stages()
    run_audit(args.start, args.end)
