"""
test_admit_logic.py — admit_to_multifactor 三关检验单元测试

覆盖 [P1-1] IC 方向一致性（原实现检验因子值符号，与 IC 符号无关）与
[P1-2] 增量 ICIR 边际贡献（原实现以平均相关近似）。通过 monkeypatch
_load_warehouse_series 注入合成数据，不依赖 qlib / parquet。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# admit_to_multifactor 位于 scripts/evaluation/，非包，需手动加入 path
_SCRIPTS_EVAL = Path(__file__).resolve().parents[2] / "scripts" / "evaluation"
if str(_SCRIPTS_EVAL) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_EVAL))

import admit_to_multifactor as admit  # noqa: E402

N_STOCKS = 60
YEARS = (2023, 2024, 2025)


def _build_dates():
    dates = []
    for y in YEARS:
        dates.extend(pd.bdate_range(f"{y}-01-01", f"{y}-12-31"))
    return dates


def _mi(dates):
    tuples = []
    for d in dates:
        for i in range(N_STOCKS):
            tuples.append((f"sh{i:06d}", d))
    return pd.MultiIndex.from_tuples(tuples, names=["instrument", "datetime"])


@pytest.fixture
def data():
    """合成面板：label 与 score 正相关；f1 三年 IC 同号；f2 方向漂移。"""
    rng = np.random.default_rng(42)
    dates = _build_dates()
    mi = _mi(dates)
    n_total = len(dates) * N_STOCKS
    score = np.linspace(-1.0, 1.0, N_STOCKS)

    def series(vals):
        return pd.Series(vals, index=mi)

    # f1：三年与 score 正相关 → 各年度 IC 均为正
    f1 = np.tile(score, len(dates)) + 0.2 * rng.standard_normal(n_total)
    # f2：2023 正相关，2024/2025 负相关 → IC 方向漂移
    f2_vals = []
    for y in YEARS:
        sign = 1.0 if y == 2023 else -1.0
        for _ in pd.bdate_range(f"{y}-01-01", f"{y}-12-31"):
            f2_vals.extend(sign * score + 0.2 * rng.standard_normal(N_STOCKS))
    # f3：恒正（|score|+1），用于无标签回退路径
    f3 = np.tile(np.abs(score) + 1.0, len(dates)) + 0.2 * rng.standard_normal(n_total)
    # label：与 score 正相关
    label = np.tile(score, len(dates)) + 0.5 * rng.standard_normal(n_total)

    return {
        "f1": series(f1),
        "f2": series(f2_vals),
        "f3": series(f3),
        "label": series(label).rename("LABEL_5D"),
    }


# ───────────────────────── P1-1：IC 方向一致性 ─────────────────────────

def test_direction_consistent_ic_sign(data, monkeypatch):
    """三年 IC 同号（正）→ 通过。"""
    monkeypatch.setattr(admit, "_load_warehouse_series", lambda name: data["f1"])
    passed, detail = admit.check_direction_consistency("f1", data["label"])
    assert passed is True
    assert "IC 方向一致" in detail


def test_direction_inconsistent_ic_sign(data, monkeypatch):
    """2023 正 / 2024-2025 负 → 拒绝。"""
    monkeypatch.setattr(admit, "_load_warehouse_series", lambda name: data["f2"])
    passed, detail = admit.check_direction_consistency("f2", data["label"])
    assert passed is False
    assert "IC 方向不一致" in detail


def test_direction_fallback_without_label(data, monkeypatch):
    """标签缺失时回退因子值符号校验（恒正因子 → 通过）。"""
    monkeypatch.setattr(admit, "_load_warehouse_series", lambda name: data["f3"])
    passed, detail = admit.check_direction_consistency("f3", label_frame=None)
    assert passed is True
    assert "方向一致" in detail


# ─────────────────────── P1-2：增量 ICIR 边际贡献 ───────────────────────

def test_marginal_improves(data, monkeypatch):
    """加入与 label 强相关的因子 → 组合 ICIR 提升 → 通过。"""
    store = {"f_existing": data["f1"], "f1": data["f1"]}
    monkeypatch.setattr(admit, "_load_warehouse_series", lambda name: store.get(name))
    existing = [{"name": "f_existing"}]
    passed, detail = admit.check_marginal_contribution("f1", existing, label_frame=data["label"])
    assert passed is True
    assert "边际贡献" in detail


def test_marginal_negative(data, monkeypatch):
    """加入与已有池相反信号（rank 组合后 IC 塌缩）→ 增量 ICIR 为负 → 拒绝。"""
    neg = -data["f1"]
    store = {"f_existing": data["f1"], "neg": neg}
    monkeypatch.setattr(admit, "_load_warehouse_series", lambda name: store.get(name))
    existing = [{"name": "f_existing"}]
    passed, detail = admit.check_marginal_contribution("neg", existing, label_frame=data["label"])
    assert passed is False
    assert "边际贡献为负" in detail


def test_marginal_empty_pool(data):
    """候选池为空 → 自动通过。"""
    passed, detail = admit.check_marginal_contribution("f1", [], label_frame=data["label"])
    assert passed is True


# ───────────────────────── 三关综合判定 ─────────────────────────

def test_admit_factor_all_gates_pass(data, monkeypatch):
    """三关全过 → admitted=True。"""
    monkeypatch.setattr(admit, "_load_warehouse_series", lambda name: data["f1"])
    corr_matrix = pd.DataFrame({"f1": [1.0]}, index=["f1"])
    info = {"name": "f1", "category": "test", "_tier": "satellite"}
    result = admit.admit_factor("f1", info, [], corr_matrix, label_frame=data["label"])
    assert result["admitted"] is True


def test_admit_factor_direction_reject(data, monkeypatch):
    """第三关（IC 方向漂移）拒绝 → admitted=False。"""
    monkeypatch.setattr(admit, "_load_warehouse_series", lambda name: data["f2"])
    corr_matrix = pd.DataFrame({"f2": [1.0]}, index=["f2"])
    info = {"name": "f2", "category": "test", "_tier": "satellite"}
    result = admit.admit_factor("f2", info, [], corr_matrix, label_frame=data["label"])
    assert result["admitted"] is False
    assert result["direction_check"]["passed"] is False
