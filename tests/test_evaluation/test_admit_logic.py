"""
test_admit_logic.py — factor_stats.py 三关准入（公共统计核心）单元测试

覆盖 [P1-1] IC 方向一致性（按年度 IC 符号判断）与
[P1-2] 增量 ICIR 边际贡献（NW 修正）及三关综合判定。

[2026-10-12] 单脚本化：run_eval.py 的 Phase 2 三关已抽入
src/qlworks/evaluation/factor_stats.py（唯一统计口径，参数化无全局副作用），
本测试直接传参注入合成数据，不依赖 qlib / parquet / 脚本全局状态。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qlworks.evaluation import factor_stats as fs  # noqa: E402

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
def window():
    """准入窗口 mock：与合成数据年份一致（2023-2025）。"""
    return {
        "sub_periods": [(str(y), f"{y}-01-01", f"{y}-12-31") for y in YEARS],
        "corr_start": "2023-01-01",
        "corr_end": "2025-12-31",
    }


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


def _kw(window):
    """从 window fixture 展开公共参数（factor_stats 参数化签名）。"""
    return {
        "start": window["corr_start"],
        "end": window["corr_end"],
        "sub_periods": window["sub_periods"],
    }


# ───────────────────────── P1-1：IC 方向一致性 ─────────────────────────

def test_direction_consistent_ic_sign(data, window):
    """三年 IC 同号（正）→ 通过。"""
    passed2, detail2 = _direction_with_series("f1", data["f1"], data["label"], window)
    assert passed2 is True
    assert "IC 方向一致" in detail2


def _direction_with_series(name, series, label_frame, window):
    """注入合成序列执行方向一致性（warehouse_root 为 None 时回退默认通过，故直接构造调用）。"""
    import qlworks.evaluation.factor_stats as _fs

    orig = _fs.load_warehouse_series
    _fs.load_warehouse_series = lambda root, n, s, e: series  # noqa: E731
    try:
        return _fs.check_direction_consistency(
            name, label_frame, **_kw(window), warehouse_root=Path("."))
    finally:
        _fs.load_warehouse_series = orig


def test_direction_inconsistent_ic_sign(data, window):
    """2023 正 / 2024-2025 负 → 拒绝。"""
    passed, detail = _direction_with_series("f2", data["f2"], data["label"], window)
    assert passed is False
    assert "IC 方向不一致" in detail


def test_direction_fallback_without_label(data, window):
    """标签缺失时回退因子值符号校验（恒正因子 → 通过）。"""
    passed, detail = _direction_with_series("f3", data["f3"], None, window)
    assert passed is True
    assert "方向一致" in detail


# ─────────────────────── P1-2：增量 ICIR 边际贡献 ───────────────────────

def _marginal_with_store(new_name, store, existing, label_frame, window):
    """注入合成 warehouse 序列执行边际贡献检验。"""
    import qlworks.evaluation.factor_stats as _fs

    orig = _fs.load_warehouse_series
    _fs.load_warehouse_series = lambda root, n, s, e: store.get(n)  # noqa: E731
    try:
        return _fs.check_marginal_contribution(
            new_name, existing, label_frame, None,
            warehouse_root=Path("."), **_kw(window))
    finally:
        _fs.load_warehouse_series = orig


def test_marginal_improves(data, window):
    """加入与 label 强相关的因子 → 组合 ICIR 提升 → 通过。"""
    store = {"f_existing": data["f1"], "f1": data["f1"]}
    existing = [{"name": "f_existing"}]
    passed, detail = _marginal_with_store("f1", store, existing, data["label"], window)
    assert passed is True
    assert "边际贡献" in detail


def test_marginal_negative(data, window):
    """加入与已有池相反信号（rank 组合后 IC 塌缩）→ 增量 ICIR 为负 → 拒绝。"""
    neg = -data["f1"]
    store = {"f_existing": data["f1"], "neg": neg}
    existing = [{"name": "f_existing"}]
    passed, detail = _marginal_with_store("neg", store, existing, data["label"], window)
    assert passed is False
    assert "边际贡献为负" in detail


def test_marginal_empty_pool(data, window):
    """候选池为空 → 自动通过。"""
    passed, detail = _marginal_with_store("f1", {"f1": data["f1"]}, [], data["label"], window)
    assert passed is True


# ───────────────────────── 三关综合判定 ─────────────────────────

def _admit_with_series(name, series, info, corr_matrix, label_frame, window):
    """注入合成序列执行三关综合判定。"""
    import qlworks.evaluation.factor_stats as _fs

    orig = _fs.load_warehouse_series
    _fs.load_warehouse_series = lambda root, n, s, e: series  # noqa: E731
    try:
        return _fs.admit_factor(
            name, info, [], corr_matrix, label_frame,
            thresholds={},
            sub_periods=window["sub_periods"],
            corr_start=window["corr_start"], corr_end=window["corr_end"],
            warehouse_root=Path("."))
    finally:
        _fs.load_warehouse_series = orig


def test_admit_factor_all_gates_pass(data, window):
    """三关全过 → admitted=True。"""
    corr_matrix = pd.DataFrame({"f1": [1.0]}, index=["f1"])
    info = {"name": "f1", "category": "test", "_tier": "satellite"}
    result = _admit_with_series("f1", data["f1"], info, corr_matrix, data["label"], window)
    assert result["admitted"] is True


def test_admit_factor_direction_reject(data, window):
    """第三关（IC 方向漂移）拒绝 → admitted=False。"""
    corr_matrix = pd.DataFrame({"f2": [1.0]}, index=["f2"])
    info = {"name": "f2", "category": "test", "_tier": "satellite"}
    result = _admit_with_series("f2", data["f2"], info, corr_matrix, data["label"], window)
    assert result["admitted"] is False
    assert result["direction_check"]["passed"] is False
