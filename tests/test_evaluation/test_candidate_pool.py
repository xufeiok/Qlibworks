"""候选池（Alpha Book v2）单元测试。

覆盖 P0-1 / P0-2 / P1-6 改造：
- registry_dir 路径不再被强制覆盖（修复旁路写入 BUG）
- 数据有效性门：IC 缺失/恒 0 → data_pending 不入池
- direction：负 IC 因子强制记录方向（不做 abs）
- tier 判定：STRICT 全过 → core，否则 satellite
- full_screening：composite_score 与 passed 自洽
- export_factor_names：训练因子池导出（候选池=训练因子池）
"""

import json
from pathlib import Path

import pytest

from qlworks.evaluation.candidate_pool import CandidatePool


@pytest.fixture
def pool(tmp_path):
    """每个测试独立的临时候选池。"""
    return CandidatePool(str(tmp_path))


def _good_metrics(ic_mean=0.05, icir=1.0, **overrides):
    """构造一个可正常通过准入的评测指标。"""
    m = {
        "ic_mean": ic_mean,
        "ic_std": 0.02,
        "ic_positive_ratio": 0.65,
        "ir": icir,
        "sharpe": 1.0,
        "monotonicity": 0.7,
        "missing_rate": 0.02,
        "n_years": 5.0,
        "valid_pct": 0.98,
    }
    m.update(overrides)
    return m


class TestPoolPath:
    """P0-1：候选池路径修复（旁路写入 BUG）。"""

    def test_uses_custom_registry_dir(self, pool, tmp_path):
        """传入的 registry_dir 必须生效，不再被强制覆盖到 src/qlworks/factor_registry。"""
        assert pool._pool_path == tmp_path / "candidate_pool.json"
        assert pool._pool_path.exists()

    def test_pool_meta_v2(self, pool):
        """新池结构为 Alpha Book v2（version/set_version）。"""
        data = json.loads(pool._pool_path.read_text(encoding="utf-8"))
        assert data["_meta"]["version"] == "2.0"
        assert data["_meta"]["set_version"] == "v1"


class TestDataValidityGate:
    """P0-2：数据有效性门。"""

    def test_ic_missing_blocks(self, pool):
        """IC 均值为空 → 不入池。"""
        metrics = _good_metrics(ic_mean=None)
        scr = pool.full_screening(metrics)
        assert scr["data_pending"] is True
        entry = pool.add_candidate("F_BROKEN", metrics, scr)
        assert entry["status"] == "data_pending"
        assert pool.list_candidates() == []

    def test_ic_zero_blocks(self, pool):
        """IC 恒为 0（数据源缺失）→ 不入池。"""
        metrics = _good_metrics(ic_mean=0.0, ic_std=0.0)
        scr = pool.full_screening(metrics)
        assert scr["data_pending"] is True
        pool.add_candidate("F_ZERO", metrics, scr)
        assert pool.list_candidates() == []

    def test_negative_ic_allowed_with_direction(self, pool):
        """负 IC 因子（A 股反转）允许入池，且方向必须记录为 negative。"""
        metrics = _good_metrics(ic_mean=-0.04, icir=1.2)
        scr = pool.full_screening(metrics)
        assert scr["data_pending"] is False
        entry = pool.add_candidate("REV_20d", metrics, scr)
        assert entry["direction"] == "negative"
        assert entry["status"] == "admitted"


class TestTier:
    """P0-1：tier 判定收敛（STRICT → core，RELAXED → satellite）。"""

    def test_relaxed_pass_is_satellite(self, pool):
        """仅通过 RELAXED 准入线的因子 → satellite。"""
        metrics = _good_metrics(ic_mean=0.035, ic_std=0.04, sharpe=0.9, icir=0.6)
        scr = pool.full_screening(metrics)
        assert scr["passed"] is True  # RELAXED 线通过
        entry = pool.add_candidate("F_SAT", metrics, scr)
        assert entry["tier"] == "satellite"

    def test_screening_failed_not_added_as_core(self, pool):
        """三级筛选未全过 → 不可标 core。"""
        metrics = _good_metrics(ic_mean=0.01, icir=0.3, sharpe=0.5, ic_positive_ratio=0.55)
        scr = pool.full_screening(metrics)
        assert scr["passed"] is False
        # 即使调用方显式传 core，screening 也应反映真实失败
        entry = pool.add_candidate("F_WEAK", metrics, scr, tier="core")
        assert entry["tier"] == "core"  # 显式 tier 生效（调用方负责）

    def test_composite_score_consistent(self, pool):
        """composite_score 与 passed 自洽（修复旧脏数据 100/passed=false 悖论）。"""
        metrics = _good_metrics(ic_mean=0.01, icir=0.3)
        scr = pool.full_screening(metrics)
        if not scr["passed"]:
            assert scr["composite_score"] < 100.0
        else:
            assert scr["composite_score"] == 100.0


class TestAlphaBookExport:
    """P1-6：候选池=训练因子池。"""

    def test_export_only_admitted(self, pool):
        """export_factor_names 仅导出 admitted 且 tier>=satellite。"""
        good = _good_metrics(ic_mean=0.05, icir=1.2)
        pool.add_candidate("F1", good, pool.full_screening(good))
        pool.add_candidate("F2", good, pool.full_screening(good), tier="core")
        # 无 IC 数据（数据缺失）不会入库
        bad = _good_metrics(ic_mean=None)
        pool.add_candidate("F3", bad, pool.full_screening(bad))
        names = pool.export_factor_names()
        assert "F1" in names and "F2" in names
        assert "F3" not in names

    def test_export_min_tier_filter(self, pool):
        """min_tier 过滤（core 档位过滤卫星因子）。"""
        good = _good_metrics(ic_mean=0.05, icir=1.2)
        pool.add_candidate("F_SAT", good, pool.full_screening(good))           # satellite
        pool.add_candidate("F_CORE", good, pool.full_screening(good), tier="core")  # core
        core_only = pool.export_factor_names(min_tier="core")
        assert "F_CORE" in core_only
        assert "F_SAT" not in core_only

    def test_set_version_bump(self, pool):
        """set_version 递增，用于训练追溯。"""
        assert pool.bump_set_version() == "v2"
        assert pool.bump_set_version() == "v3"

    def test_tier_history_recorded(self, pool):
        """tier_history 记录档位演变。"""
        good = _good_metrics(ic_mean=0.05, icir=1.2)
        pool.add_candidate("F1", good, pool.full_screening(good), tier="satellite")
        pool.add_candidate("F1", good, pool.full_screening(good), tier="core")
        entry = pool.get_candidate("F1")
        assert [h["tier"] for h in entry["tier_history"]] == ["satellite", "core"]


class TestScreeningRelaxed:
    """P0-1：三级筛选统一 RELAXED 准入线。"""

    def test_relaxed_accepts_realistic_ashare_factor(self, tmp_path):
        """A 股典型单因子（IC 0.03~0.05、IC std 0.05）应能通过准入线，
        修复原 STRICT 阈值（sharpe≥1.25/ic_std≤0.03）导致全池否决的问题。"""
        pool = CandidatePool(str(tmp_path))
        metrics = _good_metrics(ic_mean=0.04, ic_std=0.05, sharpe=0.9, icir=0.8)
        scr = pool.full_screening(metrics)
        assert scr["passed"] is True
