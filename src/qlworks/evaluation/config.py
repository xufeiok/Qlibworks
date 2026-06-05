"""评测全局配置：所有可调参数集中管理，便于实验对比。支持三级准入门槛 + 生命周期配置 + 监控告警阈值 + A 股交易约束。全周期覆盖：2010-01-01 ~ 2026-12-31"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple


def _parse_label_horizon(label_expr: str) -> int:
    m = re.search(r"Ref\(\$\w+,\s*(-\d+)\)", label_expr)
    if m:
        return abs(int(m.group(1)))
    return 5


EXTREME_EVENTS = {
    "2015_crash": ("2015-06-15", "2015-09-15"),
    "2016_meltdown": ("2016-01-04", "2016-01-28"),
    "2018_trade_war": ("2018-03-23", "2018-10-31"),
    "2020_covid": ("2020-02-03", "2020-03-23"),
    "2024_small_cap_crisis": ("2024-01-02", "2024-02-08"),
}


@dataclass
class EvalConfig:
    instruments: str = "csi500"
    start_time: str = "2010-01-01"
    end_time: str = "2026-12-31"
    freq: str = "day"

    train_end: str = "2024-12-31"
    valid_end: str = "2025-12-31"

    label_expr: str = "Ref($close, -5) / Ref($open, -1) - 1"
    label_name: str = "LABEL_5D"
    label_horizon: int = 0

    winsorize_method: str = "mad"
    winsorize_threshold: float = 5.0
    standardize_method: str = "zscore"
    neutralization: str = "industry_market"

    limit_up_pct: float = 0.095
    limit_down_pct: float = -0.095
    limit_up_exit_rule: str = "next_open"
    limit_down_exit_rule: str = "next_open"
    filter_suspended: bool = True
    suspended_volume_threshold: float = 0.0

    slippage_entry_bps: float = 10.0
    slippage_exit_bps: float = 10.0
    market_impact_bps: float = 5.0

    ic_method: str = "spearman"
    ic_annual_factor: float = 252.0

    quantiles: int = 5
    long_short_quantiles: tuple = field(default_factory=lambda: (0, 2))

    ic_threshold: float = 0.02
    icir_threshold: float = 0.4
    win_rate_threshold: float = 0.55
    ls_annual_return_threshold: float = 0.05
    ls_sharpe_threshold: float = 0.3
    satellite_composite_min: float = 40.0

    enable_fama_macbeth: bool = True
    fm_standard_errors: str = "newey_west"
    fm_nw_lags: int = 4

    enable_walk_forward: bool = True
    wf_train_months: int = 36
    wf_valid_months: int = 12
    wf_step_months: int = 12

    enable_bootstrap: bool = True
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95

    enable_capacity_analysis: bool = True
    capacity_aum_levels: list = field(default_factory=lambda: [1e8, 5e8, 1e9, 5e9, 1e10])

    enable_extreme_stress: bool = True
    extreme_event_names: list = field(default_factory=lambda: [
        "2015_crash", "2016_meltdown", "2018_trade_war",
        "2020_covid", "2024_small_cap_crisis",
    ])

    enable_ff_attribution: bool = True
    ff_model: str = "ff5"

    enable_lifecycle: bool = True
    registry_dir: str = ""
    factor_library_dir: str = ""

    monitor_freq: str = "month"
    monitor_ic_warning: float = 0.02
    monitor_ic_danger: float = 0.0
    monitor_consecutive_bad: int = 3

    warehouse_dir: str = ""
    factors_dir: str = ""
    cache_dir: str = ""
    report_dir: str = ""

    robustness_sub_periods: list = field(default_factory=lambda: [])
    robustness_sub_pools: list = field(default_factory=lambda: [])
    robustness_ls_cost: float = 0.001

    def __post_init__(self):
        if self.label_horizon == 0:
            self.label_horizon = _parse_label_horizon(self.label_expr)


_BASE_DIR = Path(__file__).resolve().parent
_QLWORKS_DIR = _BASE_DIR.parent
_PROJECT_ROOT = _QLWORKS_DIR.parent.parent

DEFAULT_CONFIG = EvalConfig(
    warehouse_dir=str(_PROJECT_ROOT / "factor_data" / "warehouse"),
    factors_dir=str(_PROJECT_ROOT / "factor_data" / "qualified_factors"),
    cache_dir=str(_PROJECT_ROOT / "factor_data" / "cache"),
    report_dir=str(_PROJECT_ROOT / "factor_data" / "reports"),
    registry_dir=str(_PROJECT_ROOT / "factor_data" / "registry"),
    factor_library_dir=str(_PROJECT_ROOT / "factor_data" / "factor_library"),
    robustness_sub_periods=[
        ("2010-01-01", "2012-12-31"),
        ("2013-01-01", "2015-12-31"),
        ("2016-01-01", "2018-12-31"),
        ("2019-01-01", "2021-12-31"),
        ("2022-01-01", "2026-12-31"),
    ],
    robustness_sub_pools=["csi300", "csi500", "csi800"],
)
