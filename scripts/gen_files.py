"""
评测全局配置：所有可调参数集中管理，便于实验对比。

支持三级准入门槛 + 生命周期配置 + 监控告警阈值。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalConfig:
    # ── 基础数据 ──
    instruments: str = "csi500"
    start_time: str = "2018-01-01"
    end_time: str = "2025-12-31"
    freq: str = "day"

    # ── 数据切分 ──
    train_end: str = "2023-12-31"
    valid_end: str = "2024-12-31"

    # ── 收益率标签 ──
    label_expr: str = "Ref($close, -5) / Ref($open, -1) - 1"
    label_name: str = "LABEL_5D"

    # ── 预处理 ──
    winsorize_method: str = "mad"
    winsorize_threshold: float = 5.0
    standardize_method: str = "zscore"
    neutralization: str = "industry_market"

    # ── IC 分析 ──
    ic_method: str = "spearman"
    ic_annual_factor: float = 252.0

    # ── 分层回测 ──
    quantiles: int = 5
    long_short_quantiles: tuple = field(default_factory=lambda: (0, 2))

    # ── 筛选标准（三级准入标准 — IC/预测能力级）──
    # 参考个人量化标准：
    #   IC 均值 > 0.05（强因子 > 0.08）
    #   IR > 1.0（顶级 > 1.5）
    #   夏普 > 1.25
    #   多空年化 > 15%
    ic_threshold: float = 0.05
    icir_threshold: float = 1.0
    win_rate_threshold: float = 0.65
    ls_annual_return_threshold: float = 0.15
    ls_sharpe_threshold: float = 1.25

    # ── 生命周期配置 ──
    enable_lifecycle: bool = True
    registry_dir: str = ""
    factor_library_dir: str = ""

    # ── 监控告警阈值 ──
    monitor_freq: str = "month"
    monitor_ic_warning: float = 0.03
    monitor_ic_danger: float = 0.0
    monitor_consecutive_bad: int = 3

    # ── 输出 ──
    qualified_dir: str = ""
    report_dir: str = ""

    # ── 稳健性检验 ──
    robustness_sub_periods: list = field(default_factory=lambda: [])
    robustness_sub_pools: list = field(default_factory=lambda: [])
    robustness_ls_cost: float = 0.001


_BASE_DIR = Path(__file__).parent
_ROOT_DIR = _BASE_DIR.parent

DEFAULT_CONFIG = EvalConfig(
    qualified_dir=str(_BASE_DIR / "qualified_factors"),
    report_dir=str(_BASE_DIR / "reports"),
    registry_dir=str(_ROOT_DIR / "factor_registry"),
    factor_library_dir=str(_ROOT_DIR / "factor_library"),
    robustness_sub_periods=[
        ("2018-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
        ("2022-01-01", "2025-12-31"),
    ],
    robustness_sub_pools=["csi300", "csi500", "csi800"],
)
