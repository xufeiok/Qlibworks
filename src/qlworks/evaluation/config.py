"""
评测全局配置：所有可调参数集中管理，便于实验对比。

支持三级准入门槛 + 生命周期配置 + 监控告警阈值。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _parse_label_horizon(label_expr: str) -> int:
    """从标签表达式自动解析持有期（交易日）。

    解析规则：从 Ref($close, -N) 中提取 N。
    例如 "Ref($close, -5) / Ref($open, -1) - 1" → 5
    如果无法解析则默认返回 5。
    """
    m = re.search(r'Ref\(\$\w+,\s*(-\d+)\)', label_expr)
    if m:
        return abs(int(m.group(1)))
    return 5


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
    label_horizon: int = 0  # 0=自动从 label_expr 解析，也可以手动指定覆盖

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
    ic_threshold: float = 0.03        # 初筛放宽至 0.03
    icir_threshold: float = 0.5       # 初筛放宽至 0.5
    win_rate_threshold: float = 0.60
    ls_annual_return_threshold: float = 0.10
    ls_sharpe_threshold: float = 0.5
    satellite_composite_min: float = 40.0  # satellite 最低综合评分

    # ── 生命周期配置 ──
    enable_lifecycle: bool = True
    registry_dir: str = ""
    factor_library_dir: str = ""

    # ── 监控告警阈值 ──
    monitor_freq: str = "month"
    monitor_ic_warning: float = 0.03
    monitor_ic_danger: float = 0.0
    monitor_consecutive_bad: int = 3

    # ── 输出目录 ──
    warehouse_dir: str = ""       # 统一数据仓库（按年分文件，不分 tier）
    factors_dir: str = ""         # 按 tier 的引用目录（软链/注册表）
    cache_dir: str = ""           # 计算缓存
    report_dir: str = ""          # HTML 报告

    # ── 稳健性检验 ──
    robustness_sub_periods: list = field(default_factory=lambda: [])
    robustness_sub_pools: list = field(default_factory=lambda: [])
    robustness_ls_cost: float = 0.001

    def __post_init__(self):
        """自动计算 label_horizon（0=从表达式解析）。"""
        if self.label_horizon == 0:
            self.label_horizon = _parse_label_horizon(self.label_expr)


# 项目根目录：从 config.py 位置 src/qlworks/evaluation/config.py 向上
# __file__ 的实际解析需用 resolve() 确保准确
_BASE_DIR = Path(__file__).resolve().parent
_QLWORKS_DIR = _BASE_DIR.parent  # src/qlworks
_PROJECT_ROOT = _QLWORKS_DIR.parent.parent  # src → Qlibworks

DEFAULT_CONFIG = EvalConfig(
    warehouse_dir=str(_PROJECT_ROOT / "factor_data" / "warehouse"),
    factors_dir=str(_PROJECT_ROOT / "factor_data" / "qualified_factors"),
    cache_dir=str(_PROJECT_ROOT / "factor_data" / "cache"),
    report_dir=str(_PROJECT_ROOT / "factor_data" / "reports"),
    registry_dir=str(_PROJECT_ROOT / "factor_data" / "registry"),
    factor_library_dir=str(_PROJECT_ROOT / "factor_data" / "factor_library"),
    robustness_sub_periods=[
        ("2018-01-01", "2020-12-31"),
        ("2021-01-01", "2023-12-31"),
        ("2024-01-01", "2026-04-30"),
    ],
    robustness_sub_pools=["csi300", "csi500", "csi800"],
)
