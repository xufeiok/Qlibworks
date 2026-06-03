"""
因子定义 Schema：YAML 读写、字段校验、dataclass 定义。
符合《因子库构建与持续迭代建议》字段规范，
以及个人三级准入标准。
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yaml
import json


# ── 因子生命周期阶段 ──
class LifecycleStage:
    EXPLORATION = "exploration"   # 探索期：初评通过，进入候选池
    ACTIVE = "active"             # 活跃期：正式评测通过，在库使用
    OBSERVATION = "observation"   # 观察期：表现退化，降低权重监控
    ARCHIVED = "archived"         # 归档期：确认失效
    REVIVAL = "revival"           # 复活期：市场变化，有望重新有效

    _ORDER = [EXPLORATION, ACTIVE, OBSERVATION, ARCHIVED, REVIVAL]

    @classmethod
    def can_transition(cls, current: str, target: str) -> bool:
        allowed = {
            cls.EXPLORATION: [cls.ACTIVE, cls.ARCHIVED],
            cls.ACTIVE: [cls.OBSERVATION, cls.ARCHIVED],
            cls.OBSERVATION: [cls.ACTIVE, cls.ARCHIVED],
            cls.ARCHIVED: [cls.REVIVAL],
            cls.REVIVAL: [cls.EXPLORATION, cls.ACTIVE],
        }
        return target in allowed.get(current, [])


# ── 因子定义 Schema ──
@dataclass
class FactorDefinition:
    name: str
    version: str = "1.0"
    category: str = ""
    sub_category: str = ""
    expression: dict = field(default_factory=lambda: {"qlib": "", "duckdb": ""})
    logic: dict = field(default_factory=lambda: {
        "return_source": "",
        "theory": "",
        "expected_direction": "positive",
    })
    parameters: dict = field(default_factory=lambda: {"lookback": 0, "freq": "daily"})
    data_source: str = "qlib"
    meaning: str = ""
    ref: str = ""
    eval_history: list = field(default_factory=list)
    lifecycle_stage: str = LifecycleStage.EXPLORATION
    lifecycle_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_yaml(cls, content: str) -> "FactorDefinition":
        data = yaml.safe_load(content)
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "FactorDefinition":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    def save_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())



# ── 因子查找（从 Qlibworks factors_repo 统一读取） ──
_FACTOR_CACHE = None


def _load_all_factors_from_repo():
    import yaml, os
    repo = str(Path(__file__).resolve().parents[2] / "Qlibworks" / "factor_data" / "factor_library")
    files = [
        "price_volume_factors", "quality_factors", "style_factors",
        "risk_factors", "sentiment_factors", "other_factors"
    ]
    all_factors = []
    for f in files:
        path = os.path.join(repo, f + ".yaml")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp)
            for fd in data.get("factors", []):
                fd["_source_file"] = f
                all_factors.append(fd)
        except Exception:
            pass
    return all_factors


def find_factor_from_qlib(name: str) -> Optional[dict]:
    global _FACTOR_CACHE
    if _FACTOR_CACHE is None:
        _FACTOR_CACHE = _load_all_factors_from_repo()
    for fd in _FACTOR_CACHE:
        if fd.get("name") == name:
            expr = fd.get("expression", "")
            if isinstance(expr, dict):
                expr = expr.get("qlib", str(expr))
            return {
                "name": name, "expr": str(expr),
                "meaning": fd.get("meaning", ""),
                "category": fd.get("category", fd.get("_source_file", "unknown")),
                "source_file": fd.get("_source_file", ""),
                "usage": fd.get("usage_scenario", ""),
            }
    return None


def list_factors_from_qlib(category: str = "") -> list:
    global _FACTOR_CACHE
    if _FACTOR_CACHE is None:
        _FACTOR_CACHE = _load_all_factors_from_repo()
    results = []
    for fd in _FACTOR_CACHE:
        if category and fd.get("_source_file") != category:
            continue
        expr = fd.get("expression", "")
        if isinstance(expr, dict):
            expr = expr.get("qlib", str(expr))
        results.append({
            "name": fd.get("name", ""), "expr": str(expr),
            "meaning": fd.get("meaning", ""),
            "category": fd.get("category", fd.get("_source_file", "unknown")),
            "source_file": fd.get("_source_file", ""),
            "usage": fd.get("usage_scenario", ""),
        })
    return results


# ── 数据质量审计报告（自动从 DataFrame 提取） ──
@dataclass
class DataQualityReport:
    missing_rate: float = 0.0
    n_years: float = 0.0
    valid_pct: float = 0.0
    n_stocks_avg: int = 0
    n_dates: int = 0
    n_total_rows: int = 0
    outlier_pct: float = 0.0

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, factor_col: str) -> "DataQualityReport":
        total = len(df)
        missing = df[factor_col].isna().sum()
        missing_rate = missing / total if total > 0 else 1.0
        dates = pd.to_datetime(df["datetime"]).unique() if "datetime" in df.columns else []
        n_dates = len(dates)
        n_years = (dates.max() - dates.min()).days / 365.25 if n_dates > 1 else 0.0
        valid = df[factor_col].notna()
        valid_pct = valid.sum() / total if total > 0 else 0.0
        n_stocks = df["instrument"].nunique() if "instrument" in df.columns else 0
        vals = df.loc[valid, factor_col]
        outlier_pct = 0.0
        if len(vals) > 10:
            median = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - median))
            if mad > 1e-12:
                lo = median - 5 * 1.4826 * mad
                hi = median + 5 * 1.4826 * mad
                outlier_pct = ((vals < lo) | (vals > hi)).sum() / len(vals)
        return cls(
            missing_rate=round(missing_rate, 4), n_years=round(n_years, 1),
            valid_pct=round(valid_pct, 4), n_stocks_avg=n_stocks,
            n_dates=n_dates, n_total_rows=total,
            outlier_pct=round(outlier_pct, 4),
        )

    def passed_stage1(self, thresholds=None) -> tuple:
        if thresholds is None:
            from .factor_def import STRICT_THRESHOLDS
            thresholds = STRICT_THRESHOLDS
        reasons = []
        if self.missing_rate > thresholds.missing_rate_max:
            reasons.append(f"缺失率 {self.missing_rate:.1%} > {thresholds.missing_rate_max:.0%}")
        if self.n_years < thresholds.min_data_years:
            reasons.append(f"数据年限 {self.n_years:.1f} < {thresholds.min_data_years:.0f} 年")
        if self.valid_pct < thresholds.min_valid_samples_pct:
            reasons.append(f"有效样本 {self.valid_pct:.1%} < {thresholds.min_valid_samples_pct:.0%}")
        if self.outlier_pct > thresholds.outlier_pct_max:
            reasons.append(f"异常值 {self.outlier_pct:.1%} > {thresholds.outlier_pct_max:.0%}")
        return len(reasons) == 0, reasons

# ── 三级准入门槛配置 ──
@dataclass
class QualificationThresholds:
    missing_rate_max: float = 0.05
    min_valid_samples_pct: float = 0.80
    min_data_years: float = 5.0
    outlier_pct_max: float = 0.01
    ic_mean_min: float = 0.05
    ic_positive_ratio_min: float = 0.65
    ir_min: float = 1.0
    sharpe_min: float = 1.25
    ls_annual_return_min: float = 15.0
    monotonicity_min: float = 0.8
    ic_std_max: float = 0.03
    rolling_24m_ic_vol_max: float = 0.30
    param_robustness_drop_max: float = 0.20
    t_stat_min: float = 2.0
    max_correlation_existing: float = 0.70
    max_correlation_barra: float = 0.50
    monitor_ic_warning: float = 0.03
    monitor_ic_danger: float = 0.0
    monitor_consecutive_bad_months: int = 3


STRICT_THRESHOLDS = QualificationThresholds()

RELAXED_THRESHOLDS = QualificationThresholds(
    ic_mean_min=0.03,
    ic_positive_ratio_min=0.55,
    ir_min=0.5,
    sharpe_min=0.8,
    ls_annual_return_min=8.0,
    monotonicity_min=0.6,
    ic_std_max=0.05,
)
