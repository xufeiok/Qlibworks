"""
候选因子池管理。

三级筛选流程：
  第一级：数据质量筛查 → 第二级：预测能力检验 → 第三级：稳定性+冗余过滤
  通过第三级 → 进入候选池（探索期）
"""

from .factor_def import DataQualityReport
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from .factor_def import QualificationThresholds, STRICT_THRESHOLDS, RELAXED_THRESHOLDS


class CandidatePool:
    """候选因子池。

    管理正在探索、但尚未正式入库的因子。
    """

    def __init__(self, registry_dir: str = ""):
        if not registry_dir:
            from .config import DEFAULT_CONFIG
            registry_dir = DEFAULT_CONFIG.registry_dir
            registry_dir = str(Path(__file__).resolve().parents[1] / "factor_registry")
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._pool_path = self.registry_dir / "candidate_pool.json"
        self._ensure_pool()

    def _ensure_pool(self):
        if not self._pool_path.exists():
            with open(self._pool_path, "w", encoding="utf-8") as f:
                json.dump({"candidate_pool": [], "last_updated": str(datetime.now())}, f, ensure_ascii=False, indent=2)

    def _save(self, data: dict):
        with open(self._pool_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> dict:
        with open(self._pool_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def stage1_data_quality_check(self, missing_rate: float, n_years: float, valid_pct: float) -> tuple:
        """第一级：数据质量筛查。

        Returns:
            (passed: bool, reasons: list)
        """
        reasons = []
        if missing_rate > STRICT_THRESHOLDS.missing_rate_max:
            reasons.append(f"缺失率 {missing_rate:.1%} > {STRICT_THRESHOLDS.missing_rate_max:.0%}")
        if n_years < STRICT_THRESHOLDS.min_data_years:
            reasons.append(f"数据年限 {n_years:.1f} < {STRICT_THRESHOLDS.min_data_years:.0f} 年")
        if valid_pct < STRICT_THRESHOLDS.min_valid_samples_pct:
            reasons.append(f"有效样本 {valid_pct:.1%} < {STRICT_THRESHOLDS.min_valid_samples_pct:.0%}")
        return len(reasons) == 0, reasons

    def stage2_predictive_check(self, ic_mean: float, ic_positive_ratio: float, ir: float) -> tuple:
        """第二级：预测能力检验。"""
        reasons = []
        if ic_mean < RELAXED_THRESHOLDS.ic_mean_min:
            reasons.append(f"IC 均值 {ic_mean:.4f} < {RELAXED_THRESHOLDS.ic_mean_min}")
        if ic_positive_ratio < RELAXED_THRESHOLDS.ic_positive_ratio_min:
            reasons.append(f"IC 胜率 {ic_positive_ratio:.1%} < {RELAXED_THRESHOLDS.ic_positive_ratio_min:.0%}")
        if ir < RELAXED_THRESHOLDS.ir_min:
            reasons.append(f"IR {ir:.2f} < {RELAXED_THRESHOLDS.ir_min}")
        return len(reasons) == 0, reasons

    def stage3_stability_check(self, ic_std: float, sharpe: float, monotonicity: float) -> tuple:
        """第三级：稳定性过滤。"""
        reasons = []
        if ic_std > STRICT_THRESHOLDS.ic_std_max:
            reasons.append(f"IC 标准差 {ic_std:.4f} > {STRICT_THRESHOLDS.ic_std_max}")
        if sharpe < STRICT_THRESHOLDS.sharpe_min:
            reasons.append(f"夏普 {sharpe:.2f} < {STRICT_THRESHOLDS.sharpe_min}")
        if monotonicity < STRICT_THRESHOLDS.monotonicity_min:
            reasons.append(f"单调性 {monotonicity:.2f} < {STRICT_THRESHOLDS.monotonicity_min}")
        return len(reasons) == 0, reasons

    def full_screening(self, metrics: dict = None, df: pd.DataFrame = None, factor_col: str = "") -> dict:
        """执行三级完整筛选。

        Args:
            metrics: 可选，预计算指标字典。如果提供则直接使用。
            df: 可选，原始数据 DataFrame。与 metrics 二选一。
            factor_col: 因子列名（df 模式时必需）

        Returns:
            {passed, stage_results: [{stage, passed, reasons}], composite_score, data_quality: DataQualityReport}
        """
        if df is not None and factor_col:
            # DataQualityReport already imported at module level
            dq = DataQualityReport.from_dataframe(df, factor_col)
            s1, r1 = dq.passed_stage1()
            metrics_from_df = {
                "missing_rate": dq.missing_rate,
                "n_years": dq.n_years,
                "valid_pct": dq.valid_pct,
                "outlier_pct": dq.outlier_pct,
            }
            # 合并外部传入的 metrics 覆盖
            if metrics:
                metrics = {**metrics_from_df, **metrics}
            else:
                metrics = metrics_from_df
        else:
            dq = None
            s1, r1 = self.stage1_data_quality_check(
                metrics.get("missing_rate", 0),
                metrics.get("n_years", 0),
                metrics.get("valid_pct", 0),
            )

        s2, r2 = self.stage2_predictive_check(
            metrics.get("ic_mean", 0),
            metrics.get("ic_positive_ratio", 0),
            metrics.get("ir", 0),
        )
        s3, r3 = self.stage3_stability_check(
            metrics.get("ic_std", 0),
            metrics.get("sharpe", 0),
            metrics.get("monotonicity", 0),
        )

        passed = s1 and s2 and s3
        stages = [
            {"stage": 1, "name": "数据质量", "passed": s1, "reasons": r1},
            {"stage": 2, "name": "预测能力", "passed": s2, "reasons": r2},
            {"stage": 3, "name": "稳定性", "passed": s3, "reasons": r3},
        ]

        result = {
            "passed": passed,
            "stage_results": stages,
            "composite_score": sum(1 for s in [s1, s2, s3]) / 3 * 100,
        }
        if dq is not None:
            result["data_quality"] = dq
        return result

    def add_candidate(self, factor_name: str, metrics: dict, screening: dict) -> dict:
        """将因子加入候选池。"""
        data = self._load()
        pool = data["candidate_pool"]

        entry = {
            "factor_name": factor_name,
            "added_date": str(datetime.now()),
            "metrics": metrics,
            "screening": screening,
            "status": "active" if screening["passed"] else "pending",
            "eval_count": 1,
        }

        # 已存在则更新
        for i, e in enumerate(pool):
            if e["factor_name"] == factor_name:
                entry["eval_count"] = e.get("eval_count", 0) + 1
                pool[i] = entry
                break
        else:
            pool.append(entry)

        # 按综合评分排序
        pool.sort(key=lambda x: x.get("screening", {}).get("composite_score", 0), reverse=True)
        data["candidate_pool"] = pool
        data["last_updated"] = str(datetime.now())
        self._save(data)
        return entry

    def remove_candidate(self, factor_name: str):
        """从候选池移除因子。"""
        data = self._load()
        data["candidate_pool"] = [e for e in data["candidate_pool"] if e["factor_name"] != factor_name]
        data["last_updated"] = str(datetime.now())
        self._save(data)

    def list_candidates(self, status: Optional[str] = None) -> list:
        """列出候选池因子。"""
        data = self._load()
        pool = data["candidate_pool"]
        if status:
            pool = [e for e in pool if e.get("status") == status]
        return pool

    def get_candidate(self, factor_name: str) -> Optional[dict]:
        """获取单个候选因子信息。"""
        data = self._load()
        for entry in data["candidate_pool"]:
            if entry["factor_name"] == factor_name:
                return entry
        return None
