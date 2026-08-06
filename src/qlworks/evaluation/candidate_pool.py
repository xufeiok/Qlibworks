"""
候选因子池管理（Alpha Book v2）。

三级筛选流程：
  第一级：数据质量筛查 → 第二级：预测能力检验 → 第三级：稳定性+冗余过滤
  通过第三级 → 进入候选池（探索期）

[单一准入通道]（P0-1 收敛）
  候选池仅由 admit_to_multifactor.py 的三关检验写入（相关性 / 边际贡献 / 方向一致）。
  evaluate() 只更新 registry.json 与 qualified_factors 分档，不再直接写候选池，
  避免旁路绕过准入检验导致无效因子混入训练因子池。

[数据有效性门]（P0-2）
  - ic_mean 缺失或恒为 0（数据源缺失）→ 判为 pending_data，不入池
  - 负 IC 因子允许入池，但强制记录 direction（A 股反转因子负 IC 是常态，方向稳定即可用）
  - 三级筛选统一使用 RELAXED 阈值作为准入线（STRICT 作为 core 判定线）
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
    写路径收敛为单一通道：admit_to_multifactor.py 三关检验通过后 add_candidate。
    """

    def __init__(self, registry_dir: str = ""):
        if not registry_dir:
            from .config import DEFAULT_CONFIG
            registry_dir = DEFAULT_CONFIG.registry_dir
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._pool_path = self.registry_dir / "candidate_pool.json"
        self._ensure_pool()

    def _ensure_pool(self):
        if not self._pool_path.exists():
            with open(self._pool_path, "w", encoding="utf-8") as f:
                json.dump({
                    "_meta": {
                        "version": "2.0",
                        "set_version": "v1",
                        "description": "多因子准入候选池（Alpha Book）— 仅由 admit_to_multifactor.py 三关检验写入，下游 train_tree-doubao.py 从此读取因子名单",
                        "pipeline_note": "单一准入通道：evaluate() 只写 registry + qualified_factors 分档，不直接写候选池",
                        "updated_at": str(datetime.now()),
                        "admit_thresholds": {
                            "max_correlation_existing": 0.70,
                            "min_oos_icir": 0.5,
                            "min_recent_3y_ic_positive_ratio": 0.60,
                            "direction_consistency_required": True,
                        },
                    },
                    "factors": [],
                    "rejected": [],
                    "stats": {"total_candidates": 0, "admitted": 0,
                              "rejected_corr": 0, "rejected_marginal": 0, "rejected_direction": 0},
                }, f, ensure_ascii=False, indent=2)

    def _save(self, data: dict):
        with open(self._pool_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> dict:
        with open(self._pool_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── 三级筛选（准入线统一用 RELAXED，STRICT 作为 core 判定线） ──

    def stage1_data_quality_check(self, missing_rate: float, n_years: float, valid_pct: float,
                                  th: QualificationThresholds = None) -> tuple:
        """第一级：数据质量筛查。

        Args:
            th: 阈值集合（默认 RELAXED 作为准入线；STRICT 用于 core 判定）
        """
        th = th or RELAXED_THRESHOLDS
        reasons = []
        if missing_rate > th.missing_rate_max:
            reasons.append(f"缺失率 {missing_rate:.1%} > {th.missing_rate_max:.0%}")
        if n_years < th.min_data_years:
            reasons.append(f"数据年限 {n_years:.1f} < {th.min_data_years:.0f} 年")
        if valid_pct < th.min_valid_samples_pct:
            reasons.append(f"有效样本 {valid_pct:.1%} < {th.min_valid_samples_pct:.0%}")
        return len(reasons) == 0, reasons

    def stage2_predictive_check(self, ic_mean: float, ic_positive_ratio: float, ir: float,
                                th: QualificationThresholds = None) -> tuple:
        """第二级：预测能力检验。"""
        th = th or RELAXED_THRESHOLDS
        reasons = []
        if ic_mean < th.ic_mean_min:
            reasons.append(f"IC 均值 {ic_mean:.4f} < {th.ic_mean_min}")
        if ic_positive_ratio < th.ic_positive_ratio_min:
            reasons.append(f"IC 胜率 {ic_positive_ratio:.1%} < {th.ic_positive_ratio_min:.0%}")
        if ir < th.ir_min:
            reasons.append(f"IR {ir:.2f} < {th.ir_min}")
        return len(reasons) == 0, reasons

    def stage3_stability_check(self, ic_std: float, sharpe: float, monotonicity: float,
                               th: QualificationThresholds = None) -> tuple:
        """第三级：稳定性过滤。"""
        th = th or RELAXED_THRESHOLDS
        reasons = []
        if ic_std > th.ic_std_max:
            reasons.append(f"IC 标准差 {ic_std:.4f} > {th.ic_std_max}")
        if sharpe < th.sharpe_min:
            reasons.append(f"夏普 {sharpe:.2f} < {th.sharpe_min}")
        if monotonicity < th.monotonicity_min:
            reasons.append(f"单调性 {monotonicity:.2f} < {th.monotonicity_min}")
        return len(reasons) == 0, reasons

    def _check_data_validity(self, metrics: dict) -> tuple:
        """[P0-2] 数据有效性门：IC 缺失或恒为 0（数据源缺失）→ pending_data。

        Returns:
            (valid: bool, reason: str)
        """
        ic_mean = metrics.get("ic_mean")
        if ic_mean is None or pd.isna(ic_mean):
            return False, "IC 均值为空（数据缺失），判为待重算"
        if float(ic_mean) == 0.0:
            ic_std = metrics.get("ic_std")
            if ic_std is None or pd.isna(ic_std) or float(ic_std) == 0.0:
                return False, "IC 恒为 0（数据源缺失），判为待重算"
        return True, ""

    def full_screening(self, metrics: dict = None, df: pd.DataFrame = None, factor_col: str = "") -> dict:
        """执行三级完整筛选（准入线 = RELAXED）。

        Args:
            metrics: 可选，预计算指标字典。如果提供则直接使用。
            df: 可选，原始数据 DataFrame。与 metrics 二选一。
            factor_col: 因子列名（df 模式时必需）

        Returns:
            {passed, stage_results: [{stage, passed, reasons}], composite_score, data_quality: DataQualityReport}
        """
        # [P0-2] 数据有效性门前置
        if metrics is not None:
            valid, reason = self._check_data_validity(metrics)
            if not valid:
                return {
                    "passed": False,
                    "data_pending": True,
                    "data_pending_reason": reason,
                    "stage_results": [],
                    "composite_score": 0.0,
                }

        if df is not None and factor_col:
            dq = DataQualityReport.from_dataframe(df, factor_col)
            s1, r1 = dq.passed_stage1(RELAXED_THRESHOLDS)
            metrics_from_df = {
                "missing_rate": dq.missing_rate,
                "n_years": dq.n_years,
                "valid_pct": dq.valid_pct,
                "outlier_pct": dq.outlier_pct,
            }
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
            "data_pending": False,
            "stage_results": stages,
            # 修复历史 BUG：原 sum(1 for s in [...]) 恒等于元素个数 3，
            # 导致 composite_score 恒为 100.0 却 passed=False（脏数据悖论）。
            "composite_score": round(sum([s1, s2, s3]) / 3 * 100, 1),
        }
        if dq is not None:
            result["data_quality"] = dq
        return result

    @staticmethod
    def _derive_direction(metrics: dict) -> str:
        """[P0-2] 从真实 IC 均值符号推导方向（不做 abs，保留反转因子负方向）。"""
        ic_mean = metrics.get("ic_mean")
        if ic_mean is None or pd.isna(ic_mean) or float(ic_mean) == 0.0:
            return "unknown"
        return "positive" if float(ic_mean) > 0 else "negative"

    def add_candidate(self, factor_name: str, metrics: dict, screening: dict,
                      direction: str = "", tier: str = "") -> dict:
        """将因子加入候选池（仅由 admit_to_multifactor.py 调用）。

        Args:
            factor_name: 因子名
            metrics: 真实评测指标（ic_mean 必须为带符号原始值，不做 abs）
            screening: full_screening 结果
            direction: 方向（可选，缺省由 ic_mean 符号推导）
            tier: 档位（可选，缺省由 screening 判定）

        Returns:
            池条目 dict；若数据有效性门未过，返回含 status="data_pending" 的条目且不入池。
        """
        # [P0-2] 数据有效性门：IC 缺失/恒 0 不入池
        if screening.get("data_pending"):
            print(f"[candidate_pool] {factor_name}: {screening.get('data_pending_reason', '')}，不入池")
            return {
                "name": factor_name,
                "status": "data_pending",
                "data_pending_reason": screening.get("data_pending_reason", ""),
            }

        data = self._load()
        pool = data.setdefault("factors", [])

        if not direction:
            direction = self._derive_direction(metrics)

        # tier 判定：STRICT 全过 → core，否则按 RELAXED 准入线 → satellite
        if not tier:
            core_ok = (
                self.stage1_data_quality_check(
                    metrics.get("missing_rate", 0), metrics.get("n_years", 0),
                    metrics.get("valid_pct", 0), STRICT_THRESHOLDS)[0]
                and self.stage2_predictive_check(
                    metrics.get("ic_mean", 0), metrics.get("ic_positive_ratio", 0),
                    metrics.get("ir", 0), STRICT_THRESHOLDS)[0]
                and self.stage3_stability_check(
                    metrics.get("ic_std", 0), metrics.get("sharpe", 0),
                    metrics.get("monotonicity", 0), STRICT_THRESHOLDS)[0]
            )
            tier = "core" if core_ok else "satellite"

        now = str(datetime.now())
        entry = {
            "name": factor_name,
            "tier": tier,
            "direction": direction,
            "category": "",
            "sub_category": "",
            "meaning": "",
            "source_file": "",
            "latest_icir": metrics.get("ir", 0),
            "admitted_at": now,
            "eval_date": now,
            "rolling_ic": {},
            "tier_history": [{"tier": tier, "at": now}],
            "status": "admitted",
            "_metrics": metrics,
            "_screening": screening,
        }

        # 已存在则更新（保留 tier_history 历史）
        for i, e in enumerate(pool):
            if e.get("name") == factor_name:
                prev_tier = e.get("tier")
                if prev_tier and prev_tier != tier:
                    entry["tier_history"] = e.get("tier_history", []) + entry["tier_history"]
                pool[i] = entry
                break
        else:
            pool.append(entry)

        # 按 ICIR 排序
        pool.sort(key=lambda x: x.get("latest_icir", 0), reverse=True)
        data["_meta"]["updated_at"] = str(datetime.now())
        data["stats"]["admitted"] = len(pool)
        data["stats"]["total_candidates"] = len(pool) + len(data.get("rejected", []))
        self._save(data)
        return entry

    def remove_candidate(self, factor_name: str):
        """从候选池移除因子。"""
        data = self._load()
        data["factors"] = [e for e in data.get("factors", []) if e.get("name") != factor_name]
        data["_meta"]["updated_at"] = str(datetime.now())
        data["stats"]["admitted"] = len(data["factors"])
        self._save(data)

    def list_candidates(self, status: Optional[str] = None) -> list:
        """列出候选池因子。"""
        data = self._load()
        pool = data.get("factors", [])
        if status:
            pool = [e for e in pool if e.get("status") == status]
        return pool

    def get_candidate(self, factor_name: str) -> Optional[dict]:
        """获取单个候选因子信息。"""
        data = self._load()
        for entry in data.get("factors", []):
            if entry.get("name") == factor_name:
                return entry
        return None

    # ── [P1-6] Alpha Book：训练因子池导出 ──

    def export_factor_names(self, min_tier: str = "satellite") -> list:
        """导出候选池因子名单（供 train_tree-doubao.py 作为训练因子池）。

        仅导出 status="admitted" 且 tier 不低于 min_tier 的因子。
        tier 等级序：core > satellite > archive。

        Args:
            min_tier: 最低档位，默认 satellite（core + satellite）

        Returns:
            因子名列表
        """
        rank = {"core": 3, "satellite": 2, "archive": 1}
        min_rank = rank.get(min_tier, 2)
        pool = self._load().get("factors", [])
        names = []
        for e in pool:
            if e.get("status") not in (None, "admitted"):
                continue
            if rank.get(e.get("tier", "satellite"), 2) >= min_rank:
                names.append(e["name"])
        return names

    def bump_set_version(self) -> str:
        """[P1-6] 因子集版本号递增（每次 admit 重建候选池时调用，用于训练追溯）。"""
        data = self._load()
        meta = data.setdefault("_meta", {})
        cur = meta.get("set_version", "v1")
        try:
            n = int(str(cur).lstrip("v")) + 1
        except (ValueError, TypeError):
            n = 2
        meta["set_version"] = f"v{n}"
        meta["updated_at"] = str(datetime.now())
        self._save(data)
        return meta["set_version"]
