"""
因子生命周期管理：5阶段流转 + 变更日志 + 状态查询。

生命周期阶段：
  探索期(exploration) → 活跃期(active) → 观察期(observation) → 归档期(archived)
                                                                  ↓
                                                              复活期(revival) → 探索期/活跃期
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .factor_def import LifecycleStage


class LifecycleManager:
    """因子生命周期管理器。

    管理因子从候选到归档的全生命周期状态流转，
    每次状态变更记录到 lifecycle_log.json。
    """

    def __init__(self, registry_dir: str = ""):
        if not registry_dir:
            from .config import DEFAULT_CONFIG
            registry_dir = DEFAULT_CONFIG.registry_dir
            registry_dir = str(Path(__file__).resolve().parents[1] / "factor_registry")
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.registry_dir / "lifecycle_log.json"
        self._ensure_log()

    def _ensure_log(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump({"events": [], "last_updated": str(datetime.now())}, f, ensure_ascii=False, indent=2)

    def _append_event(self, event: dict):
        with open(self._log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        log["events"].append(event)
        log["last_updated"] = str(datetime.now())
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    def get_lifecycle_log(self, factor_name: Optional[str] = None) -> list:
        """获取生命周期变更日志。"""
        with open(self._log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        events = log["events"]
        if factor_name:
            events = [e for e in events if e.get("factor_name") == factor_name]
        return events

    def get_current_stage(self, factor_name: str, registry: dict) -> str:
        """从注册表中获取因子当前生命周期阶段。"""
        factor_info = registry.get("factors", {}).get(factor_name, {})
        return factor_info.get("lifecycle_stage", LifecycleStage.EXPLORATION)

    def transition(
        self,
        factor_name: str,
        from_stage: str,
        to_stage: str,
        reason: str = "",
        registry_path: Optional[str] = None,
    ) -> bool:
        """执行生命周期状态迁移。

        Args:
            factor_name: 因子名称
            from_stage: 当前状态
            to_stage: 目标状态
            reason: 迁移原因
            registry_path: 注册表路径（如需同步更新）

        Returns:
            是否迁移成功
        """
        if not LifecycleStage.can_transition(from_stage, to_stage):
            return False

        event = {
            "factor_name": factor_name,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "reason": reason,
            "timestamp": str(datetime.now()),
        }
        self._append_event(event)

        # 同步更新注册表
        if registry_path:
            reg_path = Path(registry_path)
            if reg_path.exists():
                with open(reg_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                if factor_name in registry.get("factors", {}):
                    registry["factors"][factor_name]["lifecycle_stage"] = to_stage
                    registry["factors"][factor_name]["last_transition"] = str(datetime.now())
                    registry["last_updated"] = str(datetime.now())
                    with open(reg_path, "w", encoding="utf-8") as f:
                        json.dump(registry, f, ensure_ascii=False, indent=2)

        return True

    def check_degradation(
        self,
        ic_mean: float,
        icir: float,
        prev_ic_mean: float,
        prev_icir: float,
    ) -> tuple:
        """检查因子性能退化程度。

        Returns:
            (is_degraded: bool, severity: str, message: str)
            severity: none / warning / danger
        """
        ic_drop = (prev_ic_mean - ic_mean) / abs(prev_ic_mean) if prev_ic_mean != 0 else 999
        icir_drop = (prev_icir - icir) / abs(prev_icir) if prev_icir != 0 else 999

        if ic_drop > 0.5 or icir_drop > 0.5:
            return True, "danger", f"IC 降幅 {ic_drop:.1%}, ICIR 降幅 {icir_drop:.1%}"
        if ic_drop > 0.3 or icir_drop > 0.3:
            return True, "warning", f"IC 降幅 {ic_drop:.1%}, ICIR 降幅 {icir_drop:.1%}"
        return False, "none", "正常"
