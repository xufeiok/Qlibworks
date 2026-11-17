# -*- coding: utf-8 -*-
"""
factor_registry.py — 因子注册表查询模块（Registry）

[定位]
  元数据单一事实源：所有因子身份信息（家族/生命周期/来源/数据版本指纹）
  集中存于 factor_data/registry/registry.json（由 scripts/registry/upgrade_meta.py 生成）。
  数据读取仍走 warehouse（parquet 只存一份），候选池/实盘池仅引用本注册表的因子名。

[用法]
  from qlworks.core.factor_registry import FactorRegistry
  reg = FactorRegistry()                    # 自动加载 registry.json
  reg.all()                                 # 全量因子条目
  reg.by_family("momentum_price")           # 某家族因子列表
  reg.by_name("STR_20d")                    # 单因子元数据
  reg.family_count()                        # {family: n}
  reg.names(family="quality")               # 仅名称列表

[设计要点]
  - 只读视图：不直接改 warehouse / registry.json，变更走 upgrade_meta.py
  - 容错：registry.json 缺失时返回空视图并告警，不中断调用方
  - 与 pipeline_config 单一事实源互补：pipeline_config 管"全局阈值/股票池"，
    本模块管"因子个体元数据"
"""

import json
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3]
    / "factor_data" / "registry" / "registry.json"
)


class FactorRegistry:
    """因子注册表只读视图。"""

    def __init__(self, registry_path: Optional[str | Path] = None):
        self.path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._data: dict = {}
        self._factors: list[dict] = []
        self._by_name: dict[str, dict] = {}
        self.load()

    # ── 加载 ──────────────────────────────────────────────────────────
    def load(self) -> None:
        if not self.path.exists():
            print(f"[factor_registry] registry.json 不存在: {self.path} "
                  f"（先运行 scripts/registry/upgrade_meta.py 生成）")
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
            self._factors = self._data.get("factors") or []
            self._by_name = {e.get("name"): e for e in self._factors if e.get("name")}
        except Exception as e:
            print(f"[factor_registry] registry.json 解析失败: {e}")
            self._data, self._factors, self._by_name = {}, [], {}

    # ── 查询 ──────────────────────────────────────────────────────────
    def all(self) -> list[dict]:
        """全量因子条目（含 family/lifecycle_stage/version_hash/data_range）。"""
        return list(self._factors)

    def names(self, family: Optional[str] = None) -> list[str]:
        """因子名列表；family 非空时仅返回该家族。"""
        if family:
            return [e.get("name") for e in self._factors if e.get("family") == family]
        return [e.get("name") for e in self._factors]

    def by_family(self, family: str) -> list[dict]:
        """某家族的全部因子条目。"""
        return [e for e in self._factors if e.get("family") == family]

    def by_name(self, name: str) -> Optional[dict]:
        """单因子元数据；不存在返回 None。"""
        return self._by_name.get(name)

    def family_count(self) -> dict:
        """{family: 因子数}，按家族名排序。"""
        cnt: dict[str, int] = {}
        for e in self._factors:
            fam = e.get("family") or "unknown"
            cnt[fam] = cnt.get(fam, 0) + 1
        return dict(sorted(cnt.items()))

    def family_names(self) -> list[str]:
        """注册表中实际存在的家族名列表。"""
        return sorted({e.get("family") for e in self._factors})

    # ── 便捷工具 ──────────────────────────────────────────────────────
    def subset(self, family: Optional[str] = None,
               lifecycle: Optional[str] = None) -> list[str]:
        """按家族+生命周期过滤的因子名列表（供筛选/训练端取名单）。"""
        out = []
        for e in self._factors:
            if family and e.get("family") != family:
                continue
            if lifecycle and e.get("lifecycle_stage") != lifecycle:
                continue
            out.append(e.get("name"))
        return out

    def summary(self) -> str:
        """单行摘要，便于打印。"""
        cnt = self.family_count()
        fam_str = ", ".join(f"{k}={v}" for k, v in cnt.items())
        return f"FactorRegistry({self.path.name}): {len(self._factors)} 因子 [{fam_str}]"


# ── 模块级便捷函数（无需实例化）─────────────────────────────────────

_registry: Optional[FactorRegistry] = None


def get_registry() -> FactorRegistry:
    """进程内单例注册表（避免重复读盘）。"""
    global _registry
    if _registry is None:
        _registry = FactorRegistry()
    return _registry


def list_factors(family: Optional[str] = None,
                 lifecycle: Optional[str] = None) -> list[str]:
    """按家族/生命周期取因子名列表（与 FactorRegistry.subset 等价）。"""
    return get_registry().subset(family=family, lifecycle=lifecycle)


if __name__ == "__main__":
    reg = FactorRegistry()
    print(reg.summary())
    for fam in reg.family_names():
        print(f"  {fam}: {len(reg.by_family(fam))} 个")
