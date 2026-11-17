# -*- coding: utf-8 -*-
"""
因子集合（Factor Set）解析统一入口。

功能概述：
- 训练脚本（train_tree-doubao / train_linear / train_icir）不再各自硬编码
  `candidate_pool.json` 路径，统一通过本模块把"因子集合文件"解析为结构化对象。
- 支持两种集合格式：
  1) 候选池 JSON（Alpha Book，factor_data/registry/candidate_pool.json）：
     仅保留 status 为 admitted（或缺省）的因子，附带 category / ICIR 供组合层配额使用；
  2) 精选 CSV（select_factors.py 输出）：优先取 `factor_name` 列，否则取第 2 列因子名。

输入/输出：
- resolve_factor_set(spec) -> FactorSet | None
- validate_factor_set(factor_set, factor_files) -> None（因子名缺失抛 ValueError）
- list_factor_sets() -> list[str]

边界条件：
- spec 为 None/空 时返回 None，调用方回退到 factor_files 全量加载；
- spec 指向的文件不存在时抛 FileNotFoundError（不做静默回退，避免因子集无声漂移）；
- 集合内无可用因子时抛 ValueError。

性能/安全注意事项：
- 仅读取本地 JSON/CSV，不执行任何表达式；文件规模小，无性能压力。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# 项目根目录：src/qlworks/factors/registry.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_DIR = PROJECT_ROOT / "factor_data" / "registry"
# 自建因子组合目录：与候选池并列，供 --factor-set 直接引用
FACTOR_SETS_DIR = PROJECT_ROOT / "factor_data" / "factor_sets"
DEFAULT_FACTOR_SET = "factor_data/registry/candidate_pool.json"


@dataclass(frozen=True)
class FactorSet:
    """因子集合：训练/选股阶段唯一因子名单来源。"""

    path: Path
    version: str
    members: tuple[dict, ...]  # [{"name": str, "category": str, "icir": float | None}, ...]

    @property
    def names(self) -> list[str]:
        """按集合内顺序返回因子名列表。"""
        return [m["name"] for m in self.members]


def _resolve_path(spec: str | Path) -> Path:
    """相对路径按项目根解析，绝对路径原样返回。"""
    path = Path(spec)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json_set(path: Path) -> tuple[str, list[dict]]:
    """解析候选池 JSON（Alpha Book），仅保留准入因子。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    members: list[dict] = []
    seen: set[str] = set()
    for entry in data.get("factors") or []:
        name = entry.get("name")
        if not name or name in seen:
            continue
        # 候选池中未显式标注状态的历史条目视为已准入
        if entry.get("status") not in (None, "admitted"):
            continue
        icir = entry.get("latest_icir")
        if icir is None:
            icir = (entry.get("_screening") or {}).get("latest_icir")
        members.append({
            "name": str(name),
            "category": str(entry.get("category") or ""),
            "icir": icir,
        })
        seen.add(str(name))
    return str((data.get("_meta") or {}).get("set_version") or ""), members


def _read_csv_set(path: Path) -> tuple[str, list[dict]]:
    """解析精选 CSV（select_factors.py 输出），无类别信息。"""
    df = pd.read_csv(path)
    if "factor_name" in df.columns:
        series = df["factor_name"]
    elif df.shape[1] >= 2:
        series = df.iloc[:, 1]
    elif df.shape[1] == 1:
        series = df.iloc[:, 0]
    else:
        series = pd.Series(dtype=object)

    members: list[dict] = []
    seen: set[str] = set()
    for value in series.tolist():
        name = str(value).strip()
        if not name or name == "nan" or name in seen:
            continue
        members.append({"name": name, "category": "", "icir": None})
        seen.add(name)
    return path.stem, members


def resolve_factor_set(spec: str | Path | None) -> FactorSet | None:
    """解析因子集合文件；spec 为空时返回 None（调用方回退 factor_files 全量）。

    Args:
        spec: 集合文件路径（相对项目根或绝对路径），.json 候选池 / .csv 精选名单。
    Returns:
        FactorSet 或 None。
    Raises:
        FileNotFoundError: 文件不存在（不静默回退）。
        ValueError: 集合内无可用因子。
    """
    if not spec:
        return None
    path = _resolve_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"因子集合文件不存在: {path}")

    if path.suffix.lower() == ".csv":
        version, members = _read_csv_set(path)
    else:
        version, members = _read_json_set(path)

    if not members:
        raise ValueError(f"因子集合内无可用因子: {path}")
    return FactorSet(path=path, version=version or path.stem, members=tuple(members))


def validate_factor_set(factor_set: FactorSet, factor_files: str | list[str]) -> None:
    """校验集合内因子名均能在 YAML 因子库中找到，缺名直接报错。

    说明：FactorLibraryManager 对未命中的因子名是静默丢弃的，若不显式校验，
    训练端会悄悄少用因子，导致实验结果不可复现。

    Raises:
        ValueError: 存在无法解析的因子名（报文列出前 10 个）。
    """
    from qlworks.factors.manager import FactorLibraryManager

    packages = [factor_files] if isinstance(factor_files, str) else list(factor_files)
    _, found = FactorLibraryManager().get_expressions(packages, factor_names=factor_set.names)
    missing = [n for n in factor_set.names if n not in set(found)]
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"因子集合 {factor_set.path.name} 中 {len(missing)} 个因子在 YAML 因子库中不存在: "
            f"{preview}{suffix}；请检查 factor_files={packages} 或更新集合文件"
        )


def _is_factor_set_file(path: Path) -> bool:
    """轻量判断 registry 目录下文件是否为因子集合。

    约定：候选池带 `_meta`（不带下划线的 meta 是因子主注册表 registry.json），
    或成员条目显式带 status 字段；据此排除 registry.json / lifecycle_log.json 等非集合文件。
    """
    if path.suffix.lower() == ".csv":
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        factors = data.get("factors") or []
        if not factors:
            return False
        return isinstance(data.get("_meta"), dict) or any("status" in e for e in factors if isinstance(e, dict))
    except Exception:
        return False


def list_factor_sets() -> list[str]:
    """列出 registry 与 factor_sets 目录下可用的因子集合文件（相对项目根路径）。"""
    results: list[str] = []
    for directory in (REGISTRY_DIR, FACTOR_SETS_DIR):
        if not directory.exists():
            continue
        results.extend(
            p.relative_to(PROJECT_ROOT).as_posix()
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in (".json", ".csv") and _is_factor_set_file(p)
        )
    return results
