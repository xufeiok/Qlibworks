# -*- coding: utf-8 -*-
"""
upgrade_meta.py — 将 warehouse 的 meta.json 升级为因子注册表（Registry）

[目标]
  依据 factor_library 五大家族归档（quality/value/momentum_price/sentiment/risk + other），
  为 warehouse 每个因子的 meta.json 补齐注册表字段：
    family           五大家族名（quality/value/momentum_price/sentiment/risk/other/unknown）
    lifecycle_stage  生命周期阶段（保留已有值，缺省 active）
    source_file      因子定义来源 YAML（factor_library/<family>_factors.yaml）
    version_hash     数据版本指纹（data_version + updated_at + total_records 短哈希，
                     不扫描 parquet，幂等可重跑）
  并生成全库汇总 factor_data/registry/registry.json，供 factor_registry.py 查询。

[幂等性]
  重复运行不会破坏已有字段：已有 family/lifecycle_stage/source_file 保持不动，
  仅补齐缺失项并刷新 version_hash 与汇总文件。

[用法]
  python scripts/registry/upgrade_meta.py
  可选参数：
    --dry-run        只报告将变更的项，不写入
    --force          强制刷新 family/lifecycle_stage/source_file（默认仅补缺失）
    --registry-out   汇总文件输出路径（默认 factor_data/registry/registry.json）

[输出]
  factor_data/registry/registry.json
    {
      "meta": {"updated_at", "warehouse_factor_count", "mapped_count", ...},
      "family_map": {"quality": ["current_ratio", ...], ...},
      "factors": [ {name, family, category, source_file, lifecycle_stage,
                    version_hash, data_range, updated_at}, ... ]
    }
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------------------
# 路径（相对脚本位置固定，适配 Windows/Linux）
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
QLIBWORKS = SCRIPT_DIR.parents[1]                    # E:\Quant\Qlibworks
FACTOR_LIBRARY_DIR = QLIBWORKS / "factor_data" / "factor_library"
WAREHOUSE_DIR = QLIBWORKS / "factor_data" / "warehouse"
REGISTRY_DIR = QLIBWORKS / "factor_data" / "registry"

# 家族文件映射：yaml 文件名（去 _factors.yaml 后缀）→ 五大家族
FAMILY_FILE_MAP = {
    "quality": "quality",
    "value": "value",
    "momentum_price": "momentum_price",
    "sentiment": "sentiment",
    "risk": "risk",
    "other": "other",
}

# 默认生命周期阶段（仅在 meta.json 无 lifecycle_stage 时使用）
DEFAULT_LIFECYCLE = "active"

# 注册表字段（缺失时补齐；version_hash 每次刷新）
REGISTRY_FIELDS = ("family", "lifecycle_stage", "source_file", "version_hash")


# ------------------------------------------------------------------------------
# 因子库加载
# ------------------------------------------------------------------------------

def load_factor_library() -> dict:
    """扫描 factor_library 家族 yaml，返回 {factor_name: {family, category, source_file}}。

    容错：单个 yaml 解析失败仅告警，不中断全量升级。
    """
    try:
        import yaml
    except ImportError:
        print("[错误] 缺少 PyYAML，请先安装：pip install pyyaml")
        sys.exit(1)

    name_map: dict = {}
    for fname, family in FAMILY_FILE_MAP.items():
        path = FACTOR_LIBRARY_DIR / f"{fname}_factors.yaml"
        if not path.exists():
            print(f"  [警告] 家族文件不存在: {path}")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"  [错误] 解析失败 {path}: {e}")
            continue
        for factor in (data or {}).get("factors") or []:
            name = factor.get("name")
            if not name:
                continue
            name_map[name] = {
                "family": family,
                "category": factor.get("category", ""),
                "source_file": f"{fname}_factors.yaml",
            }
    print(f"  [加载] factor_library 映射 {len(name_map)} 个因子 -> 五大家族")
    return name_map


# ------------------------------------------------------------------------------
# meta.json 读写
# ------------------------------------------------------------------------------

def _short_hash(*parts) -> str:
    """基于元数据字段的稳定短指纹（不扫描 parquet，秒级完成）。"""
    payload = "|".join(str(p) for p in parts if p is not None and str(p) != "")
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _read_meta(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_meta(path: Path, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def upgrade_warehouse(name_map: dict, force: bool = False, dry_run: bool = False) -> dict:
    """遍历 warehouse 因子目录，补齐注册表字段。

    返回统计信息：
      scanned / updated / skipped_missing_meta / unmapped(未命中家族 yaml)
    """
    stats = {"scanned": 0, "updated": 0, "skipped_missing_meta": 0, "unmapped": 0}
    unmapped_names: list[str] = []

    for factor_dir in sorted(WAREHOUSE_DIR.iterdir()):
        if not factor_dir.is_dir():
            continue
        stats["scanned"] += 1
        meta_path = factor_dir / "meta.json"
        if not meta_path.exists():
            stats["skipped_missing_meta"] += 1
            continue

        name = factor_dir.name
        meta = _read_meta(meta_path)
        info = name_map.get(name, {})

        changed = False
        # family：仅补缺失（force 时覆盖）
        if "family" not in meta or force:
            if info.get("family"):
                meta["family"] = info["family"]
                changed = True
            else:
                meta["family"] = "unknown"
                changed = True
                stats["unmapped"] += 1
                unmapped_names.append(name)
        # lifecycle_stage：仅补缺失
        if "lifecycle_stage" not in meta:
            meta["lifecycle_stage"] = DEFAULT_LIFECYCLE
            changed = True
        # source_file：仅补缺失
        if "source_file" not in meta or force:
            if info.get("source_file"):
                meta["source_file"] = info["source_file"]
                changed = True
            elif "source_file" not in meta:
                meta["source_file"] = ""
                changed = True
        # category：以家族 yaml 为准（仅补缺失）
        if info.get("category") and ("category" not in meta or force):
            meta["category"] = info["category"]
            changed = True
        # version_hash：每次刷新（不扫描 parquet，成本可忽略）
        data_range = meta.get("data_range") or {}
        new_hash = _short_hash(
            meta.get("data_version"), meta.get("updated_at"),
            data_range.get("total_records"), data_range.get("last_date"),
        )
        if meta.get("version_hash") != new_hash:
            meta["version_hash"] = new_hash
            changed = True
        # registry 时间戳
        meta["registry_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if changed:
            stats["updated"] += 1
            if not dry_run:
                _write_meta(meta_path, meta)

    if unmapped_names:
        print(f"  [提示] {len(unmapped_names)} 个因子未命中家族 yaml（family=unknown）:")
        print(f"         {', '.join(unmapped_names[:15])}{'...' if len(unmapped_names) > 15 else ''}")
    return stats


# ------------------------------------------------------------------------------
# registry.json 汇总
# ------------------------------------------------------------------------------

def build_registry(name_map: dict) -> dict:
    """扫描 warehouse，构建全库注册表汇总。"""
    family_map: dict = defaultdict(list)
    factors: list[dict] = []

    for factor_dir in sorted(WAREHOUSE_DIR.iterdir()):
        if not factor_dir.is_dir():
            continue
        meta_path = factor_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = _read_meta(meta_path)
        name = factor_dir.name
        info = name_map.get(name, {})
        family = meta.get("family") or info.get("family") or "unknown"
        entry = {
            "name": name,
            "family": family,
            "category": meta.get("category") or info.get("category") or "",
            "source_file": meta.get("source_file") or info.get("source_file") or "",
            "lifecycle_stage": meta.get("lifecycle_stage", DEFAULT_LIFECYCLE),
            "version_hash": meta.get("version_hash", ""),
            "data_version": meta.get("data_version", ""),
            "updated_at": meta.get("updated_at", ""),
            "data_range": meta.get("data_range", {}),
        }
        factors.append(entry)
        family_map[family].append(name)

    registry = {
        "meta": {
            "description": "Qlibworks 因子注册表（factor registry）— 由 upgrade_meta.py 生成",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "warehouse_factor_count": len(factors),
            "family_count": {f: len(v) for f, v in sorted(family_map.items())},
        },
        "family_map": {f: sorted(v) for f, v in sorted(family_map.items())},
        "factors": factors,
    }
    return registry


# ------------------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="warehouse meta.json 升级为因子注册表")
    parser.add_argument("--dry-run", action="store_true", help="只报告变更，不写入")
    parser.add_argument("--force", action="store_true", help="强制刷新 family/source_file/category")
    parser.add_argument("--registry-out", default=None, help="registry.json 输出路径")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  warehouse meta.json → 因子注册表升级")
    print("=" * 60)
    mode = "[DRY-RUN 仅报告]" if args.dry_run else ("[FORCE 强制刷新]" if args.force else "[增量补齐]")
    print(f"  模式: {mode}")
    print(f"  factor_library: {FACTOR_LIBRARY_DIR}")
    print(f"  warehouse     : {WAREHOUSE_DIR}")

    name_map = load_factor_library()

    print(f"\n[1] 升级 warehouse meta.json...")
    stats = upgrade_warehouse(name_map, force=args.force, dry_run=args.dry_run)
    print(f"  [统计] 扫描 {stats['scanned']} 个因子目录 | 需更新 {stats['updated']} | "
          f"缺 meta {stats['skipped_missing_meta']} | 未映射 {stats['unmapped']}")

    print(f"\n[2] 生成 registry.json 汇总...")
    registry = build_registry(name_map)
    out_path = Path(args.registry_out) if args.registry_out else REGISTRY_DIR / "registry.json"
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"  [汇总] 全库 {registry['meta']['warehouse_factor_count']} 个因子")
    for fam, cnt in registry["meta"]["family_count"].items():
        print(f"         {fam:<16s}: {cnt} 个")
    print(f"  registry: {out_path}" + ("（DRY-RUN 未写入）" if args.dry_run else ""))

    print("\n完成。registry.json 可配合 src/qlworks/core/factor_registry.py 查询使用。")


if __name__ == "__main__":
    main()
