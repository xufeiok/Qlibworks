"""
训练脚本配置加载辅助模块

功能概述：
- 从 `scripts/training/configs` 目录加载 YAML 配置。
- 支持在“脚本内 [全局配置区]”和 “YAML 配置文件”之间切换。
- 为 YAML 配置补齐脚本本地默认键，避免因省略字段而报错。
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# [2026-10-26 迁移] 本模块现位于 src/qlworks/training/config.py：
#   parents[0]=.../src/qlworks/training, parents[1]=.../src/qlworks,
#   parents[2]=.../src, parents[3]=项目根。
# 确保 src 在 sys.path（本模块被 scripts/training 下各训练脚本引用）
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from qlworks.pipeline_config import (
    LABEL_EXPR, LABEL_NAME, INSTRUMENTS, START_TIME, END_TIME, ICIR_KEEP_RATIO,
)


# YAML 配置文件仍统一存放在 scripts/training/configs/，显式指向保持配置位置不变。
CONFIG_DIR = Path(__file__).resolve().parents[3] / "scripts" / "training" / "configs"

LINEAR_CONFIG_DEFAULTS: dict[str, Any] = {
    "instruments": INSTRUMENTS,
    "use_dynamic_filter": True,
    "filter_new_stocks": True,
    "filter_st": True,
    "start_time": START_TIME,
    "end_time": END_TIME,
    "model_type": "linear",
    "label_fields": [LABEL_EXPR],
    "label_names": [LABEL_NAME],
    # [五大类归档 2026-10-08] 原 reversal_momentum_factors 已并入 momentum_price_factors；
    # factor_categories 限定只取「动量与反转」子类（13 个），保持原实验口径不变
    "factor_files": ["momentum_price_factors"],
    "factor_categories": ["动量与反转"],
    "factor_cache_names": [],
    "normalize_features": True,
    "neutralize_features": True,
    "renormalize_features_after_neutralize": True,
    "normalize_labels": True,
    "neutralize_labels": True,
    "symmetric_orthogonalization": True,
    "rolling_windows": [
        {"name": "Test_2023", "train": ("2020-01-01", "2021-12-20"), "valid": ("2022-01-01", "2022-12-20"), "test": ("2023-01-01", "2023-12-31")},
        {"name": "Test_2024", "train": ("2021-01-01", "2022-12-20"), "valid": ("2023-01-01", "2023-12-20"), "test": ("2024-01-01", "2024-12-31")},
        {"name": "Test_2025", "train": ("2022-01-01", "2023-12-20"), "valid": ("2024-01-01", "2024-12-20"), "test": ("2025-01-01", "2025-12-31")},
    ],
    "top_k_factors": 20,
    "icir_stability_check": {"enabled": True, "rolling_window": 60, "keep_ratio": ICIR_KEEP_RATIO},
    "feature_selection": {"method": "filter", "algo": "f_regression", "label_col": "LABEL_5D", "remove_collinearity": False},
}


def _deep_merge_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """递归合并嵌套字典，列表与标量以 overrides 为准。"""
    result = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_name: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    功能概述：
    - 读取训练脚本 YAML 配置，并补齐默认键。

    边界条件：
    - 当配置文件不存在、为空或内容不是字典时抛出异常。
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        raise ValueError(f"配置文件为空: {config_path}")
    if not isinstance(loaded, dict):
        raise TypeError(f"配置文件顶层必须是字典: {config_path}")

    base_defaults = LINEAR_CONFIG_DEFAULTS if defaults is None else defaults
    return _deep_merge_dict(base_defaults, loaded)


def resolve_runtime_config(
    local_config: dict[str, Any],
    default_yaml_name: str,
    config_source: str = "local",
    config_name: str | None = None,
) -> dict[str, Any]:
    """
    功能概述：
    - 根据来源选择最终运行配置。

    参数说明：
    - local_config: 当前脚本 [全局配置区] 中的配置字典。
    - default_yaml_name: 当前脚本默认对应的 YAML 配置文件名（不含后缀）。
    - config_source: `local` 或 `yaml`，默认 `local`。
    - config_name: 当 `config_source=yaml` 时可显式指定 YAML 文件名；为空时回退到 default_yaml_name。
    """
    if config_source == "local":
        return deepcopy(local_config)
    if config_source == "yaml":
        return load_config(config_name or default_yaml_name, defaults=local_config)
    raise ValueError(f"不支持的配置来源: {config_source}，仅支持 local 或 yaml")
