from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a dictionary root.")
    return data


def load_config(base_config_path: str | Path, phenotype_config_path: str | Path) -> ConfigDict:
    base_path = Path(base_config_path)
    pheno_path = Path(phenotype_config_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not pheno_path.exists():
        raise FileNotFoundError(f"Phenotype config not found: {pheno_path}")

    base_cfg = _load_yaml(base_path)
    pheno_cfg = _load_yaml(pheno_path)
    merged = _deep_merge(base_cfg, pheno_cfg)

    # Normalize path keys to string-like path values
    outputs_root = merged.get("paths", {}).get("outputs_root", "outputs")
    merged.setdefault("paths", {})["outputs_root"] = str(outputs_root)
    data_root = merged.get("paths", {}).get("data_root", "")
    merged.setdefault("paths", {})["data_root"] = str(data_root)
    return merged
