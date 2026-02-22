from __future__ import annotations

from typing import Any


VALID_ANALYSIS_MODES = {"confirmatory", "exploratory"}
VALID_TUNING_SPLITS = {"val", "test"}
VALID_SELECTION_METRICS = {"roc_auc", "accuracy", "f1", "precision", "recall"}


def get_analysis_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("analysis", {}).get("mode", "confirmatory")).strip().lower()
    if mode not in VALID_ANALYSIS_MODES:
        raise ValueError(
            f"Unsupported analysis.mode='{mode}'. Valid values: {sorted(VALID_ANALYSIS_MODES)}"
        )
    return mode


def is_confirmatory_mode(config: dict[str, Any]) -> bool:
    return get_analysis_mode(config) == "confirmatory"


def get_tuning_split(config: dict[str, Any]) -> str:
    analysis_cfg = config.get("analysis", {})
    override = analysis_cfg.get("selection_split")
    if override is None or str(override).strip() == "":
        return "val" if is_confirmatory_mode(config) else "test"

    split = str(override).strip().lower()
    if split not in VALID_TUNING_SPLITS:
        raise ValueError(
            f"Unsupported analysis.selection_split='{split}'. "
            f"Valid values: {sorted(VALID_TUNING_SPLITS)}"
        )
    return split


def get_final_refit_on_train_val(config: dict[str, Any]) -> bool:
    default_refit = is_confirmatory_mode(config)
    return bool(config.get("analysis", {}).get("final", {}).get("refit_on_train_val", default_refit))


def get_final_evaluate_all_models(config: dict[str, Any]) -> bool:
    return bool(config.get("analysis", {}).get("final", {}).get("evaluate_all_models_at_locked_k", True))


def get_deterministic_torch(config: dict[str, Any]) -> bool:
    return bool(config.get("reproducibility", {}).get("torch_deterministic", False))


def get_patient_selection_metric(config: dict[str, Any]) -> str:
    metric = str(config.get("modeling", {}).get("patient_selection_metric", "roc_auc")).strip().lower()
    if metric not in VALID_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported modeling.patient_selection_metric='{metric}'. "
            f"Valid values: {sorted(VALID_SELECTION_METRICS)}"
        )
    return metric

