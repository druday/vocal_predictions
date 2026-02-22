import pytest

from voice_screening.analysis import (
    get_analysis_mode,
    get_patient_selection_metric,
    get_tuning_split,
    is_confirmatory_mode,
)
from voice_screening.config import load_config


def _base_cfg() -> dict:
    cfg = load_config("configs/base.yaml", "configs/phenotypes/parkinsons.yaml")
    return dict(cfg)


def test_default_analysis_mode_is_confirmatory() -> None:
    cfg = _base_cfg()
    assert get_analysis_mode(cfg) == "confirmatory"
    assert is_confirmatory_mode(cfg)
    assert get_tuning_split(cfg) == "val"


def test_exploratory_mode_defaults_to_test_tuning_split() -> None:
    cfg = _base_cfg()
    cfg.setdefault("analysis", {})["mode"] = "exploratory"
    cfg["analysis"].pop("selection_split", None)
    assert get_analysis_mode(cfg) == "exploratory"
    assert not is_confirmatory_mode(cfg)
    assert get_tuning_split(cfg) == "test"


def test_invalid_mode_raises() -> None:
    cfg = _base_cfg()
    cfg.setdefault("analysis", {})["mode"] = "bad-mode"
    with pytest.raises(ValueError):
        get_analysis_mode(cfg)


def test_invalid_patient_selection_metric_raises() -> None:
    cfg = _base_cfg()
    cfg.setdefault("modeling", {})["patient_selection_metric"] = "bad-metric"
    with pytest.raises(ValueError):
        get_patient_selection_metric(cfg)

