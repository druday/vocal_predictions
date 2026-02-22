import numpy as np
import pandas as pd
import pytest

from voice_screening.evaluation import aggregate_to_patient_level, find_best_threshold


def test_patient_aggregation_methods_supported() -> None:
    pred_df = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "label": [1, 1, 0, 0],
            "probability": [0.9, 0.7, 0.2, 0.4],
        }
    )

    expected = {
        "mean": {"p1": 0.8, "p2": 0.3},
        "median": {"p1": 0.8, "p2": 0.3},
        "max": {"p1": 0.9, "p2": 0.4},
        "min": {"p1": 0.7, "p2": 0.2},
        "std": {"p1": np.std([0.9, 0.7], ddof=1), "p2": np.std([0.2, 0.4], ddof=1)},
    }

    for method, method_expected in expected.items():
        out = aggregate_to_patient_level(
            pred_df=pred_df,
            participant_col="participant_id",
            label_col="label",
            prob_col="probability",
            method=method,
        )
        got = dict(zip(out["participant_id"], out["probability"]))
        assert got == pytest.approx(method_expected)


def test_patient_aggregation_invalid_method_raises() -> None:
    pred_df = pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "label": [1, 1],
            "probability": [0.8, 0.7],
        }
    )

    with pytest.raises(ValueError):
        aggregate_to_patient_level(
            pred_df=pred_df,
            participant_col="participant_id",
            label_col="label",
            prob_col="probability",
            method="unknown",
        )


def test_find_best_threshold_returns_max_f1_threshold() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.4, 0.6, 0.9], dtype=float)

    threshold, best_f1 = find_best_threshold(
        y_true=y_true,
        y_prob=y_prob,
        min_threshold=0.05,
        max_threshold=0.95,
        n_points=19,
        default_threshold=0.5,
    )

    assert threshold == pytest.approx(0.45)
    assert best_f1 == pytest.approx(1.0)
