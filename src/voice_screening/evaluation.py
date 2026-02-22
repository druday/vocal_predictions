from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    n_points: int = 19,
    default_threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Tune decision threshold on validation predictions by maximizing F1 score.

    Returns
    -------
    best_threshold, best_f1
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    mask = np.isfinite(y_prob_arr)
    if not np.all(mask):
        y_true_arr = y_true_arr[mask]
        y_prob_arr = y_prob_arr[mask]

    if y_true_arr.size == 0 or y_prob_arr.size == 0:
        return float(default_threshold), float("nan")

    if n_points <= 1:
        thresholds = np.array([float(default_threshold)], dtype=float)
    else:
        thresholds = np.linspace(float(min_threshold), float(max_threshold), int(n_points))

    best_threshold = float(default_threshold)
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        score = float(f1_score(y_true_arr, y_pred, zero_division=0))
        # Strictly ">" keeps first threshold on ties, matching notebook behavior.
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def aggregate_to_patient_level(
    pred_df: pd.DataFrame,
    participant_col: str,
    label_col: str,
    prob_col: str,
    method: str,
) -> pd.DataFrame:
    valid_methods = {"mean", "median", "max", "min", "std"}
    if method not in valid_methods:
        raise ValueError(
            "patient aggregation must be one of: mean, median, max, min, std"
        )

    out = (
        pred_df.groupby(participant_col)
        .agg(
            {
                label_col: "first",
                prob_col: method,
            }
        )
        .reset_index()
    )
    # Keep probabilities in [0, 1] for downstream thresholding even with std aggregation.
    out[prob_col] = out[prob_col].fillna(0.0).clip(0.0, 1.0)
    return out


def summarize_metrics(metrics_rows: list[dict[str, Any]], group_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(metrics_rows)
    metric_cols = ["roc_auc", "accuracy", "f1", "precision", "recall"]
    summary = (
        df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns
    ]
    return summary
