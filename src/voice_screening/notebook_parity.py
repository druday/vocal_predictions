from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from voice_screening.clustering import cluster_assignments, feature_distance_matrix, fit_hierarchical_clustering


@dataclass(frozen=True)
class NotebookSplit:
    train_participants: np.ndarray
    val_participants: np.ndarray
    test_participants: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


@dataclass(frozen=True)
class KSelectionResult:
    selected_k: int
    selected_score: float
    global_best_k: int
    global_best_score: float
    strategy: str
    reason: str
    preferred_k_min: int
    preferred_k_max: int
    near_top_delta: float
    prefer_smallest_k: bool


def clean_feature_dataframe(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Mirror notebook cleaning:
    1) Keep numeric / numeric-coercible columns.
    2) Replace inf with NaN.
    3) Fill per-column NaN using column median (or 0 fallback).
    4) Force-fill any residual NaN/inf with 0.
    """
    out = x.copy()

    numeric_cols: list[str] = []
    removed_cols: list[str] = []
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        try:
            pd.to_numeric(series, errors="raise")
            numeric_cols.append(col)
        except (TypeError, ValueError):
            removed_cols.append(col)

    out = out[numeric_cols].copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    for col in out.columns:
        series = out[col]
        if series.isna().all():
            out[col] = 0.0
            continue
        col_median = series.median()
        if pd.isna(col_median):
            out[col] = series.fillna(0.0)
        else:
            out[col] = series.fillna(float(col_median))

    out = out.fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0)
    out = out.astype(np.float32)

    return out, list(out.columns), removed_cols


def participant_train_val_test_split(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    validation_size_from_train_val: float,
    random_seed: int,
) -> NotebookSplit:
    participant_ids_arr = np.asarray(participant_ids).astype(str)
    labels_arr = np.asarray(labels).astype(int)

    # Mirror notebook semantics exactly:
    #   unique_participants = pid_full.unique()
    #   participant_labels = merged_df.groupby('participant_id')['label'].first()
    #   participant_labels.loc[unique_participants]
    unique_participants = pd.Series(participant_ids_arr).drop_duplicates().to_numpy()
    participant_labels_series = pd.Series(labels_arr, index=participant_ids_arr).groupby(level=0).first()
    participant_labels = participant_labels_series.loc[unique_participants].to_numpy()

    train_val_pids, test_pids, y_train_val, _ = train_test_split(
        unique_participants,
        participant_labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=participant_labels,
    )

    train_pids, val_pids, _, _ = train_test_split(
        train_val_pids,
        y_train_val,
        test_size=validation_size_from_train_val,
        random_state=random_seed,
        stratify=y_train_val,
    )

    train_set = set(train_pids.tolist())
    val_set = set(val_pids.tolist())
    test_set = set(test_pids.tolist())

    train_mask = np.array([pid in train_set for pid in participant_ids_arr], dtype=bool)
    val_mask = np.array([pid in val_set for pid in participant_ids_arr], dtype=bool)
    test_mask = np.array([pid in test_set for pid in participant_ids_arr], dtype=bool)

    return NotebookSplit(
        train_participants=np.asarray(train_pids).astype(str),
        val_participants=np.asarray(val_pids).astype(str),
        test_participants=np.asarray(test_pids).astype(str),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def standardize_and_clip(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    clip_abs: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.asarray(x_train, dtype=np.float32))
    val_scaled = scaler.transform(np.asarray(x_val, dtype=np.float32))
    test_scaled = scaler.transform(np.asarray(x_test, dtype=np.float32))

    train_scaled = np.clip(train_scaled, -clip_abs, clip_abs)
    val_scaled = np.clip(val_scaled, -clip_abs, clip_abs)
    test_scaled = np.clip(test_scaled, -clip_abs, clip_abs)

    return train_scaled, val_scaled, test_scaled, scaler


def fit_cluster_assignments(
    x_train_scaled: np.ndarray,
    feature_names: list[str],
    n_clusters: int,
    linkage_method: str = "ward",
) -> pd.DataFrame:
    dist = feature_distance_matrix(x_train_scaled)
    linkage = fit_hierarchical_clustering(dist, linkage_method=linkage_method)
    return cluster_assignments(feature_names, linkage, n_clusters)


def aggregate_features_by_cluster(
    x_train_scaled: np.ndarray,
    x_val_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    feature_names: list[str],
    assignments: pd.DataFrame,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Notebook-aligned cluster feature aggregation for recording-level data.
    """
    method_normalized = str(method).strip().lower()
    valid_methods = {"mean", "pca", "max", "min", "std", "median", "sum"}
    if method_normalized not in valid_methods:
        raise ValueError(f"Unsupported aggregation method: {method}")

    x_train_df = pd.DataFrame(np.asarray(x_train_scaled), columns=feature_names)
    x_val_df = pd.DataFrame(np.asarray(x_val_scaled), columns=feature_names)
    x_test_df = pd.DataFrame(np.asarray(x_test_scaled), columns=feature_names)

    cluster_to_features: dict[int, list[str]] = {}
    for cluster_id, group in assignments.groupby("cluster"):
        feats = [f for f in group["feature"].tolist() if f in x_train_df.columns]
        if feats:
            cluster_to_features[int(cluster_id)] = feats

    if not cluster_to_features:
        raise RuntimeError("No cluster assignments overlap with available feature names.")

    sorted_clusters = sorted(cluster_to_features.keys())
    cluster_feature_names = [f"cluster_{cluster_id}" for cluster_id in sorted_clusters]

    n_train = len(x_train_df)
    n_val = len(x_val_df)
    n_test = len(x_test_df)
    n_clusters = len(sorted_clusters)

    x_train_out = np.zeros((n_train, n_clusters), dtype=np.float32)
    x_val_out = np.zeros((n_val, n_clusters), dtype=np.float32)
    x_test_out = np.zeros((n_test, n_clusters), dtype=np.float32)

    for idx, cluster_id in enumerate(sorted_clusters):
        feats = cluster_to_features[cluster_id]
        train_block = x_train_df[feats].to_numpy(dtype=np.float32)
        val_block = x_val_df[feats].to_numpy(dtype=np.float32)
        test_block = x_test_df[feats].to_numpy(dtype=np.float32)

        if method_normalized == "pca":
            if train_block.shape[1] == 1:
                x_train_out[:, idx] = train_block[:, 0]
                x_val_out[:, idx] = val_block[:, 0]
                x_test_out[:, idx] = test_block[:, 0]
            else:
                pca = PCA(n_components=1, random_state=42)
                x_train_out[:, idx] = pca.fit_transform(train_block)[:, 0]
                x_val_out[:, idx] = pca.transform(val_block)[:, 0]
                x_test_out[:, idx] = pca.transform(test_block)[:, 0]
        elif method_normalized == "mean":
            x_train_out[:, idx] = np.mean(train_block, axis=1)
            x_val_out[:, idx] = np.mean(val_block, axis=1)
            x_test_out[:, idx] = np.mean(test_block, axis=1)
        elif method_normalized == "median":
            x_train_out[:, idx] = np.median(train_block, axis=1)
            x_val_out[:, idx] = np.median(val_block, axis=1)
            x_test_out[:, idx] = np.median(test_block, axis=1)
        elif method_normalized == "max":
            x_train_out[:, idx] = np.max(train_block, axis=1)
            x_val_out[:, idx] = np.max(val_block, axis=1)
            x_test_out[:, idx] = np.max(test_block, axis=1)
        elif method_normalized == "min":
            x_train_out[:, idx] = np.min(train_block, axis=1)
            x_val_out[:, idx] = np.min(val_block, axis=1)
            x_test_out[:, idx] = np.min(test_block, axis=1)
        elif method_normalized == "std":
            x_train_out[:, idx] = np.std(train_block, axis=1)
            x_val_out[:, idx] = np.std(val_block, axis=1)
            x_test_out[:, idx] = np.std(test_block, axis=1)
        elif method_normalized == "sum":
            x_train_out[:, idx] = np.sum(train_block, axis=1)
            x_val_out[:, idx] = np.sum(val_block, axis=1)
            x_test_out[:, idx] = np.sum(test_block, axis=1)

    return x_train_out, x_val_out, x_test_out, cluster_feature_names


def aggregate_recordings_to_patient_level(
    x_recording_features: np.ndarray,
    participant_ids: np.ndarray,
    aggregation_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    method = str(aggregation_method).strip().lower()
    valid_methods = {"mean", "median", "max", "min", "std"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported patient aggregation method: {aggregation_method}")

    pids = np.asarray(participant_ids).astype(str)
    unique_patients = np.unique(pids)

    n_patients = len(unique_patients)
    n_features = int(np.asarray(x_recording_features).shape[1])
    x_patient = np.zeros((n_patients, n_features), dtype=np.float32)

    for idx, patient_id in enumerate(unique_patients):
        mask = pids == patient_id
        patient_block = np.asarray(x_recording_features)[mask]

        if method == "mean":
            x_patient[idx] = np.mean(patient_block, axis=0)
        elif method == "median":
            x_patient[idx] = np.median(patient_block, axis=0)
        elif method == "max":
            x_patient[idx] = np.max(patient_block, axis=0)
        elif method == "min":
            x_patient[idx] = np.min(patient_block, axis=0)
        elif method == "std":
            x_patient[idx] = np.std(patient_block, axis=0)

    return x_patient, unique_patients


def create_patient_labels(participant_ids: np.ndarray, recording_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pids = np.asarray(participant_ids).astype(str)
    labels = np.asarray(recording_labels).astype(float)

    unique_patients = np.unique(pids)
    patient_labels = np.zeros(len(unique_patients), dtype=int)

    for idx, patient_id in enumerate(unique_patients):
        mask = pids == patient_id
        patient_labels[idx] = int(np.round(np.mean(labels[mask])))

    return patient_labels, unique_patients


def evaluate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= float(threshold)).astype(int)

    if len(np.unique(y_true_arr)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "roc_auc": auc,
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
    }


def choose_small_k_contender(
    df: pd.DataFrame,
    metric: str,
    preferred_k_min: int,
    preferred_k_max: int,
    near_top_delta: float,
    prefer_smallest_k: bool,
) -> tuple[KSelectionResult, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise RuntimeError("Cannot choose K contender from an empty dataframe.")
    score_col = str(metric).strip().lower()
    if score_col not in df.columns:
        raise RuntimeError(f"Selection metric column not found: {score_col}")

    work = df.copy()
    work["n_clusters"] = work["n_clusters"].astype(int)
    work = work.dropna(subset=[score_col])
    if work.empty:
        raise RuntimeError(f"No non-null values for metric: {score_col}")

    global_best_row = work.sort_values([score_col, "n_clusters"], ascending=[False, True]).iloc[0]
    global_best_k = int(global_best_row["n_clusters"])
    global_best_score = float(global_best_row[score_col])

    preferred = work[(work["n_clusters"] >= preferred_k_min) & (work["n_clusters"] <= preferred_k_max)].copy()
    if preferred.empty:
        preferred = work.copy()
        reason = "preferred_range_empty_fallback_global"
    else:
        reason = "near_top_in_preferred_range"

    threshold = global_best_score - float(near_top_delta)
    contenders = preferred[preferred[score_col] >= threshold].copy()
    if contenders.empty:
        contenders = preferred.sort_values([score_col, "n_clusters"], ascending=[False, True]).head(1).copy()
        reason = "best_in_preferred_range"

    if prefer_smallest_k:
        selected_row = contenders.sort_values(["n_clusters", score_col], ascending=[True, False]).iloc[0]
    else:
        selected_row = contenders.sort_values([score_col, "n_clusters"], ascending=[False, True]).iloc[0]

    selected_k = int(selected_row["n_clusters"])
    selected_score = float(selected_row[score_col])

    ranked = preferred.sort_values([score_col, "n_clusters"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank_within_scope"] = ranked.index + 1

    contenders = contenders.sort_values([score_col, "n_clusters"], ascending=[False, True]).reset_index(drop=True)
    contenders["contender_rank"] = contenders.index + 1

    selection = KSelectionResult(
        selected_k=selected_k,
        selected_score=selected_score,
        global_best_k=global_best_k,
        global_best_score=global_best_score,
        strategy="small_k_near_top",
        reason=reason,
        preferred_k_min=int(preferred_k_min),
        preferred_k_max=int(preferred_k_max),
        near_top_delta=float(near_top_delta),
        prefer_smallest_k=bool(prefer_smallest_k),
    )
    return selection, ranked, contenders


def k_selection_payload(selection: KSelectionResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selected_k": int(selection.selected_k),
        "selected_score": float(selection.selected_score),
        "global_best_k": int(selection.global_best_k),
        "global_best_score": float(selection.global_best_score),
        "strategy": selection.strategy,
        "reason": selection.reason,
        "preferred_k_min": int(selection.preferred_k_min),
        "preferred_k_max": int(selection.preferred_k_max),
        "near_top_delta": float(selection.near_top_delta),
        "prefer_smallest_k": bool(selection.prefer_smallest_k),
    }
    if extra:
        payload.update(extra)
    return payload
