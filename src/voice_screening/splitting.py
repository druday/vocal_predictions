from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train_index: np.ndarray
    test_index: np.ndarray


def create_patient_level_folds(
    df: pd.DataFrame,
    participant_col: str,
    label_col: str,
    n_splits: int,
    random_seed: int,
) -> list[FoldSplit]:
    participant_df = df[[participant_col, label_col]].drop_duplicates(subset=[participant_col]).copy()

    if participant_df[label_col].nunique() != 2:
        raise ValueError("Expected binary labels for stratified folds.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    splits: list[FoldSplit] = []
    for fold_id, (train_pid_idx, test_pid_idx) in enumerate(
        skf.split(participant_df[participant_col], participant_df[label_col])
    ):
        train_pids = set(participant_df.iloc[train_pid_idx][participant_col].tolist())
        test_pids = set(participant_df.iloc[test_pid_idx][participant_col].tolist())

        overlap = train_pids.intersection(test_pids)
        if overlap:
            raise RuntimeError(f"Leakage detected: participant overlap in fold {fold_id}: {len(overlap)}")

        train_index = df.index[df[participant_col].isin(train_pids)].to_numpy()
        test_index = df.index[df[participant_col].isin(test_pids)].to_numpy()

        splits.append(FoldSplit(fold_id=fold_id, train_index=train_index, test_index=test_index))

    return splits


def fold_membership_table(
    df: pd.DataFrame,
    splits: list[FoldSplit],
    participant_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in splits:
        test_df = df.loc[split.test_index, [participant_col]].drop_duplicates()
        for pid in test_df[participant_col].tolist():
            rows.append({participant_col: pid, "fold_id": split.fold_id})
    return pd.DataFrame(rows).sort_values(by=["fold_id", participant_col]).reset_index(drop=True)


def create_participant_train_validation_split(
    df: pd.DataFrame,
    participant_col: str,
    label_col: str,
    validation_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a participant-level inner train/validation split from an outer train fold.

    Returns
    -------
    train_index : np.ndarray
        Index values from ``df`` assigned to inner-train.
    validation_index : np.ndarray
        Index values from ``df`` assigned to inner-validation. Can be empty if
        a valid split cannot be formed.
    """
    if validation_fraction <= 0.0:
        return df.index.to_numpy(), np.array([], dtype=df.index.dtype)

    participant_df = df[[participant_col, label_col]].drop_duplicates(subset=[participant_col]).copy()
    if participant_df.empty:
        return df.index.to_numpy(), np.array([], dtype=df.index.dtype)
    if participant_df[label_col].nunique() < 2 or len(participant_df) < 4:
        return df.index.to_numpy(), np.array([], dtype=df.index.dtype)

    try:
        train_pids, val_pids = train_test_split(
            participant_df[participant_col].to_numpy(),
            test_size=validation_fraction,
            random_state=random_seed,
            stratify=participant_df[label_col].to_numpy(),
        )
    except ValueError:
        # Fall back to no inner split when class counts are too small for stratification.
        return df.index.to_numpy(), np.array([], dtype=df.index.dtype)

    train_pids_set = set(train_pids.tolist())
    val_pids_set = set(val_pids.tolist())
    if not train_pids_set or not val_pids_set:
        return df.index.to_numpy(), np.array([], dtype=df.index.dtype)

    train_index = df.index[df[participant_col].isin(train_pids_set)].to_numpy()
    validation_index = df.index[df[participant_col].isin(val_pids_set)].to_numpy()
    return train_index, validation_index
