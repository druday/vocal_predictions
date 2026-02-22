from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LabelingError(RuntimeError):
    pass


def _match_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}

    # Exact match first
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]

    # Substring match fallback
    for cand in candidates:
        cand_l = cand.lower()
        for c in columns:
            if cand_l in c.lower():
                return c

    return None


def _normalize_str_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def load_raw_tables(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(config["paths"]["data_root"])
    phenotype_file = data_root / config["files"]["phenotype_tsv"]
    static_file = data_root / config["files"]["static_features_tsv"]

    if not phenotype_file.exists():
        raise FileNotFoundError(f"Missing phenotype file: {phenotype_file}")
    if not static_file.exists():
        raise FileNotFoundError(f"Missing static feature file: {static_file}")

    phenotype_df = pd.read_csv(phenotype_file, sep="\t", low_memory=False)
    static_df = pd.read_csv(static_file, sep="\t", low_memory=False)
    return phenotype_df, static_df


def apply_case_control_labeling(phenotype_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    label_cfg = config["labeling"]
    participant_col = config["columns"]["participant_id"]

    if participant_col not in phenotype_df.columns:
        raise LabelingError(f"Required participant ID column not found: {participant_col}")

    positive_col = _match_column(phenotype_df.columns.tolist(), label_cfg["positive_column_candidates"])
    control_col = _match_column(phenotype_df.columns.tolist(), label_cfg["control_column_candidates"])

    if positive_col is None:
        raise LabelingError(
            "Could not detect positive phenotype column from candidates "
            f"{label_cfg['positive_column_candidates']}"
        )
    if control_col is None:
        raise LabelingError(
            "Could not detect control column from candidates "
            f"{label_cfg['control_column_candidates']}"
        )

    positive_values = {str(v).strip().lower() for v in label_cfg["positive_values"]}
    control_values = {str(v).strip().lower() for v in label_cfg["control_values"]}

    out = phenotype_df.copy()
    out["pd_label"] = _normalize_str_series(out[positive_col]).isin(positive_values).astype(int)
    out["control_label"] = _normalize_str_series(out[control_col]).isin(control_values).astype(int)

    out["label"] = np.nan
    out.loc[out["pd_label"] == 1, "label"] = 1
    out.loc[out["control_label"] == 1, "label"] = 0

    if label_cfg.get("include_case_control_only", True):
        out = out[(out["pd_label"] == 1) | (out["control_label"] == 1)].copy()

    # If conflicting case+control rows exist, prioritize explicit PD rows.
    out["label"] = out["label"].fillna(out["pd_label"]).astype(int)
    out["label_name"] = out["label"].map(
        {
            1: label_cfg.get("positive_class_name", "positive"),
            0: label_cfg.get("negative_class_name", "negative"),
        }
    )

    return out


def merge_label_and_static(
    labeled_phenotype_df: pd.DataFrame,
    static_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    participant_col = config["columns"]["participant_id"]
    session_col = config["columns"]["session_id"]

    merge_cols = [participant_col]
    if session_col in labeled_phenotype_df.columns and session_col in static_df.columns:
        merge_cols.append(session_col)

    keep_cols = merge_cols + ["label", "label_name", "pd_label", "control_label"]
    meta = labeled_phenotype_df[keep_cols].drop_duplicates()

    merged = static_df.merge(meta, on=merge_cols, how="inner")
    return merged


def select_feature_columns(merged_df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    exclude_cols = set(config["feature_selection"]["exclude_columns"])

    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    if not feature_cols:
        raise RuntimeError("No numeric feature columns detected after exclusion rules.")

    return feature_cols


def build_modeling_table(merged_df: pd.DataFrame, feature_cols: list[str], config: dict[str, Any]) -> pd.DataFrame:
    participant_col = config["columns"]["participant_id"]
    session_col = config["columns"]["session_id"]
    task_col = config["columns"].get("task_name", "task_name")

    keep_cols = [participant_col, "label", "label_name"]
    if session_col in merged_df.columns:
        keep_cols.append(session_col)
    if task_col in merged_df.columns:
        keep_cols.append(task_col)

    keep_cols.extend(feature_cols)
    out = merged_df[keep_cols].copy()

    # Basic cleanup: remove rows with missing label/participant
    out = out.dropna(subset=[participant_col, "label"]).copy()
    out["label"] = out["label"].astype(int)

    return out


def dataset_summary(df: pd.DataFrame, feature_cols: list[str], config: dict[str, Any]) -> dict[str, Any]:
    participant_col = config["columns"]["participant_id"]

    n_records = int(len(df))
    n_participants = int(df[participant_col].nunique())
    class_counts = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}

    per_pid = df.groupby(participant_col).size()
    return {
        "n_records": n_records,
        "n_participants": n_participants,
        "n_features": int(len(feature_cols)),
        "class_counts": class_counts,
        "recordings_per_participant": {
            "min": float(per_pid.min()),
            "median": float(per_pid.median()),
            "max": float(per_pid.max()),
            "mean": float(per_pid.mean()),
        },
    }
