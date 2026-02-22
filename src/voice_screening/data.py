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


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def _discover_phenotype_tsvs(data_root: Path) -> list[Path]:
    files = []
    for path in data_root.rglob("*.tsv"):
        parts = [part.lower() for part in path.parts]
        if "phenotype" in parts:
            files.append(path)
    return sorted(files)


def _phenotype_has_label_sources(df: pd.DataFrame, config: dict[str, Any]) -> bool:
    participant_col = config["columns"]["participant_id"]
    if participant_col not in df.columns:
        return False

    label_cfg = config["labeling"]
    has_positive = _match_column(df.columns.tolist(), label_cfg["positive_column_candidates"]) is not None
    has_control = _match_column(df.columns.tolist(), label_cfg["control_column_candidates"]) is not None
    return has_positive and has_control


def _read_tsv_header(path: Path) -> list[str] | None:
    try:
        return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    except Exception:
        return None


def _collapse_to_unique_keys(df: pd.DataFrame, keys: list[str], value_col: str) -> pd.DataFrame:
    keep = [*keys, value_col]
    out = df[keep].copy()
    out["_notna_value"] = out[value_col].notna().astype(int)
    out = out.sort_values("_notna_value", ascending=False).drop(columns=["_notna_value"])
    return out.drop_duplicates(subset=keys, keep="first")


def _build_phenotype_from_nested_tables(files: list[Path], config: dict[str, Any]) -> pd.DataFrame | None:
    participant_col = config["columns"]["participant_id"]
    session_col = config["columns"]["session_id"]
    label_cfg = config["labeling"]

    sources: list[dict[str, Any]] = []
    for path in files:
        cols = _read_tsv_header(path)
        if not cols or participant_col not in cols:
            continue
        sources.append(
            {
                "path": path,
                "cols": cols,
                "has_session": session_col in cols,
                "positive_col": _match_column(cols, label_cfg["positive_column_candidates"]),
                "control_col": _match_column(cols, label_cfg["control_column_candidates"]),
            }
        )

    if not sources:
        return None

    control_sources = [s for s in sources if s["control_col"] is not None]
    positive_sources = [s for s in sources if s["positive_col"] is not None]

    if not control_sources and not positive_sources:
        return None

    def _score_control(src: dict[str, Any]) -> tuple[int, int, int]:
        parts = [part.lower() for part in src["path"].parts]
        name = src["path"].name.lower()
        return (
            0 if src["has_session"] else 1,
            0 if "session" in parts or "session" in name else 1,
            len(parts),
        )

    def _score_positive(src: dict[str, Any]) -> tuple[int, int, int]:
        parts = [part.lower() for part in src["path"].parts]
        name = src["path"].name.lower()
        return (
            0 if "diagnosis" in parts else 1,
            0 if "parkinson" in name else 1,
            len(parts),
        )

    if control_sources:
        control_sources.sort(key=_score_control)
        base = control_sources[0]
        base_col = str(base["control_col"])
    else:
        positive_sources.sort(key=_score_positive)
        base = positive_sources[0]
        base_col = str(base["positive_col"])

    base_keys = [participant_col]
    if base["has_session"]:
        base_keys.append(session_col)

    base_usecols = [*base_keys, base_col]
    phenotype_df = _read_tsv(base["path"])[base_usecols]
    phenotype_df = _collapse_to_unique_keys(phenotype_df, base_keys, base_col)

    def _merge_source(
        dst: pd.DataFrame,
        src: dict[str, Any],
        value_col: str,
    ) -> pd.DataFrame:
        src_keys = [participant_col]
        if src["has_session"]:
            src_keys.append(session_col)

        usecols = [*src_keys, value_col]
        src_df = _read_tsv(src["path"])[usecols]
        src_df = _collapse_to_unique_keys(src_df, src_keys, value_col)

        merge_keys = [participant_col]
        if session_col in dst.columns and src["has_session"]:
            merge_keys.append(session_col)

        incoming = value_col
        if incoming in dst.columns:
            incoming = f"{value_col}__src"
            src_df = src_df.rename(columns={value_col: incoming})

        out = dst.merge(src_df, on=merge_keys, how="left")
        if incoming != value_col:
            out[value_col] = out[value_col].where(out[value_col].notna(), out[incoming])
            out = out.drop(columns=[incoming])
        return out

    for src in control_sources:
        if src["path"] == base["path"] and str(src["control_col"]) == base_col:
            continue
        phenotype_df = _merge_source(phenotype_df, src, str(src["control_col"]))

    for src in positive_sources:
        if src["path"] == base["path"] and str(src["positive_col"]) == base_col:
            continue
        phenotype_df = _merge_source(phenotype_df, src, str(src["positive_col"]))

    return phenotype_df


def _discover_static_tsv(data_root: Path, preferred_name: str) -> Path | None:
    candidates = list(data_root.rglob("*.tsv"))
    if not candidates:
        return None

    preferred = preferred_name.lower()

    def _score(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        parts = [part.lower() for part in path.parts]

        if name == preferred:
            rank = 0
        elif "static" in name and "feature" in name:
            rank = 1
        elif "features" in parts:
            rank = 2
        else:
            rank = 9
        return rank, len(parts)

    ranked = sorted(candidates, key=_score)
    if _score(ranked[0])[0] >= 9:
        return None
    return ranked[0]


def load_raw_tables(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(config["paths"]["data_root"])
    phenotype_file = data_root / config["files"]["phenotype_tsv"]
    static_file = data_root / config["files"]["static_features_tsv"]

    phenotype_df: pd.DataFrame | None = None
    if phenotype_file.exists():
        phenotype_df = _read_tsv(phenotype_file)

    if phenotype_df is None or not _phenotype_has_label_sources(phenotype_df, config):
        phenotype_candidates = _discover_phenotype_tsvs(data_root)
        if phenotype_candidates:
            discovered = _build_phenotype_from_nested_tables(phenotype_candidates, config)
            if discovered is not None:
                phenotype_df = discovered

    if phenotype_df is None:
        raise FileNotFoundError(
            f"Missing phenotype file: {phenotype_file}. "
            f"No usable phenotype TSV files were discovered under: {data_root}"
        )

    static_path = static_file
    if not static_path.exists():
        discovered_static = _discover_static_tsv(data_root, static_file.name)
        if discovered_static is None:
            raise FileNotFoundError(
                f"Missing static feature file: {static_file}. "
                f"No usable static-feature TSV was discovered under: {data_root}"
            )
        static_path = discovered_static

    static_df = _read_tsv(static_path)
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
