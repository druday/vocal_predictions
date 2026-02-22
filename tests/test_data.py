from __future__ import annotations

from pathlib import Path

import pandas as pd

from voice_screening.data import dataset_summary, load_raw_tables, select_feature_columns


def _config(data_root: Path) -> dict:
    return {
        "paths": {"data_root": str(data_root)},
        "files": {
            "phenotype_tsv": "phenotype.tsv",
            "static_features_tsv": "static_features.tsv",
        },
        "columns": {
            "participant_id": "participant_id",
            "session_id": "session_id",
        },
        "labeling": {
            "positive_column_candidates": ["parkinsons"],
            "control_column_candidates": ["is_control_participant"],
        },
    }


def test_load_raw_tables_reads_explicit_files(tmp_path: Path) -> None:
    data_root = tmp_path / "raw_data"
    data_root.mkdir(parents=True, exist_ok=True)

    phenotype = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "session_id": ["s1"],
            "parkinsons": ["yes"],
            "is_control_participant": ["no"],
        }
    )
    static = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "session_id": ["s1"],
            "feat_a": [0.1],
        }
    )
    phenotype.to_csv(data_root / "phenotype.tsv", sep="\t", index=False)
    static.to_csv(data_root / "static_features.tsv", sep="\t", index=False)

    phenotype_df, static_df = load_raw_tables(_config(data_root))
    assert set(["parkinsons", "is_control_participant"]).issubset(phenotype_df.columns)
    assert "feat_a" in static_df.columns


def test_load_raw_tables_discovers_nested_physionet_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "raw_data"
    diagnosis_dir = data_root / "physionet.org/files/b2ai-voice/3.0.0/phenotype/diagnosis"
    enrollment_dir = data_root / "physionet.org/files/b2ai-voice/3.0.0/phenotype/enrollment"
    features_dir = data_root / "physionet.org/files/b2ai-voice/3.0.0/features"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    enrollment_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "session_id": ["s1", "s2"],
            "parkinsons": ["yes", "no"],
        }
    ).to_csv(diagnosis_dir / "diagnosis.tsv", sep="\t", index=False)

    pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "session_id": ["s1", "s2"],
            "is_control_participant": ["no", "yes"],
        }
    ).to_csv(enrollment_dir / "enrollment.tsv", sep="\t", index=False)

    pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "session_id": ["s1", "s2"],
            "feat_a": [0.2, 0.3],
        }
    ).to_csv(features_dir / "static_features.tsv", sep="\t", index=False)

    phenotype_df, static_df = load_raw_tables(_config(data_root))
    assert set(["parkinsons", "is_control_participant"]).issubset(phenotype_df.columns)
    assert "feat_a" in static_df.columns


def test_dataset_summary_includes_named_recording_and_participant_counts() -> None:
    df = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2", "p3"],
            "label": [1, 1, 0, 0, 0],
        }
    )
    cfg = {
        "columns": {"participant_id": "participant_id"},
        "labeling": {
            "positive_class_name": "parkinsons",
            "negative_class_name": "control",
        },
    }

    summary = dataset_summary(df, feature_cols=["f1", "f2"], config=cfg)

    assert summary["class_counts"] == {"0": 3, "1": 2}
    assert summary["class_counts_named"] == {"parkinsons": 2, "control": 3}
    assert summary["participant_class_counts"] == {"0": 2, "1": 1}
    assert summary["participant_class_counts_named"] == {"parkinsons": 1, "control": 2}


def test_select_feature_columns_excludes_label_leakage_candidates() -> None:
    merged_df = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "session_id": ["s1", "s2"],
            "parkinsons": [1, 0],  # leakage candidate
            "is_control_participant": [0, 1],  # leakage candidate
            "label": [1, 0],  # target
            "mfcc_1_mean": [0.1, 0.2],
            "jitter_local": [0.01, 0.02],
        }
    )
    cfg = {
        "feature_selection": {
            "exclude_columns": [
                "participant_id",
                "session_id",
                "label",
                "pd_label",
                "control_label",
            ]
        },
        "labeling": {
            "positive_column_candidates": ["parkinsons"],
            "control_column_candidates": ["is_control_participant"],
            "positive_class_name": "parkinsons",
            "negative_class_name": "control",
        },
    }

    feature_cols = select_feature_columns(merged_df, cfg)
    assert "parkinsons" not in feature_cols
    assert "is_control_participant" not in feature_cols
    assert "mfcc_1_mean" in feature_cols
    assert "jitter_local" in feature_cols
