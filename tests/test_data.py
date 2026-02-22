from __future__ import annotations

from pathlib import Path

import pandas as pd

from voice_screening.data import load_raw_tables


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

