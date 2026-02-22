import pandas as pd

from voice_screening.splitting import (
    create_participant_train_validation_split,
    create_patient_level_folds,
)


def test_no_participant_leakage() -> None:
    df = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "x": list(range(8)),
        }
    )

    splits = create_patient_level_folds(
        df,
        participant_col="participant_id",
        label_col="label",
        n_splits=2,
        random_seed=42,
    )

    for split in splits:
        train_pids = set(df.loc[split.train_index, "participant_id"].tolist())
        test_pids = set(df.loc[split.test_index, "participant_id"].tolist())
        assert not train_pids.intersection(test_pids)


def test_inner_train_validation_split_is_participant_disjoint() -> None:
    df = pd.DataFrame(
        {
            "participant_id": [
                "p1",
                "p1",
                "p2",
                "p2",
                "p3",
                "p3",
                "p4",
                "p4",
                "p5",
                "p5",
                "p6",
                "p6",
            ],
            "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )

    train_idx, val_idx = create_participant_train_validation_split(
        df,
        participant_col="participant_id",
        label_col="label",
        validation_fraction=0.33,
        random_seed=42,
    )

    train_pids = set(df.loc[train_idx, "participant_id"].tolist())
    val_pids = set(df.loc[val_idx, "participant_id"].tolist())
    assert val_pids
    assert not train_pids.intersection(val_pids)
