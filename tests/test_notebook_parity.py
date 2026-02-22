import numpy as np

from voice_screening.notebook_parity import (
    participant_train_val_test_split,
    summarize_split_class_balance,
)


def _synthetic_records() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    participant_ids: list[str] = []
    labels: list[int] = []

    # Participant-level imbalance with varying recording counts per participant.
    for idx in range(30):
        pid = f"p{idx:03d}"
        y = 1 if idx < 8 else 0
        n_records = int(rng.integers(8, 35))
        participant_ids.extend([pid] * n_records)
        labels.extend([y] * n_records)

    return np.asarray(participant_ids, dtype=str), np.asarray(labels, dtype=int)


def test_balanced_split_search_is_deterministic_and_non_worse() -> None:
    participant_ids, labels = _synthetic_records()

    split_single = participant_train_val_test_split(
        participant_ids=participant_ids,
        labels=labels,
        test_size=0.2,
        validation_size_from_train_val=0.2,
        random_seed=42,
        balance_trials=1,
    )
    split_balanced = participant_train_val_test_split(
        participant_ids=participant_ids,
        labels=labels,
        test_size=0.2,
        validation_size_from_train_val=0.2,
        random_seed=42,
        balance_trials=64,
    )
    split_balanced_repeat = participant_train_val_test_split(
        participant_ids=participant_ids,
        labels=labels,
        test_size=0.2,
        validation_size_from_train_val=0.2,
        random_seed=42,
        balance_trials=64,
    )

    assert split_balanced.balance_score is not None
    assert split_single.balance_score is not None
    assert float(split_balanced.balance_score) <= float(split_single.balance_score) + 1e-12
    assert np.array_equal(split_balanced.train_mask, split_balanced_repeat.train_mask)
    assert np.array_equal(split_balanced.val_mask, split_balanced_repeat.val_mask)
    assert np.array_equal(split_balanced.test_mask, split_balanced_repeat.test_mask)


def test_split_class_balance_summary_counts_are_consistent() -> None:
    participant_ids, labels = _synthetic_records()
    split = participant_train_val_test_split(
        participant_ids=participant_ids,
        labels=labels,
        test_size=0.2,
        validation_size_from_train_val=0.2,
        random_seed=7,
        balance_trials=32,
    )

    summary = summarize_split_class_balance(
        participant_ids=participant_ids,
        labels=labels,
        split=split,
        positive_label=1,
        positive_name="parkinsons",
        negative_name="control",
    )

    assert summary["positive_name"] == "parkinsons"
    assert summary["negative_name"] == "control"
    assert summary["overall"]["recordings"]["n_total"] == int(labels.size)
    assert summary["overall"]["recordings"]["n_positive"] == int(np.sum(labels == 1))

    total_pid = int(np.unique(participant_ids).size)
    assert summary["overall"]["participants"]["n_total"] == total_pid

    split_pid_total = sum(
        int(summary["splits"][name]["participants"]["n_total"]) for name in ("train", "val", "test")
    )
    assert split_pid_total == total_pid
