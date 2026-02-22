#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voice_screening.analysis import get_analysis_mode, get_deterministic_torch
from voice_screening.config import load_config
from voice_screening.evaluation import find_best_threshold
from voice_screening.io import load_dataframe, save_json
from voice_screening.modeling import build_model, predict_positive_proba, seed_everything
from voice_screening.notebook_parity import (
    clean_feature_dataframe,
    evaluate_binary_metrics,
    participant_train_val_test_split,
    summarize_split_class_balance,
    standardize_and_clip,
)
from voice_screening.repro import update_run_manifest
from voice_screening.run import build_run_dirs
from voice_screening.splitting import (
    create_participant_train_validation_split,
    create_patient_level_folds,
    fold_membership_table,
)


MODEL_DISPLAY_NAMES = {
    "mlp": "Neural Network",
    "residual_mlp": "Residual Neural Network",
    "wide_deep_mlp": "Wide+Deep Neural Network",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "logistic_regression": "Logistic Regression",
    "elastic_net_logistic": "Elastic-Net Logistic",
    "svc_rbf": "RBF SVM",
}

MODEL_SHORT_NAMES = {
    "mlp": "nn",
    "residual_mlp": "resnn",
    "wide_deep_mlp": "wdnn",
    "random_forest": "rf",
    "extra_trees": "et",
    "hist_gradient_boosting": "hgb",
    "logistic_regression": "lr",
    "elastic_net_logistic": "enet",
    "svc_rbf": "svm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Notebook-parity baseline training on raw acoustic features."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phenotype-config", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _split_config(config: dict) -> tuple[float, float]:
    split_cfg = config.get("modeling", {}).get("split", {})
    test_fraction = float(split_cfg.get("test_fraction", 0.2))
    validation_fraction = float(split_cfg.get("validation_fraction", 0.2))
    return test_fraction, validation_fraction


def _split_balance_trials(config: dict) -> int:
    split_cfg = config.get("modeling", {}).get("split", {})
    balance_cfg = split_cfg.get("balance", {})
    if isinstance(balance_cfg, dict):
        enabled = bool(balance_cfg.get("enabled", True))
        trials = int(balance_cfg.get("trials", 64))
    else:
        enabled = bool(balance_cfg) if balance_cfg is not None else True
        trials = 64
    return max(1, trials if enabled else 1)


def _cv_splits(config: dict) -> int:
    cv_cfg = config.get("modeling", {}).get("cv", {})
    return int(cv_cfg.get("n_splits", 5))


def _permutation_importance(
    *,
    model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    random_seed: int,
    n_iterations: int,
) -> pd.DataFrame:
    y_true = np.asarray(y_eval).astype(int)
    y_prob = predict_positive_proba(model, x_eval)
    baseline = evaluate_binary_metrics(y_true, y_prob, threshold=0.5).get("roc_auc", np.nan)

    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []

    for feat_idx, feat_name in enumerate(feature_names):
        drops: list[float] = []
        for _ in range(max(1, int(n_iterations))):
            permuted = np.array(x_eval, copy=True)
            perm_indices = rng.permutation(len(permuted))
            permuted[:, feat_idx] = permuted[perm_indices, feat_idx]
            perm_prob = predict_positive_proba(model, permuted)
            perm_auc = evaluate_binary_metrics(y_true, perm_prob, threshold=0.5).get("roc_auc", np.nan)
            if np.isfinite(baseline) and np.isfinite(perm_auc):
                drops.append(float(baseline - perm_auc))

        score = float(np.mean(drops)) if drops else float("nan")
        rows.append({"feature": feat_name, "importance": score})

    out = pd.DataFrame(rows).sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["metric"] = "roc_auc_drop_permutation"
    out["n_iterations"] = int(max(1, int(n_iterations)))
    return out


def _model_importance(
    *,
    model_name: str,
    model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    random_seed: int,
    n_iterations: int,
) -> pd.DataFrame:
    if model_name in {"random_forest", "extra_trees"} and hasattr(model, "feature_importances_"):
        arr = np.asarray(model.feature_importances_, dtype=float)
        out = pd.DataFrame({"feature": feature_names, "importance": arr})
        out = out.sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1, dtype=int)
        out["metric"] = "gini_importance"
        out["n_iterations"] = 0
        return out

    if model_name in {"logistic_regression", "elastic_net_logistic"} and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        coef_abs = np.abs(np.asarray(coef, dtype=float))
        out = pd.DataFrame({"feature": feature_names, "importance": coef_abs})
        out = out.sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1, dtype=int)
        out["metric"] = "abs_coefficient"
        out["n_iterations"] = 0
        return out

    return _permutation_importance(
        model=model,
        x_eval=x_eval,
        y_eval=y_eval,
        feature_names=feature_names,
        random_seed=random_seed,
        n_iterations=n_iterations,
    )


def _fit_model_with_optional_val(
    *,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> None:
    has_val = x_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        try:
            model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
            return
        except TypeError:
            pass
    model.fit(x_train, y_train)


def _fit_and_score(
    *,
    model_name: str,
    hyperparameters: dict,
    random_seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    x_test: np.ndarray,
    y_test: np.ndarray,
    default_threshold: float,
) -> tuple[dict[str, object], np.ndarray, object]:
    model_params = dict(hyperparameters.get(model_name, {}))
    model = build_model(model_name, model_params, random_seed=random_seed)

    has_val = x_val is not None and y_val is not None and len(y_val) > 0
    _fit_model_with_optional_val(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    threshold = float(default_threshold)
    tuned_f1 = float("nan")
    if has_val:
        val_prob = predict_positive_proba(model, x_val)
        threshold, tuned_f1 = find_best_threshold(
            y_true=np.asarray(y_val).astype(int),
            y_prob=np.asarray(val_prob).astype(float),
            default_threshold=default_threshold,
        )

    y_prob = predict_positive_proba(model, x_test)
    metrics = evaluate_binary_metrics(np.asarray(y_test).astype(int), y_prob, threshold=threshold)

    row: dict[str, object] = {
        "model": model_name,
        "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "estimator_class": type(model).__name__,
        "threshold": float(threshold),
        "val_f1_at_threshold": float(tuned_f1),
        **metrics,
    }
    if hasattr(model, "best_epoch_"):
        row["best_epoch"] = int(model.best_epoch_)
    return row, np.asarray(y_prob, dtype=float), model


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.phenotype_config)

    phenotype_name = config["phenotype"]["name"]
    run_dirs = build_run_dirs(config["paths"]["outputs_root"], phenotype_name, args.run_id)

    df = load_dataframe(run_dirs.prepared / "recordings.parquet")
    feature_cols = pd.read_csv(run_dirs.prepared / "feature_columns.csv")["feature"].tolist()

    participant_col = config["columns"]["participant_id"]
    label_col = "label"
    seed = int(config["project"].get("random_seed", 42))
    seed_everything(seed, deterministic=get_deterministic_torch(config))

    x_full = df[feature_cols].copy()
    x_clean, usable_features, removed_features = clean_feature_dataframe(x_full)

    labels = df[label_col].astype(int).to_numpy()
    participant_ids = df[participant_col].astype(str).to_numpy()

    threshold_default = float(config.get("modeling", {}).get("threshold", 0.5))
    model_names = [str(name).strip().lower() for name in config.get("modeling", {}).get("models", [])]
    if not model_names:
        raise RuntimeError("No models configured under modeling.models")
    hyperparameters = dict(config.get("hyperparameters", {}))

    # Notebook-style participant CV for recording-level model stability plots.
    cv_rows: list[dict[str, object]] = []
    n_splits = _cv_splits(config)
    folds = create_patient_level_folds(
        df=pd.DataFrame({participant_col: participant_ids, label_col: labels}),
        participant_col=participant_col,
        label_col=label_col,
        n_splits=n_splits,
        random_seed=seed,
    )
    fold_membership = fold_membership_table(
        df=pd.DataFrame({participant_col: participant_ids}),
        splits=folds,
        participant_col=participant_col,
    )
    fold_membership.to_csv(run_dirs.baseline / "fold_membership.csv", index=False)

    for split in folds:
        outer_train_idx = split.train_index
        outer_test_idx = split.test_index

        outer_train_df = pd.DataFrame(
            {
                participant_col: participant_ids[outer_train_idx],
                label_col: labels[outer_train_idx],
            },
            index=outer_train_idx,
        )
        inner_train_idx, inner_val_idx = create_participant_train_validation_split(
            df=outer_train_df,
            participant_col=participant_col,
            label_col=label_col,
            validation_fraction=_split_config(config)[1],
            random_seed=seed + split.fold_id,
        )

        if len(inner_val_idx) == 0:
            inner_val_idx = inner_train_idx
            has_distinct_val = False
        else:
            has_distinct_val = True

        x_train_cv_raw = x_clean.iloc[inner_train_idx].to_numpy(dtype=np.float32)
        x_val_cv_raw = x_clean.iloc[inner_val_idx].to_numpy(dtype=np.float32)
        x_test_cv_raw = x_clean.iloc[outer_test_idx].to_numpy(dtype=np.float32)

        y_train_cv = labels[inner_train_idx]
        y_val_cv = labels[inner_val_idx]
        y_test_cv = labels[outer_test_idx]

        x_train_cv, x_val_cv, x_test_cv, _ = standardize_and_clip(
            x_train=x_train_cv_raw,
            x_val=x_val_cv_raw,
            x_test=x_test_cv_raw,
            clip_abs=8.0,
        )

        for model_name in model_names:
            row, _, _ = _fit_and_score(
                model_name=model_name,
                hyperparameters=hyperparameters,
                random_seed=seed + split.fold_id,
                x_train=x_train_cv,
                y_train=y_train_cv,
                x_val=x_val_cv if has_distinct_val else None,
                y_val=y_val_cv if has_distinct_val else None,
                x_test=x_test_cv,
                y_test=y_test_cv,
                default_threshold=threshold_default,
            )
            row.update(
                {
                    "fold_id": int(split.fold_id),
                    "n_train_records": int(len(inner_train_idx)),
                    "n_val_records": int(len(inner_val_idx)),
                    "n_test_records": int(len(outer_test_idx)),
                    "n_train_participants": int(len(np.unique(participant_ids[inner_train_idx]))),
                    "n_val_participants": int(len(np.unique(participant_ids[inner_val_idx]))),
                    "n_test_participants": int(len(np.unique(participant_ids[outer_test_idx]))),
                }
            )
            cv_rows.append(row)

    fold_metrics_df = pd.DataFrame(cv_rows)
    fold_metrics_df.to_csv(run_dirs.baseline / "fold_metrics.csv", index=False)

    metric_cols = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    cv_summary = (
        fold_metrics_df.groupby(["model", "model_display"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv_summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in cv_summary.columns
    ]
    cv_summary.to_csv(run_dirs.baseline / "metric_summary.csv", index=False)

    # Final notebook-style participant split for primary reporting and prediction plots.
    test_fraction, validation_fraction = _split_config(config)
    balance_trials = _split_balance_trials(config)
    split = participant_train_val_test_split(
        participant_ids=participant_ids,
        labels=labels,
        test_size=test_fraction,
        validation_size_from_train_val=validation_fraction,
        random_seed=seed,
        balance_trials=balance_trials,
    )

    x_train_raw = x_clean.loc[split.train_mask].to_numpy(dtype=np.float32)
    x_val_raw = x_clean.loc[split.val_mask].to_numpy(dtype=np.float32)
    x_test_raw = x_clean.loc[split.test_mask].to_numpy(dtype=np.float32)

    y_train = labels[split.train_mask]
    y_val = labels[split.val_mask]
    y_test = labels[split.test_mask]

    pid_test = participant_ids[split.test_mask]
    test_record_indices = np.where(split.test_mask)[0].astype(int)

    x_train_scaled, x_val_scaled, x_test_scaled, _ = standardize_and_clip(
        x_train=x_train_raw,
        x_val=x_val_raw,
        x_test=x_test_raw,
        clip_abs=8.0,
    )

    final_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    importance_iterations = int(
        config.get("reporting", {}).get("permutation_importance_iterations", 5)
    )

    for model_name in model_names:
        row, y_prob, model = _fit_and_score(
            model_name=model_name,
            hyperparameters=hyperparameters,
            random_seed=seed,
            x_train=x_train_scaled,
            y_train=y_train,
            x_val=x_val_scaled,
            y_val=y_val,
            x_test=x_test_scaled,
            y_test=y_test,
            default_threshold=threshold_default,
        )
        row.update(
            {
                "feature_space": "raw",
                "n_features": int(x_train_scaled.shape[1]),
                "n_train_records": int(len(y_train)),
                "n_val_records": int(len(y_val)),
                "n_test_records": int(len(y_test)),
                "n_train_participants": int(len(split.train_participants)),
                "n_val_participants": int(len(split.val_participants)),
                "n_test_participants": int(len(split.test_participants)),
            }
        )
        final_rows.append(row)

        y_pred = (y_prob >= float(row["threshold"])).astype(int)
        pred_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "participant_id": pid_test,
                    "record_index": test_record_indices,
                    "label": y_test.astype(int),
                    "probability": y_prob.astype(float),
                    "prediction": y_pred.astype(int),
                    "threshold": float(row["threshold"]),
                }
            )
        )

        importance_df = _model_importance(
            model_name=model_name,
            model=model,
            x_eval=x_val_scaled,
            y_eval=y_val,
            feature_names=usable_features,
            random_seed=seed,
            n_iterations=importance_iterations,
        )
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)
        importance_df.to_csv(run_dirs.baseline / f"{short_name}_feature_importance.csv", index=False)
        if hasattr(model, "history_"):
            save_json(model.history_, run_dirs.baseline / f"{short_name}_training_history.json")
        if model_name == "mlp":
            importance_df.to_csv(run_dirs.baseline / "feature_importance.csv", index=False)
            if hasattr(model, "history_"):
                save_json(model.history_, run_dirs.baseline / "nn_training_history.json")
        if model_name == "random_forest":
            importance_df.to_csv(run_dirs.baseline / "rf_feature_importance.csv", index=False)

    baseline_df = pd.DataFrame(final_rows)
    baseline_df.to_csv(run_dirs.baseline / "baseline_results.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(run_dirs.baseline / "recording_predictions.csv", index=False)
    (
        baseline_df[["model", "model_display", "estimator_class"]]
        .drop_duplicates()
        .sort_values(["model", "estimator_class"])
        .reset_index(drop=True)
        .to_csv(run_dirs.baseline / "model_backends.csv", index=False)
    )

    label_cfg = config.get("labeling", {})
    class_balance = summarize_split_class_balance(
        participant_ids=participant_ids,
        labels=labels,
        split=split,
        positive_label=1,
        positive_name=str(label_cfg.get("positive_class_name", "positive")),
        negative_name=str(label_cfg.get("negative_class_name", "control")),
    )

    split_summary = {
        "analysis_mode": get_analysis_mode(config),
        "train_recordings": int(split.train_mask.sum()),
        "val_recordings": int(split.val_mask.sum()),
        "test_recordings": int(split.test_mask.sum()),
        "train_participants": int(len(split.train_participants)),
        "val_participants": int(len(split.val_participants)),
        "test_participants": int(len(split.test_participants)),
        "cv_n_splits": int(n_splits),
        "split_seed": split.split_seed,
        "split_balance_score": split.balance_score,
        "split_balance_trials": int(split.balance_trials),
        "class_balance": class_balance,
    }
    save_json(split_summary, run_dirs.baseline / "split_summary.json")

    pd.DataFrame({"feature": usable_features}).to_csv(run_dirs.baseline / "feature_columns_used.csv", index=False)
    save_json(
        {
            "feature_count_input": int(len(feature_cols)),
            "feature_count_used": int(len(usable_features)),
            "removed_non_numeric_features": removed_features,
            "permutation_importance_iterations": int(importance_iterations),
        },
        run_dirs.baseline / "feature_usage.json",
    )
    update_run_manifest(
        path=run_dirs.root / "run_manifest.json",
        config=config,
        stage="03_train_baseline",
        run_id=args.run_id,
        phenotype=phenotype_name,
        extra={"models": model_names, "cv_n_splits": int(n_splits)},
        cwd=ROOT,
    )

    print(f"Baseline outputs saved under: {run_dirs.baseline}")


if __name__ == "__main__":
    main()
