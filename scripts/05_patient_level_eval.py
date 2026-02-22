#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voice_screening.analysis import (  # noqa: E402
    get_analysis_mode,
    get_deterministic_torch,
    get_final_refit_on_train_val,
    get_patient_selection_metric,
    get_tuning_split,
)
from voice_screening.config import load_config  # noqa: E402
from voice_screening.io import load_dataframe, save_json  # noqa: E402
from voice_screening.modeling import build_model, predict_positive_proba, seed_everything  # noqa: E402
from voice_screening.notebook_parity import (  # noqa: E402
    aggregate_features_by_cluster,
    aggregate_recordings_to_patient_level,
    clean_feature_dataframe,
    create_patient_labels,
    evaluate_binary_metrics,
    fit_cluster_assignments,
    participant_train_val_test_split,
    standardize_and_clip,
)
from voice_screening.repro import update_run_manifest  # noqa: E402
from voice_screening.run import build_run_dirs  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patient-level evaluation with explicit tuning/final split separation. "
            "Selection is performed on tuning split; final reported metrics are one-shot held-out test."
        )
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


def _patient_aggregation_methods(config: dict) -> list[str]:
    methods = config.get("modeling", {}).get("patient_aggregation_methods")
    if methods is None:
        methods = [config.get("modeling", {}).get("patient_aggregation", "mean")]

    ordered: list[str] = []
    for method in methods:
        method_str = str(method).strip().lower()
        if method_str and method_str not in ordered:
            ordered.append(method_str)
    return ordered or ["mean"]


def _patient_feature_sets(
    *,
    patient_agg: str,
    x_train_agg_recording: np.ndarray,
    x_val_agg_recording: np.ndarray,
    x_test_agg_recording: np.ndarray,
    pid_train: np.ndarray,
    pid_val: np.ndarray,
    pid_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_patient, pids_train_unique = aggregate_recordings_to_patient_level(
        x_recording_features=x_train_agg_recording,
        participant_ids=pid_train,
        aggregation_method=patient_agg,
    )
    x_val_patient, pids_val_unique = aggregate_recordings_to_patient_level(
        x_recording_features=x_val_agg_recording,
        participant_ids=pid_val,
        aggregation_method=patient_agg,
    )
    x_test_patient, pids_test_unique = aggregate_recordings_to_patient_level(
        x_recording_features=x_test_agg_recording,
        participant_ids=pid_test,
        aggregation_method=patient_agg,
    )

    y_train_patient, y_train_pids = create_patient_labels(pid_train, y_train)
    y_val_patient, y_val_pids = create_patient_labels(pid_val, y_val)
    y_test_patient, y_test_pids = create_patient_labels(pid_test, y_test)

    if not (
        (pids_train_unique == y_train_pids).all()
        and (pids_val_unique == y_val_pids).all()
        and (pids_test_unique == y_test_pids).all()
    ):
        raise RuntimeError(f"Participant alignment mismatch for patient aggregation method: {patient_agg}")

    return (
        x_train_patient,
        x_val_patient,
        x_test_patient,
        y_train_patient.astype(int),
        y_val_patient.astype(int),
        y_test_patient.astype(int),
        pids_test_unique.astype(str),
    )


def _fit_patient_model(
    *,
    model_name: str,
    hyperparameters: dict,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
):
    model_params = dict(hyperparameters.get(model_name, {}))
    model = build_model(model_name, model_params, random_seed=seed)
    has_val = x_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        try:
            model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
            return model
        except TypeError:
            pass
    model.fit(x_train, y_train)
    return model


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.phenotype_config)

    analysis_mode = get_analysis_mode(config)
    tuning_split = get_tuning_split(config)
    refit_on_train_val = get_final_refit_on_train_val(config)
    selection_metric = get_patient_selection_metric(config)

    phenotype_name = config["phenotype"]["name"]
    run_dirs = build_run_dirs(config["paths"]["outputs_root"], phenotype_name, args.run_id)

    best_config_path = run_dirs.cluster_models / "best_fine_tune_config.json"
    if not best_config_path.exists():
        raise FileNotFoundError(f"Missing best fine-tune config: {best_config_path}. Run script 04 first.")

    with best_config_path.open("r", encoding="utf-8") as f:
        best_config = json.load(f)

    best_k = int(best_config["n_clusters"])
    best_cluster_aggregation = str(best_config["aggregation"])

    df = load_dataframe(run_dirs.prepared / "recordings.parquet")
    feature_cols = pd.read_csv(run_dirs.prepared / "feature_columns.csv")["feature"].tolist()

    participant_col = config["columns"]["participant_id"]
    label_col = "label"
    seed = int(config["project"].get("random_seed", 42))
    seed_everything(seed, deterministic=get_deterministic_torch(config))

    x_full = df[feature_cols].copy()
    x_clean, usable_features, _ = clean_feature_dataframe(x_full)

    labels = df[label_col].astype(int).to_numpy()
    participant_ids = df[participant_col].astype(str).to_numpy()

    test_fraction, validation_fraction = _split_config(config)
    split = participant_train_val_test_split(
        participant_ids=participant_ids,
        labels=labels,
        test_size=test_fraction,
        validation_size_from_train_val=validation_fraction,
        random_seed=seed,
    )

    x_train_raw = x_clean.loc[split.train_mask].to_numpy(dtype=np.float32)
    x_val_raw = x_clean.loc[split.val_mask].to_numpy(dtype=np.float32)
    x_test_raw = x_clean.loc[split.test_mask].to_numpy(dtype=np.float32)

    y_train = labels[split.train_mask]
    y_val = labels[split.val_mask]
    y_test = labels[split.test_mask]

    pid_train = participant_ids[split.train_mask]
    pid_val = participant_ids[split.val_mask]
    pid_test = participant_ids[split.test_mask]

    x_train_scaled, x_val_scaled, x_test_scaled, _ = standardize_and_clip(
        x_train=x_train_raw,
        x_val=x_val_raw,
        x_test=x_test_raw,
        clip_abs=8.0,
    )

    assignments = fit_cluster_assignments(
        x_train_scaled=x_train_scaled,
        feature_names=usable_features,
        n_clusters=best_k,
        linkage_method=str(config.get("cluster", {}).get("linkage", "ward")),
    )

    x_train_agg_recording, x_val_agg_recording, x_test_agg_recording, _ = aggregate_features_by_cluster(
        x_train_scaled=x_train_scaled,
        x_val_scaled=x_val_scaled,
        x_test_scaled=x_test_scaled,
        feature_names=usable_features,
        assignments=assignments,
        method=best_cluster_aggregation,
    )

    patient_methods = _patient_aggregation_methods(config)
    threshold = float(config.get("modeling", {}).get("threshold", 0.5))

    model_names = [str(name).strip().lower() for name in config.get("modeling", {}).get("models", [])]
    if not model_names:
        raise RuntimeError("No models configured under modeling.models")

    hyperparameters = dict(config.get("hyperparameters", {}))

    selection_rows: list[dict[str, object]] = []

    for patient_agg in patient_methods:
        (
            x_train_patient,
            x_val_patient,
            x_test_patient,
            y_train_patient,
            y_val_patient,
            y_test_patient,
            _pids_test_unique,
        ) = _patient_feature_sets(
            patient_agg=patient_agg,
            x_train_agg_recording=x_train_agg_recording,
            x_val_agg_recording=x_val_agg_recording,
            x_test_agg_recording=x_test_agg_recording,
            pid_train=pid_train,
            pid_val=pid_val,
            pid_test=pid_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

        x_train_patient_scaled, x_val_patient_scaled, x_test_patient_scaled, _ = standardize_and_clip(
            x_train=x_train_patient,
            x_val=x_val_patient,
            x_test=x_test_patient,
            clip_abs=8.0,
        )

        x_tune = x_val_patient_scaled if tuning_split == "val" else x_test_patient_scaled
        y_tune = y_val_patient if tuning_split == "val" else y_test_patient

        for model_name in model_names:
            model = _fit_patient_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                seed=seed,
                x_train=x_train_patient_scaled,
                y_train=y_train_patient,
                x_val=x_val_patient_scaled,
                y_val=y_val_patient,
            )

            y_prob = predict_positive_proba(model, x_tune)
            metrics = evaluate_binary_metrics(y_tune, y_prob, threshold=threshold)
            row = {
                "model": model_name,
                "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "estimator_class": type(model).__name__,
                "patient_aggregation": patient_agg,
                "evaluation_split": tuning_split,
                "selection_metric": selection_metric,
                "n_features": int(x_train_patient_scaled.shape[1]),
                "threshold": float(threshold),
                "cluster_k": int(best_k),
                "cluster_aggregation": best_cluster_aggregation,
                "n_train_patients": int(len(y_train_patient)),
                "n_val_patients": int(len(y_val_patient)),
                "n_test_patients": int(len(y_test_patient)),
                **metrics,
            }
            if hasattr(model, "best_epoch_"):
                row["best_epoch"] = int(model.best_epoch_)
            selection_rows.append(row)

    selection_df = pd.DataFrame(selection_rows)
    if selection_df.empty:
        raise RuntimeError("Patient-level selection results are empty.")

    selection_df.to_csv(run_dirs.reports / "patient_level_selection_results.csv", index=False)

    selected_per_model = (
        selection_df.sort_values([selection_metric, "f1"], ascending=[False, False], na_position="last")
        .groupby("model", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected_per_model.to_csv(run_dirs.reports / "patient_level_locked_selection_per_model.csv", index=False)

    final_rows: list[dict[str, object]] = []
    final_pred_rows: list[pd.DataFrame] = []

    for _, selected in selected_per_model.iterrows():
        model_name = str(selected["model"])
        patient_agg = str(selected["patient_aggregation"])

        (
            x_train_patient,
            x_val_patient,
            x_test_patient,
            y_train_patient,
            y_val_patient,
            y_test_patient,
            pids_test_unique,
        ) = _patient_feature_sets(
            patient_agg=patient_agg,
            x_train_agg_recording=x_train_agg_recording,
            x_val_agg_recording=x_val_agg_recording,
            x_test_agg_recording=x_test_agg_recording,
            pid_train=pid_train,
            pid_val=pid_val,
            pid_test=pid_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

        if refit_on_train_val:
            x_train_ref = np.vstack([x_train_patient, x_val_patient]).astype(np.float32)
            y_train_ref = np.concatenate([y_train_patient, y_val_patient]).astype(int)

            x_train_final, _, x_test_final, _ = standardize_and_clip(
                x_train=x_train_ref,
                x_val=x_train_ref,
                x_test=x_test_patient,
                clip_abs=8.0,
            )

            model = _fit_patient_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                seed=seed,
                x_train=x_train_final,
                y_train=y_train_ref,
                x_val=None,
                y_val=None,
            )
            y_prob = predict_positive_proba(model, x_test_final)
            metrics = evaluate_binary_metrics(y_test_patient, y_prob, threshold=threshold)
            n_train_patients = int(len(y_train_ref))
            n_val_patients = 0
        else:
            x_train_final, x_val_final, x_test_final, _ = standardize_and_clip(
                x_train=x_train_patient,
                x_val=x_val_patient,
                x_test=x_test_patient,
                clip_abs=8.0,
            )
            model = _fit_patient_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                seed=seed,
                x_train=x_train_final,
                y_train=y_train_patient,
                x_val=x_val_final,
                y_val=y_val_patient,
            )
            y_prob = predict_positive_proba(model, x_test_final)
            metrics = evaluate_binary_metrics(y_test_patient, y_prob, threshold=threshold)
            n_train_patients = int(len(y_train_patient))
            n_val_patients = int(len(y_val_patient))

        y_pred = (y_prob >= threshold).astype(int)

        row = {
            "model": model_name,
            "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "estimator_class": type(model).__name__,
            "patient_aggregation": patient_agg,
            "evaluation_split": "test",
            "selection_split": tuning_split,
            "selection_metric": selection_metric,
            "selection_metric_value": float(selected.get(selection_metric, np.nan)),
            "n_features": int(x_train_patient.shape[1]),
            "threshold": float(threshold),
            "cluster_k": int(best_k),
            "cluster_aggregation": best_cluster_aggregation,
            "n_train_patients": n_train_patients,
            "n_val_patients": n_val_patients,
            "n_test_patients": int(len(y_test_patient)),
            **metrics,
        }
        if hasattr(model, "best_epoch_"):
            row["best_epoch"] = int(model.best_epoch_)

        final_rows.append(row)
        final_pred_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "patient_aggregation": patient_agg,
                    "evaluation_split": "test",
                    "selection_split": tuning_split,
                    "participant_id": pids_test_unique.astype(str),
                    "label": y_test_patient.astype(int),
                    "probability": y_prob.astype(float),
                    "prediction": y_pred.astype(int),
                    "threshold": float(threshold),
                    "cluster_k": int(best_k),
                    "cluster_aggregation": best_cluster_aggregation,
                }
            )
        )

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        raise RuntimeError("Patient-level final test results are empty.")
    final_df.to_csv(run_dirs.reports / "patient_level_final_test_results.csv", index=False)
    final_df.to_csv(run_dirs.reports / "patient_level_results.csv", index=False)

    if final_pred_rows:
        final_pred_df = pd.concat(final_pred_rows, ignore_index=True)
        final_pred_df.to_csv(run_dirs.reports / "patient_level_predictions.csv", index=False)

    final_df.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last").to_csv(
        run_dirs.reports / "patient_level_best_configurations.csv", index=False
    )
    final_df.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last").head(1).to_csv(
        run_dirs.reports / "patient_level_best_overall.csv", index=False
    )

    locked_selection = {
        "analysis_mode": analysis_mode,
        "selection_split": tuning_split,
        "selection_metric": selection_metric,
        "refit_on_train_val": bool(refit_on_train_val),
        "best_cluster_k": int(best_k),
        "best_cluster_aggregation": best_cluster_aggregation,
        "selected_aggregation_by_model": {
            str(row["model"]): str(row["patient_aggregation"]) for _, row in selected_per_model.iterrows()
        },
    }
    save_json(locked_selection, run_dirs.reports / "patient_level_locked_selection.json")

    summary_rows = [
        {
            "analysis_mode": analysis_mode,
            "selection_split": tuning_split,
            "selection_metric": selection_metric,
            "refit_on_train_val": bool(refit_on_train_val),
            "best_cluster_k": int(best_k),
            "best_cluster_aggregation": best_cluster_aggregation,
            "patient_aggregation_methods": patient_methods,
            "models": model_names,
            "n_selection_rows": int(len(selection_df)),
            "n_final_rows": int(len(final_df)),
        }
    ]
    pd.DataFrame(summary_rows).to_csv(run_dirs.reports / "patient_level_run_summary.csv", index=False)

    (
        final_df[["model", "model_display", "estimator_class"]]
        .drop_duplicates()
        .sort_values(["model", "estimator_class"])
        .reset_index(drop=True)
        .to_csv(run_dirs.reports / "patient_model_backends.csv", index=False)
    )

    update_run_manifest(
        path=run_dirs.root / "run_manifest.json",
        config=config,
        stage="05_patient_level_eval",
        run_id=args.run_id,
        phenotype=phenotype_name,
        extra={
            "analysis_mode": analysis_mode,
            "selection_split": tuning_split,
            "selection_metric": selection_metric,
            "selected_aggregation_by_model": locked_selection["selected_aggregation_by_model"],
        },
        cwd=ROOT,
    )

    print(f"Patient-level outputs saved under: {run_dirs.reports}")


if __name__ == "__main__":
    main()
