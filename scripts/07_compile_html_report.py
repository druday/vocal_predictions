#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voice_screening.analysis import get_analysis_mode, get_tuning_split
from voice_screening.config import load_config
from voice_screening.repro import update_run_manifest
from voice_screening.run import build_run_dirs


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


ACOUSTIC_FIGURES: list[tuple[str, str, str]] = [
    ("feature_category_counts.png", "Acoustic feature category counts.", "Feature inventory across acoustic families."),
    ("voice_exercises_analysis.png", "Voice exercise/task distribution.", "Task-level composition of the dataset."),
    (
        "disease_condition_cooccurrence_lower_triangle.png",
        "Disease-condition co-occurrence.",
        "Phenotype context and overlap matrix.",
    ),
    ("static_features_mfcc.png", "Static MFCC profile.", "MFCC feature family distribution."),
    ("static_features_prosody.png", "Static prosody profile.", "Prosodic feature family distribution."),
    ("static_features_formant.png", "Static formant profile.", "Formant feature family distribution."),
    ("static_features_energy.png", "Static energy profile.", "Energy feature family distribution."),
    ("static_features_other.png", "Static other-features profile.", "Remaining feature families."),
]

COARSE_CONTENDER_FIGURES: list[tuple[str, str, str]] = [
    (
        "performance_vs_clusters.png",
        "Coarse K sweep by model and aggregation.",
        "Recording-level coarse exploration over K.",
    ),
    (
        "performance_heatmap.png",
        "Coarse K heatmap.",
        "ROC-AUC across K, aggregation, and model.",
    ),
    (
        "fig_coarse_k_contenders_selection_scope.png",
        "Small-K contender region.",
        "Near-top K region targeted for explainability + generalization.",
    ),
]

FINE_TUNE_FIGURES: list[tuple[str, str, str]] = [
    (
        "fine_tune_performance_13_17.png",
        "Fine-tune local K sweep.",
        "Performance in the refined K neighborhood.",
    ),
]

K_VALIDATION_FIGURES: list[tuple[str, str, str]] = [
    (
        "cluster_validation_silhouette_cophenetic.png",
        "Silhouette + cophenetic diagnostics.",
        "Cluster validity trend across K.",
    ),
    (
        "cluster_validation_elbow_curve.png",
        "Elbow compactness proxy.",
        "Within-cluster distance and diminishing returns vs K.",
    ),
    (
        "cluster_validation_stability_bootstrap_ari.png",
        "Bootstrap stability (ARI).",
        "Feature-assignment stability across train resamples.",
    ),
    (
        "feature_clustering_dendrogram.png",
        "Feature dendrogram.",
        "Hierarchical structure of acoustic features.",
    ),
]

RECORDING_FIGURES: list[tuple[str, str, str]] = [
    ("model_comparison.png", "Baseline model comparison.", "Accuracy, ROC-AUC, and F1 at recording level."),
    (
        "fig_baseline_vs_best_cluster.png",
        "Baseline vs locked clustered.",
        "Per-model gains/losses at locked final-test K.",
    ),
    ("nn_final_roc_curve.png", "Neural network ROC.", "Recording-level ROC for the primary DL model."),
    ("nn_final_pr_curve.png", "Neural network PR.", "Recording-level precision-recall performance."),
    ("nn_final_confusion_matrix.png", "Neural network confusion matrix.", "Classification errors at recording level."),
    (
        "final_summary_all_models_levels.png",
        "Final cross-level summary heatmap.",
        "Recording vs patient-level overview.",
    ),
]

PATIENT_FIGURES: list[tuple[str, str, str]] = [
    (
        "patient_level_performance_comparison.png",
        "Aggregation strategy comparison.",
        "Patient-level ROC-AUC by model and aggregation method.",
    ),
    (
        "patient_level_metrics_line_charts.png",
        "Patient-level metric trajectories.",
        "Accuracy, ROC-AUC, and F1 trends by aggregation.",
    ),
    ("patient_level_confusion_roc.png", "Best patient configuration diagnostics.", "Confusion matrix and ROC for best patient setup."),
    (
        "recording_vs_patient_level_comparison.png",
        "Recording vs patient-level comparison.",
        "Direct comparison of held-out test performance.",
    ),
]

PROGRESSIVE_FIGURES: list[tuple[str, str, str]] = [
    (
        "progressive_selection_performance.png",
        "Progressive feature selection trajectory.",
        "Performance behavior as feature count is reduced.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a deterministic HTML report from pipeline artifacts for a given run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phenotype-config", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _safe_read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_json(path: Path, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required JSON: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _format_df_for_html(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].apply(_format_scalar)
        else:
            out[col] = out[col].fillna("").astype(str)
    return out


def _df_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return '<p class="muted">No data available for this section.</p>'
    show = df.head(max_rows).copy() if max_rows is not None else df.copy()
    show = _format_df_for_html(show)
    return show.to_html(index=False, classes="data-table", border=0, escape=False)


def _safe_model_display(model_name: Any) -> str:
    key = str(model_name)
    return MODEL_DISPLAY_NAMES.get(key, key)


def _link_to_path(from_dir: Path, target: Path, label: str | None = None) -> str:
    rel = target.relative_to(from_dir) if target.is_relative_to(from_dir) else target
    label_text = label or str(rel)
    return f'<a href="{html.escape(str(rel))}">{html.escape(label_text)}</a>'


def _figure_html(run_dirs, filename: str, title: str, caption: str) -> str:
    fig_path = run_dirs.reports / filename
    if not fig_path.exists():
        return ""
    return (
        '<figure class="panel-figure">'
        f'<img src="{html.escape(filename)}" alt="{html.escape(title)}" loading="lazy" />'
        f"<figcaption><strong>{html.escape(title)}</strong><br>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _figures_panel_html(run_dirs, figures: list[tuple[str, str, str]]) -> str:
    blocks: list[str] = []
    for filename, title, caption in figures:
        block = _figure_html(run_dirs, filename, title, caption)
        if block:
            blocks.append(block)
    if not blocks:
        return '<p class="muted">No figures available for this section.</p>'
    return '<div class="panel-grid">' + "".join(blocks) + "</div>"


def _summary_cards_html(cards: list[tuple[str, str]]) -> str:
    chunks = []
    for title, value in cards:
        chunks.append(
            '<div class="card">'
            f'<div class="card-title">{html.escape(title)}</div>'
            f'<div class="card-value">{html.escape(value)}</div>'
            "</div>"
        )
    return '<div class="card-grid">' + "".join(chunks) + "</div>"


def _key_findings_html(findings: list[str]) -> str:
    if not findings:
        return '<p class="muted">No findings generated.</p>'
    items = "".join(f"<li>{html.escape(line)}</li>" for line in findings)
    return f"<ul>{items}</ul>"


def _collapsible(title: str, body_html: str, *, open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f'<details class="table-collapse"{open_attr}>'
        f"<summary>{html.escape(title)}</summary>"
        f'<div class="table-collapse-body">{body_html}</div>'
        "</details>"
    )


def _resolve_named_class_counts(
    dataset_summary: dict[str, Any],
    *,
    default_positive_name: str = "positive",
    default_negative_name: str = "control",
) -> tuple[str, str, int, int, int | None, int | None]:
    positive_name = str(dataset_summary.get("positive_class_name", default_positive_name))
    negative_name = str(dataset_summary.get("negative_class_name", default_negative_name))

    class_counts = dataset_summary.get("class_counts", {})
    participant_counts = dataset_summary.get("participant_class_counts", {})
    class_counts_named = dataset_summary.get("class_counts_named", {})
    participant_counts_named = dataset_summary.get("participant_class_counts_named", {})

    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    rec_positive = _safe_int(class_counts_named.get(positive_name, class_counts.get("1")))
    rec_negative = _safe_int(class_counts_named.get(negative_name, class_counts.get("0")))
    pid_positive = _safe_int(participant_counts_named.get(positive_name, participant_counts.get("1")))
    pid_negative = _safe_int(participant_counts_named.get(negative_name, participant_counts.get("0")))
    rec_positive = 0 if rec_positive is None else rec_positive
    rec_negative = 0 if rec_negative is None else rec_negative
    return positive_name, negative_name, rec_positive, rec_negative, pid_positive, pid_negative


def _split_balance_rows(split_summary: dict[str, Any], level_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    balance = split_summary.get("class_balance", {})
    split_payload = balance.get("splits", {})
    for split_name in ("train", "val", "test"):
        split_data = split_payload.get(split_name, {})
        rec = split_data.get("recordings", {})
        pid = split_data.get("participants", {})
        if not rec and not pid:
            continue
        rows.append(
            {
                "level": level_name,
                "split": split_name,
                "participant_positive": pid.get("n_positive"),
                "participant_negative": pid.get("n_negative"),
                "participant_positive_rate": pid.get("positive_rate"),
                "recording_positive": rec.get("n_positive"),
                "recording_negative": rec.get("n_negative"),
                "recording_positive_rate": rec.get("positive_rate"),
            }
        )
    return rows


def _top_hyperparam_keys(model_name: str) -> list[str]:
    if model_name in {"mlp", "residual_mlp", "wide_deep_mlp"}:
        return ["hidden_dims", "lr", "weight_decay", "dropout_rate", "epochs", "patience", "batch_size"]
    if model_name in {"random_forest", "extra_trees"}:
        return ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "class_weight"]
    if model_name == "hist_gradient_boosting":
        return ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "validation_fraction"]
    if model_name == "svc_rbf":
        return ["C", "kernel", "gamma", "class_weight", "probability"]
    if model_name == "elastic_net_logistic":
        return ["loss", "penalty", "alpha", "l1_ratio", "max_iter", "class_weight"]
    if model_name == "logistic_regression":
        return ["C", "penalty", "solver", "max_iter", "class_weight"]
    return []


def _build_model_hyperparam_table(
    *,
    config: dict[str, Any],
    backend_df: pd.DataFrame,
) -> pd.DataFrame:
    model_names = [str(v).strip().lower() for v in config.get("modeling", {}).get("models", [])]
    hyper = config.get("hyperparameters", {})

    class_map: dict[str, str] = {}
    if not backend_df.empty and {"model", "estimator_class"}.issubset(backend_df.columns):
        for _, row in backend_df.iterrows():
            model = str(row.get("model", "")).strip().lower()
            estimator = str(row.get("estimator_class", "")).strip()
            if model and estimator and model not in class_map:
                class_map[model] = estimator

    rows: list[dict[str, str]] = []
    for model_name in model_names:
        params = hyper.get(model_name, {})
        if not isinstance(params, dict):
            params = {}
        keys = _top_hyperparam_keys(model_name)
        selected_items = [(k, params[k]) for k in keys if k in params]
        if not selected_items:
            selected_items = list(params.items())[:8]
        key_summary = ", ".join(f"{k}={v}" for k, v in selected_items) if selected_items else ""
        rows.append(
            {
                "model": model_name,
                "model_display": _safe_model_display(model_name),
                "estimator_class": class_map.get(model_name, ""),
                "key_hyperparameters": key_summary,
                "all_hyperparameters_json": json.dumps(params, ensure_ascii=True, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _build_recording_tables(
    baseline_df: pd.DataFrame,
    final_cluster_df: pd.DataFrame,
    all_cluster_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline_df.copy()
    if baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "model_display" not in baseline.columns and "model" in baseline.columns:
        baseline["model_display"] = baseline["model"].map(_safe_model_display)

    baseline_cols = [
        c
        for c in [
            "model_display",
            "estimator_class",
            "threshold",
            "accuracy",
            "roc_auc",
            "f1",
            "precision",
            "recall",
        ]
        if c in baseline.columns
    ]
    baseline_table = (
        baseline[baseline_cols]
        .sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )

    cluster_best = pd.DataFrame()
    if not final_cluster_df.empty and "model" in final_cluster_df.columns:
        cluster_best = final_cluster_df.copy().reset_index(drop=True)
    elif not all_cluster_df.empty and "model" in all_cluster_df.columns:
        fallback = all_cluster_df.copy()
        if "evaluation_split" in fallback.columns:
            fallback = fallback[fallback["evaluation_split"] == "test"].copy()
        if not fallback.empty:
            cluster_best = (
                fallback.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last")
                .groupby("model", as_index=False, sort=False)
                .head(1)
                .reset_index(drop=True)
            )

    if cluster_best.empty:
        return baseline_table, pd.DataFrame()

    merged = baseline.merge(cluster_best, on="model", suffixes=("_baseline", "_cluster"))
    if merged.empty:
        return baseline_table, pd.DataFrame()

    merged["model_display"] = merged["model"].map(_safe_model_display)
    merged["delta_roc_auc"] = merged["roc_auc_cluster"] - merged["roc_auc_baseline"]
    merged["delta_accuracy"] = merged["accuracy_cluster"] - merged["accuracy_baseline"]
    merged["delta_f1"] = merged["f1_cluster"] - merged["f1_baseline"]

    cols = [
        "model_display",
        "n_clusters",
        "aggregation",
        "roc_auc_baseline",
        "roc_auc_cluster",
        "delta_roc_auc",
        "accuracy_baseline",
        "accuracy_cluster",
        "delta_accuracy",
        "f1_baseline",
        "f1_cluster",
        "delta_f1",
    ]
    cols = [c for c in cols if c in merged.columns]
    comparison = merged[cols].sort_values("roc_auc_cluster", ascending=False, na_position="last")
    return baseline_table, comparison.reset_index(drop=True)


def _build_patient_tables(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if patient_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame = patient_df.copy()
    if "model_display" not in frame.columns and "model" in frame.columns:
        frame["model_display"] = frame["model"].map(_safe_model_display)

    ranked = frame.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last")

    per_model = (
        ranked.groupby("model_display", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    keep_cols = [
        "model_display",
        "estimator_class",
        "patient_aggregation",
        "evaluation_split",
        "accuracy",
        "roc_auc",
        "f1",
        "precision",
        "recall",
    ]
    keep_cols = [c for c in keep_cols if c in ranked.columns]
    ranked = ranked[keep_cols].reset_index(drop=True)
    per_model = per_model[[c for c in keep_cols if c in per_model.columns]]
    return per_model, ranked


def _build_cluster_profile_tables(cluster_profiles_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if cluster_profiles_df.empty:
        return pd.DataFrame(), []

    table = cluster_profiles_df.copy()
    for col in ["cluster_id", "n_features"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    if "cluster_importance" in table.columns:
        table["cluster_importance"] = pd.to_numeric(table["cluster_importance"], errors="coerce")

    keep_cols = [
        "cluster_id",
        "cluster_importance",
        "n_features",
        "dominant_category",
        "category_breakdown",
        "example_features",
    ]
    keep_cols = [c for c in keep_cols if c in table.columns]
    table = table[keep_cols].copy()
    if "cluster_importance" in table.columns:
        table = table.sort_values(["cluster_importance", "n_features", "cluster_id"], ascending=[False, False, True], na_position="last")
    else:
        table = table.sort_values(["n_features", "cluster_id"], ascending=[False, True], na_position="last")
    table = table.reset_index(drop=True)

    brief: list[str] = []
    top_rows = table.head(5)
    for _, row in top_rows.iterrows():
        cid = int(row["cluster_id"]) if not pd.isna(row.get("cluster_id")) else None
        dom = str(row.get("dominant_category", "")).strip()
        n_feat = int(row["n_features"]) if not pd.isna(row.get("n_features")) else None
        imp = _format_scalar(row.get("cluster_importance"))
        prefix = f"Cluster {cid}" if cid is not None else "Cluster"
        if dom and n_feat is not None:
            brief.append(f"{prefix}: {dom}-dominant ({n_feat} features, importance={imp}).")
        elif dom:
            brief.append(f"{prefix}: {dom}-dominant.")
        else:
            brief.append(f"{prefix}: profile available in table.")

    return table, brief


def _run_snapshot(
    *,
    run_dirs,
    phenotype_name: str,
    run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    resolved_config = _safe_read_json(run_dirs.prepared / "resolved_config.json")
    effective_config = resolved_config if isinstance(resolved_config, dict) and resolved_config else config
    dataset_summary = _safe_read_json(run_dirs.prepared / "dataset_summary.json")
    baseline_split = _safe_read_json(run_dirs.baseline / "split_summary.json")
    cluster_split = _safe_read_json(run_dirs.cluster_models / "split_summary.json")
    k_selection = _safe_read_json(run_dirs.cluster_models / "k_selection.json")
    best_fine = _safe_read_json(run_dirs.cluster_models / "best_fine_tune_config.json")
    locked_selection = _safe_read_json(run_dirs.cluster_models / "locked_selection.json")
    patient_locked_selection = _safe_read_json(run_dirs.reports / "patient_level_locked_selection.json")

    baseline_df = _safe_read_csv(run_dirs.baseline / "baseline_results.csv")
    all_cluster_df = _safe_read_csv(run_dirs.cluster_models / "all_recording_results.csv")
    final_cluster_df = _safe_read_csv(run_dirs.cluster_models / "final_test_results.csv")
    fine_df = _safe_read_csv(run_dirs.cluster_models / "fine_tune_results.csv")
    coarse_contenders_df = _safe_read_csv(run_dirs.cluster_models / "coarse_k_contenders_selection_scope.csv")
    patient_df = _safe_read_csv(run_dirs.reports / "patient_level_results.csv")
    patient_selection_df = _safe_read_csv(run_dirs.reports / "patient_level_selection_results.csv")
    progressive_df = _safe_read_csv(run_dirs.reports / "progressive_selection_summary.csv")
    cluster_diag_df = _safe_read_csv(run_dirs.clusters / "clustering_diagnostics.csv")
    elbow_df = _safe_read_csv(run_dirs.reports / "cluster_validation_elbow_metrics.csv")
    stability_df = _safe_read_csv(run_dirs.reports / "cluster_validation_stability_bootstrap.csv")
    figure_manifest_df = _safe_read_csv(run_dirs.reports / "figure_manifest.csv")
    final_summary_df = _safe_read_csv(run_dirs.reports / "final_summary_all_models_levels.csv")
    cluster_profiles_df = _safe_read_csv(run_dirs.cluster_models / "cluster_profiles_best.csv")

    backends_baseline = _safe_read_csv(run_dirs.baseline / "model_backends.csv")
    backends_cluster = _safe_read_csv(run_dirs.cluster_models / "model_backends.csv")
    backends_patient = _safe_read_csv(run_dirs.reports / "patient_model_backends.csv")
    backend_union = pd.concat(
        [df for df in [backends_baseline, backends_cluster, backends_patient] if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in [backends_baseline, backends_cluster, backends_patient]) else pd.DataFrame()
    if not backend_union.empty:
        backend_union = backend_union.drop_duplicates().reset_index(drop=True)
    model_hyperparams_df = _build_model_hyperparam_table(config=effective_config, backend_df=backend_union)

    baseline_table, rec_comparison = _build_recording_tables(
        baseline_df,
        final_cluster_df,
        all_cluster_df,
    )
    patient_best_per_model, patient_ranked = _build_patient_tables(patient_df)
    patient_selection_best_per_model, patient_selection_ranked = _build_patient_tables(patient_selection_df)
    cluster_profile_table, cluster_profile_brief = _build_cluster_profile_tables(cluster_profiles_df)

    fine_top = pd.DataFrame()
    if not fine_df.empty:
        fine_top = (
            fine_df.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last")
            .head(12)
            .copy()
        )
        keep = [
            "n_clusters",
            "aggregation",
            "evaluation_split",
            "model_display",
            "estimator_class",
            "roc_auc",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "best_epoch",
        ]
        keep = [c for c in keep if c in fine_top.columns]
        fine_top = fine_top[keep].reset_index(drop=True)

    if not coarse_contenders_df.empty:
        keep = [
            "n_clusters",
            "evaluation_split",
            "roc_auc",
            "accuracy",
            "f1",
            "is_near_top",
            "within_preferred_range",
        ]
        keep = [c for c in keep if c in coarse_contenders_df.columns]
        coarse_contenders_df = coarse_contenders_df[keep].reset_index(drop=True)

    if not final_summary_df.empty:
        final_cols = [
            "model_display",
            "recording_roc_auc",
            "patient_roc_auc",
            "recording_accuracy",
            "patient_accuracy",
            "recording_f1",
            "patient_f1",
        ]
        final_cols = [c for c in final_cols if c in final_summary_df.columns]
        final_summary_df = final_summary_df[final_cols].reset_index(drop=True)

    seed = resolved_config.get("project", {}).get(
        "random_seed",
        config.get("project", {}).get("random_seed", 42),
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phenotype": phenotype_name,
        "run_id": run_id,
        "random_seed": seed,
        "dataset_summary": dataset_summary,
        "baseline_split": baseline_split,
        "cluster_split": cluster_split,
        "k_selection": k_selection,
        "best_fine": best_fine,
        "locked_selection": locked_selection,
        "patient_locked_selection": patient_locked_selection,
        "baseline_table": baseline_table,
        "recording_comparison": rec_comparison,
        "fine_top": fine_top,
        "coarse_contenders": coarse_contenders_df,
        "patient_best_per_model": patient_best_per_model,
        "patient_ranked": patient_ranked,
        "patient_selection_best_per_model": patient_selection_best_per_model,
        "patient_selection_ranked": patient_selection_ranked,
        "progressive_summary": progressive_df,
        "cluster_diagnostics": cluster_diag_df,
        "cluster_elbow_metrics": elbow_df,
        "cluster_stability": stability_df,
        "model_hyperparameters": model_hyperparams_df,
        "figure_manifest": figure_manifest_df,
        "final_summary": final_summary_df,
        "cluster_profiles": cluster_profiles_df,
        "cluster_profile_table": cluster_profile_table,
        "cluster_profile_brief": cluster_profile_brief,
        "backend_tables": {
            "baseline": backends_baseline,
            "cluster": backends_cluster,
            "patient": backends_patient,
        },
    }


def _artifact_table(run_dirs) -> pd.DataFrame:
    report_dir = run_dirs.reports
    artifacts = [
        ("Prepared dataset summary", run_dirs.prepared / "dataset_summary.json"),
        ("Prepared resolved config", run_dirs.prepared / "resolved_config.json"),
        ("Run manifest", run_dirs.root / "run_manifest.json"),
        ("Baseline results", run_dirs.baseline / "baseline_results.csv"),
        ("Cluster coarse results", run_dirs.cluster_models / "coarse_results.csv"),
        ("Cluster fine-tune results", run_dirs.cluster_models / "fine_tune_results.csv"),
        ("Cluster final held-out test", run_dirs.cluster_models / "final_test_results.csv"),
        ("Cluster diagnostics (silhouette/cophenetic)", run_dirs.clusters / "clustering_diagnostics.csv"),
        ("Cluster elbow metrics", run_dirs.reports / "cluster_validation_elbow_metrics.csv"),
        ("Cluster stability metrics", run_dirs.reports / "cluster_validation_stability_bootstrap.csv"),
        ("Best fine-tune config", run_dirs.cluster_models / "best_fine_tune_config.json"),
        ("K selection metadata", run_dirs.cluster_models / "k_selection.json"),
        ("Locked clustered selection", run_dirs.cluster_models / "locked_selection.json"),
        ("Cluster profile summary", run_dirs.cluster_models / "cluster_profiles_best.csv"),
        ("Patient selection results", run_dirs.reports / "patient_level_selection_results.csv"),
        ("Patient-level results", run_dirs.reports / "patient_level_results.csv"),
        ("Patient locked selection", run_dirs.reports / "patient_level_locked_selection.json"),
        ("Figure manifest", run_dirs.reports / "figure_manifest.csv"),
    ]
    rows: list[dict[str, str]] = []
    for label, path in artifacts:
        if not path.exists():
            continue
        rel = Path(os.path.relpath(path.resolve(), start=report_dir.resolve()))
        rows.append(
            {
                "artifact": label,
                "path": f'<a href="{html.escape(str(rel))}">{html.escape(str(rel))}</a>',
            }
        )
    return pd.DataFrame(rows)


def _table_with_links_for_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    if manifest_df.empty:
        return pd.DataFrame()

    out = manifest_df.copy().sort_values("order", ascending=True).reset_index(drop=True)
    keep = ["order", "section", "filename", "description", "ppt_hint"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]
    if "filename" in out.columns:
        out["filename"] = out["filename"].apply(
            lambda x: f'<a href="{html.escape(str(x))}">{html.escape(str(x))}</a>'
        )
    return out


def _build_findings(snapshot: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    dataset_summary = snapshot.get("dataset_summary", {})
    baseline_table = snapshot["baseline_table"]
    best_fine = snapshot["best_fine"]
    patient_ranked = snapshot["patient_ranked"]
    patient_selection_ranked = snapshot.get("patient_selection_ranked", pd.DataFrame())
    k_selection = snapshot["k_selection"]
    locked_selection = snapshot.get("locked_selection", {})
    patient_locked_selection = snapshot.get("patient_locked_selection", {})
    progressive_df = snapshot["progressive_summary"]
    cluster_profile_brief = snapshot.get("cluster_profile_brief", [])

    pos_name, neg_name, rec_pos, rec_neg, pid_pos, pid_neg = _resolve_named_class_counts(
        dataset_summary,
        default_positive_name=str(snapshot.get("phenotype", "positive")),
        default_negative_name="control",
    )
    if (rec_pos + rec_neg) > 0 and pid_pos is not None and pid_neg is not None:
        findings.append(
            "Cohort composition: "
            f"{pos_name}={rec_pos} recordings / {pid_pos} participants, "
            f"{neg_name}={rec_neg} recordings / {pid_neg} participants."
        )
    elif (rec_pos + rec_neg) > 0:
        findings.append(
            "Cohort composition: "
            f"{pos_name}={rec_pos} recordings, {neg_name}={rec_neg} recordings."
        )

    if not baseline_table.empty:
        top_baseline = baseline_table.sort_values(
            ["roc_auc", "f1"], ascending=[False, False], na_position="last"
        ).iloc[0]
        findings.append(
            "Best recording-level baseline by ROC-AUC: "
            f"{top_baseline['model_display']} ({_format_scalar(top_baseline.get('roc_auc'))})."
        )

    if best_fine:
        findings.append(
            "Locked clustered recording-level configuration: "
            f"K={best_fine.get('n_clusters')}, aggregation={best_fine.get('aggregation')}, "
            f"model={best_fine.get('model_display', best_fine.get('model'))}, "
            f"final-test ROC-AUC={_format_scalar(_float_or_nan(best_fine.get('roc_auc')))}."
        )
        if best_fine.get("selection_roc_auc") is not None:
            findings.append(
                "Selection split metric for locked clustered configuration: "
                f"ROC-AUC={_format_scalar(_float_or_nan(best_fine.get('selection_roc_auc')))} "
                f"on {best_fine.get('selection_evaluation_split', 'tuning')}."
            )

    if k_selection:
        contender_list = k_selection.get("coarse_contender_k_values", [])
        findings.append(
            f"Small-K contender set from coarse sweep: {contender_list} "
            f"(selected center K={k_selection.get('selected_k')}, "
            f"tuning split={k_selection.get('tuning_split', 'tuning')})."
        )

    class_balance = snapshot.get("baseline_split", {}).get("class_balance", {})
    split_payload = class_balance.get("splits", {})
    if split_payload:
        global_pid_rate = class_balance.get("overall", {}).get("participants", {}).get("positive_rate")
        rates = []
        for split_name in ("train", "val", "test"):
            rate = split_payload.get(split_name, {}).get("participants", {}).get("positive_rate")
            if rate is not None:
                rates.append(float(rate))
        if rates and global_pid_rate is not None:
            max_dev = max(abs(float(rate) - float(global_pid_rate)) for rate in rates)
            findings.append(
                f"Participant-level class balance across splits: max positive-rate deviation={max_dev:.3f}."
            )

    if locked_selection:
        override_k = locked_selection.get("lock_override_k")
        lock_strategy = locked_selection.get("lock_strategy")
        lock_source = locked_selection.get("lock_source")
        if override_k is not None:
            findings.append(f"Manual K override applied: K={override_k} (source={lock_source}).")
        elif lock_strategy:
            findings.append(f"K lock strategy: {lock_strategy} (source={lock_source}).")

    if not patient_selection_ranked.empty:
        top_patient_sel = patient_selection_ranked.iloc[0]
        findings.append(
            "Best patient-level selection result: "
            f"{top_patient_sel['model_display']} + {top_patient_sel['patient_aggregation']} "
            f"(ROC-AUC={_format_scalar(top_patient_sel['roc_auc'])}, "
            f"split={patient_locked_selection.get('selection_split', 'tuning')})."
        )

    if not patient_ranked.empty:
        top_patient = patient_ranked.iloc[0]
        findings.append(
            "Best patient-level final test configuration by ROC-AUC: "
            f"{top_patient['model_display']} + {top_patient['patient_aggregation']} "
            f"(ROC-AUC={_format_scalar(top_patient['roc_auc'])}, "
            f"Accuracy={_format_scalar(top_patient['accuracy'])}, "
            f"F1={_format_scalar(top_patient['f1'])})."
        )

    if not progressive_df.empty:
        row = progressive_df.iloc[0]
        findings.append(
            "Progressive feature selection best ROC-AUC observed at "
            f"{_format_scalar(row.get('n_features'))} features "
            f"(ROC-AUC={_format_scalar(row.get('test_roc_auc'))})."
        )

    if cluster_profile_brief:
        findings.append(f"Cluster composition summary (top profiles): {cluster_profile_brief[0]}")

    return findings


def _report_css() -> str:
    return """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  margin: 0;
  background: #f5f7fb;
  color: #17202a;
}
.container {
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 24px 40px 24px;
}
h1, h2, h3 {
  color: #0f2e59;
  margin-top: 0;
}
h1 {
  font-size: 2rem;
}
h2 {
  margin-top: 26px;
  margin-bottom: 12px;
  font-size: 1.35rem;
  border-bottom: 2px solid #d7e1ef;
  padding-bottom: 6px;
}
h3 {
  margin-top: 18px;
  margin-bottom: 8px;
  font-size: 1.05rem;
}
.muted {
  color: #566573;
}
.meta {
  background: #eaf1fb;
  border: 1px solid #c7d8ef;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 16px;
}
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.card {
  background: #ffffff;
  border: 1px solid #d8e1ef;
  border-radius: 10px;
  padding: 12px;
}
.card-title {
  font-size: 0.82rem;
  color: #5f6b7a;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.card-value {
  font-size: 1.28rem;
  font-weight: 700;
  margin-top: 4px;
  color: #0e3a6a;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0 18px 0;
  background: #fff;
  border: 1px solid #d8e1ef;
}
.data-table th, .data-table td {
  border: 1px solid #d8e1ef;
  padding: 7px 9px;
  font-size: 0.9rem;
  vertical-align: top;
}
.data-table th {
  background: #eef4fc;
  font-weight: 600;
}
.panel-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
  gap: 12px;
}
.panel-figure {
  margin: 0;
  background: #fff;
  border: 1px solid #d8e1ef;
  border-radius: 8px;
  padding: 8px;
}
.panel-figure img {
  width: 100%;
  height: auto;
  border-radius: 6px;
  border: 1px solid #d8e1ef;
}
.panel-figure figcaption {
  font-size: 0.84rem;
  color: #2f3b4a;
  margin-top: 8px;
}
code {
  background: #ecf0f3;
  padding: 2px 5px;
  border-radius: 4px;
}
ul {
  margin-top: 6px;
}
.table-collapse {
  margin: 10px 0 14px 0;
  background: #ffffff;
  border: 1px solid #d8e1ef;
  border-radius: 8px;
  padding: 8px 10px;
}
.table-collapse > summary {
  cursor: pointer;
  font-weight: 600;
  color: #0f2e59;
  outline: none;
}
.table-collapse-body {
  margin-top: 8px;
}
"""


def build_html(snapshot: dict[str, Any], run_dirs) -> str:
    dataset_summary = snapshot["dataset_summary"]
    baseline_split = snapshot["baseline_split"]
    cluster_split = snapshot["cluster_split"]
    k_selection = snapshot["k_selection"]
    best_fine = snapshot["best_fine"]
    locked_selection = snapshot.get("locked_selection", {})
    patient_locked_selection = snapshot.get("patient_locked_selection", {})
    baseline_table = snapshot["baseline_table"]
    rec_comparison = snapshot["recording_comparison"]
    fine_top = snapshot["fine_top"]
    coarse_contenders = snapshot["coarse_contenders"]
    patient_best_per_model = snapshot["patient_best_per_model"]
    patient_ranked = snapshot["patient_ranked"]
    patient_selection_best_per_model = snapshot.get("patient_selection_best_per_model", pd.DataFrame())
    patient_selection_ranked = snapshot.get("patient_selection_ranked", pd.DataFrame())
    progressive_df = snapshot["progressive_summary"]
    cluster_diag_df = snapshot.get("cluster_diagnostics", pd.DataFrame())
    cluster_elbow_df = snapshot.get("cluster_elbow_metrics", pd.DataFrame())
    cluster_stability_df = snapshot.get("cluster_stability", pd.DataFrame())
    model_hyperparams_df = snapshot.get("model_hyperparameters", pd.DataFrame())
    cluster_profile_table = snapshot.get("cluster_profile_table", pd.DataFrame())
    cluster_profile_brief = snapshot.get("cluster_profile_brief", [])
    manifest_df = snapshot["figure_manifest"]
    final_summary_df = snapshot["final_summary"]
    backends = snapshot["backend_tables"]

    key_findings = _build_findings(snapshot)
    artifact_df = _artifact_table(run_dirs)

    n_records = _format_scalar(dataset_summary.get("n_records"))
    n_participants = _format_scalar(dataset_summary.get("n_participants"))
    n_features = _format_scalar(dataset_summary.get("n_features"))
    (
        positive_name,
        negative_name,
        rec_positive_count,
        rec_negative_count,
        pid_positive_count,
        pid_negative_count,
    ) = _resolve_named_class_counts(
        dataset_summary,
        default_positive_name=str(snapshot.get("phenotype", "positive")),
        default_negative_name="control",
    )
    baseline_balance = baseline_split.get("class_balance", {})
    overall_pid_balance = baseline_balance.get("overall", {}).get("participants", {})
    if pid_positive_count is None:
        pid_positive_count = overall_pid_balance.get("n_positive")
    if pid_negative_count is None:
        pid_negative_count = overall_pid_balance.get("n_negative")
    pid_positive_text = _format_scalar(pid_positive_count) if pid_positive_count is not None else "n/a"
    pid_negative_text = _format_scalar(pid_negative_count) if pid_negative_count is not None else "n/a"
    cohort_text = (
        f"{positive_name}={_format_scalar(rec_positive_count)} recordings / {pid_positive_text} participants; "
        f"{negative_name}={_format_scalar(rec_negative_count)} recordings / {pid_negative_text} participants."
    )
    best_k = _format_scalar(best_fine.get("n_clusters"))
    best_cluster_auc = _format_scalar(_float_or_nan(best_fine.get("roc_auc")))
    analysis_mode = str(locked_selection.get("analysis_mode", "confirmatory"))
    tuning_split = str(k_selection.get("tuning_split", k_selection.get("selection_split", "tuning")))

    patient_top_auc = ""
    if not patient_ranked.empty:
        patient_top_auc = _format_scalar(_float_or_nan(patient_ranked.iloc[0].get("roc_auc")))

    cards = [
        ("Run ID", snapshot["run_id"]),
        ("Phenotype", snapshot["phenotype"]),
        ("Random Seed", str(snapshot["random_seed"])),
        ("Analysis Mode", analysis_mode),
        ("Tuning Split", tuning_split),
        ("Records", n_records),
        ("Participants", n_participants),
        ("Features", n_features),
        (f"{positive_name.title()} Records", _format_scalar(rec_positive_count)),
        (f"{negative_name.title()} Records", _format_scalar(rec_negative_count)),
        (f"{positive_name.title()} Participants", pid_positive_text),
        (f"{negative_name.title()} Participants", pid_negative_text),
        ("Best Clustered K", best_k),
        ("Best Recording ROC-AUC", best_cluster_auc),
        ("Best Patient ROC-AUC", patient_top_auc),
    ]

    initial_table = pd.DataFrame(
        [
            {
                "run_id": snapshot["run_id"],
                "phenotype": snapshot["phenotype"],
                "random_seed": snapshot["random_seed"],
                "analysis_mode": analysis_mode,
                "tuning_split": tuning_split,
                "n_records": dataset_summary.get("n_records"),
                "n_participants": dataset_summary.get("n_participants"),
                "n_features": dataset_summary.get("n_features"),
                f"{positive_name}_records": rec_positive_count,
                f"{negative_name}_records": rec_negative_count,
                f"{positive_name}_participants": pid_positive_count,
                f"{negative_name}_participants": pid_negative_count,
                "selected_k": best_fine.get("n_clusters"),
                "selected_model": best_fine.get("model_display", best_fine.get("model")),
            }
        ]
    )

    split_rows = pd.DataFrame(
        [
            {
                "level": "recording split",
                "train_participants": baseline_split.get("train_participants"),
                "val_participants": baseline_split.get("val_participants"),
                "test_participants": baseline_split.get("test_participants"),
                "train_records": baseline_split.get("train_recordings"),
                "val_records": baseline_split.get("val_recordings"),
                "test_records": baseline_split.get("test_recordings"),
            },
            {
                "level": "cluster split",
                "train_participants": cluster_split.get("train_participants"),
                "val_participants": cluster_split.get("val_participants"),
                "test_participants": cluster_split.get("test_participants"),
                "train_records": cluster_split.get("train_recordings"),
                "val_records": cluster_split.get("val_recordings"),
                "test_records": cluster_split.get("test_recordings"),
            },
        ]
    )

    split_balance_rows = (
        _split_balance_rows(baseline_split, "recording split")
        + _split_balance_rows(cluster_split, "cluster split")
    )
    split_balance_df = pd.DataFrame(split_balance_rows)
    if not split_balance_df.empty:
        split_balance_df = split_balance_df.rename(
            columns={
                "participant_positive": f"participant_{positive_name}",
                "participant_negative": f"participant_{negative_name}",
                "recording_positive": f"recording_{positive_name}",
                "recording_negative": f"recording_{negative_name}",
            }
        )

    backend_rows: list[pd.DataFrame] = []
    for stage_name, df in backends.items():
        if df.empty:
            continue
        stage_df = df.copy()
        stage_df.insert(0, "stage", stage_name)
        backend_rows.append(stage_df)
    backend_df = pd.concat(backend_rows, ignore_index=True) if backend_rows else pd.DataFrame()

    section_counts = pd.DataFrame()
    if not manifest_df.empty and "section" in manifest_df.columns:
        section_counts = (
            manifest_df.groupby("section", as_index=False)
            .size()
            .rename(columns={"size": "n_figures"})
            .sort_values("section")
            .reset_index(drop=True)
        )
    manifest_table = _table_with_links_for_manifest(manifest_df)

    fine_context_text = ""
    if k_selection:
        fine_context_text = (
            f"Tuning split: {k_selection.get('tuning_split', 'tuning')} | "
            f"Selected small-K center: {k_selection.get('selected_k')} | "
            f"coarse contenders: {k_selection.get('coarse_contender_k_values', [])} | "
            f"fine-tune Ks: {k_selection.get('fine_tune_k_values', [])}"
        )

    k_rationale_lines: list[str] = []
    if k_selection:
        k_rationale_lines.append(
            "Coarse contender strategy: "
            f"preferred range [{k_selection.get('preferred_k_min')}, {k_selection.get('preferred_k_max')}], "
            f"near-top delta={k_selection.get('near_top_delta')}."
        )
        k_rationale_lines.append(
            f"Coarse contender set: {k_selection.get('coarse_contender_k_values', [])} "
            f"(selection split={k_selection.get('tuning_split', tuning_split)})."
        )
        if k_selection.get("selection_model"):
            k_rationale_lines.append(
                "Selector model: "
                f"{_safe_model_display(k_selection.get('selection_model'))} "
                f"(policy={k_selection.get('selection_model_policy', 'fixed')}, "
                f"source={k_selection.get('selection_model_source', 'n/a')})."
            )
    if locked_selection:
        k_rationale_lines.append(
            "Locked final configuration: "
            f"K={locked_selection.get('selected_k')}, "
            f"aggregation={locked_selection.get('selected_aggregation')}, "
            f"selection metric={locked_selection.get('selection_metric')}."
        )
        override_k = locked_selection.get("lock_override_k")
        if override_k is not None:
            k_rationale_lines.append(f"Manual override applied for final lock: K={override_k}.")
        if locked_selection.get("lock_strategy"):
            k_rationale_lines.append(
                "Lock strategy: "
                f"{locked_selection.get('lock_strategy')} "
                f"(source={locked_selection.get('lock_source')})."
            )
        balanced_gate = locked_selection.get("lock_balanced_gate_source")
        if balanced_gate:
            k_rationale_lines.append(f"Balanced gate selection path: {balanced_gate}.")
    if best_fine:
        k_rationale_lines.append(
            "Locked final-test score at selected K: "
            f"ROC-AUC={_format_scalar(_float_or_nan(best_fine.get('roc_auc')))}, "
            f"Accuracy={_format_scalar(_float_or_nan(best_fine.get('accuracy')))}, "
            f"F1={_format_scalar(_float_or_nan(best_fine.get('f1')))}."
        )

    progressive_text = ""
    if not progressive_df.empty:
        row = progressive_df.iloc[0]
        progressive_text = (
            "Best progressive ROC-AUC at "
            f"{_format_scalar(row.get('n_features'))} features "
            f"(ROC-AUC={_format_scalar(row.get('test_roc_auc'))}, "
            f"Accuracy={_format_scalar(row.get('test_accuracy'))}, "
            f"F1={_format_scalar(row.get('test_f1'))})."
        )

    patient_selection_split = str(patient_locked_selection.get("selection_split", tuning_split))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voice Screening Report - {html.escape(snapshot["phenotype"])} - {html.escape(snapshot["run_id"])}</title>
  <style>{_report_css()}</style>
</head>
<body>
  <div class="container">
    <h1>Voice-Based Phenotype Screening Report</h1>
    <div class="meta">
      <div><strong>Phenotype:</strong> {html.escape(snapshot["phenotype"])}</div>
      <div><strong>Run ID:</strong> {html.escape(snapshot["run_id"])}</div>
      <div><strong>Cohort (case vs control):</strong> {html.escape(cohort_text)}</div>
      <div><strong>Generated (UTC):</strong> {html.escape(snapshot["generated_at_utc"])}</div>
      <div><strong>Repro command:</strong> <code>make pipeline PHENOTYPE={html.escape(snapshot["phenotype"])} RUN_ID={html.escape(snapshot["run_id"])}</code></div>
      <div><strong>Compile command:</strong> <code>make compile_html PHENOTYPE={html.escape(snapshot["phenotype"])} RUN_ID={html.escape(snapshot["run_id"])}</code></div>
    </div>

    {_summary_cards_html(cards)}
    <h2>Run Snapshot</h2>
    {_collapsible("Run Snapshot Table", _df_to_html(initial_table))}

    <h2>Executive Summary</h2>
    {_key_findings_html(key_findings)}

    <h2>Acoustic Feature Landscape</h2>
    <p class="muted">Figure-first overview of the acoustic feature space and dataset composition.</p>
    {_figures_panel_html(run_dirs, ACOUSTIC_FIGURES)}

    <h2>Stage 1: Dataset Preparation</h2>
    <p class="muted">
      Cohort and static acoustic feature matrix after phenotype-specific labeling and filtering.
      <strong>Case vs control counts:</strong> {html.escape(cohort_text)}
    </p>
    {_collapsible("Dataset Summary Table", _df_to_html(pd.DataFrame([dataset_summary])))}
    {_collapsible("Split Summary Table", _df_to_html(split_rows))}
    {_collapsible("Split Class-Balance Table", _df_to_html(split_balance_df))}

    <h2>Stage 2: Coarse Small-K Contenders</h2>
    <p class="muted">{html.escape(fine_context_text)}</p>
    {_figures_panel_html(run_dirs, COARSE_CONTENDER_FIGURES)}
    {_collapsible("Coarse Small-K Contender Table", _df_to_html(coarse_contenders, max_rows=20))}

    <h2>Stage 2B: Top Fine-Tune Results</h2>
    {_figures_panel_html(run_dirs, FINE_TUNE_FIGURES)}
    {_collapsible("Top Fine-Tune Table", _df_to_html(fine_top, max_rows=12))}

    <h2>Stage 3: Models, Hyperparameters, and K Validation</h2>
    <p class="muted">Models and focused hyperparameters followed by K-validation figures (silhouette, elbow, stability).</p>
    {_collapsible("Model + Hyperparameter Summary", _df_to_html(model_hyperparams_df))}
    {_figures_panel_html(run_dirs, K_VALIDATION_FIGURES)}
    {_collapsible("K-Diagnostic: Silhouette/Cophenetic Table", _df_to_html(cluster_diag_df))}
    {_collapsible("K-Diagnostic: Elbow Metrics Table", _df_to_html(cluster_elbow_df, max_rows=30))}
    {_collapsible("K-Diagnostic: Bootstrap Stability Table", _df_to_html(cluster_stability_df, max_rows=30))}
    <h3>Final K Selection Rationale</h3>
    {_key_findings_html(k_rationale_lines)}

    <h3>Cluster Composition Profiles</h3>
    <p class="muted">Brief interpretation of what each selected-feature cluster groups together.</p>
    {_key_findings_html(cluster_profile_brief)}
    {_collapsible("Cluster Profile Table", _df_to_html(cluster_profile_table, max_rows=25))}

    <h2>Stage 4: Recording-Level Performance</h2>
    {_figures_panel_html(run_dirs, RECORDING_FIGURES)}
    {_collapsible("Recording Baseline Results Table", _df_to_html(baseline_table))}
    {_collapsible("Baseline vs Locked Clustered Final Test Table", _df_to_html(rec_comparison))}

    <h2>Stage 5: Patient-Level Performance</h2>
    <p class="muted">
      Main figures in this section reflect held-out <code>test</code> performance (aligned with final tables). Selection-split (<code>{html.escape(patient_selection_split)}</code>) tuning results are shown in the tables below.
    </p>
    {_figures_panel_html(run_dirs, PATIENT_FIGURES)}
    {_collapsible("Patient Selection Split Best-by-Model Table", _df_to_html(patient_selection_best_per_model))}
    {_collapsible("Patient Selection Split Ranked Table", _df_to_html(patient_selection_ranked, max_rows=15))}
    {_collapsible("Patient Final-Test Best-per-Model Table", _df_to_html(patient_best_per_model))}
    {_collapsible("Patient Final-Test Ranked Table", _df_to_html(patient_ranked, max_rows=15))}

    <h2>Stage 6: Progressive Feature Selection</h2>
    {_figures_panel_html(run_dirs, PROGRESSIVE_FIGURES)}
    <p class="muted">{html.escape(progressive_text)}</p>
    {_collapsible("Progressive Feature Selection Table", _df_to_html(progressive_df))}

    <h2>Stage 7: Cross-Level Summary</h2>
    {_collapsible("Cross-Level Summary Table", _df_to_html(final_summary_df))}

    <h2>Backend and Method Traceability</h2>
    {_collapsible("Backend Traceability Table", _df_to_html(backend_df))}

    <h2>Figure Inventory</h2>
    {_collapsible("Figure Count by Section", _df_to_html(section_counts))}
    {_collapsible("Figure Manifest Table", _df_to_html(manifest_table, max_rows=250))}

    <h2>Artifact Index</h2>
    {_collapsible("Artifact Index Table", _df_to_html(artifact_df))}
  </div>
</body>
</html>
"""
    return html_doc


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.phenotype_config)
    phenotype_name = config["phenotype"]["name"]
    run_dirs = build_run_dirs(config["paths"]["outputs_root"], phenotype_name, args.run_id)
    analysis_mode = get_analysis_mode(config)
    tuning_split = get_tuning_split(config)

    snapshot = _run_snapshot(
        run_dirs=run_dirs,
        phenotype_name=phenotype_name,
        run_id=args.run_id,
        config=config,
    )
    html_text = build_html(snapshot, run_dirs)

    html_path = run_dirs.reports / "manuscript_report.html"
    metadata_path = run_dirs.reports / "manuscript_report_metadata.json"

    html_path.write_text(html_text, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "report_version": "v1",
                "generated_at_utc": snapshot["generated_at_utc"],
                "run_id": snapshot["run_id"],
                "phenotype": snapshot["phenotype"],
                "random_seed": snapshot["random_seed"],
                "analysis_mode": analysis_mode,
                "tuning_split": tuning_split,
                "source_paths": {
                    "prepared": str(run_dirs.prepared),
                    "baseline": str(run_dirs.baseline),
                    "cluster_models": str(run_dirs.cluster_models),
                    "reports": str(run_dirs.reports),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    update_run_manifest(
        path=run_dirs.root / "run_manifest.json",
        config=config,
        stage="07_compile_html_report",
        run_id=args.run_id,
        phenotype=phenotype_name,
        extra={"analysis_mode": analysis_mode, "tuning_split": tuning_split},
        cwd=ROOT,
    )
    print(f"HTML report saved: {html_path}")
    print(f"Report metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
