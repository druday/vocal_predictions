#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voice_screening.analysis import get_analysis_mode, get_deterministic_torch, get_tuning_split
from voice_screening.clustering import cluster_assignments, feature_distance_matrix, fit_hierarchical_clustering
from voice_screening.config import load_config
from voice_screening.data import apply_case_control_labeling, load_raw_tables, merge_label_and_static
from voice_screening.io import load_dataframe, save_json
from voice_screening.modeling import build_model, predict_positive_proba, seed_everything
from voice_screening.notebook_parity import (
    aggregate_features_by_cluster,
    clean_feature_dataframe,
    evaluate_binary_metrics,
    fit_cluster_assignments,
    participant_train_val_test_split,
    standardize_and_clip,
)
from voice_screening.repro import update_run_manifest
from voice_screening.run import build_run_dirs


METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
}

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

DISPLAY_TO_MODEL = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
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

PPT_SECTIONS = {
    "acoustic": "01_acoustic_features",
    "cluster_decision": "02_feature_clustering_decision",
    "cluster_insights": "03_feature_clustering_insights",
    "progressive": "04_progressive_feature_selection",
    "recording": "05_recording_level_performance",
    "patient": "06_patient_level_performance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate notebook-parity figure and table artifacts aligned to manuscript/PPT flow."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phenotype-config", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _safe_read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing expected CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing expected JSON: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return plt, sns


def _make_section_dirs(run_dirs) -> dict[str, Path]:
    base = run_dirs.reports / "figures"
    base.mkdir(parents=True, exist_ok=True)
    section_dirs: dict[str, Path] = {}
    for key, folder in PPT_SECTIONS.items():
        path = base / folder
        path.mkdir(parents=True, exist_ok=True)
        section_dirs[key] = path
    return section_dirs


def _save_plot(
    *,
    plt_mod,
    report_path: Path,
    section_path: Path,
    dpi: int = 300,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    section_path.parent.mkdir(parents=True, exist_ok=True)
    plt_mod.savefig(report_path, dpi=dpi, bbox_inches="tight")
    if section_path.resolve() != report_path.resolve():
        shutil.copyfile(report_path, section_path)
    plt_mod.close()


def _record_manifest(
    manifest_rows: list[dict[str, str]],
    *,
    run_dirs,
    section_key: str,
    filename: str,
    description: str,
    ppt_hint: str,
) -> None:
    report_path = run_dirs.reports / filename
    section_path = run_dirs.reports / "figures" / PPT_SECTIONS[section_key] / filename
    manifest_rows.append(
        {
            "section": section_key,
            "ppt_section_folder": PPT_SECTIONS[section_key],
            "filename": filename,
            "description": description,
            "ppt_hint": ppt_hint,
            "report_path": str(report_path.relative_to(run_dirs.root)),
            "section_path": str(section_path.relative_to(run_dirs.root)),
        }
    )


def _split_config(config: dict) -> tuple[float, float]:
    split_cfg = config.get("modeling", {}).get("split", {})
    test_fraction = float(split_cfg.get("test_fraction", 0.2))
    validation_fraction = float(split_cfg.get("validation_fraction", 0.2))
    return test_fraction, validation_fraction


def _ordered_model_displays(config: dict, available_model_displays: list[str]) -> list[str]:
    available = [str(v) for v in available_model_displays if str(v)]
    if not available:
        return []
    configured = [str(v).strip().lower() for v in config.get("modeling", {}).get("models", [])]
    preferred = [MODEL_DISPLAY_NAMES.get(name, name) for name in configured]
    ordered: list[str] = []
    for model_display in preferred:
        if model_display in available and model_display not in ordered:
            ordered.append(model_display)
    for model_display in available:
        if model_display not in ordered:
            ordered.append(model_display)
    return ordered


def _cluster_validation_k_values(config: dict, run_dirs, diagnostics_df: pd.DataFrame) -> list[int]:
    k_values: list[int] = []
    if not diagnostics_df.empty:
        for col in ["k", "n_clusters"]:
            if col in diagnostics_df.columns:
                k_values = sorted({int(v) for v in diagnostics_df[col].dropna().astype(int).tolist()})
                if k_values:
                    return k_values

    assignment_files = sorted(run_dirs.clusters.glob("cluster_assignments_*.csv"))
    if assignment_files:
        parsed: list[int] = []
        for path in assignment_files:
            stem = path.stem
            if "_" not in stem:
                continue
            try:
                parsed.append(int(stem.rsplit("_", 1)[1]))
            except ValueError:
                continue
        k_values = sorted(set(parsed))
        if k_values:
            return k_values

    configured = config.get("cluster", {}).get("k_values", config.get("cluster", {}).get("exploration_k_values", []))
    if configured:
        return sorted({int(v) for v in configured})
    return []


def _within_cluster_mean_distance(
    *,
    assignments_df: pd.DataFrame,
    feature_names: list[str],
    dist_matrix: np.ndarray,
) -> float:
    feat_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    total_distance = 0.0
    total_pairs = 0
    for _, group in assignments_df.groupby("cluster", sort=False):
        indices = [
            feat_to_idx[str(f)]
            for f in group["feature"].astype(str).tolist()
            if str(f) in feat_to_idx
        ]
        if len(indices) < 2:
            continue
        sub = dist_matrix[np.ix_(indices, indices)]
        tri = sub[np.triu_indices(len(indices), k=1)]
        if tri.size == 0:
            continue
        total_distance += float(np.sum(tri))
        total_pairs += int(tri.size)
    if total_pairs == 0:
        return float("nan")
    return float(total_distance / total_pairs)


def _prepared_context(config: dict, run_dirs) -> dict[str, Any]:
    df = load_dataframe(run_dirs.prepared / "recordings.parquet")
    feature_cols = pd.read_csv(run_dirs.prepared / "feature_columns.csv")["feature"].tolist()

    participant_col = config["columns"]["participant_id"]
    label_col = "label"
    seed = int(config["project"].get("random_seed", 42))

    x_full = df[feature_cols].copy()
    x_clean, usable_features, removed_features = clean_feature_dataframe(x_full)
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

    return {
        "recordings_df": df,
        "feature_cols": feature_cols,
        "x_clean": x_clean,
        "usable_features": usable_features,
        "removed_features": removed_features,
        "labels": labels,
        "participant_ids": participant_ids,
        "split": split,
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pid_train": pid_train,
        "pid_val": pid_val,
        "pid_test": pid_test,
        "x_train_scaled": x_train_scaled,
        "x_val_scaled": x_val_scaled,
        "x_test_scaled": x_test_scaled,
    }


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
    baseline_prob = predict_positive_proba(model, x_eval)
    baseline_auc = evaluate_binary_metrics(y_true, baseline_prob, threshold=0.5).get("roc_auc", np.nan)

    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []
    for feat_idx, feat_name in enumerate(feature_names):
        drops: list[float] = []
        for _ in range(max(1, int(n_iterations))):
            perm = np.array(x_eval, copy=True)
            perm_idx = rng.permutation(len(perm))
            perm[:, feat_idx] = perm[perm_idx, feat_idx]
            perm_prob = predict_positive_proba(model, perm)
            perm_auc = evaluate_binary_metrics(y_true, perm_prob, threshold=0.5).get("roc_auc", np.nan)
            if np.isfinite(baseline_auc) and np.isfinite(perm_auc):
                drops.append(float(baseline_auc - perm_auc))
        rows.append({"feature": feat_name, "importance": float(np.mean(drops)) if drops else float("nan")})
    out = pd.DataFrame(rows).sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(np.asarray(y_true).astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def generate_acoustic_feature_figures(
    *,
    config: dict,
    run_dirs,
    section_dirs: dict[str, Path],
    manifest_rows: list[dict[str, str]],
    plt,
    sns,
) -> None:
    try:
        phenotype_df, static_df = load_raw_tables(config)
    except FileNotFoundError as exc:
        print(f"Skipping acoustic feature figures: {exc}")
        return

    labeled_pheno = apply_case_control_labeling(phenotype_df, config)
    merged_labeled = merge_label_and_static(labeled_pheno, static_df, config)

    checked_cols: list[str] = []
    for col in labeled_pheno.columns:
        if labeled_pheno[col].dtype != object:
            continue
        values = (
            labeled_pheno[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        )
        value_set = set(values)
        if "checked" in value_set or "unchecked" in value_set:
            checked_cols.append(col)

    # Condition co-occurrence lower triangle.
    if checked_cols:
        disease_mask = labeled_pheno["label"].astype(int) == 1
        condition_counts: dict[str, int] = {}
        for col in checked_cols:
            cnt = int(
                ((labeled_pheno[col].astype(str).str.lower() == "checked") & disease_mask).sum()
            )
            if cnt > 0:
                condition_counts[col] = cnt

        top_conditions = (
            pd.Series(condition_counts).sort_values(ascending=False).head(10).index.tolist()
            if condition_counts
            else []
        )
        if top_conditions:
            cond_matrix = (
                labeled_pheno.loc[disease_mask, top_conditions]
                .astype(str)
                .apply(lambda c: c.str.lower() == "checked")
                .astype(int)
            )
            coocc = cond_matrix.T @ cond_matrix
            pretty = [c.replace("_", " ").title()[:20] for c in top_conditions]
            coocc.index = pretty
            coocc.columns = pretty

            mask = np.triu(np.ones_like(coocc, dtype=bool), k=1)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                coocc,
                annot=True,
                fmt="d",
                cmap="YlOrRd",
                mask=mask,
                cbar_kws={"label": "Co-occurrence Count"},
                square=True,
            )
            plt.title("Condition Co-occurrence Matrix (Lower-Triangle Only)", fontsize=12, fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)

            fname = "disease_condition_cooccurrence_lower_triangle.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["acoustic"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="acoustic",
                filename=fname,
                description="Disease condition co-occurrence lower triangle heatmap.",
                ppt_hint="Slide 6-7 phenotype overview",
            )

    # Voice exercises analysis.
    if "task_name" in static_df.columns:
        task_counts = static_df["task_name"].astype(str).value_counts()
        task_summary = task_counts.rename_axis("task_name").reset_index(name="n_recordings")
        task_summary.to_csv(run_dirs.reports / "task_summary.csv", index=False)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        top_tasks = task_counts.head(20)
        axes[0, 0].barh(range(len(top_tasks)), top_tasks.values, color="steelblue", alpha=0.8)
        axes[0, 0].set_yticks(range(len(top_tasks)))
        axes[0, 0].set_yticklabels([str(t)[:40] for t in top_tasks.index], fontsize=9)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel("Number of Recordings", fontsize=11)
        axes[0, 0].set_title("Top 20 Voice Exercises/Tasks", fontsize=12, fontweight="bold")
        axes[0, 0].grid(axis="x", alpha=0.3)

        axes[0, 1].hist(task_counts.values, bins=30, color="coral", edgecolor="black", alpha=0.7)
        axes[0, 1].set_xlabel("Number of Recordings per Task", fontsize=11)
        axes[0, 1].set_ylabel("Number of Tasks", fontsize=11)
        axes[0, 1].set_title("Task Frequency Distribution", fontsize=12, fontweight="bold")
        axes[0, 1].grid(alpha=0.3)

        top_10 = task_counts.head(10)
        other_count = int(task_counts.iloc[10:].sum()) if len(task_counts) > 10 else 0
        pie_data = list(top_10.values)
        pie_labels = [str(t)[:30] for t in top_10.index]
        if other_count > 0:
            pie_data.append(other_count)
            pie_labels.append("Others")
        axes[1, 0].pie(pie_data, labels=pie_labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        axes[1, 0].set_title("Top 10 Tasks (Percentage)", fontsize=12, fontweight="bold")

        if "transcription" in static_df.columns:
            trans_len = static_df.copy()
            trans_len["transcription_length"] = trans_len["transcription"].astype(str).str.len()
            task_lengths = (
                trans_len.groupby("task_name", as_index=True)["transcription_length"]
                .mean()
                .sort_values(ascending=False)
                .head(15)
            )
            axes[1, 1].barh(range(len(task_lengths)), task_lengths.values, color="mediumseagreen", alpha=0.8)
            axes[1, 1].set_yticks(range(len(task_lengths)))
            axes[1, 1].set_yticklabels([str(t)[:30] for t in task_lengths.index], fontsize=9)
            axes[1, 1].invert_yaxis()
            axes[1, 1].set_xlabel("Average Transcription Length", fontsize=11)
            axes[1, 1].set_title("Task Complexity (by Transcription Length)", fontsize=12, fontweight="bold")
            axes[1, 1].grid(axis="x", alpha=0.3)
        else:
            axes[1, 1].axis("off")

        plt.suptitle("Voice Exercises/Tasks Analysis", fontsize=14, fontweight="bold")
        fname = "voice_exercises_analysis.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["acoustic"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="acoustic",
            filename=fname,
            description="Voice exercise/task distribution analysis.",
            ppt_hint="Slide 6 dataset overview",
        )

    # Acoustic feature category and distributions.
    id_cols = {
        config["columns"]["participant_id"],
        config["columns"].get("session_id", "session_id"),
        config["columns"].get("task_name", "task_name"),
        config["columns"].get("transcription", "transcription"),
        "label",
        "label_name",
        "pd_label",
        "control_label",
    }
    numeric_cols = merged_labeled.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in id_cols]

    if feature_cols:
        def _cat(name: str) -> list[str]:
            if name == "mfcc":
                return [f for f in feature_cols if "mfcc" in f.lower()]
            if name == "prosody":
                tokens = ["f0", "pitch", "jitter", "shimmer", "hnr"]
                return [f for f in feature_cols if any(t in f.lower() for t in tokens)]
            if name == "formant":
                tokens = ["f1", "f2", "f3", "formant"]
                return [f for f in feature_cols if any(t in f.lower() for t in tokens)]
            if name == "energy":
                tokens = ["energy", "loudness"]
                return [f for f in feature_cols if any(t in f.lower() for t in tokens)]
            return []

        mfcc = _cat("mfcc")
        prosody = _cat("prosody")
        formant = _cat("formant")
        energy = _cat("energy")
        other = [f for f in feature_cols if f not in (mfcc + prosody + formant + energy)]

        categories = {
            "MFCC": mfcc,
            "Prosody": prosody,
            "Formant": formant,
            "Energy": energy,
            "Other": other,
        }

        plt.figure(figsize=(8, 5))
        cat_names = list(categories.keys())
        cat_counts = [len(categories[n]) for n in cat_names]
        plt.bar(cat_names, cat_counts, color="cornflowerblue")
        plt.xlabel("Feature Category")
        plt.ylabel("Number of Features")
        plt.title("Number of Features in Each Acoustic Category")
        for idx, value in enumerate(cat_counts):
            plt.text(idx, value + 0.5, str(value), ha="center", va="bottom", fontsize=10)
        fname = "feature_category_counts.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["acoustic"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="acoustic",
            filename=fname,
            description="Acoustic feature category count bar chart.",
            ppt_hint="Slide 6-8 acoustic feature structure",
        )

        def _plot_group_distribution(group_name: str, feats: list[str], suffix: str) -> None:
            if not feats:
                return
            sample_feats = feats[:12]
            n_rows, n_cols = 3, 4
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
            axes = axes.flatten()
            for idx, feat in enumerate(sample_feats):
                ax = axes[idx]
                values = merged_labeled[feat].dropna()
                if len(values) == 0:
                    ax.set_title(f"{feat[:30]} (no data)")
                    ax.axis("off")
                    continue
                ax.hist(values, bins=35, alpha=0.7, color="steelblue", edgecolor="black")
                ax.set_title(feat[:30], fontsize=9, fontweight="bold")
                ax.tick_params(labelsize=8)
            for idx in range(len(sample_feats), len(axes)):
                axes[idx].axis("off")
            plt.suptitle(f"{group_name} Feature Distributions (up to 12)", fontsize=14, fontweight="bold")
            fname_local = f"static_features_{group_name.lower()}_{suffix}.png" if suffix else f"static_features_{group_name.lower()}.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname_local,
                section_path=section_dirs["acoustic"] / fname_local,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="acoustic",
                filename=fname_local,
                description=f"{group_name} feature distribution panel.",
                ppt_hint="Slide 8 feature distribution context",
            )

        _plot_group_distribution("MFCC", mfcc, "")
        _plot_group_distribution("Prosody", prosody, "")
        _plot_group_distribution("Formant", formant, "")
        _plot_group_distribution("Energy", energy, "")
        _plot_group_distribution("Other", other, "")

        def _plot_group_pd(group_name: str, feats: list[str]) -> None:
            if not feats or "label" not in merged_labeled.columns:
                return
            sample_feats = feats[:12]
            n_rows, n_cols = 3, 4
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
            axes = axes.flatten()
            for idx, feat in enumerate(sample_feats):
                ax = axes[idx]
                work = merged_labeled[[feat, "label"]].dropna()
                if work.empty:
                    ax.axis("off")
                    continue
                sns.histplot(
                    data=work,
                    x=feat,
                    hue="label",
                    bins=30,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    palette={0: "teal", 1: "crimson"},
                    ax=ax,
                )
                ax.set_title(feat[:30], fontsize=9, fontweight="bold")
                ax.tick_params(labelsize=8)
            for idx in range(len(sample_feats), len(axes)):
                axes[idx].axis("off")
            plt.suptitle(
                f"{group_name} Feature Distributions: Control vs Case (up to 12)",
                fontsize=14,
                fontweight="bold",
            )
            fname_local = f"static_features_{group_name.lower()}_pd.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname_local,
                section_path=section_dirs["acoustic"] / fname_local,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="acoustic",
                filename=fname_local,
                description=f"{group_name} feature distributions for case vs control.",
                ppt_hint="Slide 8 embedding/feature separability context",
            )

        _plot_group_pd("MFCC", mfcc)
        _plot_group_pd("Prosody", prosody)
        _plot_group_pd("Formant", formant)
        _plot_group_pd("Energy", energy)
        _plot_group_pd("Other", other)

    if "label" in merged_labeled.columns and feature_cols:
        x_for_pca = merged_labeled[feature_cols].copy()
        for col in x_for_pca.columns:
            x_for_pca[col] = pd.to_numeric(x_for_pca[col], errors="coerce")
        x_for_pca = x_for_pca.replace([np.inf, -np.inf], np.nan)
        x_for_pca = x_for_pca.fillna(x_for_pca.median(numeric_only=True))
        x_for_pca = x_for_pca.fillna(0.0)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_for_pca.to_numpy(dtype=np.float32))
        pca2 = PCA(n_components=min(2, x_scaled.shape[1]), random_state=42)
        x_pca = pca2.fit_transform(x_scaled)
        plt.figure(figsize=(8, 6))
        palette = {0: "teal", 1: "crimson"}
        for label_value in sorted(np.unique(merged_labeled["label"].astype(int).to_numpy())):
            mask = merged_labeled["label"].astype(int).to_numpy() == label_value
            plt.scatter(
                x_pca[mask, 0],
                x_pca[mask, 1] if x_pca.shape[1] > 1 else np.zeros(mask.sum()),
                s=12,
                alpha=0.6,
                color=palette.get(int(label_value), "gray"),
                label=f"Label {int(label_value)}",
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Projection of Raw Acoustic Features")
        plt.legend(loc="best")
        fname = "pca_raw_features_scatter.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["cluster_insights"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="cluster_insights",
            filename=fname,
            description="PCA scatter of raw acoustic features.",
            ppt_hint="Slide 8 dimensionality insights",
        )


def generate_cluster_decision_figures(
    *,
    config: dict,
    run_dirs,
    section_dirs: dict[str, Path],
    manifest_rows: list[dict[str, str]],
    plt,
    sns,
) -> None:
    coarse_df = _safe_read_csv(run_dirs.cluster_models / "coarse_results.csv", required=False)
    fine_df = _safe_read_csv(run_dirs.cluster_models / "fine_tune_results.csv", required=False)
    contender_df = _safe_read_csv(
        run_dirs.cluster_models / "coarse_k_contenders_selection_scope.csv", required=False
    )
    tuning_split_label = "tuning"
    if not coarse_df.empty and "evaluation_split" in coarse_df.columns:
        splits = sorted({str(v) for v in coarse_df["evaluation_split"].dropna().astype(str).tolist()})
        if splits:
            tuning_split_label = "/".join(splits)

    if not coarse_df.empty:
        # Notebook-style performance vs clusters.
        models = _ordered_model_displays(
            config,
            coarse_df["model_display"].dropna().astype(str).unique().tolist(),
        )
        colors = {
            "mean": "blue",
            "pca": "red",
            "max": "green",
            "min": "purple",
            "std": "orange",
            "median": "brown",
            "sum": "pink",
        }

        n_models = len(models)
        n_cols = 3 if n_models > 4 else 2
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
        axes_flat = np.atleast_1d(axes).flatten()
        fig.suptitle("Model Performance vs Number of Clusters", fontsize=16, fontweight="bold")

        for idx, model_name in enumerate(models):
            ax = axes_flat[idx]
            model_rows = coarse_df[coarse_df["model_display"] == model_name].copy()
            if model_rows.empty:
                ax.axis("off")
                continue
            for agg_method, agg_df in model_rows.groupby("aggregation", sort=False):
                agg_sorted = agg_df.sort_values("n_clusters")
                ax.plot(
                    agg_sorted["n_clusters"],
                    agg_sorted["roc_auc"],
                    marker="o",
                    linewidth=2,
                    color=colors.get(str(agg_method).lower(), None),
                    label=str(agg_method).upper(),
                )
            ax.set_xlabel("Number of Clusters", fontsize=11)
            ax.set_ylabel(f"ROC-AUC ({tuning_split_label})", fontsize=11)
            ax.set_title(model_name, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].axis("off")

        fname = "performance_vs_clusters.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["cluster_decision"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="cluster_decision",
            filename=fname,
            description=f"Coarse K sweep ROC-AUC ({tuning_split_label}) by model and aggregation.",
            ppt_hint="Slide 19-21 cluster-aggregated results",
        )

        # Heatmap.
        pivot = coarse_df.pivot_table(
            values="roc_auc",
            index=["n_clusters", "aggregation"],
            columns="model_display",
            aggfunc="mean",
        )
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={"label": "ROC-AUC"},
        )
        plt.title(
            "ROC-AUC Performance Heatmap: Cluster Level x Aggregation Method x Model",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("(Clusters, Aggregation)", fontsize=12)
        fname = "performance_heatmap.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["cluster_decision"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="cluster_decision",
            filename=fname,
            description="Cluster search heatmap across models and aggregations.",
            ppt_hint="Slide 19-21 cluster-aggregated results",
        )

        # Selection-scope contenders.
        fine_cfg = config.get("cluster", {}).get("fine_tune", {})
        selection_agg = str(
            fine_cfg.get(
                "cluster_aggregation",
                fine_cfg.get("aggregation", config.get("cluster", {}).get("aggregation", "mean")),
            )
        ).strip().lower()
        scope_df = coarse_df[coarse_df["aggregation"] == selection_agg].copy()
        if scope_df.empty:
            scope_df = coarse_df.copy()
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=scope_df.sort_values("n_clusters"),
            x="n_clusters",
            y="roc_auc",
            hue="model_display",
            marker="o",
        )
        if not contender_df.empty:
            sns.scatterplot(
                data=contender_df,
                x="n_clusters",
                y="roc_auc",
                hue="model_display",
                style="model_display",
                s=120,
                edgecolor="black",
                linewidth=0.8,
                legend=False,
            )
        plt.title("Coarse K Search (Selection Scope) with Contenders")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel(f"ROC-AUC ({tuning_split_label})")
        fname = "fig_coarse_k_contenders_selection_scope.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["cluster_decision"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="cluster_decision",
            filename=fname,
            description=f"Small-K contender highlighting within {tuning_split_label} selection scope.",
            ppt_hint="Slide 20-21 coarse K and contender logic",
        )

    if not fine_df.empty:
        fine_sorted = fine_df.sort_values("n_clusters")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Fine-Tuning Neural Network: Performance vs Cluster Levels",
            fontsize=16,
            fontweight="bold",
        )

        plot_specs = [("roc_auc", axes[0, 0]), ("accuracy", axes[0, 1]), ("f1", axes[1, 0])]
        for metric, ax in plot_specs:
            ax.plot(fine_sorted["n_clusters"], fine_sorted[metric], marker="o", linewidth=2)
            ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
            ax.set_xlabel("Number of Clusters", fontsize=11)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
            ax.grid(True, alpha=0.3)

        ax_all = axes[1, 1]
        for metric in ["roc_auc", "accuracy", "f1"]:
            ax_all.plot(
                fine_sorted["n_clusters"],
                fine_sorted[metric],
                marker="o",
                linewidth=2,
                label=METRIC_LABELS[metric],
            )
        ax_all.set_title("All Metrics", fontsize=12, fontweight="bold")
        ax_all.set_xlabel("Number of Clusters", fontsize=11)
        ax_all.set_ylabel("Score", fontsize=11)
        ax_all.grid(True, alpha=0.3)
        ax_all.legend()

        fname = "fine_tune_performance_13_17.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["cluster_decision"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="cluster_decision",
            filename=fname,
            description="Fine-tune window performance trends around selected K region.",
            ppt_hint="Slide 21 peak region verification",
        )


def generate_cluster_validation_figures(
    *,
    config: dict,
    run_dirs,
    section_dirs: dict[str, Path],
    manifest_rows: list[dict[str, str]],
    ctx: dict[str, Any],
    plt,
    sns,
) -> None:
    diagnostics_df = _safe_read_csv(run_dirs.clusters / "clustering_diagnostics.csv", required=False)
    k_values = _cluster_validation_k_values(config, run_dirs, diagnostics_df)
    if not k_values:
        return

    best_cfg = _safe_read_json(run_dirs.cluster_models / "best_fine_tune_config.json", required=False)
    selected_k = int(best_cfg.get("n_clusters", 0)) if best_cfg else 0

    # 1) Silhouette + cophenetic diagnostics from stage-02 clustering artifacts.
    if not diagnostics_df.empty:
        diag = diagnostics_df.copy()
        if "k" not in diag.columns and "n_clusters" in diag.columns:
            diag["k"] = diag["n_clusters"]
        if "k" in diag.columns:
            diag["k"] = diag["k"].astype(int)
            diag = diag.sort_values("k")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            if "silhouette_score" in diag.columns:
                axes[0].plot(diag["k"], diag["silhouette_score"], marker="o", linewidth=2, color="#1f77b4")
            axes[0].set_title("Silhouette Score vs K", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("Number of Clusters (K)")
            axes[0].set_ylabel("Silhouette Score")
            axes[0].grid(True, alpha=0.3)
            if selected_k > 0:
                axes[0].axvline(selected_k, color="red", linestyle="--", linewidth=1.2, label=f"Selected K={selected_k}")
                axes[0].legend(loc="best", fontsize=8)

            if "cophenetic_correlation" in diag.columns:
                axes[1].plot(diag["k"], diag["cophenetic_correlation"], marker="o", linewidth=2, color="#ff7f0e")
            axes[1].set_title("Cophenetic Correlation vs K", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Number of Clusters (K)")
            axes[1].set_ylabel("Cophenetic Correlation")
            axes[1].grid(True, alpha=0.3)
            if selected_k > 0:
                axes[1].axvline(selected_k, color="red", linestyle="--", linewidth=1.2, label=f"Selected K={selected_k}")
                axes[1].legend(loc="best", fontsize=8)

            fname = "cluster_validation_silhouette_cophenetic.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["cluster_decision"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="cluster_decision",
                filename=fname,
                description="K-diagnostics: silhouette and cophenetic trends for hierarchical feature clustering.",
                ppt_hint="Slide 12-14 cluster count diagnostics",
            )

    # 2) Elbow-style compactness proxy based on within-cluster feature distance.
    linkage_method = str(config.get("cluster", {}).get("linkage", "ward"))
    dist_full = feature_distance_matrix(np.asarray(ctx["x_train_scaled"], dtype=np.float32))
    link_full = fit_hierarchical_clustering(dist_full, linkage_method=linkage_method)
    feature_names = [str(v) for v in ctx["usable_features"]]

    elbow_rows: list[dict[str, float | int]] = []
    for k in k_values:
        assign_path = run_dirs.clusters / f"cluster_assignments_{int(k)}.csv"
        assign_df = _safe_read_csv(assign_path, required=False)
        if assign_df.empty or not {"feature", "cluster"}.issubset(assign_df.columns):
            assign_df = cluster_assignments(feature_names, link_full, int(k))

        mean_within = _within_cluster_mean_distance(
            assignments_df=assign_df,
            feature_names=feature_names,
            dist_matrix=dist_full,
        )
        elbow_rows.append(
            {
                "k": int(k),
                "within_cluster_mean_distance": float(mean_within),
            }
        )

    elbow_df = pd.DataFrame(elbow_rows).sort_values("k").reset_index(drop=True)
    if not elbow_df.empty:
        elbow_df["improvement_vs_prev_k"] = (
            elbow_df["within_cluster_mean_distance"].shift(1) - elbow_df["within_cluster_mean_distance"]
        )
        elbow_df.to_csv(run_dirs.reports / "cluster_validation_elbow_metrics.csv", index=False)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(
            elbow_df["k"],
            elbow_df["within_cluster_mean_distance"],
            marker="o",
            linewidth=2,
            color="#2ca02c",
            label="Within-cluster mean distance",
        )
        ax1.set_title("Elbow Proxy: Cluster Compactness vs K", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Within-cluster mean distance (lower is better)")
        ax1.grid(True, alpha=0.3)
        if selected_k > 0:
            ax1.axvline(selected_k, color="red", linestyle="--", linewidth=1.2, label=f"Selected K={selected_k}")
        ax1.legend(loc="upper right", fontsize=8)

        ax2 = ax1.twinx()
        ax2.bar(
            elbow_df["k"],
            elbow_df["improvement_vs_prev_k"].fillna(0.0),
            alpha=0.2,
            color="#9467bd",
            label="Improvement vs previous K",
        )
        ax2.set_ylabel("Compactness improvement vs previous K")

        fname = "cluster_validation_elbow_curve.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["cluster_decision"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="cluster_decision",
            filename=fname,
            description="Elbow-style compactness proxy using within-cluster feature distance.",
            ppt_hint="Slide 12-14 cluster compactness diagnostics",
        )

    # 3) Bootstrap stability: ARI agreement of feature assignments across resamples.
    validation_cfg = config.get("reporting", {}).get("cluster_validation", {})
    n_bootstrap = int(validation_cfg.get("n_bootstrap", 8))
    sample_fraction = float(validation_cfg.get("sample_fraction", 0.8))
    sample_fraction = float(np.clip(sample_fraction, 0.2, 1.0))
    seed = int(config.get("project", {}).get("random_seed", 42))

    x_train = np.asarray(ctx["x_train_scaled"], dtype=np.float32)
    n_samples = int(x_train.shape[0])
    labels_by_k: dict[int, list[np.ndarray]] = {int(k): [] for k in k_values}

    if n_bootstrap >= 2 and n_samples >= 20:
        rng = np.random.default_rng(seed)
        sample_n = int(max(20, round(sample_fraction * n_samples)))
        sample_n = int(min(sample_n, n_samples))

        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, size=sample_n, replace=False)
            x_sub = x_train[idx]
            dist_sub = feature_distance_matrix(x_sub)
            link_sub = fit_hierarchical_clustering(dist_sub, linkage_method=linkage_method)
            for k in k_values:
                assign_sub = cluster_assignments(feature_names, link_sub, int(k))
                labels = (
                    assign_sub.set_index("feature")["cluster"]
                    .reindex(feature_names)
                    .astype(float)
                    .fillna(-1)
                    .astype(int)
                    .to_numpy()
                )
                labels_by_k[int(k)].append(labels)

    stability_rows: list[dict[str, float | int]] = []
    for k in k_values:
        rep_labels = labels_by_k.get(int(k), [])
        if len(rep_labels) < 2:
            stability_rows.append(
                {
                    "k": int(k),
                    "n_bootstrap": int(len(rep_labels)),
                    "n_pairs": 0,
                    "ari_mean": float("nan"),
                    "ari_std": float("nan"),
                    "ari_min": float("nan"),
                    "ari_max": float("nan"),
                }
            )
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The number of unique classes is greater than 50% of the number of samples.*",
                category=UserWarning,
            )
            pair_aris = [
                float(adjusted_rand_score(a, b))
                for a, b in combinations(rep_labels, 2)
            ]
        stability_rows.append(
            {
                "k": int(k),
                "n_bootstrap": int(len(rep_labels)),
                "n_pairs": int(len(pair_aris)),
                "ari_mean": float(np.mean(pair_aris)),
                "ari_std": float(np.std(pair_aris)),
                "ari_min": float(np.min(pair_aris)),
                "ari_max": float(np.max(pair_aris)),
            }
        )

    stability_df = pd.DataFrame(stability_rows).sort_values("k").reset_index(drop=True)
    if not stability_df.empty:
        stability_df.to_csv(run_dirs.reports / "cluster_validation_stability_bootstrap.csv", index=False)

        if stability_df["ari_mean"].notna().any():
            plt.figure(figsize=(10, 6))
            x = stability_df["k"].to_numpy(dtype=float)
            y = stability_df["ari_mean"].to_numpy(dtype=float)
            y_std = stability_df["ari_std"].fillna(0.0).to_numpy(dtype=float)
            plt.plot(x, y, marker="o", linewidth=2, color="#1f77b4", label="Mean ARI")
            plt.fill_between(x, np.clip(y - y_std, -1.0, 1.0), np.clip(y + y_std, -1.0, 1.0), alpha=0.2, color="#1f77b4", label="±1 SD")
            if selected_k > 0:
                plt.axvline(selected_k, color="red", linestyle="--", linewidth=1.2, label=f"Selected K={selected_k}")
            plt.ylim([-0.05, 1.05])
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Bootstrap stability (Adjusted Rand Index)")
            plt.title("Cluster Assignment Stability vs K", fontsize=13, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=8)

            fname = "cluster_validation_stability_bootstrap_ari.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["cluster_decision"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="cluster_decision",
                filename=fname,
                description="Bootstrap K-stability using adjusted Rand index over feature-cluster assignments.",
                ppt_hint="Slide 12-14 cluster stability diagnostics",
            )


def generate_cluster_insight_figures(
    *,
    config: dict,
    run_dirs,
    section_dirs: dict[str, Path],
    manifest_rows: list[dict[str, str]],
    ctx: dict[str, Any],
    plt,
    sns,
) -> None:
    linkage_method = str(config.get("cluster", {}).get("linkage", "ward"))
    dist = feature_distance_matrix(ctx["x_train_scaled"])
    link = fit_hierarchical_clustering(dist, linkage_method=linkage_method)

    plt.figure(figsize=(14, 6))
    dendrogram(link, no_labels=True, color_threshold=0.7 * float(np.max(link[:, 2])))
    plt.title("Hierarchical Feature Clustering Dendrogram", fontsize=13, fontweight="bold")
    plt.xlabel("Feature Index")
    plt.ylabel("Ward Distance")
    fname = "feature_clustering_dendrogram.png"
    _save_plot(
        plt_mod=plt,
        report_path=run_dirs.reports / fname,
        section_path=section_dirs["cluster_insights"] / fname,
    )
    _record_manifest(
        manifest_rows,
        run_dirs=run_dirs,
        section_key="cluster_insights",
        filename=fname,
        description="Feature dendrogram from 1-|r| distance with Ward linkage.",
        ppt_hint="Slide 13 feature clustering methodology",
    )

    pca_full = PCA(n_components=min(50, ctx["x_train_scaled"].shape[1]), random_state=42)
    pca_full.fit(ctx["x_train_scaled"])
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(np.arange(1, len(explained) + 1), explained, marker="o", linewidth=1.5)
    axes[0].set_title("PCA Component Variance")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].grid(alpha=0.3)
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", linewidth=1.5, color="darkorange")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Cumulative Variance")
    axes[1].axhline(0.8, color="red", linestyle="--", linewidth=1)
    axes[1].grid(alpha=0.3)
    fname = "pca_explained_variance.png"
    _save_plot(
        plt_mod=plt,
        report_path=run_dirs.reports / fname,
        section_path=section_dirs["cluster_insights"] / fname,
    )
    _record_manifest(
        manifest_rows,
        run_dirs=run_dirs,
        section_key="cluster_insights",
        filename=fname,
        description="PCA explained and cumulative variance curves.",
        ppt_hint="Slide 8 dimensionality and embedding insights",
    )

    # Cluster importance (best fine-tune config).
    best_cfg = _safe_read_json(run_dirs.cluster_models / "best_fine_tune_config.json", required=False)
    if best_cfg:
        best_k = int(best_cfg.get("n_clusters", 0))
        importance_path = run_dirs.cluster_models / f"cluster_importance_{best_k}clusters.csv"
        cluster_imp_df = _safe_read_csv(importance_path, required=False)
        top_feats_df = _safe_read_csv(
            run_dirs.cluster_models / f"top_features_by_cluster_{best_k}clusters.csv",
            required=False,
        )
        if not cluster_imp_df.empty:
            if "cluster_id" not in cluster_imp_df.columns:
                cluster_imp_df["cluster_id"] = (
                    cluster_imp_df["feature"]
                    .astype(str)
                    .str.extract(r"(\d+)", expand=False)
                    .astype(float)
                    .astype("Int64")
                )
            top_n = min(5, len(cluster_imp_df))
            top_clusters = (
                cluster_imp_df.nsmallest(0, "importance")["cluster_id"].dropna().astype(int).tolist()
                if top_n == 0
                else cluster_imp_df.head(top_n)["cluster_id"].dropna().astype(int).tolist()
            )
            size_map: dict[int, int] = {}
            if not top_feats_df.empty and {"cluster_id", "n_features"}.issubset(top_feats_df.columns):
                size_map = {
                    int(r["cluster_id"]): int(r["n_features"])
                    for _, r in top_feats_df.iterrows()
                    if pd.notna(r["cluster_id"])
                }
            imp_sorted = cluster_imp_df.sort_values("cluster_id")
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(
                f"Feature Importance Analysis: Best Model ({best_k} Clusters)",
                fontsize=16,
                fontweight="bold",
            )
            colors = [
                "red" if int(cid) in top_clusters else "steelblue"
                for cid in imp_sorted["cluster_id"].fillna(-1).astype(int).tolist()
            ]
            axes[0].bar(range(len(imp_sorted)), imp_sorted["importance"], color=colors, alpha=0.75)
            axes[0].set_xlabel("Cluster ID", fontsize=11)
            axes[0].set_ylabel("Importance Score (Permutation)", fontsize=11)
            axes[0].set_title(f"Cluster Importance Scores (Top {top_n} in red)", fontsize=12, fontweight="bold")
            axes[0].grid(True, alpha=0.3, axis="y")
            xticks = list(range(0, len(imp_sorted), max(1, len(imp_sorted) // 10)))
            axes[0].set_xticks(xticks)
            axes[0].set_xticklabels(
                imp_sorted["cluster_id"].astype("Int64").astype(str).tolist()[:: max(1, len(imp_sorted) // 10)],
                rotation=45,
            )

            top_imp = cluster_imp_df.head(top_n).sort_values("importance", ascending=True)
            labels = []
            for _, row in top_imp.iterrows():
                cid = int(row["cluster_id"]) if pd.notna(row["cluster_id"]) else -1
                n_feats = size_map.get(cid, 0)
                labels.append(f"Cluster {cid} ({n_feats} features)")
            axes[1].barh(range(len(top_imp)), top_imp["importance"], color="crimson", alpha=0.75)
            axes[1].set_yticks(range(len(top_imp)))
            axes[1].set_yticklabels(labels, fontsize=10)
            axes[1].set_xlabel("Importance Score (Permutation)", fontsize=11)
            axes[1].set_title(f"Top {top_n} Most Important Clusters", fontsize=12, fontweight="bold")
            axes[1].grid(True, alpha=0.3, axis="x")

            fname = f"cluster_importance_{best_k}clusters.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["cluster_insights"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="cluster_insights",
                filename=fname,
                description="Permutation-based cluster importance for selected best K.",
                ppt_hint="Slide 26-29 interpretability and cluster characterization",
            )
            alias_name = "cluster_importance_best.png"
            shutil.copyfile(run_dirs.reports / fname, run_dirs.reports / alias_name)
            shutil.copyfile(
                section_dirs["cluster_insights"] / fname,
                section_dirs["cluster_insights"] / alias_name,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="cluster_insights",
                filename=alias_name,
                description="Alias of selected best-K cluster importance for report stability.",
                ppt_hint="Slide 26-29 interpretability and cluster characterization",
            )

    # SHAP figure (if available) or permutation-proxy substitute.
    shap_generated = False
    try:
        import shap  # type: ignore

        model_params = dict(config.get("hyperparameters", {}).get("logistic_regression", {}))
        model = build_model("logistic_regression", model_params, random_seed=int(config["project"]["random_seed"]))
        model.fit(ctx["x_train_scaled"], ctx["y_train"])
        sample_n = min(600, len(ctx["x_test_scaled"]))
        if sample_n > 0:
            sample_x = ctx["x_test_scaled"][:sample_n]
            explainer = shap.LinearExplainer(model, ctx["x_train_scaled"], feature_perturbation="interventional")
            shap_values = explainer.shap_values(sample_x)
            if isinstance(shap_values, list):
                shap_arr = np.asarray(shap_values[0], dtype=float)
            else:
                shap_arr = np.asarray(shap_values, dtype=float)
            mean_abs = np.abs(shap_arr).mean(axis=0)
            shap_df = pd.DataFrame({"feature": ctx["usable_features"], "importance": mean_abs})
            shap_df = shap_df.sort_values("importance", ascending=False).head(20)
            plt.figure(figsize=(10, 7))
            plt.barh(shap_df["feature"][::-1], shap_df["importance"][::-1], color="darkslateblue", alpha=0.85)
            plt.xlabel("mean(|SHAP value|)")
            plt.title("SHAP Summary (Top 20 Features)")
            fname = "shap_summary.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["cluster_insights"] / fname,
            )
            shap_df.to_csv(run_dirs.reports / "shap_summary_values.csv", index=False)
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="cluster_insights",
                filename=fname,
                description="SHAP summary for logistic baseline model (top features).",
                ppt_hint="Slide 11 interpretability insights",
            )
            shap_generated = True
    except Exception:
        shap_generated = False

    if not shap_generated:
        proxy = _safe_read_csv(run_dirs.baseline / "feature_importance.csv", required=False)
        if proxy.empty:
            proxy = _safe_read_csv(run_dirs.baseline / "rf_feature_importance.csv", required=False)
        if not proxy.empty:
            proxy = proxy.sort_values("importance", ascending=False, na_position="last").head(20)
            plt.figure(figsize=(10, 7))
            plt.barh(proxy["feature"][::-1], proxy["importance"][::-1], color="slateblue", alpha=0.85)
            plt.xlabel("Importance Score")
            plt.title("SHAP Proxy: Permutation Importance Summary (Top 20)")
            fname = "shap_summary.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["cluster_insights"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="cluster_insights",
                filename=fname,
                description="SHAP proxy using permutation importance (SHAP not available).",
                ppt_hint="Slide 11 interpretability insights",
            )


def _run_progressive_selection(config: dict, ctx: dict[str, Any], run_dirs) -> pd.DataFrame:
    settings = config.get("reporting", {}).get("progressive_selection", {})
    removal_percentage = float(settings.get("removal_percentage", 0.10))
    min_features = int(settings.get("min_features", 10))
    use_clustering = bool(settings.get("use_clustering", True))
    cluster_ratio = float(settings.get("cluster_ratio", 0.8))
    n_iterations = int(settings.get("importance_iterations", 5))
    max_steps_cfg = settings.get("max_steps")
    max_steps = None if max_steps_cfg in {None, ""} else int(max_steps_cfg)

    seed = int(config["project"].get("random_seed", 42))
    model_params = dict(config.get("hyperparameters", {}).get("mlp", {}))
    linkage_method = str(config.get("cluster", {}).get("linkage", "ward"))

    # Notebook-like initialization: standardized + clipped once before iterative steps.
    x_train_current, x_val_current, x_test_current, _ = standardize_and_clip(
        x_train=ctx["x_train_raw"],
        x_val=ctx["x_val_raw"],
        x_test=ctx["x_test_raw"],
        clip_abs=8.0,
    )
    current_feature_names = list(ctx["usable_features"])

    y_train = np.asarray(ctx["y_train"]).astype(int)
    y_val = np.asarray(ctx["y_val"]).astype(int)
    y_test = np.asarray(ctx["y_test"]).astype(int)

    rows: list[dict[str, Any]] = []
    step = 0
    while len(current_feature_names) > min_features:
        step += 1
        if max_steps is not None and step > max_steps:
            break

        scaler_step = StandardScaler()
        x_train_step = scaler_step.fit_transform(np.asarray(x_train_current, dtype=np.float32))
        x_val_step = scaler_step.transform(np.asarray(x_val_current, dtype=np.float32))
        x_test_step = scaler_step.transform(np.asarray(x_test_current, dtype=np.float32))
        x_train_step = np.clip(x_train_step, -8.0, 8.0)
        x_val_step = np.clip(x_val_step, -8.0, 8.0)
        x_test_step = np.clip(x_test_step, -8.0, 8.0)

        model = build_model("mlp", model_params, random_seed=seed + step)
        try:
            model.fit(x_train_step, y_train, x_val=x_val_step, y_val=y_val)
        except TypeError:
            model.fit(x_train_step, y_train)

        test_prob = predict_positive_proba(model, x_test_step)
        test_metrics = evaluate_binary_metrics(y_test, test_prob, threshold=0.5)

        rows.append(
            {
                "step": int(step),
                "n_features": int(len(current_feature_names)),
                "feature_names": "|".join(current_feature_names),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_roc_auc": float(test_metrics["roc_auc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
            }
        )

        if len(current_feature_names) <= min_features:
            break

        importance_df = _permutation_importance(
            model=model,
            x_eval=x_val_step,
            y_eval=y_val,
            feature_names=current_feature_names,
            random_seed=seed + step,
            n_iterations=n_iterations,
        )
        sorted_features = (
            importance_df.sort_values("importance", ascending=False, na_position="last")["feature"].tolist()
        )
        n_to_remove = max(1, int(len(current_feature_names) * removal_percentage))
        n_to_keep = max(min_features, len(current_feature_names) - n_to_remove)
        top_features = sorted_features[:n_to_keep]
        keep_indices = [current_feature_names.index(f) for f in top_features]

        x_train_current = np.asarray(x_train_current)[:, keep_indices]
        x_val_current = np.asarray(x_val_current)[:, keep_indices]
        x_test_current = np.asarray(x_test_current)[:, keep_indices]
        current_feature_names = top_features

        if use_clustering and len(current_feature_names) > min_features:
            n_clusters = max(min_features, int(len(current_feature_names) * cluster_ratio))
            if n_clusters < len(current_feature_names):
                assignments = fit_cluster_assignments(
                    x_train_scaled=np.asarray(x_train_current, dtype=np.float32),
                    feature_names=current_feature_names,
                    n_clusters=n_clusters,
                    linkage_method=linkage_method,
                )
                (
                    x_train_cluster,
                    x_val_cluster,
                    x_test_cluster,
                    cluster_feature_names,
                ) = aggregate_features_by_cluster(
                    x_train_scaled=np.asarray(x_train_current, dtype=np.float32),
                    x_val_scaled=np.asarray(x_val_current, dtype=np.float32),
                    x_test_scaled=np.asarray(x_test_current, dtype=np.float32),
                    feature_names=current_feature_names,
                    assignments=assignments,
                    method="mean",
                )
                x_train_current = x_train_cluster
                x_val_current = x_val_cluster
                x_test_current = x_test_cluster
                current_feature_names = cluster_feature_names

    out = pd.DataFrame(rows)
    out.to_csv(run_dirs.reports / "progressive_selection_results.csv", index=False)
    if not out.empty:
        best_row = out.sort_values("test_roc_auc", ascending=False, na_position="last").head(1)
        best_row.to_csv(run_dirs.reports / "progressive_selection_summary.csv", index=False)
    return out


def generate_progressive_selection_figures(
    *,
    config: dict,
    run_dirs,
    section_dirs: dict[str, Path],
    manifest_rows: list[dict[str, str]],
    ctx: dict[str, Any],
    plt,
) -> None:
    progressive_df = _safe_read_csv(run_dirs.reports / "progressive_selection_results.csv", required=False)
    if progressive_df.empty:
        progressive_df = _run_progressive_selection(config, ctx, run_dirs)
    if progressive_df.empty:
        return

    sorted_df = progressive_df.sort_values("n_features", ascending=False).copy()
    initial = sorted_df.iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Progressive Feature Selection: Performance vs Number of Features",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].plot(sorted_df["n_features"], sorted_df["test_roc_auc"], marker="o", linewidth=2, markersize=6)
    axes[0, 0].axhline(y=initial["test_roc_auc"], color="r", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Number of Features")
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[0, 0].set_title("ROC-AUC vs Number of Features", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_xaxis()

    axes[0, 1].plot(sorted_df["n_features"], sorted_df["test_accuracy"], marker="s", linewidth=2, markersize=6, color="orange")
    axes[0, 1].axhline(y=initial["test_accuracy"], color="r", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Number of Features")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy vs Number of Features", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].invert_xaxis()

    axes[1, 0].plot(sorted_df["n_features"], sorted_df["test_f1"], marker="^", linewidth=2, markersize=6, color="green")
    axes[1, 0].axhline(y=initial["test_f1"], color="r", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Number of Features")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("F1 Score vs Number of Features", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_xaxis()

    auc_drop = (float(initial["test_roc_auc"]) - sorted_df["test_roc_auc"]) * 100.0
    axes[1, 1].plot(sorted_df["n_features"], auc_drop, marker="D", linewidth=2, markersize=6, color="red")
    axes[1, 1].axhline(y=0.0, color="k", linestyle="-", alpha=0.3)
    axes[1, 1].set_xlabel("Number of Features")
    axes[1, 1].set_ylabel("Performance Drop (%)")
    axes[1, 1].set_title("Performance Drop from Initial", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].invert_xaxis()

    fname = "progressive_selection_performance.png"
    _save_plot(
        plt_mod=plt,
        report_path=run_dirs.reports / fname,
        section_path=section_dirs["progressive"] / fname,
    )
    _record_manifest(
        manifest_rows,
        run_dirs=run_dirs,
        section_key="progressive",
        filename=fname,
        description="Progressive feature selection performance panel.",
        ppt_hint="Slide 10 progressive feature selection",
    )


def _plot_cv_results_standardized(cv_df: pd.DataFrame, model_display: str, plt, report_path: Path, section_path: Path) -> None:
    if cv_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{model_display}: 5-Fold Cross-Validation Results", fontsize=16, fontweight="bold")

    folds = np.arange(1, len(cv_df) + 1)
    metrics = {
        "Accuracy": cv_df["accuracy"].to_numpy(dtype=float),
        "ROC-AUC": cv_df["roc_auc"].to_numpy(dtype=float),
        "F1 Score": cv_df["f1"].to_numpy(dtype=float),
    }

    ax1 = axes[0, 0]
    box_data = [metrics["Accuracy"], metrics["ROC-AUC"], metrics["F1 Score"]]
    bp = ax1.boxplot(box_data, labels=["Accuracy", "ROC-AUC", "F1"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["steelblue", "darkorange", "forestgreen"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax1.set_title("Test Metrics Distribution Across Folds", fontsize=13, fontweight="bold")
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[0, 1]
    ax2.plot(folds, metrics["Accuracy"], "o-", linewidth=2.5, markersize=10, color="steelblue")
    ax2.axhline(y=float(np.nanmean(metrics["Accuracy"])), color="r", linestyle="--", linewidth=2)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Across Folds", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(folds, metrics["ROC-AUC"], "o-", linewidth=2.5, markersize=10, color="darkorange")
    ax3.axhline(y=float(np.nanmean(metrics["ROC-AUC"])), color="r", linestyle="--", linewidth=2)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("ROC-AUC")
    ax3.set_title("ROC-AUC Across Folds", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(folds, metrics["F1 Score"], "o-", linewidth=2.5, markersize=10, color="forestgreen")
    ax4.axhline(y=float(np.nanmean(metrics["F1 Score"])), color="r", linestyle="--", linewidth=2)
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("F1 Score")
    ax4.set_title("F1 Across Folds", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    _save_plot(plt_mod=plt, report_path=report_path, section_path=section_path)


def _plot_binary_curves(
    *,
    df_model: pd.DataFrame,
    model_display: str,
    short_name: str,
    plt,
    sns,
    run_dirs,
    section_dirs,
    manifest_rows,
) -> None:
    if df_model.empty:
        return
    y_true = df_model["label"].astype(int).to_numpy()
    y_prob = df_model["probability"].astype(float).to_numpy()
    threshold = float(df_model["threshold"].iloc[0]) if "threshold" in df_model.columns else 0.5
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = _safe_auc(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc_val:.3f}" if np.isfinite(auc_val) else "ROC-AUC = NaN")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_display} Classification", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    roc_name = f"{short_name}_final_roc_curve.png" if short_name == "nn" else f"{short_name}_roc_curve.png"
    _save_plot(
        plt_mod=plt,
        report_path=run_dirs.reports / roc_name,
        section_path=section_dirs["recording"] / roc_name,
    )
    _record_manifest(
        manifest_rows,
        run_dirs=run_dirs,
        section_key="recording",
        filename=roc_name,
        description=f"Recording-level ROC curve for {model_display}.",
        ppt_hint="Slide 17-21 recording-level performance",
    )

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}" if np.isfinite(ap) else "AP = NaN")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_display} Classification", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left")
    pr_name = f"{short_name}_final_pr_curve.png" if short_name == "nn" else f"{short_name}_pr_curve.png"
    _save_plot(
        plt_mod=plt,
        report_path=run_dirs.reports / pr_name,
        section_path=section_dirs["recording"] / pr_name,
    )
    _record_manifest(
        manifest_rows,
        run_dirs=run_dirs,
        section_key="recording",
        filename=pr_name,
        description=f"Recording-level PR curve for {model_display}.",
        ppt_hint="Slide 17-21 recording-level performance",
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_display} Classification", fontsize=14, fontweight="bold")
    cm_name = f"{short_name}_final_confusion_matrix.png" if short_name == "nn" else f"{short_name}_confusion_matrix.png"
    _save_plot(
        plt_mod=plt,
        report_path=run_dirs.reports / cm_name,
        section_path=section_dirs["recording"] / cm_name,
    )
    _record_manifest(
        manifest_rows,
        run_dirs=run_dirs,
        section_key="recording",
        filename=cm_name,
        description=f"Recording-level confusion matrix for {model_display}.",
        ppt_hint="Slide 17-21 recording-level performance",
    )


def generate_recording_level_figures(
    *,
    config: dict,
    run_dirs,
    section_dirs: dict[str, Path],
    manifest_rows: list[dict[str, str]],
    plt,
    sns,
) -> None:
    baseline_results = _safe_read_csv(run_dirs.baseline / "baseline_results.csv", required=False)
    recording_preds = _safe_read_csv(run_dirs.baseline / "recording_predictions.csv", required=False)
    fold_metrics = _safe_read_csv(run_dirs.baseline / "fold_metrics.csv", required=False)
    cluster_final_results = _safe_read_csv(run_dirs.cluster_models / "final_test_results.csv", required=False)
    all_cluster_results = _safe_read_csv(run_dirs.cluster_models / "all_recording_results.csv", required=False)
    present_model_names: list[str] = []
    for frame in [baseline_results, cluster_final_results, recording_preds, fold_metrics]:
        if "model" in frame.columns:
            present_model_names.extend(frame["model"].dropna().astype(str).tolist())
    present_model_names = list(dict.fromkeys(present_model_names))
    configured_model_names = [str(v).strip().lower() for v in config.get("modeling", {}).get("models", [])]
    model_name_order: list[str] = []
    for model_name in configured_model_names:
        if model_name in present_model_names and model_name not in model_name_order:
            model_name_order.append(model_name)
    for model_name in present_model_names:
        if model_name not in model_name_order:
            model_name_order.append(model_name)

    if not fold_metrics.empty:
        for model_name in model_name_order:
            short_name = MODEL_SHORT_NAMES.get(model_name, model_name)
            model_cv = fold_metrics[fold_metrics["model"] == model_name].copy().sort_values("fold_id")
            if model_cv.empty:
                continue
            fname = f"{short_name}_cv_results_visualization.png"
            _plot_cv_results_standardized(
                model_cv,
                MODEL_DISPLAY_NAMES.get(model_name, model_name),
                plt,
                run_dirs.reports / fname,
                section_dirs["recording"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="recording",
                filename=fname,
                description=f"Cross-validation visualization panel for {MODEL_DISPLAY_NAMES.get(model_name, model_name)}.",
                ppt_hint="Slide 16-18 baseline performance stability",
            )

    history_path = run_dirs.baseline / "nn_training_history.json"
    if history_path.exists():
        history = _safe_read_json(history_path, required=False)
        if history and history.get("train_loss"):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            epochs = np.arange(1, len(history["train_loss"]) + 1)
            axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
            axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss Curves", fontsize=13, fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=2)
            axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy", linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Accuracy Curves", fontsize=13, fontweight="bold")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            fname = "nn_final_training_curves.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["recording"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="recording",
                filename=fname,
                description="Final neural network training curves (loss/accuracy).",
                ppt_hint="Slide 17 baseline neural model diagnostics",
            )

    nn_imp = _safe_read_csv(run_dirs.baseline / "feature_importance.csv", required=False)
    if not nn_imp.empty:
        top_df = nn_imp.sort_values("importance", ascending=False, na_position="last").head(25)
        plt.figure(figsize=(12, 8))
        plt.barh(top_df["feature"][::-1], top_df["importance"][::-1], color="royalblue", alpha=0.85)
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title("Top 25 Most Important Features (Neural Network)", fontsize=14, fontweight="bold")
        fname = "feature_importance.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["recording"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="recording",
            filename=fname,
            description="Neural network feature importance (permutation).",
            ppt_hint="Slide 26 interpretability context",
        )

    rf_imp = _safe_read_csv(run_dirs.baseline / "rf_feature_importance.csv", required=False)
    if not rf_imp.empty:
        top_df = rf_imp.sort_values("importance", ascending=False, na_position="last").head(25)
        plt.figure(figsize=(12, 8))
        plt.barh(top_df["feature"][::-1], top_df["importance"][::-1], color="seagreen", alpha=0.85)
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title("Top 25 Most Important Features (Random Forest)", fontsize=14, fontweight="bold")
        fname = "rf_feature_importance.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["recording"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="recording",
            filename=fname,
            description="Random forest feature importance plot.",
            ppt_hint="Slide 26 interpretability context",
        )

    if not recording_preds.empty:
        for model_name in model_name_order:
            short_name = MODEL_SHORT_NAMES.get(model_name, model_name)
            model_df = recording_preds[recording_preds["model"] == model_name].copy()
            if model_df.empty:
                continue
            _plot_binary_curves(
                df_model=model_df,
                model_display=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                short_name=short_name,
                plt=plt,
                sns=sns,
                run_dirs=run_dirs,
                section_dirs=section_dirs,
                manifest_rows=manifest_rows,
            )

    if not baseline_results.empty:
        model_order = _ordered_model_displays(
            config,
            baseline_results["model_display"].dropna().astype(str).unique().tolist(),
        )
        metrics = ["accuracy", "roc_auc", "f1"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
        palette = sns.color_palette("tab10", n_colors=max(1, len(model_order)))
        for metric, ax in zip(metrics, axes):
            vals = []
            for model_display in model_order:
                row = baseline_results[baseline_results["model_display"] == model_display]
                vals.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
            ax.bar(model_order, vals, color=palette[: len(model_order)], alpha=0.8)
            for idx, val in enumerate(vals):
                if np.isfinite(val):
                    ax.text(idx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_ylim([0, 1.05])
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
            ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="x", rotation=25)
        plt.suptitle("Model Comparison: Recording-Level Baseline", fontsize=15, fontweight="bold")
        fname = "model_comparison.png"
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / fname,
            section_path=section_dirs["recording"] / fname,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="recording",
            filename=fname,
            description="Recording-level baseline model comparison (accuracy, ROC-AUC, F1).",
            ppt_hint="Slide 16-18 baseline performance",
        )

    # Baseline vs locked clustered final test results (or fallback to best test rows).
    best_cluster = pd.DataFrame()
    if not cluster_final_results.empty:
        best_cluster = cluster_final_results.copy().reset_index(drop=True)
    elif not all_cluster_results.empty:
        fallback = all_cluster_results.copy()
        if "evaluation_split" in fallback.columns:
            fallback = fallback[fallback["evaluation_split"] == "test"].copy()
        if not fallback.empty:
            best_cluster = (
                fallback.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last")
                .groupby("model", as_index=False)
                .head(1)
                .reset_index(drop=True)
            )

    if not baseline_results.empty and not best_cluster.empty:
        merged_rows: list[dict[str, Any]] = []
        for _, row in baseline_results.iterrows():
            model_name = str(row["model"])
            cluster_row = best_cluster[best_cluster["model"] == model_name]
            if cluster_row.empty:
                continue
            cluster_best = cluster_row.iloc[0]
            merged_rows.append(
                {
                    "model_display": str(row["model_display"]),
                    "baseline_roc_auc": float(row["roc_auc"]),
                    "cluster_best_roc_auc": float(cluster_best["roc_auc"]),
                    "baseline_f1": float(row["f1"]),
                    "cluster_best_f1": float(cluster_best["f1"]),
                    "baseline_accuracy": float(row["accuracy"]),
                    "cluster_best_accuracy": float(cluster_best["accuracy"]),
                }
            )
        comp_df = pd.DataFrame(merged_rows)
        if not comp_df.empty:
            metrics = ["roc_auc", "accuracy", "f1"]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            x = np.arange(len(comp_df))
            width = 0.35
            for metric, ax in zip(metrics, axes):
                ax.bar(x - width / 2, comp_df[f"baseline_{metric}"], width=width, label="Baseline", alpha=0.8)
                ax.bar(
                    x + width / 2,
                    comp_df[f"cluster_best_{metric}"],
                    width=width,
                    label="Best Clustered",
                    alpha=0.8,
                )
                ax.set_ylim([0, 1.05])
                ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(comp_df["model_display"], rotation=25)
                ax.grid(True, alpha=0.3, axis="y")
                ax.legend(fontsize=9)
            plt.suptitle("Baseline vs Best Clustered Configuration (Recording Level)", fontsize=15, fontweight="bold")
            fname = "fig_baseline_vs_best_cluster.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["recording"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="recording",
                filename=fname,
                description="Baseline vs locked clustered final-test recording-level performance per model.",
                ppt_hint="Slide 19-24 baseline vs clustered final comparison",
            )


def generate_patient_level_figures(
    *,
    config: dict,
    run_dirs,
    section_dirs: dict[str, Path],
    manifest_rows: list[dict[str, str]],
    plt,
    sns,
) -> None:
    patient_df = _safe_read_csv(run_dirs.reports / "patient_level_results.csv", required=False)
    patient_selection_df = _safe_read_csv(
        run_dirs.reports / "patient_level_selection_results.csv",
        required=False,
    )
    patient_pred_df = _safe_read_csv(run_dirs.reports / "patient_level_predictions.csv", required=False)
    if patient_df.empty:
        return

    colors = {"mean": "blue", "median": "green", "max": "red", "min": "purple", "std": "orange"}

    def _split_label(df: pd.DataFrame, default: str) -> str:
        if "evaluation_split" not in df.columns:
            return default
        vals = sorted({str(v) for v in df["evaluation_split"].dropna().astype(str).tolist()})
        return "/".join(vals) if vals else default

    def _render_patient_panels(
        *,
        source_df: pd.DataFrame,
        split_label: str,
        comparison_filename: str,
        line_filename: str,
        manifest_desc_prefix: str,
        manifest_hint: str,
    ) -> None:
        if source_df.empty:
            return

        patient_methods = source_df["patient_aggregation"].dropna().astype(str).tolist()
        patient_methods = list(dict.fromkeys(patient_methods))
        model_order = _ordered_model_displays(
            config,
            source_df["model_display"].dropna().astype(str).unique().tolist(),
        )
        model_order = [m for m in model_order if m in source_df["model_display"].unique().tolist()]
        if not model_order:
            return

        # Bar comparison by patient aggregation.
        n_models = len(model_order)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
        axes_flat = np.atleast_1d(axes).flatten()
        fig.suptitle(
            f"Patient-Level Model Performance by Aggregation Method ({split_label})",
            fontsize=16,
            fontweight="bold",
        )
        for idx, model_display in enumerate(model_order):
            ax = axes_flat[idx]
            model_rows = source_df[source_df["model_display"] == model_display]
            vals = []
            for method in patient_methods:
                row = model_rows[model_rows["patient_aggregation"] == method]
                vals.append(float(row.iloc[0]["roc_auc"]) if not row.empty else np.nan)
            ax.bar(
                np.arange(len(patient_methods)),
                vals,
                color=[colors.get(m, "gray") for m in patient_methods],
                alpha=0.75,
                edgecolor="black",
            )
            ax.set_xticks(np.arange(len(patient_methods)))
            ax.set_xticklabels(patient_methods, fontsize=10)
            ax.set_ylabel("ROC-AUC")
            ax.set_title(model_display, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim([0, 1.0])
            for j, value in enumerate(vals):
                if np.isfinite(value):
                    ax.text(j, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].axis("off")
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / comparison_filename,
            section_path=section_dirs["patient"] / comparison_filename,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="patient",
            filename=comparison_filename,
            description=f"{manifest_desc_prefix} ROC-AUC by aggregation method and model ({split_label}).",
            ppt_hint=manifest_hint,
        )

        # Metric line chart.
        metrics = ["accuracy", "roc_auc", "f1"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Patient-Level Performance: Metrics by Model and Aggregation ({split_label})",
            fontsize=16,
            fontweight="bold",
        )
        x_positions = np.arange(len(model_order))
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            for method in patient_methods:
                vals = []
                for model_display in model_order:
                    row = source_df[
                        (source_df["model_display"] == model_display)
                        & (source_df["patient_aggregation"] == method)
                    ]
                    vals.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
                vals_arr = np.asarray(vals, dtype=float)
                valid = np.isfinite(vals_arr)
                if valid.any():
                    ax.plot(
                        x_positions[valid],
                        vals_arr[valid],
                        marker="o",
                        linewidth=2,
                        markersize=8,
                        label=method,
                        color=colors.get(method, None),
                    )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(model_order, rotation=35, ha="right", fontsize=10)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
            ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim([0, 1.05])
            ax.legend(loc="best", fontsize=8)
        _save_plot(
            plt_mod=plt,
            report_path=run_dirs.reports / line_filename,
            section_path=section_dirs["patient"] / line_filename,
        )
        _record_manifest(
            manifest_rows,
            run_dirs=run_dirs,
            section_key="patient",
            filename=line_filename,
            description=f"{manifest_desc_prefix} metric trajectories by aggregation ({split_label}).",
            ppt_hint=manifest_hint,
        )

    # Main patient figures should align with final held-out test tables.
    final_split_label = _split_label(patient_df, "test")
    _render_patient_panels(
        source_df=patient_df,
        split_label=final_split_label,
        comparison_filename="patient_level_performance_comparison.png",
        line_filename="patient_level_metrics_line_charts.png",
        manifest_desc_prefix="Final held-out patient-level",
        manifest_hint="Slide 23-24 patient final-test performance",
    )

    # Keep selection/tuning views as supplemental outputs.
    if not patient_selection_df.empty:
        tuning_label = _split_label(patient_selection_df, "val")
        _render_patient_panels(
            source_df=patient_selection_df,
            split_label=tuning_label,
            comparison_filename="patient_level_performance_comparison_selection.png",
            line_filename="patient_level_metrics_line_charts_selection.png",
            manifest_desc_prefix="Selection-split patient-level",
            manifest_hint="Supplemental patient aggregation tuning trends",
        )
        patient_selection_df.to_csv(
            run_dirs.reports / "patient_level_plot_data_selection.csv",
            index=False,
        )
    patient_df.to_csv(
        run_dirs.reports / "patient_level_plot_data_final_test.csv",
        index=False,
    )

    if not patient_pred_df.empty:
        best_overall = _safe_read_csv(run_dirs.reports / "patient_level_best_overall.csv", required=False)
        if best_overall.empty:
            best_overall = (
                patient_df.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last").head(1)
            )
        if not best_overall.empty:
            row = best_overall.iloc[0]
            subset = patient_pred_df[
                (patient_pred_df["model"] == row["model"])
                & (patient_pred_df["patient_aggregation"] == row["patient_aggregation"])
            ].copy()
            if not subset.empty:
                y_true = subset["label"].astype(int).to_numpy()
                y_prob = subset["probability"].astype(float).to_numpy()
                threshold = float(subset["threshold"].iloc[0]) if "threshold" in subset.columns else 0.5
                y_pred = (y_prob >= threshold).astype(int)

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=True)
                axes[0].set_title("Patient-Level Confusion Matrix", fontsize=13, fontweight="bold")
                axes[0].set_xlabel("Predicted")
                axes[0].set_ylabel("True")

                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_val = _safe_auc(y_true, y_prob)
                axes[1].plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc_val:.3f}")
                axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
                axes[1].set_title("Patient-Level ROC Curve", fontsize=13, fontweight="bold")
                axes[1].set_xlabel("False Positive Rate")
                axes[1].set_ylabel("True Positive Rate")
                axes[1].legend(loc="lower right")
                axes[1].grid(alpha=0.3)

                fname = "patient_level_confusion_roc.png"
                _save_plot(
                    plt_mod=plt,
                    report_path=run_dirs.reports / fname,
                    section_path=section_dirs["patient"] / fname,
                )
                _record_manifest(
                    manifest_rows,
                    run_dirs=run_dirs,
                    section_key="patient",
                    filename=fname,
                    description="Best patient-level confusion matrix and ROC curve.",
                    ppt_hint="Slide 23-24 patient-level best configuration",
                )

    # Recording vs patient-level comparison on final held-out test results.
    baseline_df = _safe_read_csv(run_dirs.baseline / "baseline_results.csv", required=False)
    cluster_final_df = _safe_read_csv(run_dirs.cluster_models / "final_test_results.csv", required=False)
    all_cluster_df = _safe_read_csv(run_dirs.cluster_models / "all_recording_results.csv", required=False)
    if not baseline_df.empty and (not cluster_final_df.empty or not all_cluster_df.empty):
        if not cluster_final_df.empty:
            rec_best = cluster_final_df.copy().reset_index(drop=True)
        else:
            fallback = all_cluster_df.copy()
            if "evaluation_split" in fallback.columns:
                fallback = fallback[fallback["evaluation_split"] == "test"].copy()
            rec_best = (
                fallback.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last")
                .groupby("model", as_index=False)
                .head(1)
                .reset_index(drop=True)
            )

        # Compare against final patient-level test result per model.
        patient_best = (
            patient_df.sort_values(["roc_auc", "f1"], ascending=[False, False], na_position="last")
            .groupby("model", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        rows: list[dict[str, Any]] = []
        for model_name in patient_best["model"].unique().tolist():
            rec_row = rec_best[rec_best["model"] == model_name]
            if rec_row.empty:
                rec_row = baseline_df[baseline_df["model"] == model_name]
            pat_row = patient_best[patient_best["model"] == model_name]
            if rec_row.empty or pat_row.empty:
                continue
            rec = rec_row.iloc[0]
            pat = pat_row.iloc[0]
            rows.append(
                {
                    "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "recording_accuracy": float(rec["accuracy"]),
                    "patient_accuracy": float(pat["accuracy"]),
                    "recording_roc_auc": float(rec["roc_auc"]),
                    "patient_roc_auc": float(pat["roc_auc"]),
                    "recording_f1": float(rec["f1"]),
                    "patient_f1": float(pat["f1"]),
                }
            )
        comp_df = pd.DataFrame(rows)
        if not comp_df.empty:
            metrics = ["accuracy", "roc_auc", "f1"]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            x = np.arange(len(comp_df))
            width = 0.35
            for metric, ax in zip(metrics, axes):
                ax.bar(x - width / 2, comp_df[f"recording_{metric}"], width=width, label="Recording-Level", alpha=0.8)
                ax.bar(x + width / 2, comp_df[f"patient_{metric}"], width=width, label="Patient-Level", alpha=0.8)
                ax.set_ylim([0, 1.05])
                ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(comp_df["model_display"], rotation=25)
                ax.grid(True, alpha=0.3, axis="y")
                ax.legend(fontsize=9)
            plt.suptitle("Recording-Level vs Patient-Level Performance", fontsize=15, fontweight="bold")
            fname = "recording_vs_patient_level_comparison.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["patient"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="patient",
                filename=fname,
                description="Recording-level (final test) versus patient-level (final test) metric comparison.",
                ppt_hint="Slide 24 final test-level comparison",
            )

            summary_rows = []
            for _, row in comp_df.iterrows():
                summary_rows.append(
                    {
                        "model_display": row["model_display"],
                        "recording_roc_auc": row["recording_roc_auc"],
                        "patient_roc_auc": row["patient_roc_auc"],
                        "recording_accuracy": row["recording_accuracy"],
                        "patient_accuracy": row["patient_accuracy"],
                        "recording_f1": row["recording_f1"],
                        "patient_f1": row["patient_f1"],
                    }
                )
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(run_dirs.reports / "final_summary_all_models_levels.csv", index=False)

            plt.figure(figsize=(10, 6))
            heatmap_df = summary_df.set_index("model_display")[
                [
                    "recording_roc_auc",
                    "patient_roc_auc",
                    "recording_accuracy",
                    "patient_accuracy",
                    "recording_f1",
                    "patient_f1",
                ]
            ]
            sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Score"})
            plt.title("Final Summary: All Models and Levels", fontsize=14, fontweight="bold")
            fname = "final_summary_all_models_levels.png"
            _save_plot(
                plt_mod=plt,
                report_path=run_dirs.reports / fname,
                section_path=section_dirs["patient"] / fname,
            )
            _record_manifest(
                manifest_rows,
                run_dirs=run_dirs,
                section_key="patient",
                filename=fname,
                description="Summary heatmap across recording-level and patient-level metrics.",
                ppt_hint="Slide 24 summary panel",
            )


def export_report_tables(run_dirs) -> None:
    table_pairs = [
        (run_dirs.cluster_models / "coarse_results.csv", run_dirs.reports / "table_coarse_results.csv"),
        (run_dirs.cluster_models / "fine_tune_results.csv", run_dirs.reports / "table_fine_tune_results.csv"),
        (
            run_dirs.cluster_models / "final_test_results.csv",
            run_dirs.reports / "table_recording_level_final_test_results.csv",
        ),
        (run_dirs.reports / "patient_level_results.csv", run_dirs.reports / "table_patient_level_results.csv"),
        (
            run_dirs.reports / "patient_level_selection_results.csv",
            run_dirs.reports / "table_patient_level_selection_results.csv",
        ),
        (
            run_dirs.reports / "patient_level_best_configurations.csv",
            run_dirs.reports / "table_patient_level_best_configurations.csv",
        ),
        (
            run_dirs.cluster_models / "coarse_k_contenders_selection_scope.csv",
            run_dirs.reports / "table_coarse_k_contenders_selection_scope.csv",
        ),
    ]
    for src, dst in table_pairs:
        if src.exists():
            pd.read_csv(src).to_csv(dst, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.phenotype_config)
    seed_everything(
        int(config["project"].get("random_seed", 42)),
        deterministic=get_deterministic_torch(config),
    )

    phenotype_name = config["phenotype"]["name"]
    run_dirs = build_run_dirs(config["paths"]["outputs_root"], phenotype_name, args.run_id)
    analysis_mode = get_analysis_mode(config)
    tuning_split = get_tuning_split(config)

    export_report_tables(run_dirs)

    if os.environ.get("VOICE_SCREENING_SKIP_PLOTS", "").strip().lower() in {"1", "true"}:
        update_run_manifest(
            path=run_dirs.root / "run_manifest.json",
            config=config,
            stage="06_generate_figures_tables",
            run_id=args.run_id,
            phenotype=phenotype_name,
            extra={"analysis_mode": analysis_mode, "skip_plots": True, "figure_count": 0},
            cwd=ROOT,
        )
        print(f"Tables saved under: {run_dirs.reports}")
        return

    try:
        plt, sns = _ensure_plotting()
    except Exception as exc:
        update_run_manifest(
            path=run_dirs.root / "run_manifest.json",
            config=config,
            stage="06_generate_figures_tables",
            run_id=args.run_id,
            phenotype=phenotype_name,
            extra={
                "analysis_mode": analysis_mode,
                "skip_plots": True,
                "plotting_error": str(exc),
                "figure_count": 0,
            },
            cwd=ROOT,
        )
        print(f"Plotting disabled: {exc}")
        print(f"Tables saved under: {run_dirs.reports}")
        return

    section_dirs = _make_section_dirs(run_dirs)
    manifest_rows: list[dict[str, str]] = []
    ctx = _prepared_context(config, run_dirs)

    generate_acoustic_feature_figures(
        config=config,
        run_dirs=run_dirs,
        section_dirs=section_dirs,
        manifest_rows=manifest_rows,
        plt=plt,
        sns=sns,
    )
    generate_cluster_decision_figures(
        config=config,
        run_dirs=run_dirs,
        section_dirs=section_dirs,
        manifest_rows=manifest_rows,
        plt=plt,
        sns=sns,
    )
    generate_cluster_validation_figures(
        config=config,
        run_dirs=run_dirs,
        section_dirs=section_dirs,
        manifest_rows=manifest_rows,
        ctx=ctx,
        plt=plt,
        sns=sns,
    )
    generate_cluster_insight_figures(
        config=config,
        run_dirs=run_dirs,
        section_dirs=section_dirs,
        manifest_rows=manifest_rows,
        ctx=ctx,
        plt=plt,
        sns=sns,
    )
    generate_progressive_selection_figures(
        config=config,
        run_dirs=run_dirs,
        section_dirs=section_dirs,
        manifest_rows=manifest_rows,
        ctx=ctx,
        plt=plt,
    )
    generate_recording_level_figures(
        config=config,
        run_dirs=run_dirs,
        section_dirs=section_dirs,
        manifest_rows=manifest_rows,
        plt=plt,
        sns=sns,
    )
    generate_patient_level_figures(
        config=config,
        run_dirs=run_dirs,
        section_dirs=section_dirs,
        manifest_rows=manifest_rows,
        plt=plt,
        sns=sns,
    )

    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_df["order"] = np.arange(1, len(manifest_df) + 1, dtype=int)
        manifest_df.to_csv(run_dirs.reports / "figure_manifest.csv", index=False)

    save_json(
        {
            "run_id": args.run_id,
            "phenotype": phenotype_name,
            "figure_count": int(len(manifest_rows)),
            "analysis_mode": analysis_mode,
            "tuning_split": tuning_split,
            "sections": {k: str(v.relative_to(run_dirs.root)) for k, v in section_dirs.items()},
        },
        run_dirs.reports / "figure_manifest.json",
    )
    update_run_manifest(
        path=run_dirs.root / "run_manifest.json",
        config=config,
        stage="06_generate_figures_tables",
        run_id=args.run_id,
        phenotype=phenotype_name,
        extra={"figure_count": int(len(manifest_rows)), "analysis_mode": analysis_mode},
        cwd=ROOT,
    )

    print(f"Figures and tables saved under: {run_dirs.reports}")


if __name__ == "__main__":
    main()
