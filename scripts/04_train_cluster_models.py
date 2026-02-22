#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voice_screening.analysis import (  # noqa: E402
    get_analysis_mode,
    get_deterministic_torch,
    get_final_evaluate_all_models,
    get_final_refit_on_train_val,
    get_tuning_split,
)
from voice_screening.config import load_config  # noqa: E402
from voice_screening.io import load_dataframe, save_json  # noqa: E402
from voice_screening.modeling import build_model, predict_positive_proba, seed_everything  # noqa: E402
from voice_screening.notebook_parity import (  # noqa: E402
    NotebookSplit,
    aggregate_features_by_cluster,
    choose_small_k_contender,
    clean_feature_dataframe,
    evaluate_binary_metrics,
    fit_cluster_assignments,
    k_selection_payload,
    participant_train_val_test_split,
    summarize_split_class_balance,
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
            "Clustered training with explicit exploratory/confirmatory modes: "
            "coarse K sweep, fine-tune near contender region, and locked final test evaluation."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phenotype-config", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _normalize_k_values(values: list[int], min_k: int, max_k: int, n_features: int) -> list[int]:
    cleaned = sorted({int(k) for k in values})
    upper = min(int(max_k), int(n_features))
    lower = max(2, int(min_k))
    return [k for k in cleaned if lower <= k <= upper]


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


def _cluster_aggregation_methods(config: dict) -> list[str]:
    methods = config.get("cluster", {}).get("aggregation_methods")
    if methods is None:
        methods = [config.get("cluster", {}).get("aggregation", "mean")]

    seen: list[str] = []
    for method in methods:
        method_str = str(method).strip().lower()
        if method_str and method_str not in seen:
            seen.append(method_str)
    return seen or ["mean"]


def _search_model_names(config: dict, model_names_all: list[str]) -> list[str]:
    """
    Determine which models are used for coarse/fine K search.
    Final locked evaluation still uses all configured models when enabled.
    """
    search_cfg = config.get("cluster", {}).get("search_models")
    if search_cfg is None:
        return model_names_all

    if not isinstance(search_cfg, (list, tuple)):
        search_cfg = [search_cfg]

    seen: list[str] = []
    for name in search_cfg:
        model_name = str(name).strip().lower()
        if model_name and model_name in model_names_all and model_name not in seen:
            seen.append(model_name)

    return seen or model_names_all


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


def _model_priority_order(
    *,
    model_names: list[str],
    preferred_order: list[str] | None = None,
) -> list[str]:
    ordered: list[str] = []
    if preferred_order:
        for model_name in preferred_order:
            name = str(model_name).strip().lower()
            if name in model_names and name not in ordered:
                ordered.append(name)
    for model_name in model_names:
        if model_name not in ordered:
            ordered.append(model_name)
    return ordered


def _select_selector_model(
    *,
    coarse_results: pd.DataFrame,
    fine_cfg: dict[str, object],
    model_names: list[str],
    selection_aggregation: str,
    selection_metric: str,
) -> tuple[str, str, pd.DataFrame]:
    """
    Returns:
      selection_model, selection_model_source, selection_scope_for_model
    """
    selector_cfg = fine_cfg.get("selection_model", {})
    if not isinstance(selector_cfg, dict):
        selector_cfg = {}

    policy = str(selector_cfg.get("policy", "fixed")).strip().lower()
    fixed_model = str(selector_cfg.get("fixed", fine_cfg.get("model", "mlp"))).strip().lower()
    preferred_vals = selector_cfg.get("preferred_order", [])
    if not isinstance(preferred_vals, (list, tuple)):
        preferred_vals = [preferred_vals]
    preferred_order = [str(v).strip().lower() for v in preferred_vals]
    dl_vals = selector_cfg.get("dl_models", [])
    if not isinstance(dl_vals, (list, tuple)):
        dl_vals = [dl_vals]
    dl_models = [str(v).strip().lower() for v in dl_vals]

    if fixed_model not in model_names:
        fixed_model = model_names[0]

    scope = coarse_results[coarse_results["aggregation"] == selection_aggregation].copy()
    if scope.empty:
        scope = coarse_results.copy()

    if policy == "fixed":
        model_scope = scope[scope["model"] == fixed_model].copy()
        if model_scope.empty:
            model_scope = coarse_results[coarse_results["model"] == fixed_model].copy()
        if model_scope.empty:
            model_scope = coarse_results.copy()
        return fixed_model, "fixed", model_scope

    if selection_metric not in scope.columns:
        raise RuntimeError(f"Selection metric column not found for selector model policy: {selection_metric}")

    scope = scope.dropna(subset=[selection_metric]).copy()
    if scope.empty:
        model_scope = coarse_results[coarse_results["model"] == fixed_model].copy()
        if model_scope.empty:
            model_scope = coarse_results.copy()
        return fixed_model, "fallback_fixed_metric_missing", model_scope

    model_order = _model_priority_order(model_names=model_names, preferred_order=preferred_order)
    priority = {name: idx for idx, name in enumerate(model_order)}

    ranked = scope.copy()
    ranked["model_priority"] = ranked["model"].map(lambda m: priority.get(str(m), len(priority) + 1000))
    ranked = ranked.sort_values(
        [selection_metric, "f1", "n_clusters", "model_priority"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    per_model_best = ranked.groupby("model", as_index=False, sort=False).head(1).copy()

    candidate_models = per_model_best
    model_source = "best_available"
    if policy == "best_available_prefer_dl":
        dl_rows = per_model_best[per_model_best["model"].isin(dl_models)].copy()
        if not dl_rows.empty:
            candidate_models = dl_rows
            model_source = "best_available_prefer_dl"
        else:
            model_source = "best_available_prefer_dl_fallback_all"
    elif policy != "best_available":
        raise RuntimeError(
            f"Unsupported selection model policy '{policy}'. "
            "Use one of: fixed, best_available, best_available_prefer_dl."
        )

    chosen_row = candidate_models.sort_values(
        [selection_metric, "f1", "n_clusters", "model_priority"],
        ascending=[False, False, True, True],
        na_position="last",
    ).iloc[0]
    selection_model = str(chosen_row["model"]).strip().lower()
    model_scope = scope[scope["model"] == selection_model].copy()
    if model_scope.empty:
        model_scope = coarse_results[coarse_results["model"] == selection_model].copy()
    if model_scope.empty:
        model_scope = coarse_results.copy()
    return selection_model, model_source, model_scope


def _minmax_norm(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce")
    finite = arr[np.isfinite(arr)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmax, vmin):
        return pd.Series(1.0, index=series.index)
    return (arr - vmin) / (vmax - vmin)


def _merge_lock_diagnostics(
    *,
    frame: pd.DataFrame,
    cluster_diag_df: pd.DataFrame,
    validation_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    if not cluster_diag_df.empty:
        diag = cluster_diag_df.copy()
        if "n_clusters" not in diag.columns and "k" in diag.columns:
            diag["n_clusters"] = diag["k"]
        elif "n_clusters" in diag.columns and "k" in diag.columns:
            n_clusters_num = pd.to_numeric(diag["n_clusters"], errors="coerce")
            k_num = pd.to_numeric(diag["k"], errors="coerce")
            diag["n_clusters"] = n_clusters_num.fillna(k_num)

        keep_cols = [c for c in ["n_clusters", "silhouette_score"] if c in diag.columns]
        if keep_cols:
            diag = diag[keep_cols].copy()
            diag = diag.loc[:, ~diag.columns.duplicated()].copy()
            diag["n_clusters"] = pd.to_numeric(diag["n_clusters"], errors="coerce")
            diag = diag.dropna(subset=["n_clusters"]).copy()
            diag["n_clusters"] = diag["n_clusters"].astype(int)
            diag = diag.groupby("n_clusters", as_index=False, sort=False).first()
            out = out.merge(diag, on="n_clusters", how="left")

    if not validation_metrics_df.empty:
        vm = validation_metrics_df.copy()
        if "n_clusters" in vm.columns:
            vm["n_clusters"] = pd.to_numeric(vm["n_clusters"], errors="coerce")
            vm = vm.dropna(subset=["n_clusters"]).copy()
            vm["n_clusters"] = vm["n_clusters"].astype(int)
            keep_cols = [
                c
                for c in ["n_clusters", "within_cluster_mean_distance", "elbow_gain", "stability_ari_mean"]
                if c in vm.columns
            ]
            if keep_cols:
                vm = vm[keep_cols].copy().groupby("n_clusters", as_index=False, sort=False).first()
                out = out.merge(vm, on="n_clusters", how="left")

    return out


def _coerce_override_k(selection_cfg: dict) -> int | None:
    env_override = str(os.environ.get("VOICE_SCREENING_OVERRIDE_K", "")).strip()
    raw = env_override if env_override else selection_cfg.get("override_k")
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if not raw_str:
        return None
    return int(raw_str)


def _cluster_validation_metrics_for_k_values(
    *,
    x_train_scaled: np.ndarray,
    feature_names: list[str],
    k_values: list[int],
    linkage_method: str,
    random_seed: int,
    n_bootstrap: int,
    sample_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    if not k_values:
        return pd.DataFrame(rows)

    feature_names_arr = [str(v) for v in feature_names]
    feat_to_idx = {name: idx for idx, name in enumerate(feature_names_arr)}
    n_samples = int(x_train_scaled.shape[0])
    sample_fraction = float(np.clip(sample_fraction, 0.2, 1.0))
    sample_n = int(min(n_samples, max(20, round(sample_fraction * n_samples))))
    use_bootstrap = int(n_bootstrap) >= 2 and n_samples >= 20
    rng = np.random.default_rng(int(random_seed))

    # Fit once on full training data for elbow-like compactness and full labels.
    base_assignments_by_k: dict[int, pd.DataFrame] = {}
    for k in k_values:
        assign_df = fit_cluster_assignments(
            x_train_scaled=x_train_scaled,
            feature_names=feature_names_arr,
            n_clusters=int(k),
            linkage_method=linkage_method,
        )
        base_assignments_by_k[int(k)] = assign_df

    stability_labels: dict[int, list[np.ndarray]] = {int(k): [] for k in k_values}
    if use_bootstrap:
        for _ in range(int(n_bootstrap)):
            idx = rng.choice(n_samples, size=sample_n, replace=False)
            x_sub = np.asarray(x_train_scaled[idx], dtype=np.float32)
            for k in k_values:
                assign_sub = fit_cluster_assignments(
                    x_train_scaled=x_sub,
                    feature_names=feature_names_arr,
                    n_clusters=int(k),
                    linkage_method=linkage_method,
                )
                labels = (
                    assign_sub.set_index("feature")["cluster"]
                    .reindex(feature_names_arr)
                    .astype(float)
                    .fillna(-1)
                    .astype(int)
                    .to_numpy()
                )
                stability_labels[int(k)].append(labels)

    previous_within = np.nan
    dist_full = np.asarray(np.corrcoef(x_train_scaled, rowvar=False), dtype=float)
    dist_full = np.nan_to_num(dist_full, nan=0.0, posinf=1.0, neginf=-1.0)
    dist_full = np.clip(dist_full, -1.0, 1.0)
    dist_full = 1.0 - np.abs(dist_full)
    dist_full = np.nan_to_num(dist_full, nan=1.0, posinf=1.0, neginf=1.0)
    dist_full = (dist_full + dist_full.T) / 2.0
    np.fill_diagonal(dist_full, 0.0)

    for k in sorted({int(v) for v in k_values}):
        assign_df = base_assignments_by_k[int(k)]
        total_distance = 0.0
        total_pairs = 0
        for _, group in assign_df.groupby("cluster", sort=False):
            indices = [
                feat_to_idx[str(f)]
                for f in group["feature"].astype(str).tolist()
                if str(f) in feat_to_idx
            ]
            if len(indices) < 2:
                continue
            sub = dist_full[np.ix_(indices, indices)]
            tri = sub[np.triu_indices(len(indices), k=1)]
            if tri.size == 0:
                continue
            total_distance += float(np.sum(tri))
            total_pairs += int(tri.size)
        within_mean = float(total_distance / total_pairs) if total_pairs > 0 else float("nan")
        elbow_gain = float(previous_within - within_mean) if np.isfinite(previous_within) and np.isfinite(within_mean) else float("nan")
        previous_within = within_mean

        rep_labels = stability_labels.get(int(k), [])
        if len(rep_labels) >= 2:
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
            ari_mean = float(np.mean(pair_aris)) if pair_aris else float("nan")
        else:
            ari_mean = float("nan")

        rows.append(
            {
                "n_clusters": int(k),
                "within_cluster_mean_distance": float(within_mean),
                "elbow_gain": float(elbow_gain),
                "stability_ari_mean": float(ari_mean),
            }
        )

    return pd.DataFrame(rows)


def _choose_locked_k_from_fine_results(
    *,
    fine_results: pd.DataFrame,
    selection_metric: str,
    selection_cfg: dict,
    cluster_diag_df: pd.DataFrame,
    validation_metrics_df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, dict[str, object]]:
    if fine_results.empty:
        raise RuntimeError("Fine-tune results are empty; cannot determine best configuration.")

    lock_strategy = str(selection_cfg.get("lock_strategy", "performance_only")).strip().lower()
    near_top_delta_validity = float(selection_cfg.get("validity", {}).get("near_top_delta", 0.01))
    weights_cfg = selection_cfg.get("validity", {}).get("weights", {})
    w_performance = float(weights_cfg.get("performance", 0.60))
    w_stability = float(weights_cfg.get("stability", 0.20))
    w_elbow = float(weights_cfg.get("elbow", 0.10))
    w_silhouette = float(weights_cfg.get("silhouette", 0.05))
    w_small_k = float(weights_cfg.get("small_k", 0.05))

    selection_details: dict[str, object] = {
        "lock_strategy": lock_strategy,
        "validity_near_top_delta": near_top_delta_validity,
        "validity_weights": {
            "performance": w_performance,
            "stability": w_stability,
            "elbow": w_elbow,
            "silhouette": w_silhouette,
            "small_k": w_small_k,
        },
    }

    work = fine_results.copy()
    work = work.sort_values("n_clusters").reset_index(drop=True)
    if selection_metric not in work.columns:
        raise RuntimeError(f"Selection metric column not found in fine-tune results: {selection_metric}")

    override_k = _coerce_override_k(selection_cfg)
    if override_k is not None:
        if int(override_k) not in set(pd.to_numeric(work["n_clusters"], errors="coerce").dropna().astype(int).tolist()):
            valid = sorted({int(v) for v in pd.to_numeric(work["n_clusters"], errors="coerce").dropna().astype(int).tolist()})
            raise RuntimeError(
                f"Requested override K={override_k} is not in fine-tune results. Valid K values: {valid}"
            )
        ranked = _merge_lock_diagnostics(
            frame=work,
            cluster_diag_df=cluster_diag_df,
            validation_metrics_df=validation_metrics_df,
        )
        ranked["n_clusters"] = pd.to_numeric(ranked["n_clusters"], errors="coerce").astype("Int64")
        row = ranked[ranked["n_clusters"] == int(override_k)].copy()
        row = row.sort_values([selection_metric], ascending=[False], na_position="last").iloc[0]
        selection_details["lock_source"] = "manual_override_k"
        selection_details["override_k"] = int(override_k)
        selection_details["candidate_k_values"] = [int(v) for v in ranked["n_clusters"].dropna().astype(int).tolist()]
        selection_details["candidate_count"] = int(len(ranked))
        selection_details["selected_k"] = int(row["n_clusters"])
        selection_details["selected_score_components"] = {
            selection_metric: float(row.get(selection_metric, np.nan)),
            "stability_ari_mean": float(row.get("stability_ari_mean", np.nan)),
            "accuracy": float(row.get("accuracy", np.nan)),
            "f1": float(row.get("f1", np.nan)),
        }
        return row, ranked.sort_values("n_clusters").reset_index(drop=True), selection_details

    if lock_strategy == "performance_only":
        best_idx = work[selection_metric].idxmax()
        best_row = work.loc[best_idx]
        selection_details["lock_source"] = "performance_only_argmax"
        return best_row, work, selection_details

    if lock_strategy == "stability_only":
        candidates = _merge_lock_diagnostics(
            frame=work,
            cluster_diag_df=cluster_diag_df,
            validation_metrics_df=validation_metrics_df,
        )

        stability_series = pd.to_numeric(
            candidates.get("stability_ari_mean", pd.Series(np.nan, index=candidates.index)),
            errors="coerce",
        )
        candidates["stability_ari_mean"] = stability_series

        if stability_series.notna().any():
            best_idx = candidates.sort_values(
                ["stability_ari_mean", selection_metric, "n_clusters"],
                ascending=[False, False, True],
                na_position="last",
            ).index[0]
            best_row = candidates.loc[best_idx]
            selection_details["lock_source"] = "stability_only_argmax"
            selection_details["candidate_k_values"] = [int(v) for v in candidates["n_clusters"].tolist()]
            selection_details["candidate_count"] = int(len(candidates))
            selection_details["selected_k"] = int(best_row["n_clusters"])
            selection_details["selected_score_components"] = {
                "stability_ari_mean": float(best_row.get("stability_ari_mean", np.nan)),
                selection_metric: float(best_row.get(selection_metric, np.nan)),
            }
            return best_row, candidates.sort_values("n_clusters").reset_index(drop=True), selection_details

        # Safety fallback if stability metric could not be computed.
        best_idx = work[selection_metric].idxmax()
        best_row = work.loc[best_idx]
        selection_details["lock_source"] = "stability_only_fallback_performance"
        return best_row, work, selection_details

    if lock_strategy == "balanced_stability_performance":
        balanced_cfg = selection_cfg.get("balanced", {})
        if not isinstance(balanced_cfg, dict):
            balanced_cfg = {}

        candidates = _merge_lock_diagnostics(
            frame=work,
            cluster_diag_df=cluster_diag_df,
            validation_metrics_df=validation_metrics_df,
        )

        stability_col = str(balanced_cfg.get("stability_metric", "stability_ari_mean")).strip()
        stability_delta = float(balanced_cfg.get("stability_near_top_delta", 0.03))
        roc_delta = float(balanced_cfg.get("roc_auc_near_top_delta", 0.015))
        f1_delta = float(balanced_cfg.get("f1_near_top_delta", 0.02))
        acc_delta = float(balanced_cfg.get("accuracy_near_top_delta", 0.02))
        min_votes = int(balanced_cfg.get("min_performance_votes", 2))
        prefer_smallest_k = bool(balanced_cfg.get("prefer_smallest_k", True))

        weight_cfg = balanced_cfg.get("weights", {})
        if not isinstance(weight_cfg, dict):
            weight_cfg = {}
        w_roc = float(weight_cfg.get("roc_auc", 0.53))
        w_bal_stability = float(weight_cfg.get("stability", 0.30))
        w_f1_bal = float(weight_cfg.get("f1", 0.10))
        w_acc_bal = float(weight_cfg.get("accuracy", 0.05))
        w_small_k_bal = float(weight_cfg.get("small_k", 0.02))

        candidates["stability_value"] = pd.to_numeric(
            candidates.get(stability_col, pd.Series(np.nan, index=candidates.index)),
            errors="coerce",
        )
        candidates["roc_auc"] = pd.to_numeric(candidates.get("roc_auc", pd.Series(np.nan, index=candidates.index)), errors="coerce")
        candidates["f1"] = pd.to_numeric(candidates.get("f1", pd.Series(np.nan, index=candidates.index)), errors="coerce")
        candidates["accuracy"] = pd.to_numeric(candidates.get("accuracy", pd.Series(np.nan, index=candidates.index)), errors="coerce")

        stable_mask = pd.Series(True, index=candidates.index)
        if candidates["stability_value"].notna().any():
            max_stability = float(candidates["stability_value"].max())
            stable_threshold = max_stability - stability_delta
            stable_mask = candidates["stability_value"] >= stable_threshold
            selection_details["balanced_stability_threshold"] = float(stable_threshold)
            selection_details["balanced_max_stability"] = float(max_stability)
        else:
            selection_details["balanced_stability_threshold"] = None
            selection_details["balanced_max_stability"] = None

        roc_mask = pd.Series(True, index=candidates.index)
        if candidates["roc_auc"].notna().any():
            max_roc = float(candidates["roc_auc"].max())
            roc_threshold = max_roc - roc_delta
            roc_mask = candidates["roc_auc"] >= roc_threshold
            selection_details["balanced_roc_threshold"] = float(roc_threshold)
            selection_details["balanced_max_roc_auc"] = float(max_roc)
        else:
            selection_details["balanced_roc_threshold"] = None
            selection_details["balanced_max_roc_auc"] = None

        f1_mask = pd.Series(True, index=candidates.index)
        if candidates["f1"].notna().any():
            max_f1 = float(candidates["f1"].max())
            f1_mask = candidates["f1"] >= (max_f1 - f1_delta)
            selection_details["balanced_max_f1"] = float(max_f1)
        else:
            selection_details["balanced_max_f1"] = None

        acc_mask = pd.Series(True, index=candidates.index)
        if candidates["accuracy"].notna().any():
            max_acc = float(candidates["accuracy"].max())
            acc_mask = candidates["accuracy"] >= (max_acc - acc_delta)
            selection_details["balanced_max_accuracy"] = float(max_acc)
        else:
            selection_details["balanced_max_accuracy"] = None

        performance_votes = roc_mask.astype(int) + f1_mask.astype(int) + acc_mask.astype(int)
        candidates["balanced_performance_votes"] = performance_votes

        strict_mask = stable_mask & roc_mask & (performance_votes >= max(1, min_votes))
        if strict_mask.any():
            gate_source = "stable_and_performance"
            gated = candidates[strict_mask].copy()
        else:
            fallback_mask = stable_mask & roc_mask
            if fallback_mask.any():
                gate_source = "stable_and_roc"
                gated = candidates[fallback_mask].copy()
            elif stable_mask.any():
                gate_source = "stable_only"
                gated = candidates[stable_mask].copy()
            elif roc_mask.any():
                gate_source = "roc_only"
                gated = candidates[roc_mask].copy()
            else:
                gate_source = "all_fine_k"
                gated = candidates.copy()

        gated["score_roc_auc"] = _minmax_norm(gated["roc_auc"])
        gated["score_stability"] = _minmax_norm(gated["stability_value"])
        gated["score_f1"] = _minmax_norm(gated["f1"])
        gated["score_accuracy"] = _minmax_norm(gated["accuracy"])
        gated["score_small_k"] = _minmax_norm(pd.Series(-gated["n_clusters"], index=gated.index))

        gated["balanced_score"] = (
            w_roc * gated["score_roc_auc"].fillna(0.0)
            + w_bal_stability * gated["score_stability"].fillna(0.0)
            + w_f1_bal * gated["score_f1"].fillna(0.0)
            + w_acc_bal * gated["score_accuracy"].fillna(0.0)
            + w_small_k_bal * gated["score_small_k"].fillna(0.0)
        )

        sort_ascending = [False, False, False, True] if prefer_smallest_k else [False, False, False, False]
        best_row = gated.sort_values(
            ["balanced_score", "roc_auc", "stability_value", "n_clusters"],
            ascending=sort_ascending,
            na_position="last",
        ).iloc[0]

        selection_details["lock_source"] = "balanced_stability_performance"
        selection_details["balanced_gate_source"] = gate_source
        selection_details["balanced_weights"] = {
            "roc_auc": w_roc,
            "stability": w_bal_stability,
            "f1": w_f1_bal,
            "accuracy": w_acc_bal,
            "small_k": w_small_k_bal,
        }
        selection_details["balanced_deltas"] = {
            "stability_near_top_delta": stability_delta,
            "roc_auc_near_top_delta": roc_delta,
            "f1_near_top_delta": f1_delta,
            "accuracy_near_top_delta": acc_delta,
            "min_performance_votes": int(min_votes),
        }
        selection_details["candidate_k_values"] = [int(v) for v in gated["n_clusters"].tolist()]
        selection_details["candidate_count"] = int(len(gated))
        selection_details["selected_k"] = int(best_row["n_clusters"])
        selection_details["selected_score_components"] = {
            "roc_auc": float(best_row.get("roc_auc", np.nan)),
            "stability_ari_mean": float(best_row.get("stability_value", np.nan)),
            "accuracy": float(best_row.get("accuracy", np.nan)),
            "f1": float(best_row.get("f1", np.nan)),
            "balanced_score": float(best_row.get("balanced_score", np.nan)),
            "balanced_performance_votes": int(best_row.get("balanced_performance_votes", 0)),
        }

        return best_row, gated.sort_values("n_clusters").reset_index(drop=True), selection_details

    if lock_strategy != "performance_plus_validity":
        raise RuntimeError(
            f"Unsupported selection lock strategy '{lock_strategy}'. "
            "Use one of: performance_only, stability_only, balanced_stability_performance, performance_plus_validity."
        )

    max_metric = float(pd.to_numeric(work[selection_metric], errors="coerce").max())
    candidate_mask = pd.to_numeric(work[selection_metric], errors="coerce") >= (max_metric - near_top_delta_validity)
    candidates = work[candidate_mask].copy()
    if candidates.empty:
        candidates = work.copy()

    candidates = _merge_lock_diagnostics(
        frame=candidates,
        cluster_diag_df=cluster_diag_df,
        validation_metrics_df=validation_metrics_df,
    )

    candidates["score_performance"] = _minmax_norm(candidates[selection_metric])
    candidates["score_stability"] = _minmax_norm(candidates.get("stability_ari_mean", pd.Series(np.nan, index=candidates.index)))
    candidates["score_elbow"] = _minmax_norm(candidates.get("elbow_gain", pd.Series(np.nan, index=candidates.index)))
    candidates["score_silhouette"] = _minmax_norm(candidates.get("silhouette_score", pd.Series(np.nan, index=candidates.index)))
    candidates["score_small_k"] = _minmax_norm(pd.Series(-candidates["n_clusters"], index=candidates.index))

    weighted = (
        w_performance * candidates["score_performance"].fillna(0.0)
        + w_stability * candidates["score_stability"].fillna(0.0)
        + w_elbow * candidates["score_elbow"].fillna(0.0)
        + w_silhouette * candidates["score_silhouette"].fillna(0.0)
        + w_small_k * candidates["score_small_k"].fillna(0.0)
    )
    candidates["validity_aware_score"] = weighted

    best_idx = candidates.sort_values(
        ["validity_aware_score", selection_metric, "n_clusters"],
        ascending=[False, False, True],
        na_position="last",
    ).index[0]
    best_row = candidates.loc[best_idx]

    selection_details["lock_source"] = "performance_plus_validity_weighted"
    selection_details["candidate_k_values"] = [int(v) for v in candidates["n_clusters"].tolist()]
    selection_details["candidate_count"] = int(len(candidates))
    selection_details["selected_k"] = int(best_row["n_clusters"])
    selection_details["selected_score_components"] = {
        "performance": float(best_row.get("score_performance", np.nan)),
        "stability": float(best_row.get("score_stability", np.nan)),
        "elbow": float(best_row.get("score_elbow", np.nan)),
        "silhouette": float(best_row.get("score_silhouette", np.nan)),
        "small_k": float(best_row.get("score_small_k", np.nan)),
        "validity_aware_score": float(best_row.get("validity_aware_score", np.nan)),
    }

    return best_row, candidates.sort_values("n_clusters").reset_index(drop=True), selection_details


def _single_split_arrays(
    cleaned_features: pd.DataFrame,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    split: NotebookSplit,
) -> dict[str, np.ndarray]:
    x_train_raw = cleaned_features.loc[split.train_mask].to_numpy(dtype=np.float32)
    x_val_raw = cleaned_features.loc[split.val_mask].to_numpy(dtype=np.float32)
    x_test_raw = cleaned_features.loc[split.test_mask].to_numpy(dtype=np.float32)

    y_arr = np.asarray(labels).astype(int)
    pid_arr = np.asarray(participant_ids).astype(str)

    y_train = y_arr[split.train_mask]
    y_val = y_arr[split.val_mask]
    y_test = y_arr[split.test_mask]

    pid_train = pid_arr[split.train_mask]
    pid_val = pid_arr[split.val_mask]
    pid_test = pid_arr[split.test_mask]

    return {
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pid_train": pid_train,
        "pid_val": pid_val,
        "pid_test": pid_test,
    }


def _run_model_grid(
    *,
    stage_name: str,
    k_values: list[int],
    cluster_aggregations: list[str],
    model_names: list[str],
    hyperparameters: dict,
    seed: int,
    threshold: float,
    feature_names: list[str],
    linkage_method: str,
    x_train_scaled: np.ndarray,
    x_val_scaled: np.ndarray,
    x_eval_scaled: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_eval: np.ndarray,
    assignments_dir: Path,
    clip_abs: float | None,
    evaluation_split: str,
    eval_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    x_train_df = pd.DataFrame(x_train_scaled, columns=feature_names)
    x_val_df = pd.DataFrame(x_val_scaled, columns=feature_names)
    x_eval_df = pd.DataFrame(x_eval_scaled, columns=feature_names)

    for k in k_values:
        assignment_df = fit_cluster_assignments(
            x_train_scaled=x_train_scaled,
            feature_names=feature_names,
            n_clusters=int(k),
            linkage_method=linkage_method,
        )

        k_dir = assignments_dir / f"k_{int(k)}"
        k_dir.mkdir(parents=True, exist_ok=True)
        assignment_df.to_csv(k_dir / "assignments.csv", index=False)

        for cluster_agg in cluster_aggregations:
            (
                x_train_agg,
                x_val_agg,
                x_eval_agg,
                cluster_feature_names,
            ) = aggregate_features_by_cluster(
                x_train_scaled=x_train_df.to_numpy(),
                x_val_scaled=x_val_df.to_numpy(),
                x_test_scaled=x_eval_df.to_numpy(),
                feature_names=feature_names,
                assignments=assignment_df,
                method=cluster_agg,
            )

            if clip_abs is None:
                scaler = StandardScaler()
                x_train_cluster = scaler.fit_transform(np.asarray(x_train_agg, dtype=np.float32))
                x_val_cluster = scaler.transform(np.asarray(x_val_agg, dtype=np.float32))
                x_eval_cluster = scaler.transform(np.asarray(x_eval_agg, dtype=np.float32))
            else:
                x_train_cluster, x_val_cluster, x_eval_cluster, _ = standardize_and_clip(
                    x_train=x_train_agg,
                    x_val=x_val_agg,
                    x_test=x_eval_agg,
                    clip_abs=float(clip_abs),
                )

            for model_name in model_names:
                model_params = dict(hyperparameters.get(model_name, {}))
                model = build_model(model_name, model_params, random_seed=seed)
                _fit_model_with_optional_val(
                    model=model,
                    x_train=x_train_cluster,
                    y_train=y_train,
                    x_val=x_val_cluster,
                    y_val=y_val,
                )

                y_prob = predict_positive_proba(model, x_eval_cluster)
                metrics = evaluate_binary_metrics(y_eval, y_prob, threshold=threshold)

                row = {
                    "search_stage": stage_name,
                    "evaluation_split": evaluation_split,
                    "n_clusters": int(k),
                    "aggregation": cluster_agg,
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "estimator_class": type(model).__name__,
                    "n_features": int(x_train_cluster.shape[1]),
                    "threshold": float(threshold),
                    **metrics,
                }

                if hasattr(model, "best_epoch_"):
                    row["best_epoch"] = int(model.best_epoch_)

                rows.append(row)
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "search_stage": stage_name,
                            "evaluation_split": evaluation_split,
                            "n_clusters": int(k),
                            "aggregation": cluster_agg,
                            "model": model_name,
                            "record_index": (
                                np.asarray(eval_indices).astype(int)
                                if eval_indices is not None
                                else np.arange(len(y_eval), dtype=int)
                            ),
                            "label": np.asarray(y_eval).astype(int),
                            "probability": np.asarray(y_prob).astype(float),
                        }
                    )
                )

    results_df = pd.DataFrame(rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return results_df, predictions_df


def _feature_category_name(feature_name: str) -> str:
    name = str(feature_name).lower()
    if "mfcc" in name:
        return "MFCC"
    if any(token in name for token in ["f0", "pitch", "jitter", "shimmer", "hnr"]):
        return "Prosody"
    if any(token in name for token in ["f1", "f2", "f3", "formant"]):
        return "Formant"
    if any(token in name for token in ["energy", "loudness"]):
        return "Energy"
    return "Other"


def _build_cluster_profile_table(
    *,
    assignments: pd.DataFrame,
    cluster_importance_df: pd.DataFrame,
) -> pd.DataFrame:
    importance_map: dict[int, float] = {}
    if not cluster_importance_df.empty and {"cluster_id", "importance"}.issubset(cluster_importance_df.columns):
        imp_df = cluster_importance_df.copy()
        imp_df["cluster_id"] = pd.to_numeric(imp_df["cluster_id"], errors="coerce")
        imp_df = imp_df.dropna(subset=["cluster_id"]).copy()
        for _, row in imp_df.iterrows():
            importance_map[int(row["cluster_id"])] = float(row.get("importance", np.nan))

    rows: list[dict[str, object]] = []
    for cluster_id, group in assignments.groupby("cluster", sort=True):
        features = sorted(group["feature"].astype(str).tolist())
        categories = [_feature_category_name(f) for f in features]
        category_order = ["MFCC", "Prosody", "Formant", "Energy", "Other"]
        category_counts = {name: int(categories.count(name)) for name in category_order}
        dominant = sorted(
            category_counts.items(),
            key=lambda kv: (-kv[1], category_order.index(kv[0])),
        )[0][0]
        breakdown = ", ".join(f"{name}:{count}" for name, count in category_counts.items() if count > 0)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_features": int(len(features)),
                "dominant_category": dominant,
                "category_breakdown": breakdown,
                "example_features": ", ".join(features[:6]),
                "cluster_importance": float(importance_map.get(int(cluster_id), np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["cluster_importance", "n_features", "cluster_id"], ascending=[False, False, True], na_position="last").reset_index(drop=True)


def _cluster_permutation_importance(
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
    baseline_metrics = evaluate_binary_metrics(y_true, y_prob, threshold=0.5)
    baseline_auc = float(baseline_metrics.get("roc_auc", np.nan))

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

        score = float(np.mean(drops)) if drops else float("nan")
        rows.append({"feature": feat_name, "importance": score})

    out = pd.DataFrame(rows).sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["metric"] = "roc_auc_drop_permutation"
    out["n_iterations"] = int(max(1, int(n_iterations)))
    return out


def _final_refit_arrays(
    *,
    split_arrays: dict[str, np.ndarray],
    feature_names: list[str],
    best_k: int,
    best_cluster_agg: str,
    linkage_method: str,
    refit_on_train_val: bool,
    clip_abs: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, pd.DataFrame, list[str]]:
    if refit_on_train_val:
        x_train_ref_raw = np.vstack([split_arrays["x_train_raw"], split_arrays["x_val_raw"]]).astype(np.float32)
        y_train_ref = np.concatenate([split_arrays["y_train"], split_arrays["y_val"]]).astype(int)

        x_train_ref_scaled, _, x_test_ref_scaled, _ = standardize_and_clip(
            x_train=x_train_ref_raw,
            x_val=x_train_ref_raw,
            x_test=split_arrays["x_test_raw"],
            clip_abs=clip_abs,
        )

        assignments = fit_cluster_assignments(
            x_train_scaled=x_train_ref_scaled,
            feature_names=feature_names,
            n_clusters=best_k,
            linkage_method=linkage_method,
        )

        x_train_agg, _, x_test_agg, cluster_feature_names = aggregate_features_by_cluster(
            x_train_scaled=x_train_ref_scaled,
            x_val_scaled=x_train_ref_scaled,
            x_test_scaled=x_test_ref_scaled,
            feature_names=feature_names,
            assignments=assignments,
            method=best_cluster_agg,
        )

        x_train_final, _, x_test_final, _ = standardize_and_clip(
            x_train=x_train_agg,
            x_val=x_train_agg,
            x_test=x_test_agg,
            clip_abs=clip_abs,
        )

        return (
            x_train_final,
            None,
            x_test_final,
            y_train_ref,
            None,
            split_arrays["y_test"].astype(int),
            assignments,
            cluster_feature_names,
        )

    assignments = fit_cluster_assignments(
        x_train_scaled=split_arrays["x_train_scaled"],
        feature_names=feature_names,
        n_clusters=best_k,
        linkage_method=linkage_method,
    )

    x_train_agg, x_val_agg, x_test_agg, cluster_feature_names = aggregate_features_by_cluster(
        x_train_scaled=split_arrays["x_train_scaled"],
        x_val_scaled=split_arrays["x_val_scaled"],
        x_test_scaled=split_arrays["x_test_scaled"],
        feature_names=feature_names,
        assignments=assignments,
        method=best_cluster_agg,
    )

    x_train_final, x_val_final, x_test_final, _ = standardize_and_clip(
        x_train=x_train_agg,
        x_val=x_val_agg,
        x_test=x_test_agg,
        clip_abs=clip_abs,
    )

    return (
        x_train_final,
        x_val_final,
        x_test_final,
        split_arrays["y_train"].astype(int),
        split_arrays["y_val"].astype(int),
        split_arrays["y_test"].astype(int),
        assignments,
        cluster_feature_names,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.phenotype_config)

    analysis_mode = get_analysis_mode(config)
    tuning_split = get_tuning_split(config)
    refit_on_train_val = get_final_refit_on_train_val(config)
    evaluate_all_models = get_final_evaluate_all_models(config)

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

    split_arrays = _single_split_arrays(
        cleaned_features=x_clean,
        labels=labels,
        participant_ids=participant_ids,
        split=split,
    )

    x_train_scaled, x_val_scaled, x_test_scaled, _ = standardize_and_clip(
        x_train=split_arrays["x_train_raw"],
        x_val=split_arrays["x_val_raw"],
        x_test=split_arrays["x_test_raw"],
        clip_abs=8.0,
    )

    split_arrays["x_train_scaled"] = x_train_scaled
    split_arrays["x_val_scaled"] = x_val_scaled
    split_arrays["x_test_scaled"] = x_test_scaled

    if tuning_split == "val":
        x_eval_scaled = x_val_scaled
        y_eval = split_arrays["y_val"]
        eval_indices = np.where(split.val_mask)[0].astype(int)
    else:
        x_eval_scaled = x_test_scaled
        y_eval = split_arrays["y_test"]
        eval_indices = np.where(split.test_mask)[0].astype(int)

    threshold = float(config.get("modeling", {}).get("threshold", 0.5))
    linkage_method = str(config.get("cluster", {}).get("linkage", "ward"))

    fine_cfg = config.get("cluster", {}).get("fine_tune", {})
    min_k = int(fine_cfg.get("min_k", 5))
    max_k = int(fine_cfg.get("max_k", len(usable_features)))

    coarse_values_cfg = config.get("cluster", {}).get(
        "exploration_k_values", config.get("cluster", {}).get("k_values", [])
    )
    coarse_k_values = _normalize_k_values(coarse_values_cfg, min_k, max_k, len(usable_features))
    if not coarse_k_values:
        raise RuntimeError("No valid coarse K values after filtering.")

    cluster_aggregation_methods = _cluster_aggregation_methods(config)

    model_names_all = [str(name).strip().lower() for name in config.get("modeling", {}).get("models", [])]
    if not model_names_all:
        raise RuntimeError("No models configured under modeling.models")
    search_model_names = _search_model_names(config, model_names_all)

    hyperparameters = dict(config.get("hyperparameters", {}))

    coarse_results, coarse_predictions = _run_model_grid(
        stage_name="coarse",
        k_values=coarse_k_values,
        cluster_aggregations=cluster_aggregation_methods,
        model_names=search_model_names,
        hyperparameters=hyperparameters,
        seed=seed,
        threshold=threshold,
        feature_names=usable_features,
        linkage_method=linkage_method,
        x_train_scaled=x_train_scaled,
        x_val_scaled=x_val_scaled,
        x_eval_scaled=x_eval_scaled,
        y_train=split_arrays["y_train"],
        y_val=split_arrays["y_val"],
        y_eval=y_eval,
        assignments_dir=run_dirs.cluster_models / "assignments" / "coarse",
        clip_abs=None,
        evaluation_split=tuning_split,
        eval_indices=eval_indices,
    )

    coarse_results.to_csv(run_dirs.cluster_models / "coarse_results.csv", index=False)
    if not coarse_predictions.empty:
        coarse_predictions.to_csv(run_dirs.cluster_models / "coarse_recording_predictions.csv", index=False)

    selection_aggregation = str(
        fine_cfg.get(
            "cluster_aggregation",
            fine_cfg.get("aggregation", config.get("cluster", {}).get("aggregation", "mean")),
        )
    ).strip().lower()
    selection_metric = str(fine_cfg.get("metric", "roc_auc")).strip().lower()

    selection_cfg = fine_cfg.get("selection", {})
    preferred_k_min = int(selection_cfg.get("preferred_k_min", min(coarse_k_values)))
    preferred_k_max = int(selection_cfg.get("preferred_k_max", max(coarse_k_values)))
    near_top_delta = float(selection_cfg.get("near_top_delta", 0.03))
    prefer_smallest_k = bool(selection_cfg.get("prefer_smallest_k", False))

    selection_model, selection_model_source, selection_scope = _select_selector_model(
        coarse_results=coarse_results,
        fine_cfg=fine_cfg,
        model_names=search_model_names,
        selection_aggregation=selection_aggregation,
        selection_metric=selection_metric,
    )
    selector_cfg = fine_cfg.get("selection_model", {})
    if not isinstance(selector_cfg, dict):
        selector_cfg = {}
    selection_model_policy = str(selector_cfg.get("policy", "fixed")).strip().lower()

    selection_result, coarse_ranked, coarse_contenders = choose_small_k_contender(
        df=selection_scope,
        metric=selection_metric,
        preferred_k_min=preferred_k_min,
        preferred_k_max=preferred_k_max,
        near_top_delta=near_top_delta,
        prefer_smallest_k=prefer_smallest_k,
    )

    manual_center = fine_cfg.get("center_k")
    if manual_center is not None:
        selected_k = int(manual_center)
        selected_k_source = "manual_center_k"
    else:
        selected_k = int(selection_result.selected_k)
        selected_k_source = "auto_coarse_contender"

    window = int(fine_cfg.get("window", 2))
    include_k = [int(k) for k in fine_cfg.get("include_k", [])]
    contender_k_values: list[int] = []
    if not coarse_contenders.empty and "n_clusters" in coarse_contenders.columns:
        contender_k_values = sorted({int(k) for k in coarse_contenders["n_clusters"].dropna().astype(int).tolist()})

    bridge_k_values: list[int] = []
    if len(contender_k_values) >= 2:
        for left_k, right_k in zip(contender_k_values[:-1], contender_k_values[1:]):
            if right_k - left_k > 1:
                bridge_k_values.extend(range(left_k + 1, right_k))

    fine_k_set = {selected_k}
    fine_k_set.update(range(selected_k - window, selected_k + window + 1))
    fine_k_set.update(contender_k_values)
    fine_k_set.update(bridge_k_values)
    fine_k_set.update(include_k)
    fine_k_values = _normalize_k_values(sorted(fine_k_set), min_k, max_k, len(usable_features))

    fine_models = [selection_model]
    fine_aggregations = [selection_aggregation]

    fine_results, fine_predictions = _run_model_grid(
        stage_name="fine_tune",
        k_values=fine_k_values,
        cluster_aggregations=fine_aggregations,
        model_names=fine_models,
        hyperparameters=hyperparameters,
        seed=seed,
        threshold=threshold,
        feature_names=usable_features,
        linkage_method=linkage_method,
        x_train_scaled=x_train_scaled,
        x_val_scaled=x_val_scaled,
        x_eval_scaled=x_eval_scaled,
        y_train=split_arrays["y_train"],
        y_val=split_arrays["y_val"],
        y_eval=y_eval,
        assignments_dir=run_dirs.cluster_models / "assignments" / "fine_tune",
        clip_abs=8.0,
        evaluation_split=tuning_split,
        eval_indices=eval_indices,
    )

    fine_results.to_csv(run_dirs.cluster_models / "fine_tune_results.csv", index=False)
    if not fine_predictions.empty:
        fine_predictions.to_csv(run_dirs.cluster_models / "fine_tune_recording_predictions.csv", index=False)

    cluster_diag_df = pd.read_csv(run_dirs.clusters / "clustering_diagnostics.csv") if (run_dirs.clusters / "clustering_diagnostics.csv").exists() else pd.DataFrame()
    validity_cfg = selection_cfg.get("validity", {})
    if not isinstance(validity_cfg, dict):
        validity_cfg = {}
    stability_cfg = validity_cfg.get("stability", {})
    if not isinstance(stability_cfg, dict):
        stability_cfg = {}
    n_bootstrap = int(stability_cfg.get("n_bootstrap", 8))
    sample_fraction = float(stability_cfg.get("sample_fraction", 0.8))

    validation_metrics_df = _cluster_validation_metrics_for_k_values(
        x_train_scaled=x_train_scaled,
        feature_names=usable_features,
        k_values=fine_k_values,
        linkage_method=linkage_method,
        random_seed=seed,
        n_bootstrap=n_bootstrap,
        sample_fraction=sample_fraction,
    )
    if not validation_metrics_df.empty:
        validation_metrics_df.to_csv(run_dirs.cluster_models / "fine_tune_cluster_validation_metrics.csv", index=False)

    best_row, fine_lock_ranked_df, lock_selection_details = _choose_locked_k_from_fine_results(
        fine_results=fine_results,
        selection_metric=selection_metric,
        selection_cfg=selection_cfg,
        cluster_diag_df=cluster_diag_df,
        validation_metrics_df=validation_metrics_df,
    )
    fine_lock_ranked_df.to_csv(run_dirs.cluster_models / "fine_tune_k_lock_ranking.csv", index=False)

    locked_k = int(best_row["n_clusters"])
    locked_cluster_agg = str(best_row["aggregation"]).strip().lower()
    importance_iterations = int(config.get("reporting", {}).get("permutation_importance_iterations", 20))

    x_train_final, x_val_final, x_test_final, y_train_final, y_val_final, y_test_final, assignments, cluster_feature_names = _final_refit_arrays(
        split_arrays=split_arrays,
        feature_names=usable_features,
        best_k=locked_k,
        best_cluster_agg=locked_cluster_agg,
        linkage_method=linkage_method,
        refit_on_train_val=refit_on_train_val,
        clip_abs=8.0,
    )

    final_model_names = model_names_all if evaluate_all_models else [selection_model]
    final_rows: list[dict[str, object]] = []
    final_pred_rows: list[pd.DataFrame] = []
    fitted_models: dict[str, object] = {}

    for model_name in final_model_names:
        model_params = dict(hyperparameters.get(model_name, {}))
        model = build_model(model_name, model_params, random_seed=seed)
        _fit_model_with_optional_val(
            model=model,
            x_train=x_train_final,
            y_train=y_train_final,
            x_val=x_val_final,
            y_val=y_val_final,
        )

        fitted_models[model_name] = model
        y_prob = predict_positive_proba(model, x_test_final)
        metrics = evaluate_binary_metrics(y_test_final, y_prob, threshold=threshold)
        y_pred = (y_prob >= threshold).astype(int)

        row = {
            "search_stage": "locked_final",
            "evaluation_split": "test",
            "n_clusters": int(locked_k),
            "aggregation": locked_cluster_agg,
            "model": model_name,
            "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "estimator_class": type(model).__name__,
            "n_features": int(x_train_final.shape[1]),
            "threshold": float(threshold),
            **metrics,
        }
        if hasattr(model, "best_epoch_"):
            row["best_epoch"] = int(model.best_epoch_)
        final_rows.append(row)

        final_pred_rows.append(
            pd.DataFrame(
                {
                    "search_stage": "locked_final",
                    "evaluation_split": "test",
                    "n_clusters": int(locked_k),
                    "aggregation": locked_cluster_agg,
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "record_index": np.where(split.test_mask)[0].astype(int),
                    "label": y_test_final.astype(int),
                    "probability": y_prob.astype(float),
                    "prediction": y_pred.astype(int),
                    "threshold": float(threshold),
                }
            )
        )

    final_test_results = pd.DataFrame(final_rows)
    final_test_predictions = pd.concat(final_pred_rows, ignore_index=True) if final_pred_rows else pd.DataFrame()

    final_test_results.to_csv(run_dirs.cluster_models / "final_test_results.csv", index=False)
    if not final_test_predictions.empty:
        final_test_predictions.to_csv(run_dirs.cluster_models / "final_test_recording_predictions.csv", index=False)

    selected_final_row_df = final_test_results[final_test_results["model"] == selection_model]
    if selected_final_row_df.empty:
        selected_final_row = final_test_results.sort_values(
            [selection_metric, "n_clusters"], ascending=[False, True], na_position="last"
        ).iloc[0]
    else:
        selected_final_row = selected_final_row_df.iloc[0]

    best_payload = {
        "search_stage": "fine_tune",
        "analysis_mode": analysis_mode,
        "selection_evaluation_split": tuning_split,
        "final_evaluation_split": "test",
        "selected_k_source": selected_k_source,
        "lock_strategy": lock_selection_details.get("lock_strategy"),
        "lock_source": lock_selection_details.get("lock_source"),
        "lock_candidate_count": lock_selection_details.get("candidate_count"),
        "lock_candidate_k_values": lock_selection_details.get("candidate_k_values"),
        "lock_validity_weights": lock_selection_details.get("validity_weights"),
        "lock_balanced_weights": lock_selection_details.get("balanced_weights"),
        "lock_balanced_deltas": lock_selection_details.get("balanced_deltas"),
        "lock_balanced_gate_source": lock_selection_details.get("balanced_gate_source"),
        "lock_override_k": lock_selection_details.get("override_k"),
        "lock_selected_score_components": lock_selection_details.get("selected_score_components"),
        "model": str(selection_model),
        "selection_model_policy": selection_model_policy,
        "selection_model_source": selection_model_source,
        "search_models": [str(v) for v in search_model_names],
        "all_models": [str(v) for v in model_names_all],
        "model_display": MODEL_DISPLAY_NAMES.get(selection_model, selection_model),
        "estimator_class": str(selected_final_row.get("estimator_class", "")),
        "n_clusters": int(locked_k),
        "aggregation": str(locked_cluster_agg),
        "n_features": int(selected_final_row["n_features"]),
        "threshold": float(selected_final_row["threshold"]),
        "accuracy": float(selected_final_row["accuracy"]),
        "roc_auc": float(selected_final_row["roc_auc"]),
        "f1": float(selected_final_row["f1"]),
        "precision": float(selected_final_row["precision"]),
        "recall": float(selected_final_row["recall"]),
        "selection_accuracy": float(best_row["accuracy"]),
        "selection_roc_auc": float(best_row["roc_auc"]),
        "selection_f1": float(best_row["f1"]),
        "selection_precision": float(best_row["precision"]),
        "selection_recall": float(best_row["recall"]),
    }
    if "best_epoch" in selected_final_row and not pd.isna(selected_final_row["best_epoch"]):
        best_payload["best_epoch"] = int(selected_final_row["best_epoch"])
    if "best_epoch" in best_row and not pd.isna(best_row["best_epoch"]):
        best_payload["selection_best_epoch"] = int(best_row["best_epoch"])

    # Keep legacy best prediction output for selected model only.
    selected_pred_df = final_test_predictions[final_test_predictions["model"] == selection_model].copy()
    if selected_pred_df.empty and not final_test_predictions.empty:
        selected_pred_df = final_test_predictions.head(0).copy()
    if not final_test_predictions.empty:
        selected_pred_df.to_csv(run_dirs.cluster_models / "best_fine_tune_recording_predictions.csv", index=False)

    # Interpretability on non-test data to avoid leakage into selection/reporting.
    importance_model = fitted_models.get(selection_model)
    if importance_model is None:
        raise RuntimeError(f"Selected model '{selection_model}' was not trained in final stage.")
    importance_x = x_train_final if x_val_final is None else x_val_final
    importance_y = y_train_final if y_val_final is None else y_val_final
    cluster_importance_df = _cluster_permutation_importance(
        model=importance_model,
        x_eval=importance_x,
        y_eval=importance_y,
        feature_names=cluster_feature_names,
        random_seed=seed,
        n_iterations=importance_iterations,
    )
    cluster_importance_df["cluster_id"] = (
        cluster_importance_df["feature"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype("Int64")
    )
    cluster_importance_df.to_csv(run_dirs.cluster_models / f"cluster_importance_{locked_k}clusters.csv", index=False)
    cluster_importance_df.to_csv(run_dirs.cluster_models / "cluster_importance_best.csv", index=False)

    cluster_feature_rows: list[dict[str, object]] = []
    for cluster_id, group in assignments.groupby("cluster"):
        feature_list = sorted(group["feature"].astype(str).tolist())
        cluster_feature_rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_features": int(len(feature_list)),
                "features": ", ".join(feature_list),
            }
        )
    top_features_df = pd.DataFrame(cluster_feature_rows).sort_values("cluster_id").reset_index(drop=True)
    top_features_df.to_csv(run_dirs.cluster_models / f"top_features_by_cluster_{locked_k}clusters.csv", index=False)
    top_features_df.to_csv(run_dirs.cluster_models / "top_features_by_cluster_best.csv", index=False)

    cluster_profile_df = _build_cluster_profile_table(
        assignments=assignments,
        cluster_importance_df=cluster_importance_df,
    )
    cluster_profile_df.to_csv(run_dirs.cluster_models / f"cluster_profiles_{locked_k}clusters.csv", index=False)
    cluster_profile_df.to_csv(run_dirs.cluster_models / "cluster_profiles_best.csv", index=False)

    if selection_model in fitted_models:
        selected_model_obj = fitted_models[selection_model]
        if hasattr(selected_model_obj, "history_"):
            save_json(selected_model_obj.history_, run_dirs.cluster_models / "best_fine_tune_training_history.json")

    all_results = pd.concat([coarse_results, fine_results, final_test_results], ignore_index=True)
    all_results.to_csv(run_dirs.cluster_models / "all_recording_results.csv", index=False)
    (
        all_results[["model", "model_display", "estimator_class"]]
        .drop_duplicates()
        .sort_values(["model", "estimator_class"])
        .reset_index(drop=True)
        .to_csv(run_dirs.cluster_models / "model_backends.csv", index=False)
    )

    if not coarse_ranked.empty:
        coarse_ranked.to_csv(run_dirs.cluster_models / "coarse_k_ranked_selection_scope.csv", index=False)
    if not coarse_contenders.empty:
        coarse_contenders.to_csv(run_dirs.cluster_models / "coarse_k_contenders_selection_scope.csv", index=False)

    locked_selection = {
        "analysis_mode": analysis_mode,
        "tuning_split": tuning_split,
        "selection_model": selection_model,
        "selection_model_policy": selection_model_policy,
        "selection_model_source": selection_model_source,
        "search_models": [str(v) for v in search_model_names],
        "all_models": [str(v) for v in model_names_all],
        "selection_aggregation": selection_aggregation,
        "selection_metric": selection_metric,
        "lock_strategy": lock_selection_details.get("lock_strategy"),
        "lock_source": lock_selection_details.get("lock_source"),
        "lock_candidate_count": lock_selection_details.get("candidate_count"),
        "lock_candidate_k_values": lock_selection_details.get("candidate_k_values"),
        "lock_validity_weights": lock_selection_details.get("validity_weights"),
        "lock_balanced_weights": lock_selection_details.get("balanced_weights"),
        "lock_balanced_deltas": lock_selection_details.get("balanced_deltas"),
        "lock_balanced_gate_source": lock_selection_details.get("balanced_gate_source"),
        "lock_override_k": lock_selection_details.get("override_k"),
        "lock_selected_score_components": lock_selection_details.get("selected_score_components"),
        "selected_k": int(locked_k),
        "selected_aggregation": locked_cluster_agg,
        "refit_on_train_val": bool(refit_on_train_val),
        "evaluate_all_models_at_locked_k": bool(evaluate_all_models),
    }

    k_selection = k_selection_payload(
        selection_result,
        extra={
            "analysis_mode": analysis_mode,
            "tuning_split": tuning_split,
            "selection_model": selection_model,
            "selection_model_policy": selection_model_policy,
            "selection_model_source": selection_model_source,
            "search_models": [str(v) for v in search_model_names],
            "all_models": [str(v) for v in model_names_all],
            "selection_aggregation": selection_aggregation,
            "selection_metric": selection_metric,
            "lock_strategy": lock_selection_details.get("lock_strategy"),
            "lock_source": lock_selection_details.get("lock_source"),
            "lock_candidate_count": lock_selection_details.get("candidate_count"),
            "lock_candidate_k_values": lock_selection_details.get("candidate_k_values"),
            "lock_validity_weights": lock_selection_details.get("validity_weights"),
            "lock_balanced_weights": lock_selection_details.get("balanced_weights"),
            "lock_balanced_deltas": lock_selection_details.get("balanced_deltas"),
            "lock_balanced_gate_source": lock_selection_details.get("balanced_gate_source"),
            "lock_override_k": lock_selection_details.get("override_k"),
            "lock_selected_score_components": lock_selection_details.get("selected_score_components"),
            "selected_k_source": selected_k_source,
            "manual_center_k": None if manual_center is None else int(manual_center),
            "coarse_k_values": [int(k) for k in coarse_k_values],
            "fine_tune_k_values": [int(k) for k in fine_k_values],
            "coarse_contender_k_values": [int(k) for k in contender_k_values],
            "coarse_contender_bridge_k_values": [int(k) for k in bridge_k_values],
        },
    )

    split_summary = {
        "analysis_mode": analysis_mode,
        "tuning_split": tuning_split,
        "refit_on_train_val": bool(refit_on_train_val),
        "train_recordings": int(split.train_mask.sum()),
        "val_recordings": int(split.val_mask.sum()),
        "test_recordings": int(split.test_mask.sum()),
        "train_participants": int(len(split.train_participants)),
        "val_participants": int(len(split.val_participants)),
        "test_participants": int(len(split.test_participants)),
        "test_fraction": float(test_fraction),
        "validation_fraction_from_train_val": float(validation_fraction),
        "split_seed": split.split_seed,
        "split_balance_score": split.balance_score,
        "split_balance_trials": int(split.balance_trials),
        "class_balance": summarize_split_class_balance(
            participant_ids=participant_ids,
            labels=labels,
            split=split,
            positive_label=1,
            positive_name=str(config.get("labeling", {}).get("positive_class_name", "positive")),
            negative_name=str(config.get("labeling", {}).get("negative_class_name", "control")),
        ),
    }

    save_json(k_selection, run_dirs.cluster_models / "k_selection.json")
    save_json(best_payload, run_dirs.cluster_models / "best_fine_tune_config.json")
    save_json(locked_selection, run_dirs.cluster_models / "locked_selection.json")
    save_json(split_summary, run_dirs.cluster_models / "split_summary.json")
    save_json(
        {
            "feature_count_input": int(len(feature_cols)),
            "feature_count_used": int(len(usable_features)),
            "removed_non_numeric_features": removed_features,
            "used_features": usable_features,
        },
        run_dirs.cluster_models / "feature_usage.json",
    )

    pd.DataFrame({"feature": usable_features}).to_csv(run_dirs.cluster_models / "feature_columns_used.csv", index=False)

    update_run_manifest(
        path=run_dirs.root / "run_manifest.json",
        config=config,
        stage="04_train_cluster_models",
        run_id=args.run_id,
        phenotype=phenotype_name,
        extra={
            "analysis_mode": analysis_mode,
            "tuning_split": tuning_split,
            "selected_k": int(locked_k),
            "selected_aggregation": locked_cluster_agg,
            "selection_model": selection_model,
            "selection_model_policy": selection_model_policy,
            "selection_model_source": selection_model_source,
            "search_models": [str(v) for v in search_model_names],
            "all_models": [str(v) for v in model_names_all],
            "lock_strategy": lock_selection_details.get("lock_strategy"),
            "lock_source": lock_selection_details.get("lock_source"),
            "lock_override_k": lock_selection_details.get("override_k"),
        },
        cwd=ROOT,
    )

    print(f"Clustered outputs saved under: {run_dirs.cluster_models}")


if __name__ == "__main__":
    main()
