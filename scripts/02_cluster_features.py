#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voice_screening.clustering import (
    cluster_assignments,
    clustering_diagnostics,
    feature_distance_matrix,
    fit_hierarchical_clustering,
)
from voice_screening.config import load_config
from voice_screening.io import load_dataframe, save_json
from voice_screening.preprocessing import FeaturePreprocessor
from voice_screening.repro import update_run_manifest
from voice_screening.run import build_run_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute feature clustering artifacts.")
    p.add_argument("--config", required=True)
    p.add_argument("--phenotype-config", required=True)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.phenotype_config)

    phenotype_name = config["phenotype"]["name"]
    run_dirs = build_run_dirs(config["paths"]["outputs_root"], phenotype_name, args.run_id)

    recordings = load_dataframe(run_dirs.prepared / "recordings.parquet")
    feature_cols = pd.read_csv(run_dirs.prepared / "feature_columns.csv")["feature"].tolist()

    pre = FeaturePreprocessor.from_config(standardize=False)
    x_all = pre.fit_transform(recordings[feature_cols])

    dist = feature_distance_matrix(x_all)
    linkage_matrix = fit_hierarchical_clustering(dist, config["cluster"]["linkage"])

    diagnostics_rows: list[dict[str, float | int]] = []

    k_values = config["cluster"].get("k_values", config["cluster"].get("exploration_k_values", []))
    if not k_values:
        raise RuntimeError("No cluster k_values configured.")

    for k in k_values:
        assign_df = cluster_assignments(feature_cols, linkage_matrix, int(k))
        labels = assign_df.sort_values("feature")["cluster"].to_numpy()
        diag = clustering_diagnostics(dist, linkage_matrix, labels)

        assign_path = run_dirs.clusters / f"cluster_assignments_{int(k)}.csv"
        assign_df.to_csv(assign_path, index=False)

        diagnostics_rows.append(
            {
                "k": int(k),
                "n_clusters": int(assign_df["cluster"].nunique()),
                "cophenetic_correlation": float(diag["cophenetic_correlation"]),
                "silhouette_score": float(diag["silhouette_score"]),
            }
        )

    pd.DataFrame(diagnostics_rows).sort_values("k").to_csv(
        run_dirs.clusters / "clustering_diagnostics.csv", index=False
    )

    save_json(
        {
            "k_values": [int(k) for k in k_values],
            "linkage": config["cluster"]["linkage"],
            "distance": "1 - abs(correlation)",
            "aggregation_default": config["cluster"]["aggregation"],
        },
        run_dirs.clusters / "clustering_metadata.json",
    )
    update_run_manifest(
        path=run_dirs.root / "run_manifest.json",
        config=config,
        stage="02_cluster_features",
        run_id=args.run_id,
        phenotype=phenotype_name,
        extra={"k_values": [int(k) for k in k_values]},
        cwd=ROOT,
    )

    print(f"Clustering artifacts saved under: {run_dirs.clusters}")


if __name__ == "__main__":
    main()
