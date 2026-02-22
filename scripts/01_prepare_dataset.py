#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voice_screening.config import load_config
from voice_screening.data import (
    apply_case_control_labeling,
    build_modeling_table,
    dataset_summary,
    load_raw_tables,
    merge_label_and_static,
    select_feature_columns,
)
from voice_screening.io import save_dataframe, save_json
from voice_screening.repro import update_run_manifest
from voice_screening.run import build_run_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare labeled voice dataset for modeling.")
    p.add_argument("--config", required=True, help="Path to base config YAML")
    p.add_argument("--phenotype-config", required=True, help="Path to phenotype config YAML")
    p.add_argument("--run-id", required=True, help="Run identifier")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.phenotype_config)

    phenotype_name = config["phenotype"]["name"]
    run_dirs = build_run_dirs(config["paths"]["outputs_root"], phenotype_name, args.run_id)

    phenotype_df, static_df = load_raw_tables(config)
    labeled_df = apply_case_control_labeling(phenotype_df, config)
    merged_df = merge_label_and_static(labeled_df, static_df, config)

    feature_cols = select_feature_columns(merged_df, config)
    model_df = build_modeling_table(merged_df, feature_cols, config)
    summary = dataset_summary(model_df, feature_cols, config)

    prepared_path = run_dirs.prepared / "recordings.parquet"
    feature_path = run_dirs.prepared / "feature_columns.csv"
    summary_path = run_dirs.prepared / "dataset_summary.json"
    config_path = run_dirs.prepared / "resolved_config.json"

    save_dataframe(model_df, prepared_path)
    pd.DataFrame({"feature": feature_cols}).to_csv(feature_path, index=False)
    save_json(summary, summary_path)
    save_json(config, config_path)
    update_run_manifest(
        path=run_dirs.root / "run_manifest.json",
        config=config,
        stage="01_prepare_dataset",
        run_id=args.run_id,
        phenotype=phenotype_name,
        extra={
            "n_records": int(summary.get("n_records", 0)),
            "n_participants": int(summary.get("n_participants", 0)),
            "n_features": int(summary.get("n_features", 0)),
        },
        cwd=ROOT,
    )

    print(f"Prepared dataset saved: {prepared_path}")
    print(f"Feature list saved: {feature_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
