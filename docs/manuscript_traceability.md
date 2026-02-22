# Manuscript Traceability

## Inputs
- Bridge2AI Voice phenotype table (`phenotype.tsv`)
- Bridge2AI Voice static acoustic features (`static_features.tsv`)

## Pipeline stage mapping
1. Cohort and feature preparation
   - Script: `scripts/01_prepare_dataset.py`
   - Outputs: `prepared/recordings.parquet`, `prepared/feature_columns.csv`, `prepared/dataset_summary.json`

2. Acoustic feature structure and clustering
   - Script: `scripts/02_cluster_features.py`
   - Outputs: `clusters/cluster_assignments_<K>.csv`, `clusters/clustering_diagnostics.csv`

3. Baseline raw-feature models
   - Script: `scripts/03_train_baseline.py`
   - Outputs:
     - `baseline/fold_membership.csv`
     - `baseline/fold_metrics.csv`
     - `baseline/metric_summary.csv`
     - `baseline/baseline_results.csv`
     - `baseline/recording_predictions.csv`
     - `baseline/feature_importance.csv`
     - `baseline/rf_feature_importance.csv`
   - Notes:
     - participant-level CV artifacts are exported for notebook-style baseline visualizations
     - final report metrics use participant-level train/val/test split
     - supports mixed model families:
       - linear: logistic, elastic-net logistic
       - tree ensembles: random forest, extra trees, hist gradient boosting
       - kernel: RBF SVM
       - deep: PyTorch `mlp`, `residual_mlp`, `wide_deep_mlp`

4. Cluster-aggregated models across K
   - Script: `scripts/04_train_cluster_models.py`
   - Outputs:
     - Coarse sweep (tuning split): `cluster_models/coarse_results.csv`
     - Fine-tune sweep (tuning split): `cluster_models/fine_tune_results.csv`
     - Locked final held-out test: `cluster_models/final_test_results.csv`
     - Combined: `cluster_models/all_recording_results.csv`
     - Coarse contenders/ranking:
       - `cluster_models/coarse_k_ranked_selection_scope.csv`
       - `cluster_models/coarse_k_contenders_selection_scope.csv`
     - Best configuration metadata:
       - `cluster_models/best_fine_tune_config.json`
       - `cluster_models/k_selection.json`
       - `cluster_models/locked_selection.json`
     - Interpretability:
       - `cluster_models/cluster_importance_<K>clusters.csv`
       - `cluster_models/top_features_by_cluster_<K>clusters.csv`
       - `cluster_models/cluster_profiles_<K>clusters.csv`
     - Best-config predictions:
       - `cluster_models/best_fine_tune_recording_predictions.csv`
       - `cluster_models/final_test_recording_predictions.csv`
   - Notes:
     - explainable small-K contender logic is persisted in `k_selection.json`
     - confirmatory mode selects on validation and evaluates held-out test once after locking
     - selector-model policy supports fixed selector or auto-best selector with DL preference
     - lock strategy supports manual override (`selection.override_k` or `VOICE_SCREENING_OVERRIDE_K`)
     - default lock strategy balances stability + ROC/F1/Accuracy in the fine-tune K region

5. Patient-level comparison and best configs
   - Script: `scripts/05_patient_level_eval.py`
   - Outputs:
     - Selection split results: `reports/patient_level_selection_results.csv`
     - Locked final held-out test results: `reports/patient_level_results.csv`
     - `reports/patient_level_final_test_results.csv`
     - `reports/patient_level_predictions.csv`
     - `reports/patient_level_best_configurations.csv`
     - `reports/patient_level_best_overall.csv`
     - `reports/patient_level_locked_selection.json`
     - `reports/patient_level_run_summary.csv`

6. Figures/tables
   - Script: `scripts/06_generate_figures_tables.py`
   - Outputs:
     - Notebook-parity figure aliases in `reports/*.png`
     - PPT-ordered figure folders:
       - `reports/figures/01_acoustic_features/`
       - `reports/figures/02_feature_clustering_decision/`
       - `reports/figures/03_feature_clustering_insights/`
       - `reports/figures/04_progressive_feature_selection/`
       - `reports/figures/05_recording_level_performance/`
       - `reports/figures/06_patient_level_performance/`
     - Progressive feature selection:
       - `reports/progressive_selection_results.csv`
       - `reports/progressive_selection_summary.csv`
       - `reports/progressive_selection_performance.png`
     - Figure manifest:
       - `reports/figure_manifest.csv`
       - `reports/figure_manifest.json`
     - K-validation diagnostics:
       - `reports/cluster_validation_elbow_metrics.csv`
       - `reports/cluster_validation_stability_bootstrap.csv`
       - `reports/cluster_validation_silhouette_cophenetic.png`
       - `reports/cluster_validation_elbow_curve.png`
       - `reports/cluster_validation_stability_bootstrap_ari.png`
     - Tables:
       - `reports/table_coarse_results.csv`
       - `reports/table_fine_tune_results.csv`
       - `reports/table_patient_level_results.csv`
     - `reports/table_patient_level_best_configurations.csv`
     - `reports/table_coarse_k_contenders_selection_scope.csv`

7. Deterministic HTML run report
   - Script: `scripts/07_compile_html_report.py`
   - Outputs:
     - `reports/manuscript_report.html`
     - `reports/manuscript_report_metadata.json`
     - `run_manifest.json` (updated across all pipeline stages)
   - Notes:
     - compiles end-to-end run narrative with split-separated tuning vs held-out test metrics
     - includes artifact index and figure inventory for reproducible review

## Default phenotype
- Parkinson's phenotype config: `configs/phenotypes/parkinsons.yaml`

## Extending to new phenotypes
- Add a new YAML file under `configs/phenotypes/` with label mapping rules.
- Re-run the same seven scripts with the new phenotype config.
