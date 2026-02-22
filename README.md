# Voice-Based Phenotype Screening (Reproducible Repository)

This repository provides a reproducible, **voice-only** machine learning pipeline for phenotype screening from Bridge2AI Voice acoustic features.

Default phenotype is **Parkinson's disease**, but the pipeline is config-driven so it can be replicated for other phenotypes by adding new phenotype YAML configs.

## Scope
- No synthetic data generation or synthetic data training workflows are included.
- Methods are frozen in this repository implementation (no notebook re-alignment required).
- Two explicit run modes are supported:
  - `confirmatory` (default): tune/select on validation, then one-shot held-out test.
  - `exploratory`: allows notebook-style exploration on test for hypothesis generation.
- Supports baseline (raw acoustic features) and clustered feature-space models.
- Supports patient-level aggregation strategy search (`mean`, `median`, `max`, `min`, `std`) with split-aware locking.
- Uses fixed preprocessing/scaling/clipping and decision threshold (`0.5`) unless changed in config.

## Repository layout
- `configs/`: base and phenotype-specific YAML configuration
- `scripts/`: stage scripts for data prep, clustering, training, evaluation, reporting
- `src/voice_screening/`: reusable pipeline modules
- `tests/`: core reproducibility and leakage checks
- `docs/`: reproduction and manuscript traceability docs
  - `docs/frontend_plan.md`: phased plan for a simple repository UI

## Data expectation
Place Bridge2AI Voice files under:
- `physionet.org/files/b2ai-voice/2.0.1/phenotype.tsv`
- `physionet.org/files/b2ai-voice/2.0.1/static_features.tsv`

You can override data paths in config.

### Download from PhysioNet (recommended)
Full dataset (as provided by PhysioNet):
```bash
wget -r -N -c -np --user user_name --ask-password https://physionet.org/files/b2ai-voice/3.0.0/
```

Static-only download (skip MFCC/spectrogram-heavy assets):
```bash
PHYSIONET_USER=user_name make download_data_static
```

Equivalent script call:
```bash
PHYSIONET_USER=user_name scripts/download_physionet_b2ai.sh --mode static
```

## Quickstart
1. Install environment
```bash
python -m pip install -e .[dev]
```

For frontend usage:
```bash
python -m pip install -e .[ui]
```

2. Run full pipeline (default Parkinson's)
```bash
make pipeline PHENOTYPE=parkinsons RUN_ID=run001
```

PyTorch DL methods include:
- `mlp` (`RegularizedHybridNet`)
- `residual_mlp` (residual tabular MLP)
- `wide_deep_mlp` (wide + deep tabular net)

If your execution environment blocks PyTorch/OpenMP initialization, run the pipeline in a standard local shell (not a restricted sandbox runtime).

3. Outputs
- `outputs/<phenotype>/<run_id>/prepared/`
- `outputs/<phenotype>/<run_id>/clusters/`
- `outputs/<phenotype>/<run_id>/baseline/`
- `outputs/<phenotype>/<run_id>/cluster_models/`
  - `cluster_models/coarse_results.csv` (tuning split metrics)
  - `cluster_models/fine_tune_results.csv` (tuning split metrics)
  - `cluster_models/final_test_results.csv` (locked final held-out test metrics)
  - `cluster_models/locked_selection.json`
- `outputs/<phenotype>/<run_id>/reports/`
  - `reports/patient_level_selection_results.csv` (tuning split)
  - `reports/patient_level_results.csv` (locked final held-out test)
  - `reports/patient_level_locked_selection.json`
  - `reports/manuscript_report.html`
  - `reports/manuscript_report_metadata.json`
  - `reports/figures/01_acoustic_features/`
  - `reports/figures/02_feature_clustering_decision/`
  - `reports/figures/03_feature_clustering_insights/`
  - `reports/figures/04_progressive_feature_selection/`
  - `reports/figures/05_recording_level_performance/`
  - `reports/figures/06_patient_level_performance/`
  - `reports/figure_manifest.csv`

## Simple frontend
Launch:
```bash
make frontend
```

Frontend behavior:
- First-time onboarding asks for PhysioNet username/password.
- Credentials are used only for the active session and are not stored to disk.
- If users do not have a credentialed PhysioNet account, they are instructed to contact the author.
- Frontend displays only `manuscript_report.html` as user-facing output.
- All other pipeline deliverables are saved in the repository `outputs/` folder.

## Add a new phenotype
1. Copy `configs/phenotypes/parkinsons.yaml` to a new file.
2. Update label mapping rules.
3. Run pipeline with `PHENOTYPE=<new_name>` and that phenotype config.

## Manuscript-aligned stages
1. Dataset preparation and case/control labeling
2. Correlation-based acoustic feature clustering (`1 - |r|`, Ward linkage)
3. Baseline model training/evaluation on raw features
4. Cluster-aggregated model tuning and locking using:
   - coarse K exploration (broad sweep)
   - fine-tune K sweep around an explainable high-performance region (small-K near-top objective)
   - split-aware selection (`val` in confirmatory mode)
   - locked one-shot held-out test evaluation after selection
   - models:
     - logistic regression
     - elastic-net logistic
     - random forest
     - extra trees
     - hist gradient boosting
     - RBF SVM
     - PyTorch `RegularizedHybridNet` (MLP)
     - PyTorch residual MLP
     - PyTorch wide+deep MLP
   - selector model policy:
     - coarse best model can be chosen automatically with DL preference
     - locked K from selected model is evaluated across all configured models
   - lock strategies include:
     - `balanced_stability_performance` (default): finds stable Ks, enforces near-top ROC/F1/Accuracy gates, then scores candidates with ROC + stability emphasis and small-K preference
     - `stability_only`
     - `performance_only`
     - `performance_plus_validity`
   - manual K override:
     - set `cluster.fine_tune.selection.override_k` in config, or
     - set env var `VOICE_SCREENING_OVERRIDE_K=<K>` when running pipeline
5. Patient-level model tuning and locked final evaluation
   - uses best fine-tuned cluster configuration
   - tunes patient aggregation on selection split
   - evaluates locked aggregation/model choices once on held-out test
6. Figure/table generation for manuscript reporting
   - includes notebook-parity figure families:
     - acoustic feature EDA panels
     - coarse/fine cluster-decision plots
     - K-validation diagnostics (silhouette, elbow proxy, bootstrap stability)
     - dendrogram/PCA/importance interpretability plots
     - progressive feature-selection performance
     - recording-level ROC/PR/confusion/CV/model-comparison visuals
     - patient-level aggregation and comparison visuals
7. Deterministic HTML report compilation
   - compiles all stage outputs into one artifact:
     - `reports/manuscript_report.html`
   - includes concise run summaries, split-aware metric tables, key figures, artifact index, and `run_manifest.json` traceability
