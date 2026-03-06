# Reproduction Guide

## 1. Environment
```bash
python -m pip install -e .[dev]
```

Frontend UI dependencies:
```bash
python -m pip install -e .[ui]
```

## 2. Data placement
Expected files:
- `physionet.org/files/b2ai-voice/2.0.1/phenotype.tsv`
- `physionet.org/files/b2ai-voice/2.0.1/static_features.tsv`

PhysioNet direct download (full tree):
```bash
wget -r -N -c -np --user user_name --ask-password https://physionet.org/files/b2ai-voice/3.0.0/
```

Static-only download (recommended for this repository):
```bash
PHYSIONET_USER=user_name make download_data_static
```

If your shell or VPN injects a proxy and you see errors like `Connecting to 127.0.0.1:56801` or `Proxy tunneling failed`, bypass the proxy for PhysioNet:
```bash
PHYSIONET_USER=user_name PHYSIONET_NO_PROXY=1 make download_data_static
```
You can also set `NO_PROXY=physionet.org` if direct access is allowed on your network.

## 3. Run full pipeline (default Parkinson's)
```bash
make pipeline PHENOTYPE=parkinsons RUN_ID=run001
```

If you invoke the repository from another working directory and the repo path contains spaces, quote the `make -C` argument:
```bash
make -C "/path/with spaces/vocal_predictions" pipeline PHENOTYPE=parkinsons RUN_ID=run001
```
Unquoted paths are split by the shell before `make` sees them, which produces errors such as `make: /path/with: No such file or directory`.

PyTorch DL paths include:
- `mlp` (`RegularizedHybridNet`)
- `residual_mlp` (residual tabular MLP)
- `wide_deep_mlp` (wide + deep tabular net)

If your runtime cannot load PyTorch/OpenMP, run the same command in a standard local shell instead of a restricted sandbox runtime.

## 4. Stage-by-stage execution
```bash
make prepare  PHENOTYPE=parkinsons RUN_ID=run001
make cluster  PHENOTYPE=parkinsons RUN_ID=run001
make baseline PHENOTYPE=parkinsons RUN_ID=run001
make clustered PHENOTYPE=parkinsons RUN_ID=run001
make eval PHENOTYPE=parkinsons RUN_ID=run001
make report PHENOTYPE=parkinsons RUN_ID=run001
make compile_html PHENOTYPE=parkinsons RUN_ID=run001
```

Manual K override for any run:
```bash
VOICE_SCREENING_OVERRIDE_K=16 make pipeline PHENOTYPE=parkinsons RUN_ID=run001
```

## 4. Frontend run
```bash
make frontend
```

Frontend constraints:
- first-time session asks for PhysioNet credentials,
- credentials are not persisted to disk by the UI,
- user-facing frontend output is only the compiled HTML report,
- all other outputs remain in `outputs/<phenotype>/<run_id>/`.

Compiled HTML output:
- `outputs/<phenotype>/<run_id>/reports/manuscript_report.html`
- `outputs/<phenotype>/<run_id>/reports/manuscript_report_metadata.json`

## 5. Leakage prevention
- Patient-level folds are created from unique participant IDs.
- Participants in train/test folds are disjoint for every split.
- Cluster assignments for clustered-model CV are learned per fold on train-only data.
- Preprocessing (imputation/scaling) is fit on train-only and applied to test.
- Each outer fold uses an inner participant-level validation split for notebook-aligned threshold tuning.
- `analysis.mode=confirmatory` (default) enforces:
  - clustered coarse/fine tuning on validation split only
  - patient aggregation selection on validation split only
  - locked one-shot held-out test evaluation after selection
- `analysis.mode=exploratory` is available for hypothesis exploration and is not for confirmatory claims.
- Clustered-model stage performs coarse K exploration, then fine-tunes around a contender region.
- Reporting exports contender-K tables from coarse search in preferred small-K ranges.
- Patient-level aggregation search is stored in `patient_level_selection_results.csv`; final held-out results are in `patient_level_results.csv`.
- Reporting emits PPT-ordered figure folders and a `figure_manifest.csv` for deterministic slide assembly.
- Reporting emits cluster K-validation diagnostics (silhouette, elbow proxy, bootstrap stability).
- HTML compiler emits a deterministic run-level summary report (`manuscript_report.html`) using generated figures/tables.
- End-to-end run provenance is tracked in `run_manifest.json` (config hash, environment, dependency versions, executed stages).

## 6. Non-synthetic policy
This repository intentionally excludes synthetic-data generation and synthetic-data training experiments.
