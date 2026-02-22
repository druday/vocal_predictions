# Voice-Based Phenotype Screening

This repository provides a reproducible voice-based screening workflow using Bridge2AI acoustic features. By default it runs Parkinson's disease screening, but the pipeline is phenotype-configurable so the same process can be reused for other phenotypes without hardcoded results.

The repository includes both a command-line pipeline and a beginner-friendly frontend. The frontend is designed for first-time users: it can download required PhysioNet static files, run the pipeline, and display the compiled HTML manuscript-style report. If no run has been generated yet, a bundled default report is shown so users can preview output immediately.

## Beginner quick start

1. Clone the repository (or download ZIP) and open Terminal in this folder.
2. Create and activate a Python environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install frontend dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -e ".[ui]"
```
4. Launch the frontend:
```bash
make frontend
```
5. Open `http://localhost:8501`.
6. Enter PhysioNet username/password on first launch (not stored to disk).
7. Keep `Download static files before running pipeline` checked and click `Run Pipeline`.

## What users will see

- Main user output: compiled report (`manuscript_report.html`) in the frontend.
- If a new run fails, the app can still show the bundled default report.
- Full artifacts are saved in `outputs/<phenotype>/<run_id>/`.

## Advanced documentation

For full pipeline stages, model details, configuration options, and reproducibility notes, use:

- `ReadMe_Advanced.md`
