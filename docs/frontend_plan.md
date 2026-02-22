# Frontend Plan (Simple, Reproducible)

## Goal
Provide a lightweight UI for non-technical users to:
- run the phenotype pipeline,
- inspect K-selection rationale,
- view recording/patient performance,
- browse figures and cluster descriptions,
- download/export run artifacts.

## Recommended stack
- **Backend/UI framework**: Streamlit (single-process Python app, low ops overhead)
- **Data source**: existing `outputs/<phenotype>/<run_id>/...` artifacts
- **Execution**: subprocess calls to `make ...` targets

## MVP scope
0. First-time onboarding panel
- PhysioNet username/password entry
- explicit statement: credentials are not stored to disk by the UI
- note to contact the author if no credentialed PhysioNet account
- local-data-only bypass option

1. Run setup panel
- phenotype selector
- run ID input (auto-suggest default)
- static-data download toggle (PhysioNet)
- run button for full pipeline

2. K-selection panel
- coarse contenders table
- fine-tune metrics + stability table
- selected K explanation card:
  - lock strategy
  - gate source
  - selected score components
- optional override toggle and rerun shortcut

3. Performance panel
- recording-level final test table and model ranking
- patient-level final test table and best aggregation
- side-by-side metric comparison chart

4. Cluster interpretation panel
- cluster profile table (`cluster_profiles_best.csv`)
- per-cluster feature examples and dominant category
- links to cluster importance and top-feature CSVs

5. Figures panel
- gallery grouped by report stage
- quick links to `manuscript_report.html` and `figure_manifest.csv`

6. User-visible output policy
- frontend renders only `manuscript_report.html` as user-facing output
- all other deliverables remain in repository `outputs/` directories

## Architecture
- `app/streamlit_app.py`: entrypoint
- `app/services/runs.py`: artifact loading, run discovery
- `app/services/pipeline.py`: controlled command execution and logs
- `app/components/*.py`: reusable panels/tables

## Guardrails
- show explicit warning when run is in `exploratory` mode
- block conflicting concurrent runs on same `run_id`
- validate K override against available fine-tune K values
- persist launch config with each run for traceability

## Phase plan
1. Phase 1 (MVP): run trigger + key tables + report link
2. Phase 2: figure gallery + cluster interpretation panel
3. Phase 3: run history dashboard + compare two runs
