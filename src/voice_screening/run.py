from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunDirs:
    root: Path
    prepared: Path
    clusters: Path
    baseline: Path
    cluster_models: Path
    reports: Path


def build_run_dirs(outputs_root: str | Path, phenotype_name: str, run_id: str) -> RunDirs:
    root = Path(outputs_root) / phenotype_name / run_id
    prepared = root / "prepared"
    clusters = root / "clusters"
    baseline = root / "baseline"
    cluster_models = root / "cluster_models"
    reports = root / "reports"

    for path in (root, prepared, clusters, baseline, cluster_models, reports):
        path.mkdir(parents=True, exist_ok=True)

    return RunDirs(
        root=root,
        prepared=prepared,
        clusters=clusters,
        baseline=baseline,
        cluster_models=cluster_models,
        reports=reports,
    )
