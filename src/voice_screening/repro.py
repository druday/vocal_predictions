from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DEPENDENCIES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "pyyaml",
    "matplotlib",
    "seaborn",
    "pyarrow",
]


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dependency_versions(package_names: list[str] | None = None) -> dict[str, str]:
    names = package_names or DEFAULT_DEPENDENCIES
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    root = Path(cwd or Path.cwd())
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {
            "available": True,
            "branch": branch,
            "commit_short": head,
            "dirty": bool(status),
        }
    except Exception:
        return {"available": False}


def update_run_manifest(
    *,
    path: str | Path,
    config: dict[str, Any],
    stage: str,
    run_id: str,
    phenotype: str,
    extra: dict[str, Any] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            existing = loaded

    now = datetime.now(timezone.utc).isoformat()
    stages = existing.get("stages", [])
    if not isinstance(stages, list):
        stages = []

    entry: dict[str, Any] = {"stage": stage, "completed_at_utc": now}
    if extra:
        entry["extra"] = extra
    stages.append(entry)

    manifest: dict[str, Any] = {
        "created_at_utc": existing.get("created_at_utc", now),
        "updated_at_utc": now,
        "run_id": run_id,
        "phenotype": phenotype,
        "analysis_mode": config.get("analysis", {}).get("mode", "confirmatory"),
        "random_seed": int(config.get("project", {}).get("random_seed", 42)),
        "config_sha256": config_sha256(config),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": dependency_versions(),
        "git": git_state(Path(cwd) if cwd is not None else Path.cwd()),
        "stages": stages,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest

