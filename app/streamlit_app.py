#!/usr/bin/env python3
from __future__ import annotations

import base64
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "raw_data"
PHENO_DIR = ROOT / "configs" / "phenotypes"
DEFAULT_REPORT_FILE = ROOT / "app" / "default_report" / "manuscript_report.html"
DEFAULT_BASE_URL = "https://physionet.org/files/b2ai-voice/3.0.0/"
PIPELINE_STAGE_HINTS: list[tuple[str, str, int]] = [
    ("scripts/01_prepare_dataset.py", "Stage 1/7: preparing dataset", 8),
    ("scripts/02_cluster_features.py", "Stage 2/7: clustering features", 20),
    ("scripts/03_train_baseline.py", "Stage 3/7: training baseline models", 40),
    ("scripts/04_train_cluster_models.py", "Stage 4/7: clustered-model search + lock", 64),
    ("scripts/05_patient_level_eval.py", "Stage 5/7: patient-level evaluation", 78),
    ("scripts/06_generate_figures_tables.py", "Stage 6/7: generating figures/tables", 90),
    ("scripts/07_compile_html_report.py", "Stage 7/7: compiling HTML report", 97),
    ("Pipeline completed", "Completed", 100),
]


def _available_phenotypes() -> list[str]:
    if not PHENO_DIR.exists():
        return ["parkinsons"]
    vals = sorted([p.stem for p in PHENO_DIR.glob("*.yaml") if p.is_file()])
    return vals or ["parkinsons"]


def _default_run_id() -> str:
    return "run" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _report_path(phenotype: str, run_id: str) -> Path:
    return ROOT / "outputs" / phenotype / run_id / "reports" / "manuscript_report.html"


def _discover_reports(phenotype: str) -> list[dict[str, str]]:
    root = ROOT / "outputs" / phenotype
    if not root.exists():
        return []

    rows: list[dict[str, str]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        report = run_dir / "reports" / "manuscript_report.html"
        if not report.exists():
            continue
        mtime_val = report.stat().st_mtime
        mtime = datetime.fromtimestamp(mtime_val).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "run_id": run_dir.name,
                "path": str(report),
                "label": f"{run_dir.name} ({mtime})",
                "mtime_value": str(mtime_val),
            }
        )

    rows.sort(key=lambda r: float(r["mtime_value"]), reverse=True)
    return rows


def _report_is_for_phenotype(report_file: Path, phenotype: str) -> bool:
    pheno_root = (ROOT / "outputs" / phenotype).resolve()
    try:
        report_file.resolve().relative_to(pheno_root)
    except ValueError:
        return False
    return True


def _has_static_source_table() -> bool:
    if (RAW_DATA_DIR / "static_features.tsv").exists():
        return True
    for path in RAW_DATA_DIR.rglob("*.tsv"):
        name = path.name.lower()
        if "static" in name and "feature" in name:
            return True
    return False


def _has_phenotype_source_table() -> bool:
    if (RAW_DATA_DIR / "phenotype.tsv").exists():
        return True
    for path in RAW_DATA_DIR.rglob("*.tsv"):
        parts = [part.lower() for part in path.parts]
        if "phenotype" in parts:
            return True
    return False


def _missing_required_raw_files() -> list[str]:
    missing: list[str] = []
    if not _has_phenotype_source_table():
        missing.append("phenotype TSV(s) under raw_data/**/phenotype/")
    if not _has_static_source_table():
        missing.append("static_features TSV")
    return missing


def _sanitize_logs(log_text: str, secrets: Iterable[str]) -> str:
    cleaned = str(log_text)
    for secret in secrets:
        token = str(secret or "").strip()
        if token:
            cleaned = cleaned.replace(token, "***")
    return cleaned


def _run_command(
    cmd: list[str],
    env: dict[str, str] | None = None,
    *,
    status_box=None,
    progress_bar=None,
    stage_hints: list[tuple[str, str, int]] | None = None,
    secrets: Iterable[str] = (),
    start_message: str = "Running command",
) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    progress_val = 0

    if status_box is not None:
        status_box.info(start_message)
    if progress_bar is not None:
        progress_bar.progress(progress_val, text=start_message)

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        lines.append(line)

        if stage_hints:
            for pattern, msg, pct in stage_hints:
                if pattern in line and pct >= progress_val:
                    progress_val = pct
                    if status_box is not None:
                        status_box.info(msg)
                    if progress_bar is not None:
                        progress_bar.progress(progress_val, text=msg)
                    break

    rc = proc.wait()
    safe_logs = _sanitize_logs("\n".join(lines), secrets)
    return rc, safe_logs


def _normalize_static_files() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_candidates: dict[str, list[re.Pattern[str]]] = {
        "static_features.tsv": [
            re.compile(r"^static_features\.tsv$", re.IGNORECASE),
            re.compile(r"static.*features?.*\.tsv$", re.IGNORECASE),
        ],
    }

    all_tsv = [p for p in RAW_DATA_DIR.rglob("*.tsv") if p.is_file()]

    # Only normalize phenotype file when exact filename is available.
    # Nested PhysioNet 3.0 phenotype tables are intentionally left in place;
    # the data loader resolves them directly.
    phenotype_exact = [p for p in all_tsv if p.name.lower() == "phenotype.tsv"]
    if phenotype_exact:
        phenotype_exact.sort(key=lambda p: (len(p.parts), len(str(p))))
        src = phenotype_exact[0]
        dst = RAW_DATA_DIR / "phenotype.tsv"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    for name, patterns in target_candidates.items():
        exact = [p for p in all_tsv if p.name == name]
        matches = exact
        if not matches:
            for pat in patterns:
                matches = [p for p in all_tsv if pat.search(p.name)]
                if matches:
                    break
        if not matches:
            continue
        matches.sort(key=lambda p: (len(p.parts), len(str(p))))
        src = matches[0]
        dst = RAW_DATA_DIR / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def _download_static_files(
    *,
    user: str,
    password: str,
    base_url: str,
    status_box=None,
    progress_bar=None,
) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PHYSIONET_USER"] = user
    env["PHYSIONET_PASSWORD"] = password
    rc, logs = _run_command(
        [
            str(ROOT / "scripts" / "download_physionet_b2ai.sh"),
            "--mode",
            "static",
            "--base-url",
            base_url,
            "--dest",
            str(RAW_DATA_DIR),
        ],
        env=env,
        status_box=status_box,
        progress_bar=progress_bar,
        stage_hints=[
            ("Downloading static-only files", "Downloading static files from PhysioNet", 3),
        ],
        secrets=[password],
        start_message="Preparing static-file download",
    )
    if rc != 0:
        return False, logs
    _normalize_static_files()
    missing = _missing_required_raw_files()
    if missing:
        found_tsv = sorted(str(p) for p in RAW_DATA_DIR.rglob("*.tsv") if p.is_file())
        debug = (
            "\nExpected required files were not found after download:\n"
            + "\n".join(missing)
            + "\n\nTSV files discovered under raw_data:\n"
            + ("\n".join(found_tsv) if found_tsv else "(none)")
        )
        return False, logs + debug
    return True, logs


def _inline_report_images(report_file: Path) -> str:
    html_text = report_file.read_text(encoding="utf-8")
    report_dir = report_file.parent
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".gif": "image/gif",
    }

    def repl(match: re.Match[str]) -> str:
        src = match.group(1).strip()
        if src.startswith(("http://", "https://", "data:", "file://", "mailto:")):
            return match.group(0)
        asset_path = (report_dir / src).resolve()
        if not asset_path.exists():
            return match.group(0)
        mime = mime_map.get(asset_path.suffix.lower())
        if mime is None:
            return match.group(0)
        b64 = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{b64}"'

    return re.sub(r'src="([^"]+)"', repl, html_text)


def _render_onboarding() -> None:
    st.title("First-Time Setup")
    st.write(
        "Enter your PhysioNet credentials for downloading Bridge2AI Voice static files. "
        "Credentials are only kept in memory for this active app session."
    )
    st.warning("Your PhysioNet username and password will not be stored to disk by this frontend.")
    st.info("If you do not have a credentialed PhysioNet account, please contact the author.")

    with st.form("physionet_onboarding"):
        user = st.text_input("PhysioNet Username", value=st.session_state.get("physionet_user", ""))
        password = st.text_input("PhysioNet Password", type="password")
        local_only = st.checkbox(
            "Skip download now (I already placed phenotype/static TSV files in raw_data)",
            value=False,
        )
        submitted = st.form_submit_button("Continue")
        if submitted:
            if not local_only and (not user.strip() or not password):
                st.error("Enter username/password or enable local-data-only mode.")
                return
            st.session_state["physionet_user"] = user.strip()
            st.session_state["physionet_password"] = password
            st.session_state["local_data_only"] = bool(local_only)
            st.session_state["download_base_url"] = DEFAULT_BASE_URL
            if not local_only:
                with st.spinner("Downloading static files from PhysioNet..."):
                    ok, logs = _download_static_files(
                        user=user.strip(),
                        password=password,
                        base_url=DEFAULT_BASE_URL,
                    )
                if not ok:
                    st.error("Automatic static-file download failed during setup.")
                    with st.expander("Download logs"):
                        st.code(logs)
                    return
                st.success("Static files downloaded successfully.")
            st.session_state["onboarded"] = True
            st.rerun()


def _render_existing_report_selector(
    phenotype: str,
    reports: list[dict[str, str]],
    has_bundled_default: bool,
) -> None:
    st.subheader("Open Existing Report")
    if not reports:
        if has_bundled_default:
            st.caption(
                "No existing compiled reports found for this phenotype yet. "
                "Showing the bundled default report below."
            )
        else:
            st.caption("No existing compiled reports found for this phenotype yet.")
        return

    labels = [r["label"] for r in reports]
    selected_index = 0
    current_report_path = str(st.session_state.get("last_report_file", ""))
    current_run_id = str(st.session_state.get("last_run_id", ""))
    for idx, row in enumerate(reports):
        if row["path"] == current_report_path or row["run_id"] == current_run_id:
            selected_index = idx
            break

    selected_label = st.selectbox(
        "Available runs",
        options=labels,
        index=selected_index,
        key=f"existing_report_{phenotype}",
    )
    if st.button("Open Selected Report"):
        selected = next(r for r in reports if r["label"] == selected_label)
        st.session_state["last_report_file"] = selected["path"]
        st.session_state["last_run_id"] = selected["run_id"]
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Voice Screening Frontend",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not st.session_state.get("onboarded", False):
        _render_onboarding()
        return

    st.title("Voice Screening Frontend")
    st.caption("Frontend output is limited to the compiled HTML report.")
    st.caption("All other deliverables are saved under this repository's outputs folder.")

    phenotypes = _available_phenotypes()
    default_index = phenotypes.index("parkinsons") if "parkinsons" in phenotypes else 0

    col1, col2 = st.columns(2)
    with col1:
        phenotype = st.selectbox("Phenotype", options=phenotypes, index=default_index)
        run_id = st.text_input("Run ID", value=st.session_state.get("last_run_id", _default_run_id()))
    with col2:
        base_url = st.text_input(
            "PhysioNet Base URL",
            value=str(st.session_state.get("download_base_url", DEFAULT_BASE_URL)),
        )
        download_before_run = st.checkbox(
            "Download static files before running pipeline",
            value=not st.session_state.get("local_data_only", False),
        )

    reports = _discover_reports(phenotype)
    current_report_raw = st.session_state.get("last_report_file")
    current_report = Path(current_report_raw) if current_report_raw else None
    needs_default_report = (
        current_report is None
        or not current_report.exists()
        or not _report_is_for_phenotype(current_report, phenotype)
    )
    if needs_default_report:
        if reports:
            st.session_state["last_report_file"] = reports[0]["path"]
            st.session_state["last_run_id"] = reports[0]["run_id"]
        elif DEFAULT_REPORT_FILE.exists():
            st.session_state["last_report_file"] = str(DEFAULT_REPORT_FILE)
            st.session_state["last_run_id"] = "bundled-default"

    _render_existing_report_selector(phenotype, reports, DEFAULT_REPORT_FILE.exists())

    st.subheader("Run New Pipeline")
    status_box = st.empty()
    progress_bar = st.progress(0, text="Idle")

    if st.button("Run Pipeline", type="primary"):
        st.session_state["last_run_id"] = run_id

        if download_before_run:
            user = str(st.session_state.get("physionet_user", "")).strip()
            password = str(st.session_state.get("physionet_password", ""))
            if not user or not password:
                st.error("Credentials are missing. Restart session and complete first-time setup.")
                st.stop()

            st.session_state["download_base_url"] = base_url
            ok, logs = _download_static_files(
                user=user,
                password=password,
                base_url=base_url,
                status_box=status_box,
                progress_bar=progress_bar,
            )
            if not ok:
                st.error("Static-file download failed.")
                status_box.error("Static-file download failed.")
                with st.expander("Download logs"):
                    st.code(logs)
                st.stop()
            status_box.info("Static files ready.")
            progress_bar.progress(5, text="Static files ready")

        missing_required = _missing_required_raw_files()
        if missing_required:
            st.error("Required raw files are missing. Pipeline cannot start.")
            with st.expander("How to fix"):
                st.markdown(
                    "1. Enable **Download static files before running pipeline** and run again.\n"
                    "2. Or place these files manually in `raw_data/`."
                )
                st.markdown("\n".join(f"- `{p}`" for p in missing_required))
            st.stop()

        rc, logs = _run_command(
            [
                "make",
                "pipeline",
                f"PHENOTYPE={phenotype}",
                f"RUN_ID={run_id}",
                f"PYTHON={sys.executable}",
            ],
            status_box=status_box,
            progress_bar=progress_bar,
            stage_hints=PIPELINE_STAGE_HINTS,
            secrets=[st.session_state.get("physionet_password", "")],
            start_message="Pipeline started",
        )
        if rc != 0:
            st.error("Pipeline failed.")
            status_box.error("Pipeline failed.")
            with st.expander("Pipeline logs"):
                st.code(logs)
            st.stop()

        report_file = _report_path(phenotype, run_id)
        if not report_file.exists():
            st.error(f"Pipeline completed but report was not found: {report_file}")
            status_box.error("Pipeline completed, but HTML report was not found.")
            st.stop()

        st.session_state["last_report_file"] = str(report_file)
        progress_bar.progress(100, text="Completed")
        status_box.success("Pipeline completed. Loading HTML report...")

    report_file_raw = st.session_state.get("last_report_file")
    if report_file_raw:
        report_file = Path(report_file_raw)
        if report_file.exists():
            if report_file.resolve() == DEFAULT_REPORT_FILE.resolve():
                st.info(
                    "Showing bundled default report from the repository. "
                    "Run the pipeline to generate a new run-specific report."
                )
            st.success(f"Report ready: {report_file}")
            html_report = _inline_report_images(report_file)
            components.html(html_report, height=1400, scrolling=True)


if __name__ == "__main__":
    main()
