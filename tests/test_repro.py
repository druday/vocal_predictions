from pathlib import Path

from voice_screening.repro import update_run_manifest


def test_update_run_manifest_appends_stages(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    cfg = {
        "project": {"random_seed": 42},
        "analysis": {"mode": "confirmatory"},
    }

    update_run_manifest(
        path=manifest_path,
        config=cfg,
        stage="stage_a",
        run_id="run001",
        phenotype="parkinsons",
    )
    out = update_run_manifest(
        path=manifest_path,
        config=cfg,
        stage="stage_b",
        run_id="run001",
        phenotype="parkinsons",
    )

    assert manifest_path.exists()
    assert out["run_id"] == "run001"
    assert out["phenotype"] == "parkinsons"
    assert out["analysis_mode"] == "confirmatory"
    assert len(out["stages"]) == 2
    assert out["stages"][0]["stage"] == "stage_a"
    assert out["stages"][1]["stage"] == "stage_b"

