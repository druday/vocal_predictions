from pathlib import Path

from voice_screening.config import load_config


def test_load_config_merges() -> None:
    cfg = load_config(
        Path("configs/base.yaml"),
        Path("configs/phenotypes/parkinsons.yaml"),
    )
    assert cfg["phenotype"]["name"] == "parkinsons"
    assert "paths" in cfg
    assert "modeling" in cfg
    assert cfg["analysis"]["mode"] == "confirmatory"
