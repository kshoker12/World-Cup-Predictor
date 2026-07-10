"""Tests for WC 2026 forecast-only pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from worldcup_predictor.kaggle_paths import (
    REQUIRED_MODEL_FILES,
    stage_models,
    verify_model_artifacts,
)
from worldcup_predictor.simulation.wc2026_forecast import run_wc2026_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
FIXTURE_DATA = PROJECT_ROOT / "tests" / "fixtures" / "flat_soccer_data"


def _copy_models(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_MODEL_FILES:
        shutil.copy2(src / name, dest / name)


@pytest.mark.skipif(not OUTPUT_DIR.exists(), reason="output/ models not present")
def test_verify_model_artifacts_output_dir():
    missing = verify_model_artifacts(OUTPUT_DIR)
    assert missing == []


@pytest.mark.skipif(not OUTPUT_DIR.exists(), reason="output/ models not present")
def test_stage_models(tmp_path: Path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _copy_models(OUTPUT_DIR, source)
    stage_models(source, dest)
    assert verify_model_artifacts(dest) == []
    for name in REQUIRED_MODEL_FILES:
        assert (dest / name).exists()


@pytest.mark.skipif(
    not OUTPUT_DIR.exists() or not FIXTURE_DATA.exists(),
    reason="output/ or fixture data missing",
)
def test_run_wc2026_forecast_smoke(tmp_path: Path):
    models = tmp_path / "models"
    _copy_models(OUTPUT_DIR, models)
    bundle, _ = run_wc2026_forecast(
        project_root=PROJECT_ROOT,
        models_dir=models,
        profile="fast",
        n_sims=3,
        data_root=FIXTURE_DATA,
        show_progress=False,
        print_eta=False,
    )
    forecast = bundle["forecast"]
    assert isinstance(forecast, dict)
    assert forecast["n_sims"] == 3
    assert "champion_probs" in forecast
    assert "match_win_probs" in forecast
    probs = forecast["champion_probs"]
    assert isinstance(probs, dict)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
