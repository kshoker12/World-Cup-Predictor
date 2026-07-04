"""Tests for flat vs nested Kaggle data path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldcup_predictor.kaggle_paths import DataLayout, has_international_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_nested_repo_layout(default_config):
    layout = DataLayout.resolve(PROJECT_ROOT)
    assert layout.results_csv().exists()
    assert layout.wc2026_results_csv().exists()
    assert layout.former_names_csv().exists()
    assert layout.understat_parquet() is not None
    assert layout.club_dir() is not None
    assert layout.has_club_data()


def test_flat_kaggle_layout(tmp_path: Path):
    flat = tmp_path / "soccer-data"
    flat.mkdir()
    for name in (
        "results.csv",
        "wc2026_results.csv",
        "former_names.csv",
        "fixtures.csv",
        "match_stats.csv",
        "understat_matches.parquet",
    ):
        src = PROJECT_ROOT / "data"
        if name in ("fixtures.csv", "match_stats.csv"):
            src = src / "raw" / "club" / name
        elif name == "understat_matches.parquet":
            src = src / "processed" / name
        else:
            src = src / "raw" / name
        if not src.exists():
            pytest.skip(f"Missing fixture source {src}")
        (flat / name).write_bytes(src.read_bytes())

    layout = DataLayout.resolve(PROJECT_ROOT, flat)
    assert has_international_data(flat)
    assert layout.results_csv() == flat / "results.csv"
    assert layout.wc2026_results_csv() == flat / "wc2026_results.csv"
    assert layout.former_names_csv() == flat / "former_names.csv"
    assert layout.understat_parquet() == flat / "understat_matches.parquet"
    assert layout.club_dir() == flat
    assert layout.has_club_data()
