"""Tests for score_round.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_score_round_r16_expects_six_of_eight():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from score_round import build_round_history

    forecast_path = PROJECT_ROOT / "output" / "wc2026_forecast80.json"
    if not forecast_path.exists():
        pytest.skip("forecast80 archive not present")

    payload = build_round_history(forecast_path=forecast_path)
    summary = payload["rounds"][0]["summary"]
    assert summary["correct"] == 6
    assert summary["total"] == 8

    wrong = [m for m in payload["rounds"][0]["results"] if not m["correct"]]
    wrong_pairs = {(m["home"], m["away"]) for m in wrong}
    assert ("Brazil", "Norway") in wrong_pairs
    assert ("Switzerland", "Colombia") in wrong_pairs
