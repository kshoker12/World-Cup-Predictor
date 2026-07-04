#!/usr/bin/env python3
"""Robust 2018 World Cup calibration backtest using production models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.simulation.calibration_backtest import main_for_year  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_for_year(2018))
