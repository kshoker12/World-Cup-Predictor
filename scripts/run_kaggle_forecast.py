#!/usr/bin/env python3
"""Forecast-only WC 2026 knockout simulation using pre-trained models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.kaggle_paths import resolve_work_models_dir  # noqa: E402
from worldcup_predictor.simulation.wc2026_forecast import (  # noqa: E402
    configure_forecast_runtime,
    run_wc2026_forecast,
    write_forecast_outputs,
)

configure_forecast_runtime()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WC 2026 forecast-only Monte Carlo (no training)"
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "kaggle", "backtest"],
        default="kaggle",
        help="Config profile (kaggle default uses 200k sims)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="Simulation count (default: profile value, kaggle=200000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Forecast JSON path (default: models_dir/wc2026_forecast.json)",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-eta", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir or resolve_work_models_dir(PROJECT_ROOT)
    forecast_path = args.output or models_dir / "wc2026_forecast.json"
    report_path = models_dir / "forecast_report.json"

    try:
        bundle, seconds_per_sim = run_wc2026_forecast(
            project_root=PROJECT_ROOT,
            models_dir=models_dir,
            profile=args.profile,
            n_sims=args.sims,
            seed=args.seed,
            data_root=args.data_root,
            show_progress=not args.no_progress,
            print_eta=not args.no_eta,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_forecast_outputs(
        bundle,
        forecast_path=forecast_path,
        report_path=report_path,
    )

    forecast = bundle["forecast"]
    assert isinstance(forecast, dict)
    report = bundle["report"]
    assert isinstance(report, dict)

    print(f"\nForecast complete: {forecast_path}")
    print(f"Report: {report_path}")
    print(f"Simulations: {forecast.get('n_sims')}")
    if seconds_per_sim is not None:
        print(f"Seconds per sim: {seconds_per_sim:.4f}")
    print(f"Elapsed: {report.get('elapsed_seconds')}s")
    print("\nTop 10 champion probabilities:")
    champion_probs = forecast.get("champion_probs", {})
    assert isinstance(champion_probs, dict)
    for team, prob in sorted(
        champion_probs.items(), key=lambda item: item[1], reverse=True
    )[:10]:
        print(f"  {team}: {prob:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
