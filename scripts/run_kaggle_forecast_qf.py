#!/usr/bin/env python3
"""Quarter-finals WC 2026 forecast using pre-trained models (post-R16 results)."""

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
    run_knockout_forecast,
    write_forecast_outputs,
)

configure_forecast_runtime()

QF_TOURNAMENT = (
    PROJECT_ROOT / "config" / "tournaments" / "world_cup_2026_quarterfinals.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WC 2026 quarter-finals forecast (knockout from QF, post-R16 results)"
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "kaggle", "backtest"],
        default="kaggle",
    )
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: models_dir/wc2026_forecast_qf.json",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-eta", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir or resolve_work_models_dir(PROJECT_ROOT)
    forecast_path = args.output or models_dir / "wc2026_forecast_qf.json"
    report_path = models_dir / "forecast_qf_report.json"

    from worldcup_predictor.kaggle_paths import DataLayout

    data_layout = DataLayout.resolve(PROJECT_ROOT, args.data_root)
    extra = [data_layout.wc2026_results_csv()]

    try:
        _, payload, seconds_per_sim = run_knockout_forecast(
            project_root=PROJECT_ROOT,
            models_dir=models_dir,
            tournament_path=QF_TOURNAMENT,
            profile=args.profile,
            n_sims=args.sims,
            seed=args.seed,
            data_root=args.data_root,
            extra_results_paths=extra,
            show_progress=not args.no_progress,
            print_eta=not args.no_eta,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    elapsed = payload.get("elapsed_seconds")
    bundle = {
        "forecast": payload,
        "report": {
            "profile": args.profile,
            "seed": args.seed,
            "n_sims": payload["n_sims"],
            "elapsed_seconds": elapsed,
            "seconds_per_sim": seconds_per_sim,
            "models_dir": str(models_dir),
            "data_root": str(data_layout.data_root),
            "simulation_round": "quarter_finals",
        },
    }
    write_forecast_outputs(
        bundle,
        forecast_path=forecast_path,
        report_path=report_path,
    )

    print(f"\nQF forecast complete: {forecast_path}")
    print(f"Report: {report_path}")
    print(f"Simulations: {payload.get('n_sims')}")
    if seconds_per_sim is not None:
        print(f"Seconds per sim: {seconds_per_sim:.4f}")
    print(f"Elapsed: {elapsed}s")
    print("\nTop champion probabilities:")
    champion_probs = payload.get("champion_probs", {})
    assert isinstance(champion_probs, dict)
    for team, prob in sorted(
        champion_probs.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {team}: {prob:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
