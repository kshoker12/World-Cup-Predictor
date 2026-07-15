#!/usr/bin/env python3
"""WC 2026 final and third-place forecast using pre-trained models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.predictor import load_calibrated_predictor  # noqa: E402
from worldcup_predictor.config import (  # noqa: E402
    load_profile,
    load_tournament_config,
    merge_former_name_aliases,
)
from worldcup_predictor.data.loader import load_and_clean_matches  # noqa: E402
from worldcup_predictor.kaggle_paths import (  # noqa: E402
    DataLayout,
    resolve_work_models_dir,
    verify_model_artifacts,
)
from worldcup_predictor.simulation.finals import FinalsSimulator  # noqa: E402
from worldcup_predictor.simulation.state import build_initial_pipeline  # noqa: E402
from worldcup_predictor.simulation.wc2026_forecast import (  # noqa: E402
    _apply_profile_device,
    configure_forecast_runtime,
    resolve_n_sims,
)

configure_forecast_runtime()

TOURNAMENT_PATH = (
    PROJECT_ROOT / "config" / "tournaments" / "world_cup_2026_finals.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WC 2026 final + third-place forecast"
    )
    parser.add_argument(
        "--profile", choices=["fast", "kaggle", "backtest"], default="kaggle"
    )
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir or resolve_work_models_dir(PROJECT_ROOT)
    missing = verify_model_artifacts(models_dir)
    if missing:
        print(f"ERROR: Missing model artifacts: {missing}", file=sys.stderr)
        return 1

    config_root = PROJECT_ROOT / "config"
    config = _apply_profile_device(
        load_profile(args.profile, config_root), args.profile
    )
    n_sims = resolve_n_sims(args.profile, config_root, args.sims)
    tournament = load_tournament_config(TOURNAMENT_PATH)
    data_layout = DataLayout.resolve(PROJECT_ROOT, args.data_root)
    former_names = data_layout.former_names_csv()
    config = merge_former_name_aliases(config, former_names)
    matches = load_and_clean_matches(
        data_layout.results_csv(),
        config,
        former_names_path=former_names,
        extra_results_paths=[data_layout.wc2026_results_csv()],
    )
    pipeline = build_initial_pipeline(matches, config, tournament.kickoff_date)
    predictor = load_calibrated_predictor(config, models_dir)

    print(
        f"Running final + third-place forecast: n_sims={n_sims:,}, "
        f"profile={args.profile}, seed={args.seed}"
    )
    started = time.monotonic()
    result = FinalsSimulator(
        predictor,
        config,
        pipeline,
        tournament,
        n_sims=n_sims,
        seed=args.seed,
        show_progress=not args.no_progress,
    ).run()
    elapsed = time.monotonic() - started

    payload = {
        "n_sims": result.n_sims,
        "champion_probs": result.champion_probs,
        "advancement_probs": {
            team: {"final": 1.0, "champion": probability}
            for team, probability in result.champion_probs.items()
        },
        "match_win_probs": result.match_win_probs,
        "most_likely_bracket": result.most_likely_results,
        "most_likely_bracket_count": result.most_likely_results_count,
        "most_likely_bracket_fraction": (
            result.most_likely_results_count / result.n_sims
        ),
        "sample_bracket": result.sample_results,
        "kickoff_date": tournament.kickoff_date.isoformat(),
        "tournament_config": str(TOURNAMENT_PATH),
        "simulation_mode": "finals",
        "simulation_round": "final",
        "elapsed_seconds": round(elapsed, 2),
    }
    output = args.output or models_dir / "wc2026_forecast_final.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = {
        "profile": args.profile,
        "seed": args.seed,
        "n_sims": n_sims,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_sim": round(elapsed / n_sims, 6),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (models_dir / "forecast_final_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Forecast complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
