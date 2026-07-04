#!/usr/bin/env python3
"""Run Monte Carlo forecast for a user-defined tournament config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.predictor import load_calibrated_predictor  # noqa: E402
from worldcup_predictor.config import load_config, load_tournament_config  # noqa: E402
from worldcup_predictor.data.loader import load_and_clean_matches  # noqa: E402
from worldcup_predictor.simulation.state import build_initial_pipeline  # noqa: E402
from worldcup_predictor.simulation.tournament import TournamentSimulator  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "results.csv"
FORMER_NAMES_PATH = PROJECT_ROOT / "data" / "raw" / "former_names.csv"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="World Cup Monte Carlo forecast")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Missing {args.config}", file=sys.stderr)
        return 1

    config = load_config()
    tournament = load_tournament_config(args.config)
    n_sims = args.sims or config.simulation.n_sims

    matches = load_and_clean_matches(RAW_PATH, config, FORMER_NAMES_PATH)
    pipeline = build_initial_pipeline(matches, config, tournament.kickoff_date)

    predictor = load_calibrated_predictor(config, MODELS_DIR)

    sim = TournamentSimulator(
        predictor,
        config,
        pipeline,
        tournament,
        n_sims=n_sims,
        seed=args.seed,
    )
    result = sim.run()

    print(f"Forecast {tournament.year} — {result.n_sims} simulations")
    print("\nTop 10 champion probabilities:")
    for team, prob in sorted(
        result.champion_probs.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"  {team}: {prob:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
