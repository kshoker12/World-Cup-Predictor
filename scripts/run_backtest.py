#!/usr/bin/env python3
"""Run Monte Carlo backtest for a historical World Cup."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
    parser = argparse.ArgumentParser(description="World Cup Monte Carlo backtest")
    parser.add_argument("--year", type=int, required=True, choices=[2018, 2022])
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    tournament_path = PROJECT_ROOT / "config" / "tournaments" / f"world_cup_{args.year}.yaml"
    if not tournament_path.exists():
        print(f"ERROR: Missing {tournament_path}", file=sys.stderr)
        return 1

    config = load_config()
    tournament = load_tournament_config(tournament_path)
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
        show_progress=not args.no_progress,
    )
    result = sim.run()

    print(f"Backtest {args.year} — {result.n_sims} simulations (calibrated, rho={predictor.rho:.4f})")
    print(f"Group matches per sim: {result.group_match_count}")
    print(f"Knockout matches per sim: {result.knockout_match_count}")
    print(f"\nActual champion: {tournament.actual_champion}")
    print(f"P(champion): {result.champion_probs[tournament.actual_champion]:.4f}")

    print("\nTop 5 champion probabilities:")
    for team, prob in sorted(
        result.champion_probs.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {team}: {prob:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
