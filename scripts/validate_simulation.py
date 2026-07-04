#!/usr/bin/env python3
"""Validate Monte Carlo tournament simulation output."""

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

ACTUAL_CHAMPIONS = {2018: "France", 2022: "Argentina"}


class ValidationError(Exception):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tournament simulation")
    parser.add_argument("--year", type=int, required=True, choices=[2018, 2022])
    parser.add_argument("--sims", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tournament_path = PROJECT_ROOT / "config" / "tournaments" / f"world_cup_{args.year}.yaml"
    if not MODELS_DIR.joinpath("gbm_home.txt").exists():
        print("ERROR: Models not found. Run train_gbm.py first.", file=sys.stderr)
        return 1
    if not MODELS_DIR.joinpath("calibration.json").exists():
        print("ERROR: calibration.json not found. Run fit_calibration.py first.", file=sys.stderr)
        return 1

    config = load_config()
    tournament = load_tournament_config(tournament_path)
    matches = load_and_clean_matches(RAW_PATH, config, FORMER_NAMES_PATH)

    all_teams_in_data = set(matches["home_team"]) | set(matches["away_team"])
    config_teams = [t for teams in tournament.groups.values() for t in teams]
    missing = [t for t in config_teams if t not in all_teams_in_data]
    if missing:
        print(f"ERROR: Teams not in data: {missing}", file=sys.stderr)
        return 1

    pipeline = build_initial_pipeline(matches, config, tournament.kickoff_date)
    predictor = load_calibrated_predictor(config, MODELS_DIR)

    sim = TournamentSimulator(
        predictor, config, pipeline, tournament, n_sims=args.sims, seed=args.seed
    )
    result = sim.run()

    try:
        if result.group_match_count != 48:
            raise ValidationError(
                f"Expected 48 group matches, got {result.group_match_count}"
            )
        if result.knockout_match_count != 15:
            raise ValidationError(
                f"Expected 15 knockout matches, got {result.knockout_match_count}"
            )

        prob_sum = sum(result.champion_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValidationError(f"Champion probs sum to {prob_sum}, not 1.0")

        champion = ACTUAL_CHAMPIONS[args.year]
        if result.champion_probs.get(champion, 0.0) <= 0.0:
            raise ValidationError(f"Actual champion {champion} has zero probability")

        for team in config_teams:
            adv = result.advancement_probs[team]
            if adv["champion"] > adv["final"] + 1e-9:
                raise ValidationError(f"{team}: P(champion) > P(final)")
            if adv["final"] > adv["semi_finals"] + 1e-9:
                raise ValidationError(f"{team}: P(final) > P(semi_finals)")
            if adv["semi_finals"] > adv["quarter_finals"] + 1e-9:
                raise ValidationError(f"{team}: P(semi) > P(quarters)")
            if adv["quarter_finals"] > adv["round_of_16"] + 1e-9:
                raise ValidationError(f"{team}: P(quarters) > P(R16)")

        sim2 = TournamentSimulator(
            predictor, config, pipeline, tournament, n_sims=args.sims, seed=args.seed
        )
        result2 = sim2.run()
        for team in config_teams:
            if abs(
                result.champion_probs[team] - result2.champion_probs[team]
            ) > 1e-12:
                raise ValidationError("Reproducibility failed for champion probs")

    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Simulation validation passed for {args.year} ({args.sims} sims)")
    print(f"P({ACTUAL_CHAMPIONS[args.year]} wins): "
          f"{result.champion_probs[ACTUAL_CHAMPIONS[args.year]]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
