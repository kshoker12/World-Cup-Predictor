"""Historical World Cup calibration backtests using trained production models."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from worldcup_predictor.calibration.artifacts import CalibrationArtifacts
from worldcup_predictor.calibration.predictor import load_calibrated_predictor
from worldcup_predictor.config import (
    AppConfig,
    load_profile,
    load_tournament_config,
    merge_former_name_aliases,
)
from worldcup_predictor.data.loader import load_and_clean_matches
from worldcup_predictor.kaggle_paths import DataLayout
from worldcup_predictor.simulation.state import build_initial_pipeline
from worldcup_predictor.simulation.tournament import TournamentResult, TournamentSimulator
from worldcup_predictor.simulation.wc2026_forecast import (
    _apply_profile_device,
    configure_forecast_runtime,
    run_knockout_forecast,
)


@dataclass(frozen=True)
class CalibrationMetrics:
    actual_champion: str
    p_actual_champion: float
    actual_champion_rank: int
    multiclass_brier: float
    log_score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "actual_champion": self.actual_champion,
            "p_actual_champion": self.p_actual_champion,
            "actual_champion_rank": self.actual_champion_rank,
            "multiclass_brier": self.multiclass_brier,
            "log_score": self.log_score,
        }


def compute_calibration_metrics(
    champion_probs: dict[str, float],
    actual_champion: str,
) -> CalibrationMetrics:
    if actual_champion not in champion_probs:
        raise ValueError(f"Actual champion {actual_champion!r} missing from champion_probs")

    p_actual = champion_probs[actual_champion]
    ranked = sorted(champion_probs.items(), key=lambda item: item[1], reverse=True)
    rank = next(i for i, (team, _) in enumerate(ranked, start=1) if team == actual_champion)
    brier = sum(
        (prob - (1.0 if team == actual_champion else 0.0)) ** 2
        for team, prob in champion_probs.items()
    )
    log_score = math.log(max(p_actual, 1e-15))

    return CalibrationMetrics(
        actual_champion=actual_champion,
        p_actual_champion=p_actual,
        actual_champion_rank=rank,
        multiclass_brier=brier,
        log_score=log_score,
    )


def resolve_models_dir(project_root: Path) -> Path:
    from worldcup_predictor.kaggle_paths import resolve_work_models_dir

    return resolve_work_models_dir(project_root)


def _tournament_paths(project_root: Path, year: int, full_tournament: bool) -> Path:
    if full_tournament:
        return project_root / "config" / "tournaments" / f"world_cup_{year}.yaml"
    knockout = project_root / "config" / "tournaments" / f"world_cup_{year}_knockout.yaml"
    if not knockout.exists():
        raise FileNotFoundError(f"Missing knockout config: {knockout}")
    return knockout


def _run_full_tournament_backtest(
    *,
    year: int,
    project_root: Path,
    models_dir: Path,
    profile: str,
    n_sims: int,
    seed: int,
    data_root: Path | None,
    show_progress: bool,
) -> tuple[dict[str, object], TournamentResult, float]:
    configure_forecast_runtime()
    tournament_path = project_root / "config" / "tournaments" / f"world_cup_{year}.yaml"
    if not tournament_path.exists():
        raise FileNotFoundError(f"Missing tournament config: {tournament_path}")

    config: AppConfig = _apply_profile_device(
        load_profile(profile, project_root / "config"), profile
    )
    tournament = load_tournament_config(tournament_path)
    if tournament.actual_champion is None:
        raise ValueError(f"Tournament config for {year} must set actual_champion")
    if tournament.groups is None:
        raise ValueError(f"Tournament config for {year} must define groups")

    data_layout = DataLayout.resolve(project_root, data_root)
    former_names = data_layout.former_names_csv()
    config = merge_former_name_aliases(config, former_names)

    extra: list[Path] = []
    wc2026 = data_layout.data_root / "wc2026_results.csv"
    if wc2026.exists():
        extra.append(wc2026)

    matches = load_and_clean_matches(
        data_layout.results_csv(),
        config,
        former_names_path=former_names,
        extra_results_paths=extra or None,
    )
    pipeline = build_initial_pipeline(matches, config, tournament.kickoff_date)
    predictor = load_calibrated_predictor(config, models_dir)

    started = time.monotonic()
    sim = TournamentSimulator(
        predictor,
        config,
        pipeline,
        tournament,
        n_sims=n_sims,
        seed=seed,
        show_progress=show_progress,
    )
    result = sim.run()
    elapsed = time.monotonic() - started
    return _build_payload(
        year=year,
        profile=profile,
        seed=seed,
        tournament=tournament,
        predictor_rho=predictor.rho,
        models_dir=models_dir,
        data_layout=data_layout,
        result_n_sims=result.n_sims,
        champion_probs=result.champion_probs,
        advancement_probs=result.advancement_probs,
        group_match_count=result.group_match_count,
        knockout_match_count=result.knockout_match_count,
        simulation_mode="full_tournament",
        tournament_path=tournament_path,
        elapsed=elapsed,
        seconds_per_sim=elapsed / n_sims if n_sims else None,
        match_win_probs=None,
        most_likely_bracket=None,
        most_likely_bracket_fraction=None,
    ), result, elapsed


def _build_payload(
    *,
    year: int,
    profile: str,
    seed: int,
    tournament,
    predictor_rho: float,
    models_dir: Path,
    data_layout: DataLayout,
    result_n_sims: int,
    champion_probs: dict[str, float],
    advancement_probs: dict[str, dict[str, float]],
    group_match_count: int,
    knockout_match_count: int,
    simulation_mode: str,
    tournament_path: Path,
    elapsed: float,
    seconds_per_sim: float | None,
    match_win_probs: list | None,
    most_likely_bracket: dict | None,
    most_likely_bracket_fraction: float | None,
) -> dict[str, object]:
    metrics = compute_calibration_metrics(champion_probs, tournament.actual_champion)
    artifacts = CalibrationArtifacts.load(models_dir / "calibration.json")
    payload: dict[str, object] = {
        "year": year,
        "profile": profile,
        "seed": seed,
        "n_sims": result_n_sims,
        "kickoff_date": tournament.kickoff_date.isoformat(),
        "simulation_mode": simulation_mode,
        "tournament_config": str(tournament_path),
        "group_match_count": group_match_count,
        "knockout_match_count": knockout_match_count,
        "rho": predictor_rho,
        "ensemble": {
            "w_gbm": artifacts.ensemble.w_gbm,
            "w_nn": artifacts.ensemble.w_nn,
            "w_bayesian": artifacts.ensemble.w_bayesian,
        },
        "calibration_metrics": metrics.as_dict(),
        "champion_probs": champion_probs,
        "advancement_probs": advancement_probs,
        "top_champion_probabilities": [
            {"team": team, "prob": prob}
            for team, prob in sorted(
                champion_probs.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ],
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_sim": round(seconds_per_sim, 6) if seconds_per_sim else None,
        "models_dir": str(models_dir),
        "data_root": str(data_layout.data_root),
    }
    if match_win_probs is not None:
        payload["match_win_probs"] = match_win_probs
    if most_likely_bracket is not None:
        payload["most_likely_bracket"] = most_likely_bracket
    if most_likely_bracket_fraction is not None:
        payload["most_likely_bracket_fraction"] = most_likely_bracket_fraction
    return payload


def run_calibration_backtest(
    *,
    year: int,
    project_root: Path,
    models_dir: Path,
    profile: str = "backtest",
    n_sims: int | None = None,
    seed: int = 42,
    data_root: Path | None = None,
    show_progress: bool = True,
    full_tournament: bool = False,
    print_eta: bool = True,
) -> tuple[dict[str, object], object]:
    if year not in (2018, 2022):
        raise ValueError("year must be 2018 or 2022")

    if not (models_dir / "calibration.json").exists():
        raise FileNotFoundError(
            f"Missing {models_dir / 'calibration.json'}. Train models first."
        )

    config = load_profile(profile, project_root / "config")
    sim_count = n_sims if n_sims is not None else config.simulation.n_sims

    if full_tournament:
        payload, result, _ = _run_full_tournament_backtest(
            year=year,
            project_root=project_root,
            models_dir=models_dir,
            profile=profile,
            n_sims=sim_count,
            seed=seed,
            data_root=data_root,
            show_progress=show_progress,
        )
        return payload, result

    tournament_path = _tournament_paths(project_root, year, full_tournament=False)
    tournament = load_tournament_config(tournament_path)
    if tournament.actual_champion is None:
        raise ValueError(f"Knockout config for {year} must set actual_champion")

    started = time.monotonic()
    knockout_result, forecast_payload, seconds_per_sim = run_knockout_forecast(
        project_root=project_root,
        models_dir=models_dir,
        tournament_path=tournament_path,
        profile=profile,
        n_sims=sim_count,
        seed=seed,
        data_root=data_root,
        extra_results_paths=None,
        show_progress=show_progress,
        print_eta=print_eta,
    )
    elapsed = time.monotonic() - started

    from worldcup_predictor.calibration.predictor import load_calibrated_predictor

    predictor = load_calibrated_predictor(
        _apply_profile_device(load_profile(profile, project_root / "config"), profile),
        models_dir,
    )
    data_layout = DataLayout.resolve(project_root, data_root)

    payload = _build_payload(
        year=year,
        profile=profile,
        seed=seed,
        tournament=tournament,
        predictor_rho=predictor.rho,
        models_dir=models_dir,
        data_layout=data_layout,
        result_n_sims=knockout_result.tournament.n_sims,
        champion_probs=knockout_result.tournament.champion_probs,
        advancement_probs=knockout_result.tournament.advancement_probs,
        group_match_count=0,
        knockout_match_count=knockout_result.tournament.knockout_match_count,
        simulation_mode="knockout_only",
        tournament_path=tournament_path,
        elapsed=elapsed,
        seconds_per_sim=seconds_per_sim,
        match_win_probs=forecast_payload.get("match_win_probs"),
        most_likely_bracket=forecast_payload.get("most_likely_bracket"),
        most_likely_bracket_fraction=forecast_payload.get("most_likely_bracket_fraction"),
    )
    return payload, knockout_result


def write_backtest_report(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main_for_year(year: int, argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path as _Path

    project_root = _Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(
        description=(
            f"Robust {year} World Cup calibration backtest "
            "(knockout-only MC, same engine as WC 2026 forecast)"
        )
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "backtest", "kaggle"],
        default="backtest",
        help="Simulation count profile (backtest=50k sims default)",
    )
    parser.add_argument("--sims", type=int, default=None, help="Override simulation count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=_Path, default=None)
    parser.add_argument("--data-root", type=_Path, default=None)
    parser.add_argument("--output", type=_Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--full-tournament",
        action="store_true",
        help="Slow path: simulate group stage + knockout (~4x slower)",
    )
    parser.add_argument("--no-eta", action="store_true")
    args = parser.parse_args(argv)

    models_dir = args.models_dir or resolve_models_dir(project_root)
    output = args.output or models_dir / f"wc{year}_calibration_backtest.json"

    try:
        payload, _ = run_calibration_backtest(
            year=year,
            project_root=project_root,
            models_dir=models_dir,
            profile=args.profile,
            n_sims=args.sims,
            seed=args.seed,
            data_root=args.data_root,
            show_progress=not args.no_progress,
            full_tournament=args.full_tournament,
            print_eta=not args.no_eta,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_backtest_report(payload, output)

    metrics = payload["calibration_metrics"]
    assert isinstance(metrics, dict)
    mode = payload.get("simulation_mode", "knockout_only")
    print(f"WC {year} calibration backtest ({mode}) — {payload['n_sims']} simulations")
    print(f"Actual champion: {metrics['actual_champion']}")
    print(f"P(actual champion): {metrics['p_actual_champion']:.4f}")
    print(f"Champion rank: {metrics['actual_champion_rank']}")
    print(f"Multiclass Brier: {metrics['multiclass_brier']:.4f}")
    print(f"Log score: {metrics['log_score']:.4f}")
    print(f"Elapsed: {payload['elapsed_seconds']}s")
    if payload.get("seconds_per_sim") is not None:
        print(f"Seconds per sim: {payload['seconds_per_sim']}")
    print("\nTop 5 champion probabilities:")
    for row in payload["top_champion_probabilities"][:5]:
        assert isinstance(row, dict)
        print(f"  {row['team']}: {row['prob']:.4f}")
    print(f"\nReport: {output}")
    return 0
