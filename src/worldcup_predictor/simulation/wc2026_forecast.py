"""Knockout-only Monte Carlo forecast using pre-trained production models."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from worldcup_predictor.calibration.predictor import load_calibrated_predictor
from worldcup_predictor.config import (
    AppConfig,
    load_profile,
    load_tournament_config,
    merge_former_name_aliases,
)
from worldcup_predictor.data.loader import load_and_clean_matches
from worldcup_predictor.kaggle_paths import DataLayout, verify_model_artifacts
from worldcup_predictor.simulation.knockout import KnockoutSimulationResult, KnockoutSimulator
from worldcup_predictor.simulation.state import build_initial_pipeline


def configure_forecast_runtime() -> None:
    """Suppress pandas fragmentation warnings during embedding feature assembly."""
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _tournament_config_path(project_root: Path) -> Path:
    path = project_root / "config" / "tournaments" / "world_cup_2026_knockout.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing tournament config: {path}")
    return path


def resolve_n_sims(profile: str, config_root: Path, n_sims: int | None) -> int:
    if n_sims is not None:
        return n_sims
    config = load_profile(profile, config_root)
    return config.simulation.n_sims


def _apply_profile_device(config: AppConfig, profile: str) -> AppConfig:
    nn_device = "cuda" if _cuda_available() else "cpu"
    if profile == "fast":
        return config
    return replace(
        config,
        nn=replace(config.nn, device=nn_device, num_workers=0),
    )


def run_knockout_forecast(
    *,
    project_root: Path,
    models_dir: Path,
    tournament_path: Path,
    profile: str = "kaggle",
    n_sims: int | None = None,
    seed: int = 42,
    data_root: Path | None = None,
    extra_results_paths: list[Path] | None = None,
    show_progress: bool = True,
    print_eta: bool = True,
) -> tuple[KnockoutSimulationResult, dict[str, object], float | None]:
    """Run knockout-only MC (same engine as WC 2026 forecast)."""
    configure_forecast_runtime()

    missing = verify_model_artifacts(models_dir)
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in {models_dir}: {missing}"
        )

    config_root = project_root / "config"
    sim_count = resolve_n_sims(profile, config_root, n_sims)
    config = _apply_profile_device(load_profile(profile, config_root), profile)

    tournament = load_tournament_config(tournament_path)
    if not tournament.is_knockout_only:
        raise ValueError(f"Tournament config must use mode=knockout_only: {tournament_path}")

    data_layout = DataLayout.resolve(project_root, data_root)
    former_names = data_layout.former_names_csv()
    config = merge_former_name_aliases(config, former_names)

    extra = list(extra_results_paths or [])
    matches = load_and_clean_matches(
        data_layout.results_csv(),
        config,
        former_names_path=former_names,
        extra_results_paths=extra or None,
    )
    pipeline = build_initial_pipeline(matches, config, tournament.kickoff_date)
    predictor = load_calibrated_predictor(config, models_dir)

    seconds_per_sim: float | None = None
    if print_eta and sim_count >= 25:
        warmup = KnockoutSimulator(
            predictor,
            config,
            pipeline,
            tournament,
            n_sims=min(25, sim_count),
            seed=seed,
            show_progress=False,
        )
        seconds_per_sim = warmup.benchmark(min(25, sim_count))
        eta_hours = seconds_per_sim * sim_count / 3600.0
        print(
            f"ETA benchmark: {seconds_per_sim:.3f}s/sim -> "
            f"~{eta_hours:.2f}h for {sim_count:,} simulations"
        )

    print(
        f"Running knockout forecast: n_sims={sim_count:,}, "
        f"profile={profile}, seed={seed}, tournament={tournament_path.name}"
    )
    started = time.monotonic()
    sim = KnockoutSimulator(
        predictor,
        config,
        pipeline,
        tournament,
        n_sims=sim_count,
        seed=seed,
        show_progress=show_progress,
    )
    result = sim.run()
    elapsed = time.monotonic() - started
    if seconds_per_sim is None and sim_count > 0:
        seconds_per_sim = elapsed / sim_count

    payload: dict[str, object] = {
        "n_sims": result.tournament.n_sims,
        "champion_probs": result.tournament.champion_probs,
        "advancement_probs": result.tournament.advancement_probs,
        "match_win_probs": result.match_win_probs,
        "most_likely_bracket": result.most_likely_bracket,
        "most_likely_bracket_count": result.most_likely_bracket_count,
        "most_likely_bracket_fraction": (
            result.most_likely_bracket_count / result.tournament.n_sims
            if result.tournament.n_sims
            else 0.0
        ),
        "sample_bracket": result.sample_bracket,
        "kickoff_date": tournament.kickoff_date.isoformat(),
        "tournament_config": str(tournament_path),
        "simulation_mode": "knockout_only",
        "simulation_round": tournament.start_round,
    }
    report: dict[str, object] = {
        "profile": profile,
        "seed": seed,
        "n_sims": sim_count,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_sim": round(seconds_per_sim, 6) if seconds_per_sim else None,
        "models_dir": str(models_dir),
        "data_root": str(data_layout.data_root),
        "finished_at": _utc_now(),
    }
    payload["elapsed_seconds"] = report["elapsed_seconds"]
    return result, payload, seconds_per_sim


def run_wc2026_forecast(
    *,
    project_root: Path,
    models_dir: Path,
    profile: str = "kaggle",
    n_sims: int | None = None,
    seed: int = 42,
    data_root: Path | None = None,
    show_progress: bool = True,
    print_eta: bool = True,
) -> tuple[dict[str, object], float | None]:
    data_layout = DataLayout.resolve(project_root, data_root)
    extra = [data_layout.wc2026_results_csv()]
    _, payload, seconds_per_sim = run_knockout_forecast(
        project_root=project_root,
        models_dir=models_dir,
        tournament_path=_tournament_config_path(project_root),
        profile=profile,
        n_sims=n_sims,
        seed=seed,
        data_root=data_root,
        extra_results_paths=extra,
        show_progress=show_progress,
        print_eta=print_eta,
    )
    elapsed = payload.get("elapsed_seconds")
    if seconds_per_sim is not None and elapsed is None:
        n = int(payload["n_sims"])  # type: ignore[arg-type]
        elapsed = round(n * seconds_per_sim, 2)
    report: dict[str, object] = {
        "profile": profile,
        "seed": seed,
        "n_sims": payload["n_sims"],
        "elapsed_seconds": elapsed,
        "seconds_per_sim": seconds_per_sim,
        "models_dir": str(models_dir),
        "data_root": str(data_layout.data_root),
        "finished_at": _utc_now(),
    }
    return {"forecast": payload, "report": report}, seconds_per_sim


def write_forecast_outputs(
    bundle: dict[str, object],
    *,
    forecast_path: Path,
    report_path: Path | None = None,
) -> None:
    forecast = bundle["forecast"]
    assert isinstance(forecast, dict)
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    with forecast_path.open("w", encoding="utf-8") as f:
        json.dump(forecast, f, indent=2)

    if report_path is not None:
        report = bundle["report"]
        assert isinstance(report, dict)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
