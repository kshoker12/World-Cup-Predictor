#!/usr/bin/env python3
"""Unified Kaggle/local pipeline: train, ablate, calibrate, forecast WC 2026."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.artifacts import fit_calibration  # noqa: E402
from worldcup_predictor.calibration.predictor import load_calibrated_predictor  # noqa: E402
from worldcup_predictor.config import (  # noqa: E402
    load_profile,
    load_tournament_config,
    merge_former_name_aliases,
)
from worldcup_predictor.data.loader import load_and_clean_matches  # noqa: E402
from worldcup_predictor.features.pipeline import MatchPipeline  # noqa: E402
from worldcup_predictor.kaggle_paths import DataLayout  # noqa: E402
from worldcup_predictor.models.gbm import GBMPredictor  # noqa: E402
from worldcup_predictor.models.nn.device import resolve_device  # noqa: E402
from worldcup_predictor.models.nn.embeddings import (  # noqa: E402
    augment_features_with_embeddings,
    gbm_feature_columns_with_embeddings,
)
from worldcup_predictor.models.nn.predictor import NNPredictor  # noqa: E402
from worldcup_predictor.models.sequences import build_international_sequences  # noqa: E402
from worldcup_predictor.simulation.knockout import KnockoutSimulator  # noqa: E402
from worldcup_predictor.simulation.state import build_initial_pipeline  # noqa: E402

TOURNAMENT_CONFIG = (
    PROJECT_ROOT / "config" / "tournaments" / "world_cup_2026_knockout.yaml"
)


def resolve_work_root(project_root: Path) -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        kaggle_working.mkdir(parents=True, exist_ok=True)
        return kaggle_working
    return project_root / "data"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineRunner:
    def __init__(
        self,
        *,
        profile: str,
        resume: bool,
        seed: int,
        project_root: Path,
        data_root: Path | None = None,
    ) -> None:
        self.profile_name = profile
        self.resume = resume
        self.seed = seed
        self.project_root = project_root
        self.data_layout = DataLayout.resolve(project_root, data_root)
        self.work_root = resolve_work_root(project_root)
        self.config_root = project_root / "config"
        self.processed_dir = self.work_root / "processed"
        self.models_dir = self.work_root / "models"
        self.report_path = self.models_dir / "pipeline_report.json"
        self.forecast_path = self.models_dir / "wc2026_forecast.json"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        pytensor_cache = self.work_root / ".pytensor"
        pytensor_cache.mkdir(parents=True, exist_ok=True)
        os.environ["PYTENSOR_FLAGS"] = f"compiledir={pytensor_cache}"

        self.config = load_profile(profile, self.config_root)
        nn_device = "cuda" if _cuda_available() else "cpu"
        self.config = replace(
            self.config,
            nn=replace(
                self.config.nn,
                device=nn_device if profile != "fast" else "cpu",
                num_workers=0,
            ),
        )
        self.report: dict[str, object] = {
            "profile": profile,
            "started_at": _utc_now(),
            "seed": seed,
            "data_root": str(self.data_layout.data_root),
            "work_root": str(self.work_root),
            "time_budget_hours": self.config.pipeline.time_budget_hours,
            "phases": {},
        }
        self._started = time.monotonic()

    def _elapsed_hours(self) -> float:
        return (time.monotonic() - self._started) / 3600.0

    def _budget_remaining(self) -> float:
        budget = self.config.pipeline.time_budget_hours
        if budget <= 0:
            return float("inf")
        return max(0.0, budget - self._elapsed_hours())

    def _scale_bayesian_for_budget(self) -> None:
        """Reserve forecast time up front; scale MCMC if training ran long."""
        budget = self.config.pipeline.time_budget_hours
        if budget <= 0:
            return

        forecast_reserve_h = max(1.75, budget * 0.34)
        calibration_reserve_h = max(0.05, budget * 0.01)
        bayesian_allowance = (
            budget - self._elapsed_hours() - forecast_reserve_h - calibration_reserve_h
        )
        reference_h = 2.75
        if bayesian_allowance >= reference_h:
            return

        scale = max(0.2, bayesian_allowance / reference_h)
        draws = max(250, int(self.config.bayesian.draws * scale))
        tune = max(250, int(self.config.bayesian.tune * scale))
        chains = self.config.bayesian.chains
        if bayesian_allowance < 0.75:
            chains = max(2, min(chains, 2))
        self.config = replace(
            self.config,
            bayesian=replace(
                self.config.bayesian,
                draws=draws,
                tune=tune,
                chains=chains,
            ),
        )
        print(
            "Bayesian scaled for time budget: "
            f"chains={chains}, tune={tune}, draws={draws} "
            f"(allowance={bayesian_allowance:.2f}h)"
        )

    def _plan_sim_count(
        self,
        sim: KnockoutSimulator,
        requested: int,
    ) -> tuple[int, float | None]:
        """Use remaining budget after a warmup benchmark to maximize sim count."""
        budget = self.config.pipeline.time_budget_hours
        if budget <= 0:
            return requested, None

        remaining_h = self._budget_remaining() - (3.0 / 60.0)
        if remaining_h <= 0:
            fallback = max(25_000, requested // 4)
            print(f"Budget exhausted before forecast; using n_sims={fallback}")
            return fallback, None

        warmup_n = min(100, max(25, requested // 2000))
        seconds_per_sim = sim.benchmark(warmup_n)
        usable_seconds = remaining_h * 3600.0 * 0.97
        max_by_budget = int(usable_seconds / seconds_per_sim)
        planned = min(requested, max(25_000, max_by_budget))
        print(
            f"Forecast plan: {seconds_per_sim:.3f}s/sim, "
            f"{remaining_h:.2f}h remaining -> n_sims={planned:,} "
            f"(cap={requested:,})"
        )
        return planned, seconds_per_sim

    def _record_phase(self, name: str, started: float, extra: dict | None = None) -> None:
        entry = {"seconds": round(time.monotonic() - started, 2), "finished_at": _utc_now()}
        if extra:
            entry.update(extra)
        phases = self.report.setdefault("phases", {})
        assert isinstance(phases, dict)
        phases[name] = entry

    def _artifact_exists(self, *parts: str) -> bool:
        return (self.models_dir / Path(*parts)).exists()

    def _save_report(self) -> None:
        self.report["elapsed_hours"] = round(self._elapsed_hours(), 4)
        self.report["finished_at"] = _utc_now()
        with self.report_path.open("w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)

    def run(self) -> int:
        print(f"Profile: {self.profile_name}")
        print(f"Data root: {self.data_layout.data_root}")
        print(f"Work root: {self.work_root}")

        features_path = self.processed_dir / "features.parquet"
        seq_path = self.processed_dir / "intl_sequences.npz"
        emb_features_path = self.processed_dir / "features_with_embeddings.parquet"

        t0 = time.monotonic()
        if not (self.resume and features_path.exists()):
            self._build_features(features_path)
        else:
            print(f"Skipping feature build (resume): {features_path}")
        self._record_phase("features", t0)

        t0 = time.monotonic()
        if not (self.resume and seq_path.exists()):
            self._build_sequences(features_path, seq_path)
        else:
            print(f"Skipping sequence build (resume): {seq_path}")
        self._record_phase("sequences", t0)

        t0 = time.monotonic()
        club_seq_path = self._ensure_club_sequences()
        self._record_phase(
            "club_sequences",
            t0,
            {"path": str(club_seq_path) if club_seq_path else None},
        )

        features = pd.read_parquet(features_path).reset_index(drop=True)
        val_df = features[features["split"] == "val"]
        seq = np.load(seq_path)
        val_idx = np.where(features["split"].values == "val")[0]

        t0 = time.monotonic()
        nn_path = self.models_dir / "nn_model.pt"
        if self.resume and nn_path.exists():
            nn = NNPredictor(self.config.nn)
            nn.load(self.models_dir)
        else:
            nn = self._train_nn(features, seq)
        nn_val = nn.evaluate(
            val_df,
            seq["home_seq"][val_idx],
            seq["away_seq"][val_idx],
        )
        self._record_phase("nn", t0, {"val_poisson_deviance_total": nn_val["poisson_deviance_total"]})

        t0 = time.monotonic()
        if not (self.resume and emb_features_path.exists()):
            emb_features = augment_features_with_embeddings(
                features,
                seq["home_seq"],
                seq["away_seq"],
                nn,
            )
            emb_features.to_parquet(emb_features_path, index=False)
        else:
            emb_features = pd.read_parquet(emb_features_path).reset_index(drop=True)
        self._record_phase("embeddings", t0)

        t0 = time.monotonic()
        if self.resume and self._artifact_exists("gbm_home.txt") and self.report_path.exists():
            with self.report_path.open(encoding="utf-8") as f:
                prior = json.load(f)
            gate = prior.get("gate_gbm", {})
            winner = str(gate.get("winner", "gbm_plain"))
            plain_val = {
                "poisson_deviance_total": float(
                    gate.get("plain_val_poisson_deviance_total", 0.0)
                )
            }
            emb_val = {
                "poisson_deviance_total": float(
                    gate.get("emb_val_poisson_deviance_total", 0.0)
                )
            }
            production_gbm = GBMPredictor(
                self.config.gbm,
                feature_columns=(
                    gbm_feature_columns_with_embeddings(self.config.nn.hidden_dim)
                    if winner == "gbm_emb"
                    else None
                ),
            )
            production_gbm.load(self.models_dir)
            print(f"Skipping GBM ablation (resume), winner={winner}")
        else:
            train_df = features[features["split"] == "train"]
            gbm_plain = GBMPredictor(self.config.gbm)
            gbm_plain.fit(train_df, val_df)
            plain_val = gbm_plain.evaluate(val_df)

            emb_cols = gbm_feature_columns_with_embeddings(self.config.nn.hidden_dim)
            train_emb = emb_features[emb_features["split"] == "train"]
            val_emb = emb_features[emb_features["split"] == "val"]
            gbm_emb = GBMPredictor(self.config.gbm, feature_columns=emb_cols)
            gbm_emb.fit(train_emb, val_emb)
            emb_val = gbm_emb.evaluate(val_emb)

            if emb_val["poisson_deviance_total"] < plain_val["poisson_deviance_total"]:
                winner = "gbm_emb"
                production_gbm = gbm_emb
            else:
                winner = "gbm_plain"
                production_gbm = gbm_plain
            production_gbm.save(self.models_dir)
            self.report["gate_gbm"] = {
                "winner": winner,
                "plain_val_poisson_deviance_total": plain_val["poisson_deviance_total"],
                "emb_val_poisson_deviance_total": emb_val["poisson_deviance_total"],
            }
            print(f"gate_gbm winner: {winner}")

        self._record_phase(
            "gbm_ablation",
            t0,
            {
                "winner": winner,
                "plain_val_poisson_deviance_total": plain_val["poisson_deviance_total"],
                "emb_val_poisson_deviance_total": emb_val["poisson_deviance_total"],
            },
        )

        t0 = time.monotonic()
        bayesian_path = self.models_dir / "bayesian.json"
        if self.resume and bayesian_path.exists():
            from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts

            bayesian = BayesianArtifacts.load(bayesian_path)
        else:
            self._scale_bayesian_for_budget()
            from worldcup_predictor.models.bayesian.trainer import fit_bayesian

            bayesian = fit_bayesian(features, self.config)
            bayesian.save(bayesian_path)
        self._record_phase("bayesian", t0, {"rho_mean": bayesian.rho_mean})

        t0 = time.monotonic()
        cal_path = self.models_dir / "calibration.json"
        val_for_gbm = (
            emb_features[emb_features["split"] == "val"]
            if winner == "gbm_emb"
            else val_df
        )
        if self.resume and cal_path.exists():
            from worldcup_predictor.calibration.artifacts import CalibrationArtifacts

            artifacts = CalibrationArtifacts.load(cal_path)
        else:
            artifacts = fit_calibration(
                production_gbm,
                val_for_gbm,
                self.config.calibration,
                self.config,
                max_goals=self.config.simulation.max_goals,
                nn=nn,
                val_home_seq=seq["home_seq"][val_idx],
                val_away_seq=seq["away_seq"][val_idx],
                bayesian=bayesian,
            )
            artifacts.save(cal_path)
        self._validate_calibration(artifacts)
        self._record_phase(
            "calibration",
            t0,
            {
                "w_gbm": artifacts.ensemble.w_gbm,
                "w_nn": artifacts.ensemble.w_nn,
                "w_bayesian": artifacts.ensemble.w_bayesian,
                "min_ensemble_weight": artifacts.min_ensemble_weight,
            },
        )

        t0 = time.monotonic()
        forecast, n_sims, seconds_per_sim = self._run_forecast(
            self.config.simulation.n_sims
        )
        self._record_phase(
            "forecast",
            t0,
            {
                "n_sims": n_sims,
                "seconds_per_sim": seconds_per_sim,
                "most_likely_bracket_count": forecast.get(
                    "most_likely_bracket_count", 0
                ),
            },
        )
        self._save_report()

        champion = max(
            forecast["champion_probs"].items(),
            key=lambda item: item[1],
        )
        print("\nTop 5 champion probabilities:")
        for team, prob in sorted(
            forecast["champion_probs"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]:
            print(f"  {team}: {prob:.4f}")
        print(f"\nPredicted champion: {champion[0]} ({champion[1]:.4f})")
        print(f"Report: {self.report_path}")
        print(f"Forecast: {self.forecast_path}")
        return 0

    def _build_features(self, output_path: Path) -> None:
        raw_path = self.data_layout.results_csv()
        wc2026_path = self.data_layout.wc2026_results_csv()
        former_names = self.data_layout.former_names_csv()

        config = merge_former_name_aliases(self.config, former_names)
        matches = load_and_clean_matches(
            raw_path,
            config,
            former_names_path=former_names,
            extra_results_paths=[wc2026_path],
        )
        features = MatchPipeline(config).run(matches)
        features.to_parquet(output_path, index=False)
        print(f"Wrote {len(features):,} feature rows to {output_path}")

    def _build_sequences(self, features_path: Path, output_path: Path) -> None:
        features = pd.read_parquet(features_path)
        match_cols = [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        ]
        matches = features[match_cols].copy()
        home_seq, away_seq = build_international_sequences(matches, self.config)
        np.savez(output_path, home_seq=home_seq, away_seq=away_seq)
        print(f"Wrote sequences to {output_path}")

    def _club_sequences_path(self) -> Path:
        for candidate in (
            self.processed_dir / "club_sequences.npz",
            self.data_layout.data_root / "club_sequences.npz",
            self.data_layout.data_root / "processed" / "club_sequences.npz",
            self.project_root / "data" / "processed" / "club_sequences.npz",
        ):
            if candidate.exists():
                return candidate
        return self.processed_dir / "club_sequences.npz"

    def _ensure_club_sequences(self) -> Path | None:
        if self.config.pipeline.skip_club_pretrain:
            print("Club pretrain skipped (skip_club_pretrain=true)")
            return None

        existing = self._club_sequences_path()
        if self.resume and existing.exists():
            print(f"Using existing club sequences: {existing}")
            return existing

        from worldcup_predictor.data.club_loader import (
            load_soccer_dataset,
            load_understat_matches,
        )
        from worldcup_predictor.data.club_merge import merge_club_sources
        from worldcup_predictor.models.sequences import (
            build_club_sequences,
            compute_club_norm_stats,
        )

        understat_path = self.data_layout.understat_parquet()
        club_dir = self.data_layout.club_dir()

        if understat_path is None or club_dir is None:
            print(
                "WARNING: Club data missing; skipping club pretrain. "
                "Need understat_matches.parquet plus fixtures.csv/match_stats.csv."
            )
            self.config = replace(
                self.config,
                pipeline=replace(self.config.pipeline, skip_club_pretrain=True),
            )
            return None

        club_matches_path = self.processed_dir / "club_matches.parquet"
        if not club_matches_path.exists() or not self.resume:
            understat = load_understat_matches(understat_path)
            soccer = load_soccer_dataset(club_dir)
            merged = merge_club_sources(
                understat,
                soccer,
                min_date=self.config.club.min_date,
                forward_fill_within_league=self.config.club.forward_fill_within_league,
            )
            merged.to_parquet(club_matches_path, index=False)
            print(f"Wrote {len(merged):,} club matches to {club_matches_path}")

        matches = pd.read_parquet(club_matches_path)
        norm_stats = compute_club_norm_stats(matches)
        home_seq, away_seq, y_home, y_away = build_club_sequences(
            matches,
            seq_len=self.config.nn.seq_len,
            elo_config=self.config.elo,
            norm_stats=norm_stats,
        )
        output_path = self.processed_dir / "club_sequences.npz"
        np.savez(
            output_path,
            home_seq=home_seq,
            away_seq=away_seq,
            y_home=y_home,
            y_away=y_away,
        )
        print(f"Wrote club sequences to {output_path} ({len(y_home):,} matches)")
        return output_path

    def _train_nn(self, features: pd.DataFrame, seq: np.lib.npyio.NpzFile) -> NNPredictor:
        nn = NNPredictor(self.config.nn)
        device = resolve_device(self.config.nn)
        print(f"NN device: {device}")

        club_path = self._club_sequences_path()
        if (
            not self.config.pipeline.skip_club_pretrain
            and club_path.exists()
            and self.config.nn.pretrain_epochs > 0
        ):
            club = np.load(club_path)
            nn.pretrain(
                club["home_seq"],
                club["away_seq"],
                club["y_home"],
                club["y_away"],
                show_progress=True,
            )
            nn.save_pretrain(self.models_dir)

        train_idx = np.where(features["split"].values == "train")[0]
        val_idx = np.where(features["split"].values == "val")[0]
        nn.finetune(
            seq["home_seq"],
            seq["away_seq"],
            features,
            train_idx,
            val_idx,
            show_progress=True,
        )
        nn.save(self.models_dir)
        return nn

    def _validate_calibration(self, artifacts) -> None:
        min_w = artifacts.min_ensemble_weight
        if min_w <= 0:
            return
        for name, weight in (
            ("w_gbm", artifacts.ensemble.w_gbm),
            ("w_nn", artifacts.ensemble.w_nn),
            ("w_bayesian", artifacts.ensemble.w_bayesian),
        ):
            if weight > 0 and weight + 1e-9 < min_w:
                raise RuntimeError(
                    f"Calibration weight {name}={weight:.4f} below min {min_w:.4f}"
                )
        total = (
            artifacts.ensemble.w_gbm
            + artifacts.ensemble.w_nn
            + artifacts.ensemble.w_bayesian
        )
        if abs(total - 1.0) > 1e-4:
            raise RuntimeError(f"Ensemble weights sum to {total:.4f}, expected 1.0")

    def _run_forecast(
        self, requested_sims: int
    ) -> tuple[dict[str, object], int, float | None]:
        tournament_path = TOURNAMENT_CONFIG
        if not tournament_path.exists():
            tournament_path = (
                self.config_root / "tournaments" / "world_cup_2026_knockout.yaml"
            )
        tournament = load_tournament_config(tournament_path)

        raw_path = self.data_layout.results_csv()
        wc2026_path = self.data_layout.wc2026_results_csv()
        former_names = self.data_layout.former_names_csv()

        config = merge_former_name_aliases(self.config, former_names)
        matches = load_and_clean_matches(
            raw_path,
            config,
            former_names_path=former_names,
            extra_results_paths=[wc2026_path],
        )
        pipeline = build_initial_pipeline(matches, config, tournament.kickoff_date)
        predictor = load_calibrated_predictor(config, self.models_dir)

        benchmark_sim = KnockoutSimulator(
            predictor,
            config,
            pipeline,
            tournament,
            n_sims=requested_sims,
            seed=self.seed,
            show_progress=False,
        )
        n_sims, seconds_per_sim = self._plan_sim_count(benchmark_sim, requested_sims)

        sim = KnockoutSimulator(
            predictor,
            config,
            pipeline,
            tournament,
            n_sims=n_sims,
            seed=self.seed,
        )
        result = sim.run()
        payload = {
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
        }
        with self.forecast_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload, n_sims, seconds_per_sim


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified Kaggle pipeline")
    parser.add_argument(
        "--profile",
        choices=["fast", "kaggle"],
        default="fast",
        help="Training profile (fast=smoke test, kaggle=full run)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip completed phases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data root (flat Kaggle dataset or local test path)",
    )
    args = parser.parse_args()

    runner = PipelineRunner(
        profile=args.profile,
        resume=args.resume,
        seed=args.seed,
        project_root=PROJECT_ROOT,
        data_root=args.data_root,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
