"""Bayesian model artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from worldcup_predictor.features.pipeline import FEATURE_COLUMNS


@dataclass(frozen=True)
class BayesianArtifacts:
    rho_mean: float
    rho_std: float
    intercept_mean: float
    n_matches: int
    n_teams: int
    team_index: dict[str, int]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    beta_mean: dict[str, float]
    att_mean: dict[str, float]
    def_mean: dict[str, float]
    chains: int
    draws: int
    tune: int
    rhat_rho: float | None = None
    ess_rho: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> BayesianArtifacts:
        att_raw = data.get("att_mean", {})
        def_raw = data.get("def_mean", {})
        return cls(
            rho_mean=float(data["rho_mean"]),
            rho_std=float(data["rho_std"]),
            intercept_mean=float(data.get("intercept_mean", 0.0)),
            n_matches=int(data["n_matches"]),
            n_teams=int(data["n_teams"]),
            team_index={str(k): int(v) for k, v in data["team_index"].items()},
            feature_means={str(k): float(v) for k, v in data["feature_means"].items()},
            feature_stds={str(k): float(v) for k, v in data["feature_stds"].items()},
            beta_mean={str(k): float(v) for k, v in data.get("beta_mean", {}).items()},
            att_mean={str(k): float(v) for k, v in att_raw.items()},
            def_mean={str(k): float(v) for k, v in def_raw.items()},
            chains=int(data.get("chains", 2)),
            draws=int(data.get("draws", 500)),
            tune=int(data.get("tune", 500)),
            rhat_rho=(
                float(data["rhat_rho"]) if data.get("rhat_rho") is not None else None
            ),
            ess_rho=(
                float(data["ess_rho"]) if data.get("ess_rho") is not None else None
            ),
        )

    @classmethod
    def load(cls, path: Path) -> BayesianArtifacts:
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_posterior_summary(
        cls,
        summary: dict,
        *,
        data,
        att_mean: dict[str, float],
        def_mean: dict[str, float],
        chains: int,
        draws: int,
        tune: int,
    ) -> BayesianArtifacts:
        rho_row = summary["rho"]
        beta_mean: dict[str, float] = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            key = f"beta[{i}]"
            if key not in summary:
                key = f"beta[{col}]"
            if key in summary:
                beta_mean[col] = float(summary[key]["mean"])
        feature_means = {
            col: float(data.feature_means[i]) for i, col in enumerate(FEATURE_COLUMNS)
        }
        feature_stds = {
            col: float(data.feature_stds[i]) for i, col in enumerate(FEATURE_COLUMNS)
        }
        return cls(
            rho_mean=float(rho_row["mean"]),
            rho_std=float(rho_row["sd"]),
            intercept_mean=float(summary["intercept"]["mean"]),
            n_matches=data.n_matches,
            n_teams=data.n_teams,
            team_index=dict(data.team_index),
            feature_means=feature_means,
            feature_stds=feature_stds,
            beta_mean=beta_mean,
            att_mean=att_mean,
            def_mean=def_mean,
            chains=chains,
            draws=draws,
            tune=tune,
            rhat_rho=float(rho_row["r_hat"]) if "r_hat" in rho_row else None,
            ess_rho=float(rho_row["ess_bulk"]) if "ess_bulk" in rho_row else None,
        )
