#!/usr/bin/env python3
"""Validate Bayesian model artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.artifacts import (  # noqa: E402
    combine_from_artifacts,
    fit_calibration,
)
from worldcup_predictor.calibration.dixon_coles import fit_dixon_coles_rho
from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts  # noqa: E402
from worldcup_predictor.models.gbm import GBMPredictor  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
INTL_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
BAYESIAN_PATH = MODELS_DIR / "bayesian.json"


class ValidationError(Exception):
    pass


def _load_nn_if_available(config):
    nn_path = MODELS_DIR / "nn_model.pt"
    if not nn_path.exists():
        return None, None, None
    from worldcup_predictor.models.nn.predictor import NNPredictor

    nn = NNPredictor(config.nn)
    nn.load(MODELS_DIR)
    seq = np.load(INTL_SEQ_PATH)
    return nn, seq["home_seq"], seq["away_seq"]


def main() -> int:
    if not BAYESIAN_PATH.exists():
        print(f"ERROR: Missing {BAYESIAN_PATH}. Run train_bayesian.py first.", file=sys.stderr)
        return 1

    config = load_config()
    artifacts = BayesianArtifacts.load(BAYESIAN_PATH)

    try:
        if artifacts.n_matches < 100:
            raise ValidationError(
                f"Too few matches in Bayesian fit: {artifacts.n_matches}"
            )
        if artifacts.n_teams < 10:
            raise ValidationError(
                f"Too few teams in Bayesian fit: {artifacts.n_teams}"
            )
        if not (
            config.calibration.rho_min
            <= artifacts.rho_mean
            <= config.calibration.rho_max
        ):
            raise ValidationError(
                f"rho_mean={artifacts.rho_mean} outside "
                f"[{config.calibration.rho_min}, {config.calibration.rho_max}]"
            )
        if artifacts.rhat_rho is not None and artifacts.rhat_rho > 1.1:
            raise ValidationError(
                f"R-hat for rho={artifacts.rhat_rho:.4f} > 1.1 (poor convergence)"
            )
        if artifacts.ess_rho is not None and artifacts.ess_rho < 100:
            raise ValidationError(
                f"ESS for rho={artifacts.ess_rho:.1f} < 100"
            )

        if not artifacts.att_mean or not artifacts.def_mean:
            raise ValidationError("att_mean/def_mean missing; re-run train_bayesian.py")

        from worldcup_predictor.models.bayesian.predictor import BayesianPredictor

        if FEATURES_PATH.exists():
            df = pd.read_parquet(FEATURES_PATH).reset_index(drop=True)
            val_df = df[df["split"] == "val"].head(5)
            pred = BayesianPredictor(artifacts).predict_lambda(val_df)
            if (pred["lambda_home"] <= 0).any() or (pred["lambda_away"] <= 0).any():
                raise ValidationError("BayesianPredictor produced non-positive lambda")

        if FEATURES_PATH.exists() and (MODELS_DIR / "gbm_home.txt").exists():
            df = pd.read_parquet(FEATURES_PATH).reset_index(drop=True)
            val_df = df[df["split"] == "val"]
            val_idx = np.where(df["split"].values == "val")[0]

            gbm = GBMPredictor(config.gbm)
            gbm.load(MODELS_DIR)
            nn, home_seq, away_seq = _load_nn_if_available(config)

            cal_artifacts = fit_calibration(
                gbm,
                val_df,
                config.calibration,
                config,
                max_goals=config.simulation.max_goals,
                nn=nn,
                val_home_seq=home_seq[val_idx] if home_seq is not None else None,
                val_away_seq=away_seq[val_idx] if away_seq is not None else None,
                bayesian=artifacts,
                show_progress=False,
            )
            lh_gbm, la_gbm = cal_artifacts.scaling_gbm.apply(
                gbm.predict_lambda(val_df)["lambda_home"].to_numpy(),
                gbm.predict_lambda(val_df)["lambda_away"].to_numpy(),
            )
            lh_nn = la_nn = None
            if nn is not None and cal_artifacts.ensemble.w_nn > 0:
                nn_raw = nn.predict_lambda(
                    val_df,
                    home_seq[val_idx],
                    away_seq[val_idx],
                )
                lh_nn, la_nn = cal_artifacts.scaling_nn.apply(
                    nn_raw["lambda_home"].to_numpy(),
                    nn_raw["lambda_away"].to_numpy(),
                )
            lh_bayes = la_bayes = None
            if cal_artifacts.ensemble.w_bayesian > 0:
                from worldcup_predictor.models.bayesian.predictor import BayesianPredictor

                bayes_raw = BayesianPredictor(artifacts).predict_lambda(val_df)
                lh_bayes, la_bayes = cal_artifacts.scaling_bayesian.apply(
                    bayes_raw["lambda_home"].to_numpy(),
                    bayes_raw["lambda_away"].to_numpy(),
                )
            lh, la = combine_from_artifacts(
                lh_gbm,
                la_gbm,
                cal_artifacts.ensemble,
                lh_nn,
                la_nn,
                lh_bayes,
                la_bayes,
            )
            mle_rho = fit_dixon_coles_rho(
                val_df["home_score"].to_numpy(),
                val_df["away_score"].to_numpy(),
                lh,
                la,
                rho_min=config.calibration.rho_min,
                rho_max=config.calibration.rho_max,
                max_goals=config.simulation.max_goals,
            )
            print("Comparison (informational):")
            print(f"  Bayesian rho_mean: {artifacts.rho_mean:.4f}")
            print(f"  Standalone MLE rho: {mle_rho.rho:.4f}")

    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("All Bayesian validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
