#!/usr/bin/env python3
"""Evaluate per-model and ensemble match metrics on the held-out test split.

Compares LightGBM, LSTM, Bayesian, and the calibrated ensemble on single-match
prediction quality. Writes a JSON report and prints a summary table.

Example:
    python scripts/evaluate_test_metrics.py --models-dir output
    python scripts/evaluate_test_metrics.py --models-dir output --output output/test_metrics.json

Requires:
    data/processed/features.parquet
    data/processed/intl_sequences.npz
    Trained artifacts in --models-dir (gbm_*.txt, nn_model.pt, bayesian.json, calibration.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import gammaln

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.artifacts import CalibrationArtifacts  # noqa: E402
from worldcup_predictor.calibration.ensemble import combine_lambda  # noqa: E402
from worldcup_predictor.config import load_profile  # noqa: E402
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS  # noqa: E402
from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts  # noqa: E402
from worldcup_predictor.models.bayesian.predictor import BayesianPredictor  # noqa: E402
from worldcup_predictor.models.gbm import GBMPredictor  # noqa: E402
from worldcup_predictor.models.metrics import evaluate_goals, evaluate_wdl  # noqa: E402
from worldcup_predictor.models.nn.embeddings import compute_embedding_diff  # noqa: E402
from worldcup_predictor.models.nn.predictor import NNPredictor  # noqa: E402

DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.parquet"
DEFAULT_SEQUENCES = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"
WDL_CHUNK_SIZE = 1024
NN_BATCH_SIZE = 512


def _log(msg: str) -> None:
    print(msg, flush=True)


def _progress(
    iterable,
    *,
    desc: str,
    total: int | None = None,
    disable: bool = False,
):
    """tqdm on stdout so progress is visible in non-TTY terminals."""
    from tqdm import tqdm

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=disable,
        file=sys.stdout,
    )


@dataclass(frozen=True)
class ModelMetrics:
    poisson_deviance_total: float
    poisson_deviance_home: float
    poisson_deviance_away: float
    goal_mae: float
    wdl_log_loss: float
    wdl_brier: float
    outcome_accuracy: float
    mean_log_prob_actual: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate match-level metrics on the test split."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory with gbm_*.txt, nn_model.pt, bayesian.json, calibration.json",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Feature parquet from build_features.py",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=DEFAULT_SEQUENCES,
        help="International sequences npz from build_nn_sequences.py",
    )
    parser.add_argument(
        "--profile",
        default="kaggle",
        help="Config profile overlay (default: kaggle for production artifacts)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "val"),
        help="Evaluation split (default: test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (default: <models-dir>/test_metrics.json)",
    )
    parser.add_argument(
        "--wdl-chunk-size",
        type=int,
        default=WDL_CHUNK_SIZE,
        help="Chunk size for vectorized WDL computation",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device for LSTM inference (default: cpu, avoids MPS hangs)",
    )
    parser.add_argument(
        "--nn-batch-size",
        type=int,
        default=NN_BATCH_SIZE,
        help="Batch size for LSTM forward passes",
    )
    return parser.parse_args()


def _require_paths(args: argparse.Namespace) -> None:
    missing: list[str] = []
    for label, path in (
        ("features", args.features),
        ("sequences", args.sequences),
        ("models dir", args.models_dir),
    ):
        if not path.exists():
            missing.append(f"{label}: {path}")
    required_models = (
        "gbm_home.txt",
        "gbm_away.txt",
        "gbm_meta.json",
        "calibration.json",
    )
    for name in required_models:
        path = args.models_dir / name
        if not path.exists():
            missing.append(f"model artifact: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n  " + "\n  ".join(missing)
        )


def _load_split_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    _log(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features).reset_index(drop=True)
    split_df = df[df["split"] == args.split].copy()
    if split_df.empty:
        raise ValueError(f"No rows for split={args.split!r}")

    _log(f"Loading sequences from {args.sequences}")
    seq = np.load(args.sequences)
    split_idx = np.where(df["split"].values == args.split)[0]
    home_seq = seq["home_seq"][split_idx]
    away_seq = seq["away_seq"][split_idx]

    train_df = df[df["split"] == "train"]
    _log(
        f"Split {args.split}: {len(split_df):,} matches "
        f"(train reference n={len(train_df):,})"
    )
    return split_df, home_seq, away_seq, train_df


def _load_gbm(models_dir: Path, config) -> GBMPredictor:
    _log(f"Loading LightGBM models from {models_dir} ...")
    gbm = GBMPredictor(config.gbm)
    gbm.load(models_dir)
    _log(
        f"LightGBM ready ({len(gbm.feature_columns)} features, "
        f"{gbm.model_home.num_trees()} home trees)"
    )
    return gbm


def _load_nn(models_dir: Path, config) -> NNPredictor:
    if not (models_dir / "nn_model.pt").exists():
        raise FileNotFoundError(f"Missing {models_dir / 'nn_model.pt'}")
    _log(f"Loading LSTM weights from {models_dir} (device={config.nn.device}) ...")
    nn = NNPredictor(config.nn)
    nn.load(models_dir)
    _log(f"LSTM ready on {nn.device}")
    return nn


def _load_bayesian(models_dir: Path) -> BayesianPredictor:
    path = models_dir / "bayesian.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    _log(f"Loading Bayesian artifacts from {path} ...")
    artifacts = BayesianArtifacts.load(path)
    if not artifacts.att_mean:
        raise ValueError("bayesian.json has no team attack/defense means")
    _log(f"Bayesian ready ({len(artifacts.att_mean)} teams)")
    return BayesianPredictor(artifacts)


def _predict_nn_batched(
    nn: NNPredictor,
    home_seq: np.ndarray,
    away_seq: np.ndarray,
    *,
    batch_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if nn.model is None:
        raise RuntimeError("NN model not loaded")

    model = nn.model
    model.eval()
    device = nn.device
    model.to(device)

    n = len(home_seq)
    lambda_home = np.zeros(n, dtype=np.float64)
    lambda_away = np.zeros(n, dtype=np.float64)
    batches = list(range(0, n, batch_size))
    for start in _progress(
        batches,
        desc="LSTM batches",
        total=len(batches),
        disable=not show_progress,
    ):
        end = min(start + batch_size, n)
        home_t = torch.as_tensor(home_seq[start:end], dtype=torch.float32, device=device)
        away_t = torch.as_tensor(away_seq[start:end], dtype=torch.float32, device=device)
        lh, la = model(home_t, away_t)
        lambda_home[start:end] = lh.detach().cpu().numpy()
        lambda_away[start:end] = la.detach().cpu().numpy()
    return lambda_home, lambda_away


def _gbm_feature_frame(
    gbm: GBMPredictor,
    nn: NNPredictor,
    features: pd.DataFrame,
    home_seq: np.ndarray,
    away_seq: np.ndarray,
    *,
    show_progress: bool,
) -> pd.DataFrame:
    emb_cols = [c for c in gbm.feature_columns if c.startswith("emb_diff_")]
    if not emb_cols:
        return features[gbm.feature_columns]

    _log("Computing LSTM embedding differences for GBM inputs...")
    emb = compute_embedding_diff(nn, home_seq, away_seq, batch_size=NN_BATCH_SIZE)
    emb_df = pd.DataFrame(
        emb[:, [int(c.removeprefix("emb_diff_")) for c in emb_cols]],
        columns=emb_cols,
        index=features.index,
    )
    base_cols = [c for c in FEATURE_COLUMNS if c in gbm.feature_columns]
    return pd.concat([features[base_cols], emb_df], axis=1)


def _predict_component_lambdas(
    *,
    gbm: GBMPredictor,
    nn: NNPredictor,
    bayesian: BayesianPredictor,
    artifacts: CalibrationArtifacts,
    features: pd.DataFrame,
    home_seq: np.ndarray,
    away_seq: np.ndarray,
    show_progress: bool,
    nn_batch_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    steps = [
        "LightGBM",
        "LSTM",
        "Bayesian",
        "Ensemble",
    ]
    for step in _progress(steps, desc="Predict lambdas", disable=not show_progress):
        if step == "LightGBM":
            gbm_features = _gbm_feature_frame(
                gbm, nn, features, home_seq, away_seq, show_progress=show_progress
            )
            raw = gbm.predict_lambda(gbm_features)
            lh, la = artifacts.scaling_gbm.apply(
                raw["lambda_home"].to_numpy(),
                raw["lambda_away"].to_numpy(),
            )
            preds[step] = (lh, la)
        elif step == "LSTM":
            lh_raw, la_raw = _predict_nn_batched(
                nn,
                home_seq,
                away_seq,
                batch_size=nn_batch_size,
                show_progress=show_progress,
            )
            lh, la = artifacts.scaling_nn.apply(lh_raw, la_raw)
            preds[step] = (lh, la)
        elif step == "Bayesian":
            raw = bayesian.predict_lambda(features)
            lh, la = artifacts.scaling_bayesian.apply(
                raw["lambda_home"].to_numpy(),
                raw["lambda_away"].to_numpy(),
            )
            preds[step] = (lh, la)
        elif step == "Ensemble":
            lh_gbm, la_gbm = preds["LightGBM"]
            lh_nn, la_nn = preds["LSTM"]
            lh_bayes, la_bayes = preds["Bayesian"]
            lh, la = combine_lambda(
                lh_gbm,
                la_gbm,
                artifacts.ensemble,
                lh_nn,
                la_nn,
                lh_bayes,
                la_bayes,
            )
            preds[step] = (lh, la)
    return preds


def _naive_lambdas(train_df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    mean_home = float(train_df["home_score"].mean())
    mean_away = float(train_df["away_score"].mean())
    return (
        np.full(n, mean_home, dtype=float),
        np.full(n, mean_away, dtype=float),
    )


def _batch_wdl_probabilities(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    *,
    max_goals: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Dixon-Coles W/D/L probabilities for many matches."""
    lh = np.clip(np.asarray(lambda_home, dtype=float), 1e-9, None)
    la = np.clip(np.asarray(lambda_away, dtype=float), 1e-9, None)
    goals = np.arange(max_goals + 1, dtype=float)
    log_ph = (
        -lh[:, None]
        + goals[None, :] * np.log(lh)[:, None]
        - gammaln(goals[None, :] + 1.0)
    )
    log_pa = (
        -la[:, None]
        + goals[None, :] * np.log(la)[:, None]
        - gammaln(goals[None, :] + 1.0)
    )
    log_grid = log_ph[:, :, None] + log_pa[:, None, :]
    grid = np.exp(log_grid)

    if rho != 0.0:
        tau = np.ones_like(grid)
        tau[:, 0, 0] = 1.0 - lh * la * rho
        tau[:, 0, 1] = 1.0 + lh * rho
        tau[:, 1, 0] = 1.0 + la * rho
        tau[:, 1, 1] = 1.0 - rho
        grid = grid * np.maximum(tau, 1e-12)

    grid /= grid.sum(axis=(1, 2), keepdims=True)
    p_draw = np.einsum("nii->n", grid)
    p_home = np.tril(grid, k=-1).sum(axis=(1, 2))
    p_away = np.triu(grid, k=1).sum(axis=(1, 2))
    return p_home, p_draw, p_away


def _wdl_probabilities_chunked(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    *,
    max_goals: int,
    rho: float,
    chunk_size: int,
    show_progress: bool,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(lambda_home)
    p_home = np.empty(n, dtype=float)
    p_draw = np.empty(n, dtype=float)
    p_away = np.empty(n, dtype=float)
    starts = range(0, n, chunk_size)
    for start in _progress(
        list(starts),
        desc=desc,
        disable=not show_progress,
    ):
        end = min(start + chunk_size, n)
        ph, pd_, pa = _batch_wdl_probabilities(
            lambda_home[start:end],
            lambda_away[start:end],
            max_goals=max_goals,
            rho=rho,
        )
        p_home[start:end] = ph
        p_draw[start:end] = pd_
        p_away[start:end] = pa
    return p_home, p_draw, p_away


def _outcome_labels(y_home: np.ndarray, y_away: np.ndarray) -> np.ndarray:
    y_h = np.asarray(y_home, dtype=int)
    y_a = np.asarray(y_away, dtype=int)
    return np.where(y_h > y_a, 0, np.where(y_h == y_a, 1, 2))


def _compute_metrics(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    *,
    max_goals: int,
    rho: float,
    chunk_size: int,
    show_progress: bool,
    label: str,
) -> ModelMetrics:
    goal_metrics = evaluate_goals(y_home, y_away, lambda_home, lambda_away)
    p_home, p_draw, p_away = _wdl_probabilities_chunked(
        lambda_home,
        lambda_away,
        max_goals=max_goals,
        rho=rho,
        chunk_size=chunk_size,
        show_progress=show_progress,
        desc=f"WDL {label}",
    )
    wdl_metrics = evaluate_wdl(y_home, y_away, p_home, p_draw, p_away)

    outcomes = _outcome_labels(y_home, y_away)
    probs = np.stack([p_home, p_draw, p_away], axis=1)
    pred_class = probs.argmax(axis=1)
    actual_prob = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-15, 1.0)

    return ModelMetrics(
        poisson_deviance_total=goal_metrics["poisson_deviance_total"],
        poisson_deviance_home=goal_metrics["poisson_deviance_home"],
        poisson_deviance_away=goal_metrics["poisson_deviance_away"],
        goal_mae=0.5 * (goal_metrics["mae_home"] + goal_metrics["mae_away"]),
        wdl_log_loss=wdl_metrics["wdl_log_loss"],
        wdl_brier=wdl_metrics["wdl_brier"],
        outcome_accuracy=float(np.mean(pred_class == outcomes)),
        mean_log_prob_actual=float(np.mean(np.log(actual_prob))),
    )


def _improvement_pct(baseline: float, improved: float, *, lower_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    if lower_is_better:
        return 100.0 * (baseline - improved) / baseline
    return 100.0 * (improved - baseline) / baseline


def _best_single_model(
    results: dict[str, ModelMetrics],
) -> tuple[str, ModelMetrics]:
    candidates = {
        name: metrics
        for name, metrics in results.items()
        if name not in {"Ensemble", "NaiveMean"}
    }
    best_name = min(
        candidates,
        key=lambda name: candidates[name].poisson_deviance_total,
    )
    return best_name, candidates[best_name]


def _print_table(results: dict[str, ModelMetrics]) -> None:
    headers = (
        "Model",
        "PoissonDev",
        "GoalMAE",
        "WDL_LL",
        "WDL_Brier",
        "OutcomeAcc",
        "LogProb",
    )
    print()
    print(
        f"{headers[0]:<12} | {headers[1]:>10} | {headers[2]:>8} | "
        f"{headers[3]:>8} | {headers[4]:>9} | {headers[5]:>10} | {headers[6]:>8}"
    )
    print("-" * 88)
    for name, m in results.items():
        print(
            f"{name:<12} | {m.poisson_deviance_total:10.4f} | {m.goal_mae:8.4f} | "
            f"{m.wdl_log_loss:8.4f} | {m.wdl_brier:9.4f} | "
            f"{m.outcome_accuracy:10.3%} | {m.mean_log_prob_actual:8.4f}"
        )
    print()
    print("Lower is better for PoissonDev, GoalMAE, WDL_LL, WDL_Brier.")
    print("Higher is better for OutcomeAcc and LogProb (mean log-prob of actual W/D/L).")


def _print_comparison(best_name: str, best: ModelMetrics, ensemble: ModelMetrics) -> None:
    print(f"Best single model on Poisson deviance: {best_name}")
    print(
        "Ensemble vs best single "
        f"(Poisson deviance): "
        f"{_improvement_pct(best.poisson_deviance_total, ensemble.poisson_deviance_total, lower_is_better=True):+.2f}%"
    )
    print(
        "Ensemble vs best single (WDL log loss): "
        f"{_improvement_pct(best.wdl_log_loss, ensemble.wdl_log_loss, lower_is_better=True):+.2f}%"
    )
    print(
        "Ensemble vs best single (outcome accuracy): "
        f"{_improvement_pct(best.outcome_accuracy, ensemble.outcome_accuracy, lower_is_better=False):+.2f}%"
    )
    print(
        "Ensemble vs best single (mean log-prob actual): "
        f"{_improvement_pct(best.mean_log_prob_actual, ensemble.mean_log_prob_actual, lower_is_better=False):+.2f}%"
    )


def main() -> int:
    args = _parse_args()
    show_progress = not args.no_progress
    _require_paths(args)

    config = load_profile(args.profile)
    config = replace(config, nn=replace(config.nn, device=args.device))
    _log(f"Using LSTM device: {config.nn.device}")

    _log(f"Loading calibration from {args.models_dir / 'calibration.json'} ...")
    artifacts = CalibrationArtifacts.load(args.models_dir / "calibration.json")
    rho = artifacts.dixon_coles.rho
    max_goals = config.simulation.max_goals
    _log(
        f"Calibration ready (rho={rho:.4f}, "
        f"weights GBM/LSTM/Bayes="
        f"{artifacts.ensemble.w_gbm:.2f}/"
        f"{artifacts.ensemble.w_nn:.2f}/"
        f"{artifacts.ensemble.w_bayesian:.2f})"
    )

    split_df, home_seq, away_seq, train_df = _load_split_data(args)
    y_home = split_df["home_score"].to_numpy()
    y_away = split_df["away_score"].to_numpy()

    gbm = _load_gbm(args.models_dir, config)
    nn = _load_nn(args.models_dir, config)
    bayesian = _load_bayesian(args.models_dir)

    _log("Running component predictions ...")
    lambda_preds = _predict_component_lambdas(
        gbm=gbm,
        nn=nn,
        bayesian=bayesian,
        artifacts=artifacts,
        features=split_df,
        home_seq=home_seq,
        away_seq=away_seq,
        show_progress=show_progress,
        nn_batch_size=args.nn_batch_size,
    )
    lambda_preds["NaiveMean"] = _naive_lambdas(train_df, len(split_df))

    results: dict[str, ModelMetrics] = {}
    metric_steps = list(lambda_preds.keys())
    for name in _progress(metric_steps, desc="Score metrics", disable=not show_progress):
        lh, la = lambda_preds[name]
        results[name] = _compute_metrics(
            y_home,
            y_away,
            lh,
            la,
            max_goals=max_goals,
            rho=rho,
            chunk_size=args.wdl_chunk_size,
            show_progress=show_progress,
            label=name,
        )

    best_name, best_metrics = _best_single_model(results)
    ensemble_metrics = results["Ensemble"]

    report = {
        "generated_at": _utc_now(),
        "split": args.split,
        "n_matches": int(len(split_df)),
        "models_dir": str(args.models_dir.resolve()),
        "profile": args.profile,
        "rho_ensemble": rho,
        "metric_definitions": {
            "poisson_deviance_total": (
                "Sum of mean Poisson deviances for home and away goals. "
                "Primary training loss; lower is better."
            ),
            "goal_mae": "Average absolute error between predicted λ and actual goals; lower is better.",
            "wdl_log_loss": (
                "Multiclass log loss for win/draw/loss from Dixon-Coles score grid; lower is better."
            ),
            "wdl_brier": "Multiclass Brier score for W/D/L probabilities; lower is better.",
            "outcome_accuracy": "Fraction of matches where argmax W/D/L equals the actual result.",
            "mean_log_prob_actual": (
                "Mean log probability assigned to the actual W/D/L outcome; higher is better."
            ),
        },
        "models": {name: metrics.to_dict() for name, metrics in results.items()},
        "ensemble_vs_best_single": {
            "best_single_model": best_name,
            "best_single_poisson_deviance": best_metrics.poisson_deviance_total,
            "ensemble_poisson_deviance": ensemble_metrics.poisson_deviance_total,
            "poisson_deviance_improvement_pct": _improvement_pct(
                best_metrics.poisson_deviance_total,
                ensemble_metrics.poisson_deviance_total,
                lower_is_better=True,
            ),
            "wdl_log_loss_improvement_pct": _improvement_pct(
                best_metrics.wdl_log_loss,
                ensemble_metrics.wdl_log_loss,
                lower_is_better=True,
            ),
            "outcome_accuracy_improvement_pct": _improvement_pct(
                best_metrics.outcome_accuracy,
                ensemble_metrics.outcome_accuracy,
                lower_is_better=False,
            ),
            "mean_log_prob_actual_improvement_pct": _improvement_pct(
                best_metrics.mean_log_prob_actual,
                ensemble_metrics.mean_log_prob_actual,
                lower_is_better=False,
            ),
        },
    }

    out_path = args.output or (args.models_dir / f"{args.split}_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_table(results)
    _print_comparison(best_name, best_metrics, ensemble_metrics)
    print(f"Saved report to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
