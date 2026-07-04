#!/usr/bin/env python3
"""Train NN: club pretrain then international fine-tune."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.nn.device import resolve_device  # noqa: E402
from worldcup_predictor.models.nn.predictor import NNPredictor  # noqa: E402

CLUB_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "club_sequences.npz"
INTL_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train neural network")
    parser.add_argument(
        "--phase",
        choices=["pretrain", "finetune", "all"],
        default="all",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    args = parser.parse_args()
    show_progress = not args.no_progress

    config = load_config()
    nn_overrides: dict[str, int | str] = {}
    if args.pretrain_epochs is not None:
        nn_overrides["pretrain_epochs"] = args.pretrain_epochs
    if args.finetune_epochs is not None:
        nn_overrides["finetune_epochs"] = args.finetune_epochs
    if args.device is not None:
        nn_overrides["device"] = args.device
    if nn_overrides:
        from dataclasses import replace

        config = replace(config, nn=replace(config.nn, **nn_overrides))
    nn = NNPredictor(config.nn)
    device = resolve_device(config.nn)
    print(f"Using device: {device}")

    if args.phase in ("pretrain", "all"):
        if not CLUB_SEQ_PATH.exists():
            print(f"ERROR: Missing {CLUB_SEQ_PATH}", file=sys.stderr)
            return 1
        data = np.load(CLUB_SEQ_PATH)
        result = nn.pretrain(
            data["home_seq"],
            data["away_seq"],
            data["y_home"],
            data["y_away"],
            show_progress=show_progress,
        )
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        nn.save_pretrain(MODELS_DIR)
        print(f"Pretrain complete: best val loss={result.best_val_loss:.4f}")

    if args.phase == "finetune":
        nn.load_pretrain(MODELS_DIR)
    elif args.phase == "all" and (MODELS_DIR / "nn_pretrain.pt").exists():
        pass

    if args.phase in ("finetune", "all"):
        if not INTL_SEQ_PATH.exists() or not FEATURES_PATH.exists():
            print("ERROR: Missing intl sequences or features.parquet", file=sys.stderr)
            return 1
        import pandas as pd

        features = pd.read_parquet(FEATURES_PATH).reset_index(drop=True)
        seq = np.load(INTL_SEQ_PATH)
        train_idx = np.where(features["split"].values == "train")[0]
        val_idx = np.where(features["split"].values == "val")[0]
        metrics = nn.finetune(
            seq["home_seq"],
            seq["away_seq"],
            features,
            train_idx,
            val_idx,
            show_progress=show_progress,
        )
        nn.save(
            MODELS_DIR,
            meta={
                "val_poisson_deviance_total": metrics.val["poisson_deviance_total"],
            },
        )
        print("Finetune validation metrics:")
        for k, v in sorted(metrics.val.items()):
            print(f"  {k}: {v:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
