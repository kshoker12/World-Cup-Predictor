"""Neural network predictor for lambda home/away."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from worldcup_predictor.config import NNConfig
from worldcup_predictor.models.metrics import evaluate_goals
from worldcup_predictor.models.nn.dataset import MatchSequenceDataset
from worldcup_predictor.models.nn.device import resolve_device
from worldcup_predictor.models.nn.model import MatchSequenceLSTM
from worldcup_predictor.models.nn.trainer import TrainResult, make_loaders, train_model


@dataclass
class NNMetrics:
    train: dict[str, float]
    val: dict[str, float]


class NNPredictor:
    def __init__(self, nn_config: NNConfig) -> None:
        self.nn_config = nn_config
        self.model: MatchSequenceLSTM | None = None
        self.device = resolve_device(nn_config)

    def _build_model(self) -> MatchSequenceLSTM:
        return MatchSequenceLSTM(
            input_dim=self.nn_config.feature_dim,
            hidden_dim=self.nn_config.hidden_dim,
            num_layers=self.nn_config.num_layers,
            dropout=self.nn_config.dropout,
        )

    def pretrain(
        self,
        home_seq: np.ndarray,
        away_seq: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        val_fraction: float = 0.1,
        *,
        show_progress: bool = True,
    ) -> TrainResult:
        n = len(y_home)
        split = max(1, int(n * (1 - val_fraction)))
        self.model = self._build_model()
        train_loader = make_loaders(
            home_seq[:split],
            away_seq[:split],
            y_home[:split],
            y_away[:split],
            batch_size=self.nn_config.pretrain_batch_size,
            num_workers=self.nn_config.num_workers,
        )
        val_loader = DataLoader(
            MatchSequenceDataset(
                home_seq[split:],
                away_seq[split:],
                y_home[split:],
                y_away[split:],
            ),
            batch_size=self.nn_config.pretrain_batch_size,
            shuffle=False,
            num_workers=self.nn_config.num_workers,
        )
        return train_model(
            self.model,
            train_loader,
            val_loader,
            self.nn_config,
            epochs=self.nn_config.pretrain_epochs,
            lr=self.nn_config.pretrain_lr,
            batch_size=self.nn_config.pretrain_batch_size,
            patience=self.nn_config.early_stopping_patience,
            show_progress=show_progress,
        )

    def finetune(
        self,
        home_seq: np.ndarray,
        away_seq: np.ndarray,
        features_df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        *,
        show_progress: bool = True,
    ) -> NNMetrics:
        if self.model is None:
            self.model = self._build_model()

        train_loader = make_loaders(
            home_seq[train_idx],
            away_seq[train_idx],
            features_df.iloc[train_idx]["home_score"].to_numpy(dtype=np.float32),
            features_df.iloc[train_idx]["away_score"].to_numpy(dtype=np.float32),
            batch_size=self.nn_config.finetune_batch_size,
            num_workers=self.nn_config.num_workers,
        )
        val_loader = DataLoader(
            MatchSequenceDataset(
                home_seq[val_idx],
                away_seq[val_idx],
                features_df.iloc[val_idx]["home_score"].to_numpy(dtype=np.float32),
                features_df.iloc[val_idx]["away_score"].to_numpy(dtype=np.float32),
            ),
            batch_size=self.nn_config.finetune_batch_size,
            shuffle=False,
            num_workers=self.nn_config.num_workers,
        )
        train_model(
            self.model,
            train_loader,
            val_loader,
            self.nn_config,
            epochs=self.nn_config.finetune_epochs,
            lr=self.nn_config.finetune_lr,
            batch_size=self.nn_config.finetune_batch_size,
            patience=self.nn_config.early_stopping_patience,
            show_progress=show_progress,
            freeze_encoder_epochs=self.nn_config.freeze_encoder_epochs,
        )
        train_pred = self.predict_lambda_from_sequences(
            home_seq[train_idx], away_seq[train_idx]
        )
        val_pred = self.predict_lambda_from_sequences(
            home_seq[val_idx], away_seq[val_idx]
        )
        return NNMetrics(
            train=evaluate_goals(
                features_df.iloc[train_idx]["home_score"].to_numpy(),
                features_df.iloc[train_idx]["away_score"].to_numpy(),
                train_pred["lambda_home"].to_numpy(),
                train_pred["lambda_away"].to_numpy(),
            ),
            val=evaluate_goals(
                features_df.iloc[val_idx]["home_score"].to_numpy(),
                features_df.iloc[val_idx]["away_score"].to_numpy(),
                val_pred["lambda_home"].to_numpy(),
                val_pred["lambda_away"].to_numpy(),
            ),
        )

    @torch.no_grad()
    def predict_lambda_from_sequences(
        self,
        home_seq: np.ndarray,
        away_seq: np.ndarray,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        self.model.eval()
        device = self.device
        self.model.to(device)
        home_t = torch.as_tensor(home_seq, dtype=torch.float32, device=device)
        away_t = torch.as_tensor(away_seq, dtype=torch.float32, device=device)
        lh, la = self.model(home_t, away_t)
        return pd.DataFrame(
            {
                "lambda_home": lh.cpu().numpy(),
                "lambda_away": la.cpu().numpy(),
            }
        )

    def predict_lambda(
        self,
        features: pd.DataFrame,
        home_seq: np.ndarray,
        away_seq: np.ndarray,
    ) -> pd.DataFrame:
        pred = self.predict_lambda_from_sequences(home_seq, away_seq)
        if len(pred) != len(features):
            raise ValueError("Sequence batch size must match features rows")
        return pred.reset_index(drop=True)

    def evaluate(
        self,
        features: pd.DataFrame,
        home_seq: np.ndarray,
        away_seq: np.ndarray,
    ) -> dict[str, float]:
        pred = self.predict_lambda(features, home_seq, away_seq)
        return evaluate_goals(
            features["home_score"].to_numpy(),
            features["away_score"].to_numpy(),
            pred["lambda_home"].to_numpy(),
            pred["lambda_away"].to_numpy(),
        )

    def save(self, directory: Path, *, meta: dict | None = None) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "nn_model.pt")
        torch.save(self.model.state_dict(), directory / "nn_pretrain.pt")
        payload = {
            "feature_dim": self.nn_config.feature_dim,
            "hidden_dim": self.nn_config.hidden_dim,
            "num_layers": self.nn_config.num_layers,
            "dropout": self.nn_config.dropout,
            "seq_len": self.nn_config.seq_len,
        }
        if meta:
            payload.update(meta)
        with (directory / "nn_meta.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_pretrain(self, directory: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "nn_pretrain.pt")

    def load(self, directory: Path) -> None:
        from dataclasses import replace

        meta_path = directory / "nn_meta.json"
        nn_config = self.nn_config
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            overrides = {
                k: meta[k]
                for k in ("feature_dim", "hidden_dim", "num_layers", "dropout", "seq_len")
                if k in meta
            }
            nn_config = replace(nn_config, **overrides)
            self.nn_config = nn_config
        self.model = self._build_model()
        weights_path = directory / "nn_model.pt"
        if not weights_path.exists():
            weights_path = directory / "nn_pretrain.pt"
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)

    def load_pretrain(self, directory: Path) -> None:
        self.model = self._build_model()
        state = torch.load(
            directory / "nn_pretrain.pt", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state)
