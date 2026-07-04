"""Training loop for neural network models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from worldcup_predictor.config import NNConfig
from worldcup_predictor.models.nn.dataset import MatchSequenceDataset
from worldcup_predictor.models.nn.device import resolve_device
from worldcup_predictor.models.nn.loss import poisson_nll
from worldcup_predictor.models.nn.model import MatchSequenceLSTM
from worldcup_predictor.models.metrics import poisson_deviance
from worldcup_predictor.utils.progress import progress


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_run: int


def _run_epoch(
    model: MatchSequenceLSTM,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    show_progress: bool,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    iterator = progress(loader, desc=desc, disable=not show_progress)
    for home_seq, away_seq, y_home, y_away in iterator:
        home_seq = home_seq.to(device)
        away_seq = away_seq.to(device)
        y_home = y_home.to(device)
        y_away = y_away.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        lh, la = model(home_seq, away_seq)
        loss = poisson_nll(y_home, y_away, lh, la)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def evaluate_deviance(
    model: MatchSequenceLSTM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    y_h: list[float] = []
    y_a: list[float] = []
    lh: list[float] = []
    la: list[float] = []
    for home_seq, away_seq, home_y, away_y in loader:
        home_seq = home_seq.to(device)
        away_seq = away_seq.to(device)
        pred_h, pred_a = model(home_seq, away_seq)
        y_h.extend(home_y.numpy().tolist())
        y_a.extend(away_y.numpy().tolist())
        lh.extend(pred_h.cpu().numpy().tolist())
        la.extend(pred_a.cpu().numpy().tolist())
    return poisson_deviance(np.array(y_h), np.array(lh)) + poisson_deviance(
        np.array(y_a), np.array(la)
    )


def train_model(
    model: MatchSequenceLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: NNConfig,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    show_progress: bool = True,
    freeze_encoder_epochs: int = 0,
) -> TrainResult:
    device = resolve_device(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.weight_decay
    )

    best_val = float("inf")
    best_state = None
    stale = 0
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        if epoch < freeze_encoder_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True

        _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            desc=f"Train epoch {epoch + 1}/{epochs}",
            show_progress=show_progress,
        )
        val_loss = evaluate_deviance(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_val_loss=best_val, epochs_run=epochs_run)


def make_loaders(
    home_seq: np.ndarray,
    away_seq: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = MatchSequenceDataset(home_seq, away_seq, y_home, y_away)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
