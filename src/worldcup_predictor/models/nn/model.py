"""LSTM sequence model for match goal rates."""

from __future__ import annotations

import torch
from torch import nn


class MatchSequenceLSTM(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequences)
        return hidden[-1]

    def forward(
        self,
        home_seq: torch.Tensor,
        away_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        home_emb = self.encode(home_seq)
        away_emb = self.encode(away_seq)
        logits = self.head(torch.cat([home_emb, away_emb], dim=-1))
        rates = torch.nn.functional.softplus(logits)
        return rates[:, 0], rates[:, 1]
