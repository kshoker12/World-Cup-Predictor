"""Dataset wrappers for sequence training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MatchSequenceDataset(Dataset):
    def __init__(
        self,
        home_seq: np.ndarray,
        away_seq: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
    ) -> None:
        self.home_seq = torch.as_tensor(home_seq, dtype=torch.float32)
        self.away_seq = torch.as_tensor(away_seq, dtype=torch.float32)
        self.y_home = torch.as_tensor(y_home, dtype=torch.float32)
        self.y_away = torch.as_tensor(y_away, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_home)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.home_seq[idx],
            self.away_seq[idx],
            self.y_home[idx],
            self.y_away[idx],
        )
