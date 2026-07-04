"""Poisson loss for expected goals."""

from __future__ import annotations

import torch


def poisson_nll(
    y_home: torch.Tensor,
    y_away: torch.Tensor,
    lambda_home: torch.Tensor,
    lambda_away: torch.Tensor,
) -> torch.Tensor:
    lam_h = torch.clamp(lambda_home, min=1e-6)
    lam_a = torch.clamp(lambda_away, min=1e-6)
    loss = lam_h - y_home * torch.log(lam_h) + lam_a - y_away * torch.log(lam_a)
    return loss.mean()
