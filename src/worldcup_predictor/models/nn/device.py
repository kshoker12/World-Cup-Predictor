"""PyTorch device resolution."""

from __future__ import annotations

from worldcup_predictor.config import NNConfig


def resolve_device(config: NNConfig):
    import os

    # macOS: LightGBM and PyTorch both link OpenMP; allow coexistence in one process.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch

    preference = config.device.lower()
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")
