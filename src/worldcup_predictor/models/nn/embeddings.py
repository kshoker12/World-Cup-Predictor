"""Export NN encoder embeddings for GBM feature augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from worldcup_predictor.models.nn.predictor import NNPredictor

EMBEDDING_PREFIX = "emb_diff_"


def embedding_column_names(hidden_dim: int) -> list[str]:
    return [f"{EMBEDDING_PREFIX}{i}" for i in range(hidden_dim)]


@torch.no_grad()
def compute_embedding_diff(
    nn_predictor: NNPredictor,
    home_seq: np.ndarray,
    away_seq: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Return home-minus-away encoder embeddings, shape (n, hidden_dim)."""
    if nn_predictor.model is None:
        raise RuntimeError("NN model must be trained before exporting embeddings")

    model = nn_predictor.model
    model.eval()
    device = nn_predictor.device
    model.to(device)

    n = len(home_seq)
    hidden_dim = nn_predictor.nn_config.hidden_dim
    output = np.zeros((n, hidden_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        home_t = torch.as_tensor(home_seq[start:end], dtype=torch.float32, device=device)
        away_t = torch.as_tensor(away_seq[start:end], dtype=torch.float32, device=device)
        home_emb = model.encode(home_t)
        away_emb = model.encode(away_t)
        diff = (home_emb - away_emb).cpu().numpy()
        output[start:end] = diff
    return output


def augment_features_with_embeddings(
    features: pd.DataFrame,
    home_seq: np.ndarray,
    away_seq: np.ndarray,
    nn_predictor: NNPredictor,
) -> pd.DataFrame:
    """Append emb_diff_* columns aligned row-wise with features."""
    emb = compute_embedding_diff(nn_predictor, home_seq, away_seq)
    cols = embedding_column_names(emb.shape[1])
    emb_df = pd.DataFrame(emb, columns=cols, index=features.index)
    return pd.concat([features, emb_df], axis=1)


def gbm_feature_columns_with_embeddings(hidden_dim: int) -> list[str]:
    from worldcup_predictor.features.pipeline import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS) + embedding_column_names(hidden_dim)
