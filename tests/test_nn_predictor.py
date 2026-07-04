import numpy as np
import pytest


def test_nn_save_load_roundtrip(default_config, tmp_path):
    pytest.importorskip("torch")
    import torch

    from worldcup_predictor.models.nn.model import MatchSequenceLSTM
    from worldcup_predictor.models.nn.predictor import NNPredictor

    nn = NNPredictor(default_config.nn)
    nn.model = MatchSequenceLSTM(
        input_dim=default_config.nn.feature_dim,
        hidden_dim=default_config.nn.hidden_dim,
        num_layers=default_config.nn.num_layers,
        dropout=default_config.nn.dropout,
    )
    nn.model.to(nn.device)
    nn.save(tmp_path)

    loaded = NNPredictor(default_config.nn)
    loaded.load(tmp_path)

    home = np.random.randn(3, 10, 10).astype(np.float32)
    away = np.random.randn(3, 10, 10).astype(np.float32)
    pred = loaded.predict_lambda_from_sequences(home, away)
    assert len(pred) == 3
    assert np.all(pred["lambda_home"] > 0)
    assert np.all(pred["lambda_away"] > 0)
