def test_resolve_device_cpu():
    from worldcup_predictor.config import NNConfig
    from worldcup_predictor.models.nn.device import resolve_device

    device = resolve_device(NNConfig(device="cpu"))
    assert device.type == "cpu"


def test_lstm_forward():
    import torch

    from worldcup_predictor.models.nn.model import MatchSequenceLSTM

    model = MatchSequenceLSTM(input_dim=10, hidden_dim=8, num_layers=1, dropout=0.0)
    home = torch.randn(4, 10, 10)
    away = torch.randn(4, 10, 10)
    lh, la = model(home, away)
    assert lh.shape == (4,)
    assert la.shape == (4,)
    assert (lh > 0).all()
    assert (la > 0).all()


def test_poisson_nll_finite():
    import torch

    from worldcup_predictor.models.nn.loss import poisson_nll

    y = torch.tensor([1.0, 0.0])
    lam = torch.tensor([1.2, 0.8])
    loss = poisson_nll(y, y, lam, lam)
    assert torch.isfinite(loss)
