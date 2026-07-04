def test_device_auto_fallback():
    from worldcup_predictor.config import NNConfig
    from worldcup_predictor.models.nn.device import resolve_device

    device = resolve_device(NNConfig(device="auto"))
    assert device.type in {"cpu", "mps", "cuda"}
