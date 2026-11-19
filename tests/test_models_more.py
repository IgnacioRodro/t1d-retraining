"""Models added in iteration 19: ARX with forgetting, more networks, foundation adapters and the long context."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from t1dretrain.models.arx import RecursiveARX  # noqa: E402
from t1dretrain.models.base import EXOG, TrainData  # noqa: E402
from t1dretrain.models.reference import Persistence  # noqa: E402
from t1dretrain.sim.samples import make_samples  # noqa: E402


def grid_of(values, break_at=None):
    n = len(values)
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) * 5, unit="min")
    hours = (ts - ts.normalize()).total_seconds() / 3600
    grid = pd.DataFrame({"patient_id": "p", "slot": np.arange(n), "ts": ts, "glucose": values, "target_ok": True,
                         "glucose_input": values, "segment": 0, "iob": 1.0, "cob": 0.0, "insulin_activity": 0.5,
                         "carb_absorption": 0.0, "hour_sin": np.sin(2 * np.pi * hours / 24),
                         "hour_cos": np.cos(2 * np.pi * hours / 24)})
    if break_at is not None:
        grid.loc[break_at:, "segment"] = 1
    grid["day"] = grid["ts"].dt.normalize()
    return grid


def test_arx_uses_the_trend_and_the_level():
    rng = np.random.default_rng(0)
    slope = np.zeros(20 * 288)
    y = np.zeros(20 * 288)
    y[0] = 150
    for t in range(1, len(y)):  # mean-reverting glucose with momentum
        slope[t] = 0.9 * slope[t - 1] + rng.normal(0, 0.8) - 0.002 * (y[t - 1] - 150)
        y[t] = y[t - 1] + slope[t]
    grid = grid_of(y)
    days = sorted(grid["day"].unique())
    common = dict(horizons_min=[30, 60], lookback_h=2, exog_columns=EXOG)
    train = make_samples(grid, stride_min=5, allowed_days=set(days[:14]), **common)
    test = make_samples(grid, stride_min=30, allowed_days=set(days[14:]), **common)
    model = RecursiveARX(0.999)
    model.fit(TrainData(train, np.zeros(0), np.zeros(0), 5.0))
    error = lambda p: np.sqrt(np.nanmean((p - test.targets) ** 2, axis=0))  # noqa: E731
    assert (error(model.predict(test)) < 0.9 * error(Persistence().predict(test))).all()


def test_long_context_does_not_change_the_origins():
    values = 120 + 10 * np.sin(np.arange(3 * 288) / 30)
    grid = grid_of(values, break_at=400)
    plain = make_samples(grid, [30], lookback_h=1, stride_min=30)
    long = make_samples(grid, [30], lookback_h=1, stride_min=30, context_h=24, context_exog=("iob",))
    assert (plain.origin_slot == long.origin_slot).all()
    assert np.allclose(long.context[:, -1], long.inputs[:, -1])                    # ends at the origin
    first = long.origin_slot[long.origin_slot >= 400][0]
    row = long.context[long.origin_slot == first][0]
    assert np.isnan(row[: len(row) - (first - 400) - 1]).all()                     # nothing before its segment
    assert long.context_exog["iob"].shape == long.context.shape


@pytest.mark.parametrize("architecture", ["tide", "tsmixerx", "lstm", "patchtst"])
def test_more_networks_train_save_and_reload(architecture, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from test_learned import pool_data

    from t1dretrain.models.neural import SCHEDULE, NeuralForecastModel

    s, held = pool_data(3, days=4)
    model = NeuralForecastModel(architecture, 1, {"mlp_units": 32, "max_steps": 20, "val_every": 10})
    model.train_on(s, held, {k: model.params[k] for k in SCHEDULE})
    model.save(tmp_path / "n.pt")
    again = NeuralForecastModel(architecture, 1, start="population", population=tmp_path / "n.pt")
    again.fit(TrainData(s, np.zeros(0), np.zeros(0), s.step_min))
    test = s.subset(held)
    assert np.allclose(again.predict(test), model.predict(test), atol=1e-4)
    assert again.name == f"{architecture}_pop" and len(again.head_parameters()) > 0


def test_foundation_adapters_with_random_weights():
    pytest.importorskip("chronos")
    pytest.importorskip("tsfm_public")
    from chronos import Chronos2Pipeline
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction

    from t1dretrain.models.foundation import Chronos2Backend, FoundationModel, TTMBackend

    grid = grid_of(140 + 30 * np.sin(np.arange(3 * 288) / 40))
    s = make_samples(grid, [30, 60], lookback_h=2, stride_min=120, context_h=24, context_exog=("iob", "cob"))
    config = Chronos2CoreConfig(d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4, chronos_config={
        "context_length": 512, "output_patch_size": 16, "input_patch_size": 16, "input_patch_stride": 16,
        "quantiles": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], "max_output_patches": 2,
        "use_reg_token": True})
    pipeline = Chronos2Pipeline(Chronos2Model(config).eval())
    ttm = TinyTimeMixerForPrediction(TinyTimeMixerConfig(context_length=128, prediction_length=12,
                                                         num_input_channels=1)).eval()
    ttm_freq = TinyTimeMixerForPrediction(TinyTimeMixerConfig(  # r2 checkpoints that expect a frequency token
        context_length=128, prediction_length=12, num_input_channels=1, resolution_prefix_tuning=True)).eval()
    for model in (FoundationModel("chronos2", 24, backend=Chronos2Backend(pipeline, True, 32)),
                  FoundationModel("chronos2_univariate", 24, backend=Chronos2Backend(pipeline, False, 32)),
                  FoundationModel("ttm", 24, backend=TTMBackend(ttm, "cpu", 32)),
                  FoundationModel("ttm", 24, backend=TTMBackend(ttm_freq, "cpu", 32))):
        out = model.predict(s)
        assert out.shape == (s.n, 2) and np.isfinite(out).all()
    with pytest.raises(ValueError):
        FoundationModel("ttm", 24).predict(make_samples(grid, [30], lookback_h=2, stride_min=120))  # no context


def test_a_cuda_failure_restarts_the_fit_on_the_cpu(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from test_learned import pool_data

    from t1dretrain.models import neural

    s, held = pool_data(5, days=3)
    original = neural.NeuralForecastModel._fit

    def flaky(self, train):  # the first attempt on a GPU fails as a broken CUDA context would
        if self.device != "cpu":
            raise RuntimeError("CUDA error: unspecified launch failure")
        return original(self, train)

    monkeypatch.setattr(neural.NeuralForecastModel, "_fit", flaky)
    monkeypatch.setitem(neural._CUDA_STATE, "broken", False)
    model = neural.NeuralForecastModel("nhits", 1, {"mlp_units": 16, "max_steps": 10, "val_every": 5})
    model.device = "cuda:0"
    log = model.fit(TrainData(s, np.zeros(0), np.zeros(0), s.step_min, validation=held))
    assert model.device == "cpu" and "unspecified launch failure" in log.details["cuda_error"]
    assert neural.NeuralForecastModel("tide", 1, device="cuda:1").device == "cpu"  # the process stays on the CPU
    monkeypatch.setitem(neural._CUDA_STATE, "broken", False)


def test_library_defects_are_corrected():
    """D74: TiDE's output reads the encoder and the exogenous inputs; the LSTM's first step reads the last input."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from t1dretrain.models.neural import DEFAULT_PARAMS, build_network, hist_channels

    hist, futr = hist_channels(True, True), ["hour_sin", "hour_cos"]
    torch.manual_seed(0)
    n, window, steps = 4, 72, 24
    y, exog, time = torch.randn(n, window, 1), torch.randn(n, window, len(hist)), torch.randn(n, window + steps, 2)

    def batch(y, exog):
        return {"insample_y": y, "insample_mask": torch.ones(n, window, 1), "hist_exog": exog, "futr_exog": time,
                "stat_exog": torch.zeros(n, 0)}

    tide = build_network("tide", window, steps, {**DEFAULT_PARAMS, "mlp_units": 32}, hist, futr, 1).eval()
    lstm = build_network("lstm", window, steps, {**DEFAULT_PARAMS, "mlp_units": 32}, hist, futr, 1).eval()
    late = y.clone()
    late[:, -1] += 2.0
    with torch.no_grad():
        assert (tide(batch(y, exog)) - tide(batch(y, 3 * exog))).abs().max() > 1e-3
        out = lstm(batch(y, exog))
        assert out.shape == (n, steps, 1)
        assert (out[:, 0] - lstm(batch(late, exog))[:, 0]).abs().max() > 1e-4


def test_networks_of_an_older_version_are_refused(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from test_learned import pool_data

    from t1dretrain.models.neural import NeuralForecastModel

    s, held = pool_data(4, days=3)
    model = NeuralForecastModel("tide", 1, {"mlp_units": 16, "max_steps": 5, "val_every": 5})
    model.train_on(s, held, {k: model.params[k] for k in ("learning_rate", "batch_size", "max_steps", "val_every",
                                                           "patience")})
    model.save(tmp_path / "tide.pt")
    saved = torch.load(tmp_path / "tide.pt", weights_only=False)
    saved["version"] = 1
    torch.save(saved, tmp_path / "old.pt")
    NeuralForecastModel("tide", 1, start="population", population=tmp_path / "tide.pt").load(tmp_path / "tide.pt")
    with pytest.raises(ValueError, match="--families tide"):
        NeuralForecastModel("tide", 1, start="population", population=tmp_path / "old.pt").load(tmp_path / "old.pt")


def test_longer_contexts_take_smaller_batches():
    """The batch size is set for the default 72 h of context; 14 days take proportionally fewer forecasts (script 43
    ran out of GPU memory with 256), shorter contexts keep it."""
    from t1dretrain.models.foundation import FoundationModel, scaled_batch

    assert scaled_batch(256, 72) == 256 and scaled_batch(256, 24) == 256
    assert scaled_batch(256, 168) == 109 and scaled_batch(256, 336) == 54 and scaled_batch(4, 10_000) == 1
    assert FoundationModel("chronos2", 336, batch_size=256).batch_size == 54
