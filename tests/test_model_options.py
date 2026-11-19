"""Model options of scripts 71 and 73: a loss that weighs the lows (D96) and extra inputs (the wearables, D98)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.models.arx import RecursiveARX  # noqa: E402
from t1dretrain.models.base import EXOG, TrainData, low_weights  # noqa: E402
from t1dretrain.sim.samples import make_samples, training_series  # noqa: E402
from t1dretrain.sim.stage0 import input_columns, make_model, model_options  # noqa: E402

ORIGIN = pd.Timestamp("2020-01-01")
DAYS = 20


def toy_grid(seed: int, lows: bool = True) -> pd.DataFrame:
    """A daily cycle that dips below 70 mg/dL every night, with steps that lower the glucose an hour later."""
    rng = np.random.default_rng(seed)
    n = DAYS * 288
    ts = ORIGIN + pd.to_timedelta(np.arange(n) * 5, unit="min")
    hours = (ts - ts.normalize()).total_seconds().to_numpy() / 3600
    steps = np.where(rng.random(n) < 0.05, rng.uniform(200, 900, n), 0.0)
    effect = np.convolve(steps, np.r_[np.zeros(12), np.full(12, 0.01)])[:n]
    glucose = 135 + (55 if lows else 30) * np.sin(2 * np.pi * hours / 24) - 3 * effect + rng.normal(0, 4, n)
    grid = pd.DataFrame({"patient_id": "p", "slot": np.arange(n), "ts": ts, "glucose": glucose, "target_ok": True,
                         "glucose_input": glucose, "segment": 0, "iob": 1.0, "cob": 0.0, "insulin_activity": 0.5,
                         "carb_absorption": 0.0, "hour_sin": np.sin(2 * np.pi * hours / 24),
                         "hour_cos": np.cos(2 * np.pi * hours / 24),
                         "steps_15min": pd.Series(steps).rolling(3, min_periods=1).sum().to_numpy(),
                         "asleep": 0.0})
    grid.loc[rng.random(n) < 0.2, "steps_15min"] = np.nan  # the device is not always worn
    grid["day"] = grid["ts"].dt.normalize()
    return grid


def data(grid: pd.DataFrame, extra: tuple = ()):
    days = sorted(grid["day"].unique())
    train_days, test_days = set(days[:14]), set(days[14:])
    common = dict(horizons_min=[30, 60], lookback_h=2, min_window_share=0.8, recent_complete_min=30,
                  exog_columns=EXOG + tuple(extra))
    train = make_samples(grid, stride_min=5, allowed_days=train_days, trajectory=True, **common)
    test = make_samples(grid, stride_min=30, allowed_days=test_days, **common)
    series, segment, step = training_series(grid, train_days)
    return TrainData(train, series, segment, step, validation=train.day >= np.datetime64(days[12])), test


def test_low_weights_and_options():
    w = low_weights(np.array([[60.0, 80.0], [np.nan, 69.9]]), {"threshold": 70, "factor": 5})
    assert w.tolist() == [[5.0, 1.0], [1.0, 5.0]] and low_weights(np.ones((2, 2)), None) is None
    assert model_options({}) == {} and model_options({"extra_inputs": ["asleep"], "low_loss": {"factor": 2,
                                                                                          "threshold": 70}}) == {
        "low_loss": {"factor": 2, "threshold": 70}, "extra": ("asleep",)}
    assert input_columns("persistence", {"extra_inputs": ["asleep"]}) == ()
    assert input_columns("lightgbm_ft", {"extra_inputs": ["asleep"]}) == EXOG + ("asleep",)
    with pytest.raises(ValueError, match="LightGBM and ARX only"):
        make_model("lstm_ft", 1, {"low_loss": {"threshold": 70, "factor": 5}, "population_dir": "x"}, "cpu")
    with pytest.raises(ValueError, match="no scale"):
        RecursiveARX(extra=("glucagon",))


def test_low_loss_lowers_the_forecasts_of_lows():
    train, test = data(toy_grid(1))
    low = test.targets[:, 0] < 70
    plain = RecursiveARX()
    plain.fit(train)
    weighted = RecursiveARX(low_loss={"threshold": 70, "factor": 20})
    weighted.fit(train)
    assert weighted.predict(test)[low, 0].mean() < plain.predict(test)[low, 0].mean()
    # the online update with the same weights is the batch fit on the samples known before each origin
    online, _ = weighted.online_predict(train.samples, test.subset(np.arange(5)))
    assert np.allclose(online, weighted.predict(test.subset(np.arange(5))), atol=0.05)


def test_lightgbm_on_top_of_the_population_equals_continuing_it(tmp_path):
    pytest.importorskip("lightgbm")
    from t1dretrain.models.gbdt import LightGBMDirect

    params = {"num_leaves": 7, "min_data_in_leaf": 30, "max_rounds": 40, "learning_rate": 0.1,
              "early_stopping_rounds": 10, "feature_fraction": 1.0}
    finetune = {"num_leaves": 7, "min_data_in_leaf": 20, "max_rounds": 30, "learning_rate": 0.05,
                "early_stopping_rounds": 5, "feature_fraction": 1.0}
    pool, _ = data(toy_grid(2), extra=("asleep", "steps_15min"))
    population = LightGBMDirect(1, params)
    population.fit(TrainData(pool.samples, None, None, 5.0, validation=pool.validation))
    population.save(tmp_path / "pop")
    train, test = data(toy_grid(3), extra=("asleep", "steps_15min"))
    continued = LightGBMDirect(1, params, "finetune", tmp_path / "pop", finetune)
    continued.fit(train)
    on_top = LightGBMDirect(1, params, "finetune", tmp_path / "pop", finetune, extra=("asleep",))  # constant input
    log = on_top.fit(train)
    assert on_top.personal and log.details["rounds"] == continued.fit(train).details["rounds"]
    assert np.allclose(on_top.predict(test), continued.predict(test), atol=1e-3)
    with_steps = LightGBMDirect(1, params, "finetune", tmp_path / "pop", finetune, extra=("steps_15min",))
    with_steps.fit(train)
    assert np.isfinite(with_steps.predict(test)).all()
    with pytest.raises(NotImplementedError):
        with_steps.continue_fit(train)
    zero_shot = LightGBMDirect(1, params, "population", tmp_path / "pop", finetune, extra=("steps_15min",))
    zero_shot.fit(train)
    assert np.allclose(zero_shot.predict(test), population.predict(test))
    weighted = LightGBMDirect(1, params, "finetune", tmp_path / "pop", finetune,
                              low_loss={"threshold": 70, "factor": 20})
    weighted.fit(train)
    low = test.targets[:, 0] < 70
    assert weighted.predict(test)[low, 0].mean() <= continued.predict(test)[low, 0].mean() + 0.5


def test_arx_reads_the_wearables():
    grid = toy_grid(4)
    train, test = data(grid, extra=("steps_15min",))
    with_steps = RecursiveARX(extra=("steps_15min",))
    with_steps.fit(train)
    without = RecursiveARX()
    without.fit(train)
    rmse = lambda m: np.sqrt(np.nanmean((m.predict(test)[:, 1] - test.targets[:, 1]) ** 2))  # noqa: E731
    assert rmse(with_steps) < rmse(without)
