"""Learned models of stage 0 (LightGBM, N-HiTS): no look-ahead, and they learn what can be learned.

A random walk cannot be forecast better than by persistence: a model that beats it clearly would be seeing the
future. A daily cycle can be learned from the time of day: a model that does not beat persistence there is broken.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.models.base import EXOG, TrainData  # noqa: E402
from t1dretrain.models.reference import Persistence  # noqa: E402
from t1dretrain.sim.samples import make_samples, training_series  # noqa: E402

ORIGIN = pd.Timestamp("2020-01-01")
DAYS = 20


def toy_grid(values: np.ndarray) -> pd.DataFrame:
    n = len(values)
    ts = ORIGIN + pd.to_timedelta(np.arange(n) * 5, unit="min")
    hours = (ts - ts.normalize()).total_seconds() / 3600
    grid = pd.DataFrame({"patient_id": "p", "slot": np.arange(n), "ts": ts, "glucose": values, "target_ok": True,
                         "glucose_input": values, "segment": 0, "iob": 1.0, "cob": 0.0, "insulin_activity": 0.5,
                         "carb_absorption": 0.0, "hour_sin": np.sin(2 * np.pi * hours / 24),
                         "hour_cos": np.cos(2 * np.pi * hours / 24)})
    grid["day"] = grid["ts"].dt.normalize()
    return grid


def skill_at_30(model, values: np.ndarray) -> float:
    grid = toy_grid(values)
    days = sorted(grid["day"].unique())
    train_days, test_days = set(days[:14]), set(days[14:])
    common = dict(horizons_min=[30, 60], lookback_h=2, min_window_share=0.8, recent_complete_min=30, exog_columns=EXOG)
    train = make_samples(grid, stride_min=5, allowed_days=train_days, trajectory=True, **common)
    test = make_samples(grid, stride_min=30, allowed_days=test_days, **common)
    series, segment, step = training_series(grid, train_days)
    model.fit(TrainData(train, series, segment, step, validation=train.day >= np.datetime64(days[12])))
    error = lambda p: np.sqrt(np.nanmean((p[:, 0] - test.targets[:, 0]) ** 2))  # noqa: E731
    return 1 - error(model.predict(test)) / error(Persistence().predict(test))


def random_walk(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 200 + np.cumsum(rng.normal(0, 1.0, DAYS * 288))


def daily_cycle(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(DAYS * 288) * 5 / 60
    return 150 + 50 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 4, len(t))


@pytest.mark.parametrize("kind", ["lightgbm", "nhits"])
def test_no_look_ahead_on_a_random_walk(kind):
    model = make(kind)
    assert skill_at_30(model, random_walk(1)) < 0.05


@pytest.mark.parametrize("kind", ["lightgbm", "nhits"])
def test_learns_a_daily_cycle(kind):
    model = make(kind)
    assert skill_at_30(model, daily_cycle(2)) > 0.2


def make(kind: str):
    if kind == "lightgbm":
        pytest.importorskip("lightgbm")
        from t1dretrain.models.gbdt import LightGBMDirect

        return LightGBMDirect(seed=1)
    pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from t1dretrain.models.neural import NHITSPersonal

    return NHITSPersonal(seed=1, params={"max_steps": 300, "mlp_units": 64})


# ---------------------------------------------------------------------------- starting points (D62)


def pool_data(seed: int, days: int = DAYS):
    """Samples of a daily cycle with a trajectory, the last two days held out."""
    grid = toy_grid(daily_cycle(seed)[: days * 288])
    all_days = sorted(grid["day"].unique())
    common = dict(horizons_min=[30, 60], lookback_h=2, min_window_share=0.8, recent_complete_min=30, exog_columns=EXOG)
    s = make_samples(grid, stride_min=15, trajectory=True, **common)
    return s, s.day >= np.datetime64(all_days[-2])


@pytest.mark.parametrize("kind", ["lightgbm", "nhits"])
def test_population_model_round_trip_and_finetune(kind, tmp_path):
    from t1dretrain.sim.stage0 import validation_mask

    s, held = pool_data(3)
    if kind == "lightgbm":
        pytest.importorskip("lightgbm")
        from t1dretrain.models.gbdt import LightGBMDirect as Model

        params = {"max_rounds": 50}
        pop = Model(seed=1, params=params)
        pop.train_on(s, held, pop.params)
        path = tmp_path / "lightgbm_seed1"
    else:
        pytest.importorskip("torch")
        pytest.importorskip("neuralforecast")
        from t1dretrain.models.neural import SCHEDULE
        from t1dretrain.models.neural import NHITSPersonal as Model

        params = {"max_steps": 60, "mlp_units": 32, "val_every": 20}
        pop = Model(seed=1, params=params)
        pop.train_on(s, held, {k: pop.params[k] for k in SCHEDULE})
        path = tmp_path / "nhits_seed1.pt"
    pop.save(path)
    test = s.subset(held)
    zero_shot = Model(seed=1, start="population", population=path)
    window, _ = pool_data(4, days=5)
    data = TrainData(window, np.full(0, np.nan), np.zeros(0), window.step_min,
                     validation=validation_mask(window.day, 1))
    zero_shot.fit(data)
    assert np.allclose(zero_shot.predict(test), pop.predict(test), atol=1e-4)  # saved = trained
    tuned = Model(seed=1, start="finetune", population=path, finetune={"max_rounds": 20} if kind == "lightgbm"
                  else {"max_steps": 40, "val_every": 10})
    tuned.fit(data)
    val = window.subset(data.validation)
    error = lambda m: np.nanmean((m.predict(val) - val.targets) ** 2)  # noqa: E731
    assert error(tuned) <= error(zero_shot) + 1e-6  # a fine-tuning that hurts the validation days is undone


def test_samples_concatenate_and_survive_a_round_trip(tmp_path):
    from t1dretrain.sim.samples import concat_samples, load_samples, save_samples

    a, _ = pool_data(5, days=4)
    b, _ = pool_data(6, days=3)
    b.patient_id = "q"
    both = concat_samples([a, b])
    assert both.n == a.n + b.n and set(both.patient) == {"p", "q"}
    save_samples(both, tmp_path / "s")
    back = load_samples(tmp_path / "s")
    assert np.allclose(back.inputs, both.inputs, equal_nan=True) and list(back.exog) == list(both.exog)
    assert (back.origin_ts == both.origin_ts).all() and (back.patient == both.patient).all()


def test_validation_days_are_the_last_days_with_samples_and_names_split():
    from t1dretrain.sim.stage0 import split_name, validation_mask

    day = np.array(["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-05"], dtype="datetime64[D]")
    assert validation_mask(day, 2).tolist() == [False, True, True, True]  # 3 and 4 Jan have no samples
    assert not validation_mask(day[:2], 2).any()                           # too few days: nothing held out
    assert split_name("nhits_ft") == ("nhits", "finetune") and split_name("lightgbm") == ("lightgbm", "scratch")
    assert split_name("ar3") == ("ar3", "")


def test_head_finetuning_changes_only_the_output_layers(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from t1dretrain.models.neural import SCHEDULE, NHITSPersonal

    s, held = pool_data(7, days=4)
    pop = NHITSPersonal(seed=1, params={"max_steps": 20, "mlp_units": 16, "val_every": 10})
    pop.train_on(s, held, {k: pop.params[k] for k in SCHEDULE})
    pop.save(tmp_path / "n.pt")
    before = {k: v.clone() for k, v in pop.net.state_dict().items()}
    tuned = NHITSPersonal(seed=1, start="finetune", population=tmp_path / "n.pt",
                          finetune={"train": "head", "max_steps": 20, "val_every": 1000, "learning_rate": 1e-2})
    tuned.load(tmp_path / "n.pt")
    tuned.train_on(s, None, tuned.finetune)
    after = tuned.net.state_dict()
    heads = {f"blocks.{b}.layers.{len(pop.net.blocks[b].layers) - 1}" for b in range(len(pop.net.blocks))}
    for k, v in after.items():
        changed = not np.allclose(v.cpu().numpy(), before[k].cpu().numpy())
        assert changed == any(k.startswith(h + ".") for h in heads), k
