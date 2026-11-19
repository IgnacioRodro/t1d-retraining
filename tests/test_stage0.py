"""Forecasting samples, reference models and the hold-out guard (scripts 10+)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.data.splits import group_patients  # noqa: E402
from t1dretrain.eval.stats import bootstrap_mean  # noqa: E402
from t1dretrain.models.base import TrainData  # noqa: E402
from t1dretrain.models.reference import AutoRegressive, LinearExtrapolation, Persistence  # noqa: E402
from t1dretrain.sim.samples import make_samples, training_series  # noqa: E402

ORIGIN = pd.Timestamp("2020-01-01")


def toy_grid(values, step=5.0, break_at=None, insulin_known=True, drop=()):
    n = len(values)
    slot = np.arange(n) + 1000
    grid = pd.DataFrame({"patient_id": "p", "slot": slot,
                         "ts": ORIGIN + pd.to_timedelta(np.arange(n) * step, unit="min"),
                         "glucose": np.asarray(values, dtype=float), "target_ok": True,
                         "glucose_input": np.asarray(values, dtype=float), "segment": 0,
                         "insulin_known": insulin_known})
    if break_at is not None:
        grid.loc[break_at:, "segment"] = 1
    grid["day"] = grid["ts"].dt.normalize()
    return grid.drop(index=list(drop)).reset_index(drop=True)


def test_samples_are_causal_and_aligned():
    values = np.arange(600, dtype=float) + 50
    s = make_samples(toy_grid(values), [30, 60], lookback_h=1, stride_min=30, min_window_share=0.8)
    origin = s.origin_slot - 1000
    assert np.allclose(s.inputs[:, -1], values[origin])           # last input is the origin reading
    assert np.allclose(s.inputs[:, 0], values[origin - 11])       # 1 h back on the 5-min grid
    ok = np.isfinite(s.targets[:, 0])
    assert np.allclose(s.targets[ok, 0], values[origin[ok] + 6])  # 30 min ahead
    assert (s.origin_slot % 6 == 0).all()                         # origins on the 30-min stride


def test_samples_never_cross_a_clock_break_and_respect_conditions():
    values = np.full(600, 120.0)
    s = make_samples(toy_grid(values, break_at=300), [30], lookback_h=1, stride_min=5, min_window_share=0.8)
    origin = s.origin_slot - 1000
    assert not ((origin >= 300) & (origin < 311)).any()           # window would start before the break
    assert np.isnan(s.targets[(origin >= 294) & (origin < 300), 0]).all()  # target after the break
    none = make_samples(toy_grid(values, insulin_known=False), [30], 1, 30, origin_conditions=("insulin_known",))
    assert none.n == 0


def test_window_completeness_rules():
    values = np.full(600, 120.0)
    grid = toy_grid(values)
    grid.loc[200:202, "glucose_input"] = np.nan                   # 3 missing inputs
    s = make_samples(grid, [30], lookback_h=1, stride_min=5, recent_complete_min=30)
    origin = s.origin_slot - 1000
    assert not ((origin >= 200) & (origin <= 208)).any()          # the last 30 min must be complete


def test_reference_models():
    values = 100 + 0.5 * np.arange(600)                           # a straight line, 0.5 mg/dL per slot
    s = make_samples(toy_grid(values), [30, 60], lookback_h=1, stride_min=30)
    assert np.allclose(Persistence().predict(s)[:, 1], s.inputs[:, -1])
    ok = np.isfinite(s.targets).all(axis=1)
    assert np.allclose(LinearExtrapolation().predict(s)[ok], s.targets[ok], atol=1e-3)


def test_ar3_recovers_its_coefficients():
    rng = np.random.default_rng(0)
    phi = np.array([0.6, 0.25, 0.1])
    y = np.full(5000, 140.0)
    for t in range(3, len(y)):
        y[t] = 7.0 + phi @ y[t - 3:t][::-1] + rng.normal(0, 3)
    grid = toy_grid(y)
    series, segment, step = training_series(grid, set(grid["day"]))
    model = AutoRegressive(3)
    model.fit(TrainData(make_samples(grid, [30], 1, 30), series, segment, step))
    assert np.allclose(model.coef[1:], phi, atol=0.03)


def test_confirm_group_is_held_out(tmp_path):
    (tmp_path / "toy.yaml").write_text(yaml.safe_dump({"groups": {"pool": ["a"], "screen": ["b"], "confirm": ["c"]}}))
    assert group_patients("toy", "screen", splits_dir=tmp_path) == ["b"]
    with pytest.raises(PermissionError):
        group_patients("toy", "confirm", splits_dir=tmp_path)
    assert group_patients("toy", "confirm", allow_confirm=True, splits_dir=tmp_path) == ["c"]


def test_bootstrap_interval_covers_the_mean():
    mean, low, high = bootstrap_mean(np.arange(100, dtype=float))
    assert low < mean < high and np.isclose(mean, 49.5)


def test_dmd_of_a_pair_is_the_larger_of_its_models():
    """D75: each learned model has its own DMD; models without seed noise take the smallest one."""
    from t1dretrain.eval.stage0_report import DMDTable

    noise = pd.DataFrame({"period": "p", "horizon_min": 30, "model": ["lightgbm_ft", "patchtst_ft"],
                          "dmd": [0.003, 0.016]})
    dmd = DMDTable(noise)
    assert dmd.model("p", 30, "lightgbm_ft") == 0.003 and dmd.model("p", 30, "ar3") == 0.003
    assert dmd.pair("p", 30, "lightgbm_ft", "ar3") == 0.003 and dmd.pair("p", 30, "patchtst_ft", "ar3") == 0.016
    assert DMDTable(noise, ["patchtst_ft"]).model("p", 30, "arx") == 0.016
    assert len(DMDTable(None)) == 0 and np.isnan(DMDTable(None).pair("p", 30, "a", "b"))


def test_twins_change_nothing_for_the_full_models():
    """D90: a glucose-only twin keeps its own DMD but does not set the floor of the models without seed noise, and
    every model's DMD comes from its own bootstrap stream, so adding twins leaves the full models' DMDs as they were."""
    from t1dretrain.eval.stage0_report import DMDTable, noise_floor

    noise = pd.DataFrame({"period": "p", "horizon_min": 30, "model": ["lightgbm_ft", "lightgbm_ft_noexog"],
                          "dmd": [0.003, 0.001]})
    dmd = DMDTable(noise)
    assert dmd.model("p", 30, "lightgbm_ft_noexog") == 0.001 and dmd.model("p", 30, "arx") == 0.003

    rng = np.random.default_rng(0)
    rows = [{"period": "p", "horizon_min": 30, "model": m, "seed": s, "patient_id": f"x{i}",
             "skill": 0.2 + 0.01 * rng.standard_normal()}
            for m in ("lightgbm_ft", "lightgbm_ft_noexog") for s in (1, 2, 3) for i in range(12)]
    by_patient = pd.DataFrame(rows)
    alone = noise_floor(by_patient[by_patient["model"] == "lightgbm_ft"], 200, 7)
    both = noise_floor(by_patient, 200, 7)
    full = both[both["model"] == "lightgbm_ft"].reset_index(drop=True)
    assert alone.equals(full)
    assert (both["dmd_stage"] == alone["dmd"].iloc[0]).all()
