"""Model inputs of script 05: insulin and carbohydrate models, the grid, and causality."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.features import carbs as carb_model  # noqa: E402
from t1dretrain.features.build import build_patient  # noqa: E402
from t1dretrain.features.grid import glucose_grid, interpolate  # noqa: E402
from t1dretrain.features.insulin import ExponentialInsulin, delivery_per_minute, on_board  # noqa: E402

ORIGIN = pd.Timestamp("2020-01-01")
SETTINGS = {"insulin_action_min": 360, "insulin_peak_min": 75, "insulin_delay_min": 0, "long_acting_spread_h": 24,
            "carb_absorption_min": 180, "max_interp_gap_min": 15, "meal_check_min_carbs_g": 20}


def insulin_rows(rows):
    frame = pd.DataFrame(rows, columns=["ts", "kind", "units", "rate_u_per_h", "duration_min"])
    frame["ts"] = pd.to_datetime(frame["ts"])
    return frame


def test_exponential_curve():
    model = ExponentialInsulin()
    t = np.arange(0, 361, 0.25)
    assert np.isclose(model.remaining(0), 1) and np.isclose(model.remaining(360), 0)
    assert np.isclose(np.trapezoid(model.activity(t), t), 1, atol=1e-4)
    assert t[np.argmax(model.activity(t))] == 75
    assert np.allclose(-np.gradient(model.remaining(t), t)[4:-4], model.activity(t)[4:-4], atol=1e-5)


def test_iob_is_causal_and_follows_the_curve():
    n = 24 * 60
    bolus, _ = delivery_per_minute(insulin_rows([["2020-01-01 08:00:30", "bolus", 4, np.nan, np.nan]]), ORIGIN, n)
    iob, _ = on_board(bolus, ExponentialInsulin())
    assert iob[: 8 * 60].max() == 0  # nothing before the bolus, not even in the minute it was given
    assert np.isclose(iob[8 * 60 + 1], 4)  # first bin after the bolus time
    assert np.isclose(iob[8 * 60 + 61], 4 * ExponentialInsulin().remaining(60))


def test_basal_overlaps_injections_and_five_minute_records():
    n = 3 * 24 * 60
    rows = [["2020-01-01 00:00", "basal_rate", np.nan, 1.0, 600],   # replaced at 02:00 by the next rate
            ["2020-01-01 02:00", "basal_rate", np.nan, 2.0, 60],
            ["2020-01-02 22:00", "basal_injection", 24, np.nan, np.nan],
            ["2020-01-01 10:05", "total_5min", 0.5, np.nan, np.nan]]
    _, basal = delivery_per_minute(insulin_rows(rows), ORIGIN, n)
    assert np.isclose(basal[: 3 * 60].sum(), 2 * 1.0 + 1 * 2.0)  # no double counting of the first segment
    assert np.isclose(basal[: 10 * 60 + 5].sum() - basal[: 10 * 60].sum(), 0.5)
    day3 = basal[2 * 24 * 60:]
    assert np.isclose(basal[(24 + 22) * 60 + 5], 1 / 60) and np.isclose(day3.sum() + basal[46 * 60:48 * 60].sum(), 24)


def test_cob_linear_absorption():
    grams = carb_model.carbs_per_minute(pd.DataFrame({"ts": [ORIGIN + pd.Timedelta(minutes=60)], "carbs_g": [60.0]}),
                                        ORIGIN, 600)
    cob, rate = carb_model.on_board(grams, 180)
    assert cob[59] == 0 and np.isclose(cob[60], 60) and np.isclose(cob[150], 30) and np.isclose(cob[240], 0)
    assert np.isclose(rate[100] * 60, 20)


def clean_series(days=2, step=5.0, jitter_s=20, seed=0):
    rng = np.random.default_rng(seed)
    n = int(days * 1440 / step)
    ts = ORIGIN + pd.to_timedelta(np.arange(n) * step * 60 + 130 + rng.uniform(-jitter_s, jitter_s, n), unit="s")
    return pd.DataFrame({"patient_id": "p", "ts": ts, "glucose_mgdl": 120 + 30 * np.sin(np.arange(n) / 20),
                         "source": "dexcom_g6", "interval_min": step, "flag": "", "clock_break": False})


def test_grid_snaps_every_reading_near_its_slot():
    clean = clean_series()
    grid, info = glucose_grid(clean, 15)
    assert info["step_min"] == 5 and info["collisions_dropped"] == 0
    assert grid["observed"].sum() == len(clean)
    assert grid["snap_offset_s"].abs().max() < 30
    assert (np.diff(grid["slot"]) == 1).all()


def test_interpolation_limits():
    values = np.array([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0, np.nan, 10.0])
    segment = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    out, filled = interpolate(values, segment, 2)
    assert np.allclose(out[:4], [1, 2, 3, 4])
    assert np.isnan(out[4:7]).all()  # three missing slots: too long
    assert np.isnan(out[8])  # never across a segment boundary (clock break)


def test_spikes_and_empty_days():
    clean = clean_series(days=3)
    clean = clean[(clean["ts"] < ORIGIN + pd.Timedelta(days=1)) | (clean["ts"] >= ORIGIN + pd.Timedelta(days=2))]
    clean = clean.reset_index(drop=True)
    clean.loc[10, "flag"] = "spike"
    grid, _ = glucose_grid(clean, 15)
    assert grid["ts"].dt.normalize().nunique() == 2  # the empty middle day is left out
    spike = grid[grid["slot"] == grid.loc[grid["spike"], "slot"].iloc[0]]
    assert not spike["target_ok"].iloc[0] and spike["interpolated"].iloc[0]  # replaced by its neighbours


def test_build_patient_end_to_end(tmp_path):
    clean = clean_series(days=2)
    ProcessedStore("toy", root=tmp_path, stage="clean").write_frame("cgm", "p", clean)
    source = ProcessedStore("toy", root=tmp_path)
    source.write("insulin", "p", insulin_rows([["2020-01-01 12:00:30", "bolus", 5, np.nan, np.nan]]))
    source.write("carbs", "p", pd.DataFrame({"ts": [pd.Timestamp("2020-01-01 12:00:30")], "carbs_g": [50.0]}))
    days = pd.DataFrame({"day": pd.date_range(ORIGIN, periods=2), "valid": True, "loop": "open", "period": "study",
                         "insulin_complete": True, "carbs_n": [1, 0]})
    result = build_patient({"dataset": "toy", "patient_id": "p", "derived_root": tmp_path, "settings": SETTINGS,
                            "days": days, "real_dates": False, "example": True})
    assert result.error == "", result.error
    grid = ProcessedStore("toy", root=tmp_path, stage="features").read_frame("grid", "p")
    before = grid[grid["ts"] + pd.to_timedelta(grid["snap_offset_s"], unit="s") < pd.Timestamp("2020-01-01 12:00:30")]
    after = grid[grid["ts"] > pd.Timestamp("2020-01-01 13:05")].iloc[0]
    assert before["iob"].max() == 0 and before["cob"].max() == 0  # nothing from the future
    assert 0 < after["iob"] < 5 and 0 < after["cob"] < 50
    assert grid["dow"].isna().all() and grid["insulin_known"].any()
    assert result.example is not None
