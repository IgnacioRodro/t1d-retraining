"""Stage 3b (script 42; D89): other insulin and carbohydrate curves, and the patient's own curve from its past."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.features.build import evaluation_times, insulin_carb_inputs  # noqa: E402
from t1dretrain.sim import physiology as phys  # noqa: E402

FEATURES = {"insulin_action_min": 360, "insulin_peak_min": 75, "insulin_delay_min": 0, "long_acting_spread_h": 24,
            "carb_absorption_min": 180}
TRUE = phys.PhysiologySet(240.0, 75.0, 120.0)
SETTINGS = {"horizons_min": [30, 60], "lookback_hours": 2.0, "origin_stride_min": 30.0, "train_stride_min": 15.0,
            "min_window_share": 0.8, "recent_complete_min": 30.0, "window_days": 14, "test_days": 21,
            "validation_days": 2, "clarke_point_sample": 100, "seeds": [1], "arx": {"ridge": 1.0},
            "lightgbm": {"max_rounds": 30}}


def toy_patient(tmp_path, days=35):
    """A patient whose glucose follows its carbohydrates and insulin with a 240-min insulin action and 120-min
    absorption; the saved grid has the inputs of the default curves (360 and 180 min), as script 05 makes them.
    (ARX, with the glucose changes of the last hour, forecasts this patient about as well with any insulin action
    duration; the absorption time is what tells the curves apart.)"""
    rng = np.random.default_rng(3)
    start = pd.Timestamp("2020-01-01")
    meals = [start + pd.Timedelta(days=d, hours=h) + pd.Timedelta(minutes=int(rng.integers(-40, 40)))
             for d in range(days) for h in (7.5, 13, 20)]
    grams = rng.uniform(30, 90, len(meals)).round()
    carbs = pd.DataFrame({"ts": meals, "carbs_g": grams})
    insulin = pd.DataFrame({"ts": meals, "kind": "bolus", "units": (grams / 10).round(1), "rate_u_per_h": np.nan,
                            "duration_min": np.nan})
    store = ProcessedStore("toy", root=tmp_path)
    store.write("carbs", "p", carbs)
    store.write("insulin", "p", insulin)

    n = days * 288
    ts = start + pd.to_timedelta(np.arange(n) * 5, unit="min")
    grid = pd.DataFrame({"patient_id": "p", "slot": np.arange(n), "ts": ts})
    truth, _ = insulin_carb_inputs(evaluation_times(grid), ts[0], insulin, carbs, TRUE.settings(FEATURES))
    g = np.empty(n)
    g[0] = 140.0
    for t in range(1, n):  # mg/dL per 5 min: 4 per gram absorbed, 45 per unit acting, a pull back to 140
        g[t] = g[t - 1] + 5 / 60 * (4 * truth["carb_absorption"][t] - 45 * truth["insulin_activity"][t]) \
            + 0.01 * (140 - g[t - 1]) + rng.normal(0, 1.5)
    g = np.clip(g, 40, 400)
    hours = (ts - ts.normalize()).total_seconds() / 3600
    grid = grid.assign(glucose=g, target_ok=True, glucose_input=g, segment=0, valid_day=True, loop="open",
                       insulin_known=True, period="post_randomization",
                       hour_sin=np.sin(2 * np.pi * hours / 24), hour_cos=np.cos(2 * np.pi * hours / 24))
    saved, _ = insulin_carb_inputs(evaluation_times(grid), ts[0], insulin, carbs, FEATURES)
    for name, values in saved.items():
        grid[name] = values.astype(np.float32)
    grid["day"] = grid["ts"].dt.normalize()
    ProcessedStore("toy", root=tmp_path, stage="features").write_frame("grid", "p", grid)
    return pd.Timestamp("2020-01-15")


def test_sets_and_saved_grid_times():
    sets = phys.parameter_sets([240, 360], [75], [120, 180])
    assert [s.name for s in sets][0] == "dia240_peak75_carb120" and len(sets) == 4
    assert phys.default_set(FEATURES) in sets
    with pytest.raises(ValueError):
        phys.parameter_sets([120], [75], [180])  # the peak must come before half the action
    grid = pd.DataFrame({"ts": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:05"]), "observed": [True, False],
                         "snap_offset_s": np.array([-61.5, np.nan], dtype=np.float32)})
    assert list(evaluation_times(grid)) == [pd.Timestamp("2019-12-31 23:58:58.500"), pd.Timestamp("2020-01-01 00:05")]
    # a reading on a minute boundary, saved as a float32 offset from an odd slot time, stays on its minute
    slot, reading = pd.Timestamp("2018-06-13 17:19:27.986620194"), pd.Timestamp("2018-06-13 17:15:00")
    saved = pd.DataFrame({"ts": [slot], "observed": [True],
                          "snap_offset_s": np.array([(reading - slot).total_seconds()], dtype=np.float32)})
    assert evaluation_times(saved).iloc[0] == reading


def test_prequential_choice_uses_only_known_targets():
    days = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    known = np.array(["2020-01-01 10:00", "2020-01-02 10:00"], dtype="datetime64[ns]")
    targets = np.array([[100.0], [100.0]])
    persistence = np.array([[90.0], [90.0]])
    forecasts = np.array([[[99.0], [80.0]], [[95.0], [100.0]]])  # set 0 best on day 1's origin, set 1 on day 2's
    chosen, scores = phys.prequential_choice(forecasts, persistence, targets, known, days, default=1)
    assert list(chosen) == [0, 1]  # day 3 sees both origins: set 1 (errors 25 + 0) beats set 0 (1 + 400)
    assert scores.shape == (2, 2)


def test_the_patients_own_curve_is_found_from_its_past(tmp_path):
    pytest.importorskip("lightgbm")
    anchor = toy_patient(tmp_path)
    sets = phys.parameter_sets([240, 360, 480], [75], [120, 180, 240])
    job = {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "post_randomization",
           "anchor_day": 14, "anchor": anchor, "settings": SETTINGS, "stream": 0, "sets": sets, "features": FEATURES,
           "lightgbm": "lightgbm", "interval": 7, "forgetting": 1.0}
    r = phys.evaluate_physiology(job)
    assert not r.error, r.error
    d = r.days
    arx = d[d["model"] == "arx"]
    sse = arx.groupby("policy")["sse"].sum()
    assert sse[f"fixed_{TRUE.name}"] < sse["fixed_dia360_peak75_carb180"]  # the true curves forecast better
    assert sse["personal"] < sse["fixed_dia360_peak75_carb180"]
    cover = {p: set(zip(g["day"], g["horizon_min"])) for p, g in d[d["policy"] != "reference"].groupby("policy")}
    assert all(c == cover["personal"] for c in cover.values())               # every arm on the same days
    info = pd.DataFrame(r.info)
    choices = info[info["chosen_set"].notna()].sort_values("choice_day")
    assert choices["chosen_set"].str.endswith("_carb120").all()            # the true absorption, from the past alone
    assert (choices["chosen_score"] > choices["default_score"]).all()
    fits = info[info["chosen_set"].isna()]
    assert fits.loc[fits["model"] == "arx", "default_gap"].max() < 1e-3      # the saved grid has the default inputs
    gbm = fits[fits["model"] == "lightgbm"]
    assert set(gbm["policy"]) == {"never", "restart", "restart_personal"}
    assert (gbm.loc[gbm["policy"].isin(["never", "restart"]), "physiology_set"] == "dia360_peak75_carb180").all()
    assert gbm.loc[gbm["policy"] == "restart_personal", "physiology_set"].str.endswith("_carb120").all()
    assert set(d.loc[d["model"] == "lightgbm", "policy"]) == {"never", "restart", "restart_personal"}
    assert np.isfinite(d["sse"]).all()


def test_no_data_before_the_anchor_leaves_the_patient_out(tmp_path):
    anchor = toy_patient(tmp_path)
    path = tmp_path / "features" / "toy" / "grid" / "p.parquet"
    grid = pd.read_parquet(path)
    early = grid["day"] < anchor
    grid.loc[early, ["glucose", "glucose_input"]] = np.nan
    grid.loc[early, ["target_ok", "valid_day"]] = False
    grid.to_parquet(path, index=False)
    job = {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "post_randomization",
           "anchor_day": 14, "anchor": anchor, "settings": SETTINGS, "stream": 0,
           "sets": phys.parameter_sets([240, 360], [75], [180]), "features": FEATURES, "lightgbm": None,
           "interval": 7}
    r = phys.evaluate_physiology(job)
    assert not r.error, r.error
    assert set(r.days["policy"]) <= {"reference"}
    assert pd.DataFrame(r.info)["excluded"].notna().all()
