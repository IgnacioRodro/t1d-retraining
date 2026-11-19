"""Q4 (scripts 60 and 61; D94): regularity indices and their association with the outcomes of refitting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.features import regularity as rg  # noqa: E402

DAYS = list(pd.date_range("2020-01-01", periods=14, freq="D"))


def meals(jitter_min: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for day in DAYS:
        for hour, grams in ((8, 40), (13, 70), (20, 60)):
            shift = rng.normal(0, jitter_min) if jitter_min else 0.0
            rows.append({"ts": day + pd.Timedelta(hours=hour) + pd.Timedelta(minutes=shift), "carbs_g": grams})
    return pd.DataFrame(rows)


def test_timing_and_profile_of_regular_and_irregular_meals():
    regular = rg.merge_events(meals(0)["ts"], meals(0)["carbs_g"].to_numpy(), 30)
    irregular = rg.merge_events(meals(90, 1)["ts"], meals(90, 1)["carbs_g"].to_numpy(), 30)
    assert rg.time_jitter(regular) == 0.0 and rg.time_jitter(irregular) > 20
    hourly = rg.hourly_by_day(regular["ts"], regular["size"].to_numpy(), DAYS)
    assert hourly.shape == (14, 24) and np.isclose(rg.profile_stability(hourly), 1.0)
    noisy = rg.hourly_by_day(irregular["ts"], irregular["size"].to_numpy(), DAYS)
    assert rg.profile_stability(noisy) < rg.profile_stability(hourly)
    adi, cv2 = rg.intermittency(hourly)
    assert np.isclose(adi, 8.0) and cv2 > 0
    assert rg.circular_distance(np.array([1430.0]), np.array([10.0]))[0, 0] == 20.0
    merged = rg.merge_events(pd.Series(pd.to_datetime(["2020-01-01 08:00", "2020-01-01 08:20", "2020-01-01 09:00"])),
                             np.array([10.0, 20.0, 5.0]), 30)
    assert merged["size"].tolist() == [30.0, 5.0]
    assert rg.circular_sd_minutes(np.array([1430.0, 10.0, 0.0, 1435.0])) < 15


def test_patient_indices_use_only_the_given_days():
    ts = pd.date_range("2020-01-01", periods=16 * 288, freq="5min")
    grid = pd.DataFrame({"ts": ts, "day": ts.normalize(),
                         "glucose": 140 + 40 * np.sin(2 * np.pi * (ts.hour * 60 + ts.minute) / 1440.0)})
    carbs = meals(15)
    carbs.loc[len(carbs)] = {"ts": pd.Timestamp("2020-01-16 03:00"), "carbs_g": 50}  # outside the days
    insulin = pd.DataFrame({"ts": carbs["ts"], "kind": "bolus", "units": carbs["carbs_g"] / 10})
    out = rg.patient_indices(grid, insulin, carbs, pd.DataFrame(columns=["ts", "steps"]),
                             pd.DataFrame(columns=["start", "end", "stage"]), DAYS,
                             {"merge_min": 30, "min_meal_g": 10, "min_bolus_u": 0.5})
    assert out["window_days_used"] == 14 and out["meal_events"] == 42 and out["bolus_events"] == 42
    assert np.isclose(out["meals_per_day"], 3.0) and out["glucose_profile_stability"] > 0.99
    assert 0 < out["tir_pct"] <= 100 and np.isfinite(out["meal_time_jitter_min"])
    no_insulin = rg.patient_indices(grid, insulin, carbs, pd.DataFrame(columns=["ts", "steps"]),
                                    pd.DataFrame(columns=["start", "end", "stage"]), DAYS,
                                    {"merge_min": 30, "min_meal_g": 10, "min_bolus_u": 0.5}, uses_insulin=False)
    assert "bolus_time_jitter_min" not in no_insulin


def test_association_recovers_a_planted_effect_and_holm():
    pytest.importorskip("statsmodels")
    from t1dretrain.eval import regularity_analysis as ra

    rng = np.random.default_rng(3)
    n = 300
    index = rng.normal(30, 10, n)
    cohort = np.where(np.arange(n) < 150, "a/x", "b/y")
    outcome = 0.01 * (index - 30) / 10 + np.where(cohort == "a/x", 0.02, 0.0) + rng.normal(0, 0.003, n)
    frame = pd.DataFrame({"cohort": cohort, "patient_id": [f"p{i}" for i in range(n)], "gain_refit": outcome,
                          "meal_time_jitter_min": index, "tir_pct": rng.normal(60, 10, n),
                          "glucose_cv_pct": rng.normal(35, 5, n), "glucose_mean": rng.normal(160, 20, n)})
    r = ra.association(frame, "gain_refit", "meal_time_jitter_min", list(ra.ADJUSTERS), 30)
    planted = 100 * 0.01 * index.std(ddof=1) / 10  # points per SD of the index in the sample
    assert abs(r["coef"] - planted) < 0.05 and r["low"] < planted < r["high"] and r["p"] < 1e-6
    assert ra.association(frame.head(20), "gain_refit", "meal_time_jitter_min", [], 30).get("coef") is None
    p = pd.Series([0.01, 0.04, np.nan, 0.03])
    assert np.allclose(ra.holm(p).dropna().to_numpy(), [0.03, 0.06, 0.06])


def test_patient_outcomes_from_day_metrics():
    from t1dretrain.eval import regularity_analysis as ra

    rows = []
    days = pd.date_range("2020-02-01", periods=28, freq="D")
    for k, day in enumerate(days):
        for model, policy, sse in (("persistence", "reference", 400.0), ("m", "never", 100.0 + 2 * k),
                                   ("m", "restart_every7", 81.0), ("m", "restart_every1", 64.0),
                                   ("m_noexog", "never", 144.0), ("m_noexog", "restart_every7", 121.0)):
            rows.append({"cohort": "c/x", "period": "whole", "patient_id": "p", "anchor_day": 14, "model": model,
                         "policy": policy, "horizon_min": 30, "day": day, "n": 20, "sse": 20 * sse,
                         "days_since_anchor": k, "age_days": k, "fit_day": days[0]})
    frame = pd.DataFrame(rows)
    out = ra.with_contrasts(ra.patient_outcomes(frame, {"m": {"interval": 7}}, 84))
    m = out[(out["model"] == "m") & (out["horizon_min"] == 30)].iloc[0]
    never = 1 - np.sqrt(np.mean(100.0 + 2 * np.arange(28))) / 20
    assert np.isclose(m["skill_never"], never) and np.isclose(m["gain_refit"], 1 - 9 / 20 - never)
    assert np.isclose(m["loss_daily"], (1 - 9 / 20) - (1 - 8 / 20)) and m["ageing"] < 0
    twin = out[(out["model"] == "m_noexog") & (out["horizon_min"] == 30)].iloc[0]
    assert np.isclose(twin["gain_refit"], (1 - 11 / 20) - (1 - 12 / 20)) and np.isnan(twin["loss_daily"])
    contrast = out[(out["model"] == "m" + ra.INPUTS) & (out["horizon_min"] == 30)].iloc[0]
    assert np.isclose(contrast["gain_refit"], m["gain_refit"] - twin["gain_refit"])
    assert set(out["horizon_min"]) == {0, 30}
