"""Forecast lows (scripts 70 and 71; D95, D96) and the comparison of a model variant with the standard model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.eval import low_glucose as lg  # noqa: E402
from t1dretrain.eval import variant_analysis as va  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.sim import cohort_runs as cr  # noqa: E402


def histogram() -> pd.DataFrame:
    """A glucose trace with lows; persistence-like forecasts (unbiased, noisy) and a forecast shrunk towards the
    middle (less noise): the second alarms less at 70 mg/dL but is not a worse detector."""
    rng = np.random.default_rng(0)
    reference = np.clip(140 + 45 * rng.standard_normal(40000), 40, 400)
    noisy = reference + rng.normal(0, 18, reference.size)
    shrunk = 140 + 0.8 * (reference - 140) + rng.normal(0, 9, reference.size)
    acc = ClarkeAccumulator(["model", "horizon_min"], sample_size=0)
    acc.add(("persistence", 30), reference, noisy)
    acc.add(("learned", 30), reference, shrunk)
    return acc.histogram()


def test_shrinkage_curves_and_matched_alarms():
    h = histogram()
    keys = ["model", "horizon_min"]
    shrink = lg.shrinkage(h, keys).set_index("model")
    assert shrink.loc["learned", "mean_forecast_in_lows"] > shrink.loc["persistence", "mean_forecast_in_lows"]
    assert shrink.loc["learned", "mean_reference_of_forecast_lows"] < \
        shrink.loc["persistence", "mean_reference_of_forecast_lows"]
    curves = lg.alarm_curves(h, keys)
    at70 = curves[(curves["event"] == "below70") & (curves["alarm"] == 70)].set_index("model")
    assert at70.loc["learned", "sensitivity_pct"] < at70.loc["persistence", "sensitivity_pct"]
    assert at70.loc["learned", "precision_pct"] > at70.loc["persistence", "precision_pct"]
    sens = curves[curves["event"] == "below70"].groupby("model")["sensitivity_pct"]
    assert (sens.apply(lambda s: s.is_monotonic_increasing)).all()  # nested alarms
    m = lg.matched(curves, keys).set_index(["model", "event"]).loc[(slice(None), "below70"), :]
    learned = m.loc[("learned", "below70")]
    assert learned["sensitivity_at_reference_precision_pct"] > learned["reference_sensitivity_pct"]
    assert learned["alarm_for_reference_precision"] > 70
    assert np.isclose(m.loc[("persistence", "below70"), "sensitivity_at_reference_precision_pct"],
                      at70.loc["persistence", "sensitivity_pct"])
    assert learned["average_precision_pct"] > m.loc[("persistence", "below70"), "average_precision_pct"]
    ranges = lg.range_table(h, keys)
    low = ranges[(ranges["model"] == "learned") & (ranges["range"] == "55-69")].iloc[0]
    assert low["bias"] > 5 and abs(ranges.groupby("model")["share_pct"].sum() - 100).max() < 1e-9
    calibration = lg.calibration(h, keys)
    given = calibration[(calibration["kind"] == "reference_given_forecast") & (calibration["model"] == "learned")]
    middle = given[(given["bin"] > 100) & (given["bin"] < 180)]
    assert np.abs(middle["mean"] - middle["bin"]).mean() < 8  # E[reference | forecast] near the diagonal


def test_crossings_of_a_curve():
    curve = pd.DataFrame({"alarm": [40, 50, 60, 70], "sensitivity_pct": [5, 20, 40, 60],
                          "precision_pct": [30, 60, 50, 40]})
    y, alarm, note = lg._at(curve, "precision_pct", "sensitivity_pct", 45)  # the most sensitive alarm that precise
    assert note == "" and np.isclose(alarm, 65) and np.isclose(y, 50)
    assert lg._at(curve, "precision_pct", "sensitivity_pct", 70)[2] == "below"
    assert lg._at(curve, "precision_pct", "sensitivity_pct", 35) == (60.0, 70.0, "beyond")
    y, alarm, _ = lg._at(curve, "sensitivity_pct", "precision_pct", 30)
    assert np.isclose(alarm, 55) and np.isclose(y, 55)
    assert np.isclose(lg.average_precision(curve), (5 * 30 + 15 * 60 + 20 * 50 + 20 * 40) / 100)


def day_frame() -> pd.DataFrame:
    rows = []
    for variant, shift in (("standard", 0.0), ("x5", 1.0)):
        for patient in range(6):
            for model, policy, sse in (("persistence", "reference", 400.0), ("m", "never", 100.0),
                                       ("m", "restart_every3", 81.0)):
                if model == "m":
                    sse = sse + (10.0 if variant == "x5" else 0.0) - patient
                true = 20
                hit = 6 + int(4 * shift) if model == "m" else 9
                rows.append({"cohort": "c/x", "variant": variant, "period": "whole", "patient_id": f"p{patient}",
                             "anchor_day": 14, "model": model, "policy": policy, "horizon_min": 30, "n": 100,
                             "sse": 100 * sse, "below70_true": true, "below70_alarm": hit + 2, "below70_hit": hit})
    return pd.DataFrame(rows)


def test_variant_against_standard_and_lows():
    days = day_frame()
    skill = va.skill_table(days)
    lows = lg.patient_lows(days, va.KEYS, min_events=10)
    versus = va.against_standard(skill, lows, 200, 1)
    s = versus[(versus["measure"] == "skill") & (versus["policy"] == "never")].iloc[0]
    assert s["variant"] == "x5" and s["difference"] < 0 and s["patients"] == 6
    e = versus[(versus["measure"] == "sensitivity_70") & (versus["policy"] == "never")].iloc[0]
    assert np.isclose(e["difference"], 100 * 4 / 20)
    gain = va.refit_gain(skill, 200, 1)
    assert set(gain["variant"]) == {"standard", "x5"} and (gain["gain"] > 0).all()
    assert gain["gain_difference"].notna().sum() == 1
    metrics = va.pooled_metrics(days.assign(se=0.0, sae=0.0, sape=0.0, **{f"zone_{z}": 0 for z in "ABCDE"},
                                            below54_true=0, below54_alarm=0, below54_hit=0))
    assert {"rmse", "low70_sensitivity_pct", "clarke_A_pct"} <= set(metrics.columns)


def test_cache_key_options():
    settings = {k: 1 for k in cr.SETTING_KEYS}
    plain = cr.cache_key("c", "valid", "ar3", "never", settings, {})
    assert plain == cr.cache_key("c", "valid", "ar3", "never", {**settings, "low_loss": None}, {})
    assert plain != cr.cache_key("c", "valid", "ar3", "never", {**settings, "low_loss": {"factor": 5}}, {})
    assert plain != cr.cache_key("c", "valid", "ar3", "never", {**settings, "extra_inputs": ["asleep"]}, {})
