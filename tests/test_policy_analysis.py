"""Stage 2 analysis (script 31; D83, D84): daily gain, its growth, the optimal interval and the data hypothesis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.eval import policy_analysis as pa  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402

PER_DOUBLING = 0.02
SHARE = {"never": 0.0, "growing_every1": 1.0, "growing_every7": 0.95, "growing_every14": 0.6}


def synthetic_days(patients=20, anchors=(14, 42, 70), test_days=84, seed=0):
    """Day metrics where the gain of refitting is PER_DOUBLING * log2((history + t) / history) (history = anchor)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(patients):
        level = 0.2 + rng.normal(0, 0.03)
        for anchor in anchors:
            for t in range(test_days):
                day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=anchor + t)
                rmse_p = 20.0 * np.exp(rng.normal(0, 0.2))
                gain = PER_DOUBLING * np.log2((anchor + t) / anchor)
                base = dict(period="p", patient_id=f"P{p}", anchor_day=anchor, horizon_min=60, day=day, n=48,
                            days_since_anchor=t, age_days=0, seed=0, fit_day=day)
                rows.append({**base, "model": "persistence", "policy": "reference", "sse": 48 * rmse_p ** 2})
                noise = rng.normal(0, 0.01)
                for policy, share in SHARE.items():
                    skill = level + share * gain + noise + rng.normal(0, 0.002)
                    rows.append({**base, "model": "m_ft", "policy": policy, "sse": 48 * (rmse_p * (1 - skill)) ** 2})
    return pd.DataFrame(rows)


class FixedDMD:
    def __len__(self):
        return 1

    def model(self, period, horizon, model):
        return 0.003


@pytest.fixture(scope="module")
def days():
    return synthetic_days()


def test_daily_gain_and_its_growth(days):
    pytest.importorskip("statsmodels")
    main = days[days["anchor_day"] == 14]
    gain = pa.daily_gain(main, ["growing_every1", "growing_every14"], "never")
    g1 = gain[gain["policy"] == "growing_every1"]
    expected = PER_DOUBLING * np.log2((14 + g1["days_since_anchor"]) / 14)
    assert abs(g1["gain"].mean() - expected.mean()) < 0.002
    weekly = pa.weekly_gain(gain)
    w = weekly[weekly["policy"] == "growing_every1"].sort_values("week")
    assert w["gain_mean"].iloc[-1] > w["gain_mean"].iloc[0]  # the gain grows with the data
    fit = pa.growth_fit(g1)
    assert fit["spec"] and fit["growth_per_week"] > 0 and fit["growth_low"] > 0


def test_optimal_interval_is_the_longest_within_the_dmd(days):
    skill = pr.patient_skill(days[days["anchor_day"] == 14])
    table = pa.interval_table(skill, [1, 7, 14], FixedDMD(), 500, 1)
    t = table.set_index("interval_days")
    assert t.loc[1, "verdict_to_daily"] == "reference" and t.loc[7, "verdict_to_daily"] == "tie"
    assert t.loc[14, "verdict_to_daily"] == "worse"
    assert t.loc[14, "share_of_daily_gain"] == pytest.approx(0.6, abs=0.05)
    optimal = pa.optimal_intervals(table)
    assert optimal["optimal_interval_days"].iloc[0] == 7 and optimal["model_interval_days"].iloc[0] == 7
    conf = pa.confirmation_policies(optimal, ["m_ft"], {"m_ft": []})
    assert conf["m_ft"]["policies"] == ["never", "growing_every7", "growing_every1"]


def test_later_anchors_gain_less_and_follow_the_data_ratio(days):
    pytest.importorskip("statsmodels")
    skill = pr.patient_skill(days, by=("anchor_day",))
    common = pa.common_patients(skill, [14, 42, 70], ["never", "growing_every1"])
    assert len(common[("p", "m_ft")]) == 20
    gains = pr.paired(skill, lambda p: "never", FixedDMD(), 200, 1, by=("anchor_day",))
    g1 = gains[gains["policy"] == "growing_every1"].set_index("anchor_day")["skill_difference"]
    assert list(g1.index) == [14, 42, 70] and g1[14] > g1[42] > g1[70] > 0
    change = pa.anchor_change(skill, "growing_every1", 14, FixedDMD(), 500, 1).set_index("anchor_day")
    assert change.loc[70, "change"] < change.loc[42, "change"] < 0 and change.loc[70, "verdict"] == "negative"
    gain = pa.daily_gain(days, ["growing_every1"], "never")
    fits = pa.data_ratio_fits(gain, {14: 14, 42: 42, 70: 70}).iloc[0]
    assert fits["per_doubling"] == pytest.approx(PER_DOUBLING, abs=0.003)
    for anchor in (42, 70):  # the anchor adds nothing once the data ratio is in the model
        assert abs(fits[f"anchor_{anchor}"]) < 0.004
    curve = pa.ratio_curve(gain, {14: 14, 42: 42, 70: 70})
    assert set(curve["anchor_day"]) == {14, 42, 70} and curve["gain_mean"].notna().all()
