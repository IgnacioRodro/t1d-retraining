"""Recovery after real changes (script 72; D97): events, weekly gains, change against control, trained after."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.eval import recovery as rc  # noqa: E402

CHANGE = pd.Timestamp("2020-02-01")


def days_of(cohort: str, patients: int, jump: float, refit_cut: float) -> pd.DataFrame:
    """Day metrics after a change: never's error is ``jump`` higher than persistence-relative baseline; the daily
    refit removes ``refit_cut`` of the extra error from its second day on."""
    rows = []
    for p in range(patients):
        for d in range(14):
            day = CHANGE + pd.Timedelta(days=d)
            for model, policy, sse in (("persistence", "reference", 400.0), ("m", "never", 200.0 + jump),
                                       ("m", "restart_every1", 200.0 + (jump * (1 - refit_cut) if d else jump))):
                rows.append({"cohort": cohort, "period": "whole", "patient_id": f"{cohort}-{p}", "anchor_day": 30,
                             "horizon_min": 60, "day": day, "model": model, "policy": policy, "n": 40,
                             "sse": 40 * (sse + p), "fit_day": day if policy == "restart_every1" else CHANGE,
                             "age_days": 0 if policy == "restart_every1" else d, "days_since_anchor": d})
    return pd.DataFrame(rows)


def test_select_events():
    events = pd.DataFrame({"patient_id": ["a", "b", "c", "d"], "kind": ["loop_change"] * 3 + ["randomization"],
                           "valid_days_before": [20, 5, 20, 20], "valid_days_after": [20, 20, 20, 20]})
    arms = pd.Series({"a": "CLC", "b": "CLC", "c": "SAP", "d": "CLC"})
    chosen, counts = rc.select_events(events, {"kind": "loop_change", "arm": "CLC"}, arms, None, 14)
    assert chosen["patient_id"].tolist() == ["a"] and counts == {"events": 3, "in_arm": 2, "enough_days": 1}
    chosen, counts = rc.select_events(events, {"kind": "loop_change"}, arms, ["a", "c"], 14)
    assert chosen["patient_id"].tolist() == ["a", "c"] and counts["in_group"] == 2


def test_weekly_gain_and_contrast():
    days = pd.concat([days_of("change", 6, 150.0, 0.8), days_of("control", 5, 0.0, 0.0)], ignore_index=True)
    skill = rc.daily(days, min_origins=12, clip=(-1.0, 1.0))
    assert set(skill["event"].str.split("@").str[1]) == {"30"} and set(skill["week"]) == {0, 1}
    gains = rc.daily_gains(skill)
    weekly = rc.weekly_gain(gains)
    change = weekly[(weekly["cohort"] == "change")].set_index("week")["gain"]
    assert (change > 0).all() and (weekly.loc[weekly["cohort"] == "control", "gain"].abs() < 1e-9).all()
    contrast = rc.group_contrast(rc.event_measures(skill, gains), "change", "control", 300, 1)
    never = contrast[contrast["measure"] == "skill"].iloc[0]
    gain = contrast[contrast["measure"] == "gain"].iloc[0]
    assert never["difference"] < 0 and never["high"] < 0
    assert gain["difference"] > 0 and gain["low"] > 0 and gain["change_events"] == 6
    weekly_skill = rc.weekly_skill(skill)
    assert {"events", "skill", "low", "high"} <= set(weekly_skill.columns)


def test_against_trained_after():
    before = rc.daily(days_of("change", 4, 150.0, 0.8), 12, (-1.0, 1.0))
    after_days = days_of("change", 4, 0.0, 0.0)
    after = rc.daily(after_days[after_days["policy"] != "restart_every1"], 12, (-1.0, 1.0))
    table = rc.against_trained_after(before, after, 300, 1).set_index(["policy", "week"])
    assert table.loc[("never", 0), "difference"] < 0 and table.loc[("never", 0), "patients"] == 4
    assert table.loc[("restart_every1", 1), "difference"] > table.loc[("never", 1), "difference"]
    assert np.isfinite(table["low"]).all()
