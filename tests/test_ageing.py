"""Stage 1 (script 20): the plan of fits and the rule that a refit never learns from a test day."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.sim.ageing import plan_fits  # noqa: E402
from t1dretrain.sim.samples import known_before, make_samples  # noqa: E402

A = pd.Timestamp("2020-03-01")


def test_frozen_rolling_and_growing_plans():
    frozen = plan_fits(A, 84, 14, 1, "frozen")
    assert len(frozen) == 1 and frozen[0].train_start == A - pd.Timedelta(days=14)
    assert frozen[0].test_end == A + pd.Timedelta(days=84)
    rolling = plan_fits(A, 84, 14, 7, "rolling")
    assert len(rolling) == 12 and rolling[-1].fit_day == A + pd.Timedelta(days=77)
    assert all(f.fit_day - f.train_start == pd.Timedelta(days=14) for f in rolling)     # always the last 14 days
    growing = plan_fits(A, 84, 14, 1, "growing")
    assert len(growing) == 84 and {f.train_start for f in growing} == {A - pd.Timedelta(days=14)}
    covered = [f.fit_day + pd.Timedelta(days=k) for f in rolling for k in range((f.test_end - f.fit_day).days)]
    assert covered == list(pd.date_range(A, periods=84, freq="D"))                        # every test day exactly once


def test_training_targets_end_before_the_fit_day():
    n = 3 * 288
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) * 5, unit="min")
    grid = pd.DataFrame({"patient_id": "p", "slot": np.arange(n), "ts": ts, "glucose": 120.0, "target_ok": True,
                         "glucose_input": 120.0, "segment": 0})
    grid["day"] = grid["ts"].dt.normalize()
    s = make_samples(grid, [30, 120], lookback_h=1, stride_min=5)
    end = pd.Timestamp("2020-01-03")
    keep = s.subset(known_before(s, end))
    last_target = keep.origin_ts.astype("datetime64[ns]") + np.timedelta64(120, "m")
    assert (last_target < np.datetime64(end)).all()
    assert keep.n == ((s.origin_ts.astype("datetime64[ns]") + np.timedelta64(120, "m")) < np.datetime64(end)).sum()
    assert (keep.origin_ts.astype("datetime64[ns]") < np.datetime64(end - pd.Timedelta(minutes=120))).all()
