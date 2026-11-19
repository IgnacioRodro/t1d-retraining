"""Confirmation (scripts 50 and 51; D92): what each cohort runs, the forecast lows and the draw of patients."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.eval import confirmation as cf  # noqa: E402


def test_arms_come_from_the_screening():
    policies = ["never", "growing_every3", "growing_every1"]
    assert cf.confirmation_arms(policies, "warm", 3, daily=True) == \
        ["never", "restart_every3", "restart_every1", "warm_every3"]
    assert cf.confirmation_arms(policies, "restart", 3, daily=False) == ["never", "restart_every3"]
    arx = ["never", "growing_every7", "growing_every1", "online"]
    assert cf.confirmation_arms(arx, "online_l1", 7, daily=True) == \
        ["never", "restart_every7", "restart_every1", "online_l1"]
    assert cf.confirmation_arms(["never", "growing_every1"], None, None, daily=False) == ["never", "restart_every1"]
    assert cf.screen_names("restart_every3", 3) == ("growing_every3", "restart")
    assert cf.screen_names("warm_every3", 3) == (None, "warm") and cf.screen_names("online_l1", 3)[0] == "online"


def test_cohorts_without_insulin_run_the_glucose_only_models():
    models = ["lightgbm_ft", "lightgbm_ft_noexog", "ar3", "arx", "arx_noexog"]
    assert cf.models_for_rule(models, "closed_insulin") == models
    assert cf.models_for_rule(models, "open") == ["lightgbm_ft_noexog", "ar3", "arx_noexog"]


def test_forecast_lows_and_draws():
    table = pd.DataFrame({"n": [10], "below70_true": [4], "below70_alarm": [5], "below70_hit": [3],
                          "below54_true": [0], "below54_alarm": [1], "below54_hit": [0]})
    out = cf.with_lows(table)
    assert np.isclose(out["low70_sensitivity_pct"].iloc[0], 75) and np.isclose(out["low70_precision_pct"].iloc[0], 60)
    assert np.isnan(out["low54_sensitivity_pct"].iloc[0]) and "below70_true" not in out
    patients = [f"P{i}" for i in range(50)]
    first = cf.draw_patients(patients, 10, 7, "loop/closed_insulin")
    assert len(first) == 10 and first == cf.draw_patients(list(reversed(patients)), 10, 7, "loop/closed_insulin")
    assert first != cf.draw_patients(patients, 10, 7, "dclp3/sap_open_glucose")
    assert cf.draw_patients(patients, None, 7, "x") == sorted(patients)


def test_concurrent_writes_of_one_cache_entry(tmp_path):
    """Several jobs of one patient write its persistence at once (script 73 failed on the shared temporary file)."""
    from concurrent.futures import ThreadPoolExecutor

    from t1dretrain.sim.cache import cache_valid, read_cache, write_cache

    entry = {"path": str(tmp_path / "persistence" / "p__whole__14.pkl"), "key": "k", "recompute": None}
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: write_cache(entry, {"days": i, "info": []}), range(64)))
    assert cache_valid(entry) and read_cache(entry)["days"] in range(64)
    assert sorted(p.name for p in (tmp_path / "persistence").iterdir()) == ["p__whole__14.pkl"]
