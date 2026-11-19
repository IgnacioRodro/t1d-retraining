"""Stage 2 (script 30; D80, D81): policies, shared fits, drift triggers and sample weights."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from t1dretrain.sim import policies as pol  # noqa: E402

A = pd.Timestamp("2020-03-01")


def test_windows_policies_and_fit_days():
    assert [p.name for p in pol.policy_grid(["growing", "rolling7"], [1, 7])] == \
        ["never", "growing_every1", "growing_every7", "rolling7_every1", "rolling7_every7"]
    assert len(pol.policy_grid(["growing", "rolling7", "rolling14", "rolling28", "exp14"], [1, 3, 7, 14])) == 21
    for name in ("never", "rolling14_every3", "exp14_every7", "adwin_growing", "even_page_hinkley_exp14"):
        assert pol.parse_policy(name).name == name
    assert pol.parse_policy("even_adwin_growing").matched == "adwin"
    assert pol.parse_window("rolling28").start(A, A - pd.Timedelta(days=14)) == A - pd.Timedelta(days=14)
    assert pol.parse_window("rolling7").start(A, A - pd.Timedelta(days=14)) == A - pd.Timedelta(days=7)
    w = pol.parse_window("exp14").weights(np.array([A - pd.Timedelta(days=14), A], dtype="datetime64[ns]"), A)
    np.testing.assert_allclose(w, [0.5, 1.0])
    assert len(pol.fixed_fit_days(pol.parse_policy("growing_every7"), A, 84)) == 12
    assert pol.fixed_fit_days(pol.parse_policy("never"), A, 84) == [A]
    even = pol.even_fit_days(3, A, 84)
    assert even == [A] + [A + pd.Timedelta(days=d) for d in (21, 42, 63)]
    assert pol.even_fit_days(0, A, 84) == [A]


def features_store(tmp_path, days=40, change_day=28, gap=None):
    """A toy patient on the features grid: a daily cycle whose amplitude changes at ``change_day``; no glucose on the
    days ``gap`` (first, last) if given."""
    from t1dretrain.data.store import ProcessedStore

    rng = np.random.default_rng(0)
    n = days * 288
    t = np.arange(n) * 5 / 60
    amplitude = np.where(np.arange(n) < change_day * 288, 40, 80)
    values = 150 + amplitude * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5, n)
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) * 5, unit="min")
    hours = (ts - ts.normalize()).total_seconds() / 3600
    grid = pd.DataFrame({"patient_id": "p", "slot": np.arange(n), "ts": ts, "glucose": values, "target_ok": True,
                         "glucose_input": values, "segment": 0, "valid_day": True, "loop": "open",
                         "insulin_known": True, "period": "post_randomization",
                         "iob": rng.normal(1, 0.3, n), "cob": rng.normal(10, 3, n),
                         "insulin_activity": rng.normal(0.5, 0.1, n), "carb_absorption": rng.normal(5, 1, n),
                         "hour_sin": np.sin(2 * np.pi * hours / 24), "hour_cos": np.cos(2 * np.pi * hours / 24)})
    grid["day"] = grid["ts"].dt.normalize()
    if gap:
        off = (grid["day"] >= ts[0] + pd.Timedelta(days=gap[0])) & (grid["day"] <= ts[0] + pd.Timedelta(days=gap[1]))
        grid.loc[off, ["glucose", "glucose_input"]] = np.nan
        grid.loc[off, ["target_ok", "valid_day"]] = False
    ProcessedStore("toy", root=tmp_path, stage="features").write_frame("grid", "p", grid)
    return pd.Timestamp("2020-01-15")  # the anchor: 14 days after the first day


SETTINGS = {"horizons_min": [30, 60], "lookback_hours": 2.0, "origin_stride_min": 30.0, "train_stride_min": 15.0,
            "min_window_share": 0.8, "recent_complete_min": 30.0, "window_days": 14, "test_days": 21,
            "validation_days": 2, "clarke_point_sample": 100, "seeds": [1], "arx": {},
            "triggers": {"adwin": {"delta": 0.5, "clock": 1, "grace_period": 5, "min_window_length": 2},
                         "page_hinkley": {"min_instances": 10, "delta": 0.0, "threshold": 0.5, "mode": "up"}}}
POLICIES = ["never", "growing_every1", "growing_every7", "rolling7_every3", "rolling28_every1", "exp14_every7",
            "adwin_growing", "even_adwin_growing", "page_hinkley_growing", "even_page_hinkley_growing"]


@pytest.mark.parametrize("model", ["ar3", "arx"])
def test_policies_share_fits_cover_every_day_and_match_triggers(tmp_path, model):
    pytest.importorskip("river")
    anchor = features_store(tmp_path)
    job = {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "post_randomization",
           "anchor_day": 14, "anchor": anchor, "model": model, "policies": POLICIES, "settings": SETTINGS,
           "stream": 0}
    r = pol.evaluate_policies(job)
    assert not r.error, r.error
    d = r.days
    assert set(d["policy"]) == set(POLICIES) | {"reference"}
    test_days = sorted(d.loc[d["policy"] == "reference", "day"].unique())
    assert len(test_days) == 21
    for name in POLICIES:  # every policy predicts every test day once per horizon
        p = d[d["policy"] == name]
        assert len(p) == 2 * len(test_days) and not p.duplicated(["day", "horizon_min"]).any()
    by = d.set_index(["policy", "day", "horizon_min"])["sse"]
    first = (anchor, 30)
    assert by[("never", *first)] == by[("growing_every1", *first)] == by[("rolling28_every1", *first)]
    day7 = (anchor + pd.Timedelta(days=7), 60)
    assert by[("growing_every7", *day7)] == pytest.approx(by[("growing_every1", *day7)])  # the same fit
    info = pd.DataFrame(r.info)
    distinct = info.drop_duplicates(["fit_day", "train_start"]).shape[0]
    assert distinct < len(info)  # fits are shared between policies
    for trigger in ("adwin_growing", "page_hinkley_growing"):
        assert len(r.refits[f"even_{trigger}|0"]) == len(r.refits[f"{trigger}|0"])
    if model == "arx":  # ARX (unlike AR(3), close to persistence here) gets worse after the change: detected
        assert len(r.refits["adwin_growing|0"]) > 1


def test_sample_weights_change_the_fits():
    from test_learned import pool_data

    from t1dretrain.models.base import TrainData
    from t1dretrain.models.reference import AutoRegressive

    s, held = pool_data(2, days=4)
    series = np.asarray(s.inputs[:, -1], dtype=float)
    plain = AutoRegressive()
    plain.fit(TrainData(s, series, np.zeros(len(series), int), s.step_min))
    weighted = AutoRegressive()
    w = np.linspace(0.01, 1, len(series))
    weighted.fit(TrainData(s, series, np.zeros(len(series), int), s.step_min, series_weight=w))
    assert not np.allclose(plain.coef, weighted.coef)
    pytest.importorskip("lightgbm")
    from t1dretrain.models.gbdt import LightGBMDirect

    train = TrainData(s, np.zeros(0), np.zeros(0), s.step_min, validation=held)
    a, b = LightGBMDirect(1, {"max_rounds": 30}), LightGBMDirect(1, {"max_rounds": 30})
    a.fit(train)
    b.fit(TrainData(s, np.zeros(0), np.zeros(0), s.step_min, validation=held,
                    sample_weight=np.linspace(0.01, 1, s.n)))
    assert not np.allclose(a.predict(s), b.predict(s))


def toy_job(tmp_path, model, policies, gap):
    anchor = features_store(tmp_path, gap=gap)
    return {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "post_randomization",
            "anchor_day": 14, "anchor": anchor, "model": model, "policies": policies, "settings": SETTINGS,
            "stream": 0}


def test_a_failed_refit_keeps_the_model_in_use(tmp_path):
    job = toy_job(tmp_path, "arx", ["never", "rolling7_every1"], gap=(14, 20))  # no glucose in the first test week
    r = pol.evaluate_policies(job)
    assert not r.error, r.error
    info = pd.DataFrame(r.info)
    rolling = info[info["policy"] == "rolling7_every1"]
    failed = rolling[rolling["error"].notna()]
    assert len(failed) and failed["served_by"].notna().all()  # the day-21 fit has no samples: the old one serves
    d = r.days
    covered = {p: set(d.loc[d["policy"] == p, "day"]) for p in ("reference", "never", "rolling7_every1")}
    assert covered["never"] == covered["rolling7_every1"] == covered["reference"]  # the same origins
    assert (d.loc[d["policy"] == "rolling7_every1", "age_days"] >= 1).any()  # served by an older fit


def test_no_model_at_the_anchor_leaves_the_patient_out(tmp_path):
    r = pol.evaluate_policies(toy_job(tmp_path, "arx", ["never", "growing_every7"], gap=(0, 13)))
    assert not r.error, r.error
    assert set(r.days["policy"]) == {"reference"}
    info = pd.DataFrame(r.info)
    assert len(info) == 2 and info["excluded"].notna().all()
    assert all(v == [] for v in r.refits.values())


def test_a_network_runs_its_policies_and_logs_compute_time(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    job = toy_job(tmp_path, "nhits", ["never", "growing_every7", "exp14_every7"], gap=None)
    job["settings"] = {**SETTINGS, "nhits": {"max_steps": 10, "mlp_units": 16, "val_every": 5}}
    r = pol.evaluate_policies(job)
    assert not r.error, r.error
    info = pd.DataFrame(r.info)
    assert "error" not in info or info["error"].isna().all(), info.get("error")
    assert (info["compute_seconds"] > 0).all()
    assert set(r.days["policy"]) == {"reference", "never", "growing_every7", "exp14_every7"}


def test_online_arx_equals_a_fit_on_what_is_known_at_each_origin():
    from test_learned import pool_data

    from t1dretrain.models.arx import RecursiveARX
    from t1dretrain.models.base import TrainData
    from t1dretrain.sim.samples import known_before

    s, held = pool_data(4, days=5)
    test = s.subset(held)
    model = RecursiveARX(forgetting=0.995, ridge=1.0)
    pred, updates = model.online_predict(s, test)
    assert pred.shape == test.targets.shape and updates > 1
    for row in (0, test.n // 2, test.n - 1):  # the same as a batch fit on the samples known by that origin
        origin = pd.Timestamp(test.origin_ts[row])
        train = s.subset(known_before(s, origin))
        batch = RecursiveARX(forgetting=0.995, ridge=1.0)
        batch.fit(TrainData(train, np.zeros(0), np.zeros(0), s.step_min))
        np.testing.assert_allclose(pred[row], batch.predict(test.subset(np.arange(test.n) == row))[0], rtol=1e-6)


def test_online_policy_runs_for_arx(tmp_path):
    r = pol.evaluate_policies(toy_job(tmp_path, "arx", ["never", "online"], gap=None))
    assert not r.error, r.error
    d = r.days
    assert set(d["policy"]) == {"reference", "never", "online"}
    assert (d.loc[d["policy"] == "online", "age_days"] == 0).all()
    assert len(d[d["policy"] == "online"]) == len(d[d["policy"] == "never"])
    by = d.groupby("policy")["sse"].sum()
    assert by["online"] < by["never"]  # the change at day 28 is followed
    assert pol.online_policies("arx") == ["online"] and pol.online_policies("lightgbm_ft") == []
    assert pol.parse_policy("online").kind == "online"


def test_halving_keeps_the_best_third_and_the_references():
    from t1dretrain.eval import policy_report as pr

    rank = pd.DataFrame({"period": "p", "model": "arx", "policy": ["online", "a", "b", "c", "never"],
                         "skill_mean": [0.30, 0.20, 0.18, 0.15, 0.10], "patients": 10})
    h = pr.halving(rank, keep=2, references={"arx": ["never", "online"]}, outside=["online"]).set_index("policy")
    assert h.loc["a", "rank"] == 1 and h.loc["b", "reason"] == "top third"  # online does not take a place
    assert h.loc["c", "reason"] == "dropped" and h.loc["never", "reason"] == h.loc["online", "reason"] == "reference"


def test_later_anchor_starts_with_every_day_since_day_0(tmp_path):
    """D83: with window_days = the anchor day, the first fit and the growing window start at day 0."""
    job = toy_job(tmp_path, "ar3", ["never", "growing_every7"], gap=None)
    later = pd.Timestamp("2020-01-21")  # day 20
    job.update(anchor=later, anchor_day=20, settings={**SETTINGS, "window_days": 20, "test_days": 14})
    r = pol.evaluate_policies(job)
    assert not r.error, r.error
    info = pd.DataFrame(r.info)
    assert (pd.to_datetime(info["train_start"]) == pd.Timestamp("2020-01-01")).all()
    assert info.loc[info["policy"] == "never", "train_days"].iloc[0] == 20


def test_origins_follow_the_cohort_rule_and_low_glucose_is_counted(tmp_path):
    """The confirmation (script 50): a closed-loop cohort uses closed-loop days only; with event thresholds, the
    day metrics count the forecast lows (D17, HANDOFF 7.8)."""
    from t1dretrain.sim.stage0 import day_metrics

    anchor = features_store(tmp_path)
    settings = {"horizons_min": [30, 60], "lookback_hours": 2.0, "origin_stride_min": 30.0, "train_stride_min": 15.0,
                "min_window_share": 0.8, "recent_complete_min": 30.0, "window_days": 14, "test_days": 21,
                "validation_days": 2, "clarke_point_sample": 10, "seeds": [1]}
    job = {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "post_randomization",
           "anchor_day": 14, "anchor": anchor, "model": "ar3", "settings": settings, "stream": 0}
    assert pol.JobData(job).test.n > 0 and pol.JobData({**job, "usable": "open"}).test.n == pol.JobData(job).test.n
    assert pol.JobData({**job, "usable": "closed_insulin"}).test.n == 0  # the toy patient is in open loop
    reference = np.array([[60.0, 50.0], [120.0, 65.0], [80.0, 100.0]])
    forecast = np.array([[65.0, 52.0], [60.0, 80.0], [90.0, 50.0]])
    table = day_metrics(forecast, reference, np.array(["2020-01-15"] * 3, dtype="datetime64[D]"), [30, 60],
                        (70, 54)).set_index("horizon_min")
    assert table.loc[30, ["below70_true", "below70_alarm", "below70_hit"]].tolist() == [1, 2, 1]
    assert table.loc[60, ["below70_true", "below70_alarm", "below70_hit", "below54_true", "below54_hit"]].tolist() == \
        [2, 2, 1, 1, 1]
    assert "below70_true" not in day_metrics(forecast, reference, np.array(["2020-01-15"] * 3, dtype="datetime64[D]"),
                                             [30, 60]).columns


def test_context_models_read_a_long_history(tmp_path):
    """M2 (script 43, D99): the test samples of a context-only model carry the days of history before each origin,
    also before the anchor; other models carry none."""
    from t1dretrain.data.store import ProcessedStore
    from t1dretrain.sim.policies import JobData

    n = 30 * 288
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) * 5, unit="min")
    values = 140 + 30 * np.sin(np.arange(n) / 50.0)
    grid = pd.DataFrame({"patient_id": "p", "slot": np.arange(n), "ts": ts, "glucose": values, "target_ok": True,
                         "glucose_input": values, "segment": 0, "valid_day": True, "loop": "open",
                         "insulin_known": True, "period": "whole", "iob": 1.0, "cob": 0.0, "insulin_activity": 0.5,
                         "carb_absorption": 0.0, "hour_sin": 0.0, "hour_cos": 1.0})
    grid["day"] = grid["ts"].dt.normalize()
    ProcessedStore("toy", root=tmp_path, stage="features").write_frame("grid", "p", grid)
    settings = {"horizons_min": [30, 60], "lookback_hours": 2.0, "origin_stride_min": 60.0, "train_stride_min": 60.0,
                "min_window_share": 0.8, "recent_complete_min": 30, "window_days": 14, "test_days": 7,
                "clarke_point_sample": 0, "validation_days": 2, "chronos2": {"context_hours": 72.0}}
    job = {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "whole", "anchor_day": 14,
           "anchor": pd.Timestamp("2020-01-15"), "settings": settings, "stream": 0}
    data = JobData({**job, "model": "chronos2"})
    assert data.test.context.shape == (data.test.n, 72 * 12)
    assert np.isfinite(data.test.context[0]).all()  # the first origin reads three days before the anchor
    assert JobData({**job, "model": "ar3"}).test.context is None
