"""Stage 3 (script 40; D86): the arms share the fit at the anchor, cover the same origins and keep their fits apart."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from t1dretrain.sim import mechanisms as mech  # noqa: E402

SETTINGS = {"horizons_min": [30, 60], "lookback_hours": 2.0, "origin_stride_min": 30.0, "train_stride_min": 15.0,
            "min_window_share": 0.8, "recent_complete_min": 30.0, "window_days": 14, "test_days": 21,
            "validation_days": 2, "clarke_point_sample": 100, "seeds": [1], "arx": {},
            "lightgbm": {"max_rounds": 40}, "finetune": {"lightgbm": {"max_rounds": 20}}}


def job_for(tmp_path, model, arms, interval=7, gap=None):
    from test_policies import features_store

    anchor = features_store(tmp_path, gap=gap)
    return {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "post_randomization",
            "anchor_day": 14, "anchor": anchor, "model": model, "arms": arms, "interval": interval,
            "settings": SETTINGS, "stream": 0}


def test_arm_names_and_models():
    assert mech.parse_arm("online_l0.995") == ("online", 0.995) and mech.parse_arm("warm") == ("warm", None)
    assert mech.parse_arm("leaves_d0") == ("leaves", 0.0)
    for wrong in ("sometimes", "leaves", "online_l0", "leaves_d1.5"):
        with pytest.raises(ValueError):
            mech.parse_arm(wrong)
    settings = {"finetune": {"lstm": {"train": "head", "learning_rate": 3e-5}}, "arx": {"ridge": 1.0}}
    assert mech.arm_model("lstm_ft", "restart_all", settings)[1]["finetune"]["lstm"] == \
        {"train": "all", "learning_rate": 3e-5}
    assert mech.arm_model("lightgbm_ft_noexog", "scratch", settings)[0] == "lightgbm_noexog"
    assert mech.arm_model("lstm_ft", "restart_other", settings)[1]["finetune"]["lstm"]["train"] == "all"
    other = {**settings, "finetune": {"tide": {"train": "all"}}}
    assert mech.arm_model("tide_ft_noexog", "restart_other", other)[1]["finetune"]["tide"]["train"] == "head"
    assert mech.arm_model("arx", "online_l0.99", settings)[1]["arx"] == {"ridge": 1.0, "forgetting": 0.99}
    assert [d.day for d in mech.fit_days_of(pd.Timestamp("2020-01-15"), 21, 7)] == [15, 22, 29]


def test_lightgbm_arms_share_the_anchor_fit_and_cover_the_same_origins(tmp_path):
    pytest.importorskip("lightgbm")
    arms = ["never", "restart", "warm", "leaves_d0.9", "leaves_d0", "scratch"]
    r = mech.evaluate_mechanisms(job_for(tmp_path, "lightgbm", arms))
    assert not r.error, r.error
    d = r.days
    cover = {a: set(zip(d.loc[d["policy"] == a, "day"], d.loc[d["policy"] == a, "horizon_min"])) for a in arms}
    assert all(cover[a] == cover["never"] for a in arms)
    by = d.set_index(["policy", "day", "horizon_min"])["sse"]
    first = (pd.Timestamp("2020-01-15"), 30)
    assert by[("never", *first)] == by[("restart", *first)] == by[("warm", *first)] == by[("leaves_d0", *first)]
    later = (pd.Timestamp("2020-01-30"), 60)
    anchored = ("never", "restart", "warm", "leaves_d0.9", "leaves_d0")
    assert len({round(by[(a, *later)], 3) for a in anchored}) == 5  # all different, the two decays too
    info = pd.DataFrame(r.info)
    assert set(info.loc[info["fit_day"] == pd.Timestamp("2020-01-15").date(), "weighting"]) >= {"anchor"}
    later_fits = info[info["fit_day"] > pd.Timestamp("2020-01-15").date()]
    assert set(later_fits["weighting"]) == {"restart", "warm", "leaves_d0.9", "leaves_d0", "scratch"}  # kept apart
    assert (info["compute_seconds"] >= 0).all() and len(r.refits["restart|1"]) == 3


def test_arx_forgetting_arms(tmp_path):
    r = mech.evaluate_mechanisms(job_for(tmp_path, "arx", ["never", "online_l0.99", "online_l0.999", "online_l1"]))
    assert not r.error, r.error
    sse = r.days[r.days["policy"] != "reference"].groupby("policy")["sse"].sum()
    assert sse["online_l0.999"] < sse["never"] and len(set(sse.round(3))) == 4
    info = pd.DataFrame(r.info)
    assert set(info["weighting"]) == {"anchor", "online_l0.99", "online_l0.999", "online_l1"}


def test_no_model_at_the_anchor_leaves_every_arm_out(tmp_path):
    r = mech.evaluate_mechanisms(job_for(tmp_path, "arx", ["never", "online_l0.999"], gap=(0, 13)))
    assert not r.error, r.error
    assert set(r.days["policy"]) == {"reference"}
    assert pd.DataFrame(r.info)["excluded"].notna().all()


def test_a_network_continues_its_own_training(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    job = job_for(tmp_path, "nhits", ["never", "restart", "warm"], interval=10)
    job["settings"] = {**SETTINGS, "nhits": {"max_steps": 10, "mlp_units": 16, "val_every": 5},
                       "finetune": {"nhits": {"max_steps": 10, "val_every": 5}}}
    r = mech.evaluate_mechanisms(job)
    assert not r.error, r.error
    info = pd.DataFrame(r.info)
    warm = info[(info["policy"] == "warm") & (info["weighting"] == "warm")]
    assert len(warm) == 2 and (warm["start"] == "warm").all()
    assert "error" not in warm or warm["error"].isna().all()
    assert np.isfinite(r.days["sse"]).all()


def test_an_arm_can_name_its_own_interval(tmp_path):
    """The confirmation (script 50): restart every 7 days and every day in one job, without a job interval."""
    assert mech.split_interval("restart_every3") == ("restart", 3) and mech.split_interval("warm") == ("warm", None)
    assert mech.parse_arm("restart_other_every14") == ("restart_other", None)
    assert mech.parse_arm("leaves_d0.5_every3") == ("leaves", 0.5)
    for wrong in ("never_every3", "online_l1_every7"):
        with pytest.raises(ValueError):
            mech.parse_arm(wrong)
    job = {**job_for(tmp_path, "arx", ["never", "restart_every7", "restart_every1", "online_l1"]), "interval": None}
    r = mech.evaluate_mechanisms(job)
    assert not r.error, r.error
    assert len(r.refits["restart_every7|0"]) == 3 and len(r.refits["restart_every1|0"]) == 21
    d = r.days.set_index(["policy", "day", "horizon_min"])["sse"]
    first = (pd.Timestamp("2020-01-15"), 30)
    assert d[("never", *first)] == d[("restart_every7", *first)] == d[("restart_every1", *first)]
    no_interval = mech.evaluate_mechanisms({**job, "arms": ["restart"]})  # a refitting arm needs an interval
    assert "needs a refit interval" in no_interval.error


def test_the_confirmation_takes_the_best_arm_that_never_loses():
    """D91: the arm with the largest gain over never among those that beat the reference and never lose to it."""
    from t1dretrain.eval.mechanism_analysis import confirmation_mechanisms

    rows = []
    for arm, verdicts in {"warm": ("better", "better", "tie"), "restart_other": ("worse", "worse", "worse"),
                          "big": ("better", "worse", "better")}.items():
        rows += [{"period": "p", "model": "lstm_ft", "policy": arm, "horizon_min": h, "verdict": v}
                 for h, v in zip((30, 60, 120), verdicts)]
    rows += [{"period": "p", "model": "tide_ft", "policy": "warm", "horizon_min": h, "verdict": "tie"}
             for h in (30, 60, 120)]
    vs_reference = pd.DataFrame(rows)
    table = pd.DataFrame({"period": "p", "model": ["lstm_ft"] * 4 + ["tide_ft"] * 2,
                          "arm": ["restart", "warm", "restart_other", "big", "restart", "warm"],
                          "gain_over_never": [0.020, 0.024, 0.017, 0.030, 0.019, 0.012]})
    chosen = confirmation_mechanisms(vs_reference, table, ["lstm_ft", "lstm_ft_noexog", "tide_ft"],
                                     {"lstm_ft": "restart", "tide_ft": "restart"}, "p")
    assert chosen["lstm_ft"]["arm"] == "warm" and chosen["lstm_ft"]["twin"] == "lstm_ft_noexog"
    assert chosen["tide_ft"]["arm"] == "restart" and "lstm_ft_noexog" not in chosen


def test_incremental_arms_continue_on_the_last_days_and_can_skip(tmp_path, monkeypatch):
    """IS-LSTM (Shen and Kleinberg 2025; script 75, D103): the model in use continued on the last 7 days every 7
    days; with the t-test, a refit is skipped (the model kept) when the glucose of the week did not change."""
    pytest.importorskip("lightgbm")
    assert mech.parse_arm("incremental_w7") == ("incremental", (7, None))
    assert mech.parse_arm("incremental_w7_t0.05_every7") == ("incremental", (7, 0.05))
    for wrong in ("incremental_w0", "incremental_w7_t1.5", "incremental_wx"):
        with pytest.raises(ValueError):
            mech.parse_arm(wrong)
    arms = ["never", "warm", "incremental_w7", "incremental_w7_t0.05"]
    r = mech.evaluate_mechanisms(job_for(tmp_path, "lightgbm", arms))
    assert not r.error, r.error
    info = pd.DataFrame(r.info)
    always = info[(info["policy"] == "incremental_w7") & (info["weighting"] == "incremental_w7")]
    assert len(always) == 2 and (always["train_days"] <= 7).all() and (~always["skipped"].astype(bool)).all()
    warm = info[(info["policy"] == "warm") & (info["weighting"] == "warm")]
    assert (warm["train_days"] > always["train_days"].to_numpy()).all()  # warm reads the growing window
    tested = info[(info["policy"] == "incremental_w7_t0.05") & (info["weighting"] == "incremental_w7_t0.05")]
    assert len(tested) == 2 and tested["p_value"].notna().all()
    d = r.days.set_index(["policy", "day", "horizon_min"])["sse"]
    first = (pd.Timestamp("2020-01-15"), 30)
    assert d[("never", *first)] == d[("incremental_w7", *first)] == d[("incremental_w7_t0.05", *first)]

    monkeypatch.setattr(mech, "glucose_changed", lambda *args: (False, 0.9))  # no change: every refit skipped
    r = mech.evaluate_mechanisms(job_for(tmp_path, "lightgbm", ["never", "incremental_w7_t0.05"]))
    assert not r.error, r.error
    info = pd.DataFrame(r.info)
    skipped = info[info["weighting"] == "incremental_w7_t0.05"]
    assert skipped["skipped"].astype(bool).all() and (skipped["compute_seconds"] == 0).all()
    sse = r.days[r.days["policy"] != "reference"].groupby("policy")["sse"].sum()
    assert sse["incremental_w7_t0.05"] == pytest.approx(sse["never"])  # the anchor model served throughout
    assert r.refits["incremental_w7_t0.05|1"] == ["2020-01-15"]


def test_the_week_test_sees_a_shift_and_not_noise():
    class Series:
        def __init__(self, values, start):
            self.values = np.asarray(values, float)
            self.day = (pd.Timestamp(start) + pd.to_timedelta(np.arange(len(values)) // 288, unit="D")).values

        def between(self, start, end):
            keep = (self.day >= np.datetime64(start)) & (self.day < np.datetime64(end))
            return np.where(keep, self.values, np.nan), None, 5.0

    rng = np.random.default_rng(0)
    flat = Series(140 + rng.normal(0, 30, 288 * 14), "2020-01-01")
    shifted = Series(np.r_[140 + rng.normal(0, 30, 288 * 7), 170 + rng.normal(0, 30, 288 * 7)], "2020-01-01")
    day = pd.Timestamp("2020-01-15")
    assert mech.glucose_changed(shifted, day, 7, 0.05)[0]
    changed, p = mech.glucose_changed(flat, day, 7, 1e-6)
    assert not changed and p > 1e-6
