"""Validation of the method (script 93, D112): designs against the simulator, the ageing estimator, the verdicts with
the DMD, the paired design, and the comparison of runs."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from t1dretrain.eval import method_validation as mv  # noqa: E402
from t1dretrain.sim import naive_splits as ns  # noqa: E402


def toy_job(tmp_path, model="arx", **settings):
    from test_policies import SETTINGS, features_store

    anchor = features_store(tmp_path, days=40)
    return {"dataset": "toy", "derived_root": tmp_path, "patient_id": "p", "period": "post_randomization",
            "anchor_day": 14, "anchor": anchor, "model": model, "settings": {**SETTINGS, **settings}, "stream": 0,
            "refit_every": 7, "share": 0.2, "split_seed": 1}


def test_designs_hold_out_what_they_say(tmp_path):
    from t1dretrain.sim.policies import JobData

    data = JobData(toy_job(tmp_path))
    for split in ns.SPLITS:
        d = ns.held_out(data, split, 0.2, ns.split_rng(1, "p", split))
        held, train = d["held"], d["train"]
        assert held.any() and train.any() and not train.all()
        if split == "random_sample":  # the held-out origins are not trained on, their neighbours are
            assert not np.isin(data.pool.origin_ts[train], data.test.origin_ts[held]).any()
            assert np.isin(data.pool.day[train], np.unique(data.test.day[held])).any()
        if split == "random_day":  # no sample of a held-out day; days after them are trained on
            assert not np.isin(data.pool.day[train], np.unique(data.test.day[held])).any()
            assert (data.pool.day[train] > data.test.day[held].min()).any()
        if split == "chronological":  # every target of the training samples precedes the held-out days
            last = data.pool.origin_ts[train].astype("datetime64[ns]") + np.timedelta64(60, "m")
            assert (last < np.datetime64(d["cut"].to_datetime64(), "ns")).all()
            assert (data.test.day[held] >= np.datetime64(d["cut"].to_datetime64(), "ns")).all()
    again = ns.held_out(data, "random_day", 0.2, ns.split_rng(1, "p", "random_day"))["held"]
    assert np.array_equal(again, ns.held_out(data, "random_day", 0.2, ns.split_rng(1, "p", "random_day"))["held"])


@pytest.mark.parametrize("model", ["arx", "lightgbm"])
def test_every_design_is_scored_with_the_simulator_on_its_origins(tmp_path, model):
    if model == "lightgbm":
        pytest.importorskip("lightgbm")
    job = toy_job(tmp_path, model, lightgbm={"max_rounds": 30}, test_days=21)
    r = ns.evaluate_splits(job)
    assert not r.error, r.error
    d = r.days
    assert set(d["split"]) == set(ns.SPLITS) and set(d["arm"]) == set(ns.ARMS)
    n = d.groupby(["split", "arm", "horizon_min"])["n"].sum().unstack("arm")
    assert (n.nunique(axis=1) == 1).all()  # every arm on the same origins
    sse = d.groupby(["split", "arm"])["sse"].sum().unstack("arm")
    # the record of 35 days, last 20 % = 7 days: the cut is day 28 = anchor + 14, a refit day of the refitted model
    assert sse.loc["chronological", "design"] == pytest.approx(sse.loc["chronological", "refitted"])
    info = pd.DataFrame(r.info)
    sim = info[info["split"] == "simulator"].iloc[0]
    assert sim["fits"] == 3 and sim["fits_failed"] == 0  # anchor, +7 and +14 days
    chrono = info[info["split"] == "chronological"].iloc[0]
    assert chrono["train_days_after_first_held"] == 0
    assert info.loc[info["split"] == "random_day", "train_days_after_first_held"].iloc[0] > 0
    with pytest.raises(ValueError):
        ns.check_model("nhits_ft")


def test_split_skill_and_inflation():
    rows = []
    rng = np.random.default_rng(0)
    for p in range(12):
        for arm, sse in (("persistence", 100.0), ("frozen", 64.0), ("refitted", 49.0), ("design", 36.0)):
            rows.append({"model": "m", "split": "random_sample", "arm": arm, "horizon_min": 30, "patient_id": f"P{p}",
                         "day": 0, "n": 10, "sse": 10 * sse * (1 + 0.05 * rng.normal()), "zone_A": 9, "zone_B": 1,
                         "zone_C": 0, "zone_D": 0, "zone_E": 0})
    skill = mv.split_skill(pd.DataFrame(rows))
    assert set(skill["arm"]) == {"frozen", "refitted", "design"} and (skill["clarke_AB_pct"] == 100).all()
    t = mv.inflation(skill, 200, 1).iloc[0]
    assert t["skill_design"] == pytest.approx(0.4, abs=0.03) and t["skill_frozen"] == pytest.approx(0.2, abs=0.03)
    assert t["minus_frozen"] > t["minus_refitted"] > 0 and t["patients_higher_than_frozen"] == 12


def test_ageing_estimator_is_unbiased_and_narrower_only_with_shared_noise():
    base = {"seed": 1, "repetitions": 40, "patients": 30, "days": 84, "slope_pp": -0.2, "day_sd": 0.1,
            "reference_sd": 0.1, "phi": 0.2, "slope_sd_pp": 0.0, "replicates": 300, "min_weeks": 6}
    summary = mv.simulation_summary(pd.DataFrame(mv.ageing_case({**base, "correlation": 0.9})
                                                 + mv.ageing_case({**base, "correlation": 0.0})))
    s = summary.set_index(["correlation", "label"])
    high, none = s.loc[(0.9, "model")], s.loc[(0.0, "model")]
    assert abs(high["adjusted_bias"]) < 0.03 and abs(none["adjusted_bias"]) < 0.04
    assert high["width_ratio"] < 0.6 < 1.2 < none["width_ratio"]  # about sqrt(2 (1 - r)): 0.45 and 1.41
    assert high["adjusted_coverage"] >= 0.85 and s.loc[(0.9, "control"), "adjusted_excludes_zero"] <= 0.2
    assert high["adjusted_excludes_zero"] > high["raw_excludes_zero"]


def test_variance_components_find_the_noise_of_the_generator():
    rng = np.random.default_rng(4)
    daily = mv.simulate_daily(rng, 40, 84, -0.2, 0.7, 0.1, 0.08, 0.3, 0.3)
    daily["horizon_min"] = 60
    c = mv.variance_components(daily, "model", "reference").iloc[0]
    assert c["correlation"] == pytest.approx(0.7, abs=0.06)
    assert c["day_sd"] == pytest.approx(0.1, rel=0.12) and c["reference_day_sd"] == pytest.approx(0.08, rel=0.12)
    assert c["phi"] == pytest.approx(0.3, abs=0.12) and c["slope_sd_pp"] == pytest.approx(0.3, abs=0.15)


def test_real_width_ratio():
    ageing = pd.DataFrame({"label": ["lightgbm_ft never", "chronos2 1 d"], "model": ["lightgbm_ft", "chronos2"],
                           "policy": ["never", "context"], "horizon_min": [30, 30], "patients": [38, 38],
                           "slope_per_week": [-0.1, 0.0], "low": [-0.2, -0.05], "high": [0.0, 0.05],
                           "raw_slope_per_week": [-0.1, 0.0], "raw_low": [-0.3, -0.1], "raw_high": [0.1, 0.1]})
    r = mv.real_width_ratio(ageing).set_index("label")
    assert r.loc["lightgbm_ft never", "width_ratio"] == pytest.approx(0.5) and r.loc["chronos2 1 d", "kind"] == \
        "foundation"


def seeds_table(shift: float = 0.004, patients: int = 40):
    """Three seeds of two models: seed 2 of each a little better on every patient (a training effect)."""
    rng = np.random.default_rng(2)
    rows = []
    for model in ("lightgbm_ft", "lstm_ft"):
        level = rng.normal(0.2, 0.05, patients)
        for seed in (0, 1, 2):
            skill = level + rng.normal(0, 0.003, patients) + (shift if seed == 2 else 0)
            rows += [{"period": "p", "model": model, "seed": seed, "horizon_min": 30, "patient_id": f"P{i}",
                      "skill": s} for i, s in enumerate(skill)]
    return pd.DataFrame(rows)


def test_seed_comparisons_tell_the_rule_from_the_interval():
    table = seeds_table()
    noise = pd.DataFrame({"period": ["p", "p"], "model": ["lightgbm_ft", "lstm_ft"], "horizon_min": [30, 30],
                          "dmd": [0.004, 0.004]})
    c = mv.seed_comparisons(table, noise, [0, 1, 3], [1.0], 300, 1)
    assert len(c) == 2 * 3 * 4 and set(c["rule"]) <= {"tie", "better", "worse", "uncertain"}
    calibration = mv.calibration_summary(c).set_index("family")
    assert calibration.loc["all", "interval_false_finding"] > calibration.loc["all", "rule_false_finding"]
    power = mv.power_summary(c).set_index(["effect_kind", "effect"])
    assert power.loc[("dmd", 3.0), "rule_better"] == 1.0 and power.loc[("dmd", 0.0), "rule_better"] < 1.0
    assert power.loc[("pp", 1.0), "rule_better"] == 1.0


def test_pairing_and_crossover_widths():
    rng = np.random.default_rng(3)
    level = rng.normal(0.2, 0.05, 40)
    skill = pd.DataFrame({"group": "g", "patient_id": np.tile([f"P{i}" for i in range(40)], 2),
                          "model": np.repeat(["a", "b"], 40),
                          "skill": np.concatenate([level + 0.01 + rng.normal(0, 0.005, 40),
                                                   level + rng.normal(0, 0.005, 40)])})
    p = mv.pairing_efficiency(skill, [("a", "b")], ["group"], 400, 1).iloc[0]
    assert p["width_ratio"] > 3 and p["correlation"] > 0.9 and p["parallel_patients_per_arm"] > 40
    rows = []
    for i in range(30):
        for day in range(20):
            hard = rng.uniform(0.5, 2.0)  # a day that is hard for every arm
            for policy, factor in (("reference", 1.0), ("never", 0.8), ("restart", 0.7)):
                rows.append({"group": "g", "patient_id": f"P{i}", "day": day, "policy": policy, "n": 48,
                             "sse": 48 * 400 * hard * factor * rng.uniform(0.95, 1.05)})
    x = mv.crossover_efficiency(pd.DataFrame(rows), [("restart", "never")], ["group"], 300, 1, draws=5).iloc[0]
    assert x["difference"] > 0 and x["width_ratio"] > 1.5


def test_compare_outputs_and_run_costs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, value in ((a, 1.0), (b, 1.0 + 1e-13)):
        (root / "10_x").mkdir(parents=True)
        pd.DataFrame({"x": [1, 2], "y": ["u", "v"]}).to_csv(root / "10_x" / "same.csv", index=False)
        pd.DataFrame({"x": [value, 2.0]}).to_csv(root / "10_x" / "close.csv", index=False)
        pd.DataFrame({"x": [value * (2 if root == b else 1)]}).to_csv(root / "10_x" / "other.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(a / "10_x" / "only.csv", index=False)
    status = mv.compare_outputs(a, b).set_index("file")["status"]
    assert status["same.csv"] == "identical" and status["close.csv"] == "equal"
    assert status["other.csv"] == "different" and status["only.csv"] == "only in first"
    manifest = {"package_version": "1.0", "duration_seconds": 7.0, "resources": {"device": "cpu", "n_workers": 3}}
    (a / "10_x" / "run_manifest.json").write_text(json.dumps(manifest))
    log = pd.DataFrame({"patient_id": ["p", "p", "p"], "fit_day": ["d1", "d1", "d2"],
                        "policy": ["never", "restart", "restart"], "test_origins": [10, 5, 5],
                        "compute_seconds": [3600.0, 3600.0, 1800.0]})
    log.to_csv(a / "10_x" / "fit_log.csv", index=False)
    cost = mv.run_costs(a).iloc[0]
    assert cost["stage"] == "stage 0" and cost["fits"] == 2 and cost["fit_compute_hours"] == pytest.approx(1.5)
    assert mv.rerun_scripts(a, b) == []  # no manifest in b: not a script that ran again
    (b / "10_x" / "run_manifest.json").write_text(json.dumps({"package_version": "1.0", "duration_seconds": 8.0}))
    assert mv.rerun_scripts(a, b) == ["10_x"]
