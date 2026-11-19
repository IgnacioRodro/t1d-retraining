"""Stage 1 analysis (script 21, D78), the glucose-only twins (D76, D90) and the TTM checkpoint (D77)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from t1dretrain.eval import ageing_analysis as aa  # noqa: E402
from t1dretrain.sim.stage0 import mechanism_of, population_path, split_name, split_variant  # noqa: E402


def synthetic_days(slope_per_week=-0.01, recency=0.004, data=0.01, patients=24, seed=0):
    """Day metrics of one model in three versions and of persistence, with known effects on the daily skill."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(patients):
        level = 0.18 + rng.normal(0, 0.03)
        for anchor in (14, 42):
            for d in range(84):
                day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=anchor + d)
                base = dict(period="p", patient_id=f"P{p}", anchor_day=anchor, horizon_min=30, day=day, n=48,
                            days_since_anchor=d, fit_day=day)
                rmse_p = 20.0 * np.exp(rng.normal(0, 0.2))
                rows.append({**base, "model": "persistence", "version": "reference", "sse": 48 * rmse_p ** 2,
                             "age_days": d})
                noise = rng.normal(0, 0.04)
                skills = {"frozen": level + slope_per_week * d / 7, "rolling": level + recency,
                          "growing": level + recency + data}
                for version, s in skills.items():
                    rmse = rmse_p * (1 - (s + noise))
                    rows.append({**base, "model": "m_ft", "version": version, "sse": 48 * rmse ** 2,
                                 "age_days": d if version == "frozen" else 0})
    return pd.DataFrame(rows)


def test_mixed_models_recover_the_slope_and_the_version_effects():
    pytest.importorskip("statsmodels")
    days = synthetic_days()
    skill = aa.daily_skill(days, min_origins=12)
    assert set(skill["model"]) == {"m_ft"} and not skill["clipped"].any()
    ageing = aa.ageing_fit(skill)
    assert ageing["spec"] and abs(ageing["slope_per_week"] + 0.01) < 0.002
    assert ageing["slope_low"] < ageing["slope_per_week"] < ageing["slope_high"] < 0
    effects = {r["effect"].split(" ")[0]: r for r in aa.version_fit(skill)}
    assert abs(effects["data"]["difference"] - 0.01) < 0.003
    # recency against a frozen model that ages: the mean over 12 weeks of the frozen loss is added
    assert effects["recency"]["difference"] > 0.004


def test_version_effects_fall_back_to_within_patient_least_squares(monkeypatch):
    pytest.importorskip("statsmodels")
    skill = aa.daily_skill(synthetic_days(patients=10), min_origins=12)
    monkeypatch.setattr(aa, "fit_mixed", lambda *a, **k: (None, "", "not converged"))  # as LightGBM at 60 min
    rows = aa.version_fit(skill)
    effects = {r["effect"].split(" ")[0]: r for r in rows}
    assert {r["spec"] for r in rows} == {"within (patient-clustered)"}
    assert abs(effects["data"]["difference"] - 0.01) < 0.004 and effects["data"]["se"] > 0


def test_useful_life_and_tie_rule():
    life = aa.useful_life(-0.001, -0.0015, -0.0005, dmd=0.003, follow_up_days=84)
    assert life["useful_life_days"] == pytest.approx(21.0) and life["ages_within_follow_up"]
    assert life["useful_life_low"] == pytest.approx(14.0) and life["useful_life_high"] == pytest.approx(42.0)
    flat = aa.useful_life(-0.0002, -0.0010, 0.0006, dmd=0.003, follow_up_days=84)
    assert not flat["ages_within_follow_up"] and np.isinf(flat["useful_life_high"])
    assert aa.verdict(0.002, 0.001, 0.003, 0.003) == "tie" and aa.verdict(0.01, 0.005, 0.015, 0.003) == "positive"


def test_screening_takes_the_best_per_mechanism_and_slower_ageing():
    fresh = pd.DataFrame([{"period": "p", "model": m, "version": v, "horizon_min": 30, "skill": s}
                          for m, v, s in [("a_ft", "growing", 0.22), ("b_ft", "growing", 0.21),
                                          ("c_ft", "growing", 0.15), ("d_ft", "growing", 0.12), ("ar3", "growing", 0.15),
                                          ("bad", "rolling", -0.1), ("a_ft", "rolling", 0.2)]])
    slopes = pd.DataFrame([{"model": m, "horizon_min": 30, "slope_per_week": s, "slope_se": 0.0002}
                           for m, s in [("a_ft", -0.002), ("b_ft", -0.002), ("c_ft", -0.0021), ("d_ft", 0.0),
                                        ("ar3", -0.0005), ("bad", 0.0)]])
    mechanisms = {"a_ft": "global_finetune", "b_ft": "global_finetune", "c_ft": "global_finetune",
                  "d_ft": "global_finetune", "ar3": "re_estimation", "bad": "re_estimation"}
    table = aa.screening(fresh, slopes, mechanisms, per_mechanism=2).set_index("model")
    assert table.loc["a_ft", "passes"] and table.loc["b_ft", "passes"] and table.loc["ar3", "passes"]
    assert not table.loc["c_ft", "passes"]  # third of its mechanism and ages as fast as the best
    assert table.loc["d_ft", "passes"] and "slowly" in table.loc["d_ft", "decision"]
    assert not table.loc["bad", "passes"] and table.loc["a_ft", "best_fresh_version"] == "growing"


def test_twin_names_and_files():
    assert split_variant("lightgbm_ft_noexog") == ("lightgbm_ft", "noexog")
    assert split_name("lstm_ft_noexog") == ("lstm", "finetune") and split_name("arx_noexog") == ("arx", "")
    assert population_path("x", "lstm", 2, "noexog").name == "lstm_noexog_seed2.pt"
    assert population_path("x", "lightgbm", 1, "noexog").name == "lightgbm_noexog_seed1"
    assert mechanism_of("arx_noexog") == "sample_wise" and mechanism_of("tide_ft") == "global_finetune"


def test_every_model_with_inputs_has_a_glucose_only_twin(tmp_path):
    """D90: the twins of the models that go on to stages 1-3, their availability and their shared settings."""
    from t1dretrain.sim.stage0 import per_model, twin_available, twin_families, twin_of, with_twins

    families = ["lightgbm", "nhits", "tide", "tsmixerx", "lstm"]
    models = ["ar3", "arx", "lightgbm_ft", "nhits_ft", "patchtst_ft", "lightgbm", "tide_pop", "tsmixerx_ft"]
    assert with_twins(models, families) == ["ar3", "arx", "arx_noexog", "lightgbm_ft", "lightgbm_ft_noexog",
                                            "nhits_ft", "nhits_ft_noexog", "patchtst_ft", "lightgbm", "tide_pop",
                                            "tsmixerx_ft", "tsmixerx_ft_noexog"]
    assert with_twins(["arx", "arx_noexog"], families) == ["arx", "arx_noexog"]   # listed already: kept once
    assert twin_of("lightgbm_ft", None) is None and with_twins(models, None) == models   # twins switched off
    assert twin_of("lstm_ft_noexog", families) is None
    assert twin_families({"glucose_only_twins": {"enabled": True, "families": ["lstm"]}}) == ["lstm"]
    assert twin_families({"glucose_only_twins": {"enabled": False, "families": ["lstm"]}}) is None
    assert twin_families({}) == []                                                    # ARX's twin only
    assert per_model({"tide_ft": 7}, "tide_ft_noexog") == 7 and per_model({}, "arx_noexog") == 1
    assert not twin_available("lstm_ft_noexog", tmp_path, [1, 2])
    for seed in (1, 2):
        population_path(tmp_path, "lstm", seed, "noexog").write_bytes(b"")
    assert twin_available("lstm_ft_noexog", tmp_path, [1, 2]) and twin_available("arx_noexog", tmp_path, [1])
    assert mechanism_of("lightgbm_ft") == "continue_training" and mechanism_of("ar3") == "re_estimation"


def test_glucose_only_models_ignore_the_exogenous_inputs():
    from test_learned import pool_data

    from t1dretrain.models.arx import RecursiveARX, arx_features
    from t1dretrain.models.base import TrainData

    s, held = pool_data(2, days=4)
    rng = np.random.default_rng(0)
    for name in ("iob", "cob", "insulin_activity", "carb_absorption"):  # the toy series has flat inputs
        s.exog[name] = rng.normal(1.0, 0.5, np.shape(s.exog[name])).astype(np.float32)
    assert arx_features(s, exog=False).shape[1] == arx_features(s).shape[1] - 4
    train = TrainData(s, np.zeros(0), np.zeros(0), s.step_min, validation=held)
    full, plain = RecursiveARX(), RecursiveARX(exog=False)
    full.fit(train), plain.fit(train)
    changed = s.subset(np.arange(s.n))
    for name in ("iob", "cob", "insulin_activity", "carb_absorption"):
        changed.exog[name] = np.asarray(changed.exog[name]) + 5.0
    np.testing.assert_allclose(plain.predict(changed), plain.predict(s))
    assert not np.allclose(full.predict(changed), full.predict(s))
    pytest.importorskip("lightgbm")
    from t1dretrain.models.gbdt import LightGBMDirect, features

    assert features(s, exog=False).shape[1] == s.inputs.shape[1]
    model = LightGBMDirect(1, {"exog": False, "max_rounds": 20})
    model.fit(train)
    np.testing.assert_allclose(model.predict(changed), model.predict(s))


def test_ttm_loads_the_named_checkpoint(monkeypatch):
    pytest.importorskip("tsfm_public")
    import importlib

    library = importlib.import_module("tsfm_public.toolkit.get_model")  # a module and a function share the name

    from t1dretrain.models import foundation

    calls = []

    def fake(model_path, **options):
        calls.append(options)
        from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction

        return TinyTimeMixerForPrediction(TinyTimeMixerConfig(context_length=64, prediction_length=12,
                                                              num_input_channels=1))

    monkeypatch.setattr(library, "get_model", fake)
    foundation._load("ttm", "ibm-granite/granite-timeseries-ttm-r2", "cpu", 864, 24, 32, "512-96-r2")
    foundation._load("ttm", "ibm-granite/granite-timeseries-ttm-r2", "cpu", 864, 24, 32, "1024-96-r2")
    assert calls[0]["model_revision"] == "main" and calls[1]["model_revision"] == "1024-96-r2"


def test_ageing_against_a_model_that_cannot_age():
    """A frozen model losing 0.2 points a week, on days whose difficulty moves both models together: the slope against
    the reference finds the loss with a narrower interval than the slope of the skill alone; the reference's twin
    (another fixed-context model) gives about 0."""
    rng = np.random.default_rng(3)
    rows = []
    for p in range(30):
        level = 0.18 + rng.normal(0, 0.02)
        for day in range(84):
            shared = rng.normal(0, 0.05) + 0.03 * np.sin(day / 9 + p)  # days that are hard for every model
            week = day // 7
            values = {"chronos2 14 d": level + shared, "timesfm 14 d": level - 0.01 + shared + rng.normal(0, 0.005),
                      "lightgbm_ft never": level + 0.02 - 0.002 * day / 7 + shared + rng.normal(0, 0.005)}
            for name, skill in values.items():
                model, policy = name.split(" ", 1)
                rows.append({"label": name, "model": model, "policy": "context" if "d" in policy else policy,
                             "patient_id": f"P{p}", "horizon_min": 30, "day": day, "week": week, "skill": skill})
    slopes, curve = aa.against_reference(pd.DataFrame(rows), "chronos2 14 d", replicates=400, seed=1)
    s = slopes.set_index("label")
    frozen, twin = s.loc["lightgbm_ft never"], s.loc["timesfm 14 d"]
    assert frozen["slope_per_week"] == pytest.approx(-0.2, abs=0.03)
    assert frozen["high"] < 0 and frozen["change_12_weeks"] == pytest.approx(12 * frozen["slope_per_week"])
    assert frozen["high"] - frozen["low"] < (frozen["raw_high"] - frozen["raw_low"]) / 3
    assert abs(twin["slope_per_week"]) < 0.03 and twin["low"] < 0 < twin["high"]
    assert set(curve["label"]) == {"lightgbm_ft never", "timesfm 14 d"} and curve["week"].nunique() == 12
