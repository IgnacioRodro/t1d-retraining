"""Supplement tables (script 92): the refitted arm of each model, the oriented regularity intervals and the cells."""

from __future__ import annotations

import numpy as np
import pandas as pd

from t1dretrain.eval import supplement_tables as st


def replication() -> pd.DataFrame:
    rows = []
    for model, finding, policy in (("lightgbm_ft", "gain_over_never", "restart_every3"),
                                   ("lstm_ft", "gain_over_never", "restart_every3"),
                                   ("lstm_ft", "mechanism_vs_restart", "warm_every3"),
                                   ("lstm_ft_noexog", "mechanism_vs_restart", "warm_every3"),
                                   ("arx", "gain_over_never", "restart_every7"),
                                   ("arx", "mechanism_vs_restart", "online_l1")):
        rows.append({"cohort": "replace_bg/core", "model": model, "finding": finding, "policy": policy})
    return pd.DataFrame(rows)


def test_refitted_arm_is_the_selected_mechanism_or_the_restart():
    r = replication()
    assert st.refit_policy(r, "lightgbm_ft") == "restart_every3"
    assert st.refit_policy(r, "lstm_ft") == "warm_every3"
    assert st.refit_policy(r, "arx") == "online_l1"
    assert st.refit_policy(r, "tide_ft") is None


def test_twins_stand_in_where_the_full_model_did_not_run():
    metrics = pd.DataFrame({"cohort": ["a", "a", "b", "b"],
                            "model": ["persistence", "lstm_ft", "persistence", "lstm_ft_noexog"]})
    assert st.cohort_models(metrics, "a", ["persistence", "lstm_ft"]) == ["persistence", "lstm_ft"]
    assert st.cohort_models(metrics, "b", ["persistence", "lstm_ft"]) == ["persistence", "lstm_ft_noexog"]
    assert st.model_label("lstm_ft_noexog", {"lstm_ft": "LSTM"}) == "LSTM, glucose only"


def test_regularity_interval_is_turned_with_the_coefficient():
    associations = pd.DataFrame([
        {"outcome": "ageing", "model": "lstm_ft", "horizon_min": 0, "index": "meal_time_jitter_min",
         "regular_when": "lower", "patients": 200, "coef": -0.05, "low": -0.10, "high": 0.02, "coef_regular": 0.05,
         "p_holm": 0.4},
        {"outcome": "ageing", "model": "lstm_ft", "horizon_min": 0, "index": "glucose_profile_stability",
         "regular_when": "higher", "patients": 300, "coef": 0.03, "low": -0.01, "high": 0.07, "coef_regular": 0.03,
         "p_holm": 0.9},
        {"outcome": "ageing", "model": "lstm_ft", "horizon_min": 30, "index": "meal_time_jitter_min",
         "regular_when": "lower", "patients": 200, "coef": 9, "low": 8, "high": 10, "coef_regular": -9,
         "p_holm": 0.0}])
    table = st.regularity_table(associations, ["lstm_ft"], ["ageing"],
                                ["meal_time_jitter_min", "glucose_profile_stability"])
    assert len(table) == 2                                  # only the mean over horizons
    meal = table[table["index"] == "meal_time_jitter_min"].iloc[0]
    assert (meal["coef"], meal["low"], meal["high"]) == (0.05, -0.02, 0.10)
    cells = st.regularity_cells(table, ["ageing"], {"meal_time_jitter_min": "Meal timing"}, {"lstm_ft": "LSTM"})
    assert cells.loc[0, "ageing"] == r"+0.05 (\textminus{}0.02, +0.10)"
    assert cells.loc[0, "index"] == "Meal timing" and cells.loc[1, "index"] == "glucose\\_profile\\_stability"


def test_error_and_lows_tables_read_the_frozen_and_refitted_arms():
    rows = []
    for model, policy, rmse in (("persistence", "reference", 25.0), ("lstm_ft", "never", 20.0),
                                ("lstm_ft", "warm_every3", 19.0)):
        for h in (30, 60, 120):
            rows.append({"cohort": "replace_bg/core", "model": model, "policy": policy, "horizon_min": h,
                         "rmse": rmse * h / 30, "clarke_AB_pct": 98.0, "clarke_D_pct": 1.5,
                         "low70_sensitivity_pct": 40.0, "low70_precision_pct": 60.0, "low70_share_pct": 3.0})
    metrics = pd.DataFrame(rows)
    table = st.error_table(metrics, replication(), ["replace_bg/core"], ["persistence", "lstm_ft"])
    lstm = table[table["model"] == "lstm_ft"].iloc[0]
    assert lstm["rmse_frozen_60"] == 40.0 and lstm["rmse_refit_120"] == 76.0
    assert np.isnan(table[table["model"] == "persistence"].iloc[0]["rmse_refit_30"])
    cells = st.error_cells(table, {"replace_bg/core": "REPLACE-BG"}, {"lstm_ft": "LSTM"})
    assert cells["cohort"].tolist() == ["REPLACE-BG", ""] and cells.loc[0, "rmse_refit"] == "--"
    lows = st.lows_table(metrics, replication(), ["replace_bg/core"], ["persistence", "lstm_ft"])
    assert lows["state"].tolist() == ["reference", "frozen", "refitted"]
    assert st.lows_cells(lows, {"replace_bg/core": "REPLACE-BG"}, {}).loc[0, "cohort"] == r"REPLACE-BG (3.0\%)"


def test_sex_counts_follow_the_order_and_add_a_total():
    counts = pd.DataFrame({"cohort": ["loop/closed_insulin", "replace_bg/core"], "women": [69, 51], "men": [42, 54],
                           "other": [0, 0], "not recorded": [9, 0]})
    table = st.sex_counts_table(counts, {}, ["replace_bg/core", "loop/closed_insulin"])
    assert table["cohort"].tolist() == ["replace_bg/core", "loop/closed_insulin", "total"]
    assert table.iloc[-1]["women"] == 120 and table.iloc[-1]["not recorded"] == 9
