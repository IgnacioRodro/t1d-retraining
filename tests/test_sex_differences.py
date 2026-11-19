"""Women against men (script 62, D109): only women and men enter, and the adjusted difference is recovered."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from t1dretrain.eval import sex_differences as sd


def test_adjusted_difference_women_minus_men():
    rng = np.random.default_rng(0)
    n = 120
    female = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    tir = rng.normal(60, 10, n)
    cohort = np.where(np.arange(n) % 2 == 0, "loop/closed_insulin", "replace_bg/core")
    gain = 0.02 + 0.005 * female + 0.0005 * (tir - 60) + rng.normal(0, 0.002, n)
    ids = [f"P{i}" for i in range(n)]
    outcomes = pd.DataFrame({"cohort": cohort, "patient_id": ids, "model": "lightgbm_ft", "horizon_min": 0,
                             "skill_never": 0.2, "ageing": np.nan, "gain_refit": gain, "loss_daily": np.nan})
    patients = pd.DataFrame({"dataset": [c.split("/")[0] for c in cohort], "patient_id": ids,
                             "sex": np.where(female == 1, "female", "male")})
    patients.loc[0, "sex"] = "other"  # left out of the comparison
    adjusters = pd.DataFrame({"cohort": cohort, "patient_id": ids, "tir_pct": tir})
    table = sd.sex_differences(outcomes, patients, adjusters, ["tir_pct"], min_per_sex=15)
    row = table[table["outcome"] == "gain_refit"].iloc[0]
    assert row["women"] == n // 2 - 1 and row["men"] == n // 2
    assert row["difference"] == pytest.approx(0.5, abs=0.15)  # points
    assert row["low"] < row["difference"] < row["high"]
    assert "difference" not in table[table["outcome"] == "skill_never"].dropna(axis=1).columns  # constant outcome


def test_sex_groups_for_the_r6_metrics():
    groups = sd.sex_groups(pd.DataFrame({"dataset": ["a"] * 3, "patient_id": ["1", "2", "3"],
                                         "sex": pd.Series(["female", "other", pd.NA], dtype="string")}))
    assert groups["sex_group"].tolist() == ["women", "other", "not recorded"]
