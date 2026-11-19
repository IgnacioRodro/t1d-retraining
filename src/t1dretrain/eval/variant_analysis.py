"""A model variant against the standard model (scripts 71 and 73; D96, D98).

The variant (a loss that weighs the lows, D96; the wearables as extra inputs,
D98) runs on the same patients, anchors, arms and test days as the standard
model, so every comparison is paired per patient. The day metrics carry a
``variant`` column (``standard`` for the model as it is); each variant brings
its own persistence, the same forecasts, so that the skill of every variant is
taken against persistence on the same origins.

- ``skill_table``: per cohort, variant, model, arm, horizon and patient, the
  pooled RMSE and the skill against persistence;
- ``variant_summary``: the mean over patients with a bootstrap interval;
- ``against_standard``: every variant minus the standard model, paired per
  patient: skill (points), RMSE (mg/dL) and, given per-patient lows, their
  sensitivity and precision;
- ``refit_gain``: the gain of every refit over never within each variant, and
  the difference of that gain between the variant and the standard model (does
  the variant change what refitting brings?);
- ``pooled_metrics``: the pooled forecasts (rule R6: RMSE, MAE, MARD, bias,
  Clarke zones) and the forecast lows.
"""

from __future__ import annotations

import pandas as pd

from . import policy_report as pr
from .confirmation import low_sums, with_lows
from .stage0_report import SUMS, finish
from .stats import bootstrap_mean, paired_by

STANDARD = "standard"
KEYS = ["cohort", "variant", "model", "policy", "horizon_min"]


def skill_table(days: pd.DataFrame) -> pd.DataFrame:
    """Per cohort, variant, period, model, arm, horizon and patient: RMSE and skill against persistence."""
    return pr.patient_skill(days, by=("cohort", "variant"))


def variant_summary(skill: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for key, g in skill.groupby(KEYS, sort=False):
        mean, low, high = bootstrap_mean(g["skill"].to_numpy(), replicates, seed)
        rows.append({**dict(zip(KEYS, key)), "patients": int(g["patient_id"].nunique()), "skill": mean,
                     "skill_low": low, "skill_high": high, "rmse": float(g["rmse"].mean())})
    return pd.DataFrame(rows)


def against_standard(skill: pd.DataFrame, lows: pd.DataFrame | None, replicates: int, seed: int) -> pd.DataFrame:
    """Every variant minus the standard model, paired per patient (skill and RMSE; sensitivity and precision of
    the forecast lows per threshold when ``lows`` is given). RMSE: positive when the variant errs more."""
    variants = [v for v in skill["variant"].unique() if v != STANDARD]
    pairs = {v: STANDARD for v in variants}
    parts = [paired_by(skill, KEYS, "variant", pairs, ["skill", "rmse"], replicates, seed)]
    if lows is not None and not lows.empty:
        lows = lows[lows["model"] != "persistence"]
        t = paired_by(lows, KEYS + ["threshold"], "variant", pairs, ["sensitivity_pct", "precision_pct"],
                      replicates, seed)
        if not t.empty:
            t["measure"] = t["measure"].str.replace("_pct", "", regex=False) + t["threshold"].map(lambda x: f"_{x}")
            parts.append(t.drop(columns=["threshold"]))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def refit_gain(skill: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per cohort, model, refit arm and horizon: the gain over never of every variant (mean over patients with its
    interval) and, paired per patient, the variant's gain minus the standard model's."""
    keys = ["cohort", "variant", "model", "horizon_min", "patient_id"]
    wide = skill.pivot_table(index=keys, columns="policy", values="skill").reset_index()
    arms = [c for c in wide.columns if c not in keys and c not in ("never", "reference")]
    if "never" not in wide or not arms:
        return pd.DataFrame()
    long = wide.melt(id_vars=keys + ["never"], value_vars=arms, var_name="policy", value_name="refit")
    long["gain"] = long["refit"] - long["never"]
    long = long.dropna(subset=["gain"])
    rows = []
    for key, g in long.groupby(["cohort", "variant", "model", "policy", "horizon_min"], sort=False):
        mean, low, high = bootstrap_mean(g["gain"].to_numpy(), replicates, seed)
        rows.append({**dict(zip(["cohort", "variant", "model", "policy", "horizon_min"], key)),
                     "patients": int(len(g)), "gain": mean, "low": low, "high": high})
    table = pd.DataFrame(rows)
    pairs = {v: STANDARD for v in long["variant"].unique() if v != STANDARD}
    diff = paired_by(long, KEYS, "variant", pairs, ["gain"], replicates, seed)
    if not diff.empty:
        diff = diff.rename(columns={"difference": "gain_difference", "low": "difference_low",
                                    "high": "difference_high", "patients": "paired_patients"})
        table = table.merge(diff[["cohort", "variant", "model", "policy", "horizon_min", "paired_patients",
                                  "gain_difference", "difference_low", "difference_high"]],
                            on=["cohort", "variant", "model", "policy", "horizon_min"], how="left")
    return table


def pooled_metrics(days: pd.DataFrame) -> pd.DataFrame:
    """Pooled forecasts per cohort, variant, model, arm and horizon (R6) with the forecast lows."""
    sums = SUMS + low_sums(days)
    pooled = days.groupby(KEYS, as_index=False)[sums].sum()
    return with_lows(finish(pooled))
