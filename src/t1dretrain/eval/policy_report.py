"""Tables of stage 2 (script 30; HANDOFF 7.4, D80, D81): skill, cost and refits of every refit policy.

The skill of a policy on one patient is pooled over the patient's test days
(``1 - RMSE(model) / RMSE(persistence)`` on the same origins), as in scripts
12 and 20; means over patients come with bootstrap intervals over patients.
The cost of a policy is the computing time of the fits it uses (counted as
if it ran alone), per test day; for networks without the time spent waiting
for a GPU turn (``compute_seconds``, D73).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import bootstrap_mean

KEYS = ["period", "model", "policy", "horizon_min"]
FIT_KEYS = ["model", "patient_id", "period", "anchor_day", "seed", "fit_day", "train_start", "weighting"]  # one fit


def patient_skill(days: pd.DataFrame, by: tuple = ()) -> pd.DataFrame:
    """Per period, model, policy, horizon and patient (and the columns ``by``, e.g. the anchor): pooled RMSE and
    skill against persistence."""
    by = list(by)
    pooled = days.groupby(KEYS + by + ["patient_id"], as_index=False)[["n", "sse"]].sum()
    pooled["rmse"] = np.sqrt(pooled["sse"] / pooled["n"])
    where = ["period", "horizon_min"] + by + ["patient_id"]
    reference = pooled[pooled["model"] == "persistence"].set_index(where)["rmse"]
    pooled = pooled[pooled["model"] != "persistence"].copy()
    index = pooled.set_index(where).index
    pooled["rmse_persistence"] = index.map(reference)
    pooled["skill"] = 1 - pooled["rmse"] / pooled["rmse_persistence"]
    return pooled.drop(columns=["sse"])


def ranking(skill: pd.DataFrame) -> pd.DataFrame:
    """Per period, model and policy: mean skill over patients, then over horizons (the ranking of halving)."""
    per_horizon = skill.groupby(KEYS)["skill"].mean().reset_index()
    table = per_horizon.groupby(["period", "model", "policy"])["skill"].mean().rename("skill_mean").reset_index()
    patients = skill.groupby(["period", "model", "policy"])["patient_id"].nunique().rename("patients").reset_index()
    table = table.merge(patients, on=["period", "model", "policy"])
    table["rank"] = table.groupby(["period", "model"])["skill_mean"].rank(ascending=False, method="first")
    return table


def halving(rank: pd.DataFrame, keep: int, references: dict, outside=()) -> pd.DataFrame:
    """Round 1 of successive halving: the ``keep`` best policies of each model and its reference policies. The
    policies in ``outside`` (online) do not compete for a place: they go on only as references."""
    rank = rank.copy()
    compete = ~rank["policy"].isin(list(outside))
    rank["rank"] = rank[compete].groupby(["period", "model"])["skill_mean"].rank(ascending=False, method="first")
    rows = []
    for _, r in rank.iterrows():
        refs = references.get(r["model"], [])
        top = r["rank"] <= keep
        reason = "top third" if top else "reference" if r["policy"] in refs else "dropped"
        rows.append({**r.to_dict(), "kept": reason != "dropped", "reason": reason})
    return pd.DataFrame(rows).sort_values(["period", "model", "rank"]).reset_index(drop=True)


def cost_table(fit_log: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Per period, model and policy: fits, computing seconds per test day (mean over patient-anchors).

    A fit shared by several policies may have been timed more than once (in different runs, round 1 and round 2):
    every policy is charged the median of its times, so that policies using the same fits cost the same."""
    if fit_log.empty or "compute_seconds" not in fit_log:
        return pd.DataFrame(columns=["period", "model", "policy", "fits_per_anchor", "seconds_per_day"])
    log = fit_log.copy()
    log["compute_seconds"] = pd.to_numeric(log["compute_seconds"], errors="coerce").fillna(
        pd.to_numeric(log.get("fit_seconds"), errors="coerce")).fillna(0.0)
    keys = [c for c in FIT_KEYS if c in log]
    if len(keys) == len(FIT_KEYS):
        log["compute_seconds"] = log.groupby(keys, dropna=False)["compute_seconds"].transform("median")
    per_anchor = log.groupby(["period", "model", "policy", "patient_id", "anchor_day"]).agg(
        fits=("fit_day", "size"), seconds=("compute_seconds", "sum")).reset_index()
    per_anchor["seconds_per_day"] = per_anchor["seconds"] / max(1, int(test_days))
    return per_anchor.groupby(["period", "model", "policy"]).agg(
        fits_per_anchor=("fits", "mean"), seconds_per_day=("seconds_per_day", "mean")).reset_index()


def refits_table(refits: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Per period, model and policy: refits per week after the first fit (mean over patient-anchors)."""
    if refits.empty:
        return pd.DataFrame(columns=["period", "model", "policy", "refits_per_week"])
    r = refits.copy()
    r["refits_per_week"] = 7.0 * (r["fits"] - 1).clip(lower=0) / max(1, int(test_days))
    return r.groupby(["period", "model", "policy"])["refits_per_week"].mean().reset_index()


def summary_table(skill: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per period, model, policy and horizon: patients, mean skill and RMSE with bootstrap intervals."""
    rows = []
    for key, g in skill.groupby(KEYS):
        s = bootstrap_mean(g["skill"].to_numpy(), replicates, seed)
        r = bootstrap_mean(g["rmse"].to_numpy(), replicates, seed)
        rows.append({**dict(zip(KEYS, key)), "patients": int(len(g)), "skill_mean": s[0], "skill_low": s[1],
                     "skill_high": s[2], "patient_rmse_mean": r[0], "patient_rmse_low": r[1], "patient_rmse_high": r[2]})
    return pd.DataFrame(rows)


def paired(skill: pd.DataFrame, against: dict, dmd, replicates: int, seed: int, by: tuple = ()) -> pd.DataFrame:
    """Paired differences per patient between each policy and its comparison policy (``against``: policy ->
    policy, or a callable policy -> policy or None): skill (with the tie rule of D61 and the model's DMD, D75) and
    RMSE in mg/dL (positive: the comparison policy has the larger error). ``by``: more grouping columns."""
    rows = []
    by = list(by)
    for key, g in skill.groupby(["period", "model", "horizon_min"] + by):
        period, model, horizon = key[:3]
        extra = dict(zip(by, key[3:]))
        wide = g.pivot_table(index="patient_id", columns="policy", values="skill")
        rmse = g.pivot_table(index="patient_id", columns="policy", values="rmse")
        for policy in wide.columns:
            other = against(policy) if callable(against) else against.get(policy)
            if other is None or other not in wide.columns or other == policy:
                continue
            d = (wide[policy] - wide[other]).dropna().to_numpy()
            if len(d) < 3:
                continue
            mean, low, high = bootstrap_mean(d, replicates, seed)
            e = (rmse[other] - rmse[policy]).dropna().to_numpy()
            floor = dmd.model(period, horizon, model) if dmd is not None and len(dmd) else np.nan
            verdict = "tie" if np.isfinite(floor) and abs(mean) < floor else \
                "better" if low > 0 else "worse" if high < 0 else "uncertain"
            rows.append({"period": period, "model": model, "horizon_min": horizon, **extra, "policy": policy,
                         "against": other, "patients": int(len(d)), "skill_difference": mean, "low": low,
                         "high": high, "patients_better": int((d > 0).sum()),
                         "rmse_saved_mgdl": float(np.mean(e)) if len(e) else np.nan, "dmd": floor,
                         "verdict": verdict})
    return pd.DataFrame(rows)


def pareto(table: pd.DataFrame, x: str = "seconds_per_day", y: str = "skill_mean") -> pd.Series:
    """Whether each row is on the front: no other row has at least its skill at most its cost, one strictly."""
    on = []
    for _, r in table.iterrows():
        better = table[(table[y] >= r[y]) & (table[x] <= r[x]) & ((table[y] > r[y]) | (table[x] < r[x]))]
        on.append(better.empty)
    return pd.Series(on, index=table.index)
