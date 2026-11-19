"""Q4 analysis (script 61; HANDOFF 7.7, D94): does the regularity of a patient's habits go with how the model ages
and how much refitting gains?

The unit is the patient of the confirmation (script 50), on the first
``outcome_days`` test days of every cohort. Per patient, model and horizon:

- ``skill_never``: the skill of the frozen model (never), pooled over the days;
- ``ageing``: how fast the frozen model ages, the least-squares slope of its
  daily skill against the weeks since the anchor (skill per week; negative:
  it ages);
- ``gain_refit``: what refitting at the model's optimal interval adds to never
  (``restart_every<k>`` - never);
- ``loss_daily``: what the optimal interval loses against refitting every day
  (``restart_every<k>`` - ``restart_every1``; only where the cohort runs it).
  H3 (more regular patients tolerate longer intervals) predicts a smaller loss
  in the more regular patients;
- for every model with a glucose-only twin (D90), the same outcomes of the
  full model minus those of its twin (``<model>:inputs``): whether what insulin
  and carbohydrates add depends on how regular the patient is.

Association: least squares of the outcome on the index (in standard deviations
over the patients), the glycaemic control at startup (time in range, CV and
mean glucose, also standardised) and the cohort, with HC3 errors; the
coefficient is in skill points per standard deviation of the index, and per
standard deviation towards the more regular side when the index has one. The
primary indices (D94, fixed before the analysis) are corrected together with
Holm within each outcome, model and horizon; the others, among themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.regularity import ADJUSTERS, INDICES
from ..sim.stage0 import is_twin, split_variant
from . import ageing_analysis as aa
from . import policy_report as pr

OUTCOMES = {"skill_never": "skill of the frozen model (never)",
            "ageing": "skill lost per week by the frozen model (slope; negative: it ages)",
            "gain_refit": "gain of refitting at the optimal interval over never",
            "loss_daily": "optimal interval minus refitting every day (H3: closer to 0 when more regular)"}
EXPECTED = {"ageing": "+", "gain_refit": "-", "loss_daily": "+"}  # H3: sign of the effect of being more regular
INPUTS = ":inputs"  # suffix of the full-minus-twin contrasts
TERTILES = {0.0: "low", 1.0: "middle", 2.0: "high"}  # of the index's value (not of regularity)


def patient_outcomes(days: pd.DataFrame, plan: dict, outcome_days: int, min_origins: int = 12,
                     clip: tuple = (-1.0, 1.0)) -> pd.DataFrame:
    """Per cohort, patient, model and horizon (and the mean over horizons, ``horizon_min`` 0): the outcomes."""
    d = days[days["days_since_anchor"] < int(outcome_days)]
    rows = []
    for cohort, c in d.groupby("cohort", sort=False):
        skill = pr.patient_skill(c)
        wide = skill.pivot_table(index=["patient_id", "model", "horizon_min"], columns="policy", values="skill")
        daily = aa.daily_skill(c[(c["policy"].isin(["never", "reference"]))].rename(columns={"policy": "version"}),
                               min_origins, clip)
        daily = daily[daily["version"] == "never"]
        slopes = daily.groupby(["patient_id", "model", "horizon_min"]).apply(_slope, include_groups=False)
        for (patient, model, horizon), r in wide.iterrows():
            k = (plan.get(split_variant(model)[0]) or {}).get("interval")
            arm = f"restart_every{int(k)}" if k else None
            row = {"cohort": cohort, "patient_id": patient, "model": model, "horizon_min": int(horizon),
                   "skill_never": r.get("never", np.nan), "ageing": slopes.get((patient, model, horizon), np.nan)}
            refit = r.get(arm, np.nan) if arm else np.nan
            row["gain_refit"] = refit - row["skill_never"]
            row["loss_daily"] = refit - r.get("restart_every1", np.nan) if arm != "restart_every1" else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    mean = out.groupby(["cohort", "patient_id", "model"], as_index=False)[list(OUTCOMES)].mean()
    return pd.concat([out, mean.assign(horizon_min=0)], ignore_index=True)


def _slope(frame: pd.DataFrame) -> float:
    if len(frame) < 14 or frame["weeks"].std() == 0:
        return np.nan
    return float(np.polyfit(frame["weeks"].to_numpy(float), frame["skill"].to_numpy(float), 1)[0])


def with_contrasts(outcomes: pd.DataFrame) -> pd.DataFrame:
    """The outcomes plus, for every model with its twin, full minus twin (``<model>:inputs``)."""
    if outcomes.empty:
        return outcomes
    keys = ["cohort", "patient_id", "horizon_min"]
    parts = [outcomes]
    for twin in sorted(m for m in outcomes["model"].unique() if is_twin(m)):
        full = split_variant(twin)[0]
        a = outcomes[outcomes["model"] == full].set_index(keys)[list(OUTCOMES)]
        b = outcomes[outcomes["model"] == twin].set_index(keys)[list(OUTCOMES)]
        both = a.index.intersection(b.index)
        if len(both):
            parts.append((a.loc[both] - b.loc[both]).reset_index().assign(model=full + INPUTS))
    return pd.concat(parts, ignore_index=True)


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std()
    return (values - values.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else values * np.nan


def association(frame: pd.DataFrame, outcome: str, index: str, adjust: list[str], min_patients: int) -> dict:
    """Least squares of ``outcome`` on the standardised ``index``, the standardised ``adjust`` columns and the cohort
    (HC3 errors): coefficient in skill points per SD, its 95 % interval and p, the Spearman correlation."""
    import statsmodels.api as sm
    from scipy.stats import spearmanr

    columns = [outcome, index, "cohort"] + [a for a in adjust if a != index]
    f = frame[columns].dropna()
    out = {"patients": int(len(f)), "cohorts": int(f["cohort"].nunique()) if len(f) else 0}
    if len(f) < min_patients or f[index].std() == 0:
        return out
    x = pd.DataFrame({"index": zscore(f[index])})
    for a in columns[3:]:
        x[a] = zscore(f[a])
    x = x.loc[:, x.notna().all()]
    if f["cohort"].nunique() > 1:
        x = x.join(pd.get_dummies(f["cohort"], prefix="cohort", drop_first=True, dtype=float))
    x = sm.add_constant(x, has_constant="add")
    fit = sm.OLS(f[outcome].to_numpy(float), x.to_numpy(float)).fit(cov_type="HC3")
    k = list(x.columns).index("index")
    low, high = fit.conf_int()[k]
    rho, p_rho = spearmanr(f[index], f[outcome])
    return {**out, "coef": 100 * float(fit.params[k]), "low": 100 * float(low), "high": 100 * float(high),
            "p": float(fit.pvalues[k]), "spearman": float(rho), "spearman_p": float(p_rho),
            "outcome_mean": 100 * float(f[outcome].mean()), "outcome_sd": 100 * float(f[outcome].std())}


def holm(p: pd.Series) -> pd.Series:
    """Holm-adjusted p-values (NaN kept)."""
    valid = p.dropna()
    if valid.empty:
        return p
    order = valid.sort_values()
    m = len(order)
    adjusted = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adjusted, 1.0), index=order.index).reindex(p.index)


def associations(outcomes: pd.DataFrame, indices: pd.DataFrame, primary: list[str], min_patients: int,
                 models: list[str] | None = None) -> pd.DataFrame:
    """Every outcome, model, horizon and index (and every adjuster on its own): see ``association``; Holm within
    each outcome, model and horizon, for the primary indices and for the others separately."""
    data = outcomes.merge(indices, on=["cohort", "patient_id"], how="inner")
    adjust = list(ADJUSTERS)
    names = [i for i in INDICES if i in data] + adjust
    rows = []
    for (model, horizon), g in data.groupby(["model", "horizon_min"]):
        if models is not None and model not in models:
            continue
        for outcome in OUTCOMES:
            if g[outcome].notna().sum() < min_patients:
                continue
            for name in names:
                r = association(g, outcome, name, [] if name in adjust else adjust, min_patients)
                regular = INDICES.get(name, ("", "", "", ""))[2]
                sign = {"lower": -1.0, "higher": 1.0}.get(regular, np.nan)
                rows.append({"outcome": outcome, "model": model, "horizon_min": int(horizon), "index": name,
                             "group": INDICES[name][0] if name in INDICES else "adjuster",
                             "primary": name in primary, "regular_when": regular, **r,
                             "coef_regular": r.get("coef", np.nan) * sign if np.isfinite(sign) else np.nan})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    for _, g in table.groupby(["outcome", "model", "horizon_min", "primary"]):
        table.loc[g.index, "p_holm"] = holm(g["p"]) if "p" in g else np.nan
    table["expected"] = table["outcome"].map(EXPECTED).fillna("")
    table["agrees"] = [("" if not e or not np.isfinite(c) or not np.isfinite(p) or p >= 0.05 else
                        "yes" if (c > 0) == (e == "+") else "no")
                       for e, c, p in zip(table["expected"], table["coef_regular"], table["p_holm"])]
    return table


def tertiles(outcomes: pd.DataFrame, indices: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Mean outcome (points) by tertile of each index within its cohort (low, middle, high value of the index)."""
    data = outcomes.merge(indices, on=["cohort", "patient_id"], how="inner")
    rows = []
    for name in names:
        if name not in data:
            continue
        d = data.dropna(subset=[name]).copy()
        if d.empty:
            continue
        d["tertile"] = d.groupby(["cohort", "model", "horizon_min"])[name].transform(_tertile).map(TERTILES)
        d = d.dropna(subset=["tertile"])
        for (model, horizon, tertile), g in d.groupby(["model", "horizon_min", "tertile"], observed=True):
            for outcome in OUTCOMES:
                v = g[outcome].dropna()
                if len(v) < 3:
                    continue
                half = 1.96 * v.std() / np.sqrt(len(v))
                rows.append({"index": name, "outcome": outcome, "model": model, "horizon_min": int(horizon),
                             "tertile": tertile, "patients": int(len(v)), "mean": 100 * v.mean(),
                             "low": 100 * (v.mean() - half), "high": 100 * (v.mean() + half)})
    return pd.DataFrame(rows)


def tertile_of(indices: pd.DataFrame, name: str) -> pd.DataFrame:
    """Each patient's tertile of ``name`` within the cohort (for the metrics of R6)."""
    d = indices[["cohort", "patient_id", name]].dropna().copy()
    d["tertile"] = d.groupby("cohort")[name].transform(_tertile).map(TERTILES)
    return d.dropna(subset=["tertile"])[["cohort", "patient_id", "tertile"]]


def _tertile(values: pd.Series) -> pd.Series:
    """0, 1 or 2: the tertile of every value (by rank, so ties are split); NaN with fewer than 6 values."""
    if len(values) < 6:
        return pd.Series(np.nan, index=values.index)
    return pd.Series(pd.qcut(values.rank(method="first"), 3, labels=False), index=values.index).astype(float)
