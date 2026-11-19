"""Do the outcomes of the confirmation differ between women and men? (script 62; SAGER, D109)

A brief check for the supplement, asked by the SAGER guidelines: whether the
results were analysed by sex. It reuses the per-patient outcomes of the
regularity analysis (script 61): the skill of the frozen model, how fast it
ages, the gain of refitting at the optimal interval and what that interval loses
against refitting every day, for every model, its glucose-only twin and their
difference (what insulin and carbohydrates add).

For every outcome, model and horizon: least squares of the outcome on being a
woman (1) or a man (0), adjusted for the glycaemic control at startup (time in
range, CV and mean glucose, standardised) and the cohort, with HC3 errors, as
in the regularity analysis. The coefficient is the difference women minus men
in skill points. Patients of another gender or without the datum are left out
(too few for a group of their own). Holm within each outcome and horizon,
across models. The study was not designed or powered for this comparison; the
check reports differences and their intervals, not a test of a hypothesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regularity_analysis import OUTCOMES, holm, zscore

SEXES = ("female", "male")


def with_sex(outcomes: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """The outcomes with a column ``female`` (1 woman, 0 man), only for patients recorded as woman or man.
    ``patients`` has ``dataset``, ``patient_id`` and ``sex`` (script 06); the cohort key starts with the dataset."""
    sex = patients[patients["sex"].isin(SEXES)][["dataset", "patient_id", "sex"]]
    data = outcomes.assign(dataset=outcomes["cohort"].str.split("/").str[0])
    data = data.merge(sex, on=["dataset", "patient_id"], how="inner")
    return data.assign(female=(data["sex"] == "female").astype(float)).drop(columns=["sex"])


def difference(frame: pd.DataFrame, outcome: str, adjust: list[str], min_per_sex: int) -> dict:
    """Women minus men in ``outcome`` (skill points), adjusted for the standardised ``adjust`` columns and the
    cohort (HC3): coefficient, 95 % interval and p, with the patients and the raw means of each group."""
    import statsmodels.api as sm

    columns = [outcome, "female", "cohort"] + [a for a in adjust if a in frame]
    f = frame[columns].dropna()
    women, men = f[f["female"] == 1], f[f["female"] == 0]
    out = {"women": int(len(women)), "men": int(len(men)), "cohorts": int(f["cohort"].nunique()) if len(f) else 0,
           "mean_women": 100 * float(women[outcome].mean()) if len(women) else np.nan,
           "mean_men": 100 * float(men[outcome].mean()) if len(men) else np.nan}
    if len(women) < min_per_sex or len(men) < min_per_sex or f[outcome].nunique() < 2:
        return out
    x = pd.DataFrame({"female": f["female"]})
    for a in columns[3:]:
        x[a] = zscore(f[a])
    x = x.loc[:, x.notna().all()]
    if f["cohort"].nunique() > 1:
        x = x.join(pd.get_dummies(f["cohort"], prefix="cohort", drop_first=True, dtype=float))
    x = x.loc[:, x.std() > 0]
    x = sm.add_constant(x, has_constant="add")
    fit = sm.OLS(f[outcome].to_numpy(float), x.to_numpy(float)).fit(cov_type="HC3")
    k = list(x.columns).index("female")
    low, high = fit.conf_int()[k]
    return {**out, "difference": 100 * float(fit.params[k]), "low": 100 * float(low), "high": 100 * float(high),
            "p": float(fit.pvalues[k])}


def sex_differences(outcomes: pd.DataFrame, patients: pd.DataFrame, adjusters: pd.DataFrame, adjust: list[str],
                    min_per_sex: int) -> pd.DataFrame:
    """Every outcome, model and horizon of ``outcomes`` (script 61): see ``difference``; Holm within each outcome
    and horizon across models. ``adjusters`` has ``cohort``, ``patient_id`` and the ``adjust`` columns."""
    data = with_sex(outcomes, patients).merge(adjusters, on=["cohort", "patient_id"], how="left")
    rows = []
    for (model, horizon), g in data.groupby(["model", "horizon_min"]):
        for outcome in OUTCOMES:
            if outcome not in g or g[outcome].notna().sum() == 0:
                continue
            rows.append({"outcome": outcome, "model": model, "horizon_min": int(horizon),
                         **difference(g, outcome, adjust, min_per_sex)})
    table = pd.DataFrame(rows)
    if table.empty or "p" not in table:
        return table
    table["p_holm"] = np.nan
    for _, g in table.groupby(["outcome", "horizon_min"]):
        table.loc[g.index, "p_holm"] = holm(g["p"])
    return table


def sex_groups(patients: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and patient, the group of the R6 metrics: woman, man, other or not recorded."""
    group = patients["sex"].astype("string").map({"female": "women", "male": "men", "other": "other"})
    return pd.DataFrame({"dataset": patients["dataset"], "patient_id": patients["patient_id"],
                         "sex_group": group.fillna("not recorded")})
