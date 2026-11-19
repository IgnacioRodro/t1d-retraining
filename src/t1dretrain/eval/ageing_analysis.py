"""Stage 1 analysis (script 21; HANDOFF 7.3, D78): how fast each model ages, its useful life and screening 1.

The unit is the daily skill of 7.3: on one patient, anchor and test day,
``1 - RMSE(model) / RMSE(persistence)`` over the same origins (persistence has
no parameters, so its error measures how hard the day is). Days with fewer
than ``min_origins`` origins are left out, and the daily skill is clipped to
``clip`` (a day where persistence is almost exact can give a huge negative
ratio); the number of clipped days is reported.

- **Ageing** (frozen version only): a linear mixed model per model and
  horizon, ``skill ~ weeks since the fit``, with a random intercept and slope
  per patient and a random intercept per anchor within the patient
  (statsmodels MixedLM, REML). If it does not converge, simpler random parts
  are tried in turn (the one used is reported; a variance estimated at zero is
  a valid fit and only noted). The fixed slope is the skill lost per week
  without refitting. As a check, and as the estimate when no mixed model
  converges, the within-anchor least-squares slope (an intercept per patient
  and anchor) with standard errors clustered by patient.
- **Useful life**: the days until the fitted loss reaches the DMD of the model
  (D75), ``7 * DMD / -slope``, with the interval from the slope's interval. A
  model *ages within the follow-up* when the slope's interval is below zero
  and its useful life is at most the follow-up (84 days).
- **Version effects**: ``skill ~ version`` with a random intercept per patient
  and per anchor, on the days where the three versions were tested; recency
  (rolling - frozen), amount of data (growing - rolling) and both (growing -
  frozen) with their intervals, and the tie rule of D61 with the model's DMD.
  If no mixed model converges, the same contrasts from least squares with an
  intercept per patient and anchor and errors clustered by patient.
- **Screening 1** (7.3): per mechanism, the ``per_mechanism`` models with the
  best fresh skill (better of rolling and growing, patient means, mean over
  horizons) pass, plus any model that ages clearly more slowly than the best of
  its mechanism (difference of slopes above zero with its interval, at some
  horizon). A model that does not beat persistence at any horizon is dropped
  (persistence already beats linear extrapolation in stage 0, so this is the
  stricter form of the rule of 7.3).
- **Against a model that cannot age** (script 43, D99): the slope of a
  forecaster's daily skill minus that of a zero-shot model with a fixed context
  on the same days (``against_reference``).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from .stats import bootstrap_mean

KEYS = ["period", "patient_id", "anchor_day", "horizon_min", "day"]
EFFECTS = {"recency (rolling - frozen)": ("rolling", "frozen"), "data (growing - rolling)": ("growing", "rolling"),
           "both (growing - frozen)": ("growing", "frozen")}
SPECS = ("slope + anchor", "intercept + anchor", "intercept")


def daily_skill(days: pd.DataFrame, min_origins: int = 12, clip: tuple[float, float] = (-1.0, 1.0)) -> pd.DataFrame:
    """Daily skill of every model and version against persistence on the same patient, anchor, horizon and day."""
    days = days[days["n"] >= min_origins]
    reference = days[days["model"] == "persistence"].set_index(KEYS)
    rmse_ref = np.sqrt(reference["sse"] / reference["n"]).rename("rmse_persistence")
    d = days[days["model"] != "persistence"].join(rmse_ref, on=KEYS, how="inner")
    d = d[d["rmse_persistence"] > 0].copy()
    raw = 1 - np.sqrt(d["sse"] / d["n"]) / d["rmse_persistence"]
    d["clipped"] = (raw < clip[0]) | (raw > clip[1])
    d["skill"] = raw.clip(*clip)
    d["weeks"] = d["age_days"] / 7.0
    columns = KEYS + ["model", "version", "fit_day", "age_days", "days_since_anchor", "weeks", "n", "skill", "clipped"]
    return d[[c for c in columns if c in d.columns]].reset_index(drop=True)


def _mixed(formula: str, data: pd.DataFrame, spec: str):
    import statsmodels.formula.api as smf

    vc = {"anchor": "0 + C(anchor_day)"} if "anchor" in spec else None
    re = "~weeks" if spec.startswith("slope") else "~1"
    model = smf.mixedlm(formula, data, groups=data["patient_id"], re_formula=re, vc_formula=vc)
    return model.fit(reml=True, method=["lbfgs", "powell"])


def fit_mixed(formula: str, data: pd.DataFrame, specs=SPECS) -> tuple[object | None, str, str]:
    """The first random-effects structure of ``specs`` that converges: (result, spec, warnings)."""
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    notes = []
    for spec in specs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = _mixed(formula, data, spec)
            except Exception as error:  # noqa: BLE001
                notes.append(f"{spec}: {type(error).__name__}: {str(error)[:120]}")
                continue
        # a variance estimated at zero (boundary, singular covariance) is a valid fit: that random part is not needed
        messages = sorted({str(w.message).split(".")[0][:70] for w in caught
                           if issubclass(w.category, ConvergenceWarning)})
        if result.converged and np.all(np.isfinite(result.bse)):
            if messages:
                notes.append(f"{spec}: " + "; ".join(messages))
            return result, spec, "; ".join(notes)
        notes.append(f"{spec}: not converged")
    return None, "", "; ".join(notes)


def within_fit(formula: str, f: pd.DataFrame):
    """Least squares with an intercept per patient and anchor (one factor, so the design has full rank) and
    standard errors clustered by patient."""
    import statsmodels.formula.api as smf

    f = f.assign(unit=f["patient_id"].astype(str) + "|" + f["anchor_day"].astype(str))
    groups = pd.factorize(f["patient_id"])[0]
    return smf.ols(f"{formula} + C(unit)", f).fit(cov_type="cluster", cov_kwds={"groups": groups})


def within_slope(f: pd.DataFrame) -> dict:
    """Least-squares slope of the daily skill on the weeks within each patient and anchor (clustered errors)."""
    result = within_fit("skill ~ weeks", f)
    low, high = result.conf_int().loc["weeks"]
    return {"slope_per_week": float(result.params["weeks"]), "slope_se": float(result.bse["weeks"]),
            "slope_low": float(low), "slope_high": float(high), "slope_p": float(result.pvalues["weeks"]),
            "intercept": float(f["skill"].mean() - result.params["weeks"] * f["weeks"].mean())}


def ageing_fit(frame: pd.DataFrame) -> dict:
    """Mixed model of the frozen daily skill over the weeks since the fit, for one model and horizon (and the
    within-anchor slope as a check, or as the estimate when no mixed model converges)."""
    started = time.time()
    f = frame[frame["version"] == "frozen"]
    out = {"days": int(len(f)), "patients": int(f["patient_id"].nunique()),
           "anchors": int(f[["patient_id", "anchor_day"]].drop_duplicates().shape[0]),
           "clipped_days": int(f["clipped"].sum())}
    if len(f) < 50 or out["patients"] < 3:
        return {**out, "spec": "", "note": "too few days", "seconds": round(time.time() - started, 1)}
    within = within_slope(f)
    check = {"within_slope": within["slope_per_week"], "within_se": within["slope_se"]}
    result, spec, note = fit_mixed("skill ~ weeks", f)
    if result is None:
        return {**out, **within, **check, "spec": "within (patient-clustered)", "note": note,
                "seconds": round(time.time() - started, 1)}
    low, high = result.conf_int().loc["weeks"]
    return {**out, "intercept": float(result.params["Intercept"]), "slope_per_week": float(result.params["weeks"]),
            "slope_se": float(result.bse["weeks"]), "slope_low": float(low), "slope_high": float(high),
            "slope_p": float(result.pvalues["weeks"]), **check, "spec": spec, "note": note,
            "seconds": round(time.time() - started, 1)}


def version_fit(frame: pd.DataFrame) -> list[dict]:
    """Mixed model of the daily skill by version (days where every version was tested), for one model and
    horizon: the recency, data and combined effects with their intervals."""
    versions = sorted(set(frame["version"]))
    if not {"frozen", "rolling", "growing"} <= set(versions):
        return []
    wide = frame.pivot_table(index=["patient_id", "anchor_day", "day"], columns="version", values="skill")
    complete = wide.dropna(subset=["frozen", "rolling", "growing"]).index
    f = frame.set_index(["patient_id", "anchor_day", "day"]).loc[complete].reset_index()
    f = f[f["version"].isin(["frozen", "rolling", "growing"])].copy()
    formula = "skill ~ C(version, Treatment(reference='frozen'))"
    result, spec, note = fit_mixed(formula, f, specs=("intercept + anchor", "intercept"))
    if result is None:  # the same contrasts within each patient and anchor, errors clustered by patient
        result, spec = within_fit(formula, f), "within (patient-clustered)"
    name = {v: f"C(version, Treatment(reference='frozen'))[T.{v}]" for v in ("rolling", "growing")}
    params, cov = result.params, result.cov_params()
    rows = []
    for effect, (a, b) in EFFECTS.items():
        weights = pd.Series(0.0, index=params.index)
        if a != "frozen":
            weights[name[a]] += 1.0
        if b != "frozen":
            weights[name[b]] -= 1.0
        estimate = float(weights @ params)
        se = float(np.sqrt(weights @ cov.loc[params.index, params.index] @ weights))
        rows.append({"effect": effect, "difference": estimate, "se": se, "low": estimate - 1.96 * se,
                     "high": estimate + 1.96 * se, "days": int(len(complete)),
                     "patients": int(f["patient_id"].nunique()), "spec": spec, "note": note})
    return rows


def useful_life(slope: float, low: float, high: float, dmd: float, follow_up_days: int) -> dict:
    """Days until the fitted loss of skill reaches the DMD (``inf`` when the model does not lose skill)."""
    def days(s: float) -> float:
        return 7.0 * dmd / -s if np.isfinite(s) and s < 0 and np.isfinite(dmd) else np.inf

    life, shortest, longest = days(slope), days(low), days(high)
    ages = bool(np.isfinite(high) and high < 0 and life <= follow_up_days)
    return {"useful_life_days": life, "useful_life_low": shortest, "useful_life_high": longest,
            "ages_within_follow_up": ages}


def verdict(difference: float, low: float, high: float, dmd: float) -> str:
    """The tie rule of D61 with the DMD of D75."""
    if np.isfinite(dmd) and abs(difference) < dmd:
        return "tie"
    return "positive" if low > 0 else "negative" if high < 0 else "uncertain"


def fresh_skill(days: pd.DataFrame) -> pd.DataFrame:
    """Per model, version and horizon: mean over patients of the patient skill pooled over every anchor and test
    day (the same measure as the patient means of scripts 12 and 20)."""
    keys = ["period", "model", "version", "horizon_min", "patient_id"]
    pooled = days.groupby(keys, as_index=False)[["n", "sse"]].sum()
    pooled["rmse"] = np.sqrt(pooled["sse"] / pooled["n"])
    reference = pooled[pooled["model"] == "persistence"].set_index(["period", "horizon_min", "patient_id"])["rmse"]
    pooled = pooled[pooled["model"] != "persistence"].copy()
    pooled["skill"] = 1 - pooled["rmse"] / pooled.set_index(["period", "horizon_min", "patient_id"]).index.map(
        reference)
    return pooled.groupby(["period", "model", "version", "horizon_min"], as_index=False)["skill"].mean()


def screening(fresh: pd.DataFrame, slopes: pd.DataFrame, mechanisms: dict, per_mechanism: int = 2) -> pd.DataFrame:
    """Screening 1 of 7.3 for one period (see the module notes)."""
    fresh = fresh[fresh["version"].isin(["rolling", "growing"])]
    by_version = fresh.groupby(["model", "version"])["skill"].mean().unstack("version")
    best_version = by_version.idxmax(axis=1)
    rows = []
    for model, version in best_version.items():
        f = fresh[(fresh["model"] == model) & (fresh["version"] == version)]
        rows.append({"model": model, "mechanism": mechanisms.get(model, "unknown"), "best_fresh_version": version,
                     "fresh_skill_mean": float(f["skill"].mean()),
                     **{f"fresh_skill_{int(h)}": float(v) for h, v in f.set_index("horizon_min")["skill"].items()},
                     "beats_persistence": bool((f["skill"] > 0).any())})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["rank_in_mechanism"] = table.groupby("mechanism")["fresh_skill_mean"].rank(ascending=False, method="first")
    best = table.sort_values("fresh_skill_mean", ascending=False).groupby("mechanism")["model"].first()
    slower = {}
    s = slopes.set_index(["model", "horizon_min"]) if not slopes.empty and "slope_per_week" in slopes \
        else pd.DataFrame()
    for _, row in table.iterrows():
        leader = best[row["mechanism"]]
        found = []
        if leader != row["model"] and not s.empty:
            for h in sorted(slopes["horizon_min"].unique()):
                if (row["model"], h) not in s.index or (leader, h) not in s.index:
                    continue
                a, b = s.loc[(row["model"], h)], s.loc[(leader, h)]
                diff = a["slope_per_week"] - b["slope_per_week"]
                se = np.sqrt(a["slope_se"] ** 2 + b["slope_se"] ** 2)  # independent fits: a conservative interval
                if np.isfinite(diff) and np.isfinite(se) and diff - 1.96 * se > 0:
                    found.append(int(h))
        slower[row["model"]] = found
    table["ages_slower_than_best_at"] = table["model"].map(lambda m: ", ".join(f"{h} min" for h in slower[m]))
    reasons = []
    for _, row in table.iterrows():
        if not row["beats_persistence"]:
            reasons.append("dropped: does not beat persistence at any horizon")
        elif row["rank_in_mechanism"] <= per_mechanism:
            reasons.append(f"passes: rank {int(row['rank_in_mechanism'])} of its mechanism")
        elif slower[row["model"]]:
            reasons.append("passes: ages more slowly than the best of its mechanism")
        else:
            reasons.append("stays out")
    table["decision"] = reasons
    table["passes"] = table["decision"].str.startswith("passes")
    return table.sort_values(["mechanism", "rank_in_mechanism"]).reset_index(drop=True)


def weekly_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Observed daily skill by week since the fit (frozen) or since the anchor (fresh versions): mean over patients
    of the patient means, with a normal 95 % interval."""
    f = frame.copy()
    f["week"] = (f["days_since_anchor"] // 7).astype(int)
    per_patient = f.groupby(["period", "model", "version", "horizon_min", "week", "patient_id"])["skill"].mean()
    g = per_patient.groupby(level=["period", "model", "version", "horizon_min", "week"])
    table = pd.DataFrame({"patients": g.size(), "skill_mean": g.mean(), "skill_sd": g.std()}).reset_index()
    half = 1.96 * table["skill_sd"] / np.sqrt(table["patients"].clip(lower=1))
    table["skill_low"], table["skill_high"] = table["skill_mean"] - half, table["skill_mean"] + half
    return table.drop(columns="skill_sd")


def against_reference(daily: pd.DataFrame, reference: str, replicates: int, seed: int,
                      min_weeks: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ageing of every forecaster against a model that cannot age (``reference``: a zero-shot model with a fixed
    context, script 43, D99), on the same patients and days. What both share (days that are hard for every model)
    cancels in the difference, so its slope is the ageing with less noise than the slope of the skill alone.

    ``daily``: one row per forecaster (``label``, ``model``, ``policy``), patient, horizon and day, with ``week``
    (weeks since the anchor) and ``skill``. Per patient, the least-squares slope over the weeks since the anchor of the weekly mean daily skill, alone
    (``raw_``) and minus the reference's; mean over patients with at least ``min_weeks`` weeks, with a bootstrap
    interval. Returns the slopes (points of skill per week; ``change_12_weeks`` = 12 x the slope) and the mean
    weekly difference."""
    keys = ["patient_id", "horizon_min", "day"]
    ref = daily[daily["label"] == reference].set_index(keys)["skill"]
    if ref.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = daily[daily["label"] != reference].copy()
    d["reference"] = d.set_index(keys).index.map(ref)
    d = d.dropna(subset=["reference"])
    d["difference"] = d["skill"] - d["reference"]
    group = ["label", "model", "policy", "horizon_min"]
    weekly = d.groupby(group + ["patient_id", "week"], dropna=False)[["skill", "difference"]].mean().reset_index()
    rows = []
    for key, g in weekly.groupby(group, dropna=False):
        slopes = {"difference": [], "skill": []}
        for _, p in g.groupby("patient_id"):
            if p["week"].nunique() < min_weeks:
                continue
            for column in slopes:
                slopes[column].append(np.polyfit(p["week"].to_numpy(float), p[column].to_numpy(float), 1)[0])
        if len(slopes["difference"]) < 3:
            continue
        row = dict(zip(group, key)) | {"reference": reference, "patients": len(slopes["difference"])}
        for column, prefix in (("difference", ""), ("skill", "raw_")):
            mean, low, high = bootstrap_mean(100 * np.asarray(slopes[column]), replicates, seed)
            row |= {f"{prefix}slope_per_week": mean, f"{prefix}low": low, f"{prefix}high": high}
        row["change_12_weeks"] = 12 * row["slope_per_week"]
        rows.append(row)
    g = weekly.groupby(group + ["week"], dropna=False)["difference"]
    curve = pd.DataFrame({"patients": g.size(), "difference": 100 * g.mean(), "sd": 100 * g.std()}).reset_index()
    half = 1.96 * curve["sd"] / np.sqrt(curve["patients"].clip(lower=1))
    curve["low"], curve["high"] = curve["difference"] - half, curve["difference"] + half
    return pd.DataFrame(rows), curve.drop(columns="sd")
