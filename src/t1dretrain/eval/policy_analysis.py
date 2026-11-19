"""Stage 2 analysis (script 31; HANDOFF 7.4, D83, D84): how the gain of refitting grows, the optimal interval,
the later anchors and the policies for the confirmation.

Every comparison is paired on the same patient, anchor, horizon and origins.
The unit of the curves and mixed models is the daily gain of a policy over
never: the difference of their daily skills against persistence (D78), that
is (RMSE never - RMSE policy) / RMSE persistence on that day. The unit of the
tables is the patient (skill pooled over the test days, as in script 30), with
bootstrap intervals over patients and the tie rule of D61 with the model's DMD
(D75).

- **Growth of the gain**: daily gain ~ weeks since the anchor, mixed model per
  model, horizon and policy (random intercept and slope per patient; simpler
  random parts if it does not converge; least squares within the patient with
  errors clustered by patient if none does).
- **Optimal interval** (D84): for the growing window, every ``k`` days against
  every day. The optimal interval of a model and horizon is the longest ``k``
  that is better than never (beyond the DMD) and whose loss against every day
  is within the DMD (tie, or better); if not even every day is better than
  never, the optimal policy is never. The interval of the model is the
  shortest of its horizons (never only if it is never at every horizon).
- **Later anchors** (D83): the gain over never at each anchor on the patients
  with every anchor, and the change of the gain from the main anchor to each
  later one. The data hypothesis: the daily gain depends on how much the data
  have grown since the first fit, ``log2((history + days since the anchor) /
  history)``, and not on the anchor itself; a mixed model with that term, and
  one with the anchor added, check it.
- **Confirmation** (paso 3 de 7.4, D84): per model, never, the growing window
  at the optimal interval and every day, and ARX's update at every sample.
- **Insulin and carbohydrates** (D90): for the models with a glucose-only
  twin (D76), the gain of each policy over never with every input and with
  the glucose only, and their paired difference per patient (positive:
  refitting gains more when the model reads insulin and carbohydrates).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sim.stage0 import split_variant
from . import ageing_analysis as aa
from . import policy_report as pr
from .stats import bootstrap_mean

DAY_KEYS = ["period", "model", "patient_id", "anchor_day", "horizon_min", "day", "days_since_anchor"]


def interval_policy(k: int) -> str:
    return f"growing_every{int(k)}"


# ============================================================================ daily gain and its growth


def daily_gain(days: pd.DataFrame, policies: list[str], against: str = "never", min_origins: int = 12,
               clip: tuple[float, float] = (-1.0, 1.0)) -> pd.DataFrame:
    """Per policy, patient, anchor, horizon and day: daily skill of the policy minus that of ``against``."""
    wanted = set(policies) | {against}
    d = days[(days["model"] == "persistence") | days["policy"].isin(wanted)].rename(columns={"policy": "version"})
    s = aa.daily_skill(d, min_origins, clip)
    if s.empty:
        return pd.DataFrame(columns=DAY_KEYS + ["policy", "gain"])
    wide = s.pivot_table(index=DAY_KEYS, columns="version", values="skill")
    parts = []
    for p in policies:
        if p in wide.columns and against in wide.columns and p != against:
            g = (wide[p] - wide[against]).dropna().rename("gain").reset_index()
            parts.append(g.assign(policy=p))
    if not parts:
        return pd.DataFrame(columns=DAY_KEYS + ["policy", "gain"])
    gain = pd.concat(parts, ignore_index=True)
    gain["weeks"] = gain["days_since_anchor"] / 7.0
    return gain


def weekly_gain(gain: pd.DataFrame) -> pd.DataFrame:
    """Per model, policy, horizon, anchor and week since the anchor: mean over patients of the patient means of the
    daily gain, with a normal 95 % interval."""
    g = gain.assign(week=(gain["days_since_anchor"] // 7).astype(int))
    keys = ["period", "model", "policy", "horizon_min", "anchor_day", "week"]
    per_patient = g.groupby(keys + ["patient_id"])["gain"].mean()
    grouped = per_patient.groupby(level=keys)
    table = pd.DataFrame({"patients": grouped.size(), "gain_mean": grouped.mean(), "gain_sd": grouped.std()})
    table = table.reset_index()
    half = 1.96 * table["gain_sd"] / np.sqrt(table["patients"].clip(lower=1))
    table["gain_low"], table["gain_high"] = table["gain_mean"] - half, table["gain_mean"] + half
    return table.drop(columns="gain_sd")


def growth_fit(frame: pd.DataFrame) -> dict:
    """Mixed model ``gain ~ weeks`` for one model, horizon, policy and anchor: the gain at the anchor and how much
    it grows per week."""
    out = {"days": int(len(frame)), "patients": int(frame["patient_id"].nunique())}
    if len(frame) < 50 or out["patients"] < 3:
        return {**out, "spec": "", "note": "too few days"}
    result, spec, note = aa.fit_mixed("gain ~ weeks", frame, specs=("slope", "intercept"))
    if result is None:
        result, spec = aa.within_fit("gain ~ weeks", frame), "within (patient-clustered)"
        intercept = float(frame["gain"].mean() - result.params["weeks"] * frame["weeks"].mean())
    else:
        intercept = float(result.params["Intercept"])
    low, high = result.conf_int().loc["weeks"]
    return {**out, "gain_at_anchor": intercept, "growth_per_week": float(result.params["weeks"]),
            "growth_low": float(low), "growth_high": float(high), "growth_p": float(result.pvalues["weeks"]),
            "spec": spec, "note": note}


def growth_table(gain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, f in gain.groupby(["period", "model", "policy", "horizon_min", "anchor_day"]):
        rows.append({**dict(zip(["period", "model", "policy", "horizon_min", "anchor_day"], key)), **growth_fit(f)})
    return pd.DataFrame(rows)


# ============================================================================ optimal interval


def interval_table(skill: pd.DataFrame, intervals: list[int], dmd, replicates: int, seed: int) -> pd.DataFrame:
    """Every growing interval against every day (loss in skill points and mg/dL) and against never (gain), with
    the share of the daily gain it keeps."""
    names = {interval_policy(k): int(k) for k in intervals}
    daily = interval_policy(1)
    loss = pr.paired(skill, lambda p: daily if p in names and p != daily else None, dmd, replicates, seed)
    gain = pr.paired(skill, lambda p: "never" if p in names else None, dmd, replicates, seed)
    if gain.empty:
        return pd.DataFrame()
    keys = ["period", "model", "horizon_min", "policy"]
    gain = gain[keys + ["patients", "skill_difference", "low", "high", "rmse_saved_mgdl", "verdict"]].rename(
        columns={"skill_difference": "gain_over_never", "low": "gain_low", "high": "gain_high",
                 "rmse_saved_mgdl": "mgdl_saved_over_never", "verdict": "verdict_over_never"})
    table = gain.copy()
    if not loss.empty:
        loss = loss[keys + ["skill_difference", "low", "high", "rmse_saved_mgdl", "dmd", "verdict"]].rename(
            columns={"skill_difference": "difference_to_daily", "low": "daily_low", "high": "daily_high",
                     "rmse_saved_mgdl": "mgdl_to_daily", "verdict": "verdict_to_daily"})
        table = table.merge(loss, on=keys, how="left")
    table["interval_days"] = table["policy"].map(names)
    full = table[table["policy"] == daily].set_index(["period", "model", "horizon_min"])["gain_over_never"]
    table["share_of_daily_gain"] = [g / full.get((p, m, h), np.nan) if full.get((p, m, h), 0) else np.nan
                                    for p, m, h, g in zip(table["period"], table["model"], table["horizon_min"],
                                                          table["gain_over_never"])]
    table.loc[table["policy"] == daily, ["difference_to_daily", "verdict_to_daily"]] = [0.0, "reference"]
    return table.sort_values(keys[:3] + ["interval_days"]).reset_index(drop=True)


def optimal_intervals(table: pd.DataFrame) -> pd.DataFrame:
    """Per model and horizon, the longest interval better than never and within the DMD of every day (NaN: never
    is enough); per model, the shortest of them (``model_interval_days``)."""
    rows = []
    for (period, model, horizon), g in table.groupby(["period", "model", "horizon_min"]):
        ok = g[g["verdict_to_daily"].isin(["reference", "tie", "better"]) & (g["verdict_over_never"] == "better")]
        row = {"period": period, "model": model, "horizon_min": horizon,
               "intervals_tested": ", ".join(str(int(k)) for k in sorted(g["interval_days"]))}
        if ok.empty:  # not even every day beats never beyond the DMD
            daily = g[g["interval_days"] == 1]
            row.update(optimal_policy="never", optimal_interval_days=np.nan,
                       gain_over_never=float(daily["gain_over_never"].iloc[0]) if not daily.empty else np.nan)
        else:
            best = ok.sort_values("interval_days").iloc[-1]
            diff = best.get("difference_to_daily", 0.0)
            row.update(optimal_policy=best["policy"], optimal_interval_days=int(best["interval_days"]),
                       difference_to_daily=float(diff) if pd.notna(diff) else 0.0,
                       mgdl_to_daily=float(best.get("mgdl_to_daily", np.nan)),
                       gain_over_never=float(best["gain_over_never"]),
                       share_of_daily_gain=float(best["share_of_daily_gain"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["model_interval_days"] = out.groupby(["period", "model"])["optimal_interval_days"].transform("min")
    return out


# ============================================================================ later anchors


def common_patients(skill: pd.DataFrame, anchors: list[int], policies: list[str]) -> dict:
    """Per period and model, the patients with every anchor and every one of ``policies`` at each anchor."""
    out = {}
    need = pd.MultiIndex.from_product([[int(a) for a in anchors], list(policies)], names=["anchor_day", "policy"])
    s = skill[skill["policy"].isin(policies)].drop_duplicates(["period", "model", "patient_id", "anchor_day", "policy"])
    for (period, model), g in s.groupby(["period", "model"]):
        g = g[g.set_index(["anchor_day", "policy"]).index.isin(need)]
        count = g.groupby("patient_id").size()
        out[(period, model)] = sorted(count[count == len(need)].index)
    return out


def anchor_change(skill: pd.DataFrame, policy: str, base_anchor: int, dmd, replicates: int, seed: int) -> pd.DataFrame:
    """Per model, horizon and later anchor: the change of the patient gain over never from the base anchor (negative:
    refitting gains less when the model starts with more history), on patients with both anchors."""
    rows = []
    keys = ["period", "model", "horizon_min"]
    for (period, model, horizon), g in skill.groupby(keys):
        wide = g.pivot_table(index="patient_id", columns=["anchor_day", "policy"], values="skill")
        if (base_anchor, policy) not in wide.columns or (base_anchor, "never") not in wide.columns:
            continue
        base = wide[(base_anchor, policy)] - wide[(base_anchor, "never")]
        for anchor in sorted({a for a, _ in wide.columns if a != base_anchor}):
            if (anchor, policy) not in wide.columns or (anchor, "never") not in wide.columns:
                continue
            later = wide[(anchor, policy)] - wide[(anchor, "never")]
            d = (later - base).dropna().to_numpy()
            if len(d) < 3:
                continue
            mean, low, high = bootstrap_mean(d, replicates, seed)
            floor = dmd.model(period, horizon, model) if dmd is not None and len(dmd) else np.nan
            rows.append({"period": period, "model": model, "horizon_min": horizon, "policy": policy,
                         "base_anchor": base_anchor, "anchor_day": anchor, "patients": int(len(d)),
                         "gain_base": float(np.nanmean(base[later.dropna().index])),
                         "gain_anchor": float(np.nanmean(later.dropna())), "change": mean, "low": low, "high": high,
                         "dmd": floor, "verdict": aa.verdict(mean, low, high, floor)})
    return pd.DataFrame(rows)


def patient_fit(formula: str, f: pd.DataFrame):
    """Least squares with an intercept per patient and errors clustered by patient (the fallback of the data-ratio
    models: the anchor stays a fixed effect, so the intercept cannot be per patient and anchor)."""
    import statsmodels.formula.api as smf

    groups = pd.factorize(f["patient_id"])[0]
    return smf.ols(f"{formula} + C(patient_id)", f).fit(cov_type="cluster", cov_kwds={"groups": groups})


def data_ratio_fits(gain: pd.DataFrame, history: dict) -> pd.DataFrame:
    """The data hypothesis of D83, per model and horizon (daily gain of refitting every day over never, every
    anchor): ``gain ~ log2(data ratio)`` and ``gain ~ log2(data ratio) + anchor`` with a random intercept per patient
    (and per anchor within the patient when it converges)."""
    rows = []
    g = gain.copy()
    g["history_days"] = g["anchor_day"].map(lambda a: history.get(int(a), int(a)))
    g["log2_ratio"] = np.log2((g["history_days"] + g["days_since_anchor"]) / g["history_days"])
    for (period, model, horizon), f in g.groupby(["period", "model", "horizon_min"]):
        out = {"period": period, "model": model, "horizon_min": horizon, "days": int(len(f)),
               "patients": int(f["patient_id"].nunique()), "anchors": int(f["anchor_day"].nunique())}
        if len(f) < 100 or out["anchors"] < 2:
            rows.append({**out, "note": "too few days or anchors"})
            continue
        result, spec, note = aa.fit_mixed("gain ~ log2_ratio", f, specs=("intercept + anchor", "intercept"))
        if result is None:
            result, spec = patient_fit("gain ~ log2_ratio", f), "per patient (patient-clustered)"
        low, high = result.conf_int().loc["log2_ratio"]
        out.update(per_doubling=float(result.params["log2_ratio"]), per_doubling_low=float(low),
                   per_doubling_high=float(high), spec=spec, note=note)
        formula = "gain ~ log2_ratio + C(anchor_day)"
        both, spec2, note2 = aa.fit_mixed(formula, f, specs=("intercept",))
        if both is None:
            both, spec2 = patient_fit(formula, f), "per patient (patient-clustered)"
        ci = both.conf_int()
        for name in both.params.index:
            if name.startswith("C(anchor_day)"):
                anchor = name.split("[T.")[-1].rstrip("]")
                out[f"anchor_{anchor}"] = float(both.params[name])
                out[f"anchor_{anchor}_low"], out[f"anchor_{anchor}_high"] = (float(v) for v in ci.loc[name])
        out["per_doubling_with_anchor"] = float(both.params["log2_ratio"])
        out["spec_with_anchor"], out["note_with_anchor"] = spec2, note2
        rows.append(out)
    return pd.DataFrame(rows)


def ratio_curve(gain: pd.DataFrame, history: dict, bins: int = 8) -> pd.DataFrame:
    """Binned daily gain (mean over horizons) against log2(data ratio), per model and anchor: patient means per bin,
    then their mean."""
    g = gain.copy()
    g["history_days"] = g["anchor_day"].map(lambda a: history.get(int(a), int(a)))
    g["log2_ratio"] = np.log2((g["history_days"] + g["days_since_anchor"]) / g["history_days"])
    day = g.groupby(["period", "model", "anchor_day", "patient_id", "day", "log2_ratio"])["gain"].mean().reset_index()
    edges = np.linspace(0, max(day["log2_ratio"].max(), 1e-6), bins + 1)
    day["bin"] = np.clip(np.digitize(day["log2_ratio"], edges) - 1, 0, bins - 1)
    per_patient = day.groupby(["period", "model", "anchor_day", "bin", "patient_id"]).agg(
        gain=("gain", "mean"), log2_ratio=("log2_ratio", "mean")).reset_index()
    return per_patient.groupby(["period", "model", "anchor_day", "bin"]).agg(
        gain_mean=("gain", "mean"), gain_sd=("gain", "std"), patients=("patient_id", "nunique"),
        log2_ratio=("log2_ratio", "mean")).reset_index()


# ============================================================================ cost, front and confirmation


def refits_from_log(fit_log: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Per period, model and policy: refits per week after the first fit (mean over patient-anchors)."""
    if fit_log.empty:
        return pd.DataFrame(columns=["period", "model", "policy", "refits_per_week"])
    counts = fit_log.groupby(["period", "model", "policy", "patient_id", "anchor_day"])["fit_day"].nunique()
    counts = counts.rename("fits").reset_index()
    return pr.refits_table(counts, test_days)


def policy_table(skill: pd.DataFrame, fit_log: pd.DataFrame, test_days: int, kinds: dict) -> pd.DataFrame:
    """Per model and policy (main anchor): mean skill per horizon and over horizons, refits per week, computing
    seconds per test day and the front of the fixed policies."""
    if skill.empty:
        return pd.DataFrame()
    table = pr.ranking(skill).drop(columns=["rank"])
    for extra in (pr.cost_table(fit_log, test_days), refits_from_log(fit_log, test_days)):
        table = table.merge(extra, on=["period", "model", "policy"], how="left")
    by_horizon = skill.groupby(["period", "model", "policy", "horizon_min"])["skill"].mean().unstack("horizon_min")
    by_horizon.columns = [f"skill_{int(h)}" for h in by_horizon.columns]
    table = table.merge(by_horizon.reset_index(), on=["period", "model", "policy"], how="left")
    table["kind"] = table["policy"].map(kinds)
    table["front"] = False
    for _, g in table[table["kind"].isin(["never", "fixed"])].groupby(["period", "model"]):
        table.loc[g.index, "front"] = pr.pareto(g.fillna({"seconds_per_day": 0.0}))
    return table.sort_values(["period", "model", "skill_mean"], ascending=[True, True, False]).reset_index(drop=True)


def confirmation_policies(optimal: pd.DataFrame, models: list[str], online: dict) -> dict:
    """D84: per model, never, the growing window at the model's optimal interval and every day (and online)."""
    out = {}
    for model in models:
        o = optimal[optimal["model"] == model]
        k = o["model_interval_days"].iloc[0] if not o.empty else 1.0
        k = int(k) if pd.notna(k) else None  # None: never is enough at every horizon
        policies = ["never"] + ([interval_policy(k)] if k else []) + [interval_policy(1)] + list(online.get(model, []))
        out[model] = {"policies": list(dict.fromkeys(policies)), "optimal_interval_days": k,
                      "by_horizon": {int(h): (int(v) if pd.notna(v) else None)
                                     for h, v in zip(o["horizon_min"], o["optimal_interval_days"])}}
    return out


def confirmation_intervals(derived, value, models: list[str]) -> tuple[dict, str]:
    """Each model's refit interval for stages 3 and 3b: its optimal interval of script 31
    (``<derived>/runs/31_stage2_policy_analysis/confirmation.yaml``; a glucose-only twin takes that of its full
    model; 1 day when never is enough), or ``value`` days for every model when ``value`` is a number."""
    from pathlib import Path

    import yaml

    if value != "confirmation":
        return {m: int(value) for m in models}, f"interval_days = {int(value)}"
    path = Path(derived) / "runs" / "31_stage2_policy_analysis" / "confirmation.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/31_stage2_policy_analysis.py first")
    document = yaml.safe_load(path.read_text()) or {}
    found = document.get("models", {})
    out = {}
    for m in models:
        k = (found.get(split_variant(m)[0]) or {}).get("optimal_interval_days")
        out[m] = int(k) if k else 1
    return out, f"optimal intervals of script 31, release {document.get('release')}"


# ============================================================================ insulin and carbohydrates (D90)


def exog_effect(skill: pd.DataFrame, dmd, replicates: int, seed: int) -> pd.DataFrame:
    """Per model with a glucose-only twin (D90), horizon and policy (or arm): gain over never with every input and
    with the glucose only (same patients), their paired difference with the tie rule (DMD: the larger of the two
    models'), and the skill of never of both."""
    rows = []
    twins = sorted({m for m in skill["model"].unique() if split_variant(m)[1]})
    for (period, horizon), g in skill.groupby(["period", "horizon_min"]):
        for twin in twins:
            base = split_variant(twin)[0]
            full = g[g["model"] == base].pivot_table(index="patient_id", columns="policy", values="skill")
            glucose = g[g["model"] == twin].pivot_table(index="patient_id", columns="policy", values="skill")
            if "never" not in full or "never" not in glucose:
                continue
            for policy in sorted((set(full.columns) & set(glucose.columns)) - {"never"}):
                gain_full = full[policy] - full["never"]
                gain_glucose = glucose[policy] - glucose["never"]
                d = (gain_full - gain_glucose).dropna()
                if len(d) < 3:
                    continue
                mean, low, high = bootstrap_mean(d.to_numpy(), replicates, seed)
                floor = np.nanmax([dmd.model(period, horizon, base), dmd.model(period, horizon, twin)]) \
                    if dmd is not None and len(dmd) else np.nan
                rows.append({"period": period, "model": base, "twin": twin, "horizon_min": horizon,
                             "policy": policy, "patients": int(len(d)),
                             "gain_all_inputs": float(gain_full[d.index].mean()),
                             "gain_glucose_only": float(gain_glucose[d.index].mean()), "difference": mean, "low": low,
                             "high": high, "dmd": floor, "verdict": aa.verdict(mean, low, high, floor),
                             "never_all_inputs": float(full.loc[d.index, "never"].mean()),
                             "never_glucose_only": float(glucose.loc[d.index, "never"].mean())})
    return pd.DataFrame(rows)

