"""Validation of the method (script 93, D112): the tables behind the five checks of the MethodsX article.

1. **Common designs against the simulator** (``split_skill``, ``inflation``): the skill that a random split of the
   samples, a random split of the days or one chronological split report, against the skill of the simulator's
   frozen and refitted models on the same held-out origins (``sim.naive_splits``).
2. **The ageing estimator** (``simulate_daily``, ``ageing_case``, ``simulation_summary``, ``variance_components``,
   ``real_width_ratio``): the slope of the daily skill minus that of a model that cannot age (D99), on synthetic
   cohorts with a known loss whose noise is calibrated on the real day metrics, and the width of its intervals on
   the real data.
3. **The verdicts with the minimum detectable difference** (``seed_comparisons``, ``calibration_summary``,
   ``power_summary``): comparisons of a model with itself (another seed), with and without a known difference
   added, judged by the tie rule of D61/D75 and by the interval alone.
4. **The paired design** (``pairing_efficiency``, ``crossover_efficiency``): the width of the interval of a
   difference when every arm is scored on the same patients and origins, against designs that do not pair them.
5. **Reproducibility and cost** (``compare_outputs``, ``run_costs``): whether two runs give the same files, and
   what every stage cost.

Skill differences are in points of skill (fractions; the script reports percentage points).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ageing_analysis import against_reference
from .stage0_report import stable_hash
from .stats import bootstrap_mean

ZONE_COLUMNS = [f"zone_{z}" for z in "ABCDE"]
DESIGN_ARMS = ("design", "frozen", "refitted")


# ============================================================================ 1. designs against the simulator


def split_skill(days: pd.DataFrame) -> pd.DataFrame:
    """Per model, design, arm, horizon and patient: RMSE, Clarke zones A+B and skill against persistence on the same
    held-out origins (seeds pooled)."""
    sums = ["n", "sse"] + [c for c in ZONE_COLUMNS if c in days]
    keys = ["model", "split", "arm", "horizon_min", "patient_id"]
    pooled = days.groupby(keys, as_index=False)[sums].sum()
    pooled["rmse"] = np.sqrt(pooled["sse"] / pooled["n"])
    if {"zone_A", "zone_B"} <= set(pooled):
        pooled["clarke_AB_pct"] = 100 * (pooled["zone_A"] + pooled["zone_B"]) / pooled["n"]
    where = ["model", "split", "horizon_min", "patient_id"]
    reference = pooled[pooled["arm"] == "persistence"].set_index(where)["rmse"]
    pooled = pooled[pooled["arm"] != "persistence"].copy()
    pooled["rmse_persistence"] = pooled.set_index(where).index.map(reference)
    pooled["skill"] = 1 - pooled["rmse"] / pooled["rmse_persistence"]
    return pooled.drop(columns=["sse"] + [c for c in ZONE_COLUMNS if c in pooled])


def inflation(skill: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per model, design and horizon, on the patients with every arm: the mean skill of each arm (bootstrap interval
    over patients) and the difference between the design's estimate and the frozen and refitted models'."""
    rows = []
    for (model, split, horizon), g in skill.groupby(["model", "split", "horizon_min"]):
        wide = g.pivot_table(index="patient_id", columns="arm", values="skill").dropna()
        rmse = g.pivot_table(index="patient_id", columns="arm", values="rmse").reindex(wide.index)
        if len(wide) < 3 or "design" not in wide:
            continue
        row = {"model": model, "split": split, "horizon_min": int(horizon), "patients": int(len(wide))}
        for arm in DESIGN_ARMS:
            if arm in wide:
                mean, low, high = bootstrap_mean(wide[arm].to_numpy(), replicates, seed)
                row |= {f"skill_{arm}": mean, f"skill_{arm}_low": low, f"skill_{arm}_high": high,
                        f"rmse_{arm}": float(rmse[arm].mean())}
        for against in ("frozen", "refitted"):
            if against in wide:
                d = (wide["design"] - wide[against]).to_numpy()
                mean, low, high = bootstrap_mean(d, replicates, seed)
                row |= {f"minus_{against}": mean, f"minus_{against}_low": low, f"minus_{against}_high": high,
                        f"patients_higher_than_{against}": int((d > 0).sum())}
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ 2. the ageing estimator


def ar1(rng: np.random.Generator, shape: tuple[int, int], sd: float, phi: float) -> np.ndarray:
    """Stationary AR(1) noise along the last axis with marginal standard deviation ``sd``."""
    out = np.empty(shape)
    innovation = sd * np.sqrt(max(0.0, 1 - phi ** 2))
    out[:, 0] = rng.normal(0, sd, shape[0])
    for t in range(1, shape[1]):
        out[:, t] = phi * out[:, t - 1] + rng.normal(0, innovation, shape[0])
    return out


def simulate_daily(rng: np.random.Generator, patients: int, days: int, slope_pp: float, correlation: float,
                   day_sd: float, reference_sd: float, phi: float, slope_sd_pp: float, level: float = 0.15,
                   level_sd: float = 0.05) -> pd.DataFrame:
    """Daily skill of three forecasters of one synthetic cohort, in the layout of ``against_reference``:

    - ``model``: a frozen model whose skill changes by ``slope_pp`` points per week on average (each patient's slope
      drawn around it with ``slope_sd_pp``);
    - ``reference``: a model that cannot age (a zero-shot model with a fixed context);
    - ``control``: another model that cannot age, with the noise of the model (its slope against the reference
      should be 0).

    The daily skill of the model (and the control) varies around its level and trend with standard deviation
    ``day_sd``, that of the reference with ``reference_sd``; the part they share on the same patient and day (days
    that are hard for every model) gives them the correlation ``correlation``. Every part follows an AR(1) process
    from day to day with coefficient ``phi``."""
    shared_var = max(0.0, correlation) * day_sd * reference_sd
    shared = ar1(rng, (patients, days), np.sqrt(shared_var), phi)
    own = {"model": np.sqrt(max(0.0, day_sd ** 2 - shared_var)), "control": np.sqrt(max(0.0, day_sd ** 2 - shared_var)),
           "reference": np.sqrt(max(0.0, reference_sd ** 2 - shared_var))}
    day = np.arange(days)
    slopes = (slope_pp + rng.normal(0, slope_sd_pp, patients)) / 100 if slope_sd_pp > 0 else \
        np.full(patients, slope_pp / 100)
    series = {
        "model": rng.normal(level, level_sd, patients)[:, None] + slopes[:, None] * day[None, :] / 7,
        "reference": np.repeat(rng.normal(level, level_sd, patients)[:, None], days, axis=1),
        "control": np.repeat(rng.normal(level, level_sd, patients)[:, None], days, axis=1),
    }
    frames = []
    for label, base in series.items():
        skill = base + shared + ar1(rng, (patients, days), own[label], phi)
        frames.append(pd.DataFrame({"label": label, "model": label, "policy": "never",
                                    "patient_id": np.repeat([f"S{p:03d}" for p in range(patients)], days),
                                    "horizon_min": 0, "day": np.tile(day, patients),
                                    "week": np.tile(day // 7, patients), "skill": np.clip(skill, -1, 1).ravel()}))
    return pd.concat(frames, ignore_index=True)


def ageing_case(case: dict) -> list[dict]:
    """``case["repetitions"]`` synthetic cohorts of one setting, each through ``against_reference``: the slopes of
    the model and the control, against the reference and alone (points per week), with their intervals."""
    rng = np.random.default_rng([int(case["seed"]), stable_hash(json.dumps(case, sort_keys=True, default=str))])
    rows = []
    for rep in range(int(case["repetitions"])):
        daily = simulate_daily(rng, int(case["patients"]), int(case["days"]), float(case["slope_pp"]),
                               float(case["correlation"]), float(case["day_sd"]), float(case["reference_sd"]),
                               float(case["phi"]), float(case["slope_sd_pp"]))
        slopes, _ = against_reference(daily, "reference", int(case["replicates"]), int(case["seed"]) + rep,
                                      int(case.get("min_weeks", 6)))
        for r in slopes.itertuples():
            rows.append({"correlation": case["correlation"], "slope_pp": case["slope_pp"],
                         "true_slope": case["slope_pp"] if r.label == "model" else 0.0, "repetition": rep,
                         "label": r.label, "patients": r.patients, "slope": r.slope_per_week, "low": r.low,
                         "high": r.high, "raw_slope": r.raw_slope_per_week, "raw_low": r.raw_low,
                         "raw_high": r.raw_high})
    return rows


def simulation_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per setting and forecaster: bias, coverage and width of the interval of the slope against the reference and
    of the slope alone, the ratio of the widths and how often each interval excludes 0."""
    out = []
    for (correlation, slope, label), g in rows.groupby(["correlation", "slope_pp", "label"]):
        true = float(g["true_slope"].iloc[0])
        row = {"correlation": correlation, "slope_pp": slope, "label": label, "true_slope": true,
               "repetitions": int(len(g))}
        for prefix, name in (("", "adjusted"), ("raw_", "raw")):
            estimate, low, high = g[f"{prefix}slope"], g[f"{prefix}low"], g[f"{prefix}high"]
            row |= {f"{name}_mean": float(estimate.mean()), f"{name}_bias": float(estimate.mean() - true),
                    f"{name}_sd": float(estimate.std()),
                    f"{name}_coverage": float(((low <= true) & (true <= high)).mean()),
                    f"{name}_width": float((high - low).mean()),
                    f"{name}_excludes_zero": float(((high < 0) | (low > 0)).mean())}
        row["width_ratio"] = row["adjusted_width"] / row["raw_width"] if row["raw_width"] > 0 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def variance_components(daily: pd.DataFrame, model: str, reference: str, min_days: int = 42) -> pd.DataFrame:
    """Per horizon, from the real daily skill of a frozen model and of the reference on the same patients and days:
    the standard deviation of the daily skill of each around each patient's linear trend, the correlation of the
    two (the day-to-day variation they share), the lag-1 autocorrelation of the residuals, and the
    spread of the true slopes between patients (the spread of the patients' slopes against the reference minus what
    their sampling error explains, points per week)."""
    keys = ["patient_id", "horizon_min", "day"]
    m = daily[daily["label"] == model].set_index(keys)[["skill", "week"]]
    r = daily[daily["label"] == reference].set_index(keys)["skill"].rename("reference")
    both = m.join(r, how="inner").reset_index()
    rows = []
    for horizon, g in both.groupby("horizon_min"):
        residuals, slopes, sampling, levels = [], [], [], []
        for _, p in g.groupby("patient_id"):
            if len(p) < min_days:
                continue
            p = p.sort_values("day")
            t = (pd.to_datetime(p["day"]) - pd.to_datetime(p["day"]).min()).dt.days.to_numpy(float) \
                if not np.issubdtype(p["day"].dtype, np.number) else p["day"].to_numpy(float)
            fit = {}
            for column in ("skill", "reference"):
                y = p[column].to_numpy(float)
                coef = np.polyfit(t, y, 1)
                fit[column] = y - np.polyval(coef, t)
            residuals.append(pd.DataFrame({"model": fit["skill"], "reference": fit["reference"]}))
            levels.append(float(p["skill"].mean()))
            weekly = p.assign(difference=p["skill"] - p["reference"]).groupby("week")["difference"].mean()
            if len(weekly) >= 3:
                w = weekly.index.to_numpy(float)
                coef, residual, *_ = np.polyfit(w, weekly.to_numpy(float), 1, full=True)
                dof = max(1, len(w) - 2)
                s2 = float(residual[0]) / dof if len(residual) else 0.0
                slopes.append(coef[0])
                sampling.append(s2 / np.sum((w - w.mean()) ** 2))
        if len(residuals) < 3:
            continue
        res = pd.concat(residuals, ignore_index=True)
        lag = [np.corrcoef(x["model"].to_numpy()[:-1], x["model"].to_numpy()[1:])[0, 1] for x in residuals
               if len(x) > 3]
        spread = max(0.0, float(np.var(slopes, ddof=1) - np.mean(sampling))) if len(slopes) > 2 else np.nan
        rows.append({"horizon_min": int(horizon), "patients": len(residuals), "model": model, "reference": reference,
                     "level": float(np.mean(levels)), "day_sd": float(res["model"].std()),
                     "reference_day_sd": float(res["reference"].std()),
                     "correlation": float(np.clip(res["model"].corr(res["reference"]), 0, 1)),
                     "phi": float(np.nanmean(lag)), "slope_sd_pp": 100 * float(np.sqrt(spread))
                     if np.isfinite(spread) else np.nan})
    return pd.DataFrame(rows)


def real_width_ratio(ageing: pd.DataFrame) -> pd.DataFrame:
    """From the ageing table of script 43 (``ageing_vs_context.csv``): per forecaster and horizon, the width of the
    interval of the slope against the reference over that of the slope alone."""
    a = ageing.copy()
    a["width"] = a["high"] - a["low"]
    a["raw_width"] = a["raw_high"] - a["raw_low"]
    a["width_ratio"] = a["width"] / a["raw_width"]
    a["kind"] = np.where(a["model"].isin(["chronos2", "timesfm", "ttm"]), "foundation",
                         np.where(a["policy"] == "never", "frozen", "updated"))
    return a[["label", "model", "policy", "kind", "horizon_min", "patients", "slope_per_week", "low", "high",
              "raw_slope_per_week", "raw_low", "raw_high", "width", "raw_width", "width_ratio"]]


# ============================================================================ 3. verdicts with the DMD


def rule_verdict(mean: float, low: float, high: float, dmd: float) -> str:
    """The tie rule of D61/D75: tie below the DMD, else better or worse when the interval excludes 0."""
    if np.isfinite(dmd) and abs(mean) < dmd:
        return "tie"
    return "better" if low > 0 else "worse" if high < 0 else "uncertain"


def interval_verdict(low: float, high: float) -> str:
    """The interval alone (no DMD): better or worse whenever it excludes 0."""
    return "better" if low > 0 else "worse" if high < 0 else "uncertain"


def pairs_dmd(wide: pd.DataFrame, pairs: list[tuple], replicates: int, rng: np.random.Generator) -> float:
    """The DMD as script 12 computes it (``stage0_report.noise_floor``), from the given seed pairs only: the 95th
    percentile of the absolute bootstrap means of the seed differences."""
    absolute = []
    for a, b in pairs:
        d = (wide[a] - wide[b]).to_numpy()
        means = d[rng.integers(0, d.size, size=(replicates, d.size))].mean(axis=1)
        absolute.append(np.abs(means))
    return float(np.quantile(np.concatenate(absolute), 0.95))


def seed_comparisons(by_patient: pd.DataFrame, noise: pd.DataFrame, effects_dmd: list[float],
                     effects_pp: list[float], replicates: int, seed: int, period: str | None = None) -> pd.DataFrame:
    """Every learned model of stage 0 (``metrics_by_patient`` of script 12) compared with itself: for each pair of
    seeds, the per-patient skill of one seed against the other, with a known difference added (``effects_dmd``,
    multiples of the model's DMD; ``effects_pp``, percentage points). The DMD of each comparison leaves its own
    pair of seeds out (``dmd_held_out``, from the other pairs), so that the null comparison does not set its own
    threshold; ``dmd`` is the one script 12 reports. Each comparison is judged by the tie rule and by the interval
    alone."""
    data = by_patient if period is None else by_patient[by_patient["period"] == period]
    floors = noise.set_index(["period", "model", "horizon_min"])["dmd"] if not noise.empty else pd.Series(dtype=float)
    rows = []
    for (per, model, horizon), g in data.groupby(["period", "model", "horizon_min"]):
        wide = g.pivot_table(index="patient_id", columns="seed", values="skill").dropna()
        seeds = list(wide.columns)
        if len(seeds) < 3 or len(wide) < 3:
            continue
        pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
        rng = np.random.default_rng([int(seed), stable_hash(f"{per}|{model}|{int(horizon)}|null")])
        full = float(floors.get((per, model, horizon), np.nan))
        for a, b in pairs:
            held = pairs_dmd(wide, [p for p in pairs if p != (a, b)], replicates, rng)
            null = (wide[a] - wide[b]).to_numpy()
            effects = [("dmd", e, e * held) for e in effects_dmd] + [("pp", e, e / 100) for e in effects_pp]
            for kind, size, delta in effects:
                d = null + delta
                mean, low, high = bootstrap_mean(d, replicates, int(seed) + stable_hash(f"{a}|{b}|{kind}|{size}"))
                rows.append({"period": per, "model": model, "horizon_min": int(horizon), "pair": f"{a}-{b}",
                             "patients": int(len(d)), "dmd": full, "dmd_held_out": held, "effect_kind": kind,
                             "effect": size, "added": delta, "difference": mean, "low": low, "high": high,
                             "rule": rule_verdict(mean, low, high, held), "interval": interval_verdict(low, high)})
    return pd.DataFrame(rows)


def family_of(model: str) -> str:
    return model.split("_")[0]


def calibration_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Null comparisons (no difference added): per family and overall, the share of each verdict of the tie rule
    and the share of intervals that exclude 0 (false findings without the DMD)."""
    null = comparisons[(comparisons["effect_kind"] == "dmd") & (comparisons["effect"] == 0)].copy()
    null["family"] = null["model"].map(family_of)
    rows = []
    for family, g in list(null.groupby("family")) + [("all", null)]:
        rows.append({"family": family, "comparisons": int(len(g)), "models": int(g["model"].nunique()),
                     "rule_tie": float((g["rule"] == "tie").mean()),
                     "rule_uncertain": float((g["rule"] == "uncertain").mean()),
                     "rule_false_finding": float(g["rule"].isin(["better", "worse"]).mean()),
                     "interval_false_finding": float(g["interval"].isin(["better", "worse"]).mean()),
                     "abs_difference_pp_median": 100 * float(g["difference"].abs().median()),
                     "dmd_pp_median": 100 * float(g["dmd_held_out"].median())})
    return pd.DataFrame(rows)


def power_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Per added difference (in DMDs or points): how often each rule finds the model better, and how often the tie
    rule calls it a tie."""
    rows = []
    for (kind, size), g in comparisons.groupby(["effect_kind", "effect"]):
        rows.append({"effect_kind": kind, "effect": size, "comparisons": int(len(g)),
                     "added_pp_median": 100 * float(g["added"].median()),
                     "rule_better": float((g["rule"] == "better").mean()),
                     "rule_tie": float((g["rule"] == "tie").mean()),
                     "rule_uncertain": float((g["rule"] == "uncertain").mean()),
                     "rule_worse": float((g["rule"] == "worse").mean()),
                     "interval_better": float((g["interval"] == "better").mean()),
                     "interval_worse": float((g["interval"] == "worse").mean())})
    return pd.DataFrame(rows)


# ============================================================================ 4. the paired design


def unpaired_interval(a: np.ndarray, b: np.ndarray, replicates: int, seed: int) -> tuple[float, float, float]:
    """The difference of the means of two arms whose patients are resampled independently (no pairing)."""
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, a.size, size=(replicates, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(replicates, b.size))].mean(axis=1)
    d = ma - mb
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def pairing_efficiency(skill: pd.DataFrame, pairs: list[tuple[str, str]], by: list[str], replicates: int,
                       seed: int, value: str = "skill", arm: str = "model") -> pd.DataFrame:
    """Per group (``by``) and pair of arms on the same patients: the interval of the mean difference when the arms
    are paired by patient (as the method does) and when the same values are treated as two independent groups;
    the ratio of the widths, the correlation of the two arms across patients, and the patients per group that a
    parallel design (each patient in one arm) would need for the precision of the paired one."""
    rows = []
    for key, g in skill.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        wide = g.pivot_table(index="patient_id", columns=arm, values=value)
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            both = wide[[a, b]].dropna()
            if len(both) < 5:
                continue
            x, y = both[a].to_numpy(float), both[b].to_numpy(float)
            mean, low, high = bootstrap_mean(x - y, replicates, seed)
            _, ulow, uhigh = unpaired_interval(x, y, replicates, seed)
            var_d = float(np.var(x - y, ddof=1))
            need = int(np.ceil(len(x) * (np.var(x, ddof=1) + np.var(y, ddof=1)) / var_d)) if var_d > 0 else np.nan
            rows.append(dict(zip(by, key)) | {
                "arm": a, "against": b, "patients": int(len(x)), "difference": mean, "paired_low": low,
                "paired_high": high, "unpaired_low": ulow, "unpaired_high": uhigh, "paired_width": high - low,
                "unpaired_width": uhigh - ulow, "width_ratio": (uhigh - ulow) / (high - low) if high > low else np.nan,
                "correlation": float(np.corrcoef(x, y)[0, 1]), "parallel_patients_per_arm": need})
    return pd.DataFrame(rows)


def crossover_efficiency(days: pd.DataFrame, pairs: list[tuple[str, str]], by: list[str], replicates: int, seed: int,
                         draws: int = 20, arm: str = "policy") -> pd.DataFrame:
    """Per group and pair of arms: the interval of the mean per-patient skill difference when both arms are scored
    on every test day (paired by day and origin), against a crossover in time in which each patient's days are
    split at random between the two arms (each arm scored against persistence on its own days); mean width over
    ``draws`` random splits. ``days``: day metrics with ``n`` and ``sse`` per patient, arm, horizon and day, and the
    persistence rows (arm ``reference``)."""
    rows = []
    for key, g in days.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        sse = g.pivot_table(index=["patient_id", "day"], columns=arm, values="sse", aggfunc="sum")
        n = g.pivot_table(index=["patient_id", "day"], columns=arm, values="n", aggfunc="sum")
        if "reference" not in sse:
            continue
        for a, b in pairs:
            if a not in sse or b not in sse:
                continue
            keep = sse[[a, b, "reference"]].notna().all(axis=1)
            s, c = sse[keep], n[keep]
            patients = s.index.get_level_values("patient_id")

            def skill(column: str, mask: np.ndarray) -> pd.Series:
                part_s, part_n = s[column][mask].groupby(patients[mask]).sum(), c[column][mask].groupby(
                    patients[mask]).sum()
                ref_s, ref_n = s["reference"][mask].groupby(patients[mask]).sum(), c["reference"][mask].groupby(
                    patients[mask]).sum()
                return 1 - np.sqrt(part_s / part_n) / np.sqrt(ref_s / ref_n)

            everything = np.ones(len(s), dtype=bool)
            d = (skill(a, everything) - skill(b, everything)).dropna().to_numpy()
            if len(d) < 5:
                continue
            mean, low, high = bootstrap_mean(d, replicates, seed)
            rng = np.random.default_rng([int(seed), stable_hash("|".join(map(str, key)) + f"|{a}|{b}")])
            widths, means = [], []
            for k in range(int(draws)):
                to_a = rng.random(len(s)) < 0.5
                dk = (skill(a, to_a) - skill(b, ~to_a)).dropna().to_numpy()
                m, lo, hi = bootstrap_mean(dk, replicates, seed + k + 1)
                widths.append(hi - lo)
                means.append(m)
            rows.append(dict(zip(by, key)) | {
                "arm": a, "against": b, "patients": int(len(d)), "difference": mean, "paired_low": low,
                "paired_high": high, "paired_width": high - low, "crossover_difference": float(np.mean(means)),
                "crossover_width": float(np.mean(widths)),
                "width_ratio": float(np.mean(widths)) / (high - low) if high > low else np.nan})
    return pd.DataFrame(rows)


# ============================================================================ 5. reproducibility and cost


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_tables(a: Path, b: Path, rtol: float, atol: float) -> tuple[str, float]:
    """Whether two CSV files hold the same table: ``equal`` (same cells, numbers within the tolerance) or
    ``different``, with the largest absolute difference of the numeric cells."""
    try:
        x, y = pd.read_csv(a, low_memory=False), pd.read_csv(b, low_memory=False)
    except Exception:  # noqa: BLE001
        return "different", np.nan
    if list(x.columns) != list(y.columns) or x.shape != y.shape:
        return "different", np.nan
    largest = 0.0
    for column in x.columns:
        u, v = x[column], y[column]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            u, v = u.to_numpy(float), v.to_numpy(float)
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                return "different", np.nan
            ok = ~np.isnan(u)
            if ok.any():
                largest = max(largest, float(np.max(np.abs(u[ok] - v[ok]))))
                if not np.allclose(u[ok], v[ok], rtol=rtol, atol=atol):
                    return "different", largest
        elif not u.astype(str).equals(v.astype(str)):
            return "different", np.nan
    return "equal", largest


def compare_outputs(a: Path, b: Path, scripts: list[str] | None = None, rtol: float = 1e-9, atol: float = 1e-12,
                    max_mb: float = 200.0) -> pd.DataFrame:
    """Every CSV of the script folders of two results trees: ``identical`` (the same bytes), ``equal`` (the same
    table, numbers within the tolerance), ``different``, or present in one tree only. ``scripts`` limits the
    folders compared."""
    rows = []
    folders = sorted({p.name for p in a.iterdir() if p.is_dir()} | {p.name for p in b.iterdir() if p.is_dir()})
    for folder in folders:
        if scripts and folder not in scripts:
            continue
        files = set()
        for root in (a / folder, b / folder):
            if root.exists():
                files |= {p.relative_to(root) for p in root.rglob("*.csv")}
        for name in sorted(files):
            x, y = a / folder / name, b / folder / name
            row = {"script": folder, "file": str(name)}
            if not x.exists() or not y.exists():
                rows.append(row | {"status": "only in " + ("second" if not x.exists() else "first")})
                continue
            size = max(x.stat().st_size, y.stat().st_size) / 2 ** 20
            if file_digest(x) == file_digest(y):
                rows.append(row | {"status": "identical", "size_mb": round(size, 2)})
            elif size > max_mb:
                rows.append(row | {"status": "different bytes (too large to compare cells)", "size_mb": round(size, 2)})
            else:
                status, largest = compare_tables(x, y, rtol, atol)
                rows.append(row | {"status": status, "largest_difference": largest, "size_mb": round(size, 2)})
    return pd.DataFrame(rows)


def rerun_scripts(a: Path, b: Path, exclude: tuple = ()) -> list[str]:
    """The script folders of two results trees whose run manifests differ (the script ran again): only those are
    worth comparing, since a folder copied unchanged is trivially identical."""
    out = []
    for folder in sorted(p for p in a.iterdir() if p.is_dir() and p.name not in exclude):
        x, y = folder / "run_manifest.json", b / folder.name / "run_manifest.json"
        if x.exists() and y.exists() and file_digest(x) != file_digest(y):
            out.append(folder.name)
    return out


STAGES = (("0", "data"), ("1", "stage 0"), ("2", "stage 1"), ("3", "stage 2"), ("4", "stage 3"),
          ("5", "confirmation"), ("6", "secondary"), ("7", "secondary"), ("9", "paper"))


def stage_of(script: str) -> str:
    return next((name for prefix, name in STAGES if script.startswith(prefix)), "other")


PER_USE = ("policy", "arm", "test_origins", "served_by", "skipped", "excluded")


def fit_seconds(table: pd.DataFrame) -> tuple[int, float]:
    """Number of fits that ran and their compute seconds (``compute_seconds``, else ``fit_seconds``). A fit shared by
    several policies or arms is logged once per use (stages 2 and 3); it is counted once."""
    if any(c in table for c in PER_USE[:2]):
        table = table.drop_duplicates(subset=[c for c in table.columns if c not in PER_USE])
    column = "compute_seconds" if "compute_seconds" in table else "fit_seconds" if "fit_seconds" in table else None
    if column is None:
        return int(len(table)), np.nan
    seconds = pd.to_numeric(table[column], errors="coerce")
    return int(seconds.notna().sum()), float(seconds.sum())


def run_costs(results: Path, derived: Path | None = None) -> pd.DataFrame:
    """Per script with a manifest: release, finish time, wall time of the last run (cached work is not counted),
    device and workers, and the fits of its fit log (``results/<script>/fit_log.csv`` or the parquet one of
    ``<derived>/runs/<script>/``) with their compute hours."""
    rows = []
    for manifest in sorted(results.glob("*/run_manifest.json")):
        script = manifest.parent.name
        try:
            m = json.loads(manifest.read_text())
        except Exception:  # noqa: BLE001
            continue
        resources = m.get("resources") or {}
        if isinstance(resources, str):
            resources = {}
        row = {"script": script, "stage": stage_of(script), "release": m.get("package_version"),
               "finished_at": m.get("finished_at"), "wall_seconds": m.get("duration_seconds"),
               "device": resources.get("device"), "workers": resources.get("n_workers")}
        log = None
        if (manifest.parent / "fit_log.csv").exists():
            log = pd.read_csv(manifest.parent / "fit_log.csv", low_memory=False)
        elif derived is not None and (derived / "runs" / script / "fit_log.parquet").exists():
            log = pd.read_parquet(derived / "runs" / script / "fit_log.parquet")
        if log is not None and len(log):
            fits, seconds = fit_seconds(log)
            row |= {"fits": fits, "fit_compute_hours": seconds / 3600 if np.isfinite(seconds) else np.nan}
        rows.append(row)
    return pd.DataFrame(rows)
