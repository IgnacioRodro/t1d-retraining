"""Cohorts, patient groups, day 0 and training anchors (script 04).

Input: the clean tables of script 03 (``<derived>/clean/<dataset>/days.parquet``
and ``patients.parquet``). Definitions (decisions D12, D50-D52 in HANDOFF.md):

- **Usable day.** A valid CGM day (script 03) in the loop mode of the cohort,
  with complete insulin when the cohort's models need insulin
  (``USABLE_RULES``).
- **Cohort.** The patients of one dataset with at least ``min_valid_days_core``
  usable days of one kind (``COHORTS``). A patient can be in several cohorts.
  The REPLACE-BG cohort ``core`` is split into the groups pool / screen /
  confirm; the other cohorts are confirmation or secondary analyses.
- **Groups.** Drawn once (seeded), stratified by arm and by tertile of usable
  days, by systematic allocation along a random order inside each stratum, so
  that every stratum is shared out in the target proportions. Of
  ``group_balance_draws`` such draws, the one with the smallest largest
  standardised mean difference over ``group_balance_on`` is kept
  (re-randomisation).
- **Period.** ``whole`` (the whole follow-up) or ``post_randomization`` (from
  the randomisation day; REPLACE-BG and DCLP3).
- **Day 0** of a patient in a period: the first day from which the next
  ``train_window_days`` calendar days hold at least ``min_train_usable_days``
  usable days.
- **Anchor a** (days 14, 42, 70, 98 after day 0): a model is trained with the
  ``train_window_days`` days before day 0 + a and tested from day 0 + a. The
  anchor is available when that window holds at least ``min_train_usable_days``
  usable days and the first ``max_age_days`` days after it hold at least
  ``min_test_usable_days``.
- **Change events.** Candidate real changes for stage 3: randomisation
  (REPLACE-BG, DCLP3), a change of loop mode, or a change of CGM sensor, each
  between runs of at least 14 valid days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .demographics import female_indicator
from .quality import SENSOR_GROUPS
from .store import ProcessedStore

USABLE_RULES = {
    "open_insulin": "valid CGM day in open loop with complete insulin",
    "open": "valid CGM day in open loop",
    "closed_insulin": "valid CGM day in closed loop with complete insulin",
    "valid": "valid CGM day",
    "insulin": "valid CGM day with complete insulin, open or closed loop (across a loop change, script 72)",
}
PERIODS = {"replace_bg": ["whole", "post_randomization"], "dclp3": ["whole", "post_randomization"]}
GROUP_ORDER = ["pool", "screen", "confirm"]
CHANGE_MIN_RUN_DAYS = 14
CHANGE_WINDOW_DAYS = 28


@dataclass(frozen=True)
class Cohort:
    name: str
    usable: str
    role: str
    description: str
    arm: str | None = None
    min_usable_days: int | None = None  # None: min_valid_days_core


COHORTS = {
    "replace_bg": [Cohort("core", "open_insulin", "core", "core study: groups pool / screen / confirm")],
    "dclp3": [
        Cohort("sap_open_insulin", "open_insulin", "confirmation", "SAP arm with pump data", arm="SAP"),
        Cohort("sap_open_glucose", "open", "confirmation (glucose only)", "SAP arm, glucose-only models", arm="SAP"),
        Cohort("clc_closed_insulin", "closed_insulin", "closed loop", "Control-IQ arm after randomisation", arm="CLC"),
    ],
    "loop": [Cohort("closed_insulin", "closed_insulin", "closed loop", "Loop users")],
    "hupa_ucm": [
        Cohort("open_insulin", "open_insulin", "confirmation", "MDI and pumps, study window"),
        Cohort("long_glucose", "valid", "case study (glucose only)", "long raw history", min_usable_days=98),
    ],
    "brist1d_open": [
        Cohort("open_insulin", "open_insulin", "confirmation", "open-loop periods"),
        Cohort("closed_insulin", "closed_insulin", "closed loop", "closed-loop periods"),
    ],
    "t1d_uom": [
        Cohort("open_insulin", "open_insulin", "confirmation", "MDI with wearables"),
        Cohort("open_glucose", "open", "confirmation (glucose only)", "MDI, glucose-only models"),
    ],
}


def cohort_rule(dataset: str, cohort: str) -> str:
    """The usable-day rule of a cohort (``USABLE_RULES``)."""
    for c in COHORTS.get(dataset, []):
        if c.name == cohort:
            return c.usable
    raise KeyError(f"unknown cohort {dataset}/{cohort} (data.splits.COHORTS)")


# ----------------------------------------------------------------------------- usable days and counts


def usable_flags(days: pd.DataFrame) -> pd.DataFrame:
    """One boolean column ``usable_<rule>`` per usable rule."""
    valid = days["valid"].fillna(False).astype(bool).to_numpy()
    loop = days["loop"].astype(str).to_numpy()
    insulin = days["insulin_complete"].fillna(False).astype(bool).to_numpy()
    return pd.DataFrame({
        "usable_open_insulin": valid & (loop == "open") & insulin,
        "usable_open": valid & (loop == "open"),
        "usable_closed_insulin": valid & (loop == "closed") & insulin,
        "usable_valid": valid,
        "usable_insulin": valid & insulin,
    }, index=days.index)


def randomization_index(days: pd.DataFrame) -> float:
    """``day_index`` of the first post-randomisation day of one patient (NaN if none)."""
    post = days.loc[days["period"] == "post_randomization", "day_index"]
    return float(post.min()) if len(post) else float("nan")


def period_mask(days: pd.DataFrame, period: str) -> np.ndarray:
    if period == "whole":
        return np.ones(len(days), dtype=bool)
    return (days["period"] == period).to_numpy()


# ----------------------------------------------------------------------------- day 0 and anchors


def day_zero(usable: np.ndarray, window: int, min_usable: int) -> int | None:
    """First index d with at least ``min_usable`` usable days in [d, d + window)."""
    if len(usable) < 1:
        return None
    csum = np.concatenate([[0], np.cumsum(usable.astype(int))])
    end = np.minimum(np.arange(len(usable)) + window, len(usable))
    ok = np.flatnonzero(csum[end] - csum[:len(usable)] >= min_usable)
    return int(ok[0]) if ok.size else None


def anchor_rows(usable: np.ndarray, anchors: list[int], settings: dict) -> tuple[int | None, list[dict]]:
    """Day 0 (index into ``usable``) and one row per anchor."""
    window = int(settings["train_window_days"])
    min_train = int(settings["min_train_usable_days"])
    min_test = int(settings["min_test_usable_days"])
    max_age = int(settings["max_age_days"])
    d0 = day_zero(usable, window, min_train)
    rows = []
    for anchor in anchors:
        if d0 is None:
            rows.append({"anchor_day": anchor, "train_usable_days": 0, "test_usable_days": 0,
                         "test_usable_days_within_max_age": 0, "available": False})
            continue
        at = d0 + anchor
        train = int(usable[max(at - window, 0):at].sum())
        test = int(usable[at:].sum())
        test_age = int(usable[at:at + max_age].sum())
        rows.append({"anchor_day": anchor, "train_usable_days": train, "test_usable_days": test,
                     "test_usable_days_within_max_age": test_age,
                     "available": bool(train >= min_train and test_age >= min_test)})
    return d0, rows


# ----------------------------------------------------------------------------- change events


def _runs(labels: pd.Series, valid: pd.Series, day_index: pd.Series) -> pd.DataFrame:
    """Runs of equal labels (days without a label are skipped), with their valid days."""
    keep = labels.notna() & (labels.astype(str) != "unknown")
    frame = pd.DataFrame({"label": labels[keep].astype(str), "valid": valid[keep].astype(bool),
                          "day_index": day_index[keep]})
    if frame.empty:
        return pd.DataFrame(columns=["label", "start", "end", "valid_days"])
    run = (frame["label"] != frame["label"].shift()).cumsum()
    runs = frame.groupby(run).agg(label=("label", "first"), start=("day_index", "min"), end=("day_index", "max"),
                                  valid_days=("valid", "sum")).reset_index(drop=True)
    return runs


def _stable_changes(runs: pd.DataFrame) -> list[tuple[int, str, str]]:
    """(first day of the new run, from, to) between runs of >= CHANGE_MIN_RUN_DAYS valid days."""
    runs = runs[runs["valid_days"] >= CHANGE_MIN_RUN_DAYS]
    if runs.empty:
        return []
    merged = (runs["label"] != runs["label"].shift()).cumsum()
    runs = runs.groupby(merged).agg(label=("label", "first"), start=("start", "min")).reset_index(drop=True)
    return [(int(runs["start"].iloc[k]), runs["label"].iloc[k - 1], runs["label"].iloc[k]) for k in range(1, len(runs))]


def change_events(days: pd.DataFrame) -> list[dict]:
    """Candidate real change events of one patient."""
    days = days.sort_values("day_index")
    valid = days["valid"].fillna(False).astype(bool)
    index = days["day_index"]
    events = []
    r = randomization_index(days)
    if np.isfinite(r) and (days["period"] == "pre_randomization").any():
        events.append((int(r), "randomization", "pre", "post"))
    for kind, labels in [("loop_change", days["loop"]),
                         ("sensor_change", days["source"].map(lambda s: SENSOR_GROUPS.get(s, s) if isinstance(s, str)
                                                              else None))]:
        for start, before, after in _stable_changes(_runs(labels, valid, index)):
            events.append((start, kind, before, after))
    rows = []
    position = dict(zip(index, range(len(days))))
    flags = valid.to_numpy()
    for start, kind, before, after in events:
        k = position.get(start, int(np.searchsorted(index.to_numpy(), start)))
        rows.append({"kind": kind, "day_index": start, "day": days["day"].iloc[min(k, len(days) - 1)],
                     "from": before, "to": after,
                     "valid_days_before": int(flags[max(k - CHANGE_WINDOW_DAYS, 0):k].sum()),
                     "valid_days_after": int(flags[k:k + CHANGE_WINDOW_DAYS].sum())})
    return rows


# ----------------------------------------------------------------------------- one dataset (worker)


def assess_dataset(job: dict) -> dict:
    """Usable-day counts, cohorts, day 0, anchors and change events of one dataset."""
    dataset = job["dataset"]
    store = ProcessedStore(dataset, root=job["derived_root"], stage="clean")
    days = pd.read_parquet(store.root / "days.parquet")
    patients = store.read_patients()
    days = days.sort_values(["patient_id", "day_index"]).reset_index(drop=True)
    days = pd.concat([days, usable_flags(days)], axis=1)
    arms = patients.set_index("patient_id")["arm"] if "arm" in patients else pd.Series(dtype=object)
    periods = PERIODS.get(dataset, ["whole"])
    min_default = int(job["min_valid_days"])
    anchors = list(job["anchors"])

    count_rows, member_rows, anchor_out, event_rows = [], [], [], []
    for patient_id, p in days.groupby("patient_id", sort=True):
        r_index = randomization_index(p)
        row = {"dataset": dataset, "patient_id": patient_id, "arm": arms.get(patient_id),
               "first_day": p["day"].min(), "last_day": p["day"].max(), "calendar_days": len(p),
               "randomization_day_index": r_index}
        for rule in USABLE_RULES:
            column = f"usable_{rule}"
            row[column] = int(p[column].sum())
            if "post_randomization" in periods:
                row[f"{column}_post"] = int(p.loc[p["period"] == "post_randomization", column].sum())
                row[f"{column}_pre"] = int(p.loc[p["period"] == "pre_randomization", column].sum())
        count_rows.append(row)
        for cohort in COHORTS.get(dataset, []):
            if cohort.arm and row["arm"] != cohort.arm:
                continue
            needed = cohort.min_usable_days or min_default
            if row[f"usable_{cohort.usable}"] < needed:
                continue
            member_rows.append({"dataset": dataset, "patient_id": patient_id, "cohort": cohort.name,
                                "usable_rule": cohort.usable, "usable_days": row[f"usable_{cohort.usable}"]})
            for period in periods:
                inside = period_mask(p, period)
                if not inside.any():
                    continue
                part = p[inside]
                d0, rows = anchor_rows(part[f"usable_{cohort.usable}"].to_numpy(bool), anchors, job["settings"])
                base = {"dataset": dataset, "cohort": cohort.name, "patient_id": patient_id, "period": period,
                        "day0_index": int(part["day_index"].iloc[d0]) if d0 is not None else np.nan,
                        "day0": part["day"].iloc[d0] if d0 is not None else pd.NaT,
                        "period_usable_days": int(part[f"usable_{cohort.usable}"].sum()),
                        "days_after_day0": int(len(part) - d0) if d0 is not None else 0}
                base["randomization_offset"] = (r_index - base["day0_index"]) if np.isfinite(r_index) and \
                    d0 is not None else np.nan
                anchor_out += [{**base, **a} for a in rows]
        for event in change_events(p):
            event_rows.append({"dataset": dataset, "patient_id": patient_id, **event})

    covariates = patient_covariates(days, patients)
    return {"dataset": dataset, "counts": pd.DataFrame(count_rows), "members": pd.DataFrame(member_rows),
            "anchors": pd.DataFrame(anchor_out), "events": pd.DataFrame(event_rows), "covariates": covariates}


def patient_covariates(days: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Static and glycaemic covariates per patient, over valid days (for the balance table)."""
    v = days[days["valid"].fillna(False).astype(bool)].copy()
    v["w"] = v["n_readings"].astype(float)
    v["m1"] = v["glucose_mean"] * v["w"]
    v["m2"] = (v["glucose_sd"].fillna(0) ** 2 + v["glucose_mean"] ** 2) * v["w"]
    v["tir_w"] = v["tir_70_180"] * v["w"]
    g = v.groupby("patient_id")
    mean = g["m1"].sum() / g["w"].sum()
    second = g["m2"].sum() / g["w"].sum()
    out = pd.DataFrame({
        "mean_glucose": mean,
        "glucose_cv": np.sqrt(np.maximum(second - mean ** 2, 0)) / mean,
        "tir_70_180": g["tir_w"].sum() / g["w"].sum(),
        "carbs_g_per_day": g["carbs_g"].mean(),
        "boluses_per_day": g["bolus_n"].mean(),
    })
    static = patients.drop_duplicates("patient_id").set_index("patient_id")
    out = out.reindex(out.index.union(static.index))
    out.index.name = "patient_id"
    if "age" in static:
        out["age"] = pd.to_numeric(static["age"], errors="coerce")
    if "hba1c" in static:
        out["hba1c"] = pd.to_numeric(static["hba1c"], errors="coerce")
    if "sex" in static:
        out["female"] = female_indicator(static["sex"])
    if "diabetes_duration_years" in static:
        out["diabetes_duration_years"] = pd.to_numeric(static["diabetes_duration_years"], errors="coerce")
    return out.reset_index()


# ----------------------------------------------------------------------------- groups and balance


def allocate(order: list[str], sizes: dict[str, int]) -> dict[str, str]:
    """Systematic allocation: walk ``order`` and give each patient to the group furthest below its share."""
    total = len(order)
    names = list(sizes)
    counts = dict.fromkeys(names, 0)
    out = {}
    for k, patient_id in enumerate(order):
        deficit = [sizes[g] * (k + 1) / total - counts[g] for g in names]
        group = names[int(np.argmax(deficit))]
        out[patient_id] = group
        counts[group] += 1
    if counts != sizes:  # cannot happen with exact quotas; kept as a guard
        raise RuntimeError(f"allocation gave {counts}, expected {sizes}")
    return out


def max_abs_smd(values: pd.DataFrame, groups: np.ndarray) -> float:
    """Largest |SMD| over the columns of ``values`` and every pair of groups (missing values ignored)."""
    names = [g for g in GROUP_ORDER if g in set(groups)]
    worst = 0.0
    for column in values.columns:
        x = values[column].to_numpy(float)
        for k, g1 in enumerate(names):
            for g2 in names[k + 1:]:
                a, b = x[(groups == g1) & np.isfinite(x)], x[(groups == g2) & np.isfinite(x)]
                if len(a) < 2 or len(b) < 2:
                    continue
                pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                if pooled > 0:
                    worst = max(worst, abs(a.mean() - b.mean()) / pooled)
    return float(worst)


def add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Stratum of each patient: arm x tertile of usable days."""
    frame = frame.copy()
    frame["usable_days_tertile"] = pd.qcut(frame["usable_days"].rank(method="first"), 3, labels=["low", "mid", "high"])
    frame["stratum"] = frame["arm"].fillna("na").astype(str) + " | " + frame["usable_days_tertile"].astype(str)
    return frame


def assign_groups(eligible: pd.DataFrame, sizes: dict[str, int | None], seed: int,
                  balance_on: list[str] | None = None, draws: int = 1) -> tuple[pd.DataFrame, dict]:
    """Groups for the eligible patients.

    ``eligible`` has patient_id, arm, usable_days and the ``balance_on``
    covariates. Each draw orders the patients at random inside each stratum
    (arm x tertile of usable days) and allocates them systematically; with
    ``draws`` > 1 the draw with the smallest largest |SMD| over ``balance_on``
    is kept (re-randomisation, Morgan and Rubin 2012). Returns the table
    (patient_id, arm, usable_days, usable_days_tertile, stratum, group) and a
    description of the draw.
    """
    frame = eligible.sort_values("patient_id").reset_index(drop=True).copy()
    fixed = {g: int(n) for g, n in sizes.items() if n is not None}
    rest = [g for g, n in sizes.items() if n is None]
    if sum(fixed.values()) > len(frame):
        raise ValueError(f"{len(frame)} eligible patients, fewer than the {sum(fixed.values())} needed")
    quotas = {**fixed, **{g: len(frame) - sum(fixed.values()) for g in rest[:1]}}
    frame = add_strata(frame)
    columns = [c for c in (balance_on or []) if c in frame and frame[c].notna().any()]
    best = None
    for draw in range(max(int(draws), 1)):
        rng = np.random.default_rng([seed, draw])
        order = frame.assign(key=rng.permutation(len(frame))).sort_values(["stratum", "key"])["patient_id"].tolist()
        groups = allocate(order, quotas)
        score = max_abs_smd(frame[columns], frame["patient_id"].map(groups).to_numpy()) if columns else 0.0
        if best is None or score < best[0]:
            best = (score, draw, groups)
    frame["group"] = frame["patient_id"].map(best[2])
    info = {"draws": int(max(int(draws), 1)), "chosen_draw": int(best[1]), "balance_on": columns,
            "max_abs_smd": round(best[0], 4)}
    return frame[["patient_id", "arm", "usable_days", "usable_days_tertile", "stratum", "group"]], info


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference (pooled SD); for 0/1 variables this is the usual binary SMD."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def balance_table(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Mean (SD) per group and SMD for each pair of groups."""
    rows = []
    groups = [g for g in GROUP_ORDER if g in set(frame["group"])]
    for covariate in covariates:
        if covariate not in frame or frame[covariate].notna().sum() == 0:
            continue
        row = {"covariate": covariate}
        for g in groups:
            values = frame.loc[frame["group"] == g, covariate].astype(float)
            row[f"{g}_mean"] = round(float(values.mean()), 3)
            row[f"{g}_sd"] = round(float(values.std()), 3)
        for k, g1 in enumerate(groups):
            for g2 in groups[k + 1:]:
                row[f"smd_{g1}_vs_{g2}"] = round(smd(frame.loc[frame["group"] == g1, covariate],
                                                     frame.loc[frame["group"] == g2, covariate]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- reading the saved groups


def load_split(dataset: str, splits_dir=None) -> dict:
    """The split file written by script 04 (config/splits/<dataset>.yaml)."""
    import yaml

    from ..config import CONFIG_DIR

    target = (splits_dir or CONFIG_DIR / "splits") / f"{dataset}.yaml"
    if not target.exists():
        raise FileNotFoundError(f"{target} not found: run scripts/04_define_splits.py first")
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def group_patients(dataset: str, group: str, allow_confirm: bool = False, splits_dir=None) -> list[str]:
    """Patients of one core group. The confirm group is a hold-out: only the confirmation scripts (50+) may
    ask for it, with ``allow_confirm=True``."""
    if group == "confirm" and not allow_confirm:
        raise PermissionError("the confirm group is held out until the confirmation stage (scripts 50 and 51)")
    groups = load_split(dataset, splits_dir).get("groups") or {}
    if group not in groups:
        raise KeyError(f"no group {group!r} in the split file of {dataset}")
    return [str(p) for p in groups[group]]
