"""Recovery after real changes (script 72; HANDOFF 7.10, D97).

A real change (a switch to closed loop, a new sensor, the randomisation of a
trial) is the anchor: every model is trained on the days before it and then
follows the confirmation's arms (never, the refit at the optimal interval and
every day, its mechanism) over the days after it. Per group of events:

- ``select_events``: the events of script 04 (``change_events``) of one group,
  with enough valid days on each side;
- ``weekly_skill``: per model, arm, horizon and week since the change, the mean
  over events (a patient and change) of each one's mean daily skill (against
  persistence on the same days);
- ``weekly_gain``: the gain of every refit over never, week by week;
- ``group_contrast``: a group against its control (the same calendar without the
  change): the difference of their means, never's skill and the gain of every
  refit, with a bootstrap interval that resamples each group's patients;
- ``against_trained_after``: a model trained before the change against the same
  model trained after it (the confirmation's anchor, script 50), on the same
  patients and days: what the change costs a frozen model and how fast each
  refit closes the gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ageing_analysis as aa
from .stats import bootstrap_mean

GROUP_KEYS = ["cohort", "model", "policy", "horizon_min"]


def select_events(events: pd.DataFrame, group: dict, arms: pd.Series, members: list[str] | None,
                  min_days: int) -> tuple[pd.DataFrame, dict]:
    """The events of one group (``dataset``, ``kind``; the trial arm and the core group when it names them) with at
    least ``min_days`` valid days on each side, and the counts behind them."""
    e = events[(events["kind"] == group["kind"])].copy()
    counts = {"events": int(len(e))}
    if group.get("arm"):
        e = e[e["patient_id"].map(arms).eq(group["arm"])]
        counts["in_arm"] = int(len(e))
    if members is not None:
        e = e[e["patient_id"].isin(members)]
        counts["in_group"] = int(len(e))
    e = e[(e["valid_days_before"] >= min_days) & (e["valid_days_after"] >= min_days)]
    counts["enough_days"] = int(len(e))
    return e.reset_index(drop=True), counts


def daily(days: pd.DataFrame, min_origins: int, clip: tuple) -> pd.DataFrame:
    """Daily skill of every model and arm (``version``) with the cohort, from day metrics with persistence."""
    parts = []
    for cohort, d in days.groupby("cohort", sort=False):
        s = aa.daily_skill(d.rename(columns={"policy": "version"}), min_origins, clip)
        parts.append(s.assign(cohort=cohort))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty:
        out = out.rename(columns={"version": "policy"})
        out["week"] = (out["days_since_anchor"] // 7).astype(int)
        out["event"] = out["patient_id"].astype(str) + "@" + out["anchor_day"].astype(str)
    return out


def _weekly(values: pd.DataFrame, column: str, keys: list[str]) -> pd.DataFrame:
    per_event = values.groupby(keys + ["week", "event"])[column].mean()
    g = per_event.groupby(level=keys + ["week"])
    table = pd.DataFrame({"events": g.size(), column: g.mean(), "sd": g.std()}).reset_index()
    half = 1.96 * table["sd"] / np.sqrt(table["events"].clip(lower=1))
    table["low"], table["high"] = table[column] - half, table[column] + half
    return table.drop(columns="sd")


def weekly_skill(skill: pd.DataFrame) -> pd.DataFrame:
    """Per cohort, model, arm, horizon and week: mean over events of their mean daily skill (normal 95 % interval)."""
    return _weekly(skill, "skill", GROUP_KEYS)


def daily_gains(skill: pd.DataFrame) -> pd.DataFrame:
    """Per cohort, model, refit arm, horizon, event and day: daily skill of the refit minus that of never."""
    wide = skill.pivot_table(index=["cohort", "model", "horizon_min", "event", "patient_id", "day", "week"],
                             columns="policy", values="skill")
    if "never" not in wide:
        return pd.DataFrame()
    parts = [(wide[arm] - wide["never"]).dropna().rename("gain").reset_index().assign(policy=arm)
             for arm in wide.columns if arm != "never"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def weekly_gain(gains: pd.DataFrame) -> pd.DataFrame:
    """Per cohort, model, refit arm, horizon and week: mean over events of their mean daily gain over never."""
    return _weekly(gains, "gain", GROUP_KEYS) if not gains.empty else pd.DataFrame()


def event_measures(skill: pd.DataFrame, gains: pd.DataFrame, weeks: list[int] | None = None) -> pd.DataFrame:
    """Per cohort, model, horizon and event (over ``weeks``, every week when None): never's mean daily skill and
    every refit's mean daily gain over never."""
    s = skill[skill["policy"] == "never"]
    if weeks is not None:
        s = s[s["week"].isin(weeks)]
        if not gains.empty:
            gains = gains[gains["week"].isin(weeks)]
    keys = ["cohort", "model", "horizon_min", "event"]
    rows = [s.groupby(keys)["skill"].mean().rename("value").reset_index().assign(measure="skill", policy="never")]
    if not gains.empty:
        rows.append(gains.groupby(keys + ["policy"])["gain"].mean().rename("value").reset_index().assign(
            measure="gain"))
    return pd.concat(rows, ignore_index=True)


def group_contrast(measures: pd.DataFrame, change: str, control: str, replicates: int, seed: int) -> pd.DataFrame:
    """Per model, horizon, arm and measure: the mean in the change group, in the control and their difference, with
    a bootstrap interval that resamples the events of each group on its own."""
    rows = []
    rng = np.random.default_rng(seed)
    for key, g in measures.groupby(["model", "horizon_min", "policy", "measure"], sort=False):
        a = g.loc[g["cohort"] == change, "value"].dropna().to_numpy()
        b = g.loc[g["cohort"] == control, "value"].dropna().to_numpy()
        if len(a) < 3 or len(b) < 3:
            continue
        draws = (a[rng.integers(0, len(a), (replicates, len(a)))].mean(axis=1) -
                 b[rng.integers(0, len(b), (replicates, len(b)))].mean(axis=1))
        rows.append({**dict(zip(["model", "horizon_min", "policy", "measure"], key)), "change_events": len(a),
                     "control_events": len(b), "change_mean": float(a.mean()), "control_mean": float(b.mean()),
                     "difference": float(a.mean() - b.mean()), "low": float(np.quantile(draws, 0.025)),
                     "high": float(np.quantile(draws, 0.975))})
    return pd.DataFrame(rows)


def against_trained_after(before: pd.DataFrame, after: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per model, horizon, arm of the run from the change and week since the change: the daily skill of that arm
    minus the daily skill of the same model trained after the change and frozen (``after``: never at the
    confirmation's anchor), on the same patients and days; mean over patients of their mean, bootstrap interval."""
    keys = ["patient_id", "model", "horizon_min", "day"]
    ref = after[after["policy"] == "never"][keys + ["skill"]].rename(columns={"skill": "after"})
    both = before.merge(ref, on=keys, how="inner")
    if both.empty:
        return pd.DataFrame()
    both["difference"] = both["skill"] - both["after"]
    per_patient = both.groupby(["model", "horizon_min", "policy", "week", "patient_id"])["difference"].mean()
    rows = []
    for key, values in per_patient.groupby(level=[0, 1, 2, 3]):
        mean, low, high = bootstrap_mean(values.to_numpy(), replicates, seed)
        rows.append({**dict(zip(["model", "horizon_min", "policy", "week"], key)), "patients": int(len(values)),
                     "difference": mean, "low": low, "high": high})
    return pd.DataFrame(rows)
