"""Regularity of a patient's habits at the start of the follow-up (script 60; HANDOFF 7.7, Q4, D94).

Every index is computed on the usable days of the patient's startup window,
the ``window_days`` days before the anchor (the 14 days the first model is
trained on), so that it is known when the refit policy would be chosen. The
events are those of script 02 (``processed`` store): meals (carbohydrate
entries of at least ``min_meal_g`` g) and boluses (at least ``min_bolus_u``
U), each merged with the next ones within ``merge_min`` minutes.

- **Timing jitter** (``*_time_jitter_min``): for every event, the clock-time
  distance to the closest event of each other day (circular, in minutes), the
  median over those days, then the median over events. Low: the events come at
  the same times every day.
- **Profile stability** (``*_profile_stability``): the hourly profile of each
  day (24 bins, smoothed over three hours, circular) correlated with the mean
  profile of the other days (Pearson), averaged over days. High: every day
  looks like the others. For glucose, the hourly mean of the readings.
- **Amounts**: events per day and the coefficient of variation of the daily
  total (``*_per_day_cv``).
- **Intermittency** (Syntetos and Boylan): on hourly totals, the average
  interval between hours with an event (``*_adi_h``) and the squared
  coefficient of variation of the non-zero hourly totals (``*_cv2``).
- **Wearables** (BrisT1D, T1D-UOM, HUPA-UCM): coefficient of variation of the
  daily steps and the circular standard deviation of the sleep onset.
- **Glycaemic control** (adjusters, Lu et al. 2026): time in range 70-180
  mg/dL, coefficient of variation and mean of the glucose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440.0

# name -> (group, unit, when the patient is more regular: "lower", "higher" or "" (a descriptor), description)
INDICES = {
    "meal_time_jitter_min": ("meals", "min", "lower", "clock-time distance of a meal to the closest meal of other days"),
    "carb_profile_stability": ("meals", "r", "higher", "correlation of each day's carbohydrate profile with the others"),
    "meals_per_day": ("meals", "1/day", "", "meals per day"),
    "carbs_per_day_cv": ("meals", "", "lower", "coefficient of variation of the daily carbohydrates"),
    "carb_adi_h": ("meals", "h", "", "average interval between hours with a meal"),
    "carb_cv2": ("meals", "", "lower", "squared coefficient of variation of the hourly carbohydrates"),
    "bolus_time_jitter_min": ("boluses", "min", "lower", "clock-time distance of a bolus to the closest bolus of other "
                                                         "days"),
    "bolus_profile_stability": ("boluses", "r", "higher", "correlation of each day's bolus profile with the others"),
    "boluses_per_day": ("boluses", "1/day", "", "boluses per day"),
    "bolus_units_per_day_cv": ("boluses", "", "lower", "coefficient of variation of the daily bolus insulin"),
    "bolus_adi_h": ("boluses", "h", "", "average interval between hours with a bolus"),
    "bolus_cv2": ("boluses", "", "lower", "squared coefficient of variation of the hourly bolus insulin"),
    "glucose_profile_stability": ("glucose", "r", "higher", "correlation of each day's glucose profile with the others"),
    "steps_per_day_cv": ("wearables", "", "lower", "coefficient of variation of the daily steps"),
    "sleep_onset_sd_min": ("wearables", "min", "lower", "circular standard deviation of the sleep onset"),
}
ADJUSTERS = {"tir_pct": "time in range 70-180 mg/dL (%)", "glucose_cv_pct": "coefficient of variation of the glucose "
             "(%)", "glucose_mean": "mean glucose (mg/dL)"}


# ----------------------------------------------------------------------------- building blocks


def merge_events(times: pd.Series, sizes: np.ndarray, merge_min: float) -> pd.DataFrame:
    """Events closer than ``merge_min`` minutes to the previous one join it (sizes added; time of the first)."""
    if not len(times):
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"), "size": pd.Series(dtype=float)})
    order = np.argsort(times.to_numpy("datetime64[ns]"))
    ts = pd.Series(times.to_numpy("datetime64[ns]")[order])
    size = np.asarray(sizes, dtype=float)[order]
    gap = ts.diff().dt.total_seconds().fillna(np.inf).to_numpy() / 60.0
    group = np.cumsum(gap > merge_min)
    frame = pd.DataFrame({"group": group, "ts": ts, "size": size})
    return frame.groupby("group").agg(ts=("ts", "first"), size=("size", "sum")).reset_index(drop=True)


def clock_minutes(ts: pd.Series) -> np.ndarray:
    return ((ts - ts.dt.normalize()).dt.total_seconds() / 60.0).to_numpy(float)


def circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between clock times (minutes), across midnight: at most 720."""
    d = np.abs(np.asarray(a, float)[:, None] - np.asarray(b, float)[None, :]) % MINUTES_PER_DAY
    return np.minimum(d, MINUTES_PER_DAY - d)


def time_jitter(events: pd.DataFrame) -> float:
    """Median over events of the median over the other days of the distance to that day's closest event (min)."""
    if events.empty:
        return np.nan
    day = events["ts"].dt.normalize()
    minutes = clock_minutes(events["ts"])
    days = list(pd.unique(day))
    if len(days) < 2:
        return np.nan
    by_day = {d: minutes[(day == d).to_numpy()] for d in days}
    per_event = []
    for d in days:
        own = by_day[d]
        others = [circular_distance(own, by_day[o]).min(axis=1) for o in days if o != d]
        per_event.append(np.median(np.stack(others, axis=1), axis=1))
    return float(np.median(np.concatenate(per_event)))


def hourly_by_day(ts: pd.Series, values: np.ndarray, days: list, how: str = "sum") -> np.ndarray:
    """A (days x 24) table of hourly totals (``sum``) or means (``mean``, NaN without values) of ``values``."""
    out = np.zeros((len(days), 24)) if how == "sum" else np.full((len(days), 24), np.nan)
    if not len(ts):
        return out
    frame = pd.DataFrame({"day": ts.dt.normalize(), "hour": ts.dt.hour, "value": np.asarray(values, float)})
    frame = frame[frame["day"].isin(days) & np.isfinite(frame["value"])]
    table = frame.groupby(["day", "hour"])["value"].agg(how).unstack("hour").reindex(index=days, columns=range(24))
    values = table.to_numpy(float)
    return np.nan_to_num(values, nan=0.0) if how == "sum" else values


def profile_stability(profiles: np.ndarray, smooth_bins: int = 3) -> float:
    """Mean over days of the correlation of a day's profile with the mean of the other days, after a circular
    moving average of ``smooth_bins`` hours. Days with a flat profile (no events) are left out."""
    p = np.asarray(profiles, float)
    if p.shape[0] < 3:
        return np.nan
    if smooth_bins > 1:
        half = smooth_bins // 2
        p = np.mean([np.roll(p, k, axis=1) for k in range(-half, half + 1)], axis=0)
    keep = np.isfinite(p).all(axis=1) & (np.nanstd(p, axis=1) > 0)
    p = p[keep]
    if p.shape[0] < 3:
        return np.nan
    total = p.sum(axis=0)
    r = []
    for i in range(p.shape[0]):
        others = (total - p[i]) / (p.shape[0] - 1)
        if np.std(others) > 0:
            r.append(np.corrcoef(p[i], others)[0, 1])
    return float(np.mean(r)) if r else np.nan


def intermittency(hourly: np.ndarray) -> tuple[float, float]:
    """(average interval between non-zero hours, squared CV of the non-zero hours) of hourly totals."""
    h = np.asarray(hourly, float).ravel()
    nonzero = h[h > 0]
    if len(nonzero) < 2:
        return np.nan, np.nan
    return float(len(h) / len(nonzero)), float((nonzero.std() / nonzero.mean()) ** 2)


def daily_cv(totals: np.ndarray) -> float:
    t = np.asarray(totals, float)
    return float(t.std() / t.mean()) if len(t) >= 3 and t.mean() > 0 else np.nan


def circular_sd_minutes(minutes: np.ndarray) -> float:
    """Circular standard deviation of clock times (minutes)."""
    if len(minutes) < 3:
        return np.nan
    angle = 2 * np.pi * np.asarray(minutes, float) / MINUTES_PER_DAY
    r = np.hypot(np.cos(angle).mean(), np.sin(angle).mean())
    return float(np.sqrt(-2 * np.log(max(r, 1e-12))) * MINUTES_PER_DAY / (2 * np.pi))


# ----------------------------------------------------------------------------- one patient


def event_indices(events: pd.DataFrame, days: list, prefix: str, unit_name: str) -> dict:
    """Timing, profile, amount and intermittency indices of one kind of event on ``days``."""
    out = {f"{prefix}_events": int(len(events))}
    if len(events) < 5:
        return out
    hourly = hourly_by_day(events["ts"], events["size"].to_numpy(float), days)
    totals = hourly.sum(axis=1)
    adi, cv2 = intermittency(hourly)
    names = {"meal": ("meal_time_jitter_min", "carb_profile_stability", "meals_per_day", "carbs_per_day_cv",
                      "carb_adi_h", "carb_cv2"),
             "bolus": ("bolus_time_jitter_min", "bolus_profile_stability", "boluses_per_day",
                       "bolus_units_per_day_cv", "bolus_adi_h", "bolus_cv2")}[prefix]
    values = (time_jitter(events), profile_stability(hourly), len(events) / max(1, len(days)), daily_cv(totals), adi,
              cv2)
    out.update(dict(zip(names, values)))
    out[f"{unit_name}_per_day"] = float(totals.mean())
    return out


def patient_indices(grid: pd.DataFrame, insulin: pd.DataFrame, carbs: pd.DataFrame, wearables: pd.DataFrame,
                    sleep: pd.DataFrame, days: list, settings: dict, uses_insulin: bool = True) -> dict:
    """Every index of one patient on ``days`` (the usable days of the startup window)."""
    days = sorted(pd.DatetimeIndex(days))
    out = {"window_days_used": len(days)}
    if not days:
        return out
    on_days = grid[grid["day"].isin(days)]
    g = on_days["glucose"].to_numpy(float)
    g = g[np.isfinite(g)]
    if len(g):
        out.update(tir_pct=float(100 * np.mean((g >= 70) & (g <= 180))), glucose_cv_pct=float(100 * g.std() / g.mean()),
                   glucose_mean=float(g.mean()))
    profiles = hourly_by_day(on_days["ts"], on_days["glucose"].to_numpy(float), days, how="mean")
    out["glucose_profile_stability"] = profile_stability(profiles, smooth_bins=1)

    def window(frame: pd.DataFrame, column: str = "ts") -> pd.DataFrame:
        return frame[frame[column].dt.normalize().isin(days)] if not frame.empty else frame

    merge = float(settings["merge_min"])
    if not carbs.empty:
        c = window(carbs)
        c = c[c["carbs_g"] >= float(settings["min_meal_g"])]
        meals = merge_events(c["ts"], c["carbs_g"].to_numpy(float), merge)
        if len(meals):
            out.update(event_indices(meals, days, "meal", "carbs_g"))
    if uses_insulin and not insulin.empty:
        b = window(insulin)
        b = b[b["kind"].isin(["bolus", "bolus_extended"])]
        b = b[b["units"] >= float(settings["min_bolus_u"])]
        boluses = merge_events(b["ts"], b["units"].to_numpy(float), merge)
        if len(boluses):
            out.update(event_indices(boluses, days, "bolus", "bolus_units"))
    if not wearables.empty and wearables["steps"].notna().any():
        w = window(wearables.dropna(subset=["steps"]))
        steps = w.groupby(w["ts"].dt.normalize())["steps"].sum()
        if len(steps) >= 7:
            out["steps_per_day_cv"] = daily_cv(steps.to_numpy(float))
    if not sleep.empty:
        asleep = sleep[~sleep["stage"].astype(str).str.lower().isin(["wake", "awake"])].dropna(subset=["start"])
        night = (asleep["start"] - pd.Timedelta(hours=12)).dt.normalize()
        onset = asleep.groupby(night)["start"].min()
        onset = onset[(onset.index + pd.Timedelta(days=1)).isin(days) | onset.index.isin(days)]
        if len(onset) >= 5:
            out["sleep_onset_sd_min"] = circular_sd_minutes(clock_minutes(onset.reset_index(drop=True)))
    return out
