"""Per-patient description of the harmonised tables (script 02 and later checks).

Everything here is descriptive: no row is removed or altered. Statistics on
glucose, sampling and valid days use, for each calendar day, the CGM source
with the best coverage that day ("daily best source"). Two overlapping
streams (Loop: HealthKit and direct Dexcom uploads; HUPA0001P: Libre and
Enlite) then do not look like a 2.5-min sensor, and a patient who changes
sensor (HUPA0027P, BrisT1D) keeps the days of both devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GLUCOSE_BINS = np.arange(20, 610, 10)  # mg/dL bin edges for the glucose histogram
INTERVAL_EDGES = [0, 1, 2, 3, 4, 4.5, 5.5, 6, 8, 10, 14, 16, 20, 30, 60, 120, np.inf]  # minutes
INTERVAL_LABELS = ["<1", "1-2", "2-3", "3-4", "4-4.5", "4.5-5.5", "5.5-6", "6-8", "8-10", "10-14", "14-16",
                   "16-20", "20-30", "30-60", "60-120", ">=120"]
STAGES_ASLEEP = {"light", "deep", "rem", "asleep", "restless", "sleep", "unmeasurable"}


@dataclass
class PatientDescription:
    row: dict
    glucose_hist: np.ndarray
    interval_hist: np.ndarray
    hour_glucose_sum: np.ndarray = field(default_factory=lambda: np.zeros(24))
    hour_glucose_n: np.ndarray = field(default_factory=lambda: np.zeros(24))
    hour_carbs_n: np.ndarray = field(default_factory=lambda: np.zeros(24))
    hour_bolus_n: np.ndarray = field(default_factory=lambda: np.zeros(24))
    hour_hr_sum: np.ndarray = field(default_factory=lambda: np.zeros(24))
    hour_hr_n: np.ndarray = field(default_factory=lambda: np.zeros(24))
    hour_steps_sum: np.ndarray = field(default_factory=lambda: np.zeros(24))
    hour_asleep_min: np.ndarray = field(default_factory=lambda: np.zeros(24))
    sleep_nights: float = 0.0


def daily_best_source(cgm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Automatic readings of the best-covered source of each day.

    Returns the chosen readings, one row per day (day, source, n, coverage)
    and the modal interval of each source.
    """
    auto = cgm.loc[cgm["record_type"] == "cgm", ["ts", "glucose_mgdl", "source"]].copy()
    if auto.empty:
        return auto, pd.DataFrame(columns=["day", "source", "n", "coverage"]), {}
    auto["source"] = auto["source"].fillna("unknown").astype(str)
    auto = auto.sort_values("ts")
    intervals = {src: modal_interval(group["ts"]) for src, group in auto.groupby("source")}
    auto["day"] = auto["ts"].dt.normalize()
    days = auto.groupby(["day", "source"]).size().rename("n").reset_index()
    expected = 1440.0 / days["source"].map(intervals).astype(float)
    days["coverage"] = (days["n"] / expected).clip(upper=1.0).fillna(0.0)
    best = days.sort_values(["day", "coverage", "n"], ascending=[True, False, False]).drop_duplicates("day")
    chosen = auto.merge(best[["day", "source"]], on=["day", "source"]).sort_values("ts")
    return chosen, best.reset_index(drop=True), intervals


def modal_interval(ts: pd.Series) -> float:
    """Most frequent spacing (minutes, rounded) between consecutive readings, among gaps up to 30 min."""
    diffs = ts.diff().dt.total_seconds().div(60).dropna()
    diffs = diffs[(diffs > 0.5) & (diffs <= 30)]
    if diffs.empty:
        return float("nan")
    return float(diffs.round().mode().iloc[0])


def longest_run(flags: pd.Series) -> int:
    """Longest run of consecutive True values in a daily boolean series (index = consecutive dates)."""
    best = current = 0
    for value in flags:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def describe_patient(dataset: str, patient_id: str, tables: dict[str, pd.DataFrame],
                     coverage_threshold: float, window: tuple[str, str] | None = None) -> PatientDescription:
    """Summary row plus histograms for one patient.

    ``window`` (start, end) is the dataset's own study window when it has one
    (HUPA-UCM: the span of the preprocessed file); readings outside it are
    counted separately because raw device exports can include older history.
    """
    cgm = tables["cgm"]
    row: dict = {"dataset": dataset, "patient_id": patient_id}
    main, best, intervals = daily_best_source(cgm)
    used = best["source"].value_counts()
    row["cgm_sources"] = "+".join(sorted(cgm["source"].dropna().unique())) if not cgm.empty else ""
    row["cgm_main_source"] = used.index[0] if len(used) else ""
    row["cgm_sources_days"] = "+".join(f"{src}:{n}" for src, n in used.items())
    row["cgm_source_switches"] = int((best["source"] != best["source"].shift()).sum() - 1) if len(best) else 0
    row["cgm_rows"] = int(len(cgm))
    row["cgm_main_rows"] = int(len(main))
    row["scan_rows"] = int((cgm["record_type"] == "scan").sum()) if not cgm.empty else 0
    glucose_hist = np.zeros(len(GLUCOSE_BINS) - 1)
    interval_hist = np.zeros(len(INTERVAL_LABELS))
    description = PatientDescription(row, glucose_hist, interval_hist)

    calendar = pd.DatetimeIndex([])
    if not main.empty:
        ts = main["ts"]
        g = main["glucose_mgdl"].astype(float)
        first, last = ts.iloc[0], ts.iloc[-1]
        interval = modal_interval(ts)
        diffs = ts.diff().dt.total_seconds().div(60).dropna()
        all_auto = cgm.loc[cgm["record_type"] == "cgm", "ts"].sort_values()
        all_diffs = all_auto.diff().dt.total_seconds().div(60).dropna()
        calendar = pd.date_range(first.normalize(), last.normalize(), freq="D")
        per_day = best.set_index("day")["n"].reindex(calendar, fill_value=0)
        coverage = best.set_index("day")["coverage"].reindex(calendar, fill_value=0.0)
        valid = coverage >= coverage_threshold
        row.update({
            "first_ts": first, "last_ts": last,
            "span_days": round((last - first).total_seconds() / 86400, 2),
            "days_with_cgm": int((per_day > 0).sum()),
            "modal_interval_min": interval,
            "source_intervals_min": "+".join(f"{src}:{v:g}" for src, v in intervals.items()),
            "readings_per_day_median": float(per_day[per_day > 0].median()),
            "share_intervals_lt_2min": round(float((diffs < 2).mean()), 4) if len(diffs) else np.nan,
            "share_intervals_lt_2min_all_sources": round(float((all_diffs < 2).mean()), 4) if len(all_diffs) else np.nan,
            "valid_days": int(valid.sum()),
            "longest_valid_run_days": longest_run(valid),
            "mean_daily_coverage": round(float(coverage.mean()), 4),
            "longest_gap_h": round(float(diffs.max() / 60), 2) if len(diffs) else np.nan,
            "gaps_over_1h": int((diffs > 60).sum()),
            "glucose_mean": round(float(g.mean()), 1),
            "glucose_sd": round(float(g.std()), 1),
            "glucose_cv": round(float(g.std() / g.mean()), 3),
            "tir_70_180": round(float(g.between(70, 180).mean()), 4),
            "tbr_lt70": round(float((g < 70).mean()), 4),
            "tbr_lt54": round(float((g < 54).mean()), 4),
            "tar_gt180": round(float((g > 180).mean()), 4),
            "tar_gt250": round(float((g > 250).mean()), 4),
            "glucose_min": float(g.min()), "glucose_max": float(g.max()),
        })
        if window is not None:
            start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
            inside = valid[(valid.index >= start.normalize()) & (valid.index <= end.normalize())]
            row["valid_days_in_window"] = int(inside.sum())
            row["cgm_main_rows_outside_window"] = int(((ts < start) | (ts > end)).sum())
        description.glucose_hist = np.histogram(g.clip(20, 599.9), bins=GLUCOSE_BINS)[0].astype(float)
        description.interval_hist = np.histogram(diffs, bins=INTERVAL_EDGES)[0].astype(float)
        hours = ts.dt.hour.to_numpy()
        description.hour_glucose_sum = np.bincount(hours, weights=g.to_numpy(), minlength=24)
        description.hour_glucose_n = np.bincount(hours, minlength=24).astype(float)
    row["calendar_days"] = int(len(calendar))
    insulin_days = tables["insulin"]["ts"].dt.normalize().unique() if not tables["insulin"].empty else []
    cgm_days = main["ts"].dt.normalize().unique() if not main.empty else []
    observed = len(set(insulin_days) | set(cgm_days))  # days with CGM or insulin records
    row["observed_days"] = int(observed)

    insulin = tables["insulin"]
    row["insulin_rows"] = int(len(insulin))
    for kind in ["bolus", "bolus_extended", "basal_rate", "basal_injection", "basal_5min", "total_5min"]:
        row[f"insulin_{kind}_rows"] = int((insulin["kind"] == kind).sum()) if not insulin.empty else 0
    if not insulin.empty:
        ins = insulin.copy()
        ins["day"] = ins["ts"].dt.normalize()
        delivered = np.where(
            ins["kind"] == "basal_rate",
            ins["rate_u_per_h"].astype(float) * ins["duration_min"].astype(float) / 60.0,
            ins["units"].astype(float),
        )
        ins["delivered"] = delivered
        is_bolus = ins["kind"].isin(["bolus", "bolus_extended"])
        is_basal = ins["kind"].isin(["basal_rate", "basal_injection", "basal_5min"])
        is_total = ins["kind"] == "total_5min"
        days = ins["day"].nunique()
        row["insulin_days"] = int(days)
        ins["group"] = np.select([is_bolus, is_basal, is_total], ["bolus_u", "basal_u", "total_u"], default="other")
        daily = ins.groupby(["day", "group"])["delivered"].sum(min_count=1).unstack()
        for group, key in [("bolus_u", "bolus_units_per_day_median"), ("basal_u", "basal_units_per_day_median"),
                           ("total_u", "total5_units_per_day_median")]:
            row[key] = round(float(daily[group].median()), 2) if group in daily else np.nan
        row["bolus_events_per_day"] = round(float(is_bolus.sum() / days), 2) if is_bolus.any() else np.nan
        rates = ins.loc[ins["kind"] == "basal_rate"]
        row["basal_rate_missing_duration_share"] = round(float(rates["duration_min"].isna().mean()), 4) if len(rates) else np.nan
        row["basal_rate_over_24h_share"] = round(float((rates["duration_min"] > 1440).mean()), 4) if len(rates) else np.nan
        bolus_hours = ins.loc[ins["kind"] == "bolus", "ts"].dt.hour.to_numpy()
        description.hour_bolus_n = np.bincount(bolus_hours, minlength=24).astype(float)

    carbs = tables["carbs"]
    row["carb_rows"] = int(len(carbs))
    if not carbs.empty:
        per_day = carbs.groupby(carbs["ts"].dt.normalize())["carbs_g"].agg(["sum", "size"])
        row["carb_days"] = int(len(per_day))
        row["carb_events_per_day"] = round(float(len(carbs) / observed), 2) if observed else np.nan
        row["carbs_g_per_carb_day_median"] = round(float(per_day["sum"].median()), 1)
        row["carbs_g_per_event_median"] = round(float(carbs["carbs_g"].median()), 1)
        description.hour_carbs_n = np.bincount(carbs["ts"].dt.hour.to_numpy(), minlength=24).astype(float)

    wear = tables["wearables"]
    row["wearable_rows"] = int(len(wear))
    if not wear.empty:
        row["hr_days"] = int(wear.loc[wear["hr_bpm"].notna(), "ts"].dt.normalize().nunique())
        row["steps_days"] = int(wear.loc[wear["steps"].notna(), "ts"].dt.normalize().nunique())
        steps = wear.loc[wear["steps"].notna()].groupby(wear["ts"].dt.normalize())["steps"].sum()
        row["steps_per_day_median"] = round(float(steps.median()), 0) if len(steps) else np.nan
        row["hr_median"] = round(float(wear["hr_bpm"].median()), 1) if wear["hr_bpm"].notna().any() else np.nan
        row["wearable_intervals_min"] = "+".join(str(int(v)) for v in sorted(wear["interval_min"].dropna().unique()))
        hr_rows = wear[wear["hr_bpm"].notna()]
        hr_hours = hr_rows["ts"].dt.hour.to_numpy()
        description.hour_hr_sum = np.bincount(hr_hours, weights=hr_rows["hr_bpm"].to_numpy(float), minlength=24)
        description.hour_hr_n = np.bincount(hr_hours, minlength=24).astype(float)
        step_rows = wear[wear["steps"].notna()]
        description.hour_steps_sum = np.bincount(step_rows["ts"].dt.hour.to_numpy(),
                                                 weights=step_rows["steps"].to_numpy(float), minlength=24)

    sleep = tables["sleep"]
    row["sleep_rows"] = int(len(sleep))
    if not sleep.empty:
        asleep = sleep[sleep["stage"].str.lower().isin(STAGES_ASLEEP)]
        hours = (asleep["end"] - asleep["start"]).dt.total_seconds() / 3600
        night = (asleep["start"] - pd.Timedelta(hours=12)).dt.normalize()  # a night belongs to the evening's date
        per_night = hours.groupby(night).sum()
        row["sleep_nights"] = int(len(per_night))
        row["sleep_hours_per_night_median"] = round(float(per_night.median()), 2) if len(per_night) else np.nan
        row["sleep_stages"] = "+".join(sorted(sleep["stage"].dropna().str.lower().unique()))[:120]
        description.hour_asleep_min = asleep_minutes_by_hour(asleep)
        description.sleep_nights = float(len(per_night))
        if not wear.empty and wear["hr_bpm"].notna().any():
            contrast, lag, best = hr_sleep_lag(wear, asleep)
            row["hr_asleep_minus_awake"] = contrast
            row["hr_sleep_best_lag_h"] = lag
            row["hr_asleep_minus_awake_at_best_lag"] = best
    return description


def asleep_minutes_by_hour(asleep: pd.DataFrame) -> np.ndarray:
    """Minutes asleep in each hour of the day, summed over all nights."""
    start = asleep["start"].to_numpy("datetime64[m]").astype(np.int64)
    end = asleep["end"].to_numpy("datetime64[m]").astype(np.int64)
    duration = np.clip(end - start, 0, 24 * 60)
    total = int(duration.sum())
    if total == 0 or total > 20_000_000:
        return np.zeros(24)
    offsets = np.arange(total) - np.repeat(np.cumsum(duration) - duration, duration)
    minutes = np.repeat(start, duration) + offsets
    return np.bincount((minutes // 60) % 24, minlength=24).astype(float)


def hr_sleep_lag(wear: pd.DataFrame, asleep: pd.DataFrame) -> tuple[float, float, float]:
    """Heart-rate contrast at lag 0, the lag (h) that makes it most negative, and the contrast there.

    The sleep log is shifted by -12 to +12 h in 30-min steps. When both clocks
    agree the best lag is near 0; a consistent best lag of several hours across
    patients reveals a time-zone or offset error in one of the two series.
    """
    lags = np.arange(-12, 12.5, 0.5)
    contrasts = np.array([hr_sleep_contrast(wear, asleep, pd.Timedelta(hours=float(h))) for h in lags])
    at_zero = float(contrasts[lags == 0][0])
    if np.all(np.isnan(contrasts)):
        return at_zero, float("nan"), float("nan")
    k = int(np.nanargmin(contrasts))
    return at_zero, float(lags[k]), float(contrasts[k])


def hr_sleep_contrast(wear: pd.DataFrame, asleep: pd.DataFrame, shift: pd.Timedelta = pd.Timedelta(0)) -> float:
    """Median heart rate inside sleep-stage intervals minus outside them (bpm).

    A clearly negative value (about -10 bpm or lower) shows that the clocks of
    the watch's heart-rate series and of its sleep log agree; a value near zero
    hints at a time-zone or offset error between the two.
    """
    hr = wear.loc[wear["hr_bpm"].notna(), ["ts", "hr_bpm"]].sort_values("ts")
    intervals = asleep[["start", "end"]].dropna().sort_values("start")
    if hr.empty or intervals.empty:
        return float("nan")
    starts = (intervals["start"] + shift).to_numpy()
    ends = (intervals["end"] + shift).to_numpy()
    times = hr["ts"].to_numpy()
    index = np.searchsorted(starts, times, side="right") - 1
    inside = (index >= 0) & (times < ends[np.clip(index, 0, None)])
    values = hr["hr_bpm"].to_numpy(dtype=float)
    if inside.sum() < 20 or (~inside).sum() < 20:
        return float("nan")
    return round(float(np.median(values[inside]) - np.median(values[~inside])), 1)
