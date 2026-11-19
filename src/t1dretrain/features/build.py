"""One patient's model inputs (script 05): the grid table and the checks.

Output: ``<derived>/features/<dataset>/grid/<patient_id>.parquet``, one row per
slot of the patient's native grid (``features.grid``), with:

- glucose: ``glucose`` (reading or NaN), ``observed``, ``target_ok``,
  ``glucose_input``, ``interpolated``, ``at_limit``, ``spike``,
  ``snap_offset_s``, ``segment``, ``clock_break``, ``source``;
- insulin (D54): ``iob``, ``iob_bolus``, ``iob_basal`` (U) and
  ``insulin_activity`` (U/h), from all delivered insulin;
- carbohydrates (D54): ``cob`` (g) and ``carb_absorption`` (g/h);
- calendar: ``hour_sin``, ``hour_cos``; ``dow`` (0 = Monday) only where the
  dates are real (D24);
- wearables: ``hr_15min`` (mean bpm), ``steps_15min``, ``kcal_15min``,
  ``steps_60min`` (records that ended in the last 15 or 60 min) and
  ``asleep``; NaN where the dataset or the patient has none;
- day level (script 03): ``day``, ``valid_day``, ``loop``, ``period``,
  ``insulin_known`` (complete insulin on the day of the slot and on the day
  6 h earlier) and ``carbs_known`` (carbohydrates logged on the day of the
  slot and on the day 3 h earlier).

Every input of a slot is evaluated at the time of its reading when there is
one (else at the slot time) and only uses records made at or before it.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.store import ProcessedStore
from . import carbs as carb_model
from .calendar import time_of_day
from .grid import glucose_grid
from .insulin import ExponentialInsulin, delivery_per_minute, on_board

RESPONSE_OFFSETS_MIN = np.arange(-60, 241, 15)
IOB_WINDOW_H = 6
COB_WINDOW_H = 3
FLOAT_COLUMNS = ["glucose", "glucose_input", "snap_offset_s", "iob", "iob_bolus", "iob_basal", "insulin_activity",
                 "cob", "carb_absorption", "hour_sin", "hour_cos", "dow", "hr_15min", "steps_15min", "kcal_15min",
                 "steps_60min", "asleep"]


@dataclass
class FeatureResult:
    dataset: str
    patient_id: str
    row: dict
    seconds: float
    hourly: pd.DataFrame = field(default_factory=pd.DataFrame)
    responses: pd.DataFrame = field(default_factory=pd.DataFrame)
    example: dict | None = None
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    error: str = ""


def _window_stats(event_ns: np.ndarray, values: np.ndarray, at_ns: np.ndarray, window_ns: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of the finite ``values`` whose time is in (t - window, t] for each t of ``at_ns``."""
    ok = np.isfinite(values)
    times, vals = event_ns[ok], values[ok]
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    hi = np.searchsorted(times, at_ns, "right")
    lo = np.searchsorted(times, at_ns - window_ns, "right")
    return csum[hi] - csum[lo], (hi - lo).astype(float)


def wearable_columns(wear: pd.DataFrame, at: pd.Series) -> dict[str, np.ndarray]:
    nan = np.full(len(at), np.nan)
    if wear.empty:
        return {"hr_15min": nan, "steps_15min": nan.copy(), "kcal_15min": nan.copy(), "steps_60min": nan.copy()}
    wear = wear.sort_values("ts")
    event = wear["ts"].to_numpy("datetime64[ns]").astype(np.int64)
    at_ns = at.to_numpy("datetime64[ns]").astype(np.int64)
    out = {}
    for name, column, window, how in [("hr_15min", "hr_bpm", 15, "mean"), ("steps_15min", "steps", 15, "sum"),
                                      ("kcal_15min", "kcal", 15, "sum"), ("steps_60min", "steps", 60, "sum")]:
        total, count = _window_stats(event, wear[column].to_numpy(float), at_ns, window * 60e9)
        value = total / np.where(count > 0, count, 1) if how == "mean" else total
        out[name] = np.where(count > 0, value, np.nan)
    return out


def asleep_column(sleep: pd.DataFrame, at: pd.Series) -> np.ndarray:
    if sleep.empty:
        return np.full(len(at), np.nan)
    asleep = sleep[~sleep["stage"].astype(str).str.lower().isin(["wake", "awake"])].dropna(subset=["start", "end"])
    if asleep.empty:
        return np.zeros(len(at))
    start = asleep["start"].to_numpy("datetime64[ns]").astype(np.int64)
    end = asleep["end"].to_numpy("datetime64[ns]").astype(np.int64)
    order = np.argsort(start)
    start, end = start[order], np.maximum.accumulate(end[order])
    t = at.to_numpy("datetime64[ns]").astype(np.int64)
    k = np.searchsorted(start, t, "right") - 1
    inside = (k >= 0) & (t < end[np.maximum(k, 0)])
    out = inside.astype(float)
    first, last = start.min(), end.max()
    out[(t < first - 86_400e9) | (t > last + 86_400e9)] = np.nan  # no sleep record within a day: unknown
    return out


INSULIN_CARB_COLUMNS = ("iob_bolus", "iob_basal", "iob", "insulin_activity", "cob", "carb_absorption")


def insulin_carb_inputs(at: pd.Series, first_ts: pd.Timestamp, insulin: pd.DataFrame, carbs: pd.DataFrame,
                        settings: dict) -> tuple[dict, tuple]:
    """IOB, insulin activity, COB and carbohydrate absorption at the evaluation times ``at`` (D54), with the
    insulin action curve (``insulin_action_min``, ``insulin_peak_min``, ``insulin_delay_min``) and the carbohydrate
    absorption time (``carb_absorption_min``) of ``settings``. Script 05 and stage 3b (other curves, D89) use it.
    Returns the columns and (start of the minute axis, bolus and basal units per minute)."""
    start = (first_ts - pd.Timedelta(days=2)).floor("min")
    n_minutes = int((at.max() - start).total_seconds() // 60) + 2
    minute = np.floor((at - start).dt.total_seconds().to_numpy() / 60).astype(np.int64)
    model = ExponentialInsulin(float(settings["insulin_action_min"]), float(settings["insulin_peak_min"]),
                               float(settings.get("insulin_delay_min", 0.0)))
    bolus, basal = delivery_per_minute(insulin, start, n_minutes, float(settings["long_acting_spread_h"]))
    iob_bolus, act_bolus = on_board(bolus, model)
    iob_basal, act_basal = on_board(basal, model)
    columns = {"iob_bolus": iob_bolus[minute], "iob_basal": iob_basal[minute]}
    columns["iob"] = columns["iob_bolus"] + columns["iob_basal"]
    columns["insulin_activity"] = (act_bolus + act_basal)[minute] * 60.0
    grams = carb_model.carbs_per_minute(carbs, start, n_minutes)
    cob, absorption = carb_model.on_board(grams, float(settings["carb_absorption_min"]))
    columns["cob"] = cob[minute]
    columns["carb_absorption"] = absorption[minute] * 60.0
    return columns, (start, bolus, basal)


def evaluation_times(grid: pd.DataFrame) -> pd.Series:
    """The time at which every input of a saved grid was evaluated: the reading time where there is a reading
    (slot time + ``snap_offset_s``), else the slot time (as in ``build_patient``). The offset is saved as float32
    (about 30 microseconds of error over a few minutes), so the reading time is rounded to the millisecond: readings
    are time-stamped to the second or the millisecond, and a reading on a minute boundary must not fall into the
    minute before (it would leave out a dose or a meal given at that minute)."""
    if "snap_offset_s" not in grid or "observed" not in grid:
        return grid["ts"]
    offset = pd.to_timedelta(np.round(grid["snap_offset_s"].astype(float).fillna(0.0) * 1e6), unit="us")
    reading = (grid["ts"] + offset).dt.round("ms")
    return reading.where(grid["observed"].astype(bool), grid["ts"])


def build_patient(job: dict) -> FeatureResult:
    """Script 05 for one patient (in a worker process)."""
    started = time.time()
    dataset, patient_id = job["dataset"], job["patient_id"]
    settings = job["settings"]
    try:
        clean_store = ProcessedStore(dataset, root=job["derived_root"], stage="clean")
        source = ProcessedStore(dataset, root=job["derived_root"])
        clean = clean_store.read_frame("cgm", patient_id)
        if clean.empty:
            return FeatureResult(dataset, patient_id, {"dataset": dataset, "patient_id": patient_id}, 0.0,
                                 error="no clean CGM")
        grid, info = glucose_grid(clean, float(settings["max_interp_gap_min"]))
        at = grid["reading_ts"].where(grid["observed"], grid["ts"])  # evaluation time of every input

        insulin = source.read("insulin", patient_id)
        carbs = source.read("carbs", patient_id)
        columns, (start, bolus, basal) = insulin_carb_inputs(at, grid["ts"].min(), insulin, carbs, settings)
        for name, values in columns.items():
            grid[name] = values

        grid["hour_sin"], grid["hour_cos"] = time_of_day(grid["ts"])
        grid["dow"] = grid["ts"].dt.dayofweek.astype(float) if job["real_dates"] else np.nan
        for name, values in wearable_columns(source.read("wearables", patient_id), at).items():
            grid[name] = values
        grid["asleep"] = asleep_column(source.read("sleep", patient_id), at)

        days = job["days"].set_index("day")
        grid["day"] = grid["ts"].dt.normalize()
        grid["valid_day"] = grid["day"].map(days["valid"]).eq(True)
        grid["loop"] = grid["day"].map(days["loop"]).astype("string")
        grid["period"] = grid["day"].map(days["period"]).astype("string")
        complete = days["insulin_complete"].fillna(False).astype(bool)
        earlier = (grid["ts"] - pd.Timedelta(hours=IOB_WINDOW_H)).dt.normalize()
        grid["insulin_known"] = grid["day"].map(complete).eq(True) & earlier.map(complete).eq(True)
        logged = days["carbs_n"].fillna(0) > 0
        earlier = (grid["ts"] - pd.Timedelta(hours=COB_WINDOW_H)).dt.normalize()
        grid["carbs_known"] = grid["day"].map(logged).eq(True) & earlier.map(logged).eq(True)

        for column in FLOAT_COLUMNS:
            grid[column] = grid[column].astype("float32")
        grid.insert(0, "patient_id", patient_id)
        out = ProcessedStore(dataset, root=job["derived_root"], stage="features")
        out.write_frame("grid", patient_id, grid.drop(columns=["reading_ts"]))

        delivered = pd.Series(bolus + basal, index=start + pd.to_timedelta(np.arange(len(bolus)), unit="min"))
        row = describe(grid, info, delivered, bolus.sum(), carbs, days)
        row.update({"dataset": dataset, "patient_id": patient_id})
        return FeatureResult(dataset, patient_id, row, round(time.time() - started, 2),
                             hourly=hourly_profile(grid), responses=responses(grid, insulin, carbs, info["step_min"],
                                                                              float(settings["meal_check_min_carbs_g"])),
                             example=example_day(grid, insulin, carbs) if job.get("example") else None,
                             offsets=grid["snap_offset_s"].dropna().to_numpy()[:20000])
    except Exception:  # noqa: BLE001
        return FeatureResult(dataset, patient_id, {"dataset": dataset, "patient_id": patient_id},
                             round(time.time() - started, 2), error=traceback.format_exc(limit=6))


# ----------------------------------------------------------------------------- checks


def describe(grid: pd.DataFrame, info: dict, delivered: pd.Series, bolus_total: float, carbs: pd.DataFrame,
             days: pd.DataFrame) -> dict:
    valid = grid["valid_day"]
    known = grid["insulin_known"] & valid
    complete_days = days.index[days["insulin_complete"].fillna(False).astype(bool) & days["valid"].fillna(False)]
    daily = delivered.groupby(delivered.index.normalize()).sum()
    daily = daily[daily.index.isin(complete_days)]
    carbs_daily = carbs.groupby(carbs["ts"].dt.normalize())["carbs_g"].sum() if not carbs.empty else pd.Series(dtype=float)
    carbs_daily = carbs_daily[carbs_daily.index.isin(days.index[days["valid"].fillna(False)])]
    offsets = grid["snap_offset_s"].abs().dropna()
    return {
        "step_min": info["step_min"],
        "slots": int(len(grid)),
        "observed_share": round(float(grid["observed"].mean()), 4),
        "target_ok_share": round(float(grid["target_ok"].mean()), 4),
        "interpolated_share": round(float(grid["interpolated"].mean()), 4),
        "input_share": round(float(grid["glucose_input"].notna().mean()), 4),
        "collisions_dropped": info["collisions_dropped"],
        "segments": int(grid["segment"].max()) + 1,
        "snap_offset_abs_median_s": round(float(offsets.median()), 1) if len(offsets) else np.nan,
        "snap_offset_abs_p95_s": round(float(offsets.quantile(0.95)), 1) if len(offsets) else np.nan,
        "valid_day_slot_share": round(float(valid.mean()), 4),
        "insulin_known_share": round(float(known.sum() / max(valid.sum(), 1)), 4),
        "carbs_known_share": round(float((grid["carbs_known"] & valid).sum() / max(valid.sum(), 1)), 4),
        "tdi_median_u": round(float(daily.median()), 2) if len(daily) else np.nan,
        "bolus_share": round(float(bolus_total / delivered.sum()), 3) if delivered.sum() > 0 else np.nan,
        "carbs_median_g_per_day": round(float(carbs_daily.median()), 1) if len(carbs_daily) else np.nan,
        "iob_mean_u": round(float(grid.loc[known, "iob"].mean()), 2) if known.any() else np.nan,
        "cob_mean_g": round(float(grid.loc[grid["carbs_known"] & valid, "cob"].mean()), 1)
        if (grid["carbs_known"] & valid).any() else np.nan,
        "hr_share": round(float(grid["hr_15min"].notna().mean()), 4),
        "steps_share": round(float(grid["steps_15min"].notna().mean()), 4),
        "asleep_share": round(float(grid["asleep"].mean()), 4) if grid["asleep"].notna().any() else np.nan,
    }


def hourly_profile(grid: pd.DataFrame) -> pd.DataFrame:
    """Sums and counts by hour of day of IOB, COB and glucose, on valid days (pooled later per dataset)."""
    g = grid[grid["valid_day"]]
    hour = g["ts"].dt.hour
    rows = []
    for name, mask in [("iob", g["insulin_known"]), ("cob", g["carbs_known"]), ("glucose", g["observed"])]:
        values = g.loc[mask, name]
        grouped = values.groupby(hour[mask]).agg(["sum", "count"]).reindex(range(24), fill_value=0)
        rows.append(pd.DataFrame({"variable": name, "hour": range(24), "sum": grouped["sum"].to_numpy(),
                                  "count": grouped["count"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def responses(grid: pd.DataFrame, insulin: pd.DataFrame, carbs: pd.DataFrame, step_min: float,
              min_carbs_g: float) -> pd.DataFrame:
    """Glucose change around meals and correction boluses (sums and counts by minutes from the event)."""
    # positions on the grid by absolute slot (days without readings are not in the table)
    slot0 = int(grid["slot"].iloc[0])
    origin = grid["ts"].iloc[0].value - slot0 * step_min * 60e9
    values = pd.Series(grid["glucose_input"].to_numpy(float), index=grid["slot"].to_numpy())
    valid = pd.Series(grid["valid_day"].to_numpy(), index=grid["slot"].to_numpy())
    step_ns = step_min * 60e9
    offsets = RESPONSE_OFFSETS_MIN[(RESPONSE_OFFSETS_MIN % step_min) == 0]
    events = {}
    if not carbs.empty:
        meals = carbs[carbs["carbs_g"] >= min_carbs_g]["ts"]
        events["meal"] = meals.to_numpy("datetime64[ns]").astype(np.int64)
    if not insulin.empty and not carbs.empty:  # without carbohydrate records a correction cannot be told apart
        boluses = insulin[(insulin["kind"] == "bolus") & (insulin["units"] >= 1)]["ts"]
        b = boluses.to_numpy("datetime64[ns]").astype(np.int64)
        c = np.sort(carbs["ts"].to_numpy("datetime64[ns]").astype(np.int64))
        near = np.searchsorted(c, b + 3600e9, "right") - np.searchsorted(c, b - 3600e9, "left")
        events["correction_bolus"] = b[near == 0]  # no carbohydrates within 1 h either side
    rows = []
    for kind, times in events.items():
        if len(times) == 0:
            continue
        slot = np.ceil((times - origin) / step_ns).astype(np.int64)
        base = values.reindex(slot).to_numpy()
        keep = valid.reindex(slot).eq(True).to_numpy() & np.isfinite(base)
        slot, base = slot[keep], base[keep]
        for minutes in offsets:
            delta = values.reindex(slot + int(minutes / step_min)).to_numpy() - base
            delta = delta[np.isfinite(delta)]
            rows.append({"event": kind, "minutes": int(minutes), "sum": float(delta.sum()), "count": int(delta.size)})
    return pd.DataFrame(rows)


def example_day(grid: pd.DataFrame, insulin: pd.DataFrame, carbs: pd.DataFrame) -> dict | None:
    """The valid day with complete insulin and the most carbohydrate entries, for the visual check."""
    candidates = pd.DatetimeIndex(grid.loc[grid["valid_day"] & grid["insulin_known"], "day"].unique())
    if len(candidates) == 0:
        candidates = pd.DatetimeIndex(grid.loc[grid["valid_day"], "day"].unique())
    if len(candidates) == 0:
        return None
    counts = carbs.groupby(carbs["ts"].dt.normalize()).size() if not carbs.empty else pd.Series(dtype=int)
    counts = counts.reindex(candidates, fill_value=0)
    day = counts.idxmax() if counts.max() > 0 else candidates[len(candidates) // 2]
    end = day + pd.Timedelta(days=1)
    keep = ["ts", "glucose", "glucose_input", "interpolated", "iob", "cob", "insulin_activity", "carb_absorption"]
    frame = grid.loc[(grid["ts"] >= day) & (grid["ts"] < end), keep].reset_index(drop=True)
    boluses = insulin[(insulin["kind"].isin(["bolus", "bolus_extended"])) & (insulin["ts"] >= day) & (insulin["ts"] < end)]
    meals = carbs[(carbs["ts"] >= day) & (carbs["ts"] < end)] if not carbs.empty else carbs
    return {"day": day, "frame": frame, "boluses": boluses[["ts", "units"]].reset_index(drop=True),
            "carbs": meals[["ts", "carbs_g"]].reset_index(drop=True) if not meals.empty else pd.DataFrame()}
