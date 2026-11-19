"""Native-grid glucose for the models (script 05, D11, D55).

The clean series of script 03 keeps every reading at its own time. Models
need a regular grid, so each patient gets one:

- **Step**: the finest nominal interval that holds at least 10 % of the
  patient's readings (5 min, or 15 min for FreeStyle Libre history only).
  A patient who changes from Libre to Dexcom (HUPA0027P) gets a 5-min grid,
  so no reading is lost; the Libre days then have one reading every third
  slot and their inputs are interpolated as any other short gap.
- **Phase**: the slots are placed at origin + k x step, with the origin at
  the patient's typical position of readings inside the step (circular mean
  of the reading times modulo the step). Readings then sit near the middle of
  their slots, and the offset between a reading and its slot stays small.
- **Snapping**: each reading goes to the nearest slot (``snap_offset_s`` =
  reading time - slot time). When two readings fall in the same slot, the
  closer one keeps it and the other moves to the free neighbouring slot on
  its side if it is within one step; otherwise it is dropped (counted).
- **Segments**: a new segment starts at every ``clock_break`` of script 03
  (the clock jumps there); no input window or interpolation crosses it.
- **Days without readings** are left out of the table (``slot`` keeps the
  absolute position on the grid, so windows are found by slot arithmetic).
- **Inputs**: ``glucose_input`` is the reading, or a linear interpolation
  across a gap of at most ``max_missing_slots`` empty slots inside a segment
  (``interpolated``). Interpolation uses the reading that closes the gap, so
  forecast origins must be slots with a real reading: then every gap inside
  the input window was closed before the origin.
- **Targets**: ``target_ok`` marks slots with a real reading that is not a
  ``spike`` (an isolated jump of script 03). Spikes are not used as inputs
  either; readings at the sensor limits are kept and flagged ``at_limit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def grid_step(clean: pd.DataFrame, min_share: float = 0.1) -> float:
    """Finest nominal interval (min) that holds at least ``min_share`` of the patient's readings."""
    shares = clean["interval_min"].value_counts(normalize=True)
    return float(shares[shares >= min_share].index.min())


def grid_origin(times: pd.Series, step_min: float) -> pd.Timestamp:
    """Midnight of the first day plus the circular mean phase of the readings within the step."""
    step_ns = step_min * 60e9
    ns = times.to_numpy("datetime64[ns]").astype(np.int64)
    angle = 2 * np.pi * (ns % step_ns) / step_ns
    phase = (np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) % (2 * np.pi)) / (2 * np.pi) * step_ns
    return times.min().normalize() + pd.Timedelta(int(phase), unit="ns") - pd.Timedelta(minutes=step_min)


def snap(clean: pd.DataFrame, origin: pd.Timestamp, step_min: float) -> tuple[pd.DataFrame, int]:
    """Readings with their slot index; returns (readings kept, readings dropped by collisions)."""
    step_ns = step_min * 60e9
    ns = (clean["ts"] - origin).dt.total_seconds().to_numpy(float) * 1e9
    slot = np.round(ns / step_ns).astype(np.int64)
    frame = clean.assign(slot=slot, offset_ns=ns - slot * step_ns).reset_index(drop=True)
    frame["distance"] = frame["offset_ns"].abs()
    frame = frame.sort_values(["slot", "distance"], kind="stable")
    first = ~frame["slot"].duplicated()
    taken = set(frame.loc[first, "slot"].tolist())
    moved = []
    for index, row in frame[~first].iterrows():
        target = row["slot"] + (1 if row["offset_ns"] > 0 else -1)
        if target not in taken and abs(row["offset_ns"] - (target - row["slot"]) * step_ns) <= step_ns:
            taken.add(target)
            moved.append((index, target))
    keep = first.copy()
    for index, target in moved:
        frame.loc[index, "offset_ns"] -= (target - frame.loc[index, "slot"]) * step_ns
        frame.loc[index, "slot"] = target
        keep.loc[index] = True
    dropped = int((~keep).sum())
    kept = frame[keep].sort_values("slot").drop(columns="distance").reset_index(drop=True)
    return kept, dropped


def interpolate(values: np.ndarray, segment: np.ndarray, max_missing_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of NaN runs of at most ``max_missing_slots`` between two values of one segment."""
    out = values.copy()
    filled = np.zeros(len(values), dtype=bool)
    known = np.flatnonzero(np.isfinite(values))
    if len(known) < 2 or max_missing_slots < 1:
        return out, filled
    left, right = known[:-1], known[1:]
    gap = right - left - 1
    ok = (gap >= 1) & (gap <= max_missing_slots) & (segment[left] == segment[right])
    for a, b in zip(left[ok], right[ok]):
        k = np.arange(a + 1, b)
        out[k] = values[a] + (values[b] - values[a]) * (k - a) / (b - a)
        filled[k] = True
    return out, filled


def glucose_grid(clean: pd.DataFrame, max_interp_gap_min: float) -> tuple[pd.DataFrame, dict]:
    """Regular slots from the first to the last reading with the glucose columns (see module docstring)."""
    step = grid_step(clean)
    origin = grid_origin(clean["ts"], step)
    kept, dropped = snap(clean, origin, step)
    n = int(kept["slot"].max()) + 1
    slots = np.arange(int(kept["slot"].min()), n)
    frame = pd.DataFrame({"slot": slots})
    frame["ts"] = origin + pd.to_timedelta(slots * step, unit="min")
    frame = frame.merge(kept[["slot", "ts", "glucose_mgdl", "flag", "clock_break", "source", "offset_ns"]]
                        .rename(columns={"ts": "reading_ts"}), on="slot", how="left")
    frame["observed"] = frame["glucose_mgdl"].notna()
    frame["spike"] = frame["flag"].eq("spike").fillna(False).astype(bool)
    frame["at_limit"] = frame["flag"].eq("at_limit").fillna(False).astype(bool)
    frame["clock_break"] = frame["clock_break"].fillna(False).astype(bool)
    frame["segment"] = frame["clock_break"].cumsum().astype(np.int32)
    frame["target_ok"] = frame["observed"] & ~frame["spike"]
    usable = np.where(frame["target_ok"], frame["glucose_mgdl"], np.nan)
    missing = max(1, int(round(max_interp_gap_min / step)) - 1)
    frame["glucose_input"], frame["interpolated"] = interpolate(usable, frame["segment"].to_numpy(), missing)
    frame["snap_offset_s"] = frame["offset_ns"] / 1e9
    frame = frame.drop(columns=["offset_ns", "flag"]).rename(columns={"glucose_mgdl": "glucose"})
    day = frame["ts"].dt.normalize()
    frame = frame[day.isin(day[frame["observed"]])].reset_index(drop=True)  # drop days without any reading
    info = {"step_min": step, "origin": origin, "collisions_dropped": dropped, "max_missing_slots": missing}
    return frame, info
