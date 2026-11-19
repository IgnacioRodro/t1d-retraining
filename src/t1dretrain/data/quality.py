"""Data quality (script 03): one clean CGM series per patient, a day table and eligibility.

Input: the harmonised tables of script 02 (``<derived>/processed``). Output:

- ``<derived>/clean/<dataset>/cgm/<patient_id>.parquet``: one CGM series per
  patient: patient_id, ts, glucose_mgdl, source (as harmonised),
  interval_min (nominal spacing of that source), flag and clock_break (True
  on the first reading after a clock-overlap stretch: the clock jumps there,
  so no forecast window may cross it).
- ``<derived>/clean/<dataset>/days.parquet``: one row per patient and
  calendar day, with CGM coverage, validity, loop status, study period,
  insulin and carbohydrate completeness, wearables and sleep.
- ``<derived>/clean/<dataset>/patients.parquet``: the patients table of
  script 02 plus quality figures and eligibility flags.

Rules (decisions D43-D48 in HANDOFF.md; D43 revised in iterations 12 and 13):

1. Only automatic readings (record_type "cgm"); Libre scans stay in the
   processed tables.
2. Each source (as harmonised) is cleaned on its own first (rule 3).
3. The nominal interval of a source is the smallest of 1, 5 or 15 min at
   which >= 60 % of the readings have another reading one interval before or
   after (+-10 %, at least 0.25 min). The grid support of a reading is how many
   of the 6 times +-1..3 intervals away hold a reading (6 inside a regular
   series, about 0 for a scan). Readings closer than 0.6 x the interval to
   the last kept reading of the same source (3 min for 5-min sensors, 9 min
   for 15-min Libre history) collide; the one with more support is kept
   (the earlier on a tie), so Libre scans exported among the history
   readings (T1D-UOM) give way to the history. The dropped reading is
   classified by its value:
   - |difference| <= 5 mg/dL: the same reading twice ("duplicate");
   - otherwise a conflict. Collisions less than 30 min apart (or 2
     intervals, if longer) form a run. A run with three or more conflicts
     and >= 80 % of its readings regular
     (support >= 4), mean that two regular series share the same clock
     times: a device clock set back (e.g. at the end of summer time;
     REPLACE-BG, Loop) or a time-shifted copy of the data (T1D-UOM). The
     readings of that stretch are split into two tracks by continuity of
     timing (each series keeps its own phase on the sensor grid) and of
     value (each track extrapolated from its last two readings); the track
     that continues the series before the stretch is kept and the other is
     dropped ("clock overlap"). Release 0.12.0 dropped both tracks, which
     cost half the data of one T1D-UOM patient. Other conflicts are isolated
     and only the dropped reading goes.
4. Two upload streams of the same sensor (Loop: HealthKit and direct
   Dexcom, D37) are combined day by day: the better covered stream of the day
   is the primary; a reading of the other stream within 0.6 intervals of a
   primary reading is the same reading uploaded twice (dropped); the other
   readings fill the primary's gaps, but only on days when the matched pairs
   agree (median |difference| <= 5 mg/dL, or fewer than 12 pairs to judge);
   on a day when they disagree (shifted clocks) only the primary is kept.
5. One candidate (a source, or a combined stream group) per calendar day:
   the best covered that day. Different sensors are never mixed within a
   day; on the day a sensor is replaced by another, the better covered one is
   kept.
6. Readings are never altered or interpolated here. Two flags are set:
   "at_limit" (sensor reporting limit: <= 40 mg/dL, or >= 400 for Dexcom /
   Medtronic / >= 500 for Libre) and "spike" (rate of change above
   5 mg/dL/min towards and away from one reading, in opposite directions).
7. A day is valid when the chosen source delivers at least
   ``valid_day_min_cgm_coverage`` (config/project.yaml) of its expected
   readings.
8. Insulin is complete on a day when the basal is covered (pump segments
   >= 90 % of the day, 5-min records >= 90 % of slots, or a long-acting
   injection that day or the day before) and at least one bolus is recorded
   (or 5-min total insulin covers the day, BrisT1D).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .store import ProcessedStore

SENSOR_GROUPS = {"dexcom_healthkit": "dexcom_streams", "dexcom_direct": "dexcom_streams"}  # one sensor, two uploads (D37)
DUPLICATE_FRACTION = 0.6
DUPLICATE_TOLERANCE_MGDL = 5.0
OVERLAP_MIN_CONFLICTS = 3
OVERLAP_MIN_ON_GRID = 0.8
OVERLAP_LINK_MIN = 30.0  # collisions closer than this belong to the same run (a track may miss readings)
REGULAR_MIN_SUPPORT = 4  # of 6 neighbouring grid times (+-1..3 intervals) holding a reading
CANDIDATE_INTERVALS_MIN = (1.0, 5.0, 15.0)
ON_GRID_MIN_SHARE = 0.6
STREAM_AGREEMENT_MIN_PAIRS = 12
SPIKE_RATE_MGDL_PER_MIN = 5.0
LOW_LIMIT = 40.0
HIGH_LIMITS = {"freestyle_libre": 500.0, "freestyle_libre_2": 500.0}
DEFAULT_HIGH_LIMIT = 400.0
BASAL_COVERAGE_MIN = 0.9
COVERAGE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
GAP_CLASSES = [  # (label, upper bound in multiples of the nominal interval or minutes)
    ("one missing reading", "x2.5"),
    ("short (<= 3 intervals)", "x3.5"),
    ("<= 1 h", 60),
    ("1-3 h", 180),
    ("3-12 h", 720),
    ("12-24 h", 1440),
    (">= 1 day", np.inf),
]
CLEAN_COLUMNS = ["patient_id", "ts", "glucose_mgdl", "source", "interval_min", "flag", "clock_break"]
OPEN, CLOSED = "open", "closed"


@dataclass
class QualityResult:
    dataset: str
    patient_id: str
    row: dict
    days: pd.DataFrame
    gaps: pd.DataFrame
    seconds: float
    error: str = ""
    duplicate_diffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    conflict_diffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stream_diffs: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ----------------------------------------------------------------------------- CGM


def modal_interval_min(ts: pd.Series) -> float:
    diffs = np.diff(ts.to_numpy("datetime64[ns]").astype(np.int64)) / 60e9
    diffs = diffs[(diffs > 0.5) & (diffs <= 30)]
    if len(diffs) == 0:
        return float("nan")
    values, counts = np.unique(np.round(diffs), return_counts=True)
    return float(values[np.argmax(counts)])


def grid_support(times_ns: np.ndarray, interval_min: float, steps: int = 3) -> np.ndarray:
    """For each reading (sorted times), how many of the times +-1..``steps`` intervals away hold a reading
    (tolerance +-10 % of the interval, at least 0.25 min): 2 x ``steps`` inside a regular series, ~0 for a scan."""
    n = len(times_ns)
    support = np.zeros(n, dtype=np.int64)
    if n < 2:
        return support
    step = interval_min * 60e9
    tol = max(0.25, 0.1 * interval_min) * 60e9
    for k in [*range(-steps, 0), *range(1, steps + 1)]:
        target = times_ns + k * step
        index = np.searchsorted(times_ns, target - tol, side="left")
        hit = (index < n) & (times_ns[np.minimum(index, n - 1)] <= target + tol)
        support += hit
    return support


def on_grid(times_ns: np.ndarray, interval_min: float) -> np.ndarray:
    """True for readings with another reading one interval before or after (sorted times)."""
    return grid_support(times_ns, interval_min, steps=1) > 0


def nominal_interval(times_ns: np.ndarray) -> float:
    """Smallest candidate interval with most readings on its grid (fallback: the modal spacing, else 5)."""
    for candidate in CANDIDATE_INTERVALS_MIN:
        if len(times_ns) >= 3 and on_grid(times_ns, candidate).mean() >= ON_GRID_MIN_SHARE:
            return candidate
    modal = modal_interval_min(pd.Series(times_ns.astype("datetime64[ns]")))
    return modal if np.isfinite(modal) else 5.0


def greedy_keep(times_ns: np.ndarray, min_gap_ns: float, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep a reading when it is >= ``min_gap_ns`` after the last kept one; on a collision the reading
    with more grid support wins (the earlier on a tie). Returns (keep, index of the kept reading each
    dropped one collided with)."""
    n = len(times_ns)
    keep = np.ones(n, dtype=bool)
    reference = np.full(n, -1, dtype=np.int64)
    close = np.flatnonzero(np.diff(times_ns) < min_gap_ns)
    if close.size == 0:
        return keep, reference
    t = times_ns.tolist()
    g = support.tolist()
    # clusters of readings linked by spacings below min_gap; outside them every reading is kept
    starts = close[np.concatenate([[True], np.diff(close) > 1])]
    ends = close[np.concatenate([np.diff(close) > 1, [True]])] + 1
    for first, last_index in zip(starts.tolist(), ends.tolist()):
        last = first
        for i in range(first + 1, last_index + 1):
            if t[i] - t[last] >= min_gap_ns:
                last = i
            elif g[i] > g[last]:
                keep[last] = False
                reference[last] = i
                last = i
            else:
                keep[i] = False
                reference[i] = last
    for _ in range(5):  # a reading dropped against one that was later swapped out points to its replacement
        dropped = np.flatnonzero(~keep)
        stale = dropped[~keep[reference[dropped]]]
        if stale.size == 0:
            break
        reference[stale] = reference[reference[stale]]
    return keep, reference


def _track_cost(track: list[tuple[float, float]], t: np.ndarray, v: np.ndarray, interval_ns: float) -> np.ndarray:
    """How badly readings (t, v) continue a track: timing off its grid (in units of 10 s) plus the
    distance to its value extrapolated from the last two readings (in units of 10 mg/dL)."""
    t_last, v_last = track[-1]
    elapsed = t - t_last
    off_grid = np.abs(elapsed - interval_ns * np.maximum(1, np.round(elapsed / interval_ns)))
    expected = np.full(len(t), v_last)
    if len(track) > 1:
        t_prev, v_prev = track[-2]
        limit = 4.0 / 60e9  # mg/dL per ns
        slope = min(max((v_last - v_prev) / max(t_last - t_prev, 1.0), -limit), limit)
        expected = v_last + slope * elapsed
    return off_grid / 10e9 + np.abs(v - expected) / 10.0


def split_tracks(times_ns: np.ndarray, values: np.ndarray, history: list[tuple[float, float]],
                 min_gap_ns: float, interval_ns: float) -> np.ndarray:
    """Split the readings of a clock-overlap stretch (sorted) into two tracks.

    Each regular series keeps its own timing on the sensor grid (its phase)
    and changes smoothly, so a reading is scored against each track by how far
    it falls off the track's grid (10 s = 1 point) and how far it is from the
    track's value extrapolated from its last two readings (10 mg/dL = 1
    point). Readings closer than ``min_gap_ns`` form a slot (normally one of
    each track); in a slot the assignment with the lowest total score wins,
    and a lone reading joins the cheaper track. Track 1 starts from
    ``history``, the last kept readings before the stretch (the series as it
    was before the clock change). Returns True for the readings of track 1,
    which are kept.
    """
    n = len(times_ns)
    first = np.zeros(n, dtype=bool)
    if n == 0:
        return first
    slot = np.concatenate([[0], np.cumsum(np.diff(times_ns) >= min_gap_ns)])
    track1, track2 = list(history[-2:]), []
    for k in range(int(slot[-1]) + 1):
        members = np.flatnonzero(slot == k)
        t, v = times_ns[members].astype(float), values[members].astype(float)
        if not track1:
            track1.append((t[0], v[0]))
            first[members[0]] = True
            members, t, v = members[1:], t[1:], v[1:]
            if len(members) == 0:
                continue
        c1 = _track_cost(track1, t, v, interval_ns)
        c2 = _track_cost(track2, t, v, interval_ns) if track2 else None
        if len(members) == 1:
            joins_first = c2 is None or c1[0] <= c2[0]
            first[members[0]] = joins_first
            (track1 if joins_first else track2).append((t[0], v[0]))
            continue
        if c2 is None:
            choice, other = int(np.argmin(c1)), None
        else:
            pair_cost = c1[:, None] + c2[None, :]
            np.fill_diagonal(pair_cost, np.inf)
            choice, other = (int(i) for i in np.unravel_index(np.argmin(pair_cost), pair_cost.shape))
        first[members[choice]] = True
        track1.append((t[choice], v[choice]))
        if other is None:
            other = next(i for i in range(len(members)) if i != choice)
        track2.append((t[other], v[other]))
    return first


def deduplicate(times_ns: np.ndarray, values: np.ndarray,
                interval_min: float) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Rule 3 for one sorted source.

    Returns the keep mask, the counts, |diff| of the repeats and of the
    conflicts, and a mask of the first kept reading after each clock-overlap
    stretch (where the clock jumps: no forecast window should cross it).
    """
    support = grid_support(times_ns, interval_min)
    keep, reference = greedy_keep(times_ns, DUPLICATE_FRACTION * interval_min * 60e9, support)
    dropped = np.flatnonzero(~keep)
    counts = {"duplicates": 0, "isolated_conflicts": 0, "overlap_stretches": 0, "overlap_readings": 0}
    if dropped.size == 0:
        return keep, counts, np.zeros(0), np.zeros(0), np.zeros(len(keep), dtype=bool)
    diff = np.abs(values[dropped] - values[reference[dropped]])
    duplicate = diff <= DUPLICATE_TOLERANCE_MGDL
    # runs of collisions (repeats and conflicts), each within OVERLAP_LINK_MIN (or 2 intervals) of the previous one
    link = max(OVERLAP_LINK_MIN, 2 * interval_min) * 60e9
    bounds = np.flatnonzero(np.concatenate([[True], np.diff(times_ns[dropped]) > link]))
    n_conflicts = np.add.reduceat((~duplicate).astype(np.int64), bounds)
    n_regular = np.add.reduceat((support[dropped] >= REGULAR_MIN_SUPPORT).astype(np.int64), bounds)
    sizes = np.diff(np.concatenate([bounds, [len(dropped)]]))
    overlap = (n_conflicts >= OVERLAP_MIN_CONFLICTS) & (n_regular >= OVERLAP_MIN_ON_GRID * sizes)
    counts["duplicates"] = int(duplicate.sum() - (sizes - n_conflicts)[overlap].sum())
    counts["isolated_conflicts"] = int(n_conflicts[~overlap].sum())
    ends = []
    for run in np.flatnonzero(overlap):
        members = dropped[bounds[run]:bounds[run] + sizes[run]]
        partners = reference[members]
        start = min(times_ns[members].min(), times_ns[partners].min())
        end = max(times_ns[members].max(), times_ns[partners].max())
        lo, hi = np.searchsorted(times_ns, start, "left"), np.searchsorted(times_ns, end, "right")
        span = np.arange(lo, hi)
        window = span[keep[span] | np.isin(span, members)]
        before = np.flatnonzero(keep[:lo])[-2:]
        history = [(float(times_ns[i]), float(values[i])) for i in before]
        first_track = split_tracks(times_ns[window], values[window], history, DUPLICATE_FRACTION * interval_min * 60e9,
                                   interval_min * 60e9)
        keep[window] = first_track
        counts["overlap_stretches"] += 1
        counts["overlap_readings"] += int((~first_track).sum())
        ends.append(end)
    breaks = np.zeros(len(keep), dtype=bool)
    kept = np.flatnonzero(keep)
    for end in ends:
        after = np.searchsorted(times_ns[kept], end, "right")
        if after < len(kept):
            breaks[kept[after]] = True
    return keep, counts, diff[duplicate], diff[~duplicate], breaks


def combine_streams(series: pd.DataFrame) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Rule 4 for the sources of one stream group (already de-duplicated each)."""
    stats = {"cross_stream_duplicates": 0, "added_from_second_stream": 0, "stream_disagreement_days": 0,
             "stream_disagreement_readings": 0}
    if series["source"].nunique() < 2:
        return series, stats, np.zeros(0)
    per_day = series.groupby(["day", "source"])["interval_min"].sum().reset_index(name="minutes")
    primary = per_day.sort_values(["day", "minutes"], ascending=[True, False]).drop_duplicates("day")
    primary = primary.set_index("day")["source"]
    is_primary = series["source"].to_numpy() == series["day"].map(primary).to_numpy()
    first = series[is_primary].sort_values("ts")
    other = series[~is_primary].sort_values("ts")
    tolerance = pd.Timedelta(minutes=DUPLICATE_FRACTION * float(series["interval_min"].max()))
    pairs = pd.merge_asof(other[["ts", "day", "glucose_mgdl"]].reset_index(),
                          first[["ts", "glucose_mgdl"]].rename(columns={"ts": "ts_primary", "glucose_mgdl": "g_primary"}),
                          left_on="ts", right_on="ts_primary", direction="nearest", tolerance=tolerance)
    matched = pairs["g_primary"].notna().to_numpy()
    diff = (pairs["glucose_mgdl"] - pairs["g_primary"]).abs()
    by_day = pd.DataFrame({"day": pairs["day"], "diff": diff}).groupby("day")["diff"].agg(["count", "median"])
    agree = (by_day["count"] < STREAM_AGREEMENT_MIN_PAIRS) | (by_day["median"] <= DUPLICATE_TOLERANCE_MGDL)
    day_agrees = pairs["day"].map(agree).fillna(True).to_numpy(bool)
    fill = ~matched & day_agrees
    stats.update({
        "cross_stream_duplicates": int((matched & day_agrees).sum()),
        "added_from_second_stream": int(fill.sum()),
        "stream_disagreement_days": int((~agree).sum()),
        "stream_disagreement_readings": int((~day_agrees).sum()),
    })
    added = series.loc[pairs.loc[fill, "index"].to_numpy()]
    combined = pd.concat([first, added]).sort_values("ts")
    return combined, stats, diff[matched & day_agrees].to_numpy(float)


def clean_cgm(cgm: pd.DataFrame, patient_id: str) -> tuple[pd.DataFrame, dict, dict]:
    """Rules 1-6: de-duplicated sources, combined stream groups, one candidate per day, flags.

    Returns the clean series (with a ``day`` column), counts for the patient row
    and samples of absolute value differences ("duplicate", "conflict",
    "cross_stream") for ``duplicates.csv``.
    """
    auto = cgm.loc[cgm["record_type"] == "cgm", ["ts", "glucose_mgdl", "source"]].copy()
    stats = {"raw_auto_readings": int(len(auto))}
    diffs = {"duplicate": [np.zeros(0)], "conflict": [np.zeros(0)], "cross_stream": [np.zeros(0)]}
    if auto.empty:
        return pd.DataFrame(columns=CLEAN_COLUMNS + ["day"]), stats, {k: v[0] for k, v in diffs.items()}
    auto["source"] = auto["source"].fillna("unknown").astype(str)
    auto = auto.sort_values(["source", "ts"], kind="stable")
    kept = []
    totals = {"duplicates": 0, "isolated_conflicts": 0, "overlap_stretches": 0, "overlap_readings": 0}
    for _, group in auto.groupby("source", sort=False):
        times = group["ts"].to_numpy("datetime64[ns]").astype(np.int64)
        interval = nominal_interval(times)
        keep, counts, d_dup, d_conf, breaks = deduplicate(times, group["glucose_mgdl"].to_numpy(float), interval)
        for key in totals:
            totals[key] += counts[key]
        diffs["duplicate"].append(d_dup)
        diffs["conflict"].append(d_conf)
        kept.append(group[keep].assign(interval_min=interval, clock_break=breaks[keep]))
    series = pd.concat(kept, ignore_index=True)
    series["day"] = series["ts"].dt.normalize()
    series["candidate"] = series["source"].map(lambda s: SENSOR_GROUPS.get(s, s))

    streams = {"cross_stream_duplicates": 0, "added_from_second_stream": 0, "stream_disagreement_days": 0,
               "stream_disagreement_readings": 0}
    parts = []
    for candidate, group in series.groupby("candidate", sort=False):
        if candidate in SENSOR_GROUPS.values():
            group, counts, d_stream = combine_streams(group)
            for key in streams:
                streams[key] += counts[key]
            diffs["cross_stream"].append(d_stream)
        parts.append(group)
    series = pd.concat(parts, ignore_index=True)

    per_day = series.groupby(["day", "candidate"])["interval_min"].agg(["sum", "size"]).reset_index()
    best = per_day.sort_values(["day", "sum", "size"], ascending=[True, False, False]).drop_duplicates("day")
    clean = series.merge(best[["day", "candidate"]], on=["day", "candidate"]).sort_values("ts").reset_index(drop=True)
    clean["flag"] = flags(clean)
    clean["patient_id"] = patient_id
    stats.update({
        "removed_duplicates": totals["duplicates"],
        "removed_isolated_conflicts": totals["isolated_conflicts"],
        "clock_overlap_stretches": totals["overlap_stretches"],
        "removed_clock_overlap": totals["overlap_readings"],
        "removed_cross_stream_duplicates": streams["cross_stream_duplicates"],
        "removed_stream_disagreement": streams["stream_disagreement_readings"],
        "stream_disagreement_days": streams["stream_disagreement_days"],
        "added_from_second_stream": streams["added_from_second_stream"],
        "removed_other_source_days": int(len(series) - len(clean)),
        "clean_readings": int(len(clean)),
        "at_limit_readings": int((clean["flag"] == "at_limit").sum()),
        "spike_readings": int((clean["flag"] == "spike").sum()),
        "sources": "+".join(f"{s}:{n}" for s, n in best["candidate"].value_counts().items()),
        "source_switch_days": int((best["candidate"] != best["candidate"].shift()).sum() - 1),
    })
    return clean[CLEAN_COLUMNS + ["day"]], stats, {k: np.concatenate(v) for k, v in diffs.items()}


def flags(series: pd.DataFrame) -> np.ndarray:
    """'at_limit', 'spike' or '' for each reading of a sorted single series."""
    g = series["glucose_mgdl"].to_numpy(float)
    high = series["source"].map(lambda s: HIGH_LIMITS.get(s, DEFAULT_HIGH_LIMIT)).to_numpy(float)
    at_limit = (g <= LOW_LIMIT) | (g >= high)
    t = series["ts"].to_numpy("datetime64[ns]").astype(np.int64) / 60e9
    interval = series["interval_min"].to_numpy(float)
    spike = np.zeros(len(g), dtype=bool)
    if len(g) >= 3:
        dt1, dt2 = t[1:-1] - t[:-2], t[2:] - t[1:-1]
        r1, r2 = (g[1:-1] - g[:-2]) / dt1, (g[2:] - g[1:-1]) / dt2
        near = (dt1 <= 2.5 * interval[1:-1]) & (dt2 <= 2.5 * interval[1:-1])
        spike[1:-1] = near & (np.abs(r1) > SPIKE_RATE_MGDL_PER_MIN) & (np.abs(r2) > SPIKE_RATE_MGDL_PER_MIN) & \
            (np.sign(r1) != np.sign(r2))
    out = np.full(len(g), "", dtype=object)
    out[spike] = "spike"
    out[at_limit] = "at_limit"
    return out


def gap_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Gaps between consecutive clean readings, by class: number of gaps and missing minutes."""
    if len(clean) < 2:
        return pd.DataFrame(columns=["gap_class", "gaps", "missing_min"])
    t = clean["ts"].to_numpy("datetime64[ns]").astype(np.int64) / 60e9
    interval = clean["interval_min"].to_numpy(float)[1:]
    gap = np.diff(t)
    missing = gap > 1.5 * interval
    rows = []
    lower = 1.5 * interval
    for label, upper in GAP_CLASSES:
        bound = float(upper[1:]) * interval if isinstance(upper, str) else np.full(len(gap), upper)
        inside = missing & (gap > lower) & (gap <= bound)
        rows.append({"gap_class": label, "gaps": int(inside.sum()),
                     "missing_min": float((gap[inside] - interval[inside]).sum())})
        lower = np.maximum(lower, bound)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- days


def day_frame(clean: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """CGM part of the day table."""
    if clean.empty:
        return pd.DataFrame({"day": calendar})
    g = clean.groupby("day")
    counts = clean.groupby(["day", "source"]).size().reset_index(name="n")
    main = counts.sort_values(["day", "n"], ascending=[True, False]).drop_duplicates("day").set_index("day")["source"]
    days = pd.DataFrame({
        "source": main,
        "interval_min": g["interval_min"].first(),
        "n_readings": g.size(),
        "glucose_mean": g["glucose_mgdl"].mean().round(1),
        "glucose_sd": g["glucose_mgdl"].std().round(1),
        "tir_70_180": g["glucose_mgdl"].apply(lambda v: float(v.between(70, 180).mean())).round(3),
        "flags": g["flag"].apply(lambda v: int((v != "").sum())),
    })
    t = clean[["day", "ts"]].copy()
    t["gap"] = t["ts"].diff().dt.total_seconds().div(60)
    t.loc[t["day"] != t["day"].shift(), "gap"] = np.nan  # gaps are measured within the day
    days["longest_gap_min"] = t.groupby("day")["gap"].max().round(1)
    days = days.reindex(calendar)
    days.index.name = "day"
    days["n_readings"] = days["n_readings"].fillna(0).astype(int)
    minutes = clean.groupby("day")["interval_min"].sum().reindex(calendar)
    days["coverage"] = (minutes.to_numpy() / 1440.0).clip(max=1.0)
    days["coverage"] = days["coverage"].fillna(0.0).round(4)
    return days.reset_index()


def insulin_by_day(insulin: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Basal coverage, boluses and 5-min slots per day."""
    out = pd.DataFrame(index=calendar)
    out.index.name = "day"
    out["basal_coverage"] = 0.0
    out["bolus_n"] = 0
    out["injection_today_or_yesterday"] = False
    out["total5_coverage"] = 0.0
    if insulin.empty or len(calendar) == 0:
        return out.reset_index()
    origin = calendar[0].value
    minute = (insulin["ts"].to_numpy("datetime64[ns]").astype(np.int64) - origin) / 60e9
    n_days = len(calendar)

    rates = insulin["kind"].to_numpy() == "basal_rate"
    covered = np.zeros(n_days)
    if rates.any():
        start = minute[rates]
        end = start + np.nan_to_num(insulin.loc[rates, "duration_min"].to_numpy(float), nan=0.0)
        for k in range(3):  # a segment lasts at most 24 h, so it touches at most three calendar days
            day = np.floor(start / 1440).astype(np.int64) + k
            lo, hi = np.maximum(start, day * 1440.0), np.minimum(end, (day + 1) * 1440.0)
            ok = (hi > lo) & (day >= 0) & (day < n_days)
            np.add.at(covered, day[ok], hi[ok] - lo[ok])
    slots = np.zeros(n_days)
    five = np.isin(insulin["kind"].to_numpy(), ["basal_5min"])
    if five.any():
        day = np.floor((minute[five] - 1e-9) / 1440).astype(np.int64)
        ok = (day >= 0) & (day < n_days)
        np.add.at(slots, day[ok], 5.0)
    out["basal_coverage"] = np.minimum(1.0, np.maximum(covered, slots) / 1440.0).round(3)

    total = insulin["kind"].to_numpy() == "total_5min"
    if total.any():
        tslots = np.zeros(n_days)
        day = np.floor((minute[total] - 1e-9) / 1440).astype(np.int64)
        ok = (day >= 0) & (day < n_days)
        np.add.at(tslots, day[ok], 5.0)
        out["total5_coverage"] = np.minimum(1.0, tslots / 1440.0).round(3)
    bolus = np.isin(insulin["kind"].to_numpy(), ["bolus", "bolus_extended"])
    if bolus.any():
        day = np.floor(minute[bolus] / 1440).astype(np.int64)
        ok = (day >= 0) & (day < n_days)
        counts = np.bincount(day[ok], minlength=n_days)
        out["bolus_n"] = counts[:n_days]
    injections = insulin["kind"].to_numpy() == "basal_injection"
    if injections.any():
        day = np.floor(minute[injections] / 1440).astype(np.int64)
        has = np.zeros(n_days, dtype=bool)
        has[day[(day >= 0) & (day < n_days)]] = True
        out["injection_today_or_yesterday"] = has | np.concatenate([[False], has[:-1]])
    return out.reset_index()


def per_day_counts(frame: pd.DataFrame, column: str, calendar: pd.DatetimeIndex, name: str,
                   value: str | None = None) -> pd.Series:
    if frame.empty:
        return pd.Series(0, index=calendar, name=name)
    rows = frame if value is None else frame[frame[value].notna()]
    return rows.groupby(rows[column].dt.normalize()).size().reindex(calendar, fill_value=0).rename(name)


def loop_by_day(calendar: pd.DatetimeIndex, context: dict, periods: pd.DataFrame) -> np.ndarray:
    """'open', 'closed' or 'unknown' for each day (closed_probable counts as closed)."""
    dataset = context["dataset"]
    status = np.full(len(calendar), "unknown", dtype=object)
    patient_loop = str(context.get("loop") or "unknown")
    if dataset in ("replace_bg", "hupa_ucm"):
        status[:] = OPEN
    elif dataset == "loop":
        status[:] = CLOSED
    elif dataset == "dclp3":
        randomization = pd.to_datetime(context.get("randomization_ts"), errors="coerce")
        if context.get("arm") == "CLC" and pd.notna(randomization):
            status[:] = OPEN
            status[calendar >= randomization.normalize() + pd.Timedelta(days=1)] = CLOSED
        elif context.get("arm") == "SAP":
            status[:] = OPEN
    elif dataset == "brist1d_open" and not periods.empty:
        from ..io.common import loop_status

        periods = periods.sort_values("first_ts")
        for _, period in periods.iterrows():
            kind = loop_status([period["device"]])
            value = CLOSED if kind.startswith("closed") else OPEN if kind == "open" else "unknown"
            start = pd.Timestamp(period["first_ts"]).normalize()
            later = calendar >= start  # a pump is in use from its first record until the next pump appears
            status[later] = value
    else:
        if patient_loop.startswith("closed"):
            status[:] = CLOSED
        elif patient_loop == "open":
            status[:] = OPEN
    return status


def period_by_day(calendar: pd.DatetimeIndex, context: dict) -> np.ndarray:
    dataset = context["dataset"]
    period = np.full(len(calendar), "study", dtype=object)
    if dataset in ("replace_bg", "dclp3"):
        randomization = pd.to_datetime(context.get("randomization_ts"), errors="coerce")
        if pd.notna(randomization):
            period[:] = "pre_randomization"
            period[calendar >= randomization.normalize()] = "post_randomization"
    elif dataset == "hupa_ucm":
        start = pd.to_datetime(context.get("window_start"), errors="coerce")
        end = pd.to_datetime(context.get("window_end"), errors="coerce")
        period[:] = "outside_window"
        if pd.notna(start) and pd.notna(end):
            period[(calendar >= start.normalize()) & (calendar <= end.normalize())] = "in_window"
    return period


def longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for value in flags:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


# ----------------------------------------------------------------------------- one patient


def assess_patient(job: dict) -> QualityResult:
    """Script 03 for one patient (in a worker process)."""
    start = time.time()
    dataset, patient_id = job["dataset"], job["patient_id"]
    try:
        source = ProcessedStore(dataset, root=job["derived_root"])
        target = ProcessedStore(dataset, root=job["derived_root"], stage="clean")
        tables = {t: source.read(t, patient_id) for t in ["cgm", "insulin", "carbs", "wearables", "sleep"]}
        clean, stats, diffs = clean_cgm(tables["cgm"], patient_id)
        context = job["context"]
        row = {"dataset": dataset, "patient_id": patient_id, **stats}
        if clean.empty:
            return QualityResult(dataset, patient_id, row, pd.DataFrame(), pd.DataFrame(), round(time.time() - start, 2))
        target.write_frame("cgm", patient_id, clean.drop(columns="day"))
        calendar = pd.date_range(clean["day"].min(), clean["day"].max(), freq="D")
        days = day_frame(clean, calendar)
        days = days.merge(insulin_by_day(tables["insulin"], calendar), on="day", how="left")
        carbs = tables["carbs"]
        days["carbs_n"] = per_day_counts(carbs, "ts", calendar, "carbs_n").to_numpy()
        days["carbs_g"] = (carbs.groupby(carbs["ts"].dt.normalize())["carbs_g"].sum().reindex(calendar).fillna(0.0)
                           .round(1).to_numpy() if not carbs.empty else 0.0)
        wear = tables["wearables"]
        days["hr_rows"] = per_day_counts(wear, "ts", calendar, "hr_rows", "hr_bpm").to_numpy()
        days["steps_rows"] = per_day_counts(wear, "ts", calendar, "steps_rows", "steps").to_numpy()
        sleep = tables["sleep"]
        if not sleep.empty:
            asleep = sleep[~sleep["stage"].str.lower().isin(["wake", "awake"])]
            minutes = (asleep["end"] - asleep["start"]).dt.total_seconds() / 60
            days["sleep_min"] = minutes.groupby(asleep["start"].dt.normalize()).sum().reindex(calendar) \
                .fillna(0).round(0).to_numpy()
        else:
            days["sleep_min"] = 0.0
        days["loop"] = loop_by_day(calendar, context, job["periods"])
        days["period"] = period_by_day(calendar, context)
        basal_ok = (days["basal_coverage"] >= BASAL_COVERAGE_MIN) | days["injection_today_or_yesterday"]
        total_ok = days["total5_coverage"] >= BASAL_COVERAGE_MIN
        days["insulin_complete"] = (basal_ok & (days["bolus_n"] >= 1)) | total_ok
        threshold = job["coverage_threshold"]
        days["valid"] = days["coverage"] >= threshold
        days.insert(0, "patient_id", patient_id)
        days.insert(0, "dataset", dataset)
        days.insert(3, "day_index", np.arange(len(days)))

        valid = days["valid"].to_numpy()
        open_days = days["loop"].to_numpy() == OPEN
        complete = days["insulin_complete"].to_numpy()
        row.update({
            "first_day": calendar[0].date(), "last_day": calendar[-1].date(), "calendar_days": len(calendar),
            "days_with_cgm": int((days["n_readings"] > 0).sum()),
            "valid_days": int(valid.sum()),
            "valid_days_open": int((valid & open_days).sum()),
            "valid_days_closed": int((valid & (days["loop"].to_numpy() == CLOSED)).sum()),
            "valid_days_insulin_complete": int((valid & complete).sum()),
            "valid_days_open_insulin": int((valid & open_days & complete).sum()),
            "valid_days_with_carbs": int((valid & (days["carbs_n"].to_numpy() > 0)).sum()),
            "longest_valid_run": longest_run(valid),
            "mean_coverage": round(float(days["coverage"].mean()), 4),
        })
        for value in COVERAGE_THRESHOLDS:
            row[f"valid_days_at_{int(value * 100)}"] = int((days["coverage"] >= value).sum())
        for name in ["in_window", "post_randomization", "pre_randomization"]:
            if (days["period"] == name).any():
                row[f"valid_days_{name}"] = int((valid & (days["period"].to_numpy() == name)).sum())
        if diffs["conflict"].size:
            row["conflict_abs_diff_median"] = float(np.median(diffs["conflict"]))
        gaps = gap_table(clean.drop(columns="day"))
        gaps.insert(0, "patient_id", patient_id)
        return QualityResult(dataset, patient_id, row, days, gaps, round(time.time() - start, 2),
                             duplicate_diffs=diffs["duplicate"][:5000], conflict_diffs=diffs["conflict"][:5000],
                             stream_diffs=diffs["cross_stream"][:5000])
    except Exception:  # noqa: BLE001
        return QualityResult(dataset, patient_id, {"dataset": dataset, "patient_id": patient_id}, pd.DataFrame(),
                             pd.DataFrame(), round(time.time() - start, 2), traceback.format_exc(limit=6))


def eligibility(row: pd.Series, min_valid: int) -> dict:
    """Eligibility flags: at least ``min_valid`` valid days of each kind (D48)."""
    return {
        "eligible_cgm": row.get("valid_days", 0) >= min_valid,
        "eligible_open_loop": row.get("valid_days_open", 0) >= min_valid,
        "eligible_insulin": row.get("valid_days_insulin_complete", 0) >= min_valid,
        "eligible_open_loop_insulin": row.get("valid_days_open_insulin", 0) >= min_valid,
    }
