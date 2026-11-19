"""T1D-UOM (University of Manchester, Zenodo, CC BY 4.0) -> common schema.

Facts used (README extracted by script 01 and the table samples):

- One file per participant and category, named UoM<Category><ID>.csv (the
  sleep summaries are UoM<ID>sleeptime.csv).
- Timestamps are day-first (17/10/2023 00:00) although the README says
  MM/DD/YYYY (decision D29). The parser takes, per file, the format that reads
  the most values, trying day-first first; the chosen formats are reported.
- Glucose in mmol/L every 5 minutes.
- Basal: insulin_kind R = rapid insulin through a pump, basal_dose read as a
  rate in U/h (values of 1.4-1.7 every 5 minutes; as units per 5 minutes they
  would mean ~18 U/h); L = long-acting injection in units. R records every
  ~5 minutes point to automated insulin delivery: when their median spacing is
  <= 6 minutes the patient is marked loop "closed_probable" and each rate is
  carried for at most 30 minutes (otherwise 12 h).
- Bolus doses in units; nutrition carbs_g in grams.
- UoMsleep<ID>.csv: Garmin 5-min epochs (step_count, heart_rate, intensity,
  stress, sleep_level, resting heart rate). Its heart rate is NOT used
  (decision D39): in iteration 9 it showed no night-time dip at any shift of
  the clock (87 bpm at 0-2 h, 83 at noon), while its stress score, the
  activity steps and the sleep log all follow local time; its step_count
  (1,000-1,900 per 5-min epoch at every hour) and sleep_level (highest at
  midday) are not plausible either. The file only feeds the check table
  ``checks_t1d_uom_epochs_by_hour.csv``. Steps and active kcal come from the
  15-min activity epochs (UoMActivity<ID>.csv: step_count, active_Kcal,
  duration_s), whose exact duplicate rows are dropped before summing per epoch.
- UoM<ID>sleeptime.csv: nightly summaries whose sleep_levels_map_<stage>
  columns list stage intervals in Unix seconds; local time = UTC +
  start_time_offset_s.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.pipeline import Task
from .common import (PartWriter, TaskReport, find_files, find_one, mmol_to_mgdl, numeric_timedelta, parse_datetimes,
                     read_table, to_number)

DATASET = "t1d_uom"
PREFIX = "UOM-"
DATE_FORMATS = ["%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "ISO8601"]
AID_MEDIAN_GAP_MIN = 6.0
CATEGORIES = {
    "glucose": re.compile(r"UoMGlucose(\d+)\.csv", re.I),
    "garmin_epochs": re.compile(r"UoMsleep(\d+)\.csv", re.I),
    "activity": re.compile(r"UoMActivity(\d+)\.csv", re.I),
    "basal": re.compile(r"UoMBasal(\d+)\.csv", re.I),
    "bolus": re.compile(r"UoMBolus(\d+)\.csv", re.I),
    "nutrition": re.compile(r"UoMNutrition(\d+)\.csv", re.I),
    "sleeptime": re.compile(r"UoM(\d+)sleeptime\.csv", re.I),
}


def plan(root: Path, options: dict) -> list[Task]:
    files: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in find_files(root, "UoM*.csv"):
        for category, pattern in CATEGORIES.items():
            match = pattern.fullmatch(path.name)
            if match:
                files[match.group(1)][category] = path
    tasks = [Task(DATASET, f"participant_{pid}", read_participant, {"source_id": pid, "files": paths},
                  sum(p.stat().st_size for p in paths.values())) for pid, paths in sorted(files.items())]
    tasks.append(Task(DATASET, "patients", read_patients,
                      {"bmi": find_one(root, "UoMBMI.csv", required=False), "ids": sorted(files)}, 1.0))
    return tasks


def _times(values: pd.Series, report: TaskReport, category: str) -> pd.Series:
    ts, fmt = parse_datetimes(values, DATE_FORMATS)
    report.tally(category, "time format", fmt, len(values))
    report.tally(category, "time", "unparsed", int((ts.isna() & values.notna()).sum()))
    return ts


def read_participant(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    pid = PREFIX + args["source_id"]
    files = args["files"]
    if "glucose" in files:
        g = read_table(files["glucose"], ["bg_ts", "value"])
        report.rows_read += len(g)
        writer.add("cgm", pd.DataFrame({"patient_id": pid, "ts": _times(g["bg_ts"], report, "glucose"),
                                        "glucose_mgdl": mmol_to_mgdl(g["value"]), "record_type": "cgm",
                                        "source": "unspecified_cgm"}))
    insulin = []
    if "basal" in files:
        b = read_table(files["basal"], ["basal_ts", "basal_dose"], optional=["insulin_kind"])
        report.rows_read += len(b)
        kind = b["insulin_kind"].astype("string").str.strip().str.upper()
        report.count("basal", "insulin_kind", kind)
        ts, dose = _times(b["basal_ts"], report, "basal"), to_number(b["basal_dose"])
        rapid, long_acting = kind == "R", kind == "L"
        insulin.append(pd.DataFrame({"ts": ts, "kind": "basal_rate", "rate_u_per_h": dose})[rapid & dose.notna()])
        insulin.append(pd.DataFrame({"ts": ts, "kind": "basal_injection", "units": dose})[long_acting & (dose > 0)])
        rate_times = ts[rapid].dropna().drop_duplicates().sort_values()
        gap = float(rate_times.diff().dt.total_seconds().div(60).median()) if len(rate_times) > 2 else np.nan
        aid = bool(np.isfinite(gap) and gap <= AID_MEDIAN_GAP_MIN)
        report.set_meta(pid, basal_rate_median_gap_min=gap, basal_rate_cap_min=30.0 if aid else 720.0,
                        loop="closed_probable" if aid else ("open" if rapid.any() or long_acting.any() else None),
                        therapy="pump" if rapid.any() else ("mdi" if long_acting.any() else None))
    if "bolus" in files:
        b = read_table(files["bolus"], ["bolus_ts", "bolus_dose"])
        report.rows_read += len(b)
        dose = to_number(b["bolus_dose"])
        insulin.append(pd.DataFrame({"ts": _times(b["bolus_ts"], report, "bolus"), "kind": "bolus",
                                     "units": dose})[dose > 0])
    if insulin:
        writer.add("insulin", pd.concat(insulin, ignore_index=True).assign(patient_id=pid))
    if "nutrition" in files:
        n = read_table(files["nutrition"], ["meal_ts", "carbs_g"])
        report.rows_read += len(n)
        carbs = to_number(n["carbs_g"])
        writer.add("carbs", pd.DataFrame({"patient_id": pid, "ts": _times(n["meal_ts"], report, "nutrition"),
                                          "carbs_g": carbs})[carbs > 0])
    wear = []
    if "garmin_epochs" in files:
        e = read_table(files["garmin_epochs"], ["sleep_ts"], optional=EPOCH_COLUMNS)
        report.rows_read += len(e)
        ts = _times(e["sleep_ts"], report, "garmin_epochs")
        # heart rate of this file is not used (decision D39): no night-time dip at any clock shift
        report.add_check("t1d_uom_epochs_by_hour", epoch_profile(pid, ts, e))
        report.set_meta(pid, wearables_note="heart rate of the Garmin 5-min epoch file excluded (D39)")
    if "activity" in files:
        a = read_table(files["activity"], ["activity_ts"], optional=["activity_type", "active_Kcal", "step_count",
                                                                      "duration_s", "start_time_s"])
        report.rows_read += len(a)
        before = len(a)
        a = a.drop_duplicates()
        report.tally("activity", "exact duplicate rows dropped", "rows", before - len(a))
        report.count("activity", "activity_type", a["activity_type"])
        a["start"] = _times(a["activity_ts"], report, "activity")
        a["duration"] = to_number(a["duration_s"])
        a["steps"], a["kcal"] = to_number(a["step_count"]), to_number(a["active_Kcal"])
        epochs = a.groupby("start").agg(steps=("steps", "sum"), kcal=("kcal", "sum"), duration=("duration", "max"))
        epochs = epochs.reset_index()
        duration = numeric_timedelta(epochs["duration"], "s", 0, 86400, report, "activity", "duration_s")
        wear.append(pd.DataFrame({"ts": epochs["start"] + duration,
                                  "interval_min": epochs["duration"] / 60.0, "steps": epochs["steps"],
                                  "kcal": epochs["kcal"]}))
    if wear:
        writer.add("wearables", pd.concat(wear, ignore_index=True).assign(patient_id=pid))
    if "sleeptime" in files:
        writer.add("sleep", read_sleeptime(files["sleeptime"], report).assign(patient_id=pid))


EPOCH_COLUMNS = ["step_count", "heart_rate", "current_activity_type_intensity", "stress_level_value", "sleep_level",
                 "resting_heart_rate"]


def epoch_profile(pid: str, ts: pd.Series, epochs: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day profile of the Garmin 5-min epoch file (check D39).

    Heart rate from this file showed no night-time dip in iteration 8, at any
    shift of the sleep log. The profile shows, per hour, how many epochs have a
    heart rate, its mean, the file's own sleep flag, steps, intensity and
    stress, so the meaning and the clock of each column can be judged.
    """
    values = {c: to_number(epochs[c]) for c in EPOCH_COLUMNS}
    frame = pd.DataFrame({"hour": ts.dt.hour, **values}).dropna(subset=["hour"])
    hr = frame["heart_rate"].where(frame["heart_rate"] > 0)
    stress = frame["stress_level_value"].where(frame["stress_level_value"] >= 0)
    grouped = frame.assign(hr=hr, stress=stress, hr_present=hr.notna(),
                           asleep=frame["sleep_level"] == 1).groupby("hour")
    out = pd.DataFrame({
        "epochs": grouped.size(),
        "hr_present_share": grouped["hr_present"].mean().round(3),
        "hr_mean": grouped["hr"].mean().round(1),
        "sleep_level_share": grouped["asleep"].mean().round(3),
        "steps_mean": grouped["step_count"].mean().round(1),
        "intensity_mean": grouped["current_activity_type_intensity"].mean().round(1),
        "stress_mean": grouped["stress"].mean().round(1),
    }).reset_index()
    asleep = frame["sleep_level"] == 1
    out["hr_sleep_level_1_minus_0"] = round(float(hr[asleep].median() - hr[~asleep].median()), 1) \
        if asleep.any() and (~asleep).any() else np.nan
    return out.assign(patient_id=pid)


def read_sleeptime(path: Path, report: TaskReport) -> pd.DataFrame:
    s = read_table(path)
    report.rows_read += len(s)
    offset_column = next((c for c in s.columns if c.lower() == "start_time_offset_s"), None)
    offsets = to_number(s[offset_column]) if offset_column else pd.Series(0.0, index=s.index)
    rows = []
    for column in [c for c in s.columns if c.lower().startswith("sleep_levels_map_")]:
        stage = column.lower().replace("sleep_levels_map_", "")
        for text, offset in zip(s[column], offsets):
            if not isinstance(text, str) or not text.strip().startswith("["):
                continue
            try:
                intervals = json.loads(text)
            except json.JSONDecodeError:
                report.tally("sleeptime", "map", "unreadable JSON", 1)
                continue
            for item in intervals:
                start, end = item.get("startTimeInSeconds"), item.get("endTimeInSeconds")
                if start is None or end is None:
                    continue
                shift = 0.0 if pd.isna(offset) else float(offset)
                rows.append((start + shift, end + shift, stage))
    frame = pd.DataFrame(rows, columns=["start", "end", "stage"]).drop_duplicates()
    frame["start"] = pd.to_datetime(frame["start"], unit="s")
    frame["end"] = pd.to_datetime(frame["end"], unit="s")
    report.count("sleeptime", "stage", frame["stage"])
    return frame


def read_patients(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    patients = pd.DataFrame({"patient_id": [PREFIX + i for i in args["ids"]], "dataset": DATASET,
                             "source_id": args["ids"], "dates": "real dates"})
    if args.get("bmi"):
        bmi = read_table(args["bmi"], ["participant_id"], optional=["weight_kg", "height_m", "bmi"])
        number = bmi["participant_id"].astype("string").str.extract(r"(\d+)")[0]
        patients = patients.merge(pd.DataFrame({
            "patient_id": PREFIX + number, "weight_kg": to_number(bmi["weight_kg"]),
            "height_cm": to_number(bmi["height_m"]) * 100, "bmi": to_number(bmi["bmi"]),
        }), on="patient_id", how="outer")
        patients["dataset"] = DATASET
    report.payload = patients
