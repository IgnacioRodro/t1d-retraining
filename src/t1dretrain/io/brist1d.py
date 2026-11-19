"""BrisT1D-Open (University of Bristol, CC BY 4.0) -> common schema.

Facts used (README extracted by script 01):

- device_data/processed_state/PXX.csv, one long table per participant with one
  row per device reading: bg (mmol/L, CGM), insulin (total units delivered by
  the pump in the previous 5 minutes), carbs (g), hr (mean bpm over the
  previous 5 minutes), dist, steps, cals (totals over the previous 5 minutes),
  activity (user labels, not used) and device (name of the device of that row).
- Timestamps are real local times (weekday analysis is possible).
- Device names give the CGM and pump of each period; several participants
  switch sensor or pump, or move to closed loop, during the study, so device
  periods are exported (device_periods.csv). The pump of each period is read
  from the insulin and the carbohydrate rows (the pump's bolus calculator
  records the carbs, and in a few files only the carbs name the new pump).
  The loop status follows the pump names (closed / closed_probable / open /
  unknown); a change is written as "open -> closed" with the switch time.
- CGM source names are normalised ("Guardian 4" and "Medtronic Guardian 4"
  are the same sensor).
- demographic_data.csv: free-text answers (age, gender, ethnicity, diabetes
  duration), latin-1 encoded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..data.pipeline import Task
from .common import (PartWriter, TaskReport, find_files, find_one, loop_status, mmol_to_mgdl, parse_datetimes,
                     read_table, to_number)

DATASET = "brist1d_open"
PREFIX = "BRIS-"
SOURCE_ALIASES = {"guardian_4": "medtronic_guardian_4"}


def plan(root: Path, options: dict) -> list[Task]:
    files = [p for p in find_files(root, "P*.csv") if re.fullmatch(r"P\d+\.csv", p.name, re.IGNORECASE)]
    tasks = [Task(DATASET, f"device_{p.stem}", read_participant, {"path": p}, p.stat().st_size) for p in files]
    demo = find_one(root, "demographic_data.csv", required=False)
    tasks.append(Task(DATASET, "patients", read_patients, {"path": demo, "files": files}, 1.0))
    return tasks


def patient_id(stem: str) -> str:
    number = re.search(r"\d+", stem).group(0)
    return f"{PREFIX}P{int(number):02d}"


def read_participant(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    path = args["path"]
    pid = patient_id(path.stem)
    frame = read_table(path, ["timestamp"], optional=["bg", "insulin", "carbs", "hr", "dist", "steps", "cals",
                                                       "device"])
    report.rows_read += len(frame)
    ts, _ = parse_datetimes(frame["timestamp"], ["ISO8601"])
    device = frame["device"].astype("string").str.strip()
    report.count("processed_state", "device", device)
    bg = mmol_to_mgdl(frame["bg"])
    insulin = to_number(frame["insulin"])
    carbs = to_number(frame["carbs"])
    hr, steps, cals = to_number(frame["hr"]), to_number(frame["steps"]), to_number(frame["cals"])

    source = device.str.lower().str.replace(r"\s+", "_", regex=True).replace(SOURCE_ALIASES)
    writer.add("cgm", pd.DataFrame({"patient_id": pid, "ts": ts, "glucose_mgdl": bg, "record_type": "cgm",
                                    "source": source})[bg.notna()])
    writer.add("insulin", pd.DataFrame({"patient_id": pid, "ts": ts, "kind": "total_5min", "units": insulin})
               [insulin.notna()])
    writer.add("carbs", pd.DataFrame({"patient_id": pid, "ts": ts, "carbs_g": carbs})[carbs > 0])
    wearable = hr.notna() | steps.notna() | cals.notna()
    wear = pd.DataFrame({"ts": ts, "hr_bpm": hr, "steps": steps, "kcal": cals})[wearable]
    grouped = wear.groupby("ts")
    wear = pd.concat([grouped["hr_bpm"].mean(), grouped[["steps", "kcal"]].sum(min_count=1)], axis=1).reset_index()
    writer.add("wearables", wear.assign(patient_id=pid, interval_min=5.0))

    roles = {"cgm": bg.notna(), "insulin": insulin.notna(), "carbs": carbs.notna(), "watch": wearable}
    for role, mask in roles.items():
        if not mask.any():
            continue
        periods = pd.DataFrame({"ts": ts[mask], "device": device[mask].fillna("unknown")})
        for name, group in periods.groupby("device"):
            report.devices.append({"dataset": DATASET, "patient_id": pid, "role": role, "device": name,
                                   "first_ts": group["ts"].min(), "last_ts": group["ts"].max(), "rows": len(group)})
    pump_rows = pd.DataFrame({"ts": ts, "device": device})[(insulin.notna() | (carbs > 0)) & device.notna()]
    periods = pump_rows.groupby("device")["ts"].agg(["min", "max"]).sort_values("min")
    statuses = [(name, loop_status([name]), start) for name, start in zip(periods.index, periods["min"])]
    sequence, switches = [], []
    for name, status, start in statuses:
        if not sequence or sequence[-1] != status:
            if sequence:
                switches.append(f"{status} from {start:%Y-%m-%d}")
            sequence.append(status)
    cgms = device[bg.notna()].dropna().unique().tolist()
    report.set_meta(pid, insulin_devices=" -> ".join(periods.index), cgm_device=" -> ".join(cgms),
                    loop=" -> ".join(sequence) if sequence else "unknown",
                    loop_changes="; ".join(switches) if switches else None)


def read_patients(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    ids = [patient_id(p.stem) for p in args["files"]]
    patients = pd.DataFrame({"patient_id": ids, "dataset": DATASET, "source_id": [p.stem for p in args["files"]],
                             "therapy": "pump_or_mdi", "dates": "real dates"})
    if args.get("path"):
        demo = read_table(args["path"])
        demo.columns = [c.strip() for c in demo.columns]
        number = demo.iloc[:, 0].astype("string").str.extract(r"(\d+)")[0]
        info = pd.DataFrame({
            "patient_id": PREFIX + "P" + number.astype(int).astype(str).str.zfill(2),
            "age": to_number(demo.iloc[:, 1]),
            "sex": demo.iloc[:, 2].astype("string").str.strip(),
            "diabetes_duration_years": to_number(demo.iloc[:, 4].astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0]),
        })
        patients = patients.merge(info, on="patient_id", how="outer")
        patients["dataset"] = DATASET
    report.payload = patients
