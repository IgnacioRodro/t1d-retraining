"""Loop Observational Study (JAEB) -> common schema.

Facts used (LOOP ReadMe and DataGlossary extracted by script 01):

- Pipe-delimited tables split into several files (LOOPDeviceCGM1..6,
  LOOPDeviceBasal1..3); each file is read in chunks by its own task.
- Dates are de-identified with a random offset of up to 365 days per
  participant, so weekdays and seasons are not meaningful (D24, corrected).
- Local time (decision D31, revised in iteration 8). DeviceDtTm (local device
  time) is present in about 30 % of the CGM rows, mostly those uploaded from
  the Dexcom receiver; UTCDtTm is always given; TmZnOffset is only filled
  where DeviceDtTm is. Rows with DeviceDtTm keep it, unless the device clock
  is more than 14 h away from UTC (a clock that was never set: about 70 000
  rows in iteration 10), which are treated as rows without device time. Each such row also gives
  the participant's real UTC offset at that moment (DeviceDtTm - UTCDtTm), so
  phase A stores one offset observation per patient and hour (auxiliary table
  ``tz_offsets``). In phase B, ``finalize`` gives every row without device
  time the offset observed nearest in time (within 14 days), which follows
  daylight saving and travel; only rows far from any observation fall back
  to PtRoster.PtTimezoneOffset, a standard-time offset (iteration 7 used it
  for ~70 % of rows, which put meals ~1 h early during summer time).
- CGM values come from Tidepool in mmol/L (converted back with Tidepool's
  factor 18.01559 and rounded to whole mg/dL); RecordType "CGM" only. The
  same reading often arrives twice, through Apple HealthKit (Loop app) and
  through a direct Dexcom upload, so the CGM source is split by OriginName:
  "dexcom_healthkit" or "dexcom_direct". Script 03 merges the two streams.
- Basal: Rate (U/h) for Duration (ms). Bolus: Normal and Extended (units),
  Duration in ms for Tidepool and minutes for Diasend uploads
  (``common.duration_minutes``). Food: CarbsNet in CarbUnits (grams).
- Gender (D108): the baseline survey (Surveys.txt, Period "Baseline", one row
  per SubjectID, the participant ID of the other tables) asks "What is your
  gender?" with codes 1 = Male, 2 = Female, 3 = Non-binary (DataGlossary.rtf;
  form "Loop baseline general data collection"). The check table
  ``loop_survey_match`` says how many roster participants have a baseline
  survey and how often its age agrees with the roster's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.pipeline import Task
from .common import (TIDEPOOL_MGDL_PER_MMOL, PartWriter, TaskReport, duration_minutes, find_files, find_one,
                     first_valid, numeric_timedelta, parse_datetimes, read_table, to_number)

DATASET = "loop"
PREFIX = "LOOP-"
ISO = ["ISO8601"]
TIME_COLUMNS = ["PtID", "UTCDtTm"]
TIME_OPTIONAL = ["DeviceDtTm", "TmZnOffset"]
AUX_TABLES = ["tz_offsets"]
OFFSET_TOLERANCE = pd.Timedelta(days=14)
MAX_ABS_OFFSET_MIN = 14 * 60
GENDER_CODES = {"1": "male", "2": "female", "3": "non-binary"}  # Surveys.gender (DataGlossary.rtf)


def patient_ids(values: pd.Series) -> pd.Series:
    return PREFIX + values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def roster_offsets(path: Path) -> pd.Series:
    """PtID -> participant time-zone offset in hours."""
    roster = read_table(path, ["PtID"], optional=["PtTimezoneOffset"])
    offsets = pd.Series(to_number(roster["PtTimezoneOffset"]).to_numpy(), index=roster["PtID"].str.strip())
    return offsets[~offsets.index.duplicated(keep="first")]


def local_time(frame: pd.DataFrame, offsets: pd.Series, writer: PartWriter, report: TaskReport,
               source: str) -> pd.DataFrame:
    """Local time, UTC time and how the local time was obtained, for the rows of one chunk.

    Also writes the offset observations (device time - UTC) of the chunk to
    the auxiliary table ``tz_offsets`` (one row per patient and UTC hour).
    """
    device, _ = parse_datetimes(frame["DeviceDtTm"], ISO)
    utc, _ = parse_datetimes(frame["UTCDtTm"], ISO)
    observed = ((device - utc).dt.total_seconds() / 60).round()
    plausible = observed.abs() <= MAX_ABS_OFFSET_MIN
    report.tally(source, "device - UTC offset", "outside +-14 h (device time ignored)",
                 int((observed.notna() & ~plausible).sum()))
    device = device.where(plausible)  # a device clock days away from UTC was never set: use UTC + offset instead
    roster_minutes = frame["PtID"].str.strip().map(offsets) * 60
    from_row = utc + numeric_timedelta(frame["TmZnOffset"], "min", -MAX_ABS_OFFSET_MIN, MAX_ABS_OFFSET_MIN, report,
                                       source, "TmZnOffset")
    from_roster = utc + numeric_timedelta(roster_minutes, "min", -MAX_ABS_OFFSET_MIN, MAX_ABS_OFFSET_MIN, report,
                                          source, "PtTimezoneOffset (min)")
    ts = first_valid(device, from_row, from_roster)
    origin = np.select([device.notna(), from_row.notna(), from_roster.notna()],
                       ["device_time", "utc_plus_row_offset", "utc_plus_roster_offset"], default="unresolved")
    for value, n in pd.Series(origin).value_counts().items():
        report.tally(source, "local time from (phase A)", str(value), int(n))
    obs = pd.DataFrame({"patient_id": patient_ids(frame["PtID"]), "utc": utc.dt.floor("h"),
                        "offset_min": (observed / 15).round() * 15})[observed.notna() & plausible]
    if not obs.empty:
        hourly = obs.groupby(["patient_id", "utc"])["offset_min"].agg(["median", "size"]).reset_index()
        writer.add("tz_offsets", hourly.rename(columns={"median": "offset_min", "size": "n"}))
    return pd.DataFrame({"ts": ts, "utc": utc, "time_origin": origin}, index=frame.index)


def finalize(table: str, frame: pd.DataFrame, meta: dict, aux: dict) -> pd.DataFrame:
    """Phase B: rows without device time get the offset observed nearest in time (within 14 days)."""
    if "utc" not in frame or "time_origin" not in frame:
        return frame
    observations = aux.get("tz_offsets")
    needs = frame["time_origin"].isin(["utc_plus_roster_offset", "unresolved"]) & frame["utc"].notna()
    if observations is None or observations.empty or not needs.any():
        return frame
    timeline = (observations.assign(weighted=observations["offset_min"] * observations["n"])
                .groupby("utc")[["weighted", "n"]].sum())
    timeline = (timeline["weighted"] / timeline["n"]).round().rename("offset_min").reset_index().sort_values("utc")
    timeline["utc"] = pd.to_datetime(timeline["utc"]).astype("datetime64[ns]")
    rows = frame.loc[needs, ["utc"]].copy()
    rows["utc"] = pd.to_datetime(rows["utc"]).astype("datetime64[ns]")
    rows["row"] = rows.index
    matched = pd.merge_asof(rows.sort_values("utc"), timeline, on="utc", direction="nearest",
                            tolerance=OFFSET_TOLERANCE).set_index("row")
    found = matched["offset_min"].notna()
    index = matched.index[found]
    frame = frame.copy()
    frame.loc[index, "ts"] = frame.loc[index, "utc"] + pd.to_timedelta(matched.loc[index, "offset_min"], unit="min")
    frame.loc[index, "time_origin"] = "utc_plus_observed_offset"
    return frame


def describe_aux(aux: dict, meta: dict) -> dict:
    """Per-patient summary of the observed UTC offsets (for results/02.../loop_time_offsets.csv)."""
    observations = aux.get("tz_offsets")
    if observations is None or observations.empty:
        return {"offset_hours_observed": 0}
    hours = observations.groupby("offset_min")["n"].sum().sort_values(ascending=False)
    share = hours / hours.sum()
    return {
        "offset_hours_observed": int(observations["utc"].nunique()),
        "offset_main_h": float(share.index[0] / 60),
        "offset_main_share": round(float(share.iloc[0]), 3),
        "offset_values_h": "+".join(f"{v / 60:g}" for v in share.index[:5]),
        "offset_changes": int((observations.sort_values("utc")["offset_min"].diff().fillna(0) != 0).sum()),
        "first_observation_utc": str(observations["utc"].min()),
        "last_observation_utc": str(observations["utc"].max()),
    }


def plan(root: Path, options: dict) -> list[Task]:
    roster = find_one(root, "PtRoster.txt")
    uploads = find_one(root, "LOOPDeviceUploads.txt")
    tasks = []
    for path in find_files(root, "LOOPDeviceCGM*.txt"):
        tasks.append(Task(DATASET, f"cgm_{path.stem}", read_cgm, {"path": path, "roster": roster}, path.stat().st_size))
    for path in find_files(root, "LOOPDeviceBasal*.txt"):
        tasks.append(Task(DATASET, f"basal_{path.stem}", read_basal, {"path": path, "roster": roster},
                          path.stat().st_size))
    bolus = find_one(root, "LOOPDeviceBolus.txt")
    tasks.append(Task(DATASET, "bolus", read_bolus, {"path": bolus, "roster": roster, "uploads": uploads},
                      bolus.stat().st_size))
    food = find_one(root, "LOOPDeviceFood.txt")
    tasks.append(Task(DATASET, "food", read_food, {"path": food, "roster": roster}, food.stat().st_size))
    tasks.append(Task(DATASET, "patients", read_patients, {"roster": roster, "uploads": uploads,
                                                           "final": find_one(root, "LOOPPtFinalStatus.txt", False),
                                                           "surveys": find_one(root, "Surveys.txt", False)},
                      1.0))
    return tasks


def read_cgm(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    offsets = roster_offsets(args["roster"])
    source = args["path"].stem
    for chunk in read_table(args["path"], TIME_COLUMNS + ["RecordType", "CGMVal"],
                            optional=TIME_OPTIONAL + ["Units", "OriginName"],
                            chunksize=options["chunk_rows"], max_chunks=options.get("max_chunks")):
        report.rows_read += len(chunk)
        report.count("LOOPDeviceCGM", "RecordType", chunk["RecordType"])
        report.count("LOOPDeviceCGM", "Units", chunk["Units"])
        report.count("LOOPDeviceCGM", "OriginName", chunk["OriginName"])
        chunk = chunk[chunk["RecordType"].str.strip().str.upper() == "CGM"]
        units = chunk["Units"].astype("string").str.lower()
        value = to_number(chunk["CGMVal"])
        mgdl = value.where(units.str.contains("mg", na=False), (value * TIDEPOOL_MGDL_PER_MMOL).round(0))
        healthkit = chunk["OriginName"].astype("string").str.contains("healthkit", case=False, na=False)
        times = local_time(chunk, offsets, writer, report, source)
        writer.add("cgm", pd.DataFrame({
            "patient_id": patient_ids(chunk["PtID"]), "ts": times["ts"], "glucose_mgdl": mgdl, "record_type": "cgm",
            "source": np.where(healthkit, "dexcom_healthkit", "dexcom_direct"),
            "utc": times["utc"], "time_origin": times["time_origin"],
        }))


def read_basal(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    offsets = roster_offsets(args["roster"])
    source = args["path"].stem
    for chunk in read_table(args["path"], TIME_COLUMNS, optional=TIME_OPTIONAL + [
            "ParentLOOPDeviceUploadsID", "BasalType", "Duration", "ExpectedDuration", "Rate", "Percnt", "SuprRate"],
                            chunksize=options["chunk_rows"], max_chunks=options.get("max_chunks")):
        report.rows_read += len(chunk)
        report.count("LOOPDeviceBasal", "BasalType", chunk["BasalType"])
        rate = to_number(chunk["Rate"])
        percent = to_number(chunk["Percnt"])
        percent = percent.where(percent <= 1.5, percent / 100.0)
        rate = first_valid(rate, percent * to_number(chunk["SuprRate"]))
        suspended = chunk["BasalType"].astype("string").str.lower().str.contains("suspend", na=False)
        rate = rate.mask(suspended & rate.isna(), 0.0)
        no_source = pd.Series(pd.NA, index=chunk.index, dtype="string")
        duration = duration_minutes(first_valid(to_number(chunk["Duration"]), to_number(chunk["ExpectedDuration"])),
                                    chunk["ParentLOOPDeviceUploadsID"], no_source, report, "LOOPDeviceBasal")
        report.tally("LOOPDeviceBasal", "rate", "missing (dropped)", int(rate.isna().sum()))
        times = local_time(chunk, offsets, writer, report, source)
        writer.add("insulin", pd.DataFrame({
            "patient_id": patient_ids(chunk["PtID"]), "ts": times["ts"], "kind": "basal_rate", "rate_u_per_h": rate,
            "duration_min": duration, "utc": times["utc"], "time_origin": times["time_origin"],
        })[rate.notna()])


def read_bolus(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    offsets = roster_offsets(args["roster"])
    uploads = read_table(args["uploads"], ["RecID"], optional=["DataSource"])
    sources = uploads.set_index(uploads["RecID"].str.strip())["DataSource"].astype("string")
    for chunk in read_table(args["path"], TIME_COLUMNS, optional=TIME_OPTIONAL + [
            "ParentLOOPDeviceUploadsID", "BolusType", "Normal", "Extended", "Duration"],
            chunksize=options["chunk_rows"], max_chunks=options.get("max_chunks")):
        report.rows_read += len(chunk)
        report.count("LOOPDeviceBolus", "BolusType", chunk["BolusType"])
        source = chunk["ParentLOOPDeviceUploadsID"].astype("string").str.strip().map(sources)
        report.count("LOOPDeviceBolus", "upload DataSource", source)
        duration = duration_minutes(chunk["Duration"], chunk["ParentLOOPDeviceUploadsID"], source, report,
                                    "LOOPDeviceBolus")
        pid = patient_ids(chunk["PtID"])
        times = local_time(chunk, offsets, writer, report, "LOOPDeviceBolus")
        ts, utc, origin = times["ts"], times["utc"], times["time_origin"]
        normal, extended = to_number(chunk["Normal"]), to_number(chunk["Extended"])
        writer.add("insulin", pd.concat([
            pd.DataFrame({"patient_id": pid, "ts": ts, "kind": "bolus", "units": normal, "utc": utc,
                          "time_origin": origin})[normal > 0],
            pd.DataFrame({"patient_id": pid, "ts": ts, "kind": "bolus_extended", "units": extended,
                          "duration_min": duration, "utc": utc, "time_origin": origin})[extended > 0],
        ], ignore_index=True))


def read_food(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    offsets = roster_offsets(args["roster"])
    for chunk in read_table(args["path"], TIME_COLUMNS + ["CarbsNet"], optional=TIME_OPTIONAL + ["CarbUnits"],
                            chunksize=options["chunk_rows"], max_chunks=options.get("max_chunks")):
        report.rows_read += len(chunk)
        report.count("LOOPDeviceFood", "CarbUnits", chunk["CarbUnits"])
        carbs = to_number(chunk["CarbsNet"])
        grams = chunk["CarbUnits"].astype("string").str.lower().fillna("grams").str.startswith("gram")
        keep = (carbs > 0) & grams
        times = local_time(chunk, offsets, writer, report, "LOOPDeviceFood")
        writer.add("carbs", pd.DataFrame({"patient_id": patient_ids(chunk["PtID"]), "ts": times["ts"],
                                          "carbs_g": carbs, "utc": times["utc"],
                                          "time_origin": times["time_origin"]})[keep])


def read_patients(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    roster = read_table(args["roster"], ["PtID"], optional=["EnrollDt", "PtStatus", "PtCohort", "AgeAtEnrollment",
                                                            "LoopUseTimeAtEnroll", "PtTimezoneOffset"])
    report.count("PtRoster", "LoopUseTimeAtEnroll", roster["LoopUseTimeAtEnroll"])
    report.count("PtRoster", "PtCohort", roster["PtCohort"])
    patients = pd.DataFrame({
        "patient_id": patient_ids(roster["PtID"]), "dataset": DATASET, "source_id": roster["PtID"].str.strip(),
        "age": to_number(roster["AgeAtEnrollment"]), "status": roster["PtStatus"], "arm": roster["PtCohort"],
        "loop_use_at_enrolment": roster["LoopUseTimeAtEnroll"],
        "timezone_offset_h": to_number(roster["PtTimezoneOffset"]),
        "enrollment_ts": parse_datetimes(roster["EnrollDt"], ["%m/%d/%Y", "ISO8601"])[0],
        "therapy": "pump", "loop": "closed",
        "dates": "de-identified (random offset up to 365 days); weekday and season not meaningful",
    }).drop_duplicates("patient_id")
    uploads = read_table(args["uploads"], ["PtID"], optional=["DeviceManufact", "DeviceModel", "DeviceType"])
    report.count("LOOPDeviceUploads", "DeviceType", uploads["DeviceType"])
    kind = uploads["DeviceType"].astype("string").str.lower()
    name = (uploads["DeviceManufact"].fillna("") + " " + uploads["DeviceModel"].fillna("")).str.strip()
    pid = patient_ids(uploads["PtID"])

    def joined(mask: pd.Series) -> pd.Series:
        return name[mask].groupby(pid[mask]).agg(lambda s: "+".join(sorted({v for v in s if v})))

    patients["insulin_devices"] = patients["patient_id"].map(joined(kind.str.contains("pump", na=False)))
    patients["cgm_device"] = patients["patient_id"].map(joined(kind.str.contains("cgm", na=False)))
    if args.get("surveys"):
        gender = baseline_gender(args["surveys"], patients, report)
        patients["sex"] = patients["patient_id"].map(gender)
    report.payload = patients


def baseline_gender(path: Path, patients: pd.DataFrame, report: TaskReport) -> pd.Series:
    """Gender per patient from the baseline survey (codes of DataGlossary.rtf), with a check of the link to the
    roster: participants with a baseline survey, and how often its age is within a year of the roster's."""
    surveys = read_table(path, ["SubjectID", "Period"], optional=["gender", "ageAtBaseline"])
    baseline = surveys[surveys["Period"].astype("string").str.strip().str.lower() == "baseline"].copy()
    baseline["patient_id"] = patient_ids(baseline["SubjectID"])
    baseline = baseline.drop_duplicates("patient_id")
    code = baseline["gender"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    report.count("Surveys", "gender (baseline)", code)
    gender = pd.Series(code.map(GENDER_CODES).to_numpy(), index=baseline["patient_id"].to_numpy())
    roster_age = patients.set_index("patient_id")["age"]
    linked = baseline[baseline["patient_id"].isin(roster_age.index)]
    agree = (to_number(linked["ageAtBaseline"]).to_numpy()
             - roster_age.reindex(linked["patient_id"]).to_numpy())
    report.add_check("loop_survey_match", pd.DataFrame([{
        "roster_patients": int(len(patients)), "baseline_surveys": int(len(baseline)),
        "surveys_in_roster": int(len(linked)), "roster_with_gender": int(gender.reindex(patients["patient_id"])
                                                                         .notna().sum()),
        "age_within_1_year": int(np.sum(np.abs(agree) <= 1)), "age_compared": int(np.sum(np.isfinite(agree)))}]))
    return gender
