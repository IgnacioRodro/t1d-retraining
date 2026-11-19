"""DCLP3 (JAEB, iDCL protocol 3: Control-IQ vs sensor-augmented pump) -> common schema.

Facts used (ReadMe extracted by script 01 and the table samples):

- cgm.txt is the analysis CGM (Dexcom G6) of both arms, baseline and
  post-randomisation periods; DataDtTm looks like ``11DEC17:23:59:25``.
  Dates are de-identified (shifted), so weekdays are not meaningful.
- Pump_* tables come from the study t:slim X2 pumps: CommandedBasalRate is a
  rate-change log (U/h from that moment on); BolusDelivered gives BolusAmount
  and BolusType. When DataDtTm_adjusted is filled it corrects the device clock
  and is used instead of DataDtTm.
- The SAP arm used personal pumps: their insulin is not in the public tables,
  so SAP patients appear without insulin rows (checked in the summary).
- Carbohydrates: no pump bolus-calculator carbs are public (decision D20). The
  study meter (RocheMeter_a) has an optional Carbs field, kept when > 0 and
  not a control-solution test.
- Patients: PtRoster_a (arm, randomisation date), DiabScreening_a (age, sex,
  pump or injections, pump model), DiabLocalHbA1c_a.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.pipeline import Task
from .common import PartWriter, TaskReport, find_one, first_valid, parse_datetimes, read_table, to_number

DATASET = "dclp3"
PREFIX = "DCLP3-"
CGM_SOURCE = "dexcom_g6"
BASAL_RATE_CAP_MIN = 720.0
ISO = ["ISO8601"]


def patient_ids(values: pd.Series) -> pd.Series:
    return PREFIX + values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def adjusted_time(frame: pd.DataFrame, report: TaskReport, source: str) -> pd.Series:
    original, _ = parse_datetimes(frame["DataDtTm"], ISO)
    adjusted, _ = parse_datetimes(frame["DataDtTm_adjusted"], ISO)
    report.tally(source, "time", "DataDtTm_adjusted used", int(adjusted.notna().sum()))
    report.tally(source, "time", "DataDtTm used", int((adjusted.isna() & original.notna()).sum()))
    return first_valid(adjusted, original)


def plan(root: Path, options: dict) -> list[Task]:
    files = {
        "cgm": find_one(root, "cgm.txt"),
        "basal": find_one(root, "Pump_BasalRateChange.txt"),
        "bolus": find_one(root, "Pump_BolusDelivered.txt"),
        "meter": find_one(root, "RocheMeter_a.txt", required=False),
        "roster": find_one(root, "PtRoster_a.txt"),
        "screening": find_one(root, "DiabScreening_a.txt", required=False),
        "hba1c": find_one(root, "DiabLocalHbA1c_a.txt", required=False),
    }
    size = {k: (v.stat().st_size if v else 0) for k, v in files.items()}
    tasks = [
        Task(DATASET, "cgm", read_cgm, {"path": files["cgm"]}, size["cgm"]),
        Task(DATASET, "basal", read_basal, {"path": files["basal"]}, size["basal"]),
        Task(DATASET, "bolus", read_bolus, {"path": files["bolus"]}, size["bolus"]),
        Task(DATASET, "patients", read_patients, files, 1.0),
    ]
    if files["meter"]:
        tasks.append(Task(DATASET, "meter_carbs", read_meter_carbs, {"path": files["meter"]}, size["meter"]))
    return tasks


def read_cgm(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    first_post: dict[str, pd.Timestamp] = {}
    for chunk in read_table(args["path"], ["PtID", "DataDtTm", "CGM"], optional=["Period"],
                            chunksize=options["chunk_rows"], max_chunks=options.get("max_chunks")):
        report.rows_read += len(chunk)
        report.count("cgm.txt", "Period", chunk["Period"])
        ts, fmt = parse_datetimes(chunk["DataDtTm"], ["%d%b%y:%H:%M:%S", "ISO8601"])
        report.tally("cgm.txt", "time format", fmt, len(chunk))
        report.tally("cgm.txt", "time", "unparsed", int(ts.isna().sum()))
        pid = patient_ids(chunk["PtID"])
        writer.add("cgm", pd.DataFrame({"patient_id": pid, "ts": ts, "glucose_mgdl": to_number(chunk["CGM"]),
                                        "record_type": "cgm", "source": CGM_SOURCE}))
        post = chunk["Period"].astype("string").str.contains("Post", case=False, na=False)
        for patient, first in ts[post].groupby(pid[post]).min().items():
            first_post[patient] = min(first, first_post.get(patient, first))
    for patient, first in first_post.items():
        report.set_meta(patient, cgm_first_post_randomization=str(first))


def read_basal(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    for chunk in read_table(args["path"], ["PtID", "DataDtTm", "CommandedBasalRate"], optional=["DataDtTm_adjusted"],
                            chunksize=options["chunk_rows"], max_chunks=options.get("max_chunks")):
        report.rows_read += len(chunk)
        writer.add("insulin", pd.DataFrame({
            "patient_id": patient_ids(chunk["PtID"]), "ts": adjusted_time(chunk, report, "Pump_BasalRateChange"),
            "kind": "basal_rate", "rate_u_per_h": to_number(chunk["CommandedBasalRate"]),
        }))


def read_bolus(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    frame = read_table(args["path"], ["PtID", "DataDtTm", "BolusAmount"], optional=["DataDtTm_adjusted", "BolusType"])
    report.rows_read += len(frame)
    report.count("Pump_BolusDelivered", "BolusType", frame["BolusType"])
    units = to_number(frame["BolusAmount"])
    extended = frame["BolusType"].astype("string").str.contains("extend", case=False, na=False)
    out = pd.DataFrame({
        "patient_id": patient_ids(frame["PtID"]), "ts": adjusted_time(frame, report, "Pump_BolusDelivered"),
        "kind": extended.map({True: "bolus_extended", False: "bolus"}), "units": units,
    })
    writer.add("insulin", out[units > 0])


def read_meter_carbs(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    frame = read_table(args["path"], ["PtID", "DataDtTm"], optional=["Carbs", "IsQCTest", "DataDtTm_adjusted"])
    report.rows_read += len(frame)
    carbs = to_number(frame["Carbs"])
    control = frame["IsQCTest"].astype("string").str.lower().isin(["true", "1", "yes"])
    keep = (carbs > 0) & ~control
    report.tally("RocheMeter_a", "Carbs", "> 0 (kept)", int(keep.sum()))
    writer.add("carbs", pd.DataFrame({"patient_id": patient_ids(frame["PtID"]),
                                      "ts": adjusted_time(frame, report, "RocheMeter_a"), "carbs_g": carbs})[keep])


def read_patients(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    roster = read_table(args["roster"], ["PtID"], optional=["EnrollDt", "RandDt", "trtGroup", "PtStatus"])
    report.count("PtRoster_a", "trtGroup", roster["trtGroup"])
    arm = roster["trtGroup"].astype("string").str.strip()
    patients = pd.DataFrame({
        "patient_id": patient_ids(roster["PtID"]), "dataset": DATASET, "source_id": roster["PtID"].str.strip(),
        "arm": arm, "status": roster["PtStatus"],
        "randomization_ts": parse_datetimes(roster["RandDt"], ISO)[0],
        "enrollment_ts": parse_datetimes(roster["EnrollDt"], ISO)[0],
        "loop": arm.map({"CLC": "closed_after_randomization", "SAP": "open"}).fillna("unknown"),
        "cgm_device": "Dexcom G6",
        "dates": "de-identified (shifted); weekday not meaningful",
    }).drop_duplicates("patient_id")
    if args.get("screening"):
        s = read_table(args["screening"], ["PtID"], optional=["AgeAtEnrollment", "Gender", "InsModPump",
                                                              "InsModInjections", "PumpType", "DiagAge"])
        s = s.drop_duplicates("PtID")
        pump = to_number(s["InsModPump"]) == 1
        therapy = pd.Series("unknown", index=s.index).mask(to_number(s["InsModInjections"]) == 1, "mdi").mask(pump, "pump")
        patients = patients.merge(pd.DataFrame({
            "patient_id": patient_ids(s["PtID"]), "age": to_number(s["AgeAtEnrollment"]), "sex": s["Gender"],
            "diagnosis_age": to_number(s["DiagAge"]), "therapy": therapy, "baseline_pump": s["PumpType"],
        }), on="patient_id", how="left")
    if args.get("hba1c"):
        h = read_table(args["hba1c"], ["PtID", "HbA1cTestRes"], optional=["Visit", "HbA1cTestDt"])
        h["value"], h["date"] = to_number(h["HbA1cTestRes"]), parse_datetimes(h["HbA1cTestDt"], ISO)[0]
        h = h.dropna(subset=["value"]).sort_values("date").drop_duplicates("PtID")
        patients = patients.merge(pd.DataFrame({"patient_id": patient_ids(h["PtID"]), "hba1c": h["value"]}),
                                  on="patient_id", how="left")
    clc = patients["arm"] == "CLC"
    patients["insulin_devices"] = patients.get("baseline_pump", pd.Series(pd.NA, index=patients.index)).astype(
        "string").fillna("")
    patients.loc[clc, "insulin_devices"] = (patients.loc[clc, "insulin_devices"] +
                                            " -> t:slim X2 Control-IQ (after randomisation)").str.strip(" ->")
    report.payload = patients
