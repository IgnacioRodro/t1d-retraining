"""REPLACE-BG (JAEB, protocol H) -> common schema.

Facts used (from the ReadMe glossaries extracted by script 01):

- Times are given as days from enrolment plus a device clock time; dates are
  de-identified. Timestamps are built on a synthetic base date (2000-01-01 =
  enrolment day), so time of day is real but weekdays are meaningless.
- HDeviceCGM: Dexcom G4 readings in mg/dL; RecordType separates CGM readings
  from calibrations. Only RecordType == "CGM" is kept.
- HDeviceBolus: Normal (units), Extended (units over Duration), InjValue for
  injected insulin (with the insulin name). The glossary gives Duration in
  milliseconds for Tidepool uploads and minutes for Diasend uploads; that
  documented rule is applied and the magnitudes are only checked
  (``common.duration_minutes``, D33).
- HDeviceBasal: Rate (U/h) for Duration (milliseconds per the glossary, same
  per-upload check); suspends have no rate; temporary basals given as a
  percentage use Percnt x SuprRate.
- HDeviceWizard: CarbInput is carbohydrate in grams (the glossary says "mg",
  but values such as 25 or 45 are grams) for most patients, but some pumps
  were set to carbohydrate exchanges (D57). The wizard tells which through
  r = RecommendedCarb x InsulinCarbRatio / CarbInput (RecommendedCarb = units
  to cover the carbohydrates): r = 1 in grams mode (ratio in g/U); r = 15 for
  exchanges with the ratio exported in g/U; RecommendedCarb = CarbInput x
  ratio for exchanges with the ratio in U/exchange. Each record takes the mode
  its numbers fit (within 15 %); undecided records take the patient's
  majority. A patient with no decided record at all (missing or inconsistent
  wizard fields) is in exchanges when the median entry is <= 12, as for
  HUPA-UCM (D28). Exchanges are converted with 15 g each (US carbohydrate
  choice). patients.carb_input_unit, carb_unit_rule and the fit counts
  document the decision.
- Patients: HPtRoster (arm, age, randomisation day), HScreening (sex,
  diagnosis age), HLocalHbA1c, HDeviceUploads (device models).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.pipeline import Task
from .common import (PartWriter, TaskReport, daily_timedelta, duration_minutes, find_one, first_valid,
                     numeric_timedelta, read_table, to_number)

DATASET = "replace_bg"
PREFIX = "RBG-"
BASE_DATE = pd.Timestamp("2000-01-01")
CGM_SOURCE = "dexcom_g4"
CARB_EXCHANGE_G = 15.0
CARB_FIT_TOLERANCE = 0.15
LONG_ACTING = ["lantus", "levemir", "tresiba", "toujeo", "basaglar", "glargine", "detemir", "degludec", "nph"]


def patient_ids(values: pd.Series) -> pd.Series:
    return PREFIX + values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def device_time(frame: pd.DataFrame, report: TaskReport | None = None, source: str = "",
                days: str = "DeviceDtTmDaysFromEnroll", clock: str = "DeviceTm") -> pd.Series:
    return BASE_DATE + daily_timedelta(frame[days], frame[clock], report, source)


def plan(root: Path, options: dict) -> list[Task]:
    files = {
        "cgm": find_one(root, "HDeviceCGM.txt"),
        "bolus": find_one(root, "HDeviceBolus.txt"),
        "basal": find_one(root, "HDeviceBasal.txt"),
        "wizard": find_one(root, "HDeviceWizard.txt"),
        "uploads": find_one(root, "HDeviceUploads.txt"),
        "roster": find_one(root, "HPtRoster.txt"),
        "screening": find_one(root, "HScreening.txt", required=False),
        "hba1c": find_one(root, "HLocalHbA1c.txt", required=False),
    }
    size = {k: (v.stat().st_size if v else 0) for k, v in files.items()}
    return [
        Task(DATASET, "cgm", read_cgm, {"path": files["cgm"]}, size["cgm"]),
        Task(DATASET, "bolus", read_bolus, {"path": files["bolus"], "uploads": files["uploads"]}, size["bolus"]),
        Task(DATASET, "basal", read_basal, {"path": files["basal"], "uploads": files["uploads"]}, size["basal"]),
        Task(DATASET, "wizard_carbs", read_wizard, {"path": files["wizard"]}, size["wizard"]),
        Task(DATASET, "patients", read_patients, files, 1.0),
    ]


def read_cgm(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    columns = ["PtID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "RecordType", "GlucoseValue"]
    for chunk in read_table(args["path"], columns, chunksize=options["chunk_rows"], max_chunks=options.get("max_chunks")):
        report.rows_read += len(chunk)
        report.count("HDeviceCGM", "RecordType", chunk["RecordType"])
        chunk = chunk[chunk["RecordType"].str.strip().str.upper() == "CGM"]
        writer.add("cgm", pd.DataFrame({
            "patient_id": patient_ids(chunk["PtID"]),
            "ts": device_time(chunk, report, "HDeviceCGM"),
            "glucose_mgdl": to_number(chunk["GlucoseValue"]),
            "record_type": "cgm",
            "source": CGM_SOURCE,
        }))


def upload_sources(path: Path) -> pd.Series:
    """Upload RecID -> data source ('Tidepool', 'Diasend', ...)."""
    uploads = read_table(path, ["RecID"], optional=["DataSource"])
    return uploads.set_index(uploads["RecID"].str.strip())["DataSource"].astype("string")


def read_bolus(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    frame = read_table(args["path"], ["PtID", "ParentHDeviceUploadsID", "DeviceDtTmDaysFromEnroll", "DeviceTm"],
                       optional=["BolusType", "InjValue", "Insulin", "Normal", "Extended", "Duration"])
    report.rows_read += len(frame)
    report.count("HDeviceBolus", "BolusType", frame["BolusType"])
    report.count("HDeviceBolus", "Insulin", frame["Insulin"])
    sources = upload_sources(args["uploads"])
    source = frame["ParentHDeviceUploadsID"].str.strip().map(sources)
    report.count("HDeviceBolus", "upload DataSource", source)
    pid, ts = patient_ids(frame["PtID"]), device_time(frame, report, "HDeviceBolus")
    normal, extended, injected = to_number(frame["Normal"]), to_number(frame["Extended"]), to_number(frame["InjValue"])
    long_acting = frame["Insulin"].astype("string").str.lower().str.contains("|".join(LONG_ACTING), na=False)
    parts = [
        pd.DataFrame({"patient_id": pid, "ts": ts, "kind": "bolus", "units": normal})[normal > 0],
        pd.DataFrame({"patient_id": pid, "ts": ts, "kind": "bolus_extended", "units": extended,
                      "duration_min": duration_minutes(frame["Duration"], frame["ParentHDeviceUploadsID"], source,
                                                       report, "HDeviceBolus")})[extended > 0],
        pd.DataFrame({"patient_id": pid, "ts": ts, "kind": np.where(long_acting, "basal_injection", "bolus"),
                      "units": injected})[injected > 0],
    ]
    writer.add("insulin", pd.concat(parts, ignore_index=True))


def read_basal(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    frame = read_table(args["path"], ["PtID", "ParentHDeviceUploadsID", "DeviceDtTmDaysFromEnroll", "DeviceTm"],
                       optional=["BasalType", "Duration", "ExpectedDuration", "Percnt", "Rate", "SuprRate"])
    report.rows_read += len(frame)
    report.count("HDeviceBasal", "BasalType", frame["BasalType"])
    sources = upload_sources(args["uploads"])
    source = frame["ParentHDeviceUploadsID"].str.strip().map(sources)
    report.count("HDeviceBasal", "upload DataSource", source)
    rate = to_number(frame["Rate"])
    percent = to_number(frame["Percnt"])
    percent = percent.where(percent <= 1.5, percent / 100.0)  # fraction (Tidepool) or percentage
    rate = first_valid(rate, percent * to_number(frame["SuprRate"]))
    suspended = frame["BasalType"].astype("string").str.lower().str.contains("suspend", na=False)
    rate = rate.mask(suspended & rate.isna(), 0.0)
    upload = frame["ParentHDeviceUploadsID"]
    duration = first_valid(duration_minutes(frame["Duration"], upload, source, report, "HDeviceBasal"),
                           duration_minutes(frame["ExpectedDuration"], upload, source, report, "HDeviceBasal expected"))
    report.tally("HDeviceBasal", "rate", "missing after percent/suspend rules", int(rate.isna().sum()))
    writer.add("insulin", pd.DataFrame({
        "patient_id": patient_ids(frame["PtID"]), "ts": device_time(frame, report, "HDeviceBasal"), "kind": "basal_rate",
        "rate_u_per_h": rate, "duration_min": duration,
    })[rate.notna()])


def carb_input_mode(carbs: pd.Series, ratio: pd.Series, recommended: pd.Series) -> pd.Series:
    """"grams", "exchanges" or None (undecided) for each wizard record, from the wizard's own arithmetic."""
    ok = (carbs > 0) & (ratio > 0) & (recommended > 0)
    r = recommended * ratio / carbs
    grams = (r - 1).abs() <= CARB_FIT_TOLERANCE
    exchanges = ((r / CARB_EXCHANGE_G - 1).abs() <= CARB_FIT_TOLERANCE) | \
        (((recommended - carbs * ratio).abs() / recommended) <= CARB_FIT_TOLERANCE)
    mode = pd.Series(None, index=carbs.index, dtype=object)
    mode[ok & grams & ~exchanges] = "grams"
    mode[ok & exchanges & ~grams] = "exchanges"
    return mode


def read_wizard(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    frame = read_table(args["path"], ["PtID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "CarbInput"],
                       optional=["InsulinCarbRatio", "RecommendedCarb"])
    report.rows_read += len(frame)
    carbs = to_number(frame["CarbInput"])
    report.tally("HDeviceWizard", "CarbInput", "> 0", int((carbs > 0).sum()))
    report.tally("HDeviceWizard", "CarbInput", "0 or missing", int((~(carbs > 0)).sum()))
    patients = patient_ids(frame["PtID"])
    missing = pd.Series(np.nan, index=frame.index)
    mode = carb_input_mode(carbs, to_number(frame.get("InsulinCarbRatio", missing)),
                           to_number(frame.get("RecommendedCarb", missing)))
    for patient_id, records in mode.groupby(patients):
        n_grams, n_exchanges = int((records == "grams").sum()), int((records == "exchanges").sum())
        positive = carbs[records.index][carbs[records.index] > 0]
        if n_grams + n_exchanges:
            majority, rule = ("exchanges" if n_exchanges > n_grams else "grams"), "wizard arithmetic"
        else:
            small = len(positive) > 0 and float(positive.median()) <= 12
            majority, rule = ("exchanges" if small else "grams"), "median entry (no consistent wizard record)"
        mode[records.index[records.isna()]] = majority
        report.set_meta(patient_id, carb_input_unit=majority, carb_unit_rule=rule, carb_records_fit_grams=n_grams,
                        carb_records_fit_exchanges=n_exchanges,
                        carb_entry_median_raw=float(positive.median()) if len(positive) else None)
    exchanges = (mode == "exchanges") & (carbs > 0)
    report.tally("HDeviceWizard", "CarbInput", "exchanges converted to g (x15)", int(exchanges.sum()))
    grams = carbs.where(~exchanges, carbs * CARB_EXCHANGE_G)
    writer.add("carbs", pd.DataFrame({"patient_id": patients, "ts": device_time(frame, report, "HDeviceWizard"),
                                      "carbs_g": grams})[carbs > 0])


def read_patients(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    roster = read_table(args["roster"], ["PtID"], optional=["TrtGroup", "AgeAsOfEnrollDt", "RandDtDaysAfterEnroll",
                                                            "PtStatus"])
    patients = pd.DataFrame({
        "patient_id": patient_ids(roster["PtID"]),
        "dataset": DATASET,
        "source_id": roster["PtID"].str.strip(),
        "age": to_number(roster["AgeAsOfEnrollDt"]),
        "arm": roster["TrtGroup"],
        "status": roster["PtStatus"],
        "randomization_ts": BASE_DATE + numeric_timedelta(roster["RandDtDaysAfterEnroll"], "D", -3650, 3650),
        "therapy": "pump",
        "loop": "open",
        "cgm_device": "Dexcom G4 Platinum",
        "dates": "synthetic (2000-01-01 = enrolment day); weekday not meaningful",
    }).drop_duplicates("patient_id")
    if args.get("screening"):
        screening = read_table(args["screening"], ["PtID"], optional=["Gender", "DiagAge", "Weight", "Height"])
        screening = screening.drop_duplicates("PtID").assign(patient_id=lambda d: patient_ids(d["PtID"]))
        patients = patients.merge(pd.DataFrame({
            "patient_id": screening["patient_id"], "sex": screening["Gender"],
            "diagnosis_age": to_number(screening["DiagAge"]), "weight_kg": to_number(screening["Weight"]),
            "height_cm": to_number(screening["Height"]),
        }), on="patient_id", how="left")
    if args.get("hba1c"):
        hba1c = read_table(args["hba1c"], ["PtID", "HbA1cTestRes"], optional=["Visit", "HbA1cTestDtDaysAfterEnroll"])
        hba1c["value"] = to_number(hba1c["HbA1cTestRes"])
        hba1c["day"] = to_number(hba1c["HbA1cTestDtDaysAfterEnroll"])
        hba1c = hba1c.dropna(subset=["value"]).sort_values("day").drop_duplicates("PtID")  # earliest test
        patients = patients.merge(pd.DataFrame({"patient_id": patient_ids(hba1c["PtID"]), "hba1c": hba1c["value"]}),
                                  on="patient_id", how="left")
    uploads = read_table(args["uploads"], ["PtID"], optional=["DeviceModel", "DeviceType", "DataSource"])
    report.count("HDeviceUploads", "DeviceType", uploads["DeviceType"])
    report.count("HDeviceUploads", "DataSource", uploads["DataSource"])
    pumps = uploads[uploads["DeviceType"].astype("string").str.contains("insulin-pump", na=False)]
    models = pumps.groupby(patient_ids(pumps["PtID"]))["DeviceModel"].agg(
        lambda s: "+".join(sorted(set(s.dropna().astype(str)))))
    patients["insulin_devices"] = patients["patient_id"].map(models)
    report.payload = patients
