"""HUPA-UCM Diabetes Dataset (Mendeley, CC BY 4.0) -> common schema.

Hybrid use of the two folders (decision D27):

- Glucose comes from Raw_Data, at the sensor's native rate (the preprocessed
  5-min grid is interpolated and has no gaps):
    * free_style_sensor/*.csv  FreeStyle Libre exports in two layouts: the 2018-19
      Abbott/Absys export (header "ID;Hora;Tipo de registro;Histórico glucosa...",
      ';' or tab, times like 2018/06/13 17:19) and the LibreView export
      ("Dispositivo,Número de serial,Sello de tiempo del dispositivo,...", ',' or
      ';', report lines before the header, day-first times; one file uses CR-only
      line ends). Tipo de registro 0 = automatic 15-min history -> "cgm";
      1 = scan -> "scan". Overlapping exports are de-duplicated.
    * medtronic_insulin_pump/*.csv  CareLink export with a "Sensor" section whose
      "Sensor Glucose (mg/dL)" column holds Enlite readings (only one patient).
    * dexcom/*.csv  Dexcom Clarity export (Spanish), events "Niveles estimados de
      glucosa"; "Nivel bajo"/"Nivel alto" are set to 40/400 mg/dL.
- Insulin and carbohydrates come from Preprocessed/<patient>.csv, which gathers
  the pump, pen and Libre notes of every patient on one 5-min grid:
  basal_rate = basal units in the 5-min slot (e.g. 0.0917 = 1.1 U/h), so it is
  stored as kind "basal_5min"; bolus_volume_delivered -> "bolus";
  carb_input is in 10 g exchanges for most patients but in grams for some
  (Roche-pump patients), so the unit is inferred per patient (median entry
  > 12 -> grams) and reported. A patient whose basal column is all zero has no
  basal record (not a zero basal).
- Fitbit (Raw_Data/<patient>/fitbit): heart rate every few seconds, steps and
  calories per minute, one file per day (date in the file name) -> 5-min
  intervals ending at ts (mean HR, summed steps and kcal). Sleep stages from the
  *_night.csv and *_nap_#.csv files.
- No demographics are published; therapy is inferred from the device folders.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

from ..data.pipeline import Task
from .common import PartWriter, TaskReport, censored_glucose, numeric_timedelta, parse_datetimes, read_text, to_number

DATASET = "hupa_ucm"
FITBIT_DAYS_PER_TASK = 40
EXCHANGE_GRAMS = 10.0
LIBRE_FORMATS = ["%Y/%m/%d %H:%M", "%d/%m/%y %H:%M", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M",
                 "%d-%m-%y %H:%M", "%d-%m-%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "ISO8601"]
DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


def plan(root: Path, options: dict) -> list[Task]:
    raw = _single_dir(root, "Raw_Data")
    pre = _single_dir(root, "Preprocessed")
    patients = sorted(p.name for p in raw.iterdir() if p.is_dir() and re.fullmatch(r"HUPA\d{4}P", p.name))
    tasks = [Task(DATASET, "patients", read_patients, {"raw": raw, "patients": patients}, 1.0)]
    for patient in patients:
        folder = raw / patient
        sensor_files = sorted((folder).glob("free_style_sensor/*.csv")) + sorted(folder.glob("dexcom/*.csv")) + \
            sorted(folder.glob("medtronic_insulin_pump/*.csv"))
        size = sum(p.stat().st_size for p in sensor_files)
        tasks.append(Task(DATASET, f"sensor_{patient}", read_sensors, {"patient": patient, "folder": folder}, size))
        pre_file = pre / f"{patient}.csv"
        if pre_file.exists():
            tasks.append(Task(DATASET, f"preprocessed_{patient}", read_preprocessed,
                              {"patient": patient, "path": pre_file}, pre_file.stat().st_size))
        fitbit = folder / "fitbit"
        if fitbit.exists():
            days = sorted({m.group(1) for p in fitbit.glob("*.csv") if not _is_sleep(p)
                           for m in [DATE_IN_NAME.search(p.name)] if m})
            for start in range(0, len(days), FITBIT_DAYS_PER_TASK):
                batch = days[start:start + FITBIT_DAYS_PER_TASK]
                tasks.append(Task(DATASET, f"fitbit_{patient}_{start // FITBIT_DAYS_PER_TASK:03d}", read_fitbit,
                                  {"patient": patient, "folder": fitbit, "days": batch}, 4e6 * len(batch)))
            tasks.append(Task(DATASET, f"sleep_{patient}", read_sleep, {"patient": patient, "folder": fitbit}, 1e4))
    return tasks


def _single_dir(root: Path, name: str) -> Path:
    if (root / name).is_dir():
        return root / name
    matches = sorted((p for p in root.rglob(name) if p.is_dir()), key=lambda p: len(p.parts))
    if not matches:
        raise FileNotFoundError(f"Folder '{name}' not found under {root}")
    return matches[0]


def _is_sleep(path: Path) -> bool:
    return "_sleep_" in path.name


# ----------------------------------------------------------------------------- glucose


def read_libre(path: Path, report: TaskReport) -> pd.DataFrame:
    """One FreeStyle Libre export (either layout) -> rows of the cgm table (without patient_id)."""
    lines = read_text(path).split("\n")
    header_index = None
    for index, line in enumerate(lines[:20]):
        first = re.split(r"[;,\t]", line.strip(), maxsplit=1)[0].strip().strip('"')
        if first in {"ID", "Dispositivo", "Device"}:
            header_index = index
            break
    if header_index is None:
        report.notes.append(f"{path.name}: no Libre header found")
        return pd.DataFrame()
    header = lines[header_index]
    delimiter = max([";", ",", "\t"], key=header.count)
    frame = delimited_lines(lines[header_index:], delimiter)
    if frame.empty:
        report.tally("free_style_sensor", "file", "empty", 1)
        return pd.DataFrame()
    time_col = _column(frame, ["Hora", "Sello de tiempo del dispositivo", "Device Timestamp"])
    type_col = _column(frame, ["Tipo de registro", "Record Type"])
    hist_col = _column(frame, ["Histórico glucosa (mg/dL)", "Historial de glucosa mg/dL", "Historic Glucose mg/dL"],
                       contains="hist")
    scan_col = _column(frame, ["Glucosa leída (mg/dL)", "Escaneo de glucosa mg/dL", "Scan Glucose mg/dL"],
                       contains=("leída", "escaneo", "scan"))
    ts, fmt = parse_datetimes(frame[time_col], LIBRE_FORMATS)
    report.tally("free_style_sensor", "time format", fmt, len(frame))
    report.tally("free_style_sensor", "time", "unparsed", int((ts.isna() & frame[time_col].notna()).sum()))
    record_type = frame[type_col].astype("string").str.strip()
    report.count("free_style_sensor", "Tipo de registro", record_type)
    parts = []
    for code, column, label in [("0", hist_col, "cgm"), ("1", scan_col, "scan")]:
        if column is None:
            continue
        rows = record_type == code
        values, low, high = censored_glucose(frame.loc[rows, column], ["LO", "Bajo"], ["HI", "Alto"], 40.0, 500.0)
        report.tally("free_style_sensor", f"{label} censored", "low -> 40", low)
        report.tally("free_style_sensor", f"{label} censored", "high -> 500", high)
        parts.append(pd.DataFrame({"ts": ts[rows], "glucose_mgdl": values, "record_type": label,
                                   "source": "freestyle_libre"}))
    report.rows_read += len(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def delimited_lines(lines: list[str], delimiter: str) -> pd.DataFrame:
    """Header + rows -> frame of strings. Rows longer or shorter than the header are cut or padded.

    Some Libre exports carry two extra fields on part of the rows, which
    pandas would treat as bad lines and drop.
    """
    rows = [r for r in csv.reader(lines, delimiter=delimiter) if any(field.strip() for field in r)]
    if not rows:
        return pd.DataFrame()
    header = [h.strip().lstrip("\ufeff") for h in rows[0]]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in rows[1:]]
    frame = pd.DataFrame(body, columns=header, dtype="string")
    return frame.replace({"": pd.NA})


def _column(frame: pd.DataFrame, names: list[str], contains: str | tuple[str, ...] | None = None) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    if contains:
        keys = (contains,) if isinstance(contains, str) else contains
        for column in frame.columns:
            if any(k in column.lower() for k in keys) and "gluc" in column.lower():
                return column
    return None


def read_carelink_sensor(path: Path, report: TaskReport) -> pd.DataFrame:
    """Sensor-glucose rows of a Medtronic CareLink export (any section)."""
    lines = read_text(path).split("\n")
    headers = [i for i, line in enumerate(lines) if line.startswith("Index;")]
    parts = []
    for k, start in enumerate(headers):
        end = headers[k + 1] - 1 if k + 1 < len(headers) else len(lines)
        section = [l for l in lines[start:end] if not l.startswith("-------")]
        frame = pd.read_csv(io.StringIO("\n".join(section)), sep=";", dtype=str, index_col=False)
        report.rows_read += len(frame)
        if "Sensor Glucose (mg/dL)" not in frame or frame["Sensor Glucose (mg/dL)"].isna().all():
            continue
        frame = frame[frame["Sensor Glucose (mg/dL)"].notna()]
        ts, fmt = parse_datetimes(frame["Date"].str.strip() + " " + frame["Time"].str.strip(),
                                  ["%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "ISO8601"])
        report.tally("medtronic_insulin_pump", "time format", fmt, len(frame))
        parts.append(pd.DataFrame({"ts": ts, "glucose_mgdl": to_number(frame["Sensor Glucose (mg/dL)"]),
                                   "record_type": "cgm", "source": "medtronic_enlite"}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def read_dexcom(path: Path, report: TaskReport) -> pd.DataFrame:
    """Glucose events of a Dexcom Clarity export (Spanish or English headers)."""
    frame = pd.read_csv(io.StringIO(read_text(path)), sep=None, engine="python", dtype=str)
    frame.columns = [str(c).strip() for c in frame.columns]
    report.rows_read += len(frame)
    event = _column(frame, ["Tipo de evento", "Event Type"])
    time_col = next(c for c in frame.columns if c.lower().startswith(("marca temporal", "timestamp")))
    value_col = next(c for c in frame.columns if c.lower().startswith(("nivel de glucosa", "glucose value")))
    report.count("dexcom", "Tipo de evento", frame[event])
    egv = frame[frame[event].astype("string").str.strip().isin(["Niveles estimados de glucosa", "EGV"])]
    values, low, high = censored_glucose(egv[value_col], ["Nivel bajo", "Low"], ["Nivel alto", "High"], 40.0, 400.0)
    report.tally("dexcom", "censored", "low -> 40", low)
    report.tally("dexcom", "censored", "high -> 400", high)
    ts, _ = parse_datetimes(egv[time_col], ["ISO8601"])
    return pd.DataFrame({"ts": ts, "glucose_mgdl": values, "record_type": "cgm", "source": "dexcom_g6"})


def read_sensors(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    patient, folder = args["patient"], args["folder"]
    frames = []
    for path in sorted(folder.glob("free_style_sensor/*.csv")):
        frames.append(read_libre(path, report))
    for path in sorted(folder.glob("medtronic_insulin_pump/*.csv")):
        frames.append(read_carelink_sensor(path, report))
    for path in sorted(folder.glob("dexcom/*.csv")):
        frames.append(read_dexcom(path, report))
    frames = [f for f in frames if not f.empty]
    if frames:
        cgm = pd.concat(frames, ignore_index=True).assign(patient_id=patient)
        writer.add("cgm", cgm)
        report.set_meta(patient, cgm_sources="+".join(sorted(cgm["source"].unique())))


# ----------------------------------------------------------------------------- insulin and carbs


def read_preprocessed(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    patient = args["patient"]
    frame = pd.read_csv(args["path"], sep=";", dtype=str)
    report.rows_read += len(frame)
    ts, _ = parse_datetimes(frame["time"], ["ISO8601"])
    basal = to_number(frame["basal_rate"])
    bolus = to_number(frame["bolus_volume_delivered"])
    carbs = to_number(frame["carb_input"])
    parts = [pd.DataFrame({"patient_id": patient, "ts": ts, "kind": "bolus", "units": bolus})[bolus > 0]]
    if (basal.fillna(0) > 0).any():
        parts.append(pd.DataFrame({"patient_id": patient, "ts": ts, "kind": "basal_5min", "units": basal})
                     [basal.notna()])
    writer.add("insulin", pd.concat(parts, ignore_index=True))
    positive = carbs[carbs > 0]
    unit = "none"
    if len(positive):
        unit = "grams" if positive.median() > 12 else "exchanges_10g"
        grams = carbs * (1.0 if unit == "grams" else EXCHANGE_GRAMS)
        writer.add("carbs", pd.DataFrame({"patient_id": patient, "ts": ts, "carbs_g": grams})[carbs > 0])
    report.tally("Preprocessed", "carb_input unit (patients)", unit, 1)
    report.set_meta(patient, carb_input_unit=unit, carb_input_median_raw=float(positive.median()) if len(positive)
                    else None, window_start=str(ts.min()), window_end=str(ts.max()),
                    basal_recorded="yes" if (basal.fillna(0) > 0).any() else "no")


# ----------------------------------------------------------------------------- Fitbit


def read_fitbit(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    patient, folder = args["patient"], args["folder"]
    frames = []
    for day in args["days"]:
        date = pd.Timestamp(day)
        heart = folder / f"{patient}_heart_{day}.csv"
        steps = folder / f"{patient}_steps_{day}.csv"
        cals = folder / f"{patient}_cals_{day}.csv"
        bins = []
        if heart.exists():
            h = pd.read_csv(heart, dtype=str)
            report.rows_read += len(h)
            t = date + pd.to_timedelta(h.iloc[:, 0].str.strip(), errors="coerce")
            bins.append(pd.DataFrame({"ts": _bin_end(t), "hr_bpm": to_number(h.iloc[:, 1])})
                        .groupby("ts")["hr_bpm"].mean())
        for path, name in [(steps, "steps"), (cals, "kcal")]:
            if path.exists():
                m = pd.read_csv(path, dtype=str)
                report.rows_read += len(m)
                t = date + pd.to_timedelta(m.iloc[:, 0].str.strip(), errors="coerce")
                bins.append(pd.DataFrame({"ts": _bin_end(t), name: to_number(m.iloc[:, 1])})
                            .groupby("ts")[name].sum(min_count=1))
        if bins:
            frames.append(pd.concat(bins, axis=1).reset_index())
    if frames:
        out = pd.concat(frames, ignore_index=True).assign(patient_id=patient, interval_min=5.0)
        writer.add("wearables", out)


def _bin_end(t: pd.Series) -> pd.Series:
    """End of the 5-min interval that contains each time."""
    return t.dt.floor("5min") + pd.Timedelta(minutes=5)


def read_sleep(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    patient, folder = args["patient"], args["folder"]
    frames = []
    for path in sorted(folder.glob(f"{patient}_sleep_*.csv")):
        if path.name.endswith("_summary.csv"):
            continue
        s = pd.read_csv(path, dtype=str)
        report.rows_read += len(s)
        start, _ = parse_datetimes(s["dateTime"], ["ISO8601"])
        seconds = to_number(s["Duration"])
        frames.append(pd.DataFrame({"start": start, "end": start + numeric_timedelta(seconds, "s", 0, 86400, report,
                                                                                      "fitbit sleep", "Duration"),
                                    "stage": s["State"].str.strip().str.lower()}))
        report.tally("fitbit sleep", "file kind", "nap" if "_nap_" in path.name else "night", 1)
    if frames:
        sleep = pd.concat(frames, ignore_index=True).assign(patient_id=patient)
        report.count("fitbit sleep", "State", sleep["stage"])
        writer.add("sleep", sleep)


# ----------------------------------------------------------------------------- patients


def read_patients(args: dict, writer: PartWriter, report: TaskReport, options: dict) -> None:
    rows = []
    for patient in args["patients"]:
        folder = args["raw"] / patient
        devices = sorted(p.name for p in folder.iterdir() if p.is_dir())
        pump = [d for d in devices if "pump" in d]
        cgm = ["FreeStyle Libre" if d == "free_style_sensor" else "Dexcom G6" if d == "dexcom" else None
               for d in devices]
        notes = []
        for path in folder.rglob("*.csv"):
            lower = path.name.lower()
            if "no_insulin" in lower:
                notes.append("pump export without insulin")
            if "empty" in lower:
                notes.append(f"empty export: {path.parent.name}")
        rows.append({
            "patient_id": patient, "dataset": DATASET, "source_id": patient,
            "therapy": "pump" if pump else "mdi",
            "loop": "open",
            "insulin_devices": "+".join(d.replace("_insulin_pump", "") for d in pump) or "pens",
            "cgm_device": "+".join(c for c in cgm if c),
            "device_folders": "+".join(devices),
            "notes": "; ".join(sorted(set(notes))),
            "dates": "real dates",
        })
    report.payload = pd.DataFrame(rows)
