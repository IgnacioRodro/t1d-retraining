"""Common schema every dataset is converted to (script 02).

Tables (one parquet file per patient and table):

cgm        patient_id, ts, glucose_mgdl, record_type, source
           record_type: "cgm" (automatic sensor reading) or "scan" (FreeStyle Libre scan)
           source: sensor family (e.g. "dexcom_g6", "freestyle_libre"); a patient may have several
insulin    patient_id, ts, kind, units, rate_u_per_h, duration_min
           kind:
             bolus            instantaneous bolus, ``units``
             bolus_extended   extended part of a bolus, ``units`` over ``duration_min``
             basal_rate       pump basal segment, ``rate_u_per_h`` for ``duration_min``
             basal_injection  long-acting insulin injection, ``units``
             basal_5min       basal units delivered in the 5 minutes ending at ``ts``
             total_5min       basal + bolus units delivered in the 5 minutes ending at ``ts``
carbs      patient_id, ts, carbs_g
wearables  patient_id, ts, interval_min, hr_bpm, steps, kcal
           values aggregated over the ``interval_min`` minutes ending at ``ts``
sleep      patient_id, start, end, stage

Timestamps are naive local clock times (synthetic dates where the dataset gives
only days from enrolment). Glucose is always mg/dL. Absent events are absent
rows: nothing is zero-filled or interpolated at this stage; cleaning and the
choice of one CGM source per period belong to script 03.
"""

from __future__ import annotations

import pandas as pd

MGDL_PER_MMOL = 18.016

TABLES: dict[str, dict[str, str]] = {
    "cgm": {"patient_id": "string", "ts": "datetime64[ns]", "glucose_mgdl": "float32", "record_type": "string",
            "source": "string"},
    "insulin": {"patient_id": "string", "ts": "datetime64[ns]", "kind": "string", "units": "float32",
                "rate_u_per_h": "float32", "duration_min": "float32"},
    "carbs": {"patient_id": "string", "ts": "datetime64[ns]", "carbs_g": "float32"},
    "wearables": {"patient_id": "string", "ts": "datetime64[ns]", "interval_min": "float32", "hr_bpm": "float32",
                  "steps": "float32", "kcal": "float32"},
    "sleep": {"patient_id": "string", "start": "datetime64[ns]", "end": "datetime64[ns]", "stage": "string"},
}
SORT_KEYS = {"cgm": ["ts", "source", "record_type"], "insulin": ["ts", "kind"], "carbs": ["ts"], "wearables": ["ts"], "sleep": ["start"]}
INSULIN_KINDS = {"bolus", "bolus_extended", "basal_rate", "basal_injection", "basal_5min", "total_5min"}

PATIENT_COLUMNS = [
    "patient_id", "dataset", "source_id", "age", "sex", "hba1c", "therapy", "loop", "arm", "cgm_device",
    "insulin_devices", "window_start", "window_end", "notes",
]

PLAUSIBLE_GLUCOSE = (20.0, 600.0)  # mg/dL; values outside are dropped


def conform(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Columns, types, order, duplicates and plausibility for one table."""
    spec = TABLES[table]
    frame = frame.copy()
    for column in spec:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[list(spec)]
    for column, dtype in spec.items():
        if dtype.startswith("datetime"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        elif dtype == "string":
            frame[column] = frame[column].astype("string")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(dtype)
    time_column = "start" if table == "sleep" else "ts"
    frame = frame.dropna(subset=[time_column])
    if table == "cgm":
        low, high = PLAUSIBLE_GLUCOSE
        frame = frame[frame["glucose_mgdl"].between(low, high)]
    if table == "insulin":
        unknown = set(frame["kind"].dropna().unique()) - INSULIN_KINDS
        if unknown:
            raise ValueError(f"Unknown insulin kinds: {unknown}")
    frame = frame.sort_values(SORT_KEYS[table], kind="stable")
    if table == "cgm":
        # the same reading uploaded twice (overlapping exports or uploads): keep the first
        frame = frame.drop_duplicates(subset=["ts", "record_type", "source"])
    else:
        frame = frame.drop_duplicates()
    return frame.reset_index(drop=True)


def empty(table: str) -> pd.DataFrame:
    """Empty frame with the right columns."""
    return conform(table, pd.DataFrame(columns=list(TABLES[table])))
