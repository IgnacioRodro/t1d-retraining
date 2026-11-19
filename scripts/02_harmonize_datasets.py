"""02 - Harmonise the raw datasets into one common schema.

Reads every dataset folder of config/paths.yaml with its loader
(src/t1dretrain/io/<dataset>.py) and writes, per dataset and patient, the
tables cgm, insulin, carbs, wearables and sleep defined in
src/t1dretrain/data/schema.py, plus one patients table:

    <derived_data>/processed/<dataset>/<table>/<patient_id>.parquet
    <derived_data>/processed/<dataset>/patients.parquet

Nothing is cleaned, interpolated or resampled here; every reading keeps its
native time stamp and only units, clocks and names are unified. Cleaning,
the choice of one CGM source per period and valid-day rules belong to script 03
(the summaries here already use the best-covered CGM source of each day).

Parallel on all cores but two (rule R5). Phase A reads: each loader splits its
work into tasks (one big file read in chunks, one patient folder, one batch of
daily wearable files) and the tasks of all datasets share the worker pool,
largest first. Phase B merges: one job per patient concatenates its parts,
finishes derived fields (basal-rate durations), conforms the schema, writes the
parquet files and describes the patient.

DiaData is not harmonised (decisions D13 and D23: reserve only).

Outputs (results/02_harmonize_datasets/):
    dataset_summary.csv        one row per dataset: patients, days, valid days, modalities, sampling
    patient_summary.csv        one row per patient: follow-up, valid days, sampling, glucose, insulin, carbs, wearables, sleep
    patients_static.csv        the patients tables of every dataset (demographics, arm, therapy, loop, devices)
    table_rows.csv             rows written per dataset and table
    source_value_counts.csv    frequencies of the raw categories the loaders map (record types, units, time sources...)
    device_periods.csv         device use per patient and role, where the data name the device (BrisT1D)
    task_log.csv               every phase-A task and phase-B job: status, seconds, rows, errors
    glucose_histogram.csv      share of readings per 10 mg/dL bin and dataset
    sampling_intervals.csv     share of consecutive-reading intervals per bin and dataset
    diurnal_profile.csv        mean glucose, carb entries and boluses by hour of day and dataset
    wearable_diurnal.csv       mean heart rate, share of steps and share of time asleep by hour and dataset
    time_offsets.csv           Loop: UTC offsets observed from device clocks, per patient, against the roster offset
    checks_<name>.csv          loader-specific check tables (e.g. checks_t1d_uom_epochs_by_hour.csv)
    valid_days_per_patient.png
    modality_coverage.png
    sampling_intervals.png
    glucose_distribution.png
    events_per_day.png
    diurnal_check.png
    wearable_clock_check.png   heart rate and sleep by hour, and the sleep-log lag that best fits heart rate
    summary.md
    run_manifest.json

Usage:
    python scripts/02_harmonize_datasets.py
    python scripts/02_harmonize_datasets.py --datasets hupa_ucm t1d_uom
    python scripts/02_harmonize_datasets.py --max-chunks 2      # quick smoke test on big files
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_paths, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import (execute_task, list_patients, merge_meta, merge_patient,  # noqa: E402
                                      parts_dir, reset_dataset, value_counts_frame)
from t1dretrain.data.schema import TABLES  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.data.summary import GLUCOSE_BINS, INTERVAL_LABELS  # noqa: E402
from t1dretrain.io import LOADERS  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import (DATASET_COLORS, DATASET_LABELS, DATASET_LINESTYLES, DATASET_ORDER,  # noqa: E402
                                 SEQUENTIAL_BLUE, TEXT_SECONDARY, set_style)

SCRIPT_ID = "02_harmonize_datasets"
NOT_HARMONISED = {"diadata": "reserve only, never harmonised (D13, D23)"}
OVERRIDE_FROM_DATA = {"loop", "therapy", "insulin_devices", "cgm_device"}  # data-derived values win
MODALITIES = [
    ("CGM", lambda r: r["cgm_main_rows"] > 0),
    ("Libre scans", lambda r: r["scan_rows"] > 0),
    ("Bolus", lambda r: r["insulin_bolus_rows"] + r["insulin_bolus_extended_rows"] > 0),
    ("Basal", lambda r: r["insulin_basal_rate_rows"] + r["insulin_basal_5min_rows"] + r["insulin_basal_injection_rows"] > 0),
    ("Total insulin (5 min)", lambda r: r["insulin_total_5min_rows"] > 0),
    ("Carbs", lambda r: r["carb_rows"] > 0),
    ("Heart rate", lambda r: r.get("hr_days", 0) > 0),
    ("Steps", lambda r: r.get("steps_days", 0) > 0),
    ("Sleep", lambda r: r["sleep_rows"] > 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of datasets (default: all in paths.yaml)")
    parser.add_argument("--chunk-rows", type=int, default=2_000_000, help="rows per chunk for big files")
    parser.add_argument("--max-chunks", type=int, default=None, help="read at most this many chunks per big file (tests)")
    parser.add_argument("--keep-parts", action="store_true", help="keep the phase-A part files (debugging)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)  # stops if some files were not updated
    warnings.filterwarnings("ignore", message="Mean of empty slice")  # medians of empty columns are NaN on purpose
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()

    paths = load_paths(args.paths)
    derived = derived_data_dir(args.paths)
    interim = derived / "interim"
    coverage = float(config.get("valid_day_min_cgm_coverage", 0.70))
    min_valid = int(config.get("min_valid_days_core", 42))
    options = {"chunk_rows": args.chunk_rows, "max_chunks": args.max_chunks}

    wanted = args.datasets or [d for d in DATASET_ORDER if d in paths]
    status_rows, jobs = [], []
    for dataset in wanted:
        if dataset in NOT_HARMONISED:
            status_rows.append({"dataset": dataset, "status": "skipped", "detail": NOT_HARMONISED[dataset]})
            continue
        if dataset not in LOADERS:
            status_rows.append({"dataset": dataset, "status": "skipped", "detail": "no loader"})
            continue
        root = paths.get(dataset)
        if root is None or not root.exists():
            status_rows.append({"dataset": dataset, "status": "missing", "detail": f"folder not found: {root}"})
            continue
        reset_dataset(derived, interim, dataset)
        try:
            tasks = importlib.import_module(LOADERS[dataset]).plan(root, options)
        except Exception as error:  # noqa: BLE001
            status_rows.append({"dataset": dataset, "status": "plan_error", "detail": repr(error)})
            continue
        status_rows.append({"dataset": dataset, "status": "planned", "detail": f"{len(tasks)} tasks from {root}"})
        jobs += [(task, parts_dir(interim, dataset), options) for task in tasks]

    # ---------------------------------------------------------------- phase A: read
    jobs.sort(key=lambda job: job[0].cost, reverse=True)
    start = time.time()
    results = parallel_map(execute_task, jobs, resources.n_workers, label="read tasks")
    read_seconds = time.time() - start
    task_log = pd.DataFrame([{
        "phase": "read", "dataset": r.dataset, "task": r.name, "status": r.status, "seconds": r.seconds,
        "rows_read": r.rows_read, "rows_written": sum(r.rows_written.values()),
        "error": r.error.strip().splitlines()[-1] if r.error else "", "traceback": r.error, "warnings": r.warnings,
    } for r in results])
    meta = merge_meta(results)
    statics = {r.dataset: r.report.payload for r in results if r.report is not None and r.report.payload is not None}
    devices = pd.DataFrame([d for r in results if r.report is not None for d in r.report.devices])
    checks = pd.DataFrame([{"dataset": r.dataset, **c} for r in results if r.report is not None
                           for c in r.report.checks])

    # ---------------------------------------------------------------- phase B: merge per patient
    merge_jobs = []
    datasets_done = sorted({r.dataset for r in results}, key=DATASET_ORDER.index)
    for dataset in datasets_done:
        root = parts_dir(interim, dataset)
        for patient_id in list_patients(root):
            size = sum(p.stat().st_size for p in root.glob(f"*/{patient_id}/*.parquet"))
            merge_jobs.append({
                "dataset": dataset, "patient_id": patient_id, "module": LOADERS[dataset], "parts_root": root,
                "derived_root": derived, "meta": meta.get(dataset, {}).get(patient_id, {}),
                "coverage_threshold": coverage, "size": size,
            })
    merge_jobs.sort(key=lambda job: job["size"], reverse=True)
    start = time.time()
    merged = parallel_map(merge_patient, merge_jobs, resources.n_workers, label="patient merges")
    merge_seconds = time.time() - start
    task_log = pd.concat([task_log, pd.DataFrame([{
        "phase": "merge", "dataset": m.dataset, "task": m.patient_id, "status": "ok" if not m.error else "error",
        "seconds": m.seconds, "rows_read": np.nan, "rows_written": sum(m.rows.values()),
        "error": m.error.strip().splitlines()[-1] if m.error else "", "traceback": m.error, "warnings": m.warnings,
    } for m in merged])], ignore_index=True)

    ok = [m for m in merged if m.description is not None]
    merge_counts = pd.DataFrame([{"dataset": m.dataset, "source": f"phase B: {table}", "column": column,
                                  "value": value, "count": n}
                                 for m in ok for (table, column, value), n in m.counts.items()])
    time_offsets = pd.DataFrame([{"dataset": m.dataset, "patient_id": m.patient_id, **m.extra} for m in ok if m.extra])
    summary = pd.DataFrame([m.description.row for m in ok])
    table_rows = pd.DataFrame([{"dataset": m.dataset, "table": t, "patients": int(n > 0), "rows": n}
                               for m in ok for t, n in m.rows.items()])
    if not table_rows.empty:
        table_rows = table_rows.groupby(["dataset", "table"], as_index=False)[["patients", "rows"]].sum()

    # ---------------------------------------------------------------- patients tables
    static_frames = []
    for dataset in datasets_done:
        static = statics.get(dataset, pd.DataFrame(columns=["patient_id"])).copy()
        present = summary.loc[summary["dataset"] == dataset, ["patient_id", "first_ts", "last_ts"]] \
            if not summary.empty else pd.DataFrame(columns=["patient_id", "first_ts", "last_ts"])
        static = static.merge(present, on="patient_id", how="outer")
        static["dataset"] = dataset
        static["has_data"] = static["patient_id"].isin(present["patient_id"])
        extra = pd.DataFrame.from_dict(meta.get(dataset, {}), orient="index")
        if not extra.empty:
            extra.index.name = "patient_id"
            extra = extra.reset_index()
            static = static.merge(extra, on="patient_id", how="left", suffixes=("", "__data"))
            for column in [c for c in static.columns if c.endswith("__data")]:
                base = column[: -len("__data")]
                if base in OVERRIDE_FROM_DATA:
                    static[base] = static[column].where(static[column].notna(), static[base])
                else:
                    static[base] = static[base].where(static[base].notna(), static[column])
                static = static.drop(columns=column)
        if "window_start" in static:
            static["window_start"] = static["window_start"].where(static["window_start"].notna(), static["first_ts"])
            static["window_end"] = static["window_end"].where(static["window_end"].notna(), static["last_ts"])
        else:
            static["window_start"], static["window_end"] = static["first_ts"], static["last_ts"]
        static = static.drop(columns=["first_ts", "last_ts"])
        rows = summary[summary["dataset"] == dataset] if not summary.empty else summary
        clock = [c for c in rows.columns if c.startswith("cgm_time_") and rows[c].notna().any()]
        if clock:  # how local times were obtained (Loop, D41), needed by script 03
            static = static.merge(rows[["patient_id"] + clock], on="patient_id", how="left")
        store = ProcessedStore(dataset, root=derived)
        store.write_patients(static)
        if not devices.empty and dataset in set(devices["dataset"]):
            store.write_extra("device_periods", devices[devices["dataset"] == dataset])
        static_frames.append(static)
    patients_static = pd.concat(static_frames, ignore_index=True) if static_frames else pd.DataFrame()

    if not args.keep_parts:
        shutil.rmtree(interim / "02_parts", ignore_errors=True)

    # ---------------------------------------------------------------- tables for review
    if not summary.empty:
        info = patients_static[["dataset", "patient_id"] + [c for c in ["arm", "therapy", "loop"]
                                                             if c in patients_static]]
        summary = summary.merge(info, on=["dataset", "patient_id"], how="left")
        summary["eligible_min_valid_days"] = summary["valid_days"].fillna(0) >= min_valid
    dataset_summary = summarise_datasets(summary, patients_static, min_valid)
    value_counts = value_counts_frame(results)
    if not merge_counts.empty:
        merge_counts = merge_counts.groupby(["dataset", "source", "column", "value"], as_index=False)["count"].sum()
        value_counts = pd.concat([value_counts, merge_counts], ignore_index=True)
    if not time_offsets.empty and "timezone_offset_h" in patients_static:
        time_offsets = time_offsets.merge(patients_static[["dataset", "patient_id", "timezone_offset_h"]],
                                          on=["dataset", "patient_id"], how="left")
        time_offsets["main_minus_roster_h"] = time_offsets["offset_main_h"] - time_offsets["timezone_offset_h"]
    glucose_hist, interval_hist, diurnal = aggregate_histograms(ok)
    wearable_diurnal = aggregate_wearables(ok)

    out.csv(dataset_summary, "dataset_summary")
    out.csv(summary, "patient_summary")
    out.csv(patients_static, "patients_static")
    out.csv(table_rows, "table_rows")
    out.csv(value_counts, "source_value_counts")
    out.csv(devices if not devices.empty else pd.DataFrame(columns=["dataset", "patient_id", "role", "device",
                                                                    "first_ts", "last_ts", "rows"]),
            "device_periods")
    out.csv(task_log, "task_log")
    out.csv(glucose_hist, "glucose_histogram")
    out.csv(interval_hist, "sampling_intervals")
    out.csv(diurnal, "diurnal_profile")
    out.csv(wearable_diurnal, "wearable_diurnal")
    out.csv(time_offsets, "time_offsets")
    if not checks.empty:
        for name, rows in checks.groupby("check"):
            out.csv(rows.drop(columns="check").dropna(axis=1, how="all"), f"checks_{name}")

    if not summary.empty:
        out.figure(plot_valid_days(summary, min_valid), "valid_days_per_patient")
        out.figure(plot_modalities(summary), "modality_coverage")
        out.figure(plot_intervals(interval_hist), "sampling_intervals")
        out.figure(plot_glucose(glucose_hist), "glucose_distribution")
        out.figure(plot_events(summary), "events_per_day")
        out.figure(plot_diurnal(diurnal), "diurnal_check")
        if not wearable_diurnal.empty:
            out.figure(plot_wearable_clock(wearable_diurnal, summary), "wearable_clock_check")

    status = pd.DataFrame(status_rows)
    out.text(write_summary(status, dataset_summary, summary, task_log, value_counts, patients_static, read_seconds,
                           merge_seconds, resources, derived, min_valid, time_offsets, release), "summary.md")
    out.manifest(settings={
        "datasets": wanted, "derived_data": str(derived), "chunk_rows": args.chunk_rows, "max_chunks": args.max_chunks,
        "valid_day_min_cgm_coverage": coverage, "min_valid_days_core": min_valid, "tables": list(TABLES),
    }, notes="Phase A read tasks and phase B patient merges run in parallel on n_workers processes.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ aggregation


def summarise_datasets(summary: pd.DataFrame, patients: pd.DataFrame, min_valid: int) -> pd.DataFrame:
    rows = []
    for dataset in [d for d in DATASET_ORDER if not summary.empty and d in set(summary["dataset"])]:
        s = summary[summary["dataset"] == dataset]
        p = patients[patients["dataset"] == dataset] if not patients.empty else pd.DataFrame()
        with_cgm = s[s["cgm_main_rows"] > 0]
        rows.append({
            "dataset": dataset,
            "patients_listed": int(len(p)),
            "patients_with_data": int(len(s)),
            "patients_with_cgm": int(len(with_cgm)),
            "cgm_readings": int(s["cgm_main_rows"].sum()),
            "median_span_days": round(float(with_cgm["span_days"].median()), 1) if len(with_cgm) else np.nan,
            "median_valid_days": float(with_cgm["valid_days"].median()) if len(with_cgm) else np.nan,
            "total_valid_days": int(with_cgm["valid_days"].sum()) if len(with_cgm) else 0,
            f"patients_valid_ge_{min_valid}": int((with_cgm["valid_days"] >= min_valid).sum()),
            "patients_valid_ge_98": int((with_cgm["valid_days"] >= 98).sum()),
            "patients_valid_ge_180": int((with_cgm["valid_days"] >= 180).sum()),
            "median_longest_valid_run": float(with_cgm["longest_valid_run_days"].median()) if len(with_cgm) else np.nan,
            "modal_interval_min": float(with_cgm["modal_interval_min"].mode().iloc[0]) if len(with_cgm) else np.nan,
            "median_share_intervals_lt_2min": float(with_cgm["share_intervals_lt_2min"].median()) if len(with_cgm) else np.nan,
            "median_glucose_mean": float(with_cgm["glucose_mean"].median()) if len(with_cgm) else np.nan,
            "median_tir": float(with_cgm["tir_70_180"].median()) if len(with_cgm) else np.nan,
            "share_with_bolus": round(float(((s["insulin_bolus_rows"] + s["insulin_bolus_extended_rows"]) > 0).mean()), 3),
            "share_with_basal": round(float(((s["insulin_basal_rate_rows"] + s["insulin_basal_5min_rows"] +
                                              s["insulin_basal_injection_rows"]) > 0).mean()), 3),
            "share_with_total_5min": round(float((s["insulin_total_5min_rows"] > 0).mean()), 3),
            "share_with_carbs": round(float((s["carb_rows"] > 0).mean()), 3),
            "share_with_heart_rate": round(float((s.get("hr_days", pd.Series(0, index=s.index)).fillna(0) > 0).mean()), 3),
            "share_with_sleep": round(float((s["sleep_rows"] > 0).mean()), 3),
            "median_carb_events_per_day": float(s["carb_events_per_day"].median()) if "carb_events_per_day" in s else np.nan,
            "median_bolus_events_per_day": float(s["bolus_events_per_day"].median()) if "bolus_events_per_day" in s else np.nan,
        })
    return pd.DataFrame(rows)


def aggregate_histograms(merged: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g_rows, i_rows, d_rows = [], [], []
    for dataset in DATASET_ORDER:
        descs = [m.description for m in merged if m.dataset == dataset]
        if not descs:
            continue
        g = np.sum([d.glucose_hist for d in descs], axis=0)
        i = np.sum([d.interval_hist for d in descs], axis=0)
        g_share = g / g.sum() if g.sum() else g
        i_share = i / i.sum() if i.sum() else i
        for k in range(len(g)):
            g_rows.append({"dataset": dataset, "bin_low_mgdl": int(GLUCOSE_BINS[k]), "bin_high_mgdl": int(GLUCOSE_BINS[k + 1]),
                           "readings": int(g[k]), "share": round(float(g_share[k]), 6)})
        for k, label in enumerate(INTERVAL_LABELS):
            i_rows.append({"dataset": dataset, "interval_min": label, "intervals": int(i[k]),
                           "share": round(float(i_share[k]), 6)})
        g_sum = np.sum([d.hour_glucose_sum for d in descs], axis=0)
        g_n = np.sum([d.hour_glucose_n for d in descs], axis=0)
        carbs = np.sum([d.hour_carbs_n for d in descs], axis=0)
        bolus = np.sum([d.hour_bolus_n for d in descs], axis=0)
        for hour in range(24):
            d_rows.append({"dataset": dataset, "hour": hour,
                           "mean_glucose_mgdl": round(float(g_sum[hour] / g_n[hour]), 1) if g_n[hour] else np.nan,
                           "carb_entries": int(carbs[hour]),
                           "carb_share": round(float(carbs[hour] / carbs.sum()), 4) if carbs.sum() else np.nan,
                           "bolus_entries": int(bolus[hour]),
                           "bolus_share": round(float(bolus[hour] / bolus.sum()), 4) if bolus.sum() else np.nan})
    return pd.DataFrame(g_rows), pd.DataFrame(i_rows), pd.DataFrame(d_rows)


def aggregate_wearables(merged: list) -> pd.DataFrame:
    """Mean heart rate, share of steps and share of time asleep by local hour, per dataset."""
    rows = []
    for dataset in DATASET_ORDER:
        descs = [m.description for m in merged if m.dataset == dataset]
        hr_n = np.sum([d.hour_hr_n for d in descs], axis=0) if descs else np.zeros(24)
        steps = np.sum([d.hour_steps_sum for d in descs], axis=0) if descs else np.zeros(24)
        asleep = np.sum([d.hour_asleep_min for d in descs], axis=0) if descs else np.zeros(24)
        nights = sum(d.sleep_nights for d in descs)
        if not descs or (hr_n.sum() == 0 and asleep.sum() == 0):
            continue
        hr_sum = np.sum([d.hour_hr_sum for d in descs], axis=0)
        for hour in range(24):
            rows.append({"dataset": dataset, "hour": hour,
                         "mean_hr_bpm": round(float(hr_sum[hour] / hr_n[hour]), 1) if hr_n[hour] else np.nan,
                         "steps_share": round(float(steps[hour] / steps.sum()), 4) if steps.sum() else np.nan,
                         "asleep_share": round(float(asleep[hour] / (60 * nights)), 4) if nights else np.nan})
    return pd.DataFrame(rows)


# ============================================================================ figures


def _datasets(frame: pd.DataFrame) -> list[str]:
    return [d for d in DATASET_ORDER if d in set(frame["dataset"])]


def plot_valid_days(summary: pd.DataFrame, min_valid: int) -> plt.Figure:
    datasets = _datasets(summary)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 2.1 * len(datasets)), squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        s = summary[summary["dataset"] == dataset].sort_values("valid_days", ascending=False)
        x = np.arange(len(s))
        ax.bar(x, s["calendar_days"].fillna(0), color="#d9d9d6", width=0.9, label="calendar days with follow-up")
        ax.bar(x, s["valid_days"].fillna(0), color=DATASET_COLORS[dataset], width=0.9,
               label="valid days (CGM coverage >= 70 %)")
        if s["calendar_days"].max() > 400:  # very long follow-ups (raw device history): log scale keeps short ones visible
            ax.set_yscale("log")
            ax.set_ylim(1, s["calendar_days"].max() * 1.5)
        for level in (min_valid, 98):
            ax.axhline(level, color=TEXT_SECONDARY, lw=0.8, ls="--")
            ax.text(len(s) - 0.5, level, f"{level} d", va="bottom", ha="right", fontsize=8, color=TEXT_SECONDARY)
        n_ok = int((s["valid_days"] >= min_valid).sum())
        ax.set_title(f"{DATASET_LABELS[dataset]}: {n_ok} of {len(s)} patients with >= {min_valid} valid days",
                     loc="left", fontsize=10)
        ax.set_xlim(-0.6, max(len(s), 1) - 0.4)
        ax.set_xticks([])
        ax.set_ylabel("days")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(loc="upper right", ncols=2)
    axes[-1, 0].set_xlabel("patients, sorted by valid days")
    fig.suptitle("Follow-up and valid CGM days per patient", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_modalities(summary: pd.DataFrame) -> plt.Figure:
    datasets = _datasets(summary)
    share = np.zeros((len(datasets), len(MODALITIES)))
    labels = np.empty_like(share, dtype=object)
    for i, dataset in enumerate(datasets):
        s = summary[summary["dataset"] == dataset].fillna({"hr_days": 0, "steps_days": 0})
        for j, (_, rule) in enumerate(MODALITIES):
            n = int(s.apply(rule, axis=1).sum())
            share[i, j] = n / len(s)
            labels[i, j] = f"{n}/{len(s)}"
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(datasets) + 1.8))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f4f2"] + SEQUENTIAL_BLUE)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(datasets)):
        for j in range(len(MODALITIES)):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(MODALITIES)), [m for m, _ in MODALITIES], rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    ax.grid(False)
    ax.set_title("Patients with each data type (cell = patients with data / patients)", loc="left")
    fig.tight_layout()
    return fig


def plot_intervals(intervals: pd.DataFrame) -> plt.Figure:
    datasets = _datasets(intervals)
    table = intervals.pivot(index="dataset", columns="interval_min", values="share").reindex(datasets)[INTERVAL_LABELS]
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(datasets) + 1.8))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f4f2"] + SEQUENTIAL_BLUE)
    ax.imshow(table.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(datasets)):
        for j in range(len(INTERVAL_LABELS)):
            v = table.iloc[i, j]
            if v >= 0.001:
                ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(INTERVAL_LABELS)), INTERVAL_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    ax.set_xlabel("minutes between consecutive automatic readings of the main CGM source")
    ax.grid(False)
    ax.set_title("Sampling intervals (% of intervals; cells below 0.1 % left blank)", loc="left")
    fig.tight_layout()
    return fig


def plot_glucose(hist: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    for dataset in _datasets(hist):
        h = hist[hist["dataset"] == dataset]
        centre = (h["bin_low_mgdl"] + h["bin_high_mgdl"]) / 2
        ax.plot(centre, 100 * h["share"], color=DATASET_COLORS[dataset], ls=DATASET_LINESTYLES[dataset], lw=2,
                label=DATASET_LABELS[dataset])
    for level in (70, 180):
        ax.axvline(level, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.set_xlabel("glucose (mg/dL), 10 mg/dL bins")
    ax.set_ylabel("% of readings")
    ax.set_xlim(20, 450)
    ax.set_title("Glucose distribution of the main CGM source", loc="left")
    ax.legend(ncols=2)
    fig.tight_layout()
    return fig


def plot_events(summary: pd.DataFrame) -> plt.Figure:
    panels = [("bolus_events_per_day", "bolus entries per day"), ("carb_events_per_day", "carb entries per day"),
              ("basal_units_per_day_median", "basal units per day (median day)")]
    datasets = _datasets(summary)
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.8))
    for ax, (column, title) in zip(axes, panels):
        data, names, colors = [], [], []
        for dataset in datasets:
            if column not in summary:
                continue
            values = summary.loc[summary["dataset"] == dataset, column].dropna()
            values = values[values > 0]
            if len(values):
                data.append(values.to_numpy())
                names.append(f"{DATASET_LABELS[dataset]}\n(n={len(values)})")
                colors.append(DATASET_COLORS[dataset])
        if data:
            parts = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                               flierprops={"markersize": 3, "alpha": 0.5}, medianprops={"color": "#0b0b0b"})
            for patch, color in zip(parts["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(names) + 1), names, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Recorded events per patient (patients with at least one event)", fontweight="bold", x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


def plot_diurnal(diurnal: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for dataset in _datasets(diurnal):
        d = diurnal[diurnal["dataset"] == dataset]
        style = {"color": DATASET_COLORS[dataset], "ls": DATASET_LINESTYLES[dataset], "lw": 2, "marker": "o",
                 "markersize": 3, "label": DATASET_LABELS[dataset]}
        axes[0].plot(d["hour"], d["mean_glucose_mgdl"], **style)
        if d["carb_share"].notna().any():
            axes[1].plot(d["hour"], 100 * d["carb_share"], **style)
    axes[0].set_title("Mean glucose by local hour", loc="left", fontsize=10)
    axes[0].set_ylabel("mg/dL")
    axes[1].set_title("Carb entries by local hour (meal peaks should sit near 8, 14 and 21 h)", loc="left",
                      fontsize=10)
    axes[1].set_ylabel("% of carb entries")
    for ax in axes:
        ax.set_xlabel("hour of day")
        ax.set_xticks(range(0, 24, 3))
    axes[0].legend(ncols=2)
    fig.suptitle("Clock check: diurnal patterns after time-zone handling", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_wearable_clock(wear: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for dataset in _datasets(wear):
        d = wear[wear["dataset"] == dataset]
        style = {"color": DATASET_COLORS[dataset], "ls": DATASET_LINESTYLES[dataset], "lw": 2, "marker": "o",
                 "markersize": 3, "label": DATASET_LABELS[dataset]}
        if d["mean_hr_bpm"].notna().any():
            axes[0].plot(d["hour"], d["mean_hr_bpm"], **style)
        if d["asleep_share"].notna().any():
            axes[1].plot(d["hour"], 100 * d["asleep_share"], **style)
    axes[0].set_title("Mean heart rate by local hour", loc="left", fontsize=10)
    axes[0].set_ylabel("bpm")
    axes[1].set_title("Share of nights asleep at each hour (sleep log)", loc="left", fontsize=10)
    axes[1].set_ylabel("% of nights")
    for ax in axes[:2]:
        ax.set_xlabel("hour of day")
        ax.set_xticks(range(0, 24, 3))
    axes[0].legend()
    lags = summary.dropna(subset=["hr_sleep_best_lag_h"]) if "hr_sleep_best_lag_h" in summary else pd.DataFrame()
    datasets = _datasets(lags) if not lags.empty else []
    for k, dataset in enumerate(datasets):
        values = lags.loc[lags["dataset"] == dataset, "hr_sleep_best_lag_h"].to_numpy()
        jitter = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else np.zeros(1)
        axes[2].scatter(values, k + jitter, color=DATASET_COLORS[dataset], s=18, alpha=0.8)
    axes[2].axvline(0, color=TEXT_SECONDARY, lw=0.8, ls="--")
    axes[2].set_yticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    axes[2].set_xlabel("shift of the sleep log that best separates asleep and awake heart rate (h)")
    axes[2].set_title("Clock agreement per patient (0 = heart rate and sleep agree)", loc="left", fontsize=10)
    axes[2].set_xlim(-12.5, 12.5)
    fig.suptitle("Wearable clock check", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(status: pd.DataFrame, datasets: pd.DataFrame, summary: pd.DataFrame, task_log: pd.DataFrame,
                  value_counts: pd.DataFrame, patients: pd.DataFrame, read_seconds: float, merge_seconds: float,
                  resources, derived: Path, min_valid: int, time_offsets: pd.DataFrame, release: str) -> str:
    lines = ["# 02 - Harmonised datasets", "",
             f"Code release {release} (checked against src/t1dretrain/code_manifest.json).",
             f"Machine: {resources.describe()}",
             f"Read phase {read_seconds:.0f} s, merge phase {merge_seconds:.0f} s. Processed tables in `{derived / 'processed'}`.",
             ""]
    lines += ["## Datasets", "", markdown_table(status), ""]
    warnings = []
    failed = task_log[task_log["status"] != "ok"]
    for _, row in failed.iterrows():
        warnings.append(f"{row['phase']} {row['dataset']}/{row['task']} failed: {row['error']}")
    warned = task_log[task_log["warnings"].fillna("") != ""]
    for _, row in warned.head(10).iterrows():
        warnings.append(f"{row['phase']} {row['dataset']}/{row['task']} raised warnings: {row['warnings'][:400]}")
    if len(warned) > 10:
        warnings.append(f"... and {len(warned) - 10} more tasks with warnings (task_log.csv, column 'warnings')")
    for _, row in status[status["status"].isin(["missing", "plan_error"])].iterrows():
        warnings.append(f"{row['dataset']}: {row['status']} ({row['detail']})")
    if not value_counts.empty:
        unparsed = value_counts[(value_counts["value"] == "unparsed") & (value_counts["count"] > 0)]
        for _, row in unparsed.iterrows():
            warnings.append(f"{row['dataset']} {row['source']}: {row['count']:,} time stamps could not be parsed")
        out_of_range = value_counts[value_counts["value"].astype(str).str.startswith("outside [")]
        for _, row in out_of_range.iterrows():
            warnings.append(f"{row['dataset']} {row['source']} {row['column']}: {row['count']:,} values {row['value']}")
        unresolved = value_counts[value_counts["value"] == "unresolved"]
        for _, row in unresolved.iterrows():
            warnings.append(f"{row['dataset']} {row['source']}: {row['count']:,} rows without any time-zone information")
    if not summary.empty:
        no_cgm = summary[summary["cgm_main_rows"] == 0]
        if len(no_cgm):
            warnings.append(f"{len(no_cgm)} patients have no automatic CGM readings: "
                            + ", ".join(no_cgm["patient_id"].head(15)) + ("..." if len(no_cgm) > 15 else ""))
        dup = summary[summary["share_intervals_lt_2min"] > 0.02]
        if len(dup):
            warnings.append(f"{len(dup)} patients have more than 2 % of CGM intervals under 2 min (near-duplicate "
                            "readings to resolve in script 03): " + ", ".join(dup["patient_id"].head(10)))
    lines += ["## Warnings", ""] + ([f"- {w}" for w in warnings] or ["- none"]) + [""]
    if not datasets.empty:
        lines += ["## Per dataset", "", _transpose_markdown(datasets), ""]
    if not summary.empty:
        lines += ["## Patients usable for the ageing study", "",
                  f"Patients with at least {min_valid} valid days (CGM coverage >= 70 % of the day, main source):", ""]
        for dataset in _datasets(summary):
            s = summary[summary["dataset"] == dataset]
            lines.append(f"- {DATASET_LABELS[dataset]}: {int((s['valid_days'] >= min_valid).sum())} of {len(s)} "
                         f"(>= 98 days: {int((s['valid_days'] >= 98).sum())}; median longest run of valid days "
                         f"{s['longest_valid_run_days'].median():.0f})")
        lines.append("")
    final = value_counts[value_counts["column"] == "local time from (final)"] if not value_counts.empty else value_counts
    if not final.empty:
        table = final.groupby(["dataset", "value"])["count"].sum().unstack(fill_value=0)
        table = (table.T / table.sum(axis=1)).T.round(4).reset_index()
        lines += ["## How local times were obtained (share of rows, after phase B)", "", markdown_table(table), ""]
    if not time_offsets.empty and "main_minus_roster_h" in time_offsets:
        t = time_offsets.dropna(subset=["main_minus_roster_h"])
        lines += ["## Loop: UTC offsets seen on device clocks vs roster offset", "",
                  f"- Patients with observations: {len(t)}; main observed offset equal to the roster offset: "
                  f"{int((t['main_minus_roster_h'] == 0).sum())}; one hour later (summer time): "
                  f"{int((t['main_minus_roster_h'] == 1).sum())}; other: {int((~t['main_minus_roster_h'].isin([0, 1])).sum())}.",
                  f"- Patients whose offset changes at least once (summer time or travel): "
                  f"{int((t['offset_changes'] > 0).sum())}.",
                  f"- Patients without any device-clock observation (all times from the standard-time roster "
                  f"offset, up to 1 h early in summer time): "
                  f"{int((time_offsets['offset_hours_observed'] == 0).sum())} of {len(time_offsets)}.", ""]
    if "hr_sleep_best_lag_h" in summary:
        lag = summary.dropna(subset=["hr_sleep_best_lag_h"])
        rows = [{"dataset": DATASET_LABELS.get(d, d), "patients": len(g),
                 "median contrast at lag 0 (bpm)": g["hr_asleep_minus_awake"].median(),
                 "median best lag (h)": g["hr_sleep_best_lag_h"].median(),
                 "best lag within 1 h": int((g["hr_sleep_best_lag_h"].abs() <= 1).sum()),
                 "median contrast at best lag (bpm)": g["hr_asleep_minus_awake_at_best_lag"].median()}
                for d, g in lag.groupby("dataset")]
        lines += ["## Wearable clocks: heart rate against the sleep log", "", markdown_table(pd.DataFrame(rows)), ""]
    lines += ["## Checks to look at", "",
              "- `diurnal_check.png`: carb-entry peaks at breakfast, lunch and dinner hours confirm local clocks "
              "(Loop is converted from UTC; DCLP3 and REPLACE-BG use device clocks).",
              "- `sampling_intervals.png`: Dexcom data should sit at 4.5-5.5 min, FreeStyle Libre history at 14-16 min.",
              "- `source_value_counts.csv`: record types, units, bolus and basal types, time sources and censored values.",
              "- `wearable_clock_check.png` and `patient_summary.csv` (`hr_asleep_minus_awake`, `hr_sleep_best_lag_h`): "
              "a clearly negative contrast at lag 0 means that the watch's heart-rate and sleep clocks agree.",
              "- `time_offsets.csv`: Loop offsets observed on device clocks (summer time and travel).",
              "- HUPA-UCM carb units per patient: `patients_static.csv`, column `carb_input_unit`.", ""]
    return "\n".join(lines)


def _transpose_markdown(frame: pd.DataFrame) -> str:
    table = frame.set_index("dataset").T
    table.columns = [DATASET_LABELS.get(c, c) for c in table.columns]
    return markdown_table(table.reset_index().rename(columns={"index": "metric"}))


if __name__ == "__main__":
    main()
