"""05 - Model inputs: native-grid glucose, insulin and carbohydrates on board, calendar and wearables.

For every patient of a cohort (script 04), builds one table per patient on the
patient's native grid (rules in src/t1dretrain/features/ and decisions D11,
D54-D56 of HANDOFF.md):

- glucose on a regular grid at the sensor interval (5 or 15 min), phase-aligned
  to the patient's readings; inputs interpolated only across short gaps and
  never across a clock break of script 03; targets only on real readings;
- insulin on board and insulin activity from all delivered insulin (Loop /
  OpenAPS exponential curve: 6 h, peak at 75 min; long-acting injections act
  over 24 h);
- carbohydrates on board with linear absorption over 3 h;
- time of day, day of week where dates are real, wearables (heart rate,
  steps, kcal, sleep) and day-level flags (valid day, loop mode, period,
  insulin and carbohydrates known).

Every input at a slot is evaluated at the time of its reading and only uses
records made at or before it. Runs one job per patient on all cores but two
(rule R5).

Derived data:
    <derived_data>/features/<dataset>/grid/<patient_id>.parquet   one row per slot
    <derived_data>/features/<dataset>/feature_patients.parquet     the per-patient checks

Outputs (results/05_build_model_inputs/):
    feature_summary.csv     per dataset: grid, coverage, interpolation, insulin and carbohydrates, wearables
    patient_features.csv    per patient: the same, with total daily insulin and carbohydrates per day
    hourly_profile.csv      mean IOB, COB and glucose by hour of day per dataset
    glucose_response.csv    mean glucose change around meals and correction boluses per dataset
    snap_offsets.csv        distance between readings and their slots, per dataset
    task_log.csv            one row per patient job (status, seconds, warnings)
    insulin_model.png
    iob_cob_by_hour.png
    glucose_response.png
    example_days.png
    summary.md
    run_manifest.json

Usage:
    python scripts/05_build_model_inputs.py
    python scripts/05_build_model_inputs.py --datasets replace_bg hupa_ucm
    python scripts/05_build_model_inputs.py --all-patients     # also patients outside the cohorts
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.store import ProcessedStore, safe_id  # noqa: E402
from t1dretrain.features.build import RESPONSE_OFFSETS_MIN  # noqa: E402
from t1dretrain.features.calendar import real_dates  # noqa: E402
from t1dretrain.features.insulin import ExponentialInsulin  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import (DATASET_COLORS, DATASET_LABELS, DATASET_LINESTYLES, DATASET_ORDER,  # noqa: E402
                                 SEQUENTIAL_BLUE, TEXT_SECONDARY, set_style)

SCRIPT_ID = "05_build_model_inputs"
TDI_MAX = 200.0  # U/day


def tdi_minimum(age: float) -> float:
    """Lowest plausible total daily insulin (U/day) for an age: small children need very little."""
    if age is not None and np.isfinite(age):
        if age < 7:
            return 2.0
        if age < 12:
            return 5.0
    return 10.0


def tdi_implausible(frame: pd.DataFrame) -> pd.Series:
    tdi = frame["tdi_median_u"]
    low = frame["age"].map(tdi_minimum) if "age" in frame else pd.Series(10.0, index=frame.index)
    return tdi.notna() & ((tdi < low) | (tdi > TDI_MAX))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of datasets (default: all with cohorts)")
    parser.add_argument("--all-patients", action="store_true", help="build every patient with clean CGM, not only "
                        "the cohort members of script 04")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_job(job: dict):
    """Worker entry point: one patient, with its warnings recorded."""
    from t1dretrain.features.build import build_patient

    with WarningRecorder() as recorder:
        result = build_patient(job)
    result.row["warnings"] = recorder.text()
    return result


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    settings = dict(config["features"])
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)

    available = [d for d in DATASET_ORDER if (derived / "clean" / d / "days.parquet").exists()]
    datasets = [d for d in (args.datasets or available) if d in available]
    jobs, skipped, ages = [], [], {}
    for dataset in datasets:
        clean = ProcessedStore(dataset, root=derived, stage="clean")
        cohorts = clean.read_extra("cohorts")
        if cohorts.empty and not args.all_patients:
            skipped.append(f"{dataset}: no cohorts (run script 04 first)")
            continue
        patients = clean.read_patients()
        notes = patients.set_index("patient_id")["dates"] if "dates" in patients else pd.Series(dtype=object)
        if "age" in patients:
            ages.update(dict(zip(patients["patient_id"].astype(str), pd.to_numeric(patients["age"], errors="coerce"))))
        ids = clean.patients_with("cgm") if args.all_patients else sorted(set(cohorts["patient_id"].astype(str)))
        example = pick_example(cohorts)
        days = pd.read_parquet(clean.root / "days.parquet")
        by_patient = dict(tuple(days.groupby("patient_id")))
        target = derived / "features" / dataset
        if target.exists():
            shutil.rmtree(target)
        for patient_id in ids:
            path = clean.table_dir("cgm") / f"{safe_id(patient_id)}.parquet"
            if patient_id not in by_patient or not path.exists():
                continue
            jobs.append({"dataset": dataset, "patient_id": patient_id, "derived_root": derived, "settings": settings,
                         "days": by_patient[patient_id], "real_dates": real_dates(notes.get(patient_id)),
                         "example": patient_id == example, "size": path.stat().st_size})
    jobs.sort(key=lambda job: job["size"], reverse=True)
    results = parallel_map(run_job, jobs, resources.n_workers, label="patients")

    task_log = pd.DataFrame([{"dataset": r.dataset, "patient_id": r.patient_id, "status": "error" if r.error else "ok",
                              "seconds": r.seconds, "warnings": r.row.get("warnings", ""),
                              "error": r.error.strip().splitlines()[-1] if r.error else "", "traceback": r.error}
                             for r in results])
    ok = [r for r in results if not r.error]
    patients = pd.DataFrame([r.row for r in ok])
    if not patients.empty:
        patients.insert(2, "age", patients["patient_id"].map(ages))
    for dataset in datasets:
        part = patients[patients["dataset"] == dataset] if not patients.empty else patients
        if not part.empty:
            ProcessedStore(dataset, root=derived, stage="features").write_extra("feature_patients", part)
    summary = dataset_summary(patients, ok)
    hourly = pooled(ok, "hourly", ["variable", "hour"])
    response = pooled(ok, "responses", ["event", "minutes"])
    offsets = offset_table(ok)

    out.csv(summary, "feature_summary")
    out.csv(patients, "patient_features")
    out.csv(hourly, "hourly_profile")
    out.csv(response, "glucose_response")
    out.csv(offsets, "snap_offsets")
    out.csv(task_log, "task_log")
    out.figure(plot_models(settings), "insulin_model")
    if not hourly.empty:
        out.figure(plot_hourly(hourly), "iob_cob_by_hour")
    if not response.empty:
        out.figure(plot_response(response), "glucose_response")
    examples = [r for r in ok if r.example is not None]
    if examples:
        out.figure(plot_examples(examples), "example_days")
    out.text(write_summary(summary, patients, response, task_log, skipped, settings, release, resources, derived),
             "summary.md")
    out.manifest(settings={"datasets": datasets, "derived_data": str(derived), **settings,
                           "all_patients": args.all_patients},
                 notes="One job per patient on n_workers processes; tables in <derived>/features.")
    print(f"Done. Results in {out.dir}")


def pick_example(cohorts: pd.DataFrame) -> str | None:
    """The patient with the most usable days in the dataset's first cohort (for the example-day figure)."""
    if cohorts.empty:
        return None
    first = cohorts[cohorts["cohort"] == cohorts["cohort"].iloc[0]]
    return str(first.sort_values("usable_days", ascending=False)["patient_id"].iloc[0])


# ============================================================================ tables


def dataset_summary(patients: pd.DataFrame, results: list) -> pd.DataFrame:
    rows = []
    for dataset in [d for d in DATASET_ORDER if not patients.empty and d in set(patients["dataset"])]:
        p = patients[patients["dataset"] == dataset]
        w = p["slots"]
        offsets = np.concatenate([r.offsets for r in results if r.dataset == dataset and r.offsets.size] or [np.zeros(0)])

        def weighted(column: str) -> float:
            values = p[column]
            ok = values.notna()
            return round(float((values[ok] * w[ok]).sum() / w[ok].sum()), 4) if ok.any() else np.nan

        tdi = p["tdi_median_u"].dropna()
        rows.append({
            "dataset": dataset, "patients": int(len(p)),
            "step_min": "+".join(str(int(s)) for s in sorted(p["step_min"].unique())),
            "slots_millions": round(float(w.sum() / 1e6), 2),
            "observed_share": weighted("observed_share"), "target_ok_share": weighted("target_ok_share"),
            "interpolated_share": weighted("interpolated_share"), "input_share": weighted("input_share"),
            "collisions_dropped": int(p["collisions_dropped"].sum()),
            "clock_segments_extra": int((p["segments"] - 1).sum()),
            "snap_offset_abs_median_s": round(float(np.median(np.abs(offsets))), 1) if offsets.size else np.nan,
            "snap_offset_abs_p95_s": round(float(np.percentile(np.abs(offsets), 95)), 1) if offsets.size else np.nan,
            "insulin_known_share": weighted("insulin_known_share"), "carbs_known_share": weighted("carbs_known_share"),
            "tdi_median_u": round(float(tdi.median()), 1) if len(tdi) else np.nan,
            "tdi_implausible_patients": int(tdi_implausible(p).sum()),
            "bolus_share_median": round(float(p["bolus_share"].median()), 3) if p["bolus_share"].notna().any() else np.nan,
            "carbs_median_g_per_day": round(float(p["carbs_median_g_per_day"].median()), 1)
            if p["carbs_median_g_per_day"].notna().any() else np.nan,
            "iob_mean_u": round(float(p["iob_mean_u"].median()), 2) if p["iob_mean_u"].notna().any() else np.nan,
            "cob_mean_g": round(float(p["cob_mean_g"].median()), 1) if p["cob_mean_g"].notna().any() else np.nan,
            "hr_share": weighted("hr_share"), "steps_share": weighted("steps_share"),
            "asleep_share": weighted("asleep_share"),
        })
    return pd.DataFrame(rows)


def pooled(results: list, attribute: str, keys: list[str]) -> pd.DataFrame:
    frames = [getattr(r, attribute).assign(dataset=r.dataset) for r in results if not getattr(r, attribute).empty]
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True).groupby(["dataset"] + keys, as_index=False)[["sum", "count"]].sum()
    table["mean"] = (table["sum"] / table["count"].where(table["count"] > 0)).round(3)
    table["dataset_order"] = table["dataset"].map(DATASET_ORDER.index)
    return table.sort_values(["dataset_order"] + keys).drop(columns=["dataset_order", "sum"]).reset_index(drop=True)


def offset_table(results: list) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        values = [r.offsets for r in results if r.dataset == dataset and r.offsets.size]
        if not values:
            continue
        o = np.abs(np.concatenate(values))
        rows.append({"dataset": dataset, "sampled": int(o.size), **{f"abs_offset_p{q}_s": round(float(np.percentile(o, q)), 1)
                                                                    for q in (50, 75, 90, 95, 99)}})
    return pd.DataFrame(rows)


# ============================================================================ figures


def plot_models(settings: dict) -> plt.Figure:
    model = ExponentialInsulin(float(settings["insulin_action_min"]), float(settings["insulin_peak_min"]),
                               float(settings.get("insulin_delay_min", 0.0)))
    t = np.arange(0, model.action_min + 30)
    absorption = float(settings["carb_absorption_min"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(t / 60, 100 * model.remaining(t), color=SEQUENTIAL_BLUE[5], lw=2, label="insulin on board (% of dose)")
    axes[0].plot(t / 60, 100 * np.clip(1 - t / absorption, 0, None), color="#eb6834", lw=2, ls="--",
                 label="carbohydrates on board (% of entry)")
    axes[0].set_ylabel("% still to act")
    axes[0].set_title("On board after one dose or entry", loc="left", fontsize=10)
    axes[0].legend()
    axes[1].plot(t / 60, 100 * 60 * model.activity(t), color=SEQUENTIAL_BLUE[5], lw=2)
    axes[1].axvline(model.peak_min / 60 + model.delay_min / 60, color=TEXT_SECONDARY, lw=0.8, ls=":")
    axes[1].set_ylabel("% of dose acting per hour")
    axes[1].set_title(f"Insulin activity (peak {model.peak_min:.0f} min, duration {model.action_min / 60:.0f} h)",
                      loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("hours after the dose or entry")
    fig.suptitle("Insulin and carbohydrate models used for the inputs", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_hourly(hourly: pd.DataFrame) -> plt.Figure:
    titles = {"iob": "insulin on board (U)", "cob": "carbohydrates on board (g)", "glucose": "glucose (mg/dL)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, variable in zip(axes, ["iob", "cob", "glucose"]):
        h = hourly[hourly["variable"] == variable]
        for dataset in [d for d in DATASET_ORDER if d in set(h["dataset"])]:
            d = h[h["dataset"] == dataset]
            if d["mean"].notna().any():
                ax.plot(d["hour"], d["mean"], color=DATASET_COLORS[dataset], ls=DATASET_LINESTYLES[dataset], lw=2,
                        label=DATASET_LABELS[dataset])
        ax.set_title(f"Mean {titles[variable]} by hour", loc="left", fontsize=10)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("hour of day (local)")
    axes[0].legend(ncols=2)
    fig.suptitle("Daily profiles on valid days: meals and boluses should show at breakfast, lunch and dinner",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_response(response: pd.DataFrame) -> plt.Figure:
    titles = {"meal": "after a meal entry (>= 20 g)", "correction_bolus": "after a correction bolus (>= 1 U, no carbs +-1 h)"}
    events = [e for e in ["meal", "correction_bolus"] if e in set(response["event"])]
    fig, axes = plt.subplots(1, len(events), figsize=(6.5 * len(events), 4.2), squeeze=False)
    for ax, event in zip(axes[0], events):
        r = response[response["event"] == event]
        for dataset in [d for d in DATASET_ORDER if d in set(r["dataset"])]:
            d = r[r["dataset"] == dataset]
            ax.plot(d["minutes"], d["mean"], color=DATASET_COLORS[dataset], ls=DATASET_LINESTYLES[dataset], lw=2,
                    marker="o", markersize=3, label=DATASET_LABELS[dataset])
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.6)
        ax.set_xticks(RESPONSE_OFFSETS_MIN[::4])
        ax.set_xlabel("minutes from the event")
        ax.set_ylabel("mean glucose change (mg/dL)")
        ax.set_title(f"Glucose {titles[event]}", loc="left", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Timing check: glucose should rise after meals and fall after corrections", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_examples(examples: list) -> plt.Figure:
    examples = sorted(examples, key=lambda r: DATASET_ORDER.index(r.dataset))
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3.0 * len(examples)), squeeze=False)
    for ax, result in zip(axes[:, 0], examples):
        e = result.example
        f = e["frame"]
        hours = (f["ts"] - e["day"]).dt.total_seconds() / 3600
        ax.plot(hours, f["glucose_input"], color=SEQUENTIAL_BLUE[2], lw=1, label="input (interpolated in short gaps)")
        ax.scatter(hours, f["glucose"], s=6, color=SEQUENTIAL_BLUE[5], label="readings", zorder=3)
        ax.set_ylabel("glucose (mg/dL)")
        ax.set_xlim(0, 24)
        right = ax.twinx()
        right.plot(hours, f["iob"], color="#008300", lw=1.6, label="IOB (U)")
        right.plot(hours, f["cob"] / 10, color="#eb6834", lw=1.6, ls="--", label="COB (g / 10)")
        right.set_ylabel("IOB (U) · COB (g/10)")
        right.grid(False)
        for _, b in e["boluses"].iterrows():
            ax.axvline((b["ts"] - e["day"]).total_seconds() / 3600, color="#008300", lw=0.6, alpha=0.6)
        for _, c in e["carbs"].iterrows():
            x = (c["ts"] - e["day"]).total_seconds() / 3600
            ax.annotate(f"{c['carbs_g']:.0f} g", (x, ax.get_ylim()[1]), fontsize=7, color="#eb6834", ha="center",
                        va="top")
        ax.set_title(f"{DATASET_LABELS[result.dataset]} · {result.patient_id} · {e['day'].date()} "
                     "(vertical lines: boluses)", loc="left", fontsize=10)
        if ax is axes[0, 0]:
            lines = ax.get_legend_handles_labels()
            more = right.get_legend_handles_labels()
            ax.legend(lines[0] + more[0], lines[1] + more[1], loc="upper left", fontsize=8, ncols=4)
    axes[-1, 0].set_xlabel("hour of day")
    fig.suptitle("One example day per dataset", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(summary: pd.DataFrame, patients: pd.DataFrame, response: pd.DataFrame, task_log: pd.DataFrame,
                  skipped: list[str], settings: dict, release: str, resources, derived: Path) -> str:
    lines = ["# 05 - Model inputs", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Tables in `{derived / 'features'}`. Insulin: exponential curve, {settings['insulin_action_min']} min, "
             f"peak {settings['insulin_peak_min']} min; long-acting over {settings['long_acting_spread_h']} h. "
             f"Carbohydrates: linear over {settings['carb_absorption_min']} min. Inputs interpolated across gaps of "
             f"up to {settings['max_interp_gap_min']} min.", ""]
    problems = list(skipped)
    for _, row in task_log[task_log["status"] != "ok"].iterrows():
        problems.append(f"{row['dataset']}/{row['patient_id']} failed: {row['error']}")
    warned = task_log[task_log["warnings"].fillna("") != ""]
    for _, row in warned.head(10).iterrows():
        problems.append(f"{row['dataset']}/{row['patient_id']} raised warnings: {row['warnings'][:300]}")
    if not patients.empty:
        odd = patients[tdi_implausible(patients)]
        for _, row in odd.head(10).iterrows():
            problems.append(f"{row['dataset']}/{row['patient_id']} (age {row.get('age')}): median total daily insulin "
                            f"{row['tdi_median_u']} U (incomplete insulin records?)")
    if not response.empty:
        for dataset in set(response["dataset"]):
            meal = response[(response["dataset"] == dataset) & (response["event"] == "meal") & (response["minutes"] == 60)]
            if len(meal) and meal["mean"].iloc[0] <= 0:
                problems.append(f"{dataset}: glucose does not rise 60 min after meals (check the clocks)")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not summary.empty:
        table = summary.set_index("dataset").T
        table.columns = [DATASET_LABELS.get(c, c) for c in table.columns]
        lines += ["## Per dataset", "", markdown_table(table.reset_index().rename(columns={"index": "metric"})), ""]
    if not response.empty:
        wide = response.pivot_table(index=["event", "dataset"], columns="minutes", values="mean").reset_index()
        keep = ["event", "dataset"] + [m for m in [-30, 0, 30, 60, 90, 120, 180, 240] if m in wide.columns]
        lines += ["## Mean glucose change around events (mg/dL)", "",
                  markdown_table(wide[keep].assign(dataset=wide["dataset"].map(DATASET_LABELS))), ""]
    lines += ["## Checks to look at", "",
              "- `example_days.png`: IOB rises at each bolus and COB at each meal, then both decay; readings and "
              "inputs coincide except in short gaps.",
              "- `glucose_response.png`: glucose rises after meals (peak at 60-90 min) and falls after corrections.",
              "- `iob_cob_by_hour.png`: insulin and carbohydrates peak at meal times.",
              "- `feature_summary.csv`: total daily insulin (about 0.5-1 U/kg; 20-80 U/day in adults) and the share "
              "of slots with insulin and carbohydrates known.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
