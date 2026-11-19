"""03 - Data quality: one clean CGM series per patient, a day table and eligibility.

Reads the harmonised tables of script 02 and, for every patient (rules in
src/t1dretrain/data/quality.py and decisions D43-D48 of HANDOFF.md):

- cleans each CGM source on its own: drops repeated readings, drops the
  stretches where a device clock was set back (two stretches sharing clock
  times); combines the two upload streams of one sensor (Loop) where they
  agree; keeps one sensor per calendar day (the best covered), and flags
  readings at the sensor limits and isolated spikes;
- builds one row per patient-day: CGM coverage and validity, loop status
  (open / closed), study period (randomisation, HUPA-UCM window), insulin
  completeness, carbohydrates, wearables and sleep;
- counts valid days by loop status and insulin completeness and sets the
  eligibility flags (>= 42 valid days of each kind, D48).

Nothing is interpolated: the clean series keeps only real readings. Runs one
job per patient on all cores but two (rule R5).

Derived data (<derived_data>/clean/<dataset>/):
    cgm/<patient_id>.parquet   clean CGM series
    days.parquet               one row per patient-day
    patients.parquet           patients table of script 02 + quality + eligibility

Outputs (results/03_data_quality/):
    dataset_quality.csv                  per dataset: readings kept and removed, flags, valid days, eligibility
    patient_quality.csv                  per patient: the same, with study periods and coverage thresholds
    coverage_threshold_sensitivity.csv   valid days and eligible patients for coverage thresholds 50-90 %
    gap_distribution.csv                 gaps between clean readings by length class
    day_coverage_histogram.csv           distribution of daily CGM coverage
    loop_period_days.csv                 days, valid days and insulin-complete days by loop status and period
    duplicates.csv                       value differences of repeated readings, conflicts and stream pairs dropped
                                         (clock-overlap stretches keep the series that was running before them)
    task_log.csv                         one row per patient job (status, seconds, warnings)
    eligibility_funnel.png
    coverage_threshold_sensitivity.png
    gap_distribution.png
    day_coverage_distribution.png
    followup_timelines.png
    summary.md
    run_manifest.json

Usage:
    python scripts/03_data_quality.py
    python scripts/03_data_quality.py --datasets replace_bg hupa_ucm
"""

from __future__ import annotations

import argparse
import shutil
import sys
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
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.quality import COVERAGE_THRESHOLDS, GAP_CLASSES, eligibility  # noqa: E402
from t1dretrain.data.store import ProcessedStore, safe_id  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import (DATASET_COLORS, DATASET_LABELS, DATASET_LINESTYLES, DATASET_ORDER,  # noqa: E402
                                 SEQUENTIAL_BLUE, TEXT_SECONDARY, set_style)

SCRIPT_ID = "03_data_quality"
CONTEXT_COLUMNS = ["loop", "arm", "therapy", "randomization_ts", "window_start", "window_end"]
FUNNEL = [("patients_listed", "patients"), ("with_cgm", "with CGM"), ("eligible_cgm", "valid days >= {n}"),
          ("eligible_open_loop", "open-loop valid days >= {n}"),
          ("eligible_open_loop_insulin", "open loop + complete insulin >= {n}")]
TIMELINE_COLORS = {"valid_open": "#2a78d6", "valid_closed": "#eb6834", "invalid": "#c9c8c3", "none": "#ffffff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of datasets (default: all processed)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_job(job: dict):
    """Worker entry point: one patient, with its warnings recorded."""
    from t1dretrain.data.quality import assess_patient as assess

    with WarningRecorder() as recorder:
        result = assess(job)
    result.row["warnings"] = recorder.text()
    return result


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    threshold = float(config.get("valid_day_min_cgm_coverage", 0.70))
    min_valid = int(config.get("min_valid_days_core", 42))

    available = [d for d in DATASET_ORDER if (derived / "processed" / d / "patients.parquet").exists()]
    datasets = [d for d in (args.datasets or available) if d in available]
    jobs, statics = [], {}
    for dataset in datasets:
        source = ProcessedStore(dataset, root=derived)
        static = source.read_patients()
        statics[dataset] = static
        periods = source.read_extra("device_periods")
        if not periods.empty:
            periods = periods[periods["role"].isin(["insulin", "carbs"])]
        clean_root = derived / "clean" / dataset
        if clean_root.exists():
            shutil.rmtree(clean_root)
        clean_root.mkdir(parents=True, exist_ok=True)
        for patient_id in source.patients_with("cgm"):
            info = static[static["patient_id"] == patient_id]
            context = {"dataset": dataset}
            if len(info):
                context.update({c: info.iloc[0][c] for c in CONTEXT_COLUMNS if c in info})
            context = {k: (None if (not isinstance(v, str) and pd.isna(v)) else v) for k, v in context.items()}
            size = (source.table_dir("cgm") / f"{safe_id(patient_id)}.parquet").stat().st_size
            jobs.append({"dataset": dataset, "patient_id": patient_id, "derived_root": derived, "context": context,
                         "periods": periods[periods["patient_id"] == patient_id] if not periods.empty else periods,
                         "coverage_threshold": threshold, "size": size})
    jobs.sort(key=lambda job: job["size"], reverse=True)
    results = parallel_map(run_job, jobs, resources.n_workers, label="patients")

    task_log = pd.DataFrame([{"dataset": r.dataset, "patient_id": r.patient_id, "status": "error" if r.error else "ok",
                              "seconds": r.seconds, "warnings": r.row.get("warnings", ""),
                              "error": r.error.strip().splitlines()[-1] if r.error else "", "traceback": r.error}
                             for r in results])
    quality = pd.DataFrame([r.row for r in results])
    quality = pd.concat([quality, pd.DataFrame([eligibility(row, min_valid) for _, row in quality.iterrows()])],
                        axis=1)
    days = {d: pd.concat([r.days for r in results if r.dataset == d and not r.days.empty], ignore_index=True)
            for d in datasets if any(r.dataset == d and not r.days.empty for r in results)}
    gaps = pd.concat([r.gaps.assign(dataset=r.dataset) for r in results if not r.gaps.empty], ignore_index=True)

    for dataset in datasets:
        target = ProcessedStore(dataset, root=derived, stage="clean")
        if dataset in days:
            days[dataset].to_parquet(target.root / "days.parquet", index=False)
        q = quality[quality["dataset"] == dataset].drop(columns=["dataset", "warnings"], errors="ignore")
        patients = statics[dataset].merge(q, on="patient_id", how="left")
        target.write_patients(patients)

    context = pd.concat([s[["dataset", "patient_id"] + [c for c in ["loop", "arm", "therapy"] if c in s]]
                         for s in statics.values()], ignore_index=True)
    quality = quality.merge(context, on=["dataset", "patient_id"], how="left")
    dataset_quality = summarise(quality, statics, min_valid)
    sensitivity = threshold_sensitivity(quality, min_valid)
    gap_summary = summarise_gaps(gaps)
    coverage_hist = coverage_histogram(days)
    loop_days = loop_period_table(days)
    duplicates = duplicate_table(results)

    out.csv(dataset_quality, "dataset_quality")
    out.csv(quality, "patient_quality")
    out.csv(sensitivity, "coverage_threshold_sensitivity")
    out.csv(gap_summary, "gap_distribution")
    out.csv(coverage_hist, "day_coverage_histogram")
    out.csv(loop_days, "loop_period_days")
    out.csv(duplicates, "duplicates")
    out.csv(task_log, "task_log")
    if not dataset_quality.empty:
        out.figure(plot_funnel(dataset_quality, min_valid), "eligibility_funnel")
        out.figure(plot_sensitivity(sensitivity, threshold, min_valid), "coverage_threshold_sensitivity")
        out.figure(plot_gaps(gap_summary), "gap_distribution")
        out.figure(plot_coverage(coverage_hist, threshold), "day_coverage_distribution")
        out.figure(plot_timelines(days, quality), "followup_timelines")
    out.text(write_summary(dataset_quality, sensitivity, gap_summary, loop_days, duplicates, task_log, quality,
                           threshold, min_valid, release, resources, derived), "summary.md")
    out.manifest(settings={"datasets": datasets, "derived_data": str(derived),
                           "valid_day_min_cgm_coverage": threshold, "min_valid_days_core": min_valid},
                 notes="One job per patient on n_workers processes; clean data in <derived>/clean.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ tables


def summarise(quality: pd.DataFrame, statics: dict, min_valid: int) -> pd.DataFrame:
    rows = []
    for dataset in [d for d in DATASET_ORDER if d in statics]:
        q = quality[quality["dataset"] == dataset]
        raw = q["raw_auto_readings"].sum()
        clean = q["clean_readings"].sum()
        row = {
            "dataset": dataset,
            "patients_listed": int(len(statics[dataset])),
            "with_cgm": int((q["clean_readings"].fillna(0) > 0).sum()),
            "raw_auto_readings": int(raw),
            "clean_readings": int(clean),
            "removed_duplicates_share": round(float(q["removed_duplicates"].sum() / raw), 4) if raw else np.nan,
            "removed_isolated_conflicts_share": round(float(q["removed_isolated_conflicts"].sum() / raw), 4) if raw else np.nan,
            "clock_overlap_stretches": int(q["clock_overlap_stretches"].sum()),
            "removed_clock_overlap_share": round(float(q["removed_clock_overlap"].sum() / raw), 4) if raw else np.nan,
            "removed_cross_stream_share": round(float(q["removed_cross_stream_duplicates"].sum() / raw), 4) if raw else np.nan,
            "removed_stream_disagreement_share": round(float(q["removed_stream_disagreement"].sum() / raw), 4)
            if raw else np.nan,
            "stream_disagreement_days": int(q["stream_disagreement_days"].sum()),
            "added_from_second_stream_share": round(float(q["added_from_second_stream"].sum() / raw), 4) if raw else np.nan,
            "removed_other_source_share": round(float(q["removed_other_source_days"].sum() / raw), 4) if raw else np.nan,
            "at_limit_share": round(float(q["at_limit_readings"].sum() / clean), 5) if clean else np.nan,
            "spike_share": round(float(q["spike_readings"].sum() / clean), 5) if clean else np.nan,
            "valid_days": int(q["valid_days"].sum()),
            "valid_days_open": int(q["valid_days_open"].sum()),
            "valid_days_closed": int(q["valid_days_closed"].sum()),
            "valid_days_insulin_complete": int(q["valid_days_insulin_complete"].sum()),
            "valid_days_open_insulin": int(q["valid_days_open_insulin"].sum()),
            "valid_days_with_carbs": int(q["valid_days_with_carbs"].sum()),
            "median_valid_days": float(q["valid_days"].median()),
            "median_longest_valid_run": float(q["longest_valid_run"].median()),
        }
        for flag in ["eligible_cgm", "eligible_open_loop", "eligible_insulin", "eligible_open_loop_insulin"]:
            row[flag] = int(q[flag].sum())
        for days in (98, 180):
            row[f"open_loop_insulin_valid_ge_{days}"] = int((q["valid_days_open_insulin"] >= days).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_sensitivity(quality: pd.DataFrame, min_valid: int) -> pd.DataFrame:
    rows = []
    for dataset in [d for d in DATASET_ORDER if d in set(quality["dataset"])]:
        q = quality[quality["dataset"] == dataset]
        for value in COVERAGE_THRESHOLDS:
            column = f"valid_days_at_{int(value * 100)}"
            if column not in q:
                continue
            days = q[column].fillna(0)
            rows.append({"dataset": dataset, "threshold": value, "valid_days": int(days.sum()),
                         f"patients_ge_{min_valid}": int((days >= min_valid).sum()),
                         "patients_ge_98": int((days >= 98).sum())})
    return pd.DataFrame(rows)


def summarise_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    if gaps.empty:
        return pd.DataFrame()
    table = gaps.groupby(["dataset", "gap_class"], as_index=False)[["gaps", "missing_min"]].sum()
    total = table.groupby("dataset")["missing_min"].transform("sum")
    table["missing_hours"] = (table["missing_min"] / 60).round(1)
    table["share_of_missing_time"] = (table["missing_min"] / total).round(4)
    order = {label: k for k, (label, _) in enumerate(GAP_CLASSES)}
    table["order"] = table["gap_class"].map(order)
    table["dataset_order"] = table["dataset"].map(DATASET_ORDER.index)
    return table.sort_values(["dataset_order", "order"]).drop(columns=["order", "dataset_order", "missing_min"])


def coverage_histogram(days: dict) -> pd.DataFrame:
    edges = np.linspace(0, 1, 11)
    rows = []
    for dataset, frame in days.items():
        coverage = frame["coverage"].to_numpy()
        counts, _ = np.histogram(coverage, bins=edges)
        for k in range(10):
            rows.append({"dataset": dataset, "coverage_from": edges[k], "coverage_to": edges[k + 1],
                         "days": int(counts[k]), "share": round(float(counts[k] / len(coverage)), 4)})
    return pd.DataFrame(rows)


def loop_period_table(days: dict) -> pd.DataFrame:
    frames = []
    for dataset, frame in days.items():
        grouped = frame.groupby(["loop", "period"]).agg(
            days=("valid", "size"), valid_days=("valid", "sum"),
            valid_insulin_complete=("insulin_complete", lambda s: int((s & frame.loc[s.index, "valid"]).sum())),
            patients=("patient_id", "nunique")).reset_index()
        frames.append(grouped.assign(dataset=dataset))
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return table[["dataset", "loop", "period", "patients", "days", "valid_days", "valid_insulin_complete"]]


def duplicate_table(results: list) -> pd.DataFrame:
    """Value differences of what rules 3-4 dropped: repeated readings should be (almost) identical."""
    rows = []
    for dataset in DATASET_ORDER:
        mine = [r for r in results if r.dataset == dataset]
        if not mine:
            continue
        row = {"dataset": dataset}
        for name, attribute in [("duplicate", "duplicate_diffs"), ("conflict", "conflict_diffs"),
                                ("cross_stream", "stream_diffs")]:
            values = [getattr(r, attribute) for r in mine if getattr(r, attribute).size]
            d = np.concatenate(values) if values else np.zeros(0)
            row[f"{name}_sampled"] = int(d.size)
            row[f"{name}_abs_diff_median"] = float(np.median(d)) if d.size else np.nan
            row[f"{name}_abs_diff_p95"] = float(np.percentile(d, 95)) if d.size else np.nan
        row["duplicate_share_identical"] = round(float(np.mean(np.concatenate(
            [r.duplicate_diffs for r in mine if r.duplicate_diffs.size] or [np.zeros(0)]) == 0)), 4) \
            if row["duplicate_sampled"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ figures


def plot_funnel(dataset_quality: pd.DataFrame, min_valid: int) -> plt.Figure:
    datasets = [d for d in DATASET_ORDER if d in set(dataset_quality["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets), 4.2), sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        row = dataset_quality[dataset_quality["dataset"] == dataset].iloc[0]
        values = [row[key] for key, _ in FUNNEL]
        labels = [label.format(n=min_valid) for _, label in FUNNEL]
        y = np.arange(len(values))[::-1]
        ax.barh(y, values, color=DATASET_COLORS[dataset], height=0.7)
        for yy, value in zip(y, values):
            ax.text(value, yy, f" {int(value)}", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_yticks(y, labels)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontsize=10)
        ax.set_xlim(0, max(values) * 1.3 if max(values) else 1)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Patients at each eligibility step", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_sensitivity(sensitivity: pd.DataFrame, threshold: float, min_valid: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for dataset in [d for d in DATASET_ORDER if d in set(sensitivity["dataset"])]:
        s = sensitivity[sensitivity["dataset"] == dataset]
        base = s.loc[np.isclose(s["threshold"], threshold), "valid_days"]
        style = {"color": DATASET_COLORS[dataset], "ls": DATASET_LINESTYLES[dataset], "lw": 2, "marker": "o",
                 "markersize": 4, "label": DATASET_LABELS[dataset]}
        if len(base) and base.iloc[0]:
            axes[0].plot(100 * s["threshold"], 100 * s["valid_days"] / base.iloc[0], **style)
        total = s[f"patients_ge_{min_valid}"].max()
        if total:
            axes[1].plot(100 * s["threshold"], 100 * s[f"patients_ge_{min_valid}"] / total, **style)
    for ax in axes:
        ax.axvline(100 * threshold, color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_xlabel("minimum daily CGM coverage for a valid day (%)")
    axes[0].set_ylabel(f"valid days (% of those at {int(100 * threshold)} %)")
    axes[0].set_title("Valid days by coverage threshold", loc="left", fontsize=10)
    axes[1].set_ylabel(f"patients with >= {min_valid} valid days (% of the most)")
    axes[1].set_title(f"Patients with >= {min_valid} valid days by coverage threshold", loc="left", fontsize=10)
    axes[0].legend(ncols=2)
    fig.tight_layout()
    return fig


def plot_gaps(gaps: pd.DataFrame) -> plt.Figure:
    datasets = [d for d in DATASET_ORDER if d in set(gaps["dataset"])]
    labels = [label for label, _ in GAP_CLASSES]
    table = gaps.pivot(index="dataset", columns="gap_class", values="share_of_missing_time").reindex(datasets)[labels]
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(datasets) + 1.8))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f4f2"] + SEQUENTIAL_BLUE)
    ax.imshow(table.fillna(0).to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(datasets)):
        for j in range(len(labels)):
            v = table.iloc[i, j]
            if pd.notna(v) and v >= 0.001:
                ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    ax.grid(False)
    ax.set_title("Where the missing CGM time is: % of missing time by gap length", loc="left")
    fig.tight_layout()
    return fig


def plot_coverage(hist: pd.DataFrame, threshold: float) -> plt.Figure:
    datasets = [d for d in DATASET_ORDER if d in set(hist["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.8 * len(datasets), 3.6), sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        h = hist[hist["dataset"] == dataset]
        ax.bar(100 * h["coverage_from"], 100 * h["share"], width=9, align="edge", color=DATASET_COLORS[dataset])
        ax.axvline(100 * threshold, color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontsize=10)
        ax.set_xlabel("daily coverage (%)")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("% of days")
    fig.suptitle("Daily CGM coverage (days between each patient's first and last reading)", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_timelines(days: dict, quality: pd.DataFrame, per_dataset: int = 60) -> plt.Figure:
    datasets = [d for d in DATASET_ORDER if d in days]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 2.4 * len(datasets)), squeeze=False)
    colors = list(TIMELINE_COLORS.values())
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    for ax, dataset in zip(axes[:, 0], datasets):
        frame = days[dataset]
        order = quality[quality["dataset"] == dataset].sort_values("valid_days", ascending=False)["patient_id"]
        order = list(order[order.isin(frame["patient_id"])])[:per_dataset]
        width = int(frame.loc[frame["patient_id"].isin(order), "day_index"].max()) + 1
        grid = np.full((len(order), width), 3)
        for i, patient_id in enumerate(order):
            p = frame[frame["patient_id"] == patient_id]
            code = np.where(~p["valid"], 2, np.where(p["loop"] == "closed", 1, 0))
            code = np.where(p["n_readings"] == 0, 3, code)
            grid[i, p["day_index"].to_numpy()] = code
        ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set_title(f"{DATASET_LABELS[dataset]}: {len(order)} patients with most valid days", loc="left", fontsize=10)
        ax.set_ylabel("patients")
        ax.set_yticks([])
        ax.grid(False)
    axes[-1, 0].set_xlabel("days since the patient's first CGM reading")
    handles = [plt.matplotlib.patches.Patch(facecolor=c, edgecolor="#9b9a96", label=label) for c, label in
               zip(colors, ["valid day, open loop", "valid day, closed loop", "day with too few readings",
                            "no readings"])]
    fig.suptitle("Follow-up of each patient, day by day", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles=handles, loc="upper right", ncols=4, fontsize=8, frameon=False, bbox_to_anchor=(1, 1))
    return fig


# ============================================================================ summary


def write_summary(dataset_quality: pd.DataFrame, sensitivity: pd.DataFrame, gaps: pd.DataFrame,
                  loop_days: pd.DataFrame, duplicates: pd.DataFrame, task_log: pd.DataFrame, quality: pd.DataFrame,
                  threshold: float, min_valid: int, release: str, resources, derived: Path) -> str:
    lines = ["# 03 - Data quality", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Clean data in `{derived / 'clean'}`. Valid day: coverage >= {threshold:.0%}; "
             f"eligibility threshold: {min_valid} valid days (decisions D45 and D48).", ""]
    problems = []
    for _, row in task_log[task_log["status"] != "ok"].iterrows():
        problems.append(f"{row['dataset']}/{row['patient_id']} failed: {row['error']}")
    warned = task_log[task_log["warnings"].fillna("") != ""]
    for _, row in warned.head(10).iterrows():
        problems.append(f"{row['dataset']}/{row['patient_id']} raised warnings: {row['warnings'][:300]}")
    if not dataset_quality.empty:
        for _, row in dataset_quality[dataset_quality["removed_isolated_conflicts_share"] > 0.01].iterrows():
            problems.append(f"{row['dataset']}: {row['removed_isolated_conflicts_share']:.1%} of readings dropped as "
                            "isolated conflicts (readings closer than 0.6 intervals with different values; with "
                            "15-min Libre history these are scans exported among the history)")
        for _, row in dataset_quality[dataset_quality["removed_stream_disagreement_share"] > 0.01].iterrows():
            problems.append(f"{row['dataset']}: {row['removed_stream_disagreement_share']:.1%} of readings dropped on "
                            f"{row['stream_disagreement_days']} days when two upload streams of one sensor disagreed")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not dataset_quality.empty:
        table = dataset_quality.set_index("dataset").T
        table.columns = [DATASET_LABELS.get(c, c) for c in table.columns]
        lines += ["## Per dataset", "", markdown_table(table.reset_index().rename(columns={"index": "metric"})), ""]
    if not sensitivity.empty:
        lines += ["## Coverage threshold", "", markdown_table(sensitivity.assign(
            dataset=sensitivity["dataset"].map(DATASET_LABELS))), ""]
    if not loop_days.empty:
        lines += ["## Days by loop status and study period", "", markdown_table(loop_days.assign(
            dataset=loop_days["dataset"].map(DATASET_LABELS))), ""]
    if not duplicates.empty:
        lines += ["## Readings dropped as repeated (absolute value difference to the reading kept, mg/dL)", "",
                  "`duplicate`: same source, <= 5 mg/dL apart; `conflict`: same source, closer than 0.6 intervals "
                  "but different values (isolated or in clock-overlap stretches); `cross_stream`: the same reading in "
                  "two upload streams of one sensor (Loop).", "",
                  markdown_table(duplicates.assign(dataset=duplicates["dataset"].map(DATASET_LABELS))), ""]
    if not gaps.empty:
        lines += ["## Gaps", "", markdown_table(gaps.assign(dataset=gaps["dataset"].map(DATASET_LABELS))), ""]
    lines += ["## Checks to look at", "",
              "- `eligibility_funnel.png`: how many patients each criterion keeps.",
              "- `coverage_threshold_sensitivity.png`: whether 70 % is a safe choice (flat curves = robust).",
              "- `followup_timelines.png`: structure of each follow-up (run-in, randomisation, gaps, loop changes).",
              "- `duplicates.csv` and `clock_overlap_stretches` in `dataset_quality.csv`: repeated readings, and the "
              "stretches where two series shared clock times (only the series running before them is kept; the "
              "first reading after each one is marked `clock_break`).",
              "- `loop_period_days.csv`: open-loop days available per dataset and period.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
