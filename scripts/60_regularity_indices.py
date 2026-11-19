"""60 - Q4: regularity of every confirmation patient's habits, measured at the start of the follow-up.

For every patient of the confirmation (script 50), on the usable days of the
startup window (the ``window_days`` days before the anchor, the days the first
model is trained on), computes indices of how regular the patient's meals,
boluses, glucose profile and (where there are wearables) steps and sleep are,
and the glycaemic control used to adjust for them (HANDOFF 7.7, D94;
``features.regularity``). The indices are known when a refit policy would be
chosen. Cohorts without carbohydrates (DCLP3) or without insulin have only some
of them. Script 61 relates them to what refitting gained for each patient.
This script evaluates no model (rule R6 applies to script 61).

Inputs:
    results/50_confirmation_runs/jobs.csv                  (patients, cohorts and anchors of the confirmation)
    <derived_data>/clean/<dataset>/days.parquet           (usable days, script 03-04)
    <derived_data>/features/<dataset>/grid/                (glucose on the patient's grid, script 05)
    <derived_data>/processed/<dataset>/{carbs, insulin, wearables, sleep}/   (events, script 02)

Derived data:
    <derived_data>/runs/60_regularity_indices/indices.parquet

Outputs (results/60_regularity_indices/):
    indices.csv              per cohort and patient: days used, events, every index and the glycaemic control
    index_summary.csv        per cohort and index: patients, median and quartiles
    index_correlations.csv   Spearman correlation between the indices (all patients)
    task_log.csv
    index_distributions.png, index_correlations.png, example_meals.png
    summary.md
    run_manifest.json

Usage:
    python scripts/60_regularity_indices.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.splits import cohort_rule, usable_flags  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.features.regularity import ADJUSTERS, INDICES  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402

SCRIPT_ID = "60_regularity_indices"
SOURCE = "50_confirmation_runs"
INSULIN_RULES = ("open_insulin", "closed_insulin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_job(job: dict) -> dict:
    """The indices of one patient (in a worker process)."""
    import time
    import traceback

    from t1dretrain.features.regularity import patient_indices

    started = time.time()
    row = {k: job[k] for k in ("dataset", "cohort", "patient_id", "period", "anchor", "usable_rule")}
    with WarningRecorder() as recorder:
        try:
            features = ProcessedStore(job["dataset"], root=job["derived_root"], stage="features")
            source = ProcessedStore(job["dataset"], root=job["derived_root"])
            grid = features.read_frame("grid", job["patient_id"])
            if grid.empty:
                raise LookupError("no feature grid")
            if "day" not in grid:
                grid["day"] = grid["ts"].dt.normalize()
            days = list(pd.DatetimeIndex(job["days"]))
            row.update(patient_indices(grid, source.read("insulin", job["patient_id"]),
                                       source.read("carbs", job["patient_id"]),
                                       source.read("wearables", job["patient_id"]),
                                       source.read("sleep", job["patient_id"]), days, job["settings"],
                                       uses_insulin=job["usable_rule"] in INSULIN_RULES))
            error = ""
        except Exception:  # noqa: BLE001
            error = traceback.format_exc(limit=4)
    return {"row": row, "status": "error" if error else "ok", "error": error, "warnings": recorder.text()[:300],
            "seconds": round(time.time() - started, 2)}


def window_days(days: pd.DataFrame, patient_id: str, anchor: pd.Timestamp, period: str, rule: str,
                length: int) -> list:
    """The usable days of the cohort's rule in the ``length`` days before the anchor (and in the period)."""
    d = days[days["patient_id"] == patient_id]
    d = d[(d["day"] >= anchor - pd.Timedelta(days=length)) & (d["day"] < anchor)]
    if period != "whole" and "period" in d:
        d = d[d["period"] == period]
    usable = usable_flags(d)[f"usable_{rule}"].to_numpy(bool) if not d.empty else np.zeros(0, dtype=bool)
    return list(pd.DatetimeIndex(d["day"])[usable])


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    settings = dict(config["regularity"])
    length = int(config["anchors"]["train_window_days"])
    jobs_path = Path(out.dir).parent / SOURCE / "jobs.csv"
    if not jobs_path.exists():
        raise SystemExit(f"{jobs_path} not found: run scripts/50_confirmation_runs.py first")
    patients = pd.read_csv(jobs_path)
    patients["dataset"] = patients["cohort"].str.split("/").str[0]
    patients["anchor"] = pd.to_datetime(patients["anchor"])

    jobs, day_tables = [], {}
    for r in patients.itertuples():
        if r.dataset not in day_tables:
            day_tables[r.dataset] = pd.read_parquet(derived / "clean" / r.dataset / "days.parquet")
            day_tables[r.dataset]["day"] = pd.to_datetime(day_tables[r.dataset]["day"])
        rule = cohort_rule(r.dataset, r.cohort.split("/")[1])
        days = window_days(day_tables[r.dataset], r.patient_id, r.anchor, r.period, rule, length)
        jobs.append({"dataset": r.dataset, "cohort": r.cohort, "patient_id": r.patient_id, "period": r.period,
                     "anchor": r.anchor, "usable_rule": rule, "days": days, "settings": settings,
                     "derived_root": derived})
    print(f"{len(jobs)} patients in {patients['cohort'].nunique()} cohorts; startup window of {length} days")
    results = parallel_map(run_job, jobs, resources.n_workers, label="patients")

    table = pd.DataFrame([r["row"] for r in results])
    enough = table["window_days_used"].fillna(0) >= int(settings["min_window_days"])
    for column in list(INDICES) + list(ADJUSTERS):
        if column not in table:
            table[column] = np.nan
        table.loc[~enough, column] = np.nan
    task_log = pd.DataFrame([{"cohort": r["row"]["cohort"], "patient_id": r["row"]["patient_id"],
                              "status": r["status"], "seconds": r["seconds"], "warnings": r["warnings"],
                              "error": r["error"].strip().splitlines()[-1] if r["error"] else "",
                              "traceback": r["error"]} for r in results])
    order = list(dict.fromkeys(patients["cohort"]))
    summary = index_summary(table, order)
    names = [c for c in INDICES if table[c].notna().sum() >= 10]
    pairs_min = int(settings["min_patients"])  # a correlation needs as many patients as an association (D94)
    correlations = (table[names + list(ADJUSTERS)].corr(method="spearman", min_periods=pairs_min).round(3)
                    if names else pd.DataFrame())

    runs = derived / "runs" / SCRIPT_ID
    runs.mkdir(parents=True, exist_ok=True)
    table.to_parquet(runs / "indices.parquet", index=False)
    out.csv(table.round({c: 4 for c in table.select_dtypes("number").columns}), "indices")
    out.csv(summary.round(4), "index_summary")
    out.csv(correlations.reset_index().rename(columns={"index": "index_name"}), "index_correlations")
    out.csv(task_log, "task_log")
    if names:
        out.figure(plot_distributions(table, names, order), "index_distributions")
        out.figure(plot_correlations(correlations), "index_correlations")
    example = plot_examples(table, derived, jobs, settings)
    if example is not None:
        out.figure(example, "example_meals")
    out.text(write_summary(table, summary, correlations, task_log, order, settings, length, release, resources),
             "summary.md")
    out.manifest(settings={**settings, "window_days": length, "patients": int(len(table)), "cohorts": order},
                 notes="Indices on the startup window of every confirmation patient; script 61 relates them to the "
                       "confirmation's outcomes.")
    print(f"Done. Results in {out.dir}")


def index_summary(table: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for cohort in order:
        c = table[table["cohort"] == cohort]
        for name in list(INDICES) + list(ADJUSTERS):
            v = c[name].dropna()
            if v.empty:
                continue
            rows.append({"cohort": cohort, "index": name, "patients": int(len(v)), "median": v.median(),
                         "q1": v.quantile(0.25), "q3": v.quantile(0.75)})
    return pd.DataFrame(rows)


# ============================================================================ figures and summary


def plot_distributions(table: pd.DataFrame, names: list[str], order: list[str]):
    """Every index per cohort over patients: median (dot), quartiles (thick line) and 5th-95th percentiles."""
    fig, axes = model_axes(len(names), 4.2, 2.8, ncols=4)
    for ax, name in zip(axes, names):
        labels = []
        for cohort in order:
            v = table.loc[table["cohort"] == cohort, name].dropna()
            if len(v) < 3:
                continue
            y = len(labels)
            q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
            ax.plot([q[0], q[4]], [y, y], color=TEXT_SECONDARY, lw=0.8)
            ax.plot([q[1], q[3]], [y, y], color="#2a78d6", lw=4, solid_capstyle="butt")
            ax.scatter(q[2], y, color="#d1495b", s=14, zorder=3)
            labels.append(f"{cohort} ({len(v)})")
        if labels:
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
        group, unit, regular, _ = INDICES[name]
        ax.set_title(f"{name}" + (f" ({unit})" if unit else ""), loc="left", fontsize=8)
        ax.set_xlabel(f"more regular: {regular}" if regular else "descriptor", fontsize=7, color=TEXT_SECONDARY)
    fig.suptitle("Regularity at the start of the follow-up (startup window), per cohort (D94)", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_correlations(correlations: pd.DataFrame):
    import matplotlib.pyplot as plt

    names = list(correlations.columns)
    fig, ax = plt.subplots(figsize=(0.45 * len(names) + 3, 0.45 * len(names) + 2))
    image = ax.imshow(correlations.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            v = correlations.iloc[i, j]
            if np.isfinite(v) and i != j:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(image, ax=ax, shrink=0.6, label="Spearman correlation")
    ax.set_title("How the indices relate to each other and to glycaemic control (all patients)", loc="left",
                 fontsize=9)
    fig.tight_layout()
    return fig


def plot_examples(table: pd.DataFrame, derived: Path, jobs: list[dict], settings: dict):
    """Meal clock times per day of the most and the least regular patient (meal timing) of the first cohort with
    meals: a check that the index measures what it should."""
    import matplotlib.pyplot as plt

    from t1dretrain.features.regularity import merge_events

    with_meals = table.dropna(subset=["meal_time_jitter_min"])
    if with_meals.empty:
        return None
    cohort = with_meals["cohort"].iloc[0]
    c = with_meals[with_meals["cohort"] == cohort].sort_values("meal_time_jitter_min")
    picks = [c.iloc[0], c.iloc[-1]]
    by_key = {(j["cohort"], j["patient_id"]): j for j in jobs}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, row in zip(axes, picks):
        job = by_key[(row["cohort"], row["patient_id"])]
        carbs = ProcessedStore(job["dataset"], root=derived).read("carbs", row["patient_id"])
        carbs = carbs[carbs["ts"].dt.normalize().isin(job["days"]) & (carbs["carbs_g"] >= float(settings["min_meal_g"]))]
        meals = merge_events(carbs["ts"], carbs["carbs_g"].to_numpy(float), float(settings["merge_min"]))
        day = (meals["ts"].dt.normalize() - min(job["days"])).dt.days
        hour = (meals["ts"] - meals["ts"].dt.normalize()).dt.total_seconds() / 3600
        ax.scatter(hour, day, s=np.clip(meals["size"], 10, 120), color="#2a78d6", alpha=0.7, edgecolors="none")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlabel("hour of the day")
        ax.set_title(f"{row['patient_id']}: meal timing jitter {row['meal_time_jitter_min']:.0f} min", loc="left",
                     fontsize=9)
    axes[0].set_ylabel("day of the startup window")
    fig.suptitle(f"Meals of the most and the least regular patient of {cohort} (dot size: grams)", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def write_summary(table, summary, correlations, task_log, order, settings, length, release, resources) -> str:
    lines = ["# 60 - Regularity of the confirmation patients at the start of the follow-up", "",
             f"Code release {release}.", f"Machine: {resources.describe()}",
             f"{len(table)} patients of the confirmation (script 50) in {len(order)} cohorts. Indices on the usable "
             f"days of the {length} days before the anchor (at least {settings['min_window_days']} days); meals of at "
             f"least {settings['min_meal_g']} g and boluses of at least {settings['min_bolus_u']} U, joined within "
             f"{settings['merge_min']} min (D94). No model is evaluated here.", ""]
    problems = [f"{r.cohort} {r.patient_id}: {r.error}" for r in task_log[task_log["status"] == "error"].itertuples()]
    short = table[table["window_days_used"].fillna(0) < int(settings["min_window_days"])]
    problems += [f"{len(short)} patients with fewer than {settings['min_window_days']} usable days in the window: no "
                 f"indices"] if len(short) else []
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    coverage = table.groupby("cohort", sort=False).agg(
        patients=("patient_id", "size"), window_days=("window_days_used", "median"),
        meals=("meal_time_jitter_min", "count"), boluses=("bolus_time_jitter_min", "count"),
        glucose=("glucose_profile_stability", "count"), steps=("steps_per_day_cv", "count"),
        sleep=("sleep_onset_sd_min", "count")).reindex(order).reset_index()
    lines += ["## Patients with each group of indices", "", markdown_table(coverage), ""]
    if not summary.empty:
        wide = summary.assign(cell=[f"{m:.3g} ({a:.3g}-{b:.3g})" for m, a, b in
                                    zip(summary["median"], summary["q1"], summary["q3"])])
        wide = wide.pivot_table(index="index", columns="cohort", values="cell", aggfunc="first")
        wide = wide.reindex(index=[i for i in list(INDICES) + list(ADJUSTERS) if i in wide.index],
                            columns=[c for c in order if c in wide.columns])
        lines += ["## Medians (quartiles) per cohort", "", markdown_table(wide.reset_index()), ""]
    if not correlations.empty:
        pairs = []
        names = list(correlations.columns)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                v = correlations.loc[a, b]
                if np.isfinite(v) and abs(v) >= 0.5:
                    pairs.append({"index": a, "with": b, "spearman": round(float(v), 2)})
        if pairs:
            pairs = pd.DataFrame(pairs)
            pairs = pairs.reindex(pairs["spearman"].abs().sort_values(ascending=False).index).head(20)
            lines += ["## Most related indices (|Spearman| at least 0.5; the 20 largest)", "", markdown_table(pairs),
                      "", "Every pair is in `index_correlations.csv`.", ""]
    lines += ["## Checks to look at", "",
              "- `example_meals.png`: that the most and the least regular patient look like it.",
              "- `index_correlations.png`: which indices measure the same thing, and how they relate to glycaemic "
              "control (TIR, CV, mean).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
