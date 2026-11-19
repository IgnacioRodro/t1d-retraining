"""13 - Stage 0 for foundation models used zero-shot (the "context only" mechanism).

Runs Chronos-2 (with and without IOB/COB covariates), TimesFM 2.5 and Tiny
Time Mixers r2 on the screen group of the core dataset, in both follow-up
periods, on the same origins as scripts 10 and 12 (D58, D70). They are not
trained on the patient: each forecast reads the ``context_hours`` hours of
glucose up to the origin. Persistence and AR(3) run here too; the learned
models of script 12 (``compare_with``) are added from its saved day metrics,
so that every comparison is on the same patients and origins.

Paired comparisons give each foundation model against AR(3) and against
fine-tuned LightGBM, and Chronos-2 with covariates against Chronos-2 without
them. The tie threshold of a pair is the larger DMD of its two models: a learned
model's own DMD in script 12, and for models without seed noise (foundation,
AR(3)) the smallest DMD of the full learned models compared (D75). Reports RMSE and
Clarke zones for every model (rule R6).

The weights are downloaded from the Hugging Face Hub the first time and loaded
once per worker process. GPU memory limits the parallel workers to
``workers_per_gpu`` per GPU (rule R5); without CUDA, two CPU workers. The
confirm group is never touched.

Derived data:
    <derived_data>/runs/13_stage0_foundation/day_metrics.parquet

Outputs (results/13_stage0_foundation/):
    model_summary.csv        per period, model and horizon: patient RMSE and skill with bootstrap intervals
    metrics_overall.csv      per period, model, seed and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    metrics_by_patient.csv   per patient, period, model, seed and horizon
    paired_comparisons.csv   per period, horizon and pair: skill difference, interval, verdict
    fit_log.csv              per patient, period and model: context, seconds, device
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6)
    task_log.csv
    rmse_by_horizon.png
    skill_by_patient.png
    paired_comparisons.png
    clarke_grids.png
    summary.md
    run_manifest.json

Usage:
    python scripts/13_stage0_foundation.py
    python scripts/13_stage0_foundation.py --patients 2                  # quick check that the weights load
    python scripts/13_stage0_foundation.py --models chronos2 --context-hours 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import pandas as pd  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.splits import group_patients  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import (FIT_NOTE, PERIOD_LABELS, DMDTable, fit_outcomes,  # noqa: E402
                                           model_summary, overall_table, paired_comparisons, patient_table,
                                           plot_grids, plot_paired, plot_rmse, plot_skill, read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import set_style  # noqa: E402
from t1dretrain.sim.stage0 import CONTEXT_MODELS  # noqa: E402

SCRIPT_ID = "13_stage0_foundation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of foundation.models")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients (quick check)")
    parser.add_argument("--context-hours", type=float, default=None, help="history read by every model")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_job(job: dict):
    from t1dretrain.sim.stage0 import evaluate_patient

    with WarningRecorder() as recorder:
        result = evaluate_patient(job)
    result.warnings = recorder.text()
    return result


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    print(f"GPU sharing: at most {share_gpus(config)} processes compute on one GPU at a time (D73)")
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    core = config.get("core_dataset", "replace_bg")
    stage, foundation = dict(config["stage0"]), config["foundation"]
    chosen = [m for m in foundation["models"] if m in CONTEXT_MODELS and (not args.models or m in args.models)]
    models = ["persistence", "ar3"] + chosen
    hours = float(args.context_hours or foundation["context_hours"])
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]), "warmup_days": int(config["warmup_days"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]), **stage, "models": models,
                "seeds": [0], "context_hours": hours, "foundation_batch_size": int(foundation["batch_size"])}
    for m in chosen:
        settings[m] = {**(foundation.get("settings", {}).get(m) or {}), "context_hours": hours}
    group = foundation.get("group", "screen")
    core_period = config.get("core_period", "post_randomization")
    patients = group_patients(core, group)
    if args.patients:
        patients = patients[: args.patients]
    anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == "core")].drop_duplicates(["patient_id", "period"])
    day0 = {(r.patient_id, r.period): r.day0 for r in anchors.itertuples()}
    periods = [core_period] + [p for p in ["post_randomization", "whole"] if p != core_period]
    if resources.device == "cuda" and resources.n_gpus:
        devices = [f"cuda:{k}" for k in range(resources.n_gpus)]
        n_workers = min(resources.n_workers, len(devices) * int(foundation["workers_per_gpu"]))
    else:
        devices, n_workers = ["cpu"], min(resources.n_workers, 2)
    device = "cuda" if devices != ["cpu"] else "cpu"  # each worker process uses one GPU (D73)
    jobs = [{"dataset": core, "patient_id": pid, "derived_root": derived, "settings": settings, "stream": k,
             "device": device, "day0": {p: day0.get((pid, p)) for p in periods}}
            for k, pid in enumerate(patients)]
    print(f"{len(jobs)} patients, models {chosen}, context {hours:g} h, {n_workers} workers on {devices}")
    results = parallel_map(run_job, jobs, n_workers, label="patients")

    task_log = pd.DataFrame([{"patient_id": r.patient_id, "status": "error" if r.error else "ok", "seconds": r.seconds,
                              "warnings": r.warnings, "error": r.error.strip().splitlines()[-1] if r.error else "",
                              "traceback": r.error} for r in results])
    ok = [r for r in results if not r.error]
    days = pd.concat([r.days for r in ok if not r.days.empty], ignore_index=True) if ok else pd.DataFrame()
    runs = derived / "runs" / SCRIPT_ID
    runs.mkdir(parents=True, exist_ok=True)
    days.to_parquet(runs / "day_metrics.parquet", index=False)
    clarke = ClarkeAccumulator(["model", "seed", "horizon_min", "period"], sample_size=settings["clarke_point_sample"])
    for r in ok:
        clarke.merge(r.clarke)
    fit_log = pd.DataFrame([row for r in ok for row in r.info])

    learned, note = learned_from_12(derived, foundation.get("compare_with", []), patients)
    combined = pd.concat([days, learned], ignore_index=True) if not learned.empty else days
    replicates, seed = int(stage["bootstrap_replicates"]), int(config["random_seed"])
    dmd = dmd_from_12(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv",
                      foundation.get("compare_with", []))
    by_patient = patient_table(combined) if not combined.empty else pd.DataFrame()
    overall = overall_table(combined, by_patient, replicates, seed) if not combined.empty else pd.DataFrame()
    summary = model_summary(by_patient, replicates, seed) if not by_patient.empty else pd.DataFrame()
    pairs = [(m, "ar3") for m in chosen] + [(m, "lightgbm_ft") for m in chosen]
    if "chronos2" in chosen and "chronos2_univariate" in chosen:
        pairs.append(("chronos2", "chronos2_univariate"))
    paired = paired_comparisons(by_patient, dmd, replicates, seed, pairs) if not by_patient.empty else pd.DataFrame()
    histogram = clarke.histogram()

    out.csv(summary, "model_summary")
    out.csv(overall, "metrics_overall")
    out.csv(by_patient, "metrics_by_patient")
    out.csv(paired, "paired_comparisons")
    out.csv(fit_log, "fit_log")
    out.csv(fit_outcomes(fit_log), "fit_outcomes")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(histogram, "clarke_histogram")
    out.csv(clarke.sample(), "clarke_points_sample")
    out.csv(task_log, "task_log")
    if not summary.empty:
        present = [p for p in periods if p in set(summary["period"])]
        shown = core_period if core_period in present else present[0]
        order = ["persistence", "ar3"] + [m for m in foundation.get("compare_with", []) if m in set(summary["model"])] \
            + chosen
        shown_models = [m for m in order if m in set(summary["model"])]
        out.figure(plot_rmse(summary, present, shown_models, "Stage 0: foundation models zero-shot against the "
                             "references and the fine-tuned learned models, screen group"), "rmse_by_horizon")
        out.figure(plot_skill(by_patient, shown, shown_models), "skill_by_patient")
        if not paired.empty:
            out.figure(plot_paired(paired, shown), "paired_comparisons")
        grid_models = ["ar3"] + [m for m in chosen if m in set(summary["model"])]
        out.figure(plot_grids(histogram, grid_models, settings["horizons_min"], shown, 0), "clarke_grids")
    out.text(write_summary(summary, overall, paired, fit_log, task_log, settings, chosen, hours, group, len(patients),
                           core_period, note, n_workers, devices, release, resources), "summary.md")
    out.manifest(settings={k: v for k, v in settings.items() if k not in chosen} | {"group": group,
                           "patients": len(patients), "core_period": core_period, "foundation_models": chosen},
                 notes="One job per patient; day metrics in <derived>/runs/13_stage0_foundation.")
    print(f"Done. Results in {out.dir}")


def learned_from_12(derived: Path, models: list[str], patients: list[str]) -> tuple[pd.DataFrame, str]:
    """Day metrics of the learned models of script 12 on the same patients (same origins)."""
    path = derived / "runs" / "12_stage0_population_transfer" / "day_metrics.parquet"
    if not models:
        return pd.DataFrame(), "no learned models requested"
    if not path.exists():
        return pd.DataFrame(), f"{path} not found: run script 12 to compare with the learned models"
    days = pd.read_parquet(path)
    days = days[days["model"].isin(models) & days["patient_id"].isin(patients)]
    found = sorted(days["model"].unique())
    return days, f"learned models from script 12: {', '.join(found) or 'none found'}"


def dmd_from_12(path: Path, learned: list[str]) -> DMDTable:
    """DMD of each learned model compared here, from script 12 (D75); the foundation models and AR(3) take the
    smallest DMD of the full models among them (twins do not set it, D90). Empty (no ties) if script 12 has not run
    here."""
    return DMDTable(read_noise_floor(path), learned)


def write_summary(summary, overall, paired, fit_log, task_log, settings, chosen, hours, group, n_patients, core_period,
                  note, n_workers, devices, release, resources) -> str:
    lines = ["# 13 - Stage 0 for foundation models used zero-shot", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Group `{group}` of the core dataset ({n_patients} patients); models {chosen}; context {hours:g} h of "
             f"glucose up to each origin (Chronos-2 also IOB, COB and the time of day); origins every "
             f"{settings['origin_stride_min']:.0f} min on usable days; horizons {settings['horizons_min']} min; "
             f"{n_workers} workers on {devices}. {note}. Core period: `{core_period}`.", ""]
    problems = []
    for _, row in task_log[task_log["status"] != "ok"].iterrows():
        problems.append(f"{row['patient_id']} failed: {row['error']}")
    for _, row in task_log[task_log["warnings"].fillna("") != ""].head(10).iterrows():
        problems.append(f"{row['patient_id']} raised warnings: {row['warnings'][:300]}")
    if not fit_log.empty and "error" in fit_log:
        failed = fit_log[fit_log["error"].notna()]
        for (model, error), g in failed.groupby(["model", "error"]):
            problems.append(f"{model} failed for {len(g)} patient-period(s) ({', '.join(g['patient_id'].unique()[:5])}):"
                            f" {error}")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    outcomes = fit_outcomes(fit_log)
    if not outcomes.empty:
        lines += ["## Fits (a run is complete when `other_failure` is 0)", "", markdown_table(outcomes), "", FIT_NOTE, ""]
    if not summary.empty:
        lines += ["## Models (patient means; learned models of script 12 averaged over seeds; 95 % bootstrap "
                  "intervals)", "", markdown_table(summary), ""]
    if not paired.empty:
        core = paired[paired["period"] == core_period]
        lines += [f"## Paired comparisons, {PERIOD_LABELS.get(core_period, core_period)} (tie = below the DMD)", "",
                  markdown_table(core.drop(columns=["period"])), "", "The other period is in `paired_comparisons.csv`.",
                  ""]
    if not fit_log.empty and "predict_seconds" in fit_log:
        timing = fit_log.groupby("model")[["predict_seconds", "test_origins"]].median().round(2).reset_index()
        lines += ["## Median seconds per patient and period (prediction only)", "", markdown_table(timing), ""]
    if not overall.empty:
        show = ["period", "model", "seed", "horizon_min", "n", "rmse", "mae", "mard_pct", "bias", "clarke_A_pct",
                "clarke_B_pct", "clarke_AB_pct", "clarke_C_pct", "clarke_D_pct", "clarke_E_pct"]
        lines += ["## Pooled forecasts (R6)", "", markdown_table(overall[show]), ""]
    lines += ["## Checks to look at", "",
              "- Warnings: a model that failed to load (no access to the Hugging Face Hub, out of GPU memory) is listed "
              "there; the others still run.",
              "- `rmse_by_horizon.png` and `paired_comparisons.png`: where zero-shot models stand against AR(3) and the "
              "fine-tuned learned models.",
              "- `clarke_grids.png`: the grid figure for every model (R6).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
