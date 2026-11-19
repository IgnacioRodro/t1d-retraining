"""10 - Stage 0: reference models, learned models and the noise floor.

Runs on the screen group of the core dataset, for every horizon and for both
follow-up periods (after randomisation, the provisional core period of D52,
and the whole follow-up):

- reference models: persistence, linear extrapolation of the last 30 min and
  AR(3) (D59);
- learned personal models, LightGBM (direct, one booster per horizon) and
  N-HiTS (neuralforecast network, whole path), each trained three times with
  different seeds (D60).

Every model is fitted once on the 14 days from day 0 (frozen; the last 2 days
for early stopping) and tested on every later day, on the same origins
(rules in src/t1dretrain/sim/ and decisions D58-D61 of HANDOFF.md). The noise
floor (minimum detectable difference, DMD) is the 95th percentile of the
absolute difference in mean skill between two trainings of the same model
that differ only in the seed, with patients resampled by bootstrap (D61).

Reports, as every evaluation of the project (rule R6), RMSE and the share of
forecasts in each Clarke Error Grid zone, and saves what is needed to redraw
the grids. Runs one job per patient on all cores but two, with N-HiTS on the
GPUs in turn when CUDA is present (rule R5). The confirm group is never
touched before scripts 50-51.

Derived data:
    <derived_data>/runs/10_stage0_baselines_noise/day_metrics.parquet   per patient, period, model, seed, horizon, day

Outputs (results/10_stage0_baselines_noise/):
    model_summary.csv        per period, model and horizon: mean patient RMSE and skill over persistence (seeds
                             averaged) with bootstrap intervals, and the spread between seeds
    metrics_overall.csv      per period, model, seed and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    metrics_by_patient.csv   per patient, period, model, seed and horizon: n, RMSE, skill, Clarke A+B
    noise_floor.csv          per period, learned model and horizon: differences between seeds and the DMD
    day_difficulty.csv       daily RMSE of persistence by period, horizon and week of follow-up
    fit_log.csv              per patient, period, model and seed: sizes, seconds, rounds or steps, device
    clarke_zones.csv         Clarke zone counts and shares per model, seed, horizon and period (R6)
    clarke_histogram.csv     sparse 5-mg/dL histogram of (reference, forecast) with the zone of each cell (R6)
    clarke_points_sample.csv random sample of (reference, forecast) pairs per panel (R6)
    task_log.csv
    rmse_by_horizon.png
    skill_by_patient.png
    noise_floor.png
    clarke_grids.png
    difficulty_by_week.png
    summary.md
    run_manifest.json

Usage:
    python scripts/10_stage0_baselines_noise.py
    python scripts/10_stage0_baselines_noise.py --models persistence ar3      # a subset of models
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
from t1dretrain.eval.stage0_report import (difficulty_table, model_summary, noise_floor, overall_table,  # noqa: E402
                                           patient_table, plot_difficulty, plot_grids, plot_noise, plot_rmse,
                                           plot_skill)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import set_style  # noqa: E402

SCRIPT_ID = "10_stage0_baselines_noise"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--group", default=None, choices=["screen", "pool"], help="core group (default: stage0.group)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of stage0.models")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_job(job: dict):
    from t1dretrain.sim.stage0 import evaluate_patient

    with WarningRecorder() as recorder:
        result = evaluate_patient(job)
    result.warnings = recorder.text()
    return result


def device_for(k: int, resources, choice: str) -> str:
    """N-HiTS device of the k-th patient job: a GPU when CUDA is present (rule R5; each worker process uses one
    GPU, D73), else the CPU."""
    if choice == "auto" and resources.device == "cuda" and resources.n_gpus:
        return "cuda"
    return "cpu" if choice == "auto" else choice


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
    stage = dict(config["stage0"])
    if args.models:
        stage["models"] = [m for m in stage["models"] if m in args.models]
    if "persistence" not in stage["models"]:
        stage["models"] = ["persistence"] + stage["models"]  # the reference of every skill
    group = args.group or stage["group"]
    core = config.get("core_dataset", "replace_bg")
    core_period = config.get("core_period", "post_randomization")
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]), "warmup_days": int(config["warmup_days"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]), **stage}

    patients = group_patients(core, group)
    anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == "core")].drop_duplicates(["patient_id", "period"])
    day0 = {(r.patient_id, r.period): r.day0 for r in anchors.itertuples()}
    periods = [core_period] + [p for p in ["post_randomization", "whole"] if p != core_period]
    jobs = [{"dataset": core, "patient_id": pid, "derived_root": derived, "settings": settings, "stream": k,
             "device": device_for(k, resources, str(stage.get("neural_device", "auto"))),
             "day0": {p: day0.get((pid, p)) for p in periods}} for k, pid in enumerate(patients)]
    results = parallel_map(run_job, jobs, resources.n_workers, label="patients")

    task_log = pd.DataFrame([{"patient_id": r.patient_id, "status": "error" if r.error else "ok", "seconds": r.seconds,
                              "warnings": getattr(r, "warnings", ""),
                              "error": r.error.strip().splitlines()[-1] if r.error else "", "traceback": r.error}
                             for r in results])
    ok = [r for r in results if not r.error]
    days = pd.concat([r.days for r in ok if not r.days.empty], ignore_index=True)
    runs = derived / "runs" / SCRIPT_ID
    runs.mkdir(parents=True, exist_ok=True)
    days.to_parquet(runs / "day_metrics.parquet", index=False)
    clarke = ClarkeAccumulator(["model", "seed", "horizon_min", "period"], sample_size=settings["clarke_point_sample"])
    for r in ok:
        clarke.merge(r.clarke)
    fit_log = pd.DataFrame([row for r in ok for row in r.info])
    replicates, seed = int(stage["bootstrap_replicates"]), int(config["random_seed"])

    by_patient = patient_table(days)
    overall = overall_table(days, by_patient, replicates, seed)
    summary = model_summary(by_patient, replicates, seed)
    noise = noise_floor(by_patient, replicates, seed)
    difficulty = difficulty_table(days)
    histogram = clarke.histogram()

    out.csv(summary, "model_summary")
    out.csv(overall, "metrics_overall")
    out.csv(by_patient, "metrics_by_patient")
    out.csv(noise, "noise_floor")
    out.csv(difficulty, "day_difficulty")
    for column in ("coef", "rounds", "added_rounds"):
        if column in fit_log:
            fit_log[column] = fit_log[column].astype(str)
    out.csv(fit_log, "fit_log")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(histogram, "clarke_histogram")
    out.csv(clarke.sample(), "clarke_points_sample")
    out.csv(task_log, "task_log")
    if not summary.empty:
        present = [p for p in periods if p in set(summary["period"])]
        shown = core_period if core_period in present else present[0]
        models = [m for m in settings["models"] if m in set(summary["model"])]
        out.figure(plot_rmse(summary, present, models, "Stage 0: frozen models by horizon, screen group (learned "
                             "models: mean of the seeds)"), "rmse_by_horizon")
        out.figure(plot_skill(by_patient, shown, models), "skill_by_patient")
        if not noise.empty:
            out.figure(plot_noise(noise, by_patient, shown), "noise_floor")
        out.figure(plot_grids(histogram, models, settings["horizons_min"], shown, settings["seeds"][0]), "clarke_grids")
        out.figure(plot_difficulty(difficulty, present), "difficulty_by_week")
    out.text(write_summary(summary, overall, noise, difficulty, fit_log, task_log, settings, group, len(patients),
                           core_period, release, resources), "summary.md")
    out.manifest(settings={**settings, "group": group, "patients": len(patients), "core_period": core_period},
                 notes="One job per patient; day metrics in <derived>/runs/10_stage0_baselines_noise.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ summary


def write_summary(summary: pd.DataFrame, overall: pd.DataFrame, noise: pd.DataFrame, difficulty: pd.DataFrame,
                  fit_log: pd.DataFrame, task_log: pd.DataFrame, settings: dict, group: str, n_patients: int,
                  core_period: str, release: str, resources) -> str:
    lines = ["# 10 - Stage 0: reference models, learned models and the noise floor", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Group `{group}` of the core dataset ({n_patients} patients). Frozen models fitted on the "
             f"{settings['warmup_days']} days from day 0 (learned models: last {settings['validation_days']} days for "
             f"early stopping; seeds {settings['seeds']}); origins every {settings['origin_stride_min']:.0f} min on "
             f"usable days (valid, open loop, insulin known); inputs {settings['lookback_hours']:.0f} h; horizons "
             f"{settings['horizons_min']} min. Core period (provisional, D52): `{core_period}`.", ""]
    problems = []
    for _, row in task_log[task_log["status"] != "ok"].iterrows():
        problems.append(f"{row['patient_id']} failed: {row['error']}")
    for _, row in task_log[task_log["warnings"].fillna("") != ""].head(10).iterrows():
        problems.append(f"{row['patient_id']} raised warnings: {row['warnings'][:300]}")
    if not fit_log.empty and "fitted" in fit_log:
        unfitted = fit_log[(fit_log["model"] == "ar3") & (fit_log["fitted"] == False)]  # noqa: E712
        for _, row in unfitted.iterrows():
            problems.append(f"{row['patient_id']} ({row['period']}): AR(3) not fitted (too few training runs); "
                            "persistence used instead")
    if not fit_log.empty and "error" in fit_log:
        failed = fit_log[fit_log["error"].notna()]
        for (model, error), g in failed.groupby(["model", "error"]):
            problems.append(f"{model} not fitted for {len(g)} patient-period(s) ({', '.join(g['patient_id'].unique()[:5])}): "
                            f"{error}")
    if not fit_log.empty and "cuda_error" in fit_log:
        for _, row in fit_log[fit_log["cuda_error"].notna()].head(5).iterrows():
            problems.append(f"{row['patient_id']}: N-HiTS fell back to the CPU ({row['cuda_error']})")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not summary.empty:
        lines += ["## Models (patient means; learned models averaged over seeds; 95 % bootstrap intervals)", "",
                  markdown_table(summary), ""]
    if not noise.empty:
        lines += ["## Noise floor (D61)", "",
                  "`dmd`: the change in mean skill that two trainings of the same model show by chance. Later stages "
                  "call a difference between two models a tie when it is below the larger DMD of the two (D75; "
                  "`dmd_stage`, the largest of the period and horizon, is for reference).", "", markdown_table(noise),
                  ""]
    if not fit_log.empty:
        timing = fit_log.groupby("model")[["fit_seconds", "predict_seconds"]].median().round(2).reset_index()
        lines += ["## Median seconds per patient and period", "", markdown_table(timing), ""]
    if not overall.empty:
        show = ["period", "model", "seed", "horizon_min", "n", "rmse", "mae", "mard_pct", "bias", "clarke_A_pct",
                "clarke_B_pct", "clarke_AB_pct", "clarke_C_pct", "clarke_D_pct", "clarke_E_pct"]
        lines += ["## Pooled forecasts per seed (R6)", "", markdown_table(overall[show]), ""]
    if not difficulty.empty:
        first = difficulty[difficulty["week"] <= 1].groupby(["period", "horizon_min"])["rmse_median"].mean()
        last = difficulty[(difficulty["patient_days"] >= 50)].sort_values("week").groupby(
            ["period", "horizon_min"]).tail(2).groupby(["period", "horizon_min"])["rmse_median"].mean()
        lines += ["## Day difficulty (daily RMSE of persistence, median)", "",
                  markdown_table(pd.DataFrame({"first_two_weeks": first, "last_two_weeks_with_50_days": last})
                                 .round(2).reset_index()), ""]
    lines += ["## Checks to look at", "",
              "- `rmse_by_horizon.png` and `skill_by_patient.png`: where each model stands against persistence.",
              "- `noise_floor.png`: the size of a difference that seeds alone can produce (the DMD).",
              "- `clarke_grids.png`: the grid figure of the paper, drawn from `clarke_histogram.csv` (R6).",
              "- `difficulty_by_week.png`: whether days get harder or easier over the follow-up; stage 1 uses skill "
              "over persistence so that ageing is not confused with difficulty.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
