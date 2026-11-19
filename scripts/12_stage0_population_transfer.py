"""12 - Stage 0 with pretrained models: scratch, population and fine-tuned starting points.

Repeats the stage-0 protocol of script 10 on the screen group of the core
dataset, for every horizon and both follow-up periods, with the three starting
points of each learned model (D62), using the settings chosen on the pool
group by script 11:

- ``lightgbm`` / ``nhits``: trained from scratch on the patient's 14 days;
- ``<family>_pop``: the population model of the pool group, as it is
  (zero-shot), for LightGBM and the networks;
- ``<family>_ft``: the population model fine-tuned on the patient's 14 days
  (last 2 days for early stopping);
- the glucose-only twins (D76, D90) of the models that go on to stages 1-3:
  ``<family>_ft_noexog`` for every family of ``glucose_only_twins.families``
  and ``arx_noexog``, the same models reading the glucose only (no IOB, COB,
  insulin activity, carbohydrate absorption or time of day), compared with the
  full ones (added to ``transfer.models`` by ``sim.stage0.with_twins``);

plus persistence (the reference of every skill), AR(3) and ARX. Learned models run
once per seed of stage 0 (each seed has its own population model), so the
noise floor (DMD, D61) covers every starting point. Paired comparisons give,
for each family, the difference in mean skill between starting points and
against AR(3), and call it a tie when it is below the DMD of the pair (the
larger DMD of its two models, D75).

Every model is frozen after its 14 days, as in script 10; stage 1 (scripts 20
and 21) will refresh them. Reports RMSE and Clarke zones for every model
(rule R6). One job per patient on all cores but two, with the networks on the
GPUs (rule R5, D73). The confirm group is never touched.

Derived data:
    <derived_data>/runs/12_stage0_population_transfer/day_metrics.parquet

Outputs (results/12_stage0_population_transfer/):
    model_summary.csv        per period, model and horizon: patient RMSE and skill (seeds averaged), intervals
    metrics_overall.csv      per period, model, seed and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    metrics_by_patient.csv   per patient, period, model, seed and horizon
    noise_floor.csv          per period, learned model and horizon: seed-pair differences and the DMD
    paired_comparisons.csv   per period, family, horizon and pair of models: skill difference, interval, verdict
    fit_log.csv              per patient, period, model and seed: sizes, seconds, rounds or steps, device
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6)
    task_log.csv
    rmse_by_horizon.png
    skill_by_patient.png
    paired_comparisons.png
    noise_floor.png
    clarke_grids.png
    summary.md
    run_manifest.json

Usage:
    python scripts/12_stage0_population_transfer.py
    python scripts/12_stage0_population_transfer.py --models persistence ar3 lightgbm_pop lightgbm_ft
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
import yaml  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.splits import group_patients  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import (FIT_NOTE, PERIOD_LABELS, fit_outcomes, model_summary,  # noqa: E402
                                           noise_floor, overall_table, paired_comparisons, pairs_for, patient_table,
                                           plot_grids, plot_noise, plot_paired, plot_rmse, plot_skill)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import set_style  # noqa: E402
from t1dretrain.sim.stage0 import (FAMILIES, is_twin, split_name, split_variant, twin_families,  # noqa: E402
                                   with_twins)

SCRIPT_ID = "12_stage0_population_transfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of transfer.models")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients (quick check)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_job(job: dict):
    from t1dretrain.sim.stage0 import evaluate_patient

    with WarningRecorder() as recorder:
        result = evaluate_patient(job)
    result.warnings = recorder.text()
    return result


def device_for(k: int, resources) -> str:
    return "cuda" if resources.device == "cuda" and resources.n_gpus else "cpu"  # one GPU per worker process (D73)


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
    population_dir = derived / "models" / core / "population"
    choices_file = population_dir / "choices.yaml"
    if not choices_file.exists():
        raise SystemExit(f"{choices_file} not found: run scripts/11_pretrain_population.py first")
    choices = yaml.safe_load(choices_file.read_text())
    stage, transfer = dict(config["stage0"]), config["transfer"]
    available = {f for f in FAMILIES if isinstance(choices.get(f), dict)}
    models, skipped = [], []
    for name in with_twins(transfer["models"], twin_families(config)):  # each with its glucose-only twin (D90)
        family, start = split_name(name)
        variant = split_variant(name)[1]
        if family in FAMILIES and (family not in available or start not in choices[family] or
                                   (variant and variant not in choices[family])):
            skipped.append(name)
            continue  # a family, starting point or twin (D76, D90) that script 11 did not produce
        models.append(name)
    if skipped:
        print(f"Left out (script 11 did not produce them): {skipped}")
    if args.models:  # a chosen model brings its glucose-only twin (D90)
        models = [m for m in models if m in args.models or split_variant(m)[0] in args.models]
    if "persistence" not in models:
        models = ["persistence"] + models
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]), "warmup_days": int(config["warmup_days"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]), **stage,
                "models": models, "seeds": [int(s) for s in choices.get("seeds", stage["seeds"])],
                "population_dir": str(population_dir),
                "finetune": {f: choices[f].get("finetune") for f in available}}
    for f in available:
        if choices[f].get("scratch"):
            settings[f] = choices[f]["scratch"]
    group = transfer.get("group", "screen")
    core_period = config.get("core_period", "post_randomization")

    patients = group_patients(core, group)
    if args.patients:
        patients = patients[: args.patients]
    anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == "core")].drop_duplicates(["patient_id", "period"])
    day0 = {(r.patient_id, r.period): r.day0 for r in anchors.itertuples()}
    periods = [core_period] + [p for p in ["post_randomization", "whole"] if p != core_period]
    jobs = [{"dataset": core, "patient_id": pid, "derived_root": derived, "settings": settings, "stream": k,
             "device": device_for(k, resources), "day0": {p: day0.get((pid, p)) for p in periods}}
            for k, pid in enumerate(patients)]
    results = parallel_map(run_job, jobs, resources.n_workers, label="patients")

    task_log = pd.DataFrame([{"patient_id": r.patient_id, "status": "error" if r.error else "ok", "seconds": r.seconds,
                              "warnings": getattr(r, "warnings", ""),
                              "error": r.error.strip().splitlines()[-1] if r.error else "", "traceback": r.error}
                             for r in results])
    ok = [r for r in results if not r.error]
    days = pd.concat([r.days for r in ok if not r.days.empty], ignore_index=True) if ok else pd.DataFrame()
    runs = derived / "runs" / SCRIPT_ID
    runs.mkdir(parents=True, exist_ok=True)
    days.to_parquet(runs / "day_metrics.parquet", index=False)
    clarke = ClarkeAccumulator(["model", "seed", "horizon_min", "period"], sample_size=settings["clarke_point_sample"])
    for r in ok:
        clarke.merge(r.clarke)
    fit_log = pd.DataFrame([row for r in ok for row in r.info])
    replicates, seed = int(stage["bootstrap_replicates"]), int(config["random_seed"])

    by_patient = patient_table(days) if not days.empty else pd.DataFrame()
    overall = overall_table(days, by_patient, replicates, seed) if not days.empty else pd.DataFrame()
    summary = model_summary(by_patient, replicates, seed) if not by_patient.empty else pd.DataFrame()
    noise = noise_floor(by_patient, replicates, seed) if not by_patient.empty else pd.DataFrame()
    paired = paired_comparisons(by_patient, noise, replicates, seed, pairs_for(set(by_patient["model"]))) \
        if not by_patient.empty else pd.DataFrame()
    histogram = clarke.histogram()

    out.csv(summary, "model_summary")
    out.csv(overall, "metrics_overall")
    out.csv(by_patient, "metrics_by_patient")
    out.csv(noise, "noise_floor")
    out.csv(paired, "paired_comparisons")
    for column in ("coef", "rounds", "added_rounds"):
        if column in fit_log:
            fit_log[column] = fit_log[column].astype(str)
    out.csv(fit_log, "fit_log")
    out.csv(fit_outcomes(fit_log), "fit_outcomes")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(histogram, "clarke_histogram")
    out.csv(clarke.sample(), "clarke_points_sample")
    out.csv(task_log, "task_log")
    if not summary.empty:
        present = [p for p in periods if p in set(summary["period"])]
        shown = core_period if core_period in present else present[0]
        shown_models = [m for m in models if m in set(summary["model"])]
        out.figure(plot_rmse(summary, present, shown_models, "Stage 0 with pretrained models: frozen after 14 days, "
                             "screen group (learned models: mean of the seeds)"), "rmse_by_horizon")
        out.figure(plot_skill(by_patient, shown, shown_models), "skill_by_patient")
        if not paired.empty:
            out.figure(plot_paired(paired, shown), "paired_comparisons")
        if not noise.empty:
            out.figure(plot_noise(noise, by_patient, shown), "noise_floor")
        out.figure(plot_grids(histogram, shown_models, settings["horizons_min"], shown, settings["seeds"][0]),
                   "clarke_grids")
    out.text(write_summary(summary, overall, noise, paired, fit_log, task_log, settings, choices, group, len(patients),
                           core_period, release, resources), "summary.md")
    out.manifest(settings={**{k: v for k, v in settings.items() if k != "population_dir"}, "group": group,
                           "patients": len(patients), "core_period": core_period,
                           "population_models": str(population_dir), "choices_created": choices.get("created"),
                           "choices_release": choices.get("release")},
                 notes="One job per patient; day metrics in <derived>/runs/12_stage0_population_transfer.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ paired comparisons


# ============================================================================ summary


def write_summary(summary, overall, noise, paired, fit_log, task_log, settings, choices, group, n_patients, core_period,
                  release, resources) -> str:
    lines = ["# 12 - Stage 0 with pretrained models: scratch, population and fine-tuned", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Group `{group}` of the core dataset ({n_patients} patients). Models frozen after the "
             f"{settings['warmup_days']} days from day 0 (fine-tuning and scratch: last {settings['validation_days']} "
             f"days with samples for early stopping; seeds {settings['seeds']}); origins every "
             f"{settings['origin_stride_min']:.0f} min on usable days; horizons {settings['horizons_min']} min. "
             f"Population models and settings from script 11 (`choices.yaml`, release {choices.get('release')}, "
             f"{choices.get('created')}; {len(choices.get('pool', {}).get('fit', [])) + len(choices.get('pool', {}).get('check', []))} "
             f"pool patients). Core period (D52): `{core_period}`.", ""]
    problems = []
    for _, row in task_log[task_log["status"] != "ok"].iterrows():
        problems.append(f"{row['patient_id']} failed: {row['error']}")
    for _, row in task_log[task_log["warnings"].fillna("") != ""].head(10).iterrows():
        problems.append(f"{row['patient_id']} raised warnings: {row['warnings'][:300]}")
    if not fit_log.empty and "error" in fit_log:
        failed = fit_log[fit_log["error"].notna()]
        for (model, error), g in failed.groupby(["model", "error"]):
            problems.append(f"{model} not fitted for {len(g)} patient-period(s) ({', '.join(g['patient_id'].unique()[:5])}): "
                            f"{error}")
    if not fit_log.empty and "cuda_error" in fit_log:
        for _, row in fit_log[fit_log["cuda_error"].notna()].head(5).iterrows():
            problems.append(f"{row['patient_id']}: {row['model']} fell back to the CPU ({row['cuda_error']})")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    outcomes = fit_outcomes(fit_log)
    if not outcomes.empty:
        lines += ["## Fits (a run is complete when `other_failure` is 0)", "", markdown_table(outcomes), "", FIT_NOTE, ""]
    if not summary.empty:
        lines += ["## Models (patient means; learned models averaged over seeds; 95 % bootstrap intervals)", "",
                  markdown_table(summary), ""]
    if not paired.empty:
        core = paired[paired["period"] == core_period]
        lines += [f"## Paired comparisons, {PERIOD_LABELS.get(core_period, core_period)} (tie = below the DMD)", "",
                  markdown_table(core.drop(columns=["period"])), "",
                  "The other period is in `paired_comparisons.csv`.", ""]
        twins = core[core["against"].map(is_twin)]
        if not twins.empty:
            show = ["model", "horizon_min", "patients", "skill_difference", "low", "high", "patients_better", "dmd",
                    "verdict"]
            lines += ["## What insulin and carbohydrates add (D90: full model - its glucose-only twin)",
                      "", markdown_table(twins[show]), ""]
    if not noise.empty:
        lines += ["## Noise floor (D61)", "", markdown_table(noise), ""]
    if not fit_log.empty:
        timing = fit_log.groupby("model")[["fit_seconds", "predict_seconds"]].median().round(2).reset_index()
        lines += ["## Median seconds per patient and period", "", markdown_table(timing), ""]
        if "best_step" in fit_log:
            steps = fit_log[fit_log["model"].str.startswith("nhits")].groupby("model")[["steps", "best_step"]] \
                .median().reset_index()
            if not steps.empty:
                lines += ["N-HiTS training steps (median; best step 0 = the starting network was kept):", "",
                          markdown_table(steps), ""]
    if not overall.empty:
        show = ["period", "model", "seed", "horizon_min", "n", "rmse", "mae", "mard_pct", "bias", "clarke_A_pct",
                "clarke_B_pct", "clarke_AB_pct", "clarke_C_pct", "clarke_D_pct", "clarke_E_pct"]
        lines += ["## Pooled forecasts per seed (R6)", "", markdown_table(overall[show]), ""]
    lines += ["## Checks to look at", "",
              "- `rmse_by_horizon.png` and `paired_comparisons.png`: which starting point stage 1 should use for each "
              "family (fine-tuned above population above scratch is the expected order).",
              "- `noise_floor.png`: the DMD of each model; comparisons use the larger DMD of the two models (D75).",
              "- `clarke_grids.png`: the grid figure for every model (R6).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
