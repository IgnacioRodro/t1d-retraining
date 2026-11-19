"""50 - Confirmation: what the screening chose, on new patients of the core dataset and on the other datasets.

Runs, from day 14 of each cohort's period, every model of stage 2 with what
the screening chose for it; nothing is chosen here (HANDOFF 7.6, D92;
``eval.confirmation``):

- the policies of script 31 (``confirmation.yaml``, D84): never, the growing
  window refitted at the model's optimal interval and every day
  (``restart_every<k>`` and ``restart_every1``: each refit starts again from
  the population model, D65, the fits of stage 2); ARX also its update at
  every sample (``online_l1``, D87);
- the mechanism of script 41 (``mechanisms.yaml``, D91) at the optimal
  interval, when it is not the restart (e.g. ``warm_every3``);
- every glucose-only twin with the arms of its full model (D90).

Cohorts (``confirmation.cohorts``): the confirm group of the core dataset (the
hold-out, untouched until now) and the cohorts of DCLP3, Loop, BrisT1D and
T1D-UOM of script 04. Origins every ``origin_stride_min_confirm`` minutes
(D17); the day metrics also count the forecast lows below 70 and 54 mg/dL.
The population models are those of the core dataset's pool group (script
11). Patients whose grid is not of 5 min are left out (the learned models read
5-min inputs), and cohorts without insulin run only the models that read the
glucose only (AR(3) and the twins). The every-day refit runs where the cohort
asks for it (``daily``). Results are cached per cohort, model, arm and patient
(``sim.cohort_runs``, shared with scripts 71 to 73), so a run that stops can go
on later; ``--recompute`` ignores the cache. One
job per patient and model on all cores but two, networks on the GPUs (rule
R5). Script 51 adds the analysis.

Inputs:
    <derived_data>/runs/31_stage2_policy_analysis/confirmation.yaml    (D84)
    <derived_data>/runs/41_stage3_mechanism_analysis/mechanisms.yaml   (D91)
    <derived_data>/models/<core>/population/                           (script 11)
    <derived_data>/clean/<dataset>/ anchors and <derived_data>/features/<dataset>/   (scripts 04 and 05)
    results/12_stage0_population_transfer/noise_floor.csv               (DMD, D75)

Derived data:
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet   per cohort, patient, model, arm, horizon and day
    <derived_data>/runs/50_confirmation_runs/fit_log.parquet       every fit
    <derived_data>/runs/50_confirmation_runs/cache/<dataset>__<cohort>/<model>/<arm>/<patient-anchor>.pkl

Outputs (results/50_confirmation_runs/):
    cohorts.csv              per cohort: usable rule, patients with an anchor, left out (grid), drawn, models
    arm_summary.csv          per cohort, model, arm and horizon: patients, skill and RMSE with intervals
    arm_overview.csv         per cohort, model and arm: skill per horizon, gain over never, seconds per test day
    vs_never.csv             every arm against never (paired; verdict with the model's DMD, D75)
    metrics_overall.csv      per cohort, model, arm and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6), lows
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6)
    fit_log.csv / fit_outcomes.csv / task_log.csv / jobs.csv
    arms.png                 gain of every arm over never, per cohort and model
    summary.md
    run_manifest.json

Usage:
    python scripts/50_confirmation_runs.py
    python scripts/50_confirmation_runs.py --cohorts replace_bg/core --models lightgbm_ft --patients 4   # a quick run
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
from t1dretrain.data.splits import cohort_rule  # noqa: E402
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval import mechanism_analysis as ma  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.stage0_report import (FIT_NOTE, SUMS, DMDTable, color, finish, fit_outcomes,  # noqa: E402
                                           label, read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim import cohort_runs as cr  # noqa: E402
from t1dretrain.sim.mechanisms import parse_arm  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, is_twin, split_name  # noqa: E402

SCRIPT_ID = "50_confirmation_runs"
FIT_KEYS = ["cohort"] + pr.FIT_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--cohorts", nargs="*", default=None,
                        help="subset of the cohorts, as dataset/cohort (e.g. replace_bg/core loop/closed_insulin)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of the models (full names; twins follow)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients of each cohort (quick)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ main


def main() -> None:
    import time

    run_start = time.time()  # --recompute: marks the results of this run (sim.cache)
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
    stage, conf = config["stage0"], config["confirmation"]
    seed = int(config["random_seed"])
    population_dir = derived / "models" / core / "population"
    choices = cr.read_choices(population_dir)
    base = cr.base_settings(config, choices, population_dir)
    forgetting = float((base.get("arx") or {}).get("forgetting", 1.0))
    policies_doc, mechanisms_doc = cr.read_plan(derived)
    models, plan, notes = cr.plan_models(policies_doc, mechanisms_doc, args.models, population_dir, base["seeds"])
    anchor_days = [int(a) for a in conf["anchor_days"]]

    cohorts = list(conf["cohorts"])
    if args.cohorts:
        cohorts = [c for c in cohorts if cf.cohort_key(c["dataset"], c["cohort"]) in args.cohorts]
    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    jobs, cohort_rows, all_anchors, arms_of = [], [], [], {}
    for order, c in enumerate(cohorts):
        key, rule = cf.cohort_key(c["dataset"], c["cohort"]), cohort_rule(c["dataset"], c["cohort"])
        settings = {**base, "test_days": int(c.get("test_days", conf["test_days"]))}
        anchors, counts = cr.cohort_patients(c, derived, anchor_days, seed, args.patients)
        runnable = cf.models_for_rule(models, rule)
        arms_of[key] = {m: cf.confirmation_arms(plan[m]["policies"], plan[m]["mechanism"], plan[m]["interval"],
                                                bool(c.get("daily", False)), forgetting) for m in runnable}
        for arms in arms_of[key].values():
            for arm in arms:
                parse_arm(arm)  # an unknown arm stops the run here
        cohort_rows.append({"cohort": key, "dataset": c["dataset"], "period": c["period"], "usable_rule": rule,
                            "group": c.get("group", ""), **counts, "test_days": settings["test_days"],
                            "daily": bool(c.get("daily", False)), "models": " ".join(runnable)})
        all_anchors.append(anchors.assign(cohort=key))
        jobs += cr.cohort_jobs(key, c["dataset"], rule, order, anchors, arms_of[key], settings, choices, runs,
                               recompute, derived, cr.device_for(resources), stream0=len(jobs))
    cohort_table = pd.DataFrame(cohort_rows)
    print(cohort_table.to_string(index=False))
    jobs.sort(key=cr.job_order)
    print(f"Models {models}; {len(jobs)} jobs")
    statuses = parallel_map(cr.run_job, jobs, resources.n_workers, label="confirmation jobs")

    days, infos, clarke, missing = cr.collect(jobs)
    runs.mkdir(parents=True, exist_ok=True)
    days.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log = pd.DataFrame(infos)
    if not fit_log.empty:
        cr.write_fit_log(fit_log, runs / "fit_log.parquet", FIT_KEYS)
    task_log = cr.task_table(statuses)
    replicates = int(stage["bootstrap_replicates"])
    dmd = DMDTable(read_noise_floor(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in models if split_name(m)[0] in FAMILIES])
    summary, overview, vs_never, overall = cohort_tables(days, fit_log, cohort_table, dmd, replicates, seed)
    sample = clarke.sample()
    if not sample.empty:  # the histogram has every panel; the points only those of never and the reference
        sample = sample[~sample["policy"].str.contains("_every1$", regex=True)]

    out.csv(cohort_table, "cohorts")
    out.csv(summary.round(4), "arm_summary")
    out.csv(overview.round(5), "arm_overview")
    out.csv(vs_never.round(4), "vs_never")
    out.csv(overall, "metrics_overall")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(clarke.histogram(), "clarke_histogram")
    out.csv(sample, "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(cohort_outcomes(fit_log), "fit_outcomes")
    out.csv(task_log, "task_log")
    jobs_table = pd.concat(all_anchors, ignore_index=True) if all_anchors else pd.DataFrame()
    out.csv(jobs_table.drop(columns=["dataset"], errors="ignore"), "jobs")
    if not overview.empty:
        out.figure(plot_arms(overview, cohort_table), "arms")
    out.text(write_summary(cohort_table, overview, fit_log, task_log, missing, models, plan, arms_of, notes,
                           policies_doc, mechanisms_doc, release, resources, base), "summary.md")
    out.manifest(settings={**{k: v for k, v in base.items() if k != "population_dir"}, "models": models,
                           "plan": plan, "arms": arms_of, "cohorts": cohorts, "anchor_days": anchor_days,
                           "population_models": str(population_dir), "choices_release": choices.get("release"),
                           "confirmation_release": policies_doc.get("release"),
                           "mechanisms_release": mechanisms_doc.get("release")},
                 notes="One job per cohort, patient, anchor and model; day metrics in "
                       "<derived>/runs/50_confirmation_runs.")
    print(f"Done. Results in {out.dir}")


def cohort_outcomes(fit_log: pd.DataFrame) -> pd.DataFrame:
    """Fit outcomes (D72) per cohort and model, one row per arm and fit."""
    if fit_log.empty:
        return pd.DataFrame()
    log = fit_log.drop_duplicates([c for c in FIT_KEYS + ["policy"] if c in fit_log])
    parts = [fit_outcomes(g).assign(cohort=cohort) for cohort, g in log.groupby("cohort")]
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return table[["cohort"] + [c for c in table.columns if c != "cohort"]] if not table.empty else table


def cohort_tables(days: pd.DataFrame, fit_log: pd.DataFrame, cohorts: pd.DataFrame, dmd, replicates: int,
                  seed: int) -> tuple:
    """Per cohort: summary (skill and RMSE with intervals), overview (per arm, with the gain over never and the
    cost), every arm against never, and the pooled metrics (R6, forecast lows)."""
    summaries, overviews, vs_never, overall = [], [], [], []
    if days.empty:
        return (pd.DataFrame(),) * 4
    used = fit_log[fit_log["excluded"].isna()] if "excluded" in fit_log else fit_log
    sums = SUMS + cf.low_sums(days)
    for row in cohorts.itertuples():
        d = days[days["cohort"] == row.cohort]
        if d.empty:
            continue
        skill = pr.patient_skill(d)
        if skill.empty:
            continue
        log = used[used["cohort"] == row.cohort] if not used.empty else used
        references = {m: next((a for a in skill.loc[skill["model"] == m, "policy"].unique()
                               if a.startswith("restart_every") and a != "restart_every1"), "never")
                      for m in skill["model"].unique()}
        summaries.append(pr.summary_table(skill, replicates, seed).assign(cohort=row.cohort))
        overviews.append(ma.arm_table(skill, log, int(row.test_days), references).assign(cohort=row.cohort))
        vs_never.append(pr.paired(skill, lambda a: "never", dmd, replicates, seed).assign(cohort=row.cohort))
        pooled = d.groupby(["period", "model", "policy", "horizon_min"], as_index=False)[sums].sum()
        overall.append(cf.with_lows(finish(pooled)).round(4).assign(cohort=row.cohort))

    def first(frames):
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        table = pd.concat(frames, ignore_index=True)
        return table[["cohort"] + [c for c in table.columns if c != "cohort"]]

    return first(summaries), first(overviews), first(vs_never), first(overall)


# ============================================================================ figure and summary


def plot_arms(overview: pd.DataFrame, cohorts: pd.DataFrame):
    """Per cohort, the gain of every arm over never (mean over horizons) of the full models."""
    shown = [c for c in cohorts["cohort"] if c in set(overview["cohort"])]
    fig, axes = model_axes(len(shown), 5.2, 3.0, ncols=2)
    for ax, cohort in zip(axes, shown):
        o = overview[(overview["cohort"] == cohort) & (overview["arm"] != "never") &
                     ~overview["model"].map(is_twin)]
        models = list(dict.fromkeys(o["model"]))
        for i, model in enumerate(models):
            m = o[o["model"] == model].sort_values("arm")
            n = len(m)
            for j, r in enumerate(m.itertuples()):
                x = i + (j - (n - 1) / 2) * 0.8 / max(n, 1)
                ax.bar(x, 100 * r.gain_over_never, width=0.8 / max(n, 1), color=color(model),
                       alpha=0.45 + 0.5 * (j + 1) / n, edgecolor="white", lw=0.4)
                ax.text(x, 0, r.arm.replace("_every", "/"), rotation=90, fontsize=5, ha="center", va="bottom",
                        color=TEXT_SECONDARY)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(range(len(models)), [label(m) for m in models], fontsize=7, rotation=20, ha="right")
        n = int(cohorts.loc[cohorts["cohort"] == cohort, "patients"].iloc[0])
        ax.set_title(f"{cohort} ({n} patients)", loc="left", fontsize=9)
        ax.set_ylabel("gain over never (skill points)")
    fig.suptitle("Confirmation (D92): gain of every arm over never, mean over horizons (arm/k: refit every k days)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def write_summary(cohorts, overview, fit_log, task_log, missing, models, plan, arms_of, notes, policies_doc,
                  mechanisms_doc, release, resources, settings) -> str:
    lines = ["# 50 - Confirmation runs", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Models {models}: policies of script 31 (release {policies_doc.get('release')}) and mechanisms of "
             f"script 41 (release {mechanisms_doc.get('release')}); nothing is chosen here (D92). Origins every "
             f"{settings['origin_stride_min']:g} min; horizons {settings['horizons_min']} min; forecast lows below "
             f"{settings['event_thresholds']} mg/dL. Glucose-only twins (`_noexog`, D90) run the arms of their full "
             f"model.", ""]
    plan_rows = [{"model": m, "policies": " ".join(p["policies"]), "optimal_interval_days": p["interval"],
                  "mechanism": p["mechanism"]} for m, p in plan.items() if not is_twin(m)]
    if plan_rows:
        lines += ["## What each model brings from the screening", "", markdown_table(pd.DataFrame(plan_rows)), ""]
    if not cohorts.empty:
        lines += ["## Cohorts", "", "`left_out_grid`: patients whose grid is not of 5 min (the learned models read "
                  "5-min inputs); cohorts without insulin run only AR(3) and the twins.", "",
                  markdown_table(cohorts.drop(columns=["models"]).assign(daily=cohorts["daily"].map({True: "yes",
                                                                                                     False: "no"}))),
                  ""]
        arms_rows = [{"cohort": c, "model": m, "arms": " ".join(a)} for c, by in arms_of.items()
                     for m, a in by.items() if not is_twin(m)]
        lines += ["## Arms per cohort", "", markdown_table(pd.DataFrame(arms_rows)), ""]
    problems = list(notes)
    if not task_log.empty:
        problems += [f"{r.job} failed: {r.error}" for r in task_log[task_log["status"] == "error"].head(20).itertuples()]
        problems += [f"{r.job} raised warnings: {str(r.warnings)[:200]}"
                     for r in task_log[task_log["warnings"].fillna("") != ""].head(10).itertuples()]
        retry = task_log[task_log["retry_errors"].fillna("") != ""]
        problems += [f"{r.job}: fits failed, not cached (computed again next run): {str(r.retry_errors)[:300]}"
                     for r in retry.head(10).itertuples()]
    problems += [f"no result for {m}" for m in missing[:10]]
    if len(missing) > 10:
        problems.append(f"... and {len(missing) - 10} more results missing")
    if not fit_log.empty and "error" in fit_log:
        for (cohort, model, error), g in fit_log[fit_log["error"].notna()].groupby(["cohort", "model", "error"]):
            problems.append(f"{cohort} {model}: {len(g)} fits failed: {error}")
    if not fit_log.empty and "excluded" in fit_log:
        out = fit_log[fit_log["excluded"].notna()].drop_duplicates(["cohort", "model", "patient_id", "anchor_day"])
        for (cohort, model), g in out.groupby(["cohort", "model"]):
            problems.append(f"{cohort} {model}: left out for {len(g)} patients (no model at the anchor)")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    outcomes = cohort_outcomes(fit_log)
    if not outcomes.empty:
        lines += ["## Fits (a run is complete when `other_failure` is 0)", "", "One row per arm and fit.", "",
                  markdown_table(outcomes), "", FIT_NOTE, ""]
    if not overview.empty:
        o = overview.copy()
        o["gain_over_never"] = (100 * o["gain_over_never"]).round(2)
        show = [c for c in ("cohort", "model", "arm", "skill_30", "skill_60", "skill_120", "gain_over_never",
                            "seconds_per_day", "patients") if c in o.columns]
        lines += ["## Arms (mean skill over patients; gain over never in points, mean over horizons)", "",
                  markdown_table(o[show].round(4)), ""]
    lines += ["## Checks to look at", "",
              "- `arms.png`: whether refitting still beats never on new patients and datasets.",
              "- Script 51: the optimal interval against every day, the mechanism against the restart, insulin and "
              "carbohydrates, the gain over the weeks and the forecast lows, against the screening.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
