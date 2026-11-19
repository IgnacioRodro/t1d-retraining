"""73 - Wearables, exploratory: do heart rate, steps and sleep help the personal models? (HANDOFF 7.10, D98).

The few patients with wearables (BrisT1D, T1D-UOM, HUPA-UCM) on a 5-min grid,
from the confirmation's anchor (day 14), with the confirmation's arms (never
and the refit at the optimal interval; ARX also online; no every-day refit).
LightGBM and ARX, every input and glucose only (D90), each run twice on the
same patients, arms and days: as they are (``standard``) and with the
wearables at the origin as extra inputs (``wearables``; ``wearables.inputs``:
heart rate over the last 15 min, steps over the last 15 and 60 min, energy over
the last 15 min, asleep or not; NaN where the patient has none, D56):

- LightGBM: the population model of script 11 has no wearables, so every
  fit grows new boosters with every input on top of its forecast, which is the
  same as continuing its boosting (``models.gbdt``, D98);
- ARX: the wearables centred and scaled with fixed constants (a missing value
  counts as the centre).

Few patients: descriptive, paired per patient, with bootstrap intervals and
each patient's difference; nothing is tested or chosen. Patients enter when at
least ``min_coverage`` of the slots of their test days have a wearable value.
T1D-UOM's heart rate is not used (D39): its column is empty.

Inputs:
    <derived_data>/clean/<dataset>/anchors.parquet; <derived_data>/features/<dataset>/grid/  (scripts 04, 05)
    <derived_data>/runs/31_.../confirmation.yaml, <derived_data>/runs/41_.../mechanisms.yaml
    <derived_data>/models/<core>/population/   (script 11)

Derived data:
    <derived_data>/runs/73_wearables/<variant>/cache/...; <derived_data>/runs/73_wearables/day_metrics.parquet

Outputs (results/73_wearables/):
    cohorts.csv             per cohort: patients with an anchor, on a 5-min grid, with enough wearables, models
    coverage.csv            per patient: share of the test slots with heart rate, steps, energy and sleep
    skill_by_variant.csv    per cohort, variant, model, arm and horizon: skill (95 %) and RMSE
    vs_standard.csv         wearables minus standard, paired per patient: skill, RMSE, forecast lows
    per_patient.csv         every patient's skill with and without the wearables
    refit_gain.csv          gain of every refit over never with and without the wearables
    metrics_by_variant.csv  pooled forecasts (R6: RMSE, MAE, MARD, bias, Clarke zones) and forecast lows
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6)
    fit_log.csv / task_log.csv / jobs.csv
    per_patient.png
    summary.md, run_manifest.json

Usage:
    python scripts/73_wearables.py
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

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.splits import cohort_rule  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval import low_glucose as lg  # noqa: E402
from t1dretrain.eval import variant_analysis as va  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import color, label  # noqa: E402
from t1dretrain.models.base import WEARABLE_INPUTS  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim import cohort_runs as cr  # noqa: E402
from t1dretrain.sim.mechanisms import parse_arm  # noqa: E402
from t1dretrain.sim.stage0 import split_variant  # noqa: E402

SCRIPT_ID = "73_wearables"
WEARABLES = "wearables"
ARM_KINDS = ("never", "restart", "online")  # the arms that can refit a model with extra inputs (models.gbdt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def coverage(dataset: str, anchors: pd.DataFrame, derived: Path, test_days: int, inputs: list[str]) -> pd.DataFrame:
    """Per patient: the share of the slots of the test days with each wearable input, and with any of them."""
    store = ProcessedStore(dataset, root=derived, stage="features")
    rows = []
    for r in anchors.itertuples():
        grid = store.read_frame("grid", r.patient_id)
        if grid.empty:
            continue
        start = pd.Timestamp(r.anchor)
        test = grid[(grid["ts"] >= start) & (grid["ts"] < start + pd.Timedelta(days=test_days))]
        present = [c for c in inputs if c in test]
        row = {"patient_id": r.patient_id, "slots": int(len(test))}
        for c in inputs:
            row[f"{c}_share"] = float(test[c].notna().mean()) if c in present and len(test) else 0.0
        row["any_share"] = float(test[present].notna().any(axis=1).mean()) if present and len(test) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ main


def main() -> None:
    import time

    run_start = time.time()
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
    conf, cfg = config["confirmation"], config["wearables"]
    inputs = [str(c) for c in cfg["inputs"]]
    unknown = [c for c in inputs if c not in WEARABLE_INPUTS]
    if unknown:
        raise SystemExit(f"wearables.inputs: no scale for {unknown} (models.base.WEARABLE_INPUTS)")
    seed, replicates = int(config["random_seed"]), int(config["stage0"]["bootstrap_replicates"])
    min_events = int(config.get("low_glucose", {}).get("min_events", 20))
    population_dir = derived / "models" / core / "population"
    choices = cr.read_choices(population_dir)
    base = cr.base_settings(config, choices, population_dir)
    forgetting = float((base.get("arx") or {}).get("forgetting", 1.0))
    policies_doc, mechanisms_doc = cr.read_plan(derived)
    models, plan, notes = cr.plan_models(policies_doc, mechanisms_doc, list(cfg["models"]), population_dir,
                                         base["seeds"])

    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    jobs, rows, coverages, all_anchors, arms_of, by_variant = [], [], [], [], {}, {}
    for order, c in enumerate(cfg["cohorts"]):
        key, rule = cf.cohort_key(c["dataset"], c["cohort"]), cohort_rule(c["dataset"], c["cohort"])
        test_days = int(c.get("test_days", conf["test_days"]))
        anchors, counts = cr.cohort_patients(c, derived, [int(a) for a in conf["anchor_days"]], seed, None)
        cover = coverage(c["dataset"], anchors, derived, test_days, inputs)
        kept = set(cover.loc[cover["any_share"] >= float(cfg["min_coverage"]), "patient_id"]) if not cover.empty \
            else set()
        anchors = anchors[anchors["patient_id"].isin(kept)]
        counts["with_wearables"] = int(anchors["patient_id"].nunique())
        coverages.append(cover.assign(cohort=key, enters=cover["patient_id"].isin(kept)))
        runnable = cf.models_for_rule(models, rule)
        arms_of[key] = {}
        for m in runnable:
            arms = cf.confirmation_arms(plan[m]["policies"], plan[m]["mechanism"], plan[m]["interval"], False,
                                        forgetting)
            dropped = [a for a in arms if parse_arm(a)[0] not in ARM_KINDS]
            if dropped:
                notes.append(f"{key} {m}: {dropped} left out (a model with extra inputs refits by restarting)")
            arms_of[key][m] = [a for a in arms if a not in dropped]
        rows.append({"cohort": key, "usable_rule": rule, **counts, "test_days": test_days,
                     "models": " ".join(runnable)})
        if anchors.empty:
            continue
        all_anchors.append(anchors.assign(cohort=key))
        for variant, extra in ((va.STANDARD, {}), (WEARABLES, {"extra_inputs": inputs})):
            settings = {**base, "test_days": test_days, **extra}
            part = cr.cohort_jobs(key, c["dataset"], rule, order, anchors, arms_of[key], settings, choices,
                                  runs / variant, recompute, derived, cr.device_for(resources), stream0=len(jobs))
            by_variant.setdefault(variant, []).extend(part)
            jobs += part
    cohort_table = pd.DataFrame(rows)
    print(cohort_table.to_string(index=False))
    jobs.sort(key=cr.job_order)
    print(f"Models {models}; {len(jobs)} jobs")
    statuses = parallel_map(cr.run_job, jobs, resources.n_workers, label="wearable jobs")

    frames, infos, missing = [], [], []
    clarke = ClarkeAccumulator(["variant", "cohort", "model", "policy", "horizon_min", "period"], sample_size=20000)
    for variant, part in by_variant.items():
        days, info, accumulator, lost = cr.collect(part)
        frames.append(days.assign(variant=variant))
        infos += [{**row, "variant": variant} for row in info]
        clarke.merge(accumulator.tagged(variant=variant))
        missing += lost
    days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    runs.mkdir(parents=True, exist_ok=True)
    if not days.empty:
        days.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log, task_log = pd.DataFrame(infos), cr.task_table(statuses)

    summary = versus = gain = metrics = per_patient = pd.DataFrame()
    if not days.empty:
        skill = va.skill_table(days)
        lows = lg.patient_lows(days, va.KEYS, min_events)
        summary = va.variant_summary(skill, replicates, seed)
        versus = va.against_standard(skill, lows, replicates, seed)
        gain = va.refit_gain(skill, replicates, seed)
        metrics = va.pooled_metrics(days)
        per_patient = patient_table(skill)
    cover_table = pd.concat(coverages, ignore_index=True) if coverages else pd.DataFrame()

    out.csv(cohort_table, "cohorts")
    out.csv(first(cover_table).round(4), "coverage")
    out.csv(summary.round(5), "skill_by_variant")
    out.csv(versus.round(5), "vs_standard")
    out.csv(per_patient.round(5), "per_patient")
    out.csv(gain.round(5), "refit_gain")
    out.csv(metrics.round(4), "metrics_by_variant")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(clarke.histogram(), "clarke_histogram")
    out.csv(clarke.sample(), "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(task_log, "task_log")
    out.csv(pd.concat(all_anchors, ignore_index=True).drop(columns=["dataset"], errors="ignore")
            if all_anchors else pd.DataFrame(), "jobs")
    if not per_patient.empty:
        out.figure(plot_patients(per_patient, plan), "per_patient")
    out.text(write_summary(cohort_table, cover_table, versus, per_patient, gain, task_log, fit_log, missing, notes,
                           models, inputs, plan, release, resources), "summary.md")
    out.manifest(settings={"wearables": cfg, "models": models, "arms": arms_of, "plan": plan,
                           **{k: v for k, v in base.items() if k != "population_dir"}},
                 notes="Every model runs twice on the same patients, arms and days: standard and with the wearables.")
    print(f"Done. Results in {out.dir}")


def first(table: pd.DataFrame) -> pd.DataFrame:
    return table[["cohort"] + [c for c in table.columns if c != "cohort"]] if "cohort" in table else table


def patient_table(skill: pd.DataFrame) -> pd.DataFrame:
    """Every patient's skill with and without the wearables, and the difference."""
    keys = ["cohort", "model", "policy", "horizon_min", "patient_id"]
    wide = skill.pivot_table(index=keys, columns="variant", values="skill").reset_index()
    if WEARABLES not in wide or va.STANDARD not in wide:
        return pd.DataFrame()
    wide["difference"] = wide[WEARABLES] - wide[va.STANDARD]
    return wide.rename(columns={va.STANDARD: "skill_standard", WEARABLES: "skill_wearables"})


# ============================================================================ figure


def plot_patients(table: pd.DataFrame, plan: dict, horizons=(30, 60, 120)):
    """Per model: every patient's skill with the wearables minus without, for never (light) and the refit at the
    optimal interval (dark), per horizon."""
    models = list(dict.fromkeys(table["model"]))
    fig, axes = model_axes(len(models), 4.4, 3.0, ncols=2)
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, models):
        k = (plan.get(model) or plan.get(split_variant(model)[0]) or {}).get("interval")
        refit = f"restart_every{int(k)}" if k else None
        m = table[table["model"] == model]
        for i, horizon in enumerate(horizons):
            for j, (arm, alpha) in enumerate((("never", 0.4), (refit, 0.95))):
                v = m[(m["horizon_min"] == horizon) & (m["policy"] == arm)]["difference"].dropna()
                if v.empty:
                    continue
                x = i + (j - 0.5) * 0.3
                ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), 100 * v, s=12, color=color(model), alpha=alpha,
                           lw=0)
                ax.plot([x - 0.1, x + 0.1], [100 * v.mean()] * 2, color=color(model), lw=2, alpha=alpha)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(range(len(horizons)), [f"{h} min" for h in horizons])
        ax.set_ylabel("skill with - without wearables (points)")
        ax.set_title(label(model), loc="left", fontsize=9)
    fig.suptitle("Wearables as extra inputs, patient by patient (light: never; dark: refit at the optimal interval; "
                 "bar: mean)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(cohorts, cover, versus, per_patient, gain, task_log, fit_log, missing, notes, models, inputs, plan,
                  release, resources) -> str:
    lines = ["# 73 - Wearables (exploratory)", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Models {models}, with and without the wearables {inputs} at the origin, on the same patients, arms "
             "and days (D98). Few patients: descriptive, nothing is tested or chosen.", "", "## Cohorts", "",
             markdown_table(cohorts), ""]
    if not cover.empty:
        lines += ["## Wearables in the test days (share of the slots)", "",
                  markdown_table(first(cover).round(2)), ""]
    problems = list(notes)
    if not task_log.empty:
        failed = task_log[task_log["status"] == "error"].head(20)
        problems += [f"{r.job} failed: {r.error}" for r in failed.itertuples()]
    if not fit_log.empty and "error" in fit_log:
        for (variant, model, error), g in fit_log[fit_log["error"].notna()].groupby(["variant", "model", "error"]):
            problems.append(f"{variant} {model}: {len(g)} fits failed: {error}")
    problems += [f"no result for {m}" for m in missing[:10]]
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not versus.empty:
        v = versus[versus["measure"].isin(["skill", "rmse"])].copy()
        scale = np.where(v["measure"] == "skill", 100, 1)
        v["value"] = [f"{s * d:+.2f} ({s * lo:+.2f} to {s * hi:+.2f}; {n} of {p} higher)" for s, d, lo, hi, n, p in
                      zip(scale, v["difference"], v["low"], v["high"], v["patients_higher"], v["patients"])]
        table = v.pivot_table(index=["cohort", "model", "policy", "horizon_min"], columns="measure", values="value",
                              aggfunc="first").reset_index()
        lines += ["## With minus without the wearables, paired per patient (skill in points, RMSE in mg/dL)", "",
                  markdown_table(table), "", "Forecast lows in `vs_standard.csv`; every patient in `per_patient.csv`.",
                  ""]
    if not gain.empty:
        g = gain[gain["variant"] == WEARABLES].copy()
        if "gain_difference" in g and not g.empty:
            for column in ("gain", "gain_difference", "difference_low", "difference_high"):
                g[column] = (100 * g[column]).round(2)
            lines += ["## Gain of refitting over never with the wearables, and its difference to the standard models "
                      "(points)", "", markdown_table(g[["cohort", "model", "policy", "horizon_min", "patients", "gain",
                                                        "gain_difference", "difference_low", "difference_high"]]), ""]
    lines += ["## Checks to look at", "",
              "- `per_patient.png`: whether any patient gains clearly from the wearables, and with which model.",
              "- `coverage.csv`: how much of the test days the wearables cover.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
