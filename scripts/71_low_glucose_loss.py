"""71 - A loss that weighs the lows: does it make the models detect them, and at what cost? (HANDOFF 7.10, D96).

A sensitivity analysis: nothing is chosen with it. LightGBM and ARX are
trained with a loss in which every training target below ``threshold`` mg/dL
(70) weighs ``factor`` times (5 and 20), in every fit:

- LightGBM: the population model is trained again on the pool with the same
  settings and rounds as in script 11 and the weighted loss; every personal fit
  (at the anchor and every refit) uses the weighted loss in training and on
  the validation days (early stopping and the undo check);
- ARX: weighted least squares in the batch fits and in the online update.

They run on the confirmation's patients and arms (script 50, D92) of
``low_glucose.loss.cohorts``: never, the refit at the optimal interval and,
for ARX, the online update (no every-day refit), and are compared, paired per
patient, with the same models of script 50 (``variant`` = standard): skill
and RMSE, sensitivity and precision of the forecast lows, Clarke zones, the
alarm curves (does the weighted loss lift the curve, or only move along it as
a higher alarm threshold would? script 70) and the gain of refitting over never
with each loss. Results are cached per factor, cohort, model, arm and patient
(``sim.cohort_runs``).

Inputs:
    <derived_data>/models/<core>/population/ (choices.yaml, pool split) and the pool's feature grids  (script 11)
    <derived_data>/runs/31_.../confirmation.yaml, <derived_data>/runs/41_.../mechanisms.yaml       (scripts 31, 41)
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet, results/50_confirmation_runs/clarke_histogram.csv

Derived data:
    <derived_data>/runs/71_low_glucose_loss/population/x<factor>/   LightGBM population models with the loss
    <derived_data>/runs/71_low_glucose_loss/x<factor>/cache/         one result per cohort, model, arm and patient
    <derived_data>/runs/71_low_glucose_loss/day_metrics.parquet      the runs of every factor

Outputs (results/71_low_glucose_loss/):
    population_models.csv   the population models trained with each factor (rounds, seconds)
    skill_by_variant.csv    per cohort, variant, model, arm and horizon: skill (95 %) and RMSE
    vs_standard.csv         every factor minus the standard loss, paired per patient: skill, RMSE, sensitivity
                            and precision below 70 and 54 mg/dL
    refit_gain.csv          gain of every refit over never with each loss, and its difference to the standard loss
    metrics_by_variant.csv  pooled forecasts (R6: RMSE, MAE, MARD, bias, Clarke zones) and forecast lows
    range_metrics.csv, alarm_curves.csv, matched_alarms.csv   as in script 70, per variant
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   the runs of this script (R6)
    fit_log.csv / task_log.csv / jobs.csv
    alarm_curves.png, tradeoff.png
    summary.md, run_manifest.json

Usage:
    python scripts/71_low_glucose_loss.py
    python scripts/71_low_glucose_loss.py --patients 4      # a quick run
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
from t1dretrain.data.splits import cohort_rule  # noqa: E402
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval import low_glucose as lg  # noqa: E402
from t1dretrain.eval import variant_analysis as va  # noqa: E402
from t1dretrain.eval.stage0_report import color, label  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim import cohort_runs as cr  # noqa: E402
from t1dretrain.sim.mechanisms import parse_arm  # noqa: E402
from t1dretrain.sim.stage0 import is_twin, population_path  # noqa: E402

SCRIPT_ID = "71_low_glucose_loss"
SOURCE = "50_confirmation_runs"
CURVE_KEYS = ["cohort", "variant", "model", "policy", "horizon_min"]
LINESTYLES = ["--", ":", "-."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients of each cohort (quick)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def variant_name(factor: float) -> str:
    return f"x{float(factor):g}"


# ============================================================================ population models (workers)


def run_pool(job: dict):
    from t1dretrain.sim.population import pool_patient

    with WarningRecorder():
        return pool_patient(job)


def run_train(job: dict) -> dict:
    from t1dretrain.sim.population import train_population

    return train_population(job)


def population_models(derived: Path, core: str, choices: dict, loss: dict, factors: list[float], seeds: list[int],
                      config: dict, release: str, resources) -> tuple[dict, pd.DataFrame]:
    """Per factor: the folder of its LightGBM population models (trained again when the source models of script
    11 or the loss changed) and its choices.yaml; and the log of the trainings of this run."""
    import shutil
    from datetime import datetime

    from t1dretrain.sim.samples import concat_samples, save_samples

    entry = choices.get("lightgbm") or {}
    if not entry.get("population"):
        raise SystemExit("choices.yaml of script 11 has no LightGBM population model")
    root = derived / "runs" / SCRIPT_ID / "population"
    folders, todo = {}, []
    for factor in factors:
        folder = root / variant_name(factor)
        wanted = {"threshold": float(loss["threshold"]), "factor": float(factor)}
        path = folder / "choices.yaml"
        old = (yaml.safe_load(path.read_text()) or {}).get("lightgbm", {}) if path.exists() else {}
        current = (old.get("low_loss") == wanted and old.get("source_created") == entry.get("created") and
                   all(population_path(folder, "lightgbm", s).exists() for s in seeds))
        folders[factor] = folder
        if not current:
            todo.append((factor, folder, wanted))
    if not todo:
        return folders, pd.DataFrame()
    pool = list((choices.get("pool") or {}).get("fit", [])) + list((choices.get("pool") or {}).get("check", []))
    stage = config["stage0"]
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]],
                "lookback_hours": float(config["lookback_hours"]), "min_window_share": float(stage["min_window_share"]),
                "recent_complete_min": float(stage["recent_complete_min"]),
                "population_stride_min": float(choices.get("population_stride_min",
                                                           config["population"]["stride_min"]))}
    built = parallel_map(run_pool, [{"dataset": core, "patient_id": p, "derived_root": derived, "settings": settings,
                                     "day0": None} for p in pool], resources.n_workers, label="pool patients")
    failed = [r.patient_id for r in built if r.error or r.population is None]
    if failed:
        print(f"Pool patients without samples (left out): {failed}")
    samples = concat_samples([r.population for r in built if not r.error and r.population is not None])
    cache = root / "samples"
    shutil.rmtree(cache, ignore_errors=True)
    save_samples(samples, cache)
    print(f"Pool: {len(pool)} patients, {samples.n} samples; {len(todo) * len(seeds)} population models to train")
    del samples
    jobs = [{"key": f"lightgbm_{variant_name(factor)}_seed{s}", "family": "lightgbm",
             "params": entry["population"], "seed": int(s), "cache": str(cache), "low_loss": wanted,
             "out": str(population_path(folder, "lightgbm", int(s)))}
            for factor, folder, wanted in todo for s in seeds]
    threads = max(1, resources.n_workers // len(jobs))
    for job in jobs:
        job["threads"] = threads
    runs = parallel_map(run_train, jobs, min(len(jobs), resources.n_workers), label="population models",
                        threads_per_worker=threads)
    shutil.rmtree(cache, ignore_errors=True)
    failed = [r for r in runs if r["status"] != "ok"]
    if failed:
        raise SystemExit("population training failed: " + "; ".join(r.get("error", "")[-300:] for r in failed))
    created = datetime.now().isoformat(timespec="seconds")
    for factor, folder, wanted in todo:
        own = {**choices, "lightgbm": {**entry, "low_loss": wanted, "source_created": entry.get("created"),
                                       "created": created, "release": release}}
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "choices.yaml").write_text(yaml.safe_dump(own, sort_keys=False))
    table = pd.DataFrame([{k: r.get(k) for k in ("key", "seed", "status", "samples", "rounds", "seconds")}
                          for r in runs])
    return folders, table


# ============================================================================ main


def main() -> None:
    import json
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
    results = Path(out.dir).parent
    core = config.get("core_dataset", "replace_bg")
    conf, cfg = config["confirmation"], config["low_glucose"]
    loss = cfg["loss"]
    seed = int(config["random_seed"])
    replicates = int(config["stage0"]["bootstrap_replicates"])
    min_events = int(cfg.get("min_events", 20))
    factors = [float(f) for f in loss["factors"]]
    source_days = derived / "runs" / SOURCE / "day_metrics.parquet"
    if not source_days.exists():
        raise SystemExit(f"{source_days} not found: run scripts/50_confirmation_runs.py first")
    population_dir = derived / "models" / core / "population"
    choices = cr.read_choices(population_dir)
    base = cr.base_settings(config, choices, population_dir)
    forgetting = float((base.get("arx") or {}).get("forgetting", 1.0))
    policies_doc, mechanisms_doc = cr.read_plan(derived)
    models, plan, notes = cr.plan_models(policies_doc, mechanisms_doc, list(loss["models"]), population_dir,
                                         base["seeds"])
    models = [m for m in models if not is_twin(m)]
    cohorts = [c for c in conf["cohorts"] if cf.cohort_key(c["dataset"], c["cohort"]) in set(loss["cohorts"])]
    lightgbm = any(m.startswith("lightgbm") for m in models)
    folders, trained = (population_models(derived, core, choices, loss, factors, base["seeds"], config, release,
                                          resources) if lightgbm else ({}, pd.DataFrame()))

    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    jobs, cohort_rows, all_anchors, arms_of, by_variant = [], [], [], {}, {}
    for order, c in enumerate(cohorts):
        key, rule = cf.cohort_key(c["dataset"], c["cohort"]), cohort_rule(c["dataset"], c["cohort"])
        anchors, counts = cr.cohort_patients(c, derived, [int(a) for a in conf["anchor_days"]], seed, args.patients)
        runnable = cf.models_for_rule(models, rule)
        arms_of[key] = {m: cf.confirmation_arms(plan[m]["policies"], plan[m]["mechanism"], plan[m]["interval"],
                                                False, forgetting) for m in runnable}
        for arms in arms_of[key].values():
            for arm in arms:
                parse_arm(arm)
        test_days = int(c.get("test_days", conf["test_days"]))
        cohort_rows.append({"cohort": key, "usable_rule": rule, **counts, "test_days": test_days,
                            "models": " ".join(runnable)})
        all_anchors.append(anchors.assign(cohort=key))
        for factor in factors:
            name = variant_name(factor)
            own_dir = folders.get(factor, population_dir)
            own_choices = yaml.safe_load((own_dir / "choices.yaml").read_text()) if factor in folders else choices
            settings = {**base, "test_days": test_days, "population_dir": str(own_dir),
                        "low_loss": {"threshold": float(loss["threshold"]), "factor": factor}}
            part = cr.cohort_jobs(key, c["dataset"], rule, order, anchors, arms_of[key], settings, own_choices,
                                  runs / name, recompute, derived, cr.device_for(resources), stream0=len(jobs))
            by_variant.setdefault(name, []).extend(part)
            jobs += part
    cohort_table = pd.DataFrame(cohort_rows)
    print(cohort_table.to_string(index=False))
    jobs.sort(key=cr.job_order)
    print(f"Models {models}; factors {factors}; {len(jobs)} jobs")
    statuses = parallel_map(cr.run_job, jobs, resources.n_workers, label="low-loss jobs")

    frames, infos, clarke_tables, missing = [], [], [], []
    for name, part in by_variant.items():
        days, info, clarke, lost = cr.collect(part)
        frames.append(days.assign(variant=name))
        infos += [{**row, "variant": name} for row in info]
        clarke_tables.append((name, clarke))
        missing += lost
    own_days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    runs.mkdir(parents=True, exist_ok=True)
    own_days.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log = pd.DataFrame(infos)
    task_log = cr.task_table(statuses)

    test_days = {r.cohort: int(r.test_days) for r in cohort_table.itertuples()}
    standard = standard_days(source_days, arms_of, test_days)
    days = pd.concat([standard.assign(variant=va.STANDARD), own_days], ignore_index=True)
    skill = va.skill_table(days)
    lows = lg.patient_lows(days, va.KEYS, min_events)
    summary = va.variant_summary(skill, replicates, seed)
    versus = va.against_standard(skill, lows, replicates, seed)
    gain = va.refit_gain(skill, replicates, seed)
    metrics = va.pooled_metrics(days)
    histogram = histograms(results / SOURCE / "clarke_histogram.csv", clarke_tables, arms_of)
    curves = lg.alarm_curves(histogram, CURVE_KEYS)
    matched = lg.matched(curves, CURVE_KEYS, ignore=("policy", "variant"))
    ranges = lg.range_table(histogram, CURVE_KEYS)
    own_clarke = merged(clarke_tables)

    out.csv(cohort_table, "cohorts")
    out.csv(trained, "population_models")
    out.csv(summary.round(5), "skill_by_variant")
    out.csv(versus.round(5), "vs_standard")
    out.csv(gain.round(5), "refit_gain")
    out.csv(metrics.round(4), "metrics_by_variant")
    out.csv(ranges.round(4), "range_metrics")
    out.csv(curves.round(3), "alarm_curves")
    out.csv(matched.round(3), "matched_alarms")
    out.csv(own_clarke.zones(), "clarke_zones")
    out.csv(own_clarke.histogram(), "clarke_histogram")
    out.csv(own_clarke.sample(), "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(task_log, "task_log")
    out.csv(pd.concat(all_anchors, ignore_index=True).drop(columns=["dataset"], errors="ignore"), "jobs")
    if not curves.empty:
        out.figure(plot_curves(curves, models, factors), "alarm_curves")
    if not versus.empty:
        out.figure(plot_tradeoff(versus, models, factors), "tradeoff")
    manifest = json.loads((results / SOURCE / "run_manifest.json").read_text())
    out.text(write_summary(cohort_table, trained, versus, gain, matched, task_log, fit_log, missing, notes, models,
                           factors, loss, release, resources, manifest), "summary.md")
    out.manifest(settings={"loss": loss, "models": models, "arms": arms_of, "cohorts": cohorts,
                           "min_events": min_events, "bootstrap_replicates": replicates,
                           "population_models": {variant_name(f): str(p) for f, p in folders.items()},
                           "source_release": manifest.get("package_version")},
                 notes="Standard models: the day metrics and Clarke histogram of script 50 on the same patients, arms "
                       "and test days.")
    print(f"Done. Results in {out.dir}")


def standard_days(path: Path, arms_of: dict, test_days: dict) -> pd.DataFrame:
    """The day metrics of script 50 for the cohorts, models and arms of this run (and persistence), on the same
    test days."""
    models = sorted({m for by_model in arms_of.values() for m in by_model} | {"persistence"})
    days = pd.read_parquet(path, filters=[("cohort", "in", sorted(arms_of)), ("model", "in", models)])
    keep = []
    for cohort, by_model in arms_of.items():
        part = days[(days["cohort"] == cohort) & (days["days_since_anchor"] < test_days[cohort])]
        wanted = part["model"].eq("persistence")
        for model, arms in by_model.items():
            wanted |= part["model"].eq(model) & part["policy"].isin(arms)
        keep.append(part[wanted])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


def histograms(path: Path, clarke_tables: list, arms_of: dict) -> pd.DataFrame:
    """Clarke histograms of the standard models (script 50, the same cohorts, models and arms) and of every factor."""
    parts = []
    if path.exists():
        h = pd.read_csv(path)
        wanted = pd.Series(False, index=h.index)
        for cohort, by_model in arms_of.items():
            here = h["cohort"] == cohort
            wanted |= here & h["model"].eq("persistence")
            for model, arms in by_model.items():
                wanted |= here & h["model"].eq(model) & h["policy"].isin(arms)
        parts.append(h[wanted].assign(variant=va.STANDARD))
    for name, clarke in clarke_tables:
        parts.append(clarke.histogram().assign(variant=name))
    parts = [p for p in parts if not p.empty]
    return lg.pooled(pd.concat(parts, ignore_index=True), CURVE_KEYS) if parts else pd.DataFrame()


def merged(clarke_tables: list):
    from t1dretrain.eval.metrics import ClarkeAccumulator

    out = ClarkeAccumulator(["variant", "cohort", "model", "policy", "horizon_min", "period"], sample_size=20000)
    for name, clarke in clarke_tables:
        out.merge(clarke.tagged(variant=name))
    return out


# ============================================================================ figures


def plot_curves(curves: pd.DataFrame, models: list[str], factors: list[float], horizons=(30, 60)):
    """Alarm curves (lows below 70 mg/dL) of never with every loss: solid, the standard loss; dashed and dotted, the
    weighted losses; grey, persistence; dots: the alarm at 70 mg/dL."""
    c = curves[(curves["event"] == "below70") & curves["policy"].isin(["never", "reference"])]
    cohorts = list(dict.fromkeys(c["cohort"]))
    fig, axes = model_axes(len(cohorts) * len(horizons), 4.4, 3.6, ncols=len(horizons))
    styles = {va.STANDARD: "-", **{variant_name(f): LINESTYLES[i % 3] for i, f in enumerate(factors)}}
    for i, cohort in enumerate(cohorts):
        for j, horizon in enumerate(horizons):
            ax = axes[i * len(horizons) + j]
            h = c[(c["cohort"] == cohort) & (c["horizon_min"] == horizon)]
            p = h[(h["model"] == "persistence") & (h["variant"] == va.STANDARD)].sort_values("alarm")
            ax.plot(p["sensitivity_pct"], p["precision_pct"], color=color("persistence"), lw=1.0, label="Persistence")
            for model in models:
                for variant, style in styles.items():
                    m = h[(h["model"] == model) & (h["variant"] == variant)].sort_values("alarm")
                    if m.empty:
                        continue
                    ax.plot(m["sensitivity_pct"], m["precision_pct"], color=color(model), ls=style, lw=1.2,
                            label=f"{label(model)}, {variant}")
                    at = m[m["alarm"] == 70]
                    ax.scatter(at["sensitivity_pct"], at["precision_pct"], color=color(model), s=14, zorder=3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel("sensitivity (%)")
            ax.set_ylabel("precision (%)")
            ax.set_title(f"{cohort}, never, {horizon} min", loc="left", fontsize=8)
        axes[i * len(horizons)].legend(fontsize=6, loc="upper right")
    fig.suptitle("Alarm curves with the standard loss and with the lows weighted (reference below 70 mg/dL; dots: "
                 "alarm at 70 mg/dL)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_tradeoff(versus: pd.DataFrame, models: list[str], factors: list[float]):
    """Per cohort: what each weighted loss costs in skill (x) and gains in sensitivity below 70 mg/dL (y), never
    and the refit at the optimal interval, per horizon (paired means over patients, 95 % intervals)."""
    cohorts = list(dict.fromkeys(versus["cohort"]))
    fig, axes = model_axes(len(cohorts), 5.0, 3.8, ncols=2)
    markers = {30: "o", 60: "s", 120: "^"}
    for ax, cohort in zip(axes, cohorts):
        v = versus[versus["cohort"] == cohort]
        for model in models:
            for factor in factors:
                name = variant_name(factor)
                s = v[(v["model"] == model) & (v["variant"] == name) & (v["measure"] == "skill")]
                e = v[(v["model"] == model) & (v["variant"] == name) & (v["measure"] == "sensitivity_70")]
                both = s.merge(e, on=["policy", "horizon_min"], suffixes=("_s", "_e"))
                for r in both.itertuples():
                    alpha = 0.45 if r.policy == "never" else 0.95
                    ax.plot([100 * r.low_s, 100 * r.high_s], [r.difference_e] * 2, color=color(model), lw=0.8,
                            alpha=alpha)
                    ax.plot([100 * r.difference_s] * 2, [r.low_e, r.high_e], color=color(model), lw=0.8, alpha=alpha)
                    ax.scatter(100 * r.difference_s, r.difference_e, color=color(model), alpha=alpha,
                               marker=markers.get(int(r.horizon_min), "o"), s=18 + 10 * factors.index(factor))
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.7)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.7)
        ax.set_xlabel("skill, weighted - standard (points)")
        ax.set_ylabel("sensitivity below 70, weighted - standard (points)")
        ax.set_title(cohort, loc="left", fontsize=9)
    fig.suptitle("What weighting the lows costs and gains (circle 30, square 60, triangle 120 min; larger: heavier "
                 "weight; light: never, dark: refit)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(cohorts, trained, versus, gain, matched, task_log, fit_log, missing, notes, models, factors, loss,
                  release, resources, manifest) -> str:
    lines = ["# 71 - A loss that weighs the lows", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Models {models}; targets below {loss['threshold']} mg/dL weigh {factors} times in every fit "
             f"(LightGBM: population model trained again on the pool). Against the standard models of script 50 "
             f"(release {manifest.get('package_version')}) on the same patients, arms and days. D96; nothing is "
             "chosen here.", "", "## Cohorts", "", markdown_table(cohorts), ""]
    if not trained.empty:
        lines += ["## Population models trained in this run", "", markdown_table(trained), ""]
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
        v = versus[versus["measure"].isin(["skill", "rmse", "sensitivity_70", "precision_70"])].copy()
        v["value"] = [f"{(100 if m == 'skill' else 1) * d:+.2f} ({(100 if m == 'skill' else 1) * lo:+.2f} to "
                      f"{(100 if m == 'skill' else 1) * hi:+.2f})" for m, d, lo, hi in
                      zip(v["measure"], v["difference"], v["low"], v["high"])]
        table = v.pivot_table(index=["cohort", "model", "policy", "variant", "horizon_min"], columns="measure",
                              values="value", aggfunc="first").reset_index()
        lines += ["## Weighted minus standard loss, paired per patient (skill in points; RMSE in mg/dL; "
                  "sensitivity and precision below 70 mg/dL in points)", "", markdown_table(table), "",
                  "54 mg/dL and every patient count are in `vs_standard.csv`.", ""]
    if not matched.empty:
        m = matched[(matched["event"] == "below70") & matched["policy"].isin(["never"]) &
                    matched["horizon_min"].isin([30, 60])]
        lines += ["## Alarm curves of never (reference below 70 mg/dL, %)", "",
                  "A weighted loss that only moves the model along its curve changes the sensitivity at 70 mg/dL but "
                  "not the sensitivity at persistence's precision nor the average precision.", "",
                  markdown_table(m[["cohort", "model", "variant", "horizon_min", "sensitivity_pct", "precision_pct",
                                    "sensitivity_at_reference_precision_pct", "alarm_for_reference_precision",
                                    "average_precision_pct"]].round(1)), ""]
    if not gain.empty:
        g = gain[gain["horizon_min"].isin([30, 60, 120])].copy()
        for column in ("gain", "low", "high", "gain_difference", "difference_low", "difference_high"):
            if column in g:
                g[column] = (100 * g[column]).round(2)
        show = [c for c in ("cohort", "model", "policy", "variant", "horizon_min", "patients", "gain", "low", "high",
                            "gain_difference", "difference_low", "difference_high") if c in g]
        lines += ["## Gain of refitting over never with each loss (points)", "", markdown_table(g[show]), ""]
    lines += ["## Checks to look at", "",
              "- `alarm_curves.png`: whether the weighted losses lift the curve or only move along it.",
              "- `tradeoff.png`: the skill each weighted loss costs for the sensitivity it gains.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
