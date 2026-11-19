"""11 - Hyperparameters and population pretraining on the pool group.

Stage 0 (script 10) showed that personal models trained only on a patient's
first 14 days do not beat AR(3): LightGBM ties with it and N-HiTS stays at the
level of persistence. This script prepares the other two starting points of
the learned models (D62) and tunes all three, using only the pool group of the
core dataset, which is never evaluated (D12):

1. Samples of every pool patient: all usable days (population samples) and
   the stage-0 split (14 days from day 0, later test days).
2. The pool is split once into *fit* patients (80 %) and *check* patients
   (20 %). Candidate population models (a small grid per family) are trained
   on the fit patients, with early stopping on the check patients' first 14
   days.
3. On the check patients, as a new patient would be met: every candidate
   population model zero-shot (``*_pop``), and the best one fine-tuned on the
   first 14 days with a small grid of fine-tuning settings (``*_ft``). On every
   pool patient: a small grid of scratch settings (``lightgbm``, ``nhits``; the
   networks added in iteration 19 have no scratch grid, D68).
   Every candidate is scored by its skill over persistence on the later days
   (mean over patients and horizons), with RMSE and Clarke zones (R6).
4. The chosen population settings are refitted on all pool patients, once per
   seed of stage 0, with the rounds (LightGBM) or steps (networks) of the tuning
   run, and saved with the chosen scratch and fine-tuning settings.
5. Glucose-only twins (D76, D90): for the families of
   ``glucose_only_twins.families``, the same final models without exogenous
   inputs (glucose only: no IOB, COB, insulin activity, carbohydrate
   absorption or time of day), with the same settings, rounds or steps
   (``<family>_noexog_seed<k>``); the later scripts run them next to the full
   models. ``--twins-only`` trains only these, from the saved ``choices.yaml``,
   without tuning again (report in ``twins/``).

The inputs are not tuned: every full model reads them (D90).

The families are LightGBM and the networks of ``models.neural.NETWORKS``
(N-HiTS, TiDE, TSMixerx, LSTM, PatchTST) that have a ``population.candidates``
block. ``--families`` trains only some of them; the others keep their models
and their entries in ``choices.yaml`` (D71).

Script 12 compares the three starting points on the screen group.

Derived data:
    <derived_data>/models/<core>/population/   lightgbm_seed<k>/h<h>.txt, <network>_seed<k>.pt, choices.yaml
    <derived_data>/models/<core>/tuning/       candidate population models (trained on the fit patients)
    <derived_data>/models/<core>/cache/        pool samples (.npy) for the training processes

Outputs (results/11_pretrain_population/):
    pool_patients.csv        per pool patient: role (fit / check), samples, usable days, status
    tuning_runs.csv          candidate population models: settings, rounds or steps, validation error, seconds
    tuning_scores.csv        every candidate (scratch, population, fine-tuned): skill by horizon on the pool
    learning_curves.csv      validation error of the N-HiTS candidates every ``val_every`` steps
    final_models.csv         the saved population models (one per family and seed)
    chosen_settings.yaml     copy of choices.yaml
    check_summary.csv        chosen starting points on the check patients: patient RMSE and skill with intervals
    check_metrics.csv        chosen starting points, pooled forecasts: RMSE, MAE, MARD, bias, Clarke zones (R6)
    check_by_patient.csv     per check patient, model and horizon
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6, chosen starting points)
    fit_log.csv              every fit of the scoring jobs
    task_log.csv
    tuning_scores.png
    learning_curves.png
    check_skill.png
    clarke_grids.png
    summary.md
    run_manifest.json

Usage:
    python scripts/11_pretrain_population.py
    python scripts/11_pretrain_population.py --families tide tsmixerx lstm patchtst   # only these families
    python scripts/11_pretrain_population.py --twins-only      # only the glucose-only twins (D90)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.splits import group_patients  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import (FIT_NOTE, color, fit_outcomes, label, model_summary,  # noqa: E402
                                           overall_table, patient_table, plot_grids, plot_skill)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, set_style  # noqa: E402
from t1dretrain.sim.population import expand_grid, split_pool  # noqa: E402
from t1dretrain.sim.samples import concat_samples, save_samples  # noqa: E402
from t1dretrain.models.neural import NETWORKS  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, TWIN, population_path, twin_families  # noqa: E402

SCRIPT_ID = "11_pretrain_population"
START_SUFFIX = {"scratch": "", "population": "_pop", "finetune": "_ft"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--families", nargs="*", default=None, choices=FAMILIES, help="subset of model families")
    parser.add_argument("--twins-only", action="store_true",
                        help="train only the glucose-only twins of glucose_only_twins.families (D90)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ jobs (module level for the workers)


def run_pool(job: dict):
    from t1dretrain.sim.population import pool_patient

    with WarningRecorder() as recorder:
        result = pool_patient(job)
    if result.error == "" and recorder.text():
        result.info = {**(result.info or {}), "warnings": recorder.text()[:300]}
    return result


def run_score(job: dict):
    from t1dretrain.sim.population import score_patient

    with WarningRecorder() as recorder:
        result = score_patient(job)
    result.warnings = recorder.text()
    return result


def run_train(job: dict) -> dict:
    from t1dretrain.sim.population import train_population

    return train_population(job)


# ============================================================================ main


def gpu(k: int, resources) -> str:
    return "cuda" if resources.device == "cuda" and resources.n_gpus else "cpu"  # one GPU per worker process (D73)


def train_models(jobs: list[dict], resources) -> list[dict]:
    """LightGBM jobs on CPU threads and network jobs on the GPUs, all at once (rule R5; D73 for the GPU sharing)."""
    if not jobs:
        return []
    neural = [j for j in jobs if j["family"] in NETWORKS]
    trees = [j for j in jobs if j["family"] == "lightgbm"]
    for k, job in enumerate(neural):
        job["device"] = gpu(k, resources)
    n_workers = max(1, min(resources.n_workers, len(jobs)))
    threads = max(1, (resources.n_workers - min(len(neural), n_workers)) // max(1, min(len(trees), n_workers)))
    for job in trees:
        job["threads"] = threads
    return parallel_map(run_train, jobs, n_workers, label="model trainings", threads_per_worker=threads)


def score(pool: dict, variants_of: dict, settings: dict, resources, stream0: int) -> list:
    """One scoring job per patient: fit every variant on the 14-day window, test on the later days."""
    jobs = []
    for k, (pid, variants) in enumerate(variants_of.items()):
        patient = pool[pid]
        jobs.append({"patient": patient, "variants": variants, "device": gpu(k, resources),
                     "clarke_point_sample": settings["clarke_point_sample"], "stream": stream0 + k})
    return parallel_map(run_score, jobs, resources.n_workers, label="patients")


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    print(f"GPU sharing: at most {share_gpus(config)} processes compute on one GPU at a time (D73)")
    out = ScriptOutput(SCRIPT_ID, run_tag="twins" if args.twins_only else None, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    pop, stage = config["population"], config["stage0"]
    families = [f for f in FAMILIES if f in pop["candidates"] and (not args.families or f in args.families)]
    twinned = [f for f in (twin_families(config) or []) if f in families]
    core = config.get("core_dataset", "replace_bg")
    seed = int(config["random_seed"])
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]), "warmup_days": int(config["warmup_days"]),
                "train_stride_min": float(stage["train_stride_min"]), "validation_days": int(stage["validation_days"]),
                "min_window_share": float(stage["min_window_share"]),
                "recent_complete_min": float(stage["recent_complete_min"]),
                "population_stride_min": float(pop["stride_min"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"])}

    # ------------------------------------------------------------------ 1. pool samples
    patients = group_patients(core, pop["group"])
    if pop["group"] != "pool":
        raise SystemExit("population models are trained on the pool group only (D12)")
    anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == "core") & (anchors["period"] == "whole")].drop_duplicates("patient_id")
    day0 = dict(zip(anchors["patient_id"], anchors["day0"]))
    jobs = [{"dataset": core, "patient_id": pid, "derived_root": derived, "settings": settings, "day0": day0.get(pid)}
            for pid in patients]
    built = parallel_map(run_pool, jobs, resources.n_workers, label="pool patients")
    usable = [r for r in built if not r.error and r.population is not None and r.population.n]
    scored = sorted(r.patient_id for r in usable if r.warmup is not None and r.warmup.n and r.test is not None
                    and r.test.n)
    fit_ids, check_ids = split_pool(scored, float(pop["check_share"]), seed)
    fit_ids = sorted(fit_ids + [r.patient_id for r in usable if r.patient_id not in set(scored)])
    pool_table = pd.DataFrame([{**(r.info or {"patient_id": r.patient_id}), "role": "fit" if r.patient_id in fit_ids
                                else "check" if r.patient_id in check_ids else "unused",
                                "status": "error" if r.error else ("ok" if r.patient_id in fit_ids + check_ids
                                                                   else "no samples"),
                                "seconds": r.seconds, "error": r.error.strip().splitlines()[-1] if r.error else ""}
                               for r in built])

    models_dir = derived / "models" / core
    cache, tuning_dir, final_dir = models_dir / "cache", models_dir / "tuning", models_dir / "population"
    shutil.rmtree(cache, ignore_errors=True)
    by_id = {r.patient_id: r for r in usable}
    if args.twins_only:
        run_twins_only(out, by_id, fit_ids, check_ids, twinned, final_dir, cache, settings, stage,
                       pop_table=pool_table, release=release, resources=resources)
        return
    for f in families:  # only the families of this run: the others keep their models and settings
        for old in (list(tuning_dir.glob(f"{f}_*")) + list(final_dir.glob(f"{f}_seed*")) +
                    list(final_dir.glob(f"{f}_{TWIN}_seed*"))):
            shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink()
    tune = concat_samples([by_id[p].population for p in fit_ids] + [by_id[p].warmup for p in check_ids])
    held = np.zeros(tune.n, dtype=bool)
    held[sum(by_id[p].population.n for p in fit_ids):] = True
    save_samples(tune, cache / "tune")
    np.save(cache / "tune" / "validation.npy", held)
    tune_size, tune_held = int(tune.n), int(held.sum())
    del tune
    final = concat_samples([by_id[p].population for p in fit_ids + check_ids])
    save_samples(final, cache / "final")
    final_size = int(final.n)
    del final
    for r in usable:
        r.population = None  # the scoring jobs only need the 14-day windows and the test days
    print(f"Pool: {len(fit_ids)} fit and {len(check_ids)} check patients; {tune_size} tuning samples "
          f"({tune_held} held out), {final_size} final samples")

    # ------------------------------------------------------------------ 2. candidate population models
    grids = {f: {start: expand_grid(pop[block][f]["base"], pop[block][f].get("grid")) if f in (pop.get(block) or {})
                 else [] for start, block in (("scratch", "scratch"), ("population", "candidates"),
                                              ("finetune", "finetune"))}
             for f in families}  # a family without a scratch (or fine-tuning) block skips that starting point
    keys = {f: {start: [f"{f}_{start}_c{i}" for i in range(len(grids[f][start]))] for start in grids[f]}
            for f in families}
    tuning_seed = int(pop.get("tuning_seed", 1))
    train_jobs = [{"key": key, "family": f, "params": params, "seed": tuning_seed, "cache": str(cache / "tune"),
                   "out": str(population_path(tuning_dir / key, f, tuning_seed))}
                  for f in families for key, params in zip(keys[f]["population"], grids[f]["population"])]
    tuning_runs = train_models(train_jobs, resources)
    trained = {r["key"]: r for r in tuning_runs if r["status"] == "ok"}

    # ------------------------------------------------------------------ 3. scoring on the pool
    scratch_variants = [{"key": "persistence", "family": "persistence"}, {"key": "ar3", "family": "ar3"}] + \
        [{"key": key, "family": f, "start": "scratch", "params": params, "seed": tuning_seed}
         for f in families for key, params in zip(keys[f]["scratch"], grids[f]["scratch"])]
    population_variants = [{"key": key, "family": f, "start": "population", "seed": tuning_seed,
                            "population": str(population_path(tuning_dir / key, f, tuning_seed))}
                           for f in families for key in keys[f]["population"] if key in trained]
    variants_of = {pid: scratch_variants + (population_variants if pid in check_ids else []) for pid in scored}
    results = score(by_id, variants_of, settings, resources, stream0=0)
    days = collect_days(results)
    by_patient = patient_table(days) if not days.empty else pd.DataFrame()
    scores = score_table(by_patient)
    chosen = {}
    for f in families:
        chosen[f] = {"scratch": best(scores, keys[f]["scratch"]), "population": best(scores, keys[f]["population"])}
    finetune_variants = []
    for f in families:
        base = chosen[f]["population"]
        if base is None:
            continue
        finetune_variants += [{"key": key, "family": f, "start": "finetune", "seed": tuning_seed, "finetune": params,
                               "population": str(population_path(tuning_dir / base, f, tuning_seed))}
                              for key, params in zip(keys[f]["finetune"], grids[f]["finetune"])]
    if finetune_variants:
        results += score(by_id, {pid: finetune_variants for pid in check_ids}, settings, resources,
                         stream0=len(variants_of))
        days = collect_days(results)
        by_patient = patient_table(days)
        scores = score_table(by_patient)
        for f in families:
            chosen[f]["finetune"] = best(scores, keys[f]["finetune"])

    # ------------------------------------------------------------------ 4. final population models
    seeds = [int(s) for s in stage["seeds"]]
    created = datetime.now().isoformat(timespec="seconds")
    choices = {"dataset": core, "release": release, "created": created,
               "population_stride_min": settings["population_stride_min"], "seeds": seeds,
               "pool": {"fit": fit_ids, "check": check_ids, "tuning_samples": tune_size, "final_samples": final_size}}
    kept = keep_other_families(final_dir / "choices.yaml", choices, families)
    final_jobs = []
    for f in families:
        index = {start: {k: i for i, k in enumerate(keys[f][start])} for start in keys[f]}
        entry = {}
        if chosen[f].get("scratch"):
            entry["scratch"] = grids[f]["scratch"][index["scratch"][chosen[f]["scratch"]]]
        if chosen[f].get("population"):
            params = dict(grids[f]["population"][index["population"][chosen[f]["population"]]])
            run = trained[chosen[f]["population"]]
            if f == "lightgbm":
                params["max_rounds"] = [int(r) for r in run["rounds"]]
            else:  # at least one evaluation interval, even if the first evaluation was the best
                params["max_steps"] = max(int(run["best_step"]), int(params.get("val_every", 1)))
            entry["population"] = params
            final_jobs += [{"key": f"{f}_final_seed{s}", "family": f, "params": params, "seed": s,
                            "cache": str(cache / "final"), "out": str(population_path(final_dir, f, s))} for s in seeds]
            if f in twinned:
                entry[TWIN] = {"population": {**params, "exog": False}}
                final_jobs += twin_jobs(f, params, seeds, cache / "final", final_dir)
        if chosen[f].get("finetune"):
            entry["finetune"] = grids[f]["finetune"][index["finetune"][chosen[f]["finetune"]]]
        entry["chosen_keys"] = {k: v for k, v in chosen[f].items() if v}
        entry["created"], entry["release"] = created, release
        choices[f] = entry
    final_runs = train_models(final_jobs, resources)
    failed = [r for r in final_runs if r["status"] != "ok"]
    if final_jobs and not failed:
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "choices.yaml").write_text(yaml.safe_dump(to_plain({**choices, **kept}), sort_keys=False))
    shutil.rmtree(cache, ignore_errors=True)

    # ------------------------------------------------------------------ 5. report
    canonical = {chosen[f][start]: f + START_SUFFIX[start] for f in families for start in chosen[f] if chosen[f][start]}
    canonical.update({"persistence": "persistence", "ar3": "ar3"})
    check_days = days[days["patient_id"].isin(check_ids) & days["model"].isin(canonical)].copy() \
        if not days.empty else pd.DataFrame()
    if not check_days.empty:
        check_days["model"] = check_days["model"].map(canonical)
    replicates = int(stage["bootstrap_replicates"])
    check_patient = patient_table(check_days) if not check_days.empty else pd.DataFrame()
    check_overall = overall_table(check_days, check_patient, replicates, seed) if not check_days.empty else pd.DataFrame()
    check_summary = model_summary(check_patient, replicates, seed) if not check_patient.empty else pd.DataFrame()
    clarke = ClarkeAccumulator(["model", "seed", "horizon_min", "period"],
                               sample_size=settings["clarke_point_sample"])
    for r in results:
        if not r.error and r.clarke is not None:
            clarke.merge(r.clarke)
    histogram = rename_models(clarke.histogram(), canonical)
    task_log = pd.DataFrame([{"stage": "scoring", "patient_id": r.patient_id, "status": "error" if r.error else "ok",
                              "seconds": r.seconds, "warnings": r.warnings,
                              "error": r.error.strip().splitlines()[-1] if r.error else "", "traceback": r.error}
                             for r in results] +
                            [{"stage": "training", "patient_id": r["key"], "status": r["status"],
                              "seconds": r["seconds"], "warnings": "",
                              "error": r.get("error", "").strip().splitlines()[-1] if r.get("error") else "",
                              "traceback": r.get("error", "")} for r in tuning_runs + final_runs])
    fit_log = pd.DataFrame([row for r in results if not r.error for row in r.info])
    curves = pd.DataFrame([{"key": r["key"], "step": s, "val_mse": v} for r in tuning_runs
                           for s, v in (r.get("curve") or [])])
    runs_table = runs_frame(tuning_runs)
    finals_table = runs_frame(final_runs)
    if not by_patient.empty:  # AR(3) on the check patients only, the fair line for population and fine-tuned
        on_check = by_patient[by_patient["patient_id"].isin(check_ids) & (by_patient["model"] == "ar3")]
        scores = pd.concat([scores, score_table(on_check).assign(model="ar3_check")], ignore_index=True)
    scores = annotate_scores(scores, grids, keys, chosen, families)

    out.csv(pool_table, "pool_patients")
    out.csv(runs_table, "tuning_runs")
    out.csv(scores, "tuning_scores")
    out.csv(curves, "learning_curves")
    out.csv(finals_table, "final_models")
    out.text(yaml.safe_dump(to_plain(choices), sort_keys=False), "chosen_settings.yaml")
    out.csv(check_summary, "check_summary")
    out.csv(check_overall, "check_metrics")
    out.csv(check_patient, "check_by_patient")
    out.csv(rename_models(clarke.zones(), canonical), "clarke_zones")
    out.csv(histogram, "clarke_histogram")
    out.csv(rename_models(clarke.sample(), canonical), "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(fit_outcomes(fit_log), "fit_outcomes")
    out.csv(task_log, "task_log")
    if not scores.empty:
        out.figure(plot_scores(scores, families), "tuning_scores")
    if not curves.empty:
        out.figure(plot_curves(curves, runs_table, chosen), "learning_curves")
    order = ["persistence", "ar3"] + [f + START_SUFFIX[s] for f in families for s in START_SUFFIX]
    present = [m for m in order if not check_patient.empty and m in set(check_patient["model"])]
    if present:
        out.figure(plot_skill(check_patient, "whole", present), "check_skill")
        out.figure(plot_grids(histogram, present, settings["horizons_min"], "whole", tuning_seed, group="pool (check)"),
                   "clarke_grids")
    out.text(write_summary(pool_table, runs_table, scores, check_summary, check_overall, finals_table, choices, task_log,
                           fit_log, settings, families, release, resources, final_dir if final_jobs and not failed else None),
             "summary.md")
    out.manifest(settings={**settings, "population": pop, "families": families, "pool_patients": len(patients),
                           "fit_patients": len(fit_ids), "check_patients": len(check_ids)},
                 notes="Population models in <derived>/models/<core>/population (choices.yaml); candidates in tuning/.")
    print(f"Done. Results in {out.dir}")


# ============================================================================ glucose-only twins (D76, D90)


def twin_jobs(family: str, params: dict, seeds: list[int], cache: Path, final_dir: Path) -> list[dict]:
    """Final population models of a family without exogenous inputs: the chosen settings, rounds or steps."""
    return [{"key": f"{family}_{TWIN}_seed{s}", "family": family, "params": {**params, "exog": False}, "seed": s,
             "cache": str(cache), "out": str(population_path(final_dir, family, s, TWIN))} for s in seeds]


def run_twins_only(out, by_id: dict, fit_ids: list, check_ids: list, families: list[str], final_dir: Path,
                   cache: Path, settings: dict, stage: dict, pop_table: pd.DataFrame, release: str, resources) -> None:
    """``--twins-only``: the glucose-only population models of ``families`` from the saved choices.yaml (same pool
    split, settings, rounds and steps), added to it; the full models are not touched."""
    path = final_dir / "choices.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found: run script 11 without --twins-only first")
    choices = yaml.safe_load(path.read_text()) or {}
    pool = choices.get("pool") or {}
    if pool.get("fit") != fit_ids or pool.get("check") != check_ids:
        raise SystemExit("choices.yaml comes from another pool split: run script 11 without --twins-only")
    families = [f for f in families if isinstance(choices.get(f), dict) and choices[f].get("population")]
    if not families:
        raise SystemExit("no family of glucose_only_twins.families has a population model in choices.yaml")
    final = concat_samples([by_id[p].population for p in fit_ids + check_ids])
    save_samples(final, cache / "final")
    final_size = int(final.n)
    del final
    seeds = [int(s) for s in choices.get("seeds", stage["seeds"])]
    for f in families:
        for old in final_dir.glob(f"{f}_{TWIN}_seed*"):
            shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink()
    jobs = [job for f in families for job in twin_jobs(f, choices[f]["population"], seeds, cache / "final",
                                                       final_dir)]
    print(f"Glucose-only twins (D90): {len(jobs)} population models of {families} on {final_size} samples")
    runs = train_models(jobs, resources)
    shutil.rmtree(cache, ignore_errors=True)
    failed = [r for r in runs if r["status"] != "ok"]
    created = datetime.now().isoformat(timespec="seconds")
    if not failed:
        for f in families:
            choices[f][TWIN] = {"population": {**choices[f]["population"], "exog": False}, "created": created,
                                "release": release}
        path.write_text(yaml.safe_dump(to_plain(choices), sort_keys=False))
    table = runs_frame(runs)
    out.csv(table, "twin_models")
    out.csv(pop_table, "pool_patients")
    task_log = pd.DataFrame([{"stage": "training", "key": r["key"], "status": r["status"], "seconds": r["seconds"],
                              "error": r.get("error", "").strip().splitlines()[-1] if r.get("error") else "",
                              "traceback": r.get("error", "")} for r in runs])
    out.csv(task_log, "task_log")
    if not table.empty:
        out.figure(plot_twin_runs(table), "twin_models")
    lines = ["# 11 - Glucose-only twins of the population models (D76, D90)", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Families {families}; seeds {seeds}; {final_size} samples of the {len(fit_ids) + len(check_ids)} pool "
             "patients; settings, rounds and steps of the full models in `choices.yaml`, with `exog: false` (glucose "
             "only: no IOB, COB, insulin activity, carbohydrate absorption or time of day).", "", "## Warnings", ""]
    lines += [f"- {r['key']} failed: {r.get('error', '').strip().splitlines()[-1]}" for r in failed] or ["- none"]
    lines += ["", ("Saved in `choices.yaml` under `<family>: noexog`; the later scripts run the `*_ft_noexog` twins."
                   if not failed else "**choices.yaml NOT updated** (a training failed): run again."), ""]
    show = [c for c in ("key", "family", "seed", "status", "samples", "rounds", "steps", "device", "seconds")
            if c in table.columns]
    if show:
        lines += ["## Trainings", "", markdown_table(table[show]), ""]
    out.text("\n".join(lines), "summary.md")
    out.manifest(settings={**settings, "twin_families": families, "seeds": seeds},
                 notes="Glucose-only population models <family>_noexog_seed<k> next to the full ones (D90).")
    print(f"Done. Results in {out.dir}")


def plot_twin_runs(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(table) + 1.4))
    y = np.arange(len(table))[::-1]
    ax.barh(y, table["seconds"], color=[color(f + "_ft") for f in table["family"]])
    ax.set_yticks(y, table["key"], fontsize=8)
    ax.set_xlabel("training time (s)")
    ax.set_title("Glucose-only twins of the population models (D90)", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================================ helpers


def keep_other_families(path: Path, choices: dict, families: list[str]) -> dict:
    """Entries of ``choices.yaml`` for families not run now, kept if they come from the same pool split."""
    if not path.exists():
        return {}
    old = yaml.safe_load(path.read_text()) or {}
    same_pool = (old.get("pool") or {}).get("fit") == choices["pool"]["fit"] and \
        (old.get("pool") or {}).get("check") == choices["pool"]["check"]
    if not same_pool:
        print("choices.yaml comes from another pool split: the families not run now are dropped")
        return {}
    kept = {}
    for family in FAMILIES:
        if family in old and family not in families:
            entry = dict(old[family])
            entry.setdefault("created", old.get("created"))
            entry.setdefault("release", old.get("release"))
            kept[family] = entry
    return kept


def collect_days(results: list) -> pd.DataFrame:
    frames = [r.days for r in results if not r.error and r.days is not None and not r.days.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score_table(by_patient: pd.DataFrame) -> pd.DataFrame:
    """Per candidate: patients, mean skill by horizon and overall (mean over horizons, then patients)."""
    if by_patient.empty:
        return pd.DataFrame()
    b = by_patient[~by_patient["model"].isin(["persistence"])]
    wide = b.pivot_table(index=["model", "patient_id"], columns="horizon_min", values="skill")
    per_model = wide.groupby(level="model").mean()
    per_model.columns = [f"skill_{h}" for h in per_model.columns]
    per_model["skill_mean"] = wide.mean(axis=1).groupby(level="model").mean()
    per_model["patients"] = wide.groupby(level="model").size()
    rmse = b.pivot_table(index="model", columns="horizon_min", values="rmse", aggfunc="mean")
    rmse.columns = [f"rmse_{h}" for h in rmse.columns]
    ab = b.groupby("model")["clarke_AB_pct"].mean().rename("clarke_AB_pct_mean")
    return per_model.join(rmse).join(ab).reset_index().round(4)


def best(scores: pd.DataFrame, candidates: list[str]) -> str | None:
    s = scores[scores["model"].isin(candidates)] if not scores.empty else scores
    if s.empty:
        return None
    return str(s.sort_values("skill_mean", ascending=False)["model"].iloc[0])


def annotate_scores(scores: pd.DataFrame, grids: dict, keys: dict, chosen: dict, families: list[str]) -> pd.DataFrame:
    if scores.empty:
        return scores
    meta = {}
    for f in families:
        for start in grids[f]:
            for key, params in zip(keys[f][start], grids[f][start]):
                meta[key] = {"family": f, "start": start, "chosen": key == chosen[f].get(start),
                             "settings": ", ".join(f"{k}={v}" for k, v in params.items()
                                                   if k in varying(grids[f][start]))}
    meta["ar3"] = {"family": "ar3", "start": "reference", "chosen": False, "settings": "every scored pool patient"}
    meta["ar3_check"] = {"family": "ar3", "start": "reference", "chosen": False, "settings": "check patients"}
    extra = pd.DataFrame([{"model": k, **v} for k, v in meta.items()])
    table = extra.merge(scores, on="model", how="right")
    return table.sort_values(["family", "start", "model"]).reset_index(drop=True)


def varying(configs: list[dict]) -> set:
    keys = set().union(*configs) if configs else set()
    return {k for k in keys if len({str(c.get(k)) for c in configs}) > 1}


def runs_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {k: v for k, v in r.items() if k not in ("curve", "params")}
        for key in ("rounds", "validation_rmse"):
            if key in row:
                row[key] = str(row[key])
        rows.append(row)
    return pd.DataFrame(rows)


def rename_models(frame: pd.DataFrame, canonical: dict) -> pd.DataFrame:
    if frame.empty:
        return frame
    f = frame[frame["model"].isin(canonical)].copy()
    f["model"] = f["model"].map(canonical)
    return f.reset_index(drop=True)


def to_plain(value):
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


# ============================================================================ figures


def plot_scores(scores: pd.DataFrame, families: list[str]) -> plt.Figure:
    starts = [s for s in ("scratch", "population", "finetune") if s in set(scores["start"])]
    horizons = sorted(int(c.split("_")[1]) for c in scores.columns if c.startswith("skill_") and c != "skill_mean")
    fig, axes = plt.subplots(len(families), len(starts), figsize=(5.2 * len(starts), 3.8 * len(families)),
                             squeeze=False, sharey=True)
    for i, f in enumerate(families):
        for j, start in enumerate(starts):
            ax = axes[i, j]
            s = scores[(scores["family"] == f) & (scores["start"] == start)].reset_index(drop=True)
            if s.empty:
                ax.set_visible(False)
                continue
            width = 0.8 / max(1, len(s))
            for k, row in s.iterrows():
                x = np.arange(len(horizons)) + (k - (len(s) - 1) / 2) * width
                values = [row[f"skill_{h}"] for h in horizons]
                name = f + START_SUFFIX[start]
                ax.bar(x, values, width=width, color=color(name), alpha=1.0 if row["chosen"] else 0.35,
                       edgecolor="black" if row["chosen"] else "none",
                       label=(row["settings"] or row["model"]) + (" (chosen)" if row["chosen"] else ""))
            ar3 = scores[scores["model"] == ("ar3" if start == "scratch" else "ar3_check")]
            if not ar3.empty:
                for k, h in enumerate(horizons):
                    ax.hlines(ar3[f"skill_{h}"].iloc[0], k - 0.45, k + 0.45, color=TEXT_SECONDARY, ls="--", lw=1,
                              label="AR(3)" if k == 0 else None)
            ax.axhline(0, color=TEXT_SECONDARY, lw=0.6)
            ax.set_xticks(np.arange(len(horizons)), [f"{h} min" for h in horizons])
            ax.set_title(f"{label(f + START_SUFFIX[start])}", loc="left", fontsize=10)
            ax.legend(fontsize=6, loc="lower left")
            ax.grid(axis="x", visible=False)
        axes[i, 0].set_ylabel("mean skill over persistence")
    fig.suptitle("Candidates on the pool group (scratch: every pool patient; population and fine-tuned: check "
                 "patients)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_curves(curves: pd.DataFrame, runs: pd.DataFrame, chosen: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    picked = {v for c in chosen.values() for v in c.values() if v}
    for key, c in curves.groupby("key"):
        ax.plot(c["step"], c["val_mse"], lw=2 if key in picked else 1, alpha=1 if key in picked else 0.6,
                label=key + (" (chosen)" if key in picked else ""))
    ax.set_xlabel("training step")
    ax.set_ylabel("validation MSE of the path ((mg/dL / 50)^2)")
    ax.legend(fontsize=7)
    fig.suptitle("N-HiTS population candidates: error on the check patients' first 14 days", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(pool_table, runs, scores, check_summary, check_overall, finals, choices, task_log, fit_log, settings,
                  families, release, resources, final_dir) -> str:
    roles = pool_table["role"].value_counts().to_dict() if not pool_table.empty else {}
    lines = ["# 11 - Hyperparameters and population pretraining on the pool group", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Pool group: {roles.get('fit', 0)} fit and {roles.get('check', 0)} check patients "
             f"({roles.get('unused', 0)} without samples). Population samples every "
             f"{settings['population_stride_min']:.0f} min on every usable day; scoring as in stage 0 "
             f"({settings['warmup_days']} days from day 0 of the whole follow-up, tested on the later days every "
             f"{settings['origin_stride_min']:.0f} min; "
             f"horizons {settings['horizons_min']} min). Families: {', '.join(families)}.", ""]
    problems = []
    for _, row in task_log[task_log["status"] != "ok"].iterrows():
        problems.append(f"{row['stage']} {row['patient_id']} failed: {row['error']}")
    for _, row in task_log[task_log["warnings"].fillna("") != ""].head(10).iterrows():
        problems.append(f"{row['patient_id']} raised warnings: {row['warnings'][:300]}")
    if not fit_log.empty and "error" in fit_log:
        failed = fit_log[fit_log["error"].notna()]
        for (model, error), g in failed.groupby(["model", "error"]):
            problems.append(f"{model} not fitted for {len(g)} patient(s) ({', '.join(g['patient_id'].unique()[:5])}): {error}")
    if not runs.empty and "cuda_error" in runs:
        for _, row in runs[runs["cuda_error"].notna()].iterrows():
            problems.append(f"{row['key']}: the network fell back to the CPU ({row['cuda_error']})")
    if final_dir is None:
        problems.append("population models NOT saved (a final training failed or no family was chosen)")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    outcomes = fit_outcomes(fit_log)
    if not outcomes.empty:
        lines += ["## Fits (a run is complete when `other_failure` is 0)", "", markdown_table(outcomes), "", FIT_NOTE, ""]
    if final_dir is not None:
        lines += [f"Population models and `choices.yaml` saved in `{final_dir}`.", ""]
    lines += ["## Chosen settings", "", "```yaml", yaml.safe_dump(to_plain({f: choices.get(f) for f in families}),
                                                                     sort_keys=False).strip(), "```", ""]
    if not scores.empty:
        show = [c for c in scores.columns if c not in ("family",)]
        scores = scores.assign(chosen=scores["chosen"].map({True: "yes"}).fillna(""))
        lines += ["## Candidates (mean skill over persistence; scratch on every pool patient, population and "
                  "fine-tuned on the check patients)", "", markdown_table(scores[show]), ""]
    if not runs.empty:
        lines += ["## Candidate population trainings", "", markdown_table(runs.drop(columns=["status"],
                                                                                      errors="ignore")), ""]
    if not check_summary.empty:
        lines += ["## Chosen starting points on the check patients (95 % bootstrap intervals over patients)", "",
                  markdown_table(check_summary.drop(columns=["period"], errors="ignore")), ""]
    if not check_overall.empty:
        show = ["model", "horizon_min", "n", "rmse", "mae", "mard_pct", "bias", "clarke_A_pct", "clarke_B_pct",
                "clarke_AB_pct", "clarke_C_pct", "clarke_D_pct", "clarke_E_pct"]
        lines += ["## Pooled forecasts on the check patients (R6)", "", markdown_table(check_overall[show]), ""]
    if not finals.empty:
        lines += ["## Final population models (all pool patients)", "",
                  markdown_table(finals.drop(columns=["status"], errors="ignore")), ""]
    lines += ["## Checks to look at", "",
              "- `tuning_scores.png`: whether pretraining helps (population and fine-tuned above scratch and AR(3)).",
              "- `learning_curves.png`: N-HiTS candidates should improve for a while and then flatten; a best step at "
              "the first evaluation means the network learns nothing beyond persistence.",
              "- `check_skill.png` and `clarke_grids.png`: the chosen starting points on the check patients.",
              "- Script 12 repeats the comparison on the screen group with three seeds and the noise floor.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
