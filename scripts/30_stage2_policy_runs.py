"""30 - Stage 2: refit policies (how often, on which days), successive halving and drift triggers.

Runs on the screen group of the core dataset, from anchor day 14 of the core
period (D80), with the models that passed screening 1 (``shortlist.yaml`` of
script 21) plus ``stage2.extra_models`` (TSMixerx, D79). Each model is tested
on the same 84 days under refit policies (HANDOFF 7.4; ``sim.policies``):

- window: growing, rolling 7 / 14 / 28 days, or growing with exponential
  sample weights (half-life 14 days);
- frequency: every 1, 3, 7 or 14 days, or never (21 policies);
- drift triggers (D81): ADWIN and Page-Hinkley on the model's errors relative
  to persistence, each against the fixed policy with the same number of
  refits spread evenly over the test days;
- ARX also with its own mechanism, the update at every sample (``online``,
  D82), which always goes on to round 2.

Successive halving (D80): every policy on ``round1_patients`` patients; the best
third of each model's policies, plus its references (never, the growing
window at the stage-1 frequency and, for ARX, online), then go on to every
patient of the group, with the triggers. Every refit starts again from the
model's starting point (D65); a refit at the start of day *t* learns only from
targets before *t*. A refit that fails keeps the model in use; a patient and
anchor without a model at the anchor (the fit of never failed) is left out
for that model, so every policy of a model is scored on the same origins.
Results are cached per model, policy and patient, so a later run computes only
what is new or changed.

Later anchors (D83, ``stage2.later_anchors``): from anchor days 42 and 70 (and
the main anchor), a few policies (never and the growing window every 1, 7 and
14 days; ARX also online) on every patient with that anchor. Here the first
fit and the growing window start at day 0, so the model starts with 42 or 70
days of the patient's history: if the gain of refitting comes from adding
data, it should shrink at later anchors.

Glucose-only twins (D90): every model that reads insulin and carbohydrates has
a twin that reads the glucose only (D76), and the twin runs exactly the
policies of its full model: those its full model keeps after round 1 (the
selection is made on the full models), the triggers, the later anchors and,
for ARX, the update at every sample. Script 31 compares their gains with those
of the full models, patient by patient.

Outputs per policy: skill against persistence (patient means, intervals),
RMSE and Clarke zones (rule R6), refits per week and computing seconds per day;
paired differences against never (the price of not refitting, in skill points
and mg/dL); the front of skill against cost per model; and the triggers
against their matched fixed policies. Script 31 adds the analysis (optimal
interval per horizon, mixed models). One job per patient, anchor and model on
all cores but two, networks on the GPUs (rule R5, D73). The confirm group is
never touched.

Derived data:
    <derived_data>/runs/30_stage2_policy_runs/day_metrics.parquet   per patient, model, policy, horizon and day
    <derived_data>/runs/30_stage2_policy_runs/fit_log.parquet       every fit (cost per fit, for script 31)
    <derived_data>/runs/30_stage2_policy_runs/cache/<model>/<policy>/<patient-anchor>.pkl

Outputs (results/30_stage2_policy_runs/):
    halving.csv              round 1: every policy of every model on the round-1 patients, rank, kept and why
    policy_summary.csv       every policy that ran on all patients: skill and RMSE by horizon with intervals
    policy_overview.csv      per model and policy: mean skill over horizons, refits per week, seconds per day,
                             front of skill against cost
    vs_never.csv             paired differences against never: skill points, RMSE saved (mg/dL), verdict (DMD, D75)
    triggers.csv             each trigger against its matched fixed policy, with its refits per week
                             (against never: vs_never.csv)
    later_anchors.csv        D83: per model, anchor, horizon and policy, the gain over never
    metrics_overall.csv      per model, policy and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6)
    fit_log.csv / fit_outcomes.csv / task_log.csv / jobs.csv
    policy_grid.png          round 1: skill of every window and frequency per model
    vs_never.png             what each policy gains over never
    pareto.png               skill against computing time per day
    triggers.png             triggers against their matched fixed policies
    clarke_grids.png         best policy and never of every model (R6)
    later_anchors.png        D83: gain of refitting every day over never by anchor
    summary.md
    run_manifest.json

Usage:
    python scripts/30_stage2_policy_runs.py
    python scripts/30_stage2_policy_runs.py --round1-only                 # stop after round 1
    python scripts/30_stage2_policy_runs.py --skip-later-anchors          # without the runs of D83
    python scripts/30_stage2_policy_runs.py --skip-twins                  # without the glucose-only twins
    python scripts/30_stage2_policy_runs.py --models ar3 arx --patients 4  # a quick partial run
"""

from __future__ import annotations

import argparse
import sys
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
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator, plot_clarke  # noqa: E402
from t1dretrain.eval.stage0_report import (FIT_NOTE, PERIOD_LABELS, SUMS, DMDTable, color,  # noqa: E402
                                           finish, fit_outcomes, label, read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.cache import cache_valid, read_cache, retryable, write_cache  # noqa: E402
from t1dretrain.sim.policies import online_policies, parse_policy, policy_grid, trigger_policies  # noqa: E402
from t1dretrain.sim.stage0 import (FAMILIES, STATISTICAL_MODELS, model_dependencies, per_model,  # noqa: E402
                                   split_name, twin_available, twin_families, twin_of)

SCRIPT_ID = "30_stage2_policy_runs"
FIT_KEYS = pr.FIT_KEYS  # the columns that identify one fit in fit_log.csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of the stage-2 models")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients (quick runs)")
    parser.add_argument("--round1-only", action="store_true", help="stop after round 1 of successive halving")
    parser.add_argument("--skip-later-anchors", action="store_true",
                        help="leave out the runs from later anchors (stage2.later_anchors, D83)")
    parser.add_argument("--skip-twins", action="store_true",
                        help="leave out the glucose-only twins (D90; quick runs only)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ jobs and cache


def run_job(job: dict) -> dict:
    """The policies of one model, patient and anchor that have no valid cached result; each is cached."""
    import time

    from t1dretrain.sim.policies import evaluate_policies

    started = time.time()
    todo = [p for p in job["policies"] if not cache_valid(job["cache"][p])]
    for p in list(todo):  # a trigger and its matched fixed policy are computed together
        partner = p[len("even_"):] if p.startswith("even_") else f"even_{p}" if parse_policy(p).trigger else None
        if partner and partner in job["policies"] and partner not in todo:
            todo.append(partner)
    reference = not cache_valid(job["reference"])
    if not todo and not reference:
        return {"job": job["name"], "status": "cached", "policies": [], "seconds": 0.0, "warnings": "", "error": ""}
    with WarningRecorder() as recorder:
        result = evaluate_policies({**job, "policies": todo, "with_reference": reference})
    status = {"job": job["name"], "status": "error" if result.error else "ok", "policies": todo,
              "seconds": round(time.time() - started, 1), "warnings": recorder.text(), "error": result.error}
    if result.error:
        return status
    not_cached = []
    for policy in todo:
        days = result.days[result.days["policy"] == policy] if not result.days.empty else result.days
        clarke = result.clarke.subset(lambda key, policy=policy: key["policy"] == policy)
        info = [row for row in result.info if row.get("policy") == policy]
        if any(retryable(row) for row in info):  # an unexpected failure (e.g. CUDA): computed again next run
            not_cached.append(policy)
            continue
        refits = {k: v for k, v in result.refits.items() if k.split("|")[0] == policy}
        write_cache(job["cache"][policy], {"days": days, "clarke": clarke, "info": info, "refits": refits})
    if reference:
        days = result.days[result.days["policy"] == "reference"] if not result.days.empty else result.days
        write_cache(job["reference"], {"days": days, "info": [], "refits": {},
                                       "clarke": result.clarke.subset(lambda key: key["policy"] == "reference")})
    status["not_cached"] = not_cached
    status["retry_errors"] = sorted({row["error"] for row in result.info if retryable(row)})
    status["cuda_fallbacks"] = sum(1 for row in result.info if row.get("cuda_error"))
    return status


def cache_key(model: str, policy: str, settings: dict, choices: dict) -> str:
    """What a cached result depends on: model, policy, its starting point, the stage-2 settings, the simulator."""
    import hashlib
    import json

    from t1dretrain.sim.policies import CACHE_VERSION

    relevant = {k: settings[k] for k in ("horizons_min", "lookback_hours", "origin_stride_min", "train_stride_min",
                                         "validation_days", "min_window_share", "recent_complete_min", "window_days",
                                         "test_days", "seeds")}
    policy_kind = parse_policy(policy).kind if policy != "reference" else "reference"
    relevant["triggers"] = settings["triggers"] if policy_kind in ("trigger", "matched") else None
    relevant.update(model_dependencies(model, settings, choices))
    text = json.dumps({"model": model, "policy": policy, "cache_version": CACHE_VERSION, **relevant}, sort_keys=True,
                      default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def device_for(model: str, resources) -> str:
    return "cuda" if resources.device == "cuda" and resources.n_gpus else "cpu"  # one GPU per worker process (D73)


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
    stage, stage1, stage2 = config["stage0"], config["stage1"], config["stage2"]
    periods = list(stage2["periods"])
    models, shortlist_note = stage2_models(derived, stage2, periods[0], args.models)
    population_dir = derived / "models" / core / "population"
    choices = yaml.safe_load((population_dir / "choices.yaml").read_text()) \
        if (population_dir / "choices.yaml").exists() else {}
    if any(split_name(m)[0] in FAMILIES for m in models) and not choices:
        raise SystemExit(f"{population_dir / 'choices.yaml'} not found: run scripts/11_pretrain_population.py first")
    families = None if args.skip_twins else twin_families(config)
    twins = {m: t for m in models
             if (t := twin_of(m, families)) and twin_available(t, population_dir, stage2["seeds"])}
    left_out = [twin_of(m, families) for m in models if twin_of(m, families) and m not in twins]
    if left_out:
        print(f"Twins left out (no glucose-only population model of script 11): {left_out}")
    everyone = [n for m in models for n in (m, twins.get(m)) if n]
    triggers = stage2["triggers"]
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]),
                "train_stride_min": float(stage["train_stride_min"]), "validation_days": int(stage["validation_days"]),
                "min_window_share": float(stage["min_window_share"]),
                "recent_complete_min": float(stage["recent_complete_min"]),
                "window_days": int(config["anchors"]["train_window_days"]),
                "test_days": int(config["anchors"]["max_age_days"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]),
                "seeds": [int(s) for s in stage2["seeds"]], "population_dir": str(population_dir),
                "finetune": {f: (choices.get(f) or {}).get("finetune") for f in FAMILIES},
                "triggers": {t: dict(triggers[t]) for t in triggers["detectors"]}}
    for f in FAMILIES:
        settings[f] = (choices.get(f) or {}).get("scratch") or stage.get(f)
    for m in STATISTICAL_MODELS:
        settings[m] = stage.get(m) or {}
    grid = [p.name for p in policy_grid(list(stage2["windows"]), [int(k) for k in stage2["every"]])]
    trigger_names = [p.name for p in trigger_policies(list(triggers["detectors"]), triggers["window"])]
    references = {m: ["never", f"growing_every{int(per_model(stage1['refresh_days'], m, 1))}"] + online_policies(m)
                  for m in models}  # ARX: also its update at every sample (D82)

    patients = group_patients(core, stage2.get("group", "screen"))
    if args.patients:
        patients = patients[: args.patients]
    all_anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    all_anchors = all_anchors[(all_anchors["cohort"] == "core") & all_anchors["patient_id"].isin(patients) &
                              all_anchors["period"].isin(periods) & all_anchors["available"].astype(bool)].copy()
    all_anchors["anchor"] = pd.to_datetime(all_anchors["day0"]) + pd.to_timedelta(all_anchors["anchor_day"], unit="D")
    all_anchors = all_anchors.sort_values(["patient_id", "period", "anchor_day"]).reset_index(drop=True)
    main_days = [int(a) for a in stage2["anchor_days"]]
    anchors = all_anchors[all_anchors["anchor_day"].isin(main_days)].reset_index(drop=True)
    available = sorted(anchors["patient_id"].unique())
    rng = np.random.default_rng(int(config["random_seed"]))
    round1 = sorted(str(p) for p in rng.choice(available, size=min(int(stage2["round1_patients"]), len(available)),
                                                 replace=False)) if available else []
    runs = derived / "runs" / SCRIPT_ID

    def jobs_for(policies_of: dict, patient_ids: list[str], table: pd.DataFrame | None = None,
                 history: bool = False, model_list: list[str] | None = None) -> list[dict]:
        """One job per patient, anchor and model. ``history``: the first fit and the growing window start at day 0
        (the later anchors of D83), not ``window_days`` before the anchor."""
        jobs = []
        table = anchors if table is None else table
        for r in table[table["patient_id"].isin(patient_ids)].itertuples():
            own = {**settings, "window_days": int(r.anchor_day)} if history else settings
            name = f"{r.patient_id}__{r.period}__{int(r.anchor_day)}"
            if own["window_days"] != settings["window_days"]:  # its own cache files
                name += f"__h{own['window_days']}"
            reference = {"path": str(runs / "cache" / "persistence" / f"{name}.pkl"),
                         "key": cache_key("persistence", "reference", own, choices),
                         "recompute": run_start if args.recompute else None}
            for m in model_list or models:
                cache = {p: {"path": str(runs / "cache" / m / p / f"{name}.pkl"),
                             "key": cache_key(m, p, own, choices),
                             "recompute": run_start if args.recompute else None}
                         for p in policies_of[m]}
                jobs.append({"name": f"{name}__{m}", "dataset": core, "patient_id": r.patient_id, "period": r.period,
                             "anchor_day": int(r.anchor_day), "anchor": r.anchor, "derived_root": derived,
                             "settings": own, "model": m, "policies": policies_of[m], "cache": cache,
                             "reference": reference, "stream": len(jobs), "device": device_for(m, resources)})
        # the networks first: they take longest, so the GPUs start working at once
        return sorted(jobs, key=lambda j: split_name(j["model"])[0] not in FAMILIES or j["model"].startswith("lightgbm"))

    print(f"Models {models} ({shortlist_note}); glucose-only twins {list(twins.values())}; {len(available)} patients "
          f"with anchor day(s) {stage2['anchor_days']}; round 1 on {len(round1)}: {round1}")
    jobs1 = jobs_for({m: grid + online_policies(m) for m in models}, round1)
    statuses = parallel_map(run_job, jobs1, resources.n_workers, label="round-1 jobs")
    days1, _, _, _, _ = collect(jobs1)
    skill1 = pr.patient_skill(days1) if not days1.empty else pd.DataFrame()
    keep = int(np.ceil(len(grid) * float(stage2["keep_share"]) - 1e-9))
    halving = pr.halving(pr.ranking(skill1), keep, references, outside=["online"]) if not skill1.empty \
        else pd.DataFrame()
    if args.round1_only or halving.empty:
        jobs2, kept = [], {m: [] for m in models}
    else:
        kept = {m: list(halving.loc[(halving["model"] == m) & halving["kept"], "policy"]) for m in models}
        kept.update({twin: kept[m] for m, twin in twins.items()})  # a twin runs what its full model keeps (D90)
        jobs2 = jobs_for({m: kept[m] + trigger_names for m in everyone}, available, model_list=everyone)
        statuses += parallel_map(run_job, jobs2, resources.n_workers, label="round-2 jobs")
    later = stage2.get("later_anchors") or {}
    later_days = sorted(set(main_days) | {int(a) for a in later.get("anchor_days", [])})
    jobs3 = []
    if later and not args.round1_only and not args.skip_later_anchors:  # D83: more history at the start
        table = all_anchors[all_anchors["anchor_day"].isin(later_days)]
        jobs3 = jobs_for({m: list(later["policies"]) + online_policies(m) for m in everyone}, patients, table,
                         history=True, model_list=everyone)
        print(f"Later anchors {later_days}: {table['patient_id'].nunique()} patients, {len(jobs3)} jobs")
        statuses += parallel_map(run_job, jobs3, resources.n_workers, label="later-anchor jobs")

    final_jobs = jobs2 if jobs2 else jobs1
    days, infos, refits, clarke, missing = collect(final_jobs)
    days3, infos3, _, _, missing3 = collect(jobs3) if jobs3 else (pd.DataFrame(), [], None, None, [])
    missing += missing3
    runs.mkdir(parents=True, exist_ok=True)
    keys = ["patient_id", "period", "anchor_day", "model", "policy", "seed", "horizon_min", "day"]
    parts = [d for d in (days1, days, days3) if not d.empty]
    all_days = pd.concat(parts, ignore_index=True).drop_duplicates(keys) if parts else pd.DataFrame()
    all_days.to_parquet(runs / "day_metrics.parquet", index=False)
    task_log = pd.DataFrame([{"job": r["job"], "status": r["status"], "policies_run": " ".join(r["policies"]),
                              "not_cached": " ".join(r.get("not_cached", [])),
                              "retry_errors": " | ".join(r.get("retry_errors", [])),
                              "cuda_fallbacks": r.get("cuda_fallbacks", 0), "seconds": r["seconds"],
                              "warnings": r["warnings"], "error": r["error"].strip().splitlines()[-1] if r["error"] else "",
                              "traceback": r["error"]} for r in statuses])
    main_pairs = {(row["model"], row["policy"]) for row in infos}  # anchor 14 of D83: only what round 2 lacks
    fit_log = pd.DataFrame(infos + [row for row in infos3 if int(row.get("anchor_day", -1)) not in main_days or
                                    (row["model"], row["policy"]) not in main_pairs])
    if not fit_log.empty:
        fit_log["history_days"] = [int(settings["window_days"]) if int(a) in main_days else int(a)
                                   for a in fit_log["anchor_day"]]
        write_fit_log(fit_log, runs / "fit_log.parquet")
    main_log = fit_log[fit_log["anchor_day"].isin(main_days)] if not fit_log.empty else fit_log
    used = main_log[main_log["excluded"].isna()] if "excluded" in main_log else main_log  # fits the policies used
    fits = fit_log.drop_duplicates([c for c in FIT_KEYS if c in fit_log]) if not fit_log.empty else fit_log
    replicates, seed = int(stage["bootstrap_replicates"]), int(config["random_seed"])
    dmd = DMDTable(read_noise_floor(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in everyone if split_name(m)[0] in FAMILIES])
    test_days = int(settings["test_days"])
    skill = pr.patient_skill(days) if not days.empty else pd.DataFrame()
    summary = pr.summary_table(skill, replicates, seed) if not skill.empty else pd.DataFrame()
    overview = policy_overview(skill, used, refits, test_days)
    vs_never = pr.paired(skill, lambda p: "never", dmd, replicates, seed) if not skill.empty else pd.DataFrame()
    trigger_table = trigger_comparison(skill, overview, trigger_names, dmd, replicates, seed)
    overall = finish(days.groupby(["period", "model", "policy", "horizon_min"], as_index=False)[SUMS].sum()).round(4) \
        if not days.empty else pd.DataFrame()
    histogram = clarke.histogram()
    later_table = later_anchor_table(days3, dmd, replicates, seed) if not days3.empty else pd.DataFrame()
    twin_table = vs_never[vs_never["model"].isin(list(twins.values()))] if not vs_never.empty else vs_never
    panels = clarke_panels(overview, models, periods[0]) if not overview.empty else []

    out.csv(halving.round(4), "halving")
    out.csv(summary.round(4), "policy_summary")
    out.csv(overview.round(4), "policy_overview")
    out.csv(vs_never.round(4), "vs_never")
    out.csv(trigger_table.round(4), "triggers")
    out.csv(overall, "metrics_overall")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(histogram, "clarke_histogram")
    sample = clarke.sample()
    if panels and not sample.empty:  # the panels of clarke_grids.png and persistence (the histogram has them all)
        drawn = sample[["model", "policy"]].apply(tuple, axis=1).isin(set(panels) | {("persistence", "reference")})
        sample = sample[drawn]
    out.csv(sample, "clarke_points_sample")
    out.csv(later_table.round(4), "later_anchors")
    out.csv(fit_log, "fit_log")
    out.csv(fit_outcomes(fits), "fit_outcomes")
    out.csv(task_log, "task_log")
    jobs_table = all_anchors[all_anchors["anchor_day"].isin(later_days if jobs3 else main_days)]
    out.csv(jobs_table.drop(columns=["dataset"], errors="ignore").assign(
        round1=jobs_table["patient_id"].isin(round1) & jobs_table["anchor_day"].isin(main_days)), "jobs")
    period, windows = periods[0], list(stage2["windows"])
    if not halving.empty:
        out.figure(plot_grid(halving, models, windows, [int(k) for k in stage2["every"]], period), "policy_grid")
    if not vs_never.empty:
        out.figure(plot_vs_never(vs_never, models, windows, period), "vs_never")
    if not overview.empty:
        out.figure(plot_pareto(overview, models, period), "pareto")
    if not trigger_table.empty:
        out.figure(plot_triggers(trigger_table, models, period), "triggers")
    if panels and not histogram.empty:
        out.figure(plot_grids(histogram, panels, settings["horizons_min"], period), "clarke_grids")
    if not later_table.empty:
        out.figure(plot_later_anchors(later_table, models, period), "later_anchors")
    out.text(write_summary(halving, overview, vs_never, trigger_table, later_table, twin_table, fit_log, task_log,
                           missing, settings, models, shortlist_note, round1, len(available), keep, release, resources,
                           period, main_days), "summary.md")
    out.manifest(settings={**{k: v for k, v in settings.items() if k != "population_dir"}, "periods": periods,
                           "models": models, "grid": grid, "triggers_policies": trigger_names, "round1": round1,
                           "kept": kept, "patients": len(available), "population_models": str(population_dir),
                           "later_anchors": {"anchor_days": later_days, "policies": later.get("policies"),
                                             "jobs": len(jobs3)} if jobs3 else None,
                           "twins": twins,
                           "choices_release": choices.get("release")},
                 notes="One job per patient, anchor and model; day metrics in <derived>/runs/30_stage2_policy_runs.")
    print(f"Done. Results in {out.dir}")


def stage2_models(derived: Path, stage2: dict, period: str, chosen: list[str] | None) -> tuple[list[str], str]:
    """The shortlist of script 21 for the period, plus the extra models (D79), in that order."""
    path = derived / "runs" / "21_stage1_ageing_analysis" / "shortlist.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/21_stage1_ageing_analysis.py first")
    shortlist = (yaml.safe_load(path.read_text()) or {}).get("periods", {}).get(period, {}).get("models", [])
    extra = [m for m in stage2.get("extra_models", []) if m not in shortlist]
    models = [m for m in list(shortlist) + extra if not chosen or m in chosen]
    note = f"shortlist of script 21: {', '.join(shortlist)}; added: {', '.join(extra) or 'none'}"
    return models, note


def collect(jobs: list[dict]):
    """Every cached result of the jobs: day metrics, fit log, refits, Clarke data and what is missing."""
    frames, infos, refits, missing = [], [], [], []
    clarke = ClarkeAccumulator(["model", "policy", "horizon_min", "period"], sample_size=20000)
    seen = set()
    for job in jobs:
        entries = [(p, job["cache"][p]) for p in job["policies"]]
        if job["reference"]["path"] not in seen:
            entries.append(("reference", job["reference"]))
            seen.add(job["reference"]["path"])
        for policy, entry in entries:
            cached = read_cache(entry)
            if cached is None:
                missing.append(f"{job['name']}:{policy}")
                continue
            frames.append(cached["days"])
            infos += cached["info"]
            clarke.merge(cached["clarke"])
            for key, fit_days in cached.get("refits", {}).items():
                if not fit_days:  # left out: no model at the anchor
                    continue
                refits.append({"period": job["period"], "model": job["model"], "policy": key.split("|")[0],
                               "patient_id": job["patient_id"], "anchor_day": job["anchor_day"],
                               "fits": len(fit_days)})
    days = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    return days, infos, pd.DataFrame(refits), clarke, missing


def policy_overview(skill: pd.DataFrame, fit_log: pd.DataFrame, refits: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Per model and policy: mean skill over horizons, refits per week, seconds per day, and the front."""
    if skill.empty:
        return pd.DataFrame()
    table = pr.ranking(skill).drop(columns=["rank"])
    for extra in (pr.cost_table(fit_log, test_days), pr.refits_table(refits, test_days)):
        table = table.merge(extra, on=["period", "model", "policy"], how="left")
    by_horizon = skill.groupby(["period", "model", "policy", "horizon_min"])["skill"].mean().unstack("horizon_min")
    by_horizon.columns = [f"skill_{int(h)}" for h in by_horizon.columns]
    table = table.merge(by_horizon.reset_index(), on=["period", "model", "policy"], how="left")
    table["kind"] = table["policy"].map(lambda p: parse_policy(p).kind)
    table.loc[table["kind"] == "online", "refits_per_week"] = np.nan  # updated at every sample
    table["front"] = False
    for _, g in table[table["kind"].isin(["never", "fixed"])].groupby(["period", "model"]):
        table.loc[g.index, "front"] = pr.pareto(g.fillna({"seconds_per_day": 0.0}))
    return table.sort_values(["period", "model", "skill_mean"], ascending=[True, True, False]).reset_index(drop=True)


def trigger_comparison(skill: pd.DataFrame, overview: pd.DataFrame, trigger_names: list[str], dmd, replicates: int,
                       seed: int) -> pd.DataFrame:
    """Each trigger against its matched fixed policy, with the trigger's refits per week ("never fired": the two
    policies are the same policy, never)."""
    if skill.empty:
        return pd.DataFrame()
    table = pr.paired(skill, lambda p: f"even_{p}" if p in trigger_names else None, dmd, replicates, seed)
    if table.empty:
        return table
    rates = overview.set_index(["period", "model", "policy"])["refits_per_week"] if not overview.empty else {}
    table.insert(table.columns.get_loc("patients"), "refits_per_week",
                 [rates.get((r.period, r.model, r.policy), np.nan) for r in table.itertuples()])
    table.loc[table["refits_per_week"].fillna(0) == 0, "verdict"] = "never fired"
    return table


def later_anchor_table(days: pd.DataFrame, dmd, replicates: int, seed: int) -> pd.DataFrame:
    """D83: per model, anchor, horizon and policy, the paired gain over never (each anchor on its own patients); the
    first fit uses every day since day 0, so the anchor day is also the days of history at the start."""
    skill = pr.patient_skill(days, by=("anchor_day",))
    table = pr.paired(skill, lambda p: "never", dmd, replicates, seed, by=("anchor_day",))
    if table.empty:
        return table
    level = skill[skill["policy"] == "never"].groupby(["period", "model", "horizon_min", "anchor_day"])["skill"]
    table["never_skill"] = [level.mean().get((r.period, r.model, r.horizon_min, r.anchor_day), np.nan)
                            for r in table.itertuples()]
    return table.sort_values(["period", "model", "policy", "horizon_min", "anchor_day"]).reset_index(drop=True)


def clarke_panels(overview: pd.DataFrame, models: list[str], period: str) -> list[tuple[str, str]]:
    """The panels of clarke_grids.png: never and the best fixed policy of every model."""
    best = overview[(overview["period"] == period) & (overview["kind"] == "fixed")] \
        .sort_values("skill_mean", ascending=False).groupby("model")["policy"].first()
    return [(m, p) for m in models for p in ("never", best.get(m)) if p]


def write_fit_log(fit_log: pd.DataFrame, path: Path) -> None:
    """The fit log for script 31 (cost per fit): the columns it needs, in parquet."""
    columns = [c for c in FIT_KEYS + ["policy", "history_days", "test_origins", "train_days", "train_samples",
                                      "fit_seconds", "compute_seconds", "updates", "error", "excluded", "served_by"]
               if c in fit_log]
    table = fit_log[columns].copy()
    for c in ("fit_day", "train_start"):
        table[c] = pd.to_datetime(table[c].astype(str), errors="coerce")
    for c in ("weighting", "error", "excluded", "served_by", "policy", "model", "patient_id", "period"):
        if c in table:
            table[c] = table[c].astype("string")
    table.to_parquet(path, index=False)


def policy_order(policies, windows: list[str]) -> list[str]:
    """never, the fixed policies by window and frequency, online, the triggers, their matched policies."""
    def key(name: str):
        p = parse_policy(name)
        w = windows.index(p.window.key) if p.window.key in windows else len(windows)
        return ({"never": 0, "fixed": 1, "online": 2, "trigger": 3, "matched": 4}[p.kind], w, p.every or 0, name)
    return sorted(policies, key=key)


def seconds_label(x: float, _pos=None) -> str:
    if not np.isfinite(x) or x <= 0:
        return "0"
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if x >= scale * 0.999:
            return f"{x / scale:g} {unit}"
    return f"{x:.0e} s"


# ============================================================================ figures


def plot_grid(halving: pd.DataFrame, models: list[str], windows: list[str], every: list[int], period: str):
    h = halving[halving["period"] == period]
    fig, axes = model_axes(len(models), 4.2, 3.6)
    for ax, model in zip(axes, models):
        m = h[h["model"] == model].set_index("policy")
        values = np.array([[m["skill_mean"].get(f"{w}_every{k}", np.nan) for k in every] for w in windows])
        image = ax.imshow(100 * values, cmap="viridis", aspect="auto")
        for i, w in enumerate(windows):
            for j, k in enumerate(every):
                name = f"{w}_every{k}"
                if name in m.index:
                    ax.text(j, i, f"{100 * m.loc[name, 'skill_mean']:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if values[i, j] < np.nanmean(values) else "black",
                            fontweight="bold" if bool(m.loc[name, "kept"]) else "normal")
        never = 100 * m["skill_mean"].get("never", np.nan)
        online = f"; online: {100 * m.loc['online', 'skill_mean']:.1f} %" if "online" in m.index else ""
        ax.set_title(f"{label(model)}\nnever: {never:.1f} %{online}", loc="left", fontsize=9)
        ax.set_xticks(range(len(every)), [f"{k} d" for k in every], fontsize=8)
        ax.set_yticks(range(len(windows)), windows, fontsize=8)
        ax.set_xlabel("refit every")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Round 1: mean skill over horizons (%) of every policy, {PERIOD_LABELS.get(period, period)} "
                 "(bold: goes on to round 2)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_vs_never(vs_never: pd.DataFrame, models: list[str], windows: list[str], period: str):
    v = vs_never[(vs_never["period"] == period) & ~vs_never["policy"].str.startswith("even_")]
    horizons = sorted(v["horizon_min"].unique())
    rows = [(m, p) for m in models for p in policy_order(v.loc[v["model"] == m, "policy"].unique(), windows)]
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.8 * len(horizons), 0.3 * len(rows) + 1.8), sharey=True,
                             squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, horizon in zip(axes[0], horizons):
        h = v[v["horizon_min"] == horizon].set_index(["model", "policy"])
        for yy, key in zip(y, rows):
            if key not in h.index:
                continue
            r = h.loc[key]
            if np.isfinite(r["dmd"]):
                ax.fill_between([-100 * r["dmd"], 100 * r["dmd"]], yy - 0.35, yy + 0.35, color=TEXT_SECONDARY,
                                alpha=0.12, lw=0, zorder=0)
            ax.plot([100 * r["low"], 100 * r["high"]], [yy, yy], color=color(key[0]), lw=2)
            ax.scatter(100 * r["skill_difference"], yy, color=color(key[0]), s=22, zorder=3,
                       marker="o" if r["verdict"] in ("better", "worse") else "s")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(y, [f"{label(m)}, {p}" for m, p in rows], fontsize=7)
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.set_xlabel("skill points over never (95 % CI)")
    fig.suptitle(f"What each policy gains over never refitting, {PERIOD_LABELS.get(period, period)} (band: the "
                 "model's DMD; circle: beyond it)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_pareto(overview: pd.DataFrame, models: list[str], period: str):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    o = overview[(overview["period"] == period) & overview["kind"].isin(["never", "fixed", "trigger", "online"])]
    fig, axes = model_axes(len(models), 4.6, 4.0)
    for ax, model in zip(axes, models):
        m = o[(o["model"] == model) & (o["seconds_per_day"] > 0)].copy()
        fixed, trig = m[m["kind"].isin(["never", "fixed"])], m[m["kind"] == "trigger"]
        online = m[m["kind"] == "online"]
        ax.scatter(fixed["seconds_per_day"], 100 * fixed["skill_mean"], color=color(model), s=18, alpha=0.6)
        for part, marker, name in ((trig, "^", "drift trigger"), (online, "D", "update at every sample")):
            if not part.empty:
                ax.scatter(part["seconds_per_day"], 100 * part["skill_mean"], facecolors="none",
                           edgecolors=color(model), marker=marker, s=30, label=name)
        front = fixed[fixed["front"]].sort_values("seconds_per_day")
        ax.plot(front["seconds_per_day"], 100 * front["skill_mean"], color=color(model), lw=1.5)
        named = pd.concat([front, trig, online])  # policies at the same point share one label
        named = named.assign(x=named["seconds_per_day"].round(9), y=(100 * named["skill_mean"]).round(4))
        for (x, y), g in named.groupby(["x", "y"]):
            ax.annotate("\n".join(g["policy"]), (x, y), fontsize=6, textcoords="offset points", xytext=(3, 3),
                        va="bottom")
        ax.set_xscale("log")
        span = m["seconds_per_day"].max() / max(m["seconds_per_day"].min(), 1e-12) if len(m) else 1.0
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if span < 20 else (1.0,)))
        ax.xaxis.set_major_formatter(FuncFormatter(seconds_label))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.15, y=0.12)
        ax.set_title(label(model), loc="left", fontsize=9)
        ax.set_xlabel("computing time per test day")
        if not (trig.empty and online.empty):
            ax.legend(fontsize=7, loc="lower right")
    for ax in axes[::3]:
        ax.set_ylabel("mean skill over horizons (%)")
    fig.suptitle(f"Skill against cost: the front of each model's fixed policies, {PERIOD_LABELS.get(period, period)}",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_triggers(table: pd.DataFrame, models: list[str], period: str):
    t = table[table["period"] == period]
    horizons = sorted(t["horizon_min"].unique())
    rows = [(m, p) for m in models for p in sorted(t.loc[t["model"] == m, "policy"].unique())]
    rates = t.groupby(["model", "policy"])["refits_per_week"].first()
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.8 * len(horizons), 0.35 * len(rows) + 1.8), sharey=True,
                             squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, horizon in zip(axes[0], horizons):
        h = t[t["horizon_min"] == horizon].set_index(["model", "policy"])
        for yy, key in zip(y, rows):
            if key not in h.index:
                continue
            r = h.loc[key]
            if np.isfinite(r["dmd"]):
                ax.fill_between([-100 * r["dmd"], 100 * r["dmd"]], yy - 0.35, yy + 0.35, color=TEXT_SECONDARY,
                                alpha=0.12, lw=0, zorder=0)
            fired = r["verdict"] != "never fired"
            ax.plot([100 * r["low"], 100 * r["high"]], [yy, yy], color=color(key[0]), lw=2)
            ax.scatter(100 * r["skill_difference"], yy, s=22, zorder=3, edgecolors=color(key[0]),
                       facecolors=color(key[0]) if fired else "none")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(y, [f"{label(m)}, {p} ({rates.get((m, p), np.nan):.2f} refits/week)" for m, p in rows],
                      fontsize=7)
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.set_xlabel("skill points: trigger - same refits evenly spread")
    fig.suptitle(f"Drift triggers against fixed refits with the same count, {PERIOD_LABELS.get(period, period)} "
                 "(band: the model's DMD; hollow: never fired)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_later_anchors(table: pd.DataFrame, models: list[str], period: str, policy: str = "growing_every1"):
    """D83: the gain of refitting every day over never, by anchor (= days of history at the start)."""
    t = table[(table["period"] == period) & (table["policy"] == policy)]
    horizons = sorted(t["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.6 * len(horizons), 4.0), sharey=True, squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        h = t[t["horizon_min"] == horizon]
        for k, model in enumerate(models):
            m = h[h["model"] == model].sort_values("anchor_day")
            if m.empty:
                continue
            x = m["anchor_day"] + (k - len(models) / 2) * 0.8  # side by side
            ax.errorbar(x, 100 * m["skill_difference"], yerr=[100 * (m["skill_difference"] - m["low"]),
                        100 * (m["high"] - m["skill_difference"])], color=color(model), marker="o", ms=4, lw=1.4,
                        capsize=2, label=label(model))
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(sorted(t["anchor_day"].unique()))
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.set_xlabel("anchor day = days of history at the first fit")
    axes[0, 0].set_ylabel("skill points over never (95 % CI)")
    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Refitting every day against never, by the history at the start (D83), "
                 f"{PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_grids(histogram: pd.DataFrame, panels: list[tuple[str, str]], horizons: list[int], period: str):
    fig, axes = plt.subplots(len(panels), len(horizons), figsize=(3.7 * len(horizons), 3.8 * len(panels)),
                             squeeze=False)
    for i, (model, policy) in enumerate(panels):
        for j, horizon in enumerate(horizons):
            h = histogram[(histogram["model"] == model) & (histogram["policy"] == policy) &
                          (histogram["horizon_min"] == horizon) & (histogram["period"] == period)]
            plot_clarke(axes[i, j], histogram=h, title=f"{label(model)} {policy}, {horizon} min")
            if j:
                axes[i, j].set_ylabel("")
            if i < len(panels) - 1:
                axes[i, j].set_xlabel("")
    fig.suptitle(f"Clarke Error Grid, stage 2, {PERIOD_LABELS.get(period, period)}: never and the best policy "
                 "(density, log scale)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(halving, overview, vs_never, triggers, later, twins, fit_log, task_log, missing, settings,
                  models, shortlist_note, round1, n_patients, keep, release, resources, period, anchor_days) -> str:
    days = ", ".join(str(a) for a in anchor_days)
    lines = ["# 30 - Stage 2: refit policies", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Models {models} ({shortlist_note}). Screen group: {n_patients} patients from anchor day {days}, "
             f"{settings['test_days']} test days; windows and frequencies of HANDOFF 7.4; every refit from the "
             f"model's starting point. Successive halving: round 1 on {len(round1)} patients, then the best {keep} "
             f"policies of each model plus its references on all; triggers {list(settings['triggers'])} in round 2.",
             ""]
    problems = []
    if not task_log.empty:
        problems += [f"{r.job} failed: {r.error}" for r in task_log[task_log["status"] == "error"].itertuples()]
        problems += [f"{r.job} raised warnings: {str(r.warnings)[:200]}"
                     for r in task_log[task_log["warnings"].fillna("") != ""].head(10).itertuples()]
        retry = task_log[task_log["retry_errors"].fillna("") != ""] if "retry_errors" in task_log else task_log.head(0)
        problems += [f"{r.job}: fits failed, not cached (computed again next run): {str(r.retry_errors)[:300]}"
                     for r in retry.head(10).itertuples()]
    problems += [f"no result for {m}" for m in missing[:10]]
    if not fit_log.empty and "error" in fit_log:
        failed = fit_log[fit_log["error"].notna()].drop_duplicates([c for c in FIT_KEYS if c in fit_log])
        for (model, error), g in failed.groupby(["model", "error"]):
            problems.append(f"{model}: {len(g)} fits failed: {error}")
    if not fit_log.empty and "excluded" in fit_log:
        out = fit_log[fit_log["excluded"].notna()].drop_duplicates(["model", "patient_id", "anchor_day"])
        for model, g in out.groupby("model"):
            where = ", ".join(f"{p} (day {a})" for p, a in zip(g["patient_id"], g["anchor_day"]))
            problems.append(f"{model}: left out for {where}: no model at the anchor (the fit of never failed)")
    if not fit_log.empty and "served_by" in fit_log:
        kept = fit_log[fit_log["served_by"].notna()]
        for model, g in kept.groupby("model"):
            problems.append(f"{model}: {len(g)} policy refits failed and kept the model in use "
                            f"({g['policy'].nunique()} policies; `served_by` in fit_log.csv)")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    outcomes = fit_outcomes(fit_log.drop_duplicates([c for c in FIT_KEYS if c in fit_log])) \
        if not fit_log.empty else pd.DataFrame()
    if not outcomes.empty:
        lines += ["## Fits (a run is complete when `other_failure` is 0)", "",
                  "Distinct fits: a fit shared by several policies counts once.", "", markdown_table(outcomes), "",
                  FIT_NOTE, ""]
    if not halving.empty:
        kept = halving[halving["kept"]]
        lines += ["## Round 1 (successive halving): policies that go on", "",
                  markdown_table(kept[["model", "policy", "skill_mean", "rank", "reason"]].round(4)), "",
                  "Every policy of round 1 is in `halving.csv` and `policy_grid.png`.", ""]
    if not overview.empty:
        o = overview[overview["period"] == period]
        show = [c for c in ("model", "policy", "kind", "skill_mean", "skill_30", "skill_60", "skill_120",
                            "refits_per_week", "seconds_per_day", "front", "patients") if c in o.columns]
        lines += ["## Policies on every patient (mean skill over patients; front: best skill for its cost)", "",
                  markdown_table(o[show].round(4)), ""]
    if not vs_never.empty:
        best = vs_never[(vs_never["period"] == period)].sort_values("skill_difference", ascending=False) \
            .groupby(["model", "horizon_min"]).first().reset_index()
        show = ["model", "horizon_min", "policy", "skill_difference", "low", "high", "rmse_saved_mgdl", "dmd",
                "verdict"]
        lines += ["## The price of not refitting (best policy against never)", "", markdown_table(best[show].round(4)),
                  "", "The best of several policies is picked here, so its gain is somewhat optimistic; every policy "
                  "against never is in `vs_never.csv` and `vs_never.png`.", ""]
    if not triggers.empty:
        show = ["model", "horizon_min", "policy", "against", "refits_per_week", "skill_difference", "low", "high",
                "patients_better", "dmd", "verdict"]
        t = triggers[triggers["period"] == period]
        lines += ["## Drift triggers against the same number of refits evenly spread", "", markdown_table(t[show].round(4)),
                  ""]
    if not later.empty:
        t = later[later["period"] == period].copy()
        t["cell"] = [f"{100 * d:+.2f} ({v}, {n})" for d, v, n in zip(t["skill_difference"], t["verdict"], t["patients"])]
        wide = t.pivot_table(index=["model", "policy", "horizon_min"], columns="anchor_day", values="cell",
                             aggfunc="first").reset_index()
        wide.columns = [f"anchor {c}" if isinstance(c, (int, np.integer)) else c for c in wide.columns]
        lines += ["## Later anchors (D83): gain over never with more history at the start", "",
                  "Skill points over never (verdict with the model's DMD, patients). At each anchor the first fit "
                  "uses every day since day 0, so the anchor day is also the days of history; each anchor has its "
                  "own patients (script 31 compares the patients with every anchor).", "",
                  markdown_table(wide), ""]
    if not twins.empty:
        a = twins[(twins["period"] == period) & twins["policy"].isin(["growing_every1", "growing_every14",
                                                                                  "online"])].copy()
        a["cell"] = [f"{100 * d:+.2f} ({v})" for d, v in zip(a["skill_difference"], a["verdict"])]
        wide = a.pivot_table(index=["model", "policy"], columns="horizon_min", values="cell", aggfunc="first")
        wide.columns = [f"{int(c)} min" for c in wide.columns]
        lines += ["## Glucose-only twins (D90): gain over never", "",
                  "The policies of their full models, for the twins that read the glucose only (no IOB, COB or time "
                  "of day; D76). Script 31 compares their gains with those of the full models; every policy of every twin is "
                  "in `vs_never.csv`.", "", markdown_table(wide.reset_index()), ""]
    lines += ["## Checks to look at", "",
              "- `policy_grid.png`: whether the window matters more than the frequency (stage 1 says so).",
              "- `pareto.png`: which policies give the best skill for their computing time.",
              "- `triggers.png`: whether a drift trigger beats refitting as often on a fixed calendar.",
              "- `later_anchors.png`: whether the gain of refitting shrinks when the model starts with more history.",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
