"""40 - Stage 3: how each model takes in new data (restart, warm start, leaf refit, scratch; ARX forgetting).

Runs on the screen group of the core dataset from the anchor of stage 2 (day
14 after randomisation, 84 test days), every model refitted at its optimal
interval of script 31 (``confirmation.yaml``, D84) on the growing window, and
compares the ways of taking in the new data (HANDOFF 7.5, D86;
``sim.mechanisms``):

- LightGBM: restart from the population model (D65), warm start (new trees on
  the previous personal model), leaf refit with decay 0.9, 0.5 and 0 (D88), and
  a personal model from scratch;
- LSTM and TiDE: restart with the fine-tuning of script 11, restart fine-tuning
  the other layers (every layer / only the output layers), and warm start;
- ARX: the update at every sample with forgetting factors from 0.99 to 1 (the
  reference: no forgetting, the factor of the settings since D87);
- TSMixerx too (the network that ages least, D79), with the same arms as the
  other networks; ``restart_other`` fine-tunes the other layers than script 11
  chose (every layer or only the head);
- every model's glucose-only twin (D90) with the same arms and interval, to
  see whether the mechanism matters differently without insulin and
  carbohydrates.

Every arm is scored on the same origins as never; the cost of an arm counts
every fit it uses (networks without the wait for a GPU turn, D73). Results are
cached per model, arm and patient, so a later run computes only what is new or
changed; ``--recompute`` ignores the cache. Script 41 adds the analysis
(efficiency, effect of insulin and carbohydrates, forgetting factor). One job
per patient, anchor and model on all cores but two, networks on the GPUs (rule
R5). The confirm group is never touched.

Derived data:
    <derived_data>/runs/40_stage3_mechanism_runs/day_metrics.parquet   per patient, model, arm, horizon and day
    <derived_data>/runs/40_stage3_mechanism_runs/fit_log.parquet       every fit (cost per fit, for script 41)
    <derived_data>/runs/40_stage3_mechanism_runs/cache/<model>/<arm>/<patient-anchor>.pkl

Outputs (results/40_stage3_mechanism_runs/):
    arm_summary.csv          per model, arm and horizon: patients, skill and RMSE with intervals
    arm_overview.csv         per model and arm: skill per horizon, fits per anchor, computing seconds per test day
    vs_restart.csv           every arm against restart (D65; ARX: online with the settings' factor): paired
                             differences, RMSE in mg/dL and verdict with the model's DMD (D75)
    vs_never.csv             every arm against never
    metrics_overall.csv      per model, arm and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6; points of never, restart and warm)
    fit_log.csv / fit_outcomes.csv / task_log.csv / jobs.csv
    mechanisms.png           every arm against restart, per model and horizon
    summary.md
    run_manifest.json

Usage:
    python scripts/40_stage3_mechanism_runs.py
    python scripts/40_stage3_mechanism_runs.py --models lightgbm_ft arx --patients 4   # a quick partial run
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
import yaml  # noqa: E402

from t1dretrain.compute import detect_resources, parallel_map, share_gpus  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.data.splits import group_patients  # noqa: E402
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.eval import policy_analysis as pa  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import (FIT_NOTE, PERIOD_LABELS, SUMS, DMDTable, color, finish,  # noqa: E402
                                           fit_outcomes, label, read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.cache import cache_valid, read_cache, retryable, write_cache  # noqa: E402
from t1dretrain.sim.mechanisms import parse_arm  # noqa: E402
from t1dretrain.sim.stage0 import (FAMILIES, STATISTICAL_MODELS, model_dependencies, population_path,  # noqa: E402
                                   split_name, split_variant, twin_families, with_twins)

SCRIPT_ID = "40_stage3_mechanism_runs"
FIT_KEYS = pr.FIT_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--models", nargs="*", default=None, help="subset of the stage-3 models (full names)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N patients (quick runs)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of earlier runs")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ jobs and cache


def run_job(job: dict) -> dict:
    """The arms of one model, patient and anchor that have no valid cached result; each is cached."""
    import time

    from t1dretrain.sim.mechanisms import evaluate_mechanisms

    started = time.time()
    todo = [a for a in job["arms"] if not cache_valid(job["cache"][a])]
    reference = not cache_valid(job["reference"])
    if not todo and not reference:
        return {"job": job["name"], "status": "cached", "arms": [], "seconds": 0.0, "warnings": "", "error": ""}
    with WarningRecorder() as recorder:
        result = evaluate_mechanisms({**job, "arms": todo, "with_reference": reference})
    status = {"job": job["name"], "status": "error" if result.error else "ok", "arms": todo,
              "seconds": round(time.time() - started, 1), "warnings": recorder.text(), "error": result.error}
    if result.error:
        return status
    not_cached = []
    for arm in todo:
        days = result.days[result.days["policy"] == arm] if not result.days.empty else result.days
        clarke = result.clarke.subset(lambda key, arm=arm: key["policy"] == arm)
        info = [row for row in result.info if row.get("policy") == arm]
        if any(retryable(row) for row in info):  # an unexpected failure (e.g. CUDA): computed again next run
            not_cached.append(arm)
            continue
        refits = {k: v for k, v in result.refits.items() if k.split("|")[0] == arm}
        write_cache(job["cache"][arm], {"days": days, "clarke": clarke, "info": info, "refits": refits})
    if reference:
        days = result.days[result.days["policy"] == "reference"] if not result.days.empty else result.days
        write_cache(job["reference"], {"days": days, "info": [], "refits": {},
                                       "clarke": result.clarke.subset(lambda key: key["policy"] == "reference")})
    status["not_cached"] = not_cached
    status["retry_errors"] = sorted({row["error"] for row in result.info if retryable(row)})
    status["cuda_fallbacks"] = sum(1 for row in result.info if row.get("cuda_error"))
    return status


def cache_key(model: str, arm: str, settings: dict, choices: dict, interval: int) -> str:
    """What a cached result depends on: model, arm, interval, starting point, settings, the simulator."""
    import hashlib
    import json

    from t1dretrain.sim.mechanisms import CACHE_VERSION

    relevant = {k: settings[k] for k in ("horizons_min", "lookback_hours", "origin_stride_min", "train_stride_min",
                                         "validation_days", "min_window_share", "recent_complete_min", "window_days",
                                         "test_days", "seeds")}
    relevant.update(model_dependencies(model, settings, choices))
    relevant["interval"] = int(interval) if arm != "reference" else None
    text = json.dumps({"model": model, "arm": arm, "cache_version": CACHE_VERSION, **relevant}, sort_keys=True,
                      default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def device_for(resources) -> str:
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
    stage, stage3 = config["stage0"], config["stage3"]
    period = list(stage3["periods"])[0]
    population_dir = derived / "models" / core / "population"
    choices = yaml.safe_load((population_dir / "choices.yaml").read_text()) \
        if (population_dir / "choices.yaml").exists() else {}
    settings = {"horizons_min": [int(h) for h in config["horizons_min"]], "lookback_hours": float(config["lookback_hours"]),
                "origin_stride_min": float(config["origin_stride_min_screen"]),
                "train_stride_min": float(stage["train_stride_min"]), "validation_days": int(stage["validation_days"]),
                "min_window_share": float(stage["min_window_share"]),
                "recent_complete_min": float(stage["recent_complete_min"]),
                "window_days": int(config["anchors"]["train_window_days"]),
                "test_days": int(config["anchors"]["max_age_days"]),
                "clarke_point_sample": int(config["evaluation"]["clarke_point_sample"]),
                "seeds": [int(s) for s in stage3["seeds"]], "population_dir": str(population_dir),
                "finetune": {f: (choices.get(f) or {}).get("finetune") for f in FAMILIES}}
    for f in FAMILIES:
        settings[f] = (choices.get(f) or {}).get("scratch") or stage.get(f)
    for m in STATISTICAL_MODELS:
        settings[m] = stage.get(m) or {}
    arms_of, notes = stage3_arms(stage3, args.models, population_dir, settings, twin_families(config))
    intervals, interval_note = pa.confirmation_intervals(derived, stage3.get("interval_days", "confirmation"),
                                                         list(arms_of))
    if any(split_name(m)[0] in FAMILIES for m in arms_of) and not choices:
        raise SystemExit(f"{population_dir / 'choices.yaml'} not found: run scripts/11_pretrain_population.py first")
    models = list(arms_of)

    patients = group_patients(core, stage3.get("group", "screen"))
    if args.patients:
        patients = patients[: args.patients]
    anchors = ProcessedStore(core, root=derived, stage="clean").read_extra("anchors")
    anchors = anchors[(anchors["cohort"] == "core") & anchors["patient_id"].isin(patients) &
                      (anchors["period"] == period) & anchors["available"].astype(bool) &
                      anchors["anchor_day"].isin([int(a) for a in stage3["anchor_days"]])].copy()
    anchors["anchor"] = pd.to_datetime(anchors["day0"]) + pd.to_timedelta(anchors["anchor_day"], unit="D")
    anchors = anchors.sort_values(["patient_id", "anchor_day"]).reset_index(drop=True)
    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    jobs = []
    for r in anchors.itertuples():
        name = f"{r.patient_id}__{r.period}__{int(r.anchor_day)}"
        reference = {"path": str(runs / "cache" / "persistence" / f"{name}.pkl"),
                     "key": cache_key("persistence", "reference", settings, choices, 0), "recompute": recompute}
        for m in models:
            cache = {a: {"path": str(runs / "cache" / m / a / f"{name}.pkl"),
                         "key": cache_key(m, a, settings, choices, intervals[m]), "recompute": recompute}
                     for a in arms_of[m]}
            jobs.append({"name": f"{name}__{m}", "dataset": core, "patient_id": r.patient_id, "period": r.period,
                         "anchor_day": int(r.anchor_day), "anchor": r.anchor, "derived_root": derived,
                         "settings": settings, "model": m, "arms": arms_of[m], "interval": intervals[m],
                         "cache": cache, "reference": reference, "stream": len(jobs), "device": device_for(resources)})
    jobs.sort(key=lambda j: split_name(j["model"])[0] not in FAMILIES or j["model"].startswith("lightgbm"))
    print(f"Models {models}; intervals {intervals} ({interval_note}); {anchors['patient_id'].nunique()} patients; "
          f"{len(jobs)} jobs")
    statuses = parallel_map(run_job, jobs, resources.n_workers, label="stage-3 jobs")

    days, infos, clarke, missing = collect(jobs)
    runs.mkdir(parents=True, exist_ok=True)
    days.to_parquet(runs / "day_metrics.parquet", index=False)
    fit_log = pd.DataFrame(infos)
    if not fit_log.empty:
        write_fit_log(fit_log, runs / "fit_log.parquet")
    task_log = pd.DataFrame([{"job": r["job"], "status": r["status"], "arms_run": " ".join(r["arms"]),
                              "not_cached": " ".join(r.get("not_cached", [])),
                              "retry_errors": " | ".join(r.get("retry_errors", [])),
                              "cuda_fallbacks": r.get("cuda_fallbacks", 0), "seconds": r["seconds"],
                              "warnings": r["warnings"], "error": r["error"].strip().splitlines()[-1] if r["error"] else "",
                              "traceback": r["error"]} for r in statuses])
    replicates, seed = int(stage["bootstrap_replicates"]), int(config["random_seed"])
    dmd = DMDTable(read_noise_floor(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in models if split_name(m)[0] in FAMILIES])
    test_days = int(settings["test_days"])
    skill = pr.patient_skill(days) if not days.empty else pd.DataFrame()
    used = fit_log[fit_log["excluded"].isna()] if "excluded" in fit_log else fit_log
    summary = pr.summary_table(skill, replicates, seed) if not skill.empty else pd.DataFrame()
    overview = arm_overview(skill, used, test_days)
    references = {m: reference_arm(arms_of[m], float((settings.get(split_name(m)[0]) or {}).get("forgetting", 1.0)))
                  for m in models}
    vs_restart = paired_by_model(skill, references, dmd, replicates, seed) if not skill.empty else pd.DataFrame()
    vs_never = pr.paired(skill, lambda a: "never", dmd, replicates, seed) if not skill.empty else pd.DataFrame()
    overall = finish(days.groupby(["period", "model", "policy", "horizon_min"], as_index=False)[SUMS].sum()).round(4) \
        if not days.empty else pd.DataFrame()
    sample = clarke.sample()
    if not sample.empty:  # the histogram has every panel; the points only those of never, the reference and warm
        sample = sample[sample["policy"].isin(["never", "restart", "warm", "reference"] + list(references.values()))]

    out.csv(summary.round(4), "arm_summary")
    out.csv(overview.round(4), "arm_overview")
    out.csv(vs_restart.round(4), "vs_restart")
    out.csv(vs_never.round(4), "vs_never")
    out.csv(overall, "metrics_overall")
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(clarke.histogram(), "clarke_histogram")
    out.csv(sample, "clarke_points_sample")
    out.csv(fit_log, "fit_log")
    out.csv(fit_outcomes(fit_log.drop_duplicates([c for c in FIT_KEYS + ["policy"] if c in fit_log]))
            if not fit_log.empty else pd.DataFrame(), "fit_outcomes")
    out.csv(task_log, "task_log")
    out.csv(anchors.drop(columns=["dataset"], errors="ignore"), "jobs")
    if not vs_restart.empty:  # the full models; their twins are in the tables and in script 41 (D90)
        full = [m for m in models if not split_variant(m)[1]]
        out.figure(plot_mechanisms(vs_restart, full, arms_of, references, period), "mechanisms")
    out.text(write_summary(overview, vs_restart, fit_log, task_log, missing, models, arms_of, intervals,
                           interval_note, notes, references, anchors, release, resources,
                           period, {f: v["train"] for f, v in settings["finetune"].items() if v and v.get("train")}),
           "summary.md")
    out.manifest(settings={**{k: v for k, v in settings.items() if k != "population_dir"}, "period": period,
                           "models": models, "arms": arms_of, "intervals": intervals, "references": references,
                           "patients": int(anchors["patient_id"].nunique()), "population_models": str(population_dir),
                           "choices_release": choices.get("release")},
                 notes="One job per patient, anchor and model; day metrics in <derived>/runs/40_stage3_mechanism_runs.")
    print(f"Done. Results in {out.dir}")


def stage3_arms(stage3: dict, chosen: list[str] | None, population_dir: Path, settings: dict,
                families: list[str] | None) -> tuple[dict, list]:
    """The arms of every model of ``stage3.arms`` that can run, each followed by its glucose-only twin with the same
    arms (D90; ``families``: the learned families with twins). ``--models`` filters by the full model's name;
    learned models need their population models of script 11."""
    out, notes = {}, []
    for model, arms in (stage3.get("arms") or {}).items():
        if chosen and model not in chosen:
            continue
        for arm in arms:
            parse_arm(arm)  # an unknown arm stops the run here
        for name in with_twins([model], families):
            base, variant = split_variant(name)
            family = split_name(name)[0]
            if family in FAMILIES and split_name(name)[1] != "scratch":
                path = population_path(population_dir, family, int(settings["seeds"][0]), variant)
                if not path.exists():
                    notes.append(f"{name} left out: no population model {path}")
                    continue
            out[name] = list(arms)
    return out, notes


def reference_arm(arms: list[str], forgetting: float = 1.0) -> str:
    """The arm every other arm is compared with: restart (D65), or for ARX the online update with the forgetting
    factor of the settings (1 since D87, the ``online`` policy of stage 2), else the first online arm."""
    if "restart" in arms:
        return "restart"
    online = [a for a in arms if parse_arm(a)[0] == "online"]
    own = f"online_l{float(forgetting):g}"
    return own if own in online else (online[0] if online else "never")


def paired_by_model(skill: pd.DataFrame, references: dict, dmd, replicates: int, seed: int) -> pd.DataFrame:
    parts = []
    for model, reference in references.items():
        part = skill[skill["model"] == model]
        if part.empty:
            continue
        parts.append(pr.paired(part, lambda a, ref=reference: ref if a not in (ref, "never") else None, dmd,
                               replicates, seed))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect(jobs: list[dict]):
    frames, infos, missing = [], [], []
    clarke = ClarkeAccumulator(["model", "policy", "horizon_min", "period"], sample_size=20000)
    seen = set()
    for job in jobs:
        entries = [(a, job["cache"][a]) for a in job["arms"]]
        if job["reference"]["path"] not in seen:
            entries.append(("reference", job["reference"]))
            seen.add(job["reference"]["path"])
        for arm, entry in entries:
            cached = read_cache(entry)
            if cached is None:
                missing.append(f"{job['name']}:{arm}")
                continue
            frames.append(cached["days"])
            infos += cached["info"]
            clarke.merge(cached["clarke"])
    days = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    return days, infos, clarke, missing


def write_fit_log(fit_log: pd.DataFrame, path: Path) -> None:
    columns = [c for c in FIT_KEYS + ["policy", "test_origins", "train_days", "train_samples", "fit_seconds",
                                      "compute_seconds", "rounds", "added_rounds", "steps", "best_step", "error",
                                      "excluded", "served_by"] if c in fit_log]
    table = fit_log[columns].copy()
    for c in ("fit_day", "train_start"):
        if c in table:
            table[c] = pd.to_datetime(table[c].astype(str), errors="coerce")
    for c in table.columns:
        if table[c].dtype == object:
            table[c] = table[c].astype("string")
    table.to_parquet(path, index=False)


def arm_overview(skill: pd.DataFrame, fit_log: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Per model and arm: mean skill over horizons and per horizon, fits per anchor, computing seconds per day."""
    if skill.empty:
        return pd.DataFrame()
    table = pr.ranking(skill).drop(columns=["rank"])
    table = table.merge(pr.cost_table(fit_log, test_days), on=["period", "model", "policy"], how="left")
    by_horizon = skill.groupby(["period", "model", "policy", "horizon_min"])["skill"].mean().unstack("horizon_min")
    by_horizon.columns = [f"skill_{int(h)}" for h in by_horizon.columns]
    table = table.merge(by_horizon.reset_index(), on=["period", "model", "policy"], how="left")
    return table.rename(columns={"policy": "arm"}).sort_values(["period", "model", "skill_mean"],
                                                               ascending=[True, True, False]).reset_index(drop=True)


# ============================================================================ figure and summary


def plot_mechanisms(table: pd.DataFrame, models: list[str], arms_of: dict, references: dict, period: str):
    t = table[table["period"] == period]
    shown = [m for m in models if m in set(t["model"])]
    fig, axes = model_axes(len(shown), 4.4, 3.2)
    for ax, model in zip(axes, shown):
        m = t[t["model"] == model]
        arms = [a for a in arms_of[model] if a in set(m["policy"])]
        horizons = sorted(m["horizon_min"].unique())
        for j, horizon in enumerate(horizons):
            h = m[m["horizon_min"] == horizon].set_index("policy")
            for i, arm in enumerate(arms):
                r = h.loc[arm]
                y = i + (j - 1) * 0.22
                if j == 0 and np.isfinite(r["dmd"]):
                    ax.fill_betweenx([i - 0.4, i + 0.4], -100 * r["dmd"], 100 * r["dmd"], color=TEXT_SECONDARY,
                                     alpha=0.10, lw=0)
                ax.plot([100 * r["low"], 100 * r["high"]], [y, y], color=color(model), lw=1.4, alpha=0.4 + 0.3 * j)
                ax.scatter(100 * r["skill_difference"], y, color=color(model), s=16, alpha=0.4 + 0.3 * j,
                           marker="o" if r["verdict"] in ("better", "worse") else "s",
                           label=f"{horizon} min" if i == 0 else None)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(range(len(arms)), arms, fontsize=7)
        ax.set_title(f"{label(model)} (against {references[model]})", loc="left", fontsize=9)
        ax.set_xlabel("skill points against the reference arm")
    if shown:
        axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"How each model takes in new data (D86): every arm against the reference, "
                 f"{PERIOD_LABELS.get(period, period)} (band: DMD; circle: beyond it)", fontweight="bold", x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


def write_summary(overview, vs_restart, fit_log, task_log, missing, models, arms_of, intervals, interval_note, notes,
                  references, anchors, release, resources, period, trains=None) -> str:
    lines = ["# 40 - Stage 3: how each model takes in new data", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Models and arms: {arms_of}. Refit intervals (days): {intervals} ({interval_note}). "
             f"{anchors['patient_id'].nunique()} patients of the screen group, anchor day(s) "
             f"{sorted(anchors['anchor_day'].unique().tolist())}, {PERIOD_LABELS.get(period, period)}; growing window; "
             f"references: {references}. Fine-tuning chosen by script 11 (restart; restart_other is the other "
             f"one): {trains}. Glucose-only twins (`_noexog`, D90) run the arms and interval of their full model.", ""]
    problems = list(notes)
    if not task_log.empty:
        problems += [f"{r.job} failed: {r.error}" for r in task_log[task_log["status"] == "error"].itertuples()]
        problems += [f"{r.job} raised warnings: {str(r.warnings)[:200]}"
                     for r in task_log[task_log["warnings"].fillna("") != ""].head(10).itertuples()]
        retry = task_log[task_log["retry_errors"].fillna("") != ""]
        problems += [f"{r.job}: fits failed, not cached (computed again next run): {str(r.retry_errors)[:300]}"
                     for r in retry.head(10).itertuples()]
    problems += [f"no result for {m}" for m in missing[:10]]
    if not fit_log.empty and "error" in fit_log:
        for (model, error), g in fit_log[fit_log["error"].notna()].groupby(["model", "error"]):
            problems.append(f"{model}: {len(g)} fits failed: {error}")
    if not fit_log.empty and "excluded" in fit_log:
        out = fit_log[fit_log["excluded"].notna()].drop_duplicates(["model", "patient_id", "anchor_day"])
        for model, g in out.groupby("model"):
            problems.append(f"{model}: left out for {', '.join(g['patient_id'])}: no model at the anchor")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    outcomes = fit_outcomes(fit_log.drop_duplicates([c for c in FIT_KEYS + ["policy"] if c in fit_log])) \
        if not fit_log.empty else pd.DataFrame()
    if not outcomes.empty:
        lines += ["## Fits (a run is complete when `other_failure` is 0)", "", "One row per arm and fit.", "",
                  markdown_table(outcomes), "", FIT_NOTE, ""]
    if not overview.empty:
        o = overview[overview["period"] == period]
        show = [c for c in ("model", "arm", "skill_mean", "skill_30", "skill_60", "skill_120", "fits_per_anchor",
                            "seconds_per_day", "patients") if c in o.columns]
        lines += ["## Arms on every patient (mean skill over patients)", "", markdown_table(o[show].round(4)), ""]
    if not vs_restart.empty:
        v = vs_restart[vs_restart["period"] == period]
        show = ["model", "policy", "against", "horizon_min", "skill_difference", "low", "high", "rmse_saved_mgdl",
                "patients_better", "dmd", "verdict"]
        lines += ["## Every arm against the reference (positive: better than the reference)", "",
                  markdown_table(v[show].round(4)), ""]
    lines += ["## Checks to look at", "",
              "- `mechanisms.png`: whether a warm start or a leaf refit keeps the skill of restarting from the "
              "population model, and whether LightGBM from scratch catches up with enough data.",
              "- Script 41: the cost of each arm, the effect of insulin and carbohydrates, and ARX's forgetting.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
