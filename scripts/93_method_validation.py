"""93 - Validation of the method for the MethodsX article (D112): five checks of the simulator and its statistics.

The article describes the evaluation method of this project (the simulator,
the staged screening and the confirmation); this script gives the evidence
that the method does what it claims (``eval.method_validation``):

1. **designs** - Common evaluation designs against the simulator
   (``sim.naive_splits``). LightGBM fine-tuned and ARX on the screening
   patients of stage 3 (anchor, test days and origins of script 40): a random
   split of the samples, a random split of the days and one chronological
   split of the patient's record, each scored on its held-out origins together
   with the simulator's frozen and refitted models on the same origins. A few
   patients are run twice to check that a run is deterministic.
2. **ageing** - The ageing estimator of script 43 (the slope of the daily skill
   minus that of a model that cannot age, D99) on synthetic cohorts with a
   known loss, whose day-to-day noise is calibrated on the day metrics of
   scripts 40 and 43; and the width of its intervals on the real data.
3. **verdicts** - The tie rule with the DMD (D61, D75) on null comparisons
   (every learned model of stage 0 against itself with another seed, script 12)
   and with known differences added: false findings and power, against the
   interval alone.
4. **pairing** - The width of the interval of a difference when the arms are
   paired by patient and origin, against unpaired designs: the starting points
   of stage 0 (script 12) and, in the confirmation (day metrics of script 50),
   refitting against the frozen model, also against a crossover in time.
5. **runs** - Reproducibility (every table of this results tree against
   another one, ``--compare``) and the cost of every stage (run manifests and fit
   logs).

Parts 1 and the calibration of 2 and the crossover of 4 need the derived data of the server; without them they are
left out and the summary says so. Rule R6: RMSE and Clarke zones of every forecast of part 1, with the grid data.

Inputs:
    results/12_stage0_population_transfer/  metrics_by_patient.csv, noise_floor.csv, paired_comparisons.csv
    results/43_stage3_context/ageing_vs_context.csv; results/51_confirmation_analysis/replication.csv
    results/<script>/run_manifest.json and fit_log.csv of every script (part 5)
    <derived_data>/features/<core>/grid/, clean/<core>/anchors.parquet, models/<core>/population/   (part 1)
    <derived_data>/runs/31_stage2_policy_analysis/confirmation.yaml                                (part 1)
    <derived_data>/runs/40_stage3_mechanism_runs/ and 43_stage3_context/day_metrics.parquet        (part 2)
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet                                   (part 4)

Derived data:
    <derived_data>/runs/93_method_validation/cache/<model>/<patient>.pkl   (part 1, one per patient and model)

Outputs (results/93_method_validation/):
    designs_skill.csv          per model, design, arm, horizon and patient: RMSE, Clarke A+B and skill
    designs_inflation.csv      per model, design and horizon: skill of each arm and design minus frozen / refitted
    designs_metrics.csv        pooled forecasts per model, design and arm (R6: RMSE, MAE, MARD, bias, Clarke zones)
    designs_fit_log.csv / designs_determinism.csv / designs_jobs.csv
    clarke_zones.csv / clarke_histogram.csv / clarke_points_sample.csv   (R6, part 1)
    ageing_calibration.csv     variance components of the real daily skill (model and reference)
    ageing_simulation.csv / ageing_simulation_summary.csv   synthetic cohorts: bias, coverage, width ratio
    ageing_real.csv            the width ratio of the intervals on the real data (script 43)
    verdicts_comparisons.csv / verdicts_calibration.csv / verdicts_power.csv
    pairing_stage0.csv / pairing_confirmation.csv / crossover_confirmation.csv
    runs_cost.csv / stage_cost.csv / runs_identity.csv
    fig_designs, fig_ageing, fig_verdicts, fig_pairing (PDF and PNG at the printed size), with their data CSVs
    summary.md, run_manifest.json

Usage:
    python scripts/93_method_validation.py
    python scripts/93_method_validation.py --parts 2 3 4 5           # without the simulator runs
    python scripts/93_method_validation.py --patients 3               # part 1 on the first three patients
    python scripts/93_method_validation.py --compare /path/to/other/results   # part 5: identical outputs?
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

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, RESULTS_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.splits import cohort_rule  # noqa: E402
from t1dretrain.eval import method_validation as mv  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.metrics import ClarkeAccumulator  # noqa: E402
from t1dretrain.eval.stage0_report import SUMS, finish  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import (PAPER_FAMILIES, SEQUENTIAL_BLUE, TEXT_SECONDARY, paper_figure,  # noqa: E402
                                 panel_letter, set_paper_style)
from t1dretrain.sim import cohort_runs as cr  # noqa: E402
from t1dretrain.sim import naive_splits as ns  # noqa: E402
from t1dretrain.sim.cache import cache_valid, read_cache, write_cache  # noqa: E402
from t1dretrain.sim.stage0 import is_twin, split_name  # noqa: E402

SCRIPT_ID = "93_method_validation"
PARTS = {1: "designs", 2: "ageing", 3: "verdicts", 4: "pairing", 5: "runs"}
SPLIT_LABELS = {"random_sample": "Random split of samples", "random_day": "Random split of days",
                "chronological": "One chronological split"}
HORIZON_COLOURS = {30: SEQUENTIAL_BLUE[2], 60: SEQUENTIAL_BLUE[4], 120: SEQUENTIAL_BLUE[6]}
HORIZON_MARKERS = {30: "o", 60: "s", 120: "D"}


class Inputs:
    """Reads the tables of the earlier scripts and remembers which ones were used."""

    def __init__(self, results: Path) -> None:
        self.results = results
        self.used: list[str] = []

    def csv(self, script: str, name: str, required: bool = True) -> pd.DataFrame:
        target = self.results / script / f"{name}.csv"
        if not target.exists():
            if required:
                raise SystemExit(f"{target} not found: run scripts/{script}.py first (or pass --results)")
            return pd.DataFrame()
        self.used.append(f"{script}/{name}.csv")
        return pd.read_csv(target, low_memory=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parts", type=int, nargs="+", default=sorted(PARTS), choices=sorted(PARTS),
                        help="parts to run (1 designs, 2 ageing, 3 verdicts, 4 pairing, 5 runs; default all)")
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR,
                        help="folder with the results of the earlier scripts (default: results/)")
    parser.add_argument("--compare", type=Path, default=None,
                        help="another results folder: part 5 compares every table of both (identical outputs?)")
    parser.add_argument("--patients", type=int, default=None, help="part 1 on the first N patients (quick check)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached results of part 1")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ main


def main() -> None:
    import time

    run_start = time.time()
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    cfg = config["method_validation"]
    seed, replicates = int(config["random_seed"]), int(config["stage0"]["bootstrap_replicates"])
    derived = derived_data_dir(args.paths)
    inputs = Inputs(args.results.resolve())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_paper_style()
    notes, tables = [], {}

    if 1 in args.parts:
        tables |= part_designs(config, cfg["designs"], derived, args, resources, seed, replicates, run_start, out,
                               notes)
    if 2 in args.parts:
        tables |= part_ageing(cfg["ageing"], derived, inputs, resources, seed, notes)
    if 3 in args.parts:
        tables |= part_verdicts(cfg["verdicts"], inputs, replicates, seed)
    if 4 in args.parts:
        tables |= part_pairing(cfg["pairing"], derived, inputs, replicates, seed, notes)
    if 5 in args.parts:
        tables |= part_runs(derived, inputs, args.compare, notes)

    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame):
            out.csv(frame.round(6) if name != "designs_jobs" else frame, name)
    figures(tables, cfg, out)
    out.text(write_summary(tables, cfg, notes, args, release, resources), "summary.md")
    out.manifest(settings={"parts": [PARTS[p] for p in args.parts], "results": str(args.results),
                           "compare": str(args.compare) if args.compare else None,
                           "inputs": sorted(set(inputs.used)), "bootstrap_replicates": replicates, **cfg},
                 notes="Validation of the method for the MethodsX article (D112).")
    print(f"Done. Results in {out.dir}")


# ============================================================================ 1. designs against the simulator


def refit_intervals(derived: Path, models: list[str], fallback: dict) -> tuple[dict, str]:
    """The refit interval of each model: its optimal interval of script 31 (``confirmation.yaml``; ARX refits every
    day there), else the fallback of the config."""
    path = derived / "runs" / "31_stage2_policy_analysis" / "confirmation.yaml"
    plan = (yaml.safe_load(path.read_text()) or {}).get("models", {}) if path.exists() else {}
    out = {}
    for m in models:
        value = (plan.get(m) or {}).get("optimal_interval_days")
        out[m] = int(value) if value else int(fallback[m])
    source = "confirmation.yaml of script 31" if plan else "the config (confirmation.yaml of script 31 not found)"
    return out, source


def design_jobs(config: dict, cfg: dict, derived: Path, args, resources, seed: int, run_start: float) -> tuple:
    stage3 = config["stage3"]
    core = config.get("core_dataset", "replace_bg")
    population_dir = derived / "models" / core / "population"
    choices = cr.read_choices(population_dir)
    base = cr.base_settings(config, choices, population_dir)
    base.update({"origin_stride_min": float(config["origin_stride_min_screen"]),  # the protocol of script 40
                 "test_days": int(config["anchors"]["max_age_days"]), "seeds": [int(s) for s in stage3["seeds"]]})
    period = list(stage3["periods"])[0]
    cohort = {"dataset": core, "cohort": "core", "group": stage3.get("group", "screen"), "period": period}
    anchors, counts = cr.cohort_patients(cohort, derived, [int(a) for a in stage3["anchor_days"]], seed, args.patients)
    rule = cohort_rule(core, "core")
    models = list(cfg["models"])
    intervals, source = refit_intervals(derived, models, cfg["refit_every"])
    runs = derived / "runs" / SCRIPT_ID
    recompute = run_start if args.recompute else None
    jobs = []
    for r in anchors.itertuples():
        for m in models:
            arm = f"designs_k{intervals[m]}_share{cfg['held_out_share']}_v{ns.CACHE_VERSION}"
            cache = {"path": str(runs / "cache" / m / f"{r.patient_id}__{r.period}__{int(r.anchor_day)}.pkl"),
                     "key": cr.cache_key("designs", rule, m, arm, base, choices), "recompute": recompute}
            jobs.append({"name": f"{r.patient_id}__{m}", "dataset": core, "usable": rule, "patient_id": r.patient_id,
                         "period": r.period, "anchor_day": int(r.anchor_day), "anchor": r.anchor,
                         "derived_root": derived, "settings": base, "model": m, "stream": len(jobs), "device": "cpu",
                         "refit_every": intervals[m], "share": float(cfg["held_out_share"]), "split_seed": seed,
                         "cache": cache})
    return jobs, anchors, counts, intervals, source


def run_design_job(job: dict) -> dict:
    """One patient and model of part 1, cached."""
    import time

    started = time.time()
    if cache_valid(job["cache"]):
        return {"job": job["name"], "status": "cached", "seconds": 0.0, "error": ""}
    result = ns.evaluate_splits(job)
    if result.error:
        return {"job": job["name"], "status": "error", "seconds": round(time.time() - started, 1),
                "error": result.error}
    write_cache(job["cache"], {"days": result.days, "clarke": result.clarke, "info": result.info})
    return {"job": job["name"], "status": "ok", "seconds": round(time.time() - started, 1), "error": ""}


def run_twice(job: dict) -> list[dict]:
    """The same patient and model run twice from scratch, and against its cached result: are the forecasts the
    same? (LightGBM is deterministic with a fixed seed; ARX is a least-squares fit.)"""
    keys = ["seed", "split", "arm", "horizon_min", "day"]
    runs = [ns.evaluate_splits(job) for _ in range(2)]
    cached = read_cache(job["cache"])
    frames = [r.days for r in runs] + ([cached["days"]] if cached else [])
    rows = []
    if any(r.error for r in runs) or any(f.empty for f in frames):
        return [{"job": job["name"], "comparison": "runs", "error": next((r.error for r in runs if r.error), "empty")}]
    first = frames[0].set_index(keys)[["n", "sse"]].sort_index()
    for name, other in zip(("second run", "cached run"), frames[1:]):
        other = other.set_index(keys)[["n", "sse"]].sort_index()
        same_index = first.index.equals(other.index)
        diff = float(np.max(np.abs(first["sse"].to_numpy() - other["sse"].to_numpy()))) if same_index else np.nan
        rows.append({"job": job["name"], "model": job["model"], "patient_id": job["patient_id"], "comparison": name,
                     "rows": int(len(first)), "same_origins": bool(same_index and first["n"].equals(other["n"])),
                     "identical": bool(same_index and first["sse"].equals(other["sse"])),
                     "largest_sse_difference": diff})
    return rows


def part_designs(config, cfg, derived, args, resources, seed, replicates, run_start, out, notes) -> dict:
    core = config.get("core_dataset", "replace_bg")
    needed = [derived / "features" / core, derived / "models" / core / "population" / "choices.yaml"]
    if not all(p.exists() for p in needed):
        absent = ", ".join(str(p) for p in needed if not p.exists())
        notes.append(f"Part 1 (designs) left out: no derived data ({absent})")
        return {}
    jobs, anchors, counts, intervals, source = design_jobs(config, cfg, derived, args, resources, seed, run_start)
    print(f"Part 1: {anchors['patient_id'].nunique()} patients; models {list(intervals)}; refit every {intervals} "
          f"days ({source}); {len(jobs)} jobs")
    statuses = parallel_map(run_design_job, jobs, resources.n_workers, label="design jobs")
    frames, infos, missing = [], [], []
    clarke = ClarkeAccumulator(["model", "split", "arm", "horizon_min"], sample_size=20000)
    for job in jobs:
        cached = read_cache(job["cache"])
        if cached is None:
            missing.append(job["name"])
            continue
        if not cached["days"].empty:
            frames.append(cached["days"])
        infos += cached["info"]
        if cached["clarke"] is not None:
            clarke.merge(cached["clarke"])
    days = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    errors = [s for s in statuses if s["status"] == "error"]
    if errors:
        notes.append(f"Part 1: {len(errors)} jobs failed, e.g. {errors[0]['job']}: "
                     f"{errors[0]['error'].strip().splitlines()[-1]}")
    if missing:
        notes.append(f"Part 1: {len(missing)} jobs without a result")
    if days.empty:
        notes.append("Part 1: no day metrics")
        return {}
    twice = [j for m in intervals for j in [x for x in jobs if x["model"] == m][: int(cfg["determinism_patients"])]]
    determinism = pd.DataFrame([row for rows in parallel_map(run_twice, twice, resources.n_workers,
                                                             label="determinism checks") for row in rows])
    skill = mv.split_skill(days)
    sums = [c for c in SUMS if c in days]
    pooled = days.groupby(["model", "split", "arm", "horizon_min"], as_index=False)[sums].sum()
    out.csv(clarke.zones(), "clarke_zones")
    out.csv(clarke.histogram(), "clarke_histogram")
    out.csv(clarke.sample(), "clarke_points_sample")
    notes.append(f"Part 1: refit intervals {intervals} ({source}); counts {counts}")
    return {"designs_skill": skill, "designs_inflation": mv.inflation(skill, replicates, seed),
            "designs_metrics": finish(pooled).round(4), "designs_fit_log": pd.DataFrame(infos),
            "designs_determinism": determinism, "designs_jobs": pd.DataFrame(statuses)}


# ============================================================================ 2. the ageing estimator


def real_daily(derived: Path, model: str, reference: str, test_days: int, min_origins: int) -> pd.DataFrame:
    """The daily skill of a frozen model of script 40 (``"lightgbm_ft never"``) and of a zero-shot model with a
    fixed context of script 43 (``"chronos2 14 d"``), on the same patients and days, against the persistence of
    script 40 (the layout of ``ageing_analysis.against_reference``)."""
    path40 = derived / "runs" / "40_stage3_mechanism_runs" / "day_metrics.parquet"
    path43 = derived / "runs" / "43_stage3_context" / "day_metrics.parquet"
    if not path40.exists() or not path43.exists():
        return pd.DataFrame()
    columns = ["patient_id", "model", "policy", "horizon_min", "day", "days_since_anchor", "n", "sse"]
    d40 = pd.read_parquet(path40, columns=columns)
    d43 = pd.read_parquet(path43, columns=columns + ["context_days"])
    name, policy = model.split(" ", 1)
    ref_name, ref_days = reference.split(" ")[0], float(reference.split(" ")[1])
    parts = {model: d40[(d40["model"] == name) & (d40["policy"] == policy)],
             reference: d43[(d43["model"] == ref_name) & (d43["context_days"] == ref_days)],
             "persistence": d40[d40["model"] == "persistence"]}
    keys = ["patient_id", "horizon_min", "day"]
    table = []
    for label, part in parts.items():
        part = part[part["days_since_anchor"] < test_days]
        g = part.groupby(keys, as_index=False).agg(n=("n", "sum"), sse=("sse", "sum"),
                                                   days_since_anchor=("days_since_anchor", "first"))
        table.append(g.assign(label=label))
    t = pd.concat(table, ignore_index=True)
    t = t[t["n"] >= min_origins]
    ref = t[t["label"] == "persistence"].set_index(keys)
    rmse_ref = np.sqrt(ref["sse"] / ref["n"])
    d = t[t["label"] != "persistence"].copy()
    d["rmse_persistence"] = d.set_index(keys).index.map(rmse_ref)
    d = d[d["rmse_persistence"] > 0]
    d["skill"] = (1 - np.sqrt(d["sse"] / d["n"]) / d["rmse_persistence"]).clip(-1, 1)
    d["week"] = (d["days_since_anchor"] // 7).astype(int)
    d["model"], d["policy"] = d["label"].str.split(" ").str[0], "never"
    both = d.groupby(keys)["label"].transform("nunique") == 2  # the same patients and days for both
    return d[both][["label", "model", "policy", "patient_id", "horizon_min", "day", "week", "skill"]]


def part_ageing(cfg: dict, derived: Path, inputs: Inputs, resources, seed: int, notes: list) -> dict:
    out = {}
    real = inputs.csv("43_stage3_context", "ageing_vs_context", required=False)
    if not real.empty:
        out["ageing_real"] = mv.real_width_ratio(real)
    else:
        notes.append("Part 2: ageing_vs_context.csv of script 43 not found: no real width ratios")
    daily = real_daily(derived, cfg["model"], cfg["reference"], int(cfg["days"]), int(cfg["min_origins_per_day"]))
    params = dict(cfg["defaults"])
    params.setdefault("reference_sd", params["day_sd"])
    if not daily.empty:
        calibration = mv.variance_components(daily, cfg["model"], cfg["reference"])
        out["ageing_calibration"] = calibration
        row = calibration[calibration["horizon_min"] == int(cfg["horizon_min"])]
        if not row.empty:
            r = row.iloc[0]
            params = {k: float(r[k]) for k in ("day_sd", "reference_day_sd", "correlation", "phi")}
            params["reference_sd"] = params.pop("reference_day_sd")
            params["slope_sd_pp"] = float(r["slope_sd_pp"]) if np.isfinite(r["slope_sd_pp"]) else \
                float(cfg["defaults"]["slope_sd_pp"])
            notes.append(f"Part 2: noise calibrated on the real daily skill at {cfg['horizon_min']} min: "
                         + ", ".join(f"{k} {v:.3f}" for k, v in params.items()))
    else:
        notes.append("Part 2: day metrics of scripts 40 and 43 not found: synthetic noise from the config defaults "
                     + ", ".join(f"{k} {v}" for k, v in params.items()))
    real_correlation = round(float(params["correlation"]), 2)
    correlations = sorted(set(float(c) for c in cfg["correlations"]) | {real_correlation})
    cases = [{"seed": seed, "repetitions": int(cfg["repetitions"]), "patients": int(cfg["patients"]),
              "days": int(cfg["days"]), "slope_pp": float(slope), "correlation": c,
              "day_sd": float(params["day_sd"]), "reference_sd": float(params["reference_sd"]),
              "phi": float(params["phi"]), "slope_sd_pp": float(params["slope_sd_pp"]),
              "replicates": int(cfg["bootstrap_replicates"]), "min_weeks": int(cfg["min_weeks"])}
             for c in correlations for slope in cfg["slopes_pp_per_week"]]
    print(f"Part 2: {len(cases)} synthetic settings x {cfg['repetitions']} cohorts")
    rows = pd.DataFrame([r for rows in parallel_map(mv.ageing_case, cases, resources.n_workers,
                                                    label="synthetic cohorts") for r in rows])
    out["ageing_simulation"] = rows
    out["ageing_simulation_summary"] = mv.simulation_summary(rows).assign(
        calibrated=lambda t: np.isclose(t["correlation"], real_correlation))
    out["ageing_parameters"] = pd.DataFrame([params | {"source": "real day metrics" if not daily.empty else
                                                       "config defaults"}])
    return out


# ============================================================================ 3. verdicts


def part_verdicts(cfg: dict, inputs: Inputs, replicates: int, seed: int) -> dict:
    by_patient = inputs.csv("12_stage0_population_transfer", "metrics_by_patient")
    noise = inputs.csv("12_stage0_population_transfer", "noise_floor")
    comparisons = mv.seed_comparisons(by_patient, noise, [float(e) for e in cfg["effects_dmd"]],
                                      [float(e) for e in cfg["effects_pp"]], replicates, seed, cfg.get("period"))
    print(f"Part 3: {len(comparisons)} comparisons of a model with itself")
    return {"verdicts_comparisons": comparisons, "verdicts_calibration": mv.calibration_summary(comparisons),
            "verdicts_power": mv.power_summary(comparisons)}


# ============================================================================ 4. pairing


def part_pairing(cfg: dict, derived: Path, inputs: Inputs, replicates: int, seed: int, notes: list) -> dict:
    out = {}
    by_patient = inputs.csv("12_stage0_population_transfer", "metrics_by_patient")
    paired = inputs.csv("12_stage0_population_transfer", "paired_comparisons")
    skill = by_patient.groupby(["period", "model", "horizon_min", "patient_id"], as_index=False)["skill"].mean()
    pairs = list(dict.fromkeys(zip(paired["model"], paired["against"])))
    out["pairing_stage0"] = mv.pairing_efficiency(skill, pairs, ["period", "horizon_min"], replicates, seed)
    path = derived / "runs" / "50_confirmation_runs" / "day_metrics.parquet"
    replication = inputs.csv("51_confirmation_analysis", "replication", required=False)
    if not path.exists() or replication.empty:
        notes.append("Part 4: day metrics of script 50 or replication.csv of script 51 not found: stage 0 only")
        return out
    rep = replication[(replication["finding"] == "gain_over_never") & ~replication["model"].map(is_twin)]
    chosen = rep[["cohort", "model", "policy", "against"]].drop_duplicates()
    days = pd.read_parquet(path, columns=["cohort", "patient_id", "model", "policy", "horizon_min", "day", "n", "sse"])
    days = days[days["cohort"].isin(set(chosen["cohort"]))]
    confirmation, crossover = [], []
    for (cohort, model), c in chosen.groupby(["cohort", "model"]):
        d = days[(days["cohort"] == cohort) & days["model"].isin([model, "persistence"])]
        if d.empty:
            continue
        arms = list(zip(c["policy"], c["against"]))
        confirmation.append(mv.pairing_efficiency(pr.patient_skill(d.assign(period="all")), arms, ["horizon_min"],
                                                  replicates, seed, arm="policy").assign(cohort=cohort, model=model))
        crossover.append(mv.crossover_efficiency(d, arms, ["horizon_min"], replicates, seed,
                                                 int(cfg["crossover_draws"])).assign(cohort=cohort, model=model))
        print(f"Part 4: {cohort} {model}")
    out["pairing_confirmation"] = pd.concat(confirmation, ignore_index=True) if confirmation else pd.DataFrame()
    out["crossover_confirmation"] = pd.concat(crossover, ignore_index=True) if crossover else pd.DataFrame()
    return out


# ============================================================================ 5. runs


def part_runs(derived: Path, inputs: Inputs, compare: Path | None, notes: list) -> dict:
    costs = mv.run_costs(inputs.results, derived)
    out = {"runs_cost": costs}
    if not costs.empty:
        stage = costs.groupby("stage", sort=False).agg(scripts=("script", "count"), fits=("fits", "sum"),
                                                       fit_compute_hours=("fit_compute_hours", "sum"),
                                                       wall_hours=("wall_seconds", lambda s: s.sum() / 3600))
        out["stage_cost"] = stage.reset_index()
    if compare is not None:
        if not compare.exists():
            raise SystemExit(f"--compare {compare}: folder not found")
        scripts = mv.rerun_scripts(inputs.results, compare.resolve(), exclude=(SCRIPT_ID,))
        if scripts:
            out["runs_identity"] = mv.compare_outputs(inputs.results, compare.resolve(), scripts)
            notes.append(f"Part 5: scripts run again since {compare}: {', '.join(scripts)}")
        else:
            notes.append(f"Part 5: no script was run again since {compare} (same run manifests): nothing to compare")
    else:
        notes.append("Part 5: no --compare folder: the identity of outputs across runs is not checked")
    return out


# ============================================================================ figures


def figures(tables: dict, cfg: dict, out: ScriptOutput) -> None:
    for name, maker in (("fig_schema", fig_schema), ("fig_designs", fig_designs), ("fig_ageing", fig_ageing),
                        ("fig_verdicts", fig_verdicts), ("fig_pairing", fig_pairing)):
        made = maker(tables, cfg)
        if made is None:
            continue
        fig, data = made
        out.figure(fig, name, dpi=600, formats=("pdf", "png"), exact_size=True)
        out.csv(data, f"{name}_data")


LEARNS, SCORED, UNUSED = PAPER_FAMILIES["lightgbm"][1], PAPER_FAMILIES["lstm"][1], "#d9d8d4"


def fig_schema(tables: dict, cfg: dict):
    """What each design learns from and is scored on, over one patient's record: the training window before the
    anchor and the test days after it (drawn for a refit every 21 days, for legibility)."""
    window, test, every, share = 14, 84, 21, float(cfg["designs"]["held_out_share"])
    rng = np.random.default_rng(7)
    fig = paper_figure(62)
    ax = fig.subplots()
    rows, data = [], []

    def bar(y, start, end, colour, height=0.56, kind=""):
        ax.add_patch(plt.Rectangle((start, y - height / 2), end - start, height, color=colour, linewidth=0))
        data.append({"row": y, "start_day": start, "end_day": end, "kind": kind})

    y = 0
    rows.append((y, "Simulator: frozen model"))
    bar(y, -window, 0, LEARNS, kind="learns")
    bar(y, 0, test, SCORED, kind="scored")
    fits = list(range(0, test, every))
    for i, day in enumerate(fits):
        y += 0.55 if i else 1.0
        rows.append((y, f"Simulator: refitted every {every} days" if i == 0 else ""))
        stop = fits[i + 1] if i + 1 < len(fits) else test
        bar(y, -window, day, LEARNS, height=0.36, kind="learns")
        bar(y, day, stop, SCORED, height=0.36, kind="scored")
        if stop < test:
            bar(y, stop, test, UNUSED, height=0.36, kind="later fit")
    y += 1.0
    rows.append((y, "Random split of samples"))
    bar(y, -window, test, LEARNS, kind="learns")
    for x in rng.uniform(0, test, int(share * test * 6)):
        bar(y, x, x + 0.18, SCORED, kind="scored")
    y += 1.0
    rows.append((y, "Random split of days"))
    bar(y, -window, test, LEARNS, kind="learns")
    for d in rng.choice(test, int(round(share * test)), replace=False):
        bar(y, d, d + 1, SCORED, kind="scored")
    y += 1.0
    rows.append((y, "One chronological split"))
    cut = test - share * (window + test)
    bar(y, -window, cut, LEARNS, kind="learns")
    bar(y, cut, test, SCORED, kind="scored")
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.7, linestyle="--")
    ax.text(0.6, -0.62, "anchor", color=TEXT_SECONDARY, fontsize=6.5, va="bottom")
    ax.set_xlim(-window - 1, test + 1)
    ax.set_ylim(y + 0.6, -0.75)
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-14, 0, 14, 28, 42, 56, 70, 84])
    ax.set_xlabel("Days from the anchor")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LEARNS, SCORED, UNUSED)]
    ax.legend(handles, ["Learns from", "Scored on", "Forecast by a later fit"], loc="upper center",
              bbox_to_anchor=(0.5, 1.12), ncols=3)
    return fig, pd.DataFrame(data)


def model_label(model: str) -> str:
    return PAPER_FAMILIES.get(split_name(model)[0], (model,))[0]


def fig_designs(tables: dict, cfg: dict):
    """What each design reports minus what the frozen (a) and the refitted (b) model achieve on the same origins."""
    t = tables.get("designs_inflation")
    if t is None or t.empty:
        return None
    rows = [(m, s) for m in cfg["designs"]["models"] for s in ns.SPLITS
            if ((t["model"] == m) & (t["split"] == s)).any()]
    fig = paper_figure(18 + 7 * len(rows))
    axes = fig.subplots(1, 2, sharey=True)
    data = []
    for ax, against, letter, title in ((axes[0], "frozen", "a", "Design minus frozen model"),
                                       (axes[1], "refitted", "b", "Design minus refitted model")):
        for y, (model, split) in enumerate(rows):
            for k, horizon in enumerate(sorted(t["horizon_min"].unique())):
                r = t[(t["model"] == model) & (t["split"] == split) & (t["horizon_min"] == horizon)]
                if r.empty or f"minus_{against}" not in r:
                    continue
                r = r.iloc[0]
                yy = y + (k - 1) * 0.22
                ax.plot([100 * r[f"minus_{against}_low"], 100 * r[f"minus_{against}_high"]], [yy, yy],
                        color=HORIZON_COLOURS[horizon], linewidth=0.8)
                ax.plot(100 * r[f"minus_{against}"], yy, marker=HORIZON_MARKERS[horizon], linestyle="none",
                        color=HORIZON_COLOURS[horizon], markersize=3, label=f"{horizon} min" if y == 0 else None)
                data.append({"panel": letter, "model": model, "split": split, "horizon_min": horizon,
                             "difference_pp": 100 * r[f"minus_{against}"], "low": 100 * r[f"minus_{against}_low"],
                             "high": 100 * r[f"minus_{against}_high"]})
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.6, zorder=1)
        ax.set_xlabel("Skill difference (percentage points)")
        ax.set_title(title, loc="center")
        ax.grid(axis="y", visible=False)
        panel_letter(ax, letter)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{model_label(m)}: {SPLIT_LABELS[s].lower()}" for m, s in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[1].legend(loc="lower right", title="Horizon", title_fontsize=6.5)
    return fig, pd.DataFrame(data)


def fig_ageing(tables: dict, cfg: dict):
    """Synthetic cohorts: (a) width of the interval against the reference over that of the slope alone, (b) coverage
    of both intervals, (c) how often each finds the loss, as functions of the correlation of the daily skill of the
    model and the reference (the day-to-day variation they share)."""
    s = tables.get("ageing_simulation_summary")
    if s is None or s.empty:
        return None
    loss = min(float(x) for x in cfg["ageing"]["slopes_pp_per_week"])
    model = s[(s["label"] == "model") & (s["slope_pp"] == loss)].sort_values("correlation")
    control = s[(s["label"] == "control") & (s["slope_pp"] == loss)].sort_values("correlation")
    real = tables.get("ageing_real")
    fig = paper_figure(52)
    axes = fig.subplots(1, 3)
    adjusted, raw = PAPER_FAMILIES["lightgbm"][1], TEXT_SECONDARY
    x = model["correlation"]
    axes[0].plot(x, model["width_ratio"], marker="o", color=adjusted, label="Synthetic")
    if real is not None and not real.empty:
        frozen = real[real["kind"] == "frozen"]["width_ratio"]
        if len(frozen):
            axes[0].axhspan(frozen.min(), frozen.max(), color=PAPER_FAMILIES["lstm"][1], alpha=0.15, linewidth=0,
                            label="Real, frozen models")
    calibrated = model[model.get("calibrated", False)] if "calibrated" in model else pd.DataFrame()
    params = tables.get("ageing_parameters")
    real_noise = params is not None and not params.empty and params["source"].iloc[0] == "real day metrics"
    if len(calibrated):
        axes[0].plot(calibrated["correlation"], calibrated["width_ratio"], marker="o", markersize=5,
                     markerfacecolor="white", color=adjusted, linestyle="none",
                     label="Calibrated on real data" if real_noise else "Default noise")
    axes[0].axhline(1, color=TEXT_SECONDARY, linewidth=0.5, linestyle=":")
    axes[0].set_ylim(0, max(1.05, float(np.ceil(10 * model["width_ratio"].max() + 0.5)) / 10))
    axes[0].set_ylabel("Interval width, adjusted / raw")
    axes[0].legend(loc="lower left")
    for frame, style, name in ((model, "-", "loss"), (control, "--", "control")):
        axes[1].plot(frame["correlation"], frame["adjusted_coverage"], linestyle=style, marker="o", color=adjusted,
                     label=f"Adjusted, {name}")
        axes[1].plot(frame["correlation"], frame["raw_coverage"], linestyle=style, marker="s", color=raw,
                     label=f"Raw, {name}")
    axes[1].axhline(0.95, color=TEXT_SECONDARY, linewidth=0.5, linestyle=":")
    axes[1].set_ylim(0.75, 1.0)
    axes[1].set_ylabel("Coverage of the 95% interval")
    axes[1].legend(loc="lower left", ncols=2)
    axes[2].plot(x, model["adjusted_excludes_zero"], marker="o", color=adjusted, label="Adjusted")
    axes[2].plot(x, model["raw_excludes_zero"], marker="s", color=raw, label="Raw")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel(f"Loss of {abs(loss):g} pp/week found")
    axes[2].legend(loc="upper left")
    for ax, letter in zip(axes, "abc"):
        ax.set_xlabel("Correlation of daily skill, model and reference")
        panel_letter(ax, letter)
    data = pd.concat([model.assign(series="loss"), control.assign(series="control")], ignore_index=True)
    return fig, data


def fig_verdicts(tables: dict, cfg: dict):
    """(a) Null comparisons: false findings of the interval alone and of the tie rule, per family; (b) power: how
    often each finds a known difference, by its size in DMDs."""
    calibration, power = tables.get("verdicts_calibration"), tables.get("verdicts_power")
    if calibration is None or calibration.empty:
        return None
    fig = paper_figure(52)
    axes = fig.subplots(1, 2, width_ratios=[1.2, 1])
    c = calibration[calibration["family"] != "all"].copy()
    c["label"] = c["family"].map(lambda f: PAPER_FAMILIES.get(f, (f,))[0])
    y = np.arange(len(c))
    axes[0].barh(y - 0.18, 100 * c["interval_false_finding"], height=0.34, color=TEXT_SECONDARY,
                 label="Interval alone")
    axes[0].barh(y + 0.18, 100 * c["rule_false_finding"], height=0.34, color=PAPER_FAMILIES["lightgbm"][1],
                 label="Tie rule with the MDD")
    for yy, a, b in zip(y, 100 * c["interval_false_finding"], 100 * c["rule_false_finding"]):
        axes[0].text(a + 0.8, yy - 0.18, f"{a:.0f}", va="center", fontsize=6, color=TEXT_SECONDARY)
        axes[0].text(b + 0.8, yy + 0.18, f"{b:.0f}", va="center", fontsize=6, color=TEXT_SECONDARY)
    axes[0].set_yticks(y, c["label"])
    axes[0].set_ylim(len(c) - 0.5, -0.5)
    axes[0].axvline(5, color=TEXT_SECONDARY, linewidth=0.5, linestyle=":")
    axes[0].set_xlabel("Null comparisons with a finding (%)")
    axes[0].set_xlim(0, max(10.0, 100 * float(c["interval_false_finding"].max()) + 8))
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", visible=False)
    p = power[power["effect_kind"] == "dmd"].sort_values("effect")
    axes[1].plot(p["effect"], 100 * p["rule_better"], marker="o", color=PAPER_FAMILIES["lightgbm"][1],
                 label="Tie rule: better")
    axes[1].plot(p["effect"], 100 * p["rule_tie"], marker="s", color=PAPER_FAMILIES["lightgbm"][1], linestyle="--",
                 label="Tie rule: tie")
    axes[1].plot(p["effect"], 100 * p["interval_better"], marker="^", color=TEXT_SECONDARY,
                 label="Interval alone: better")
    axes[1].set_xlabel("Added difference (multiples of the MDD)")
    axes[1].set_ylabel("Comparisons (%)")
    axes[1].set_ylim(-2, 102)
    axes[1].legend(loc="center right")
    for ax, letter in zip(axes, "ab"):
        panel_letter(ax, letter)
    return fig, pd.concat([calibration.assign(panel="a"), power.assign(panel="b")], ignore_index=True)


def fig_pairing(tables: dict, cfg: dict):
    """Width of the interval of an unpaired design over that of the paired one: stage 0 (patients as two independent
    groups) and the confirmation (the same, and a crossover in time)."""
    parts = []
    s0 = tables.get("pairing_stage0")
    if s0 is not None and not s0.empty:
        parts.append(("Stage 0: independent groups", s0["width_ratio"]))
    conf, cross = tables.get("pairing_confirmation"), tables.get("crossover_confirmation")
    if conf is not None and not conf.empty:
        parts.append(("Confirmation: independent groups", conf["width_ratio"]))
    if cross is not None and not cross.empty:
        parts.append(("Confirmation: crossover in time", cross["width_ratio"]))
    if not parts:
        return None
    fig = paper_figure(14 + 9 * len(parts), width="single")
    ax = fig.subplots()
    rng = np.random.default_rng(0)
    data = []
    for y, (name, ratios) in enumerate(parts):
        r = ratios.dropna().to_numpy()
        ax.plot(r, y + rng.uniform(-0.18, 0.18, len(r)), marker="o", linestyle="none", markersize=2.2,
                color=PAPER_FAMILIES["lightgbm"][1], alpha=0.55, markeredgewidth=0)
        ax.plot([np.median(r)] * 2, [y - 0.3, y + 0.3], color=TEXT_SECONDARY, linewidth=1.2)
        data += [{"design": name, "width_ratio": v} for v in r]
    ax.axvline(1, color=TEXT_SECONDARY, linewidth=0.5, linestyle=":")
    ax.set_yticks(range(len(parts)), [p[0] for p in parts])
    ax.set_ylim(len(parts) - 0.5, -0.5)
    ax.set_xscale("log")
    ticks = [v for v in (1, 1.5, 2, 3, 4, 6, 8, 12, 16) if v <= max(2.0, max(float(np.nanmax(p[1])) for p in parts))
             * 1.1]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Interval width, unpaired / paired")
    ax.grid(axis="y", visible=False)
    return fig, pd.DataFrame(data)


# ============================================================================ summary


def pp(value: float) -> str:
    return f"{100 * value:+.2f}" if np.isfinite(value) else "--"


def write_summary(tables: dict, cfg: dict, notes: list, args, release, resources) -> str:
    lines = [f"# {SCRIPT_ID} - Validation of the method", "", f"Code release {release}. Inputs from `{args.results}`.",
             resources.describe(), "", "## Notes", ""]
    lines += [f"- {n}" for n in notes] or ["- none"]
    t = tables.get("designs_inflation")
    if t is not None and not t.empty:
        lines += ["", "## 1. Common designs against the simulator", "",
                  "Skill (pp) of each design's estimate and of the simulator's frozen and refitted models on the same "
                  "held-out origins; differences with 95 % bootstrap intervals over patients.", ""]
        view = pd.DataFrame({
            "model": t["model"], "design": t["split"], "h": t["horizon_min"], "patients": t["patients"],
            "design_skill": (100 * t["skill_design"]).round(2), "frozen": (100 * t.get("skill_frozen")).round(2),
            "refitted": (100 * t.get("skill_refitted")).round(2),
            "minus_frozen": [f"{pp(m)} ({pp(lo)}, {pp(hi)})" for m, lo, hi in
                             zip(t["minus_frozen"], t["minus_frozen_low"], t["minus_frozen_high"])],
            "minus_refitted": [f"{pp(m)} ({pp(lo)}, {pp(hi)})" for m, lo, hi in
                               zip(t.get("minus_refitted"), t.get("minus_refitted_low"),
                                   t.get("minus_refitted_high"))]})
        lines.append(markdown_table(view))
        det = tables.get("designs_determinism")
        if det is not None and not det.empty and "identical" in det:
            lines += ["", f"Determinism: {int(det['identical'].sum())} of {len(det)} repeated runs identical "
                          f"(largest SSE difference {det['largest_sse_difference'].max():.3g})."]
    s = tables.get("ageing_simulation_summary")
    if s is not None and not s.empty:
        lines += ["", "## 2. The ageing estimator", "", "Synthetic cohorts (points per week): bias, coverage and "
                  "width of the interval of the slope against the reference (adjusted) and of the slope alone (raw).",
                  "", markdown_table(s[["correlation", "slope_pp", "label", "adjusted_bias", "adjusted_coverage",
                                        "adjusted_width", "raw_bias", "raw_coverage", "raw_width", "width_ratio",
                                        "adjusted_excludes_zero", "raw_excludes_zero"]].round(3))]
        params = tables.get("ageing_parameters")
        if params is not None:
            lines += ["", "Noise of the synthetic cohorts:", "", markdown_table(params.round(4))]
    real = tables.get("ageing_real")
    if real is not None and not real.empty:
        lines += ["", "Real data (script 43): width of the adjusted interval over the raw one", "",
                  markdown_table(real.groupby(["kind", "horizon_min"])["width_ratio"].describe()[
                      ["count", "min", "50%", "max"]].round(3).reset_index())]
    c = tables.get("verdicts_calibration")
    if c is not None and not c.empty:
        lines += ["", "## 3. Verdicts with the DMD", "", "Null comparisons (a model against itself with another "
                  "seed; the DMD from the other seed pairs):", "", markdown_table(c.round(3)), "",
                  "Known differences added:", "", markdown_table(tables["verdicts_power"].round(3))]
    for name, title in (("pairing_stage0", "## 4. The paired design: stage 0"),
                        ("pairing_confirmation", "Confirmation, refitted against frozen"),
                        ("crossover_confirmation", "Confirmation, crossover in time")):
        p = tables.get(name)
        if p is not None and not p.empty:
            lines += ["", title, "", f"Width ratio unpaired / paired: median {p['width_ratio'].median():.2f} "
                      f"(range {p['width_ratio'].min():.2f}-{p['width_ratio'].max():.2f}) over {len(p)} comparisons."]
            if "correlation" in p:
                lines.append(f"Correlation of the arms across patients: median {p['correlation'].median():.2f}; "
                             f"patients per arm of a parallel design for the same precision: median "
                             f"{p['parallel_patients_per_arm'].median():.0f} (paired: median "
                             f"{p['patients'].median():.0f}).")
    stage = tables.get("stage_cost")
    if stage is not None and not stage.empty:
        lines += ["", "## 5. Runs", "", "Cost per stage (fits of the fit logs; wall time of the last run of each "
                  "script, cached work not counted):", "", markdown_table(stage.round(2))]
    identity = tables.get("runs_identity")
    if identity is not None and not identity.empty:
        lines += ["", "Identity of the outputs against the other results folder:", "",
                  markdown_table(identity["status"].value_counts().rename_axis("status").reset_index())]
        different = identity[identity["status"] == "different"]
        if len(different):
            lines += ["", "Different tables:", "", markdown_table(different.head(40))]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
