"""61 - Q4: does the regularity of a patient's habits go with how the model ages and how much refitting gains?

Relates the regularity indices of script 60 (startup window) to what happened
to every confirmation patient in script 50, for every model and its
glucose-only twin (HANDOFF 7.7, H3, D94; ``eval.regularity_analysis``):

- outcomes per patient, model and horizon on the first ``outcome_days`` test
  days: the skill of the frozen model, how fast it ages (slope of its daily
  skill), the gain of refitting at the optimal interval over never and, where
  the cohort refits every day too, what the optimal interval loses against
  every day (H3: closer to 0 when the patient is more regular);
- the same outcomes of every full model minus its twin (D90): does what insulin
  and carbohydrates add depend on how regular the patient is?
- association: least squares on the standardised index, adjusted for the
  glycaemic control at startup (time in range, CV and mean glucose) and the
  cohort, HC3 errors; the primary indices (D94) corrected together with Holm,
  the others among themselves; per cohort too for the primary indices;
- outcome by tertile of each primary index, and RMSE and Clarke zones of every
  arm by tertile of ``regularity.tertile_index`` (rule R6).

Inputs:
    <derived_data>/runs/60_regularity_indices/indices.parquet
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet
    results/50_confirmation_runs/run_manifest.json        (models, intervals and mechanisms)

Outputs (results/61_regularity_analysis/):
    patient_outcomes.csv     per cohort, patient, model (and full minus twin) and horizon (0: mean): the outcomes
    associations.csv         per outcome, model, horizon and index: patients, coefficient per SD (points), 95 % CI,
                             p, Holm p, Spearman, coefficient towards the more regular side, agreement with H3
    associations_by_cohort.csv   the primary indices per cohort (cohorts with enough patients)
    tertiles.csv             mean outcome by tertile of each primary index (points, 95 %)
    metrics_by_tertile.csv   per cohort, model, arm, horizon and tertile: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    associations.png, h3_interval.png, tertiles.png, inputs_by_regularity.png
    summary.md, run_manifest.json

Usage:
    python scripts/61_regularity_analysis.py
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

from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.pipeline import WarningRecorder  # noqa: E402
from t1dretrain.eval import regularity_analysis as ra  # noqa: E402
from t1dretrain.eval.stage0_report import SUMS, color, finish, label  # noqa: E402
from t1dretrain.features.regularity import ADJUSTERS, INDICES  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.stage0 import is_twin  # noqa: E402

SCRIPT_ID = "61_regularity_analysis"
INDICES_SOURCE = "60_regularity_indices"
RUNS_SOURCE = "50_confirmation_runs"
HORIZON_LABEL = {0: "mean over horizons"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def run_associations(job: dict) -> dict:
    """The associations of one model (in a worker process)."""
    import time

    started = time.time()
    with WarningRecorder() as recorder:
        try:
            table = ra.associations(job["outcomes"], job["indices"], job["primary"], job["min_patients"])
            by_cohort = []
            for cohort, g in job["outcomes"].groupby("cohort"):
                part = ra.associations(g, job["indices"], job["primary"], job["min_patients"])
                if not part.empty:
                    by_cohort.append(part[part["primary"]].assign(cohort=cohort))
            by_cohort = pd.concat(by_cohort, ignore_index=True) if by_cohort else pd.DataFrame()
            error = ""
        except Exception as exc:  # noqa: BLE001
            table, by_cohort, error = pd.DataFrame(), pd.DataFrame(), f"{type(exc).__name__}: {exc}"[:300]
    return {"model": job["model"], "table": table, "by_cohort": by_cohort, "error": error,
            "warnings": recorder.text()[:300], "seconds": round(time.time() - started, 1)}


def main() -> None:
    import json

    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    settings = dict(config["regularity"])
    cfg = config.get("stage2_analysis", {})
    min_origins, clip = int(cfg.get("min_origins_per_day", 12)), tuple(cfg.get("skill_clip", [-1.0, 1.0]))
    primary, min_patients = list(settings["primary"]), int(settings["min_patients"])
    for path, script in ((derived / "runs" / INDICES_SOURCE / "indices.parquet", INDICES_SOURCE),
                         (derived / "runs" / RUNS_SOURCE / "day_metrics.parquet", RUNS_SOURCE)):
        if not path.exists():
            raise SystemExit(f"{path} not found: run scripts/{script}.py first")
    indices = pd.read_parquet(derived / "runs" / INDICES_SOURCE / "indices.parquet")
    days = pd.read_parquet(derived / "runs" / RUNS_SOURCE / "day_metrics.parquet")
    manifest = json.loads((Path(out.dir).parent / RUNS_SOURCE / "run_manifest.json").read_text())["settings"]
    plan, models = manifest["plan"], [m for m in manifest["models"] if m in set(days["model"])]
    keep = ["cohort", "patient_id"] + [c for c in list(INDICES) + list(ADJUSTERS) if c in indices]
    indices = indices[keep]
    print(f"{len(indices)} patients with indices; {len(days)} day rows of script 50; models {models}")

    outcomes = ra.with_contrasts(ra.patient_outcomes(days, plan, int(settings["outcome_days"]), min_origins, clip))
    names = list(dict.fromkeys(outcomes["model"]))
    jobs = [{"model": m, "outcomes": outcomes[outcomes["model"] == m], "indices": indices, "primary": primary,
             "min_patients": min_patients} for m in names]
    results = parallel_map(run_associations, jobs, resources.n_workers, label="association models")
    table = concat([r["table"] for r in results])
    by_cohort = concat([r["by_cohort"] for r in results])
    fit_rows = pd.DataFrame([{k: r[k] for k in ("model", "error", "warnings", "seconds")} for r in results])
    present = [i for i in primary if i in indices and indices[i].notna().any()]
    bands = ra.tertiles(outcomes, indices, present)
    metrics = metrics_by_tertile(days, indices, settings["tertile_index"], int(settings["outcome_days"]))

    out.csv(outcomes.round(6), "patient_outcomes")
    out.csv(table.round(5), "associations")
    out.csv(by_cohort.round(5), "associations_by_cohort")
    out.csv(bands.round(4), "tertiles")
    out.csv(metrics, "metrics_by_tertile")
    full = [m for m in models if not is_twin(m)]
    if not table.empty:
        out.figure(plot_heatmap(table, full, primary), "associations")
        out.figure(plot_inputs(table, full, primary), "inputs_by_regularity")
    h3 = plot_h3(outcomes, indices, full, present)
    if h3 is not None:
        out.figure(h3, "h3_interval")
    if not bands.empty:
        out.figure(plot_tertiles(bands, full, present), "tertiles")
    out.text(write_summary(table, by_cohort, bands, fit_rows, outcomes, indices, full, primary, settings, manifest,
                           release, resources), "summary.md")
    out.manifest(settings={**settings, "models": models, "plan": plan, "outcomes": ra.OUTCOMES,
                           "expected_signs": ra.EXPECTED, "source_release": manifest.get("confirmation_release")},
                 notes="Indices of script 60; outcomes from the day metrics of script 50.")
    print(f"Done. Results in {out.dir}")


def concat(frames: list) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def metrics_by_tertile(days: pd.DataFrame, indices: pd.DataFrame, name: str, outcome_days: int) -> pd.DataFrame:
    """R6: n, RMSE, MAE, MARD, bias and Clarke zones per cohort, model, arm, horizon and tertile of ``name``."""
    if name not in indices:
        return pd.DataFrame()
    groups = ra.tertile_of(indices, name)
    d = days[days["days_since_anchor"] < outcome_days].merge(groups, on=["cohort", "patient_id"], how="inner")
    if d.empty:
        return pd.DataFrame()
    table = finish(d.groupby(["cohort", "model", "policy", "horizon_min", "tertile"], as_index=False)[SUMS].sum())
    return table.assign(index=name).round(4)


# ============================================================================ figures


def mean_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["horizon_min"] == 0]


def plot_heatmap(table: pd.DataFrame, models: list[str], primary: list[str]):
    """Coefficient per SD towards the more regular side (points; mean over horizons), every index and outcome of
    the full models; a dot: Holm p < 0.05."""
    import matplotlib.pyplot as plt

    t = mean_rows(table)
    t = t[t["model"].isin(models) & t["coef_regular"].notna()]
    outcomes = [o for o in ra.OUTCOMES if o in set(t["outcome"])]
    names = [i for i in INDICES if i in set(t["index"])]
    shown = [m for m in models if m in set(t["model"])]
    fig, axes = plt.subplots(1, len(outcomes), figsize=(2.2 + 0.55 * len(shown) * len(outcomes), 0.32 * len(names) + 2.2),
                             squeeze=False)
    limit = max(0.5, float(np.nanpercentile(np.abs(t["coef_regular"]), 95))) if not t.empty else 1.0
    image = None
    for k, (ax, outcome) in enumerate(zip(axes[0], outcomes)):
        o = t[t["outcome"] == outcome]
        grid = o.pivot_table(index="index", columns="model", values="coef_regular").reindex(index=names, columns=shown)
        hits = o.pivot_table(index="index", columns="model", values="p_holm").reindex(index=names, columns=shown)
        image = ax.imshow(grid.to_numpy(float), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        for i in range(len(names)):
            for j in range(len(shown)):
                p = hits.iloc[i, j]
                if np.isfinite(p) and p < 0.05:
                    ax.scatter(j, i, s=12, color="black")
        ax.set_xticks(range(len(shown)), [label(m) for m in shown], rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(names)), [n + (" *" if n in primary else "") for n in names] if k == 0 else
                      [""] * len(names), fontsize=6)
        ax.set_title(outcome, loc="left", fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), shrink=0.7, label="points per SD more regular")
    fig.suptitle("Regularity and outcomes (mean over horizons; adjusted for TIR, CV, mean glucose and cohort; "
                 "dot: Holm p < 0.05; *: primary index)", fontweight="bold", x=0.01, ha="left", fontsize=9)
    return fig


def plot_inputs(table: pd.DataFrame, models: list[str], primary: list[str]):
    """D90: how what insulin and carbohydrates add (full minus twin) changes with each primary index."""
    t = mean_rows(table)
    t = t[t["model"].str.endswith(ra.INPUTS) & t["index"].isin(primary) & t["coef"].notna()]
    outcomes = [o for o in ra.OUTCOMES if o in set(t["outcome"])]
    fig, axes = model_axes(len(outcomes), 4.6, 3.0, ncols=len(outcomes) or 1)
    for ax, outcome in zip(axes, outcomes):
        o = t[t["outcome"] == outcome]
        contrasts = [m + ra.INPUTS for m in models if m + ra.INPUTS in set(o["model"])]
        for i, name in enumerate(primary):
            for j, contrast in enumerate(contrasts):
                r = o[(o["model"] == contrast) & (o["index"] == name)]
                if r.empty:
                    continue
                r = r.iloc[0]
                sign = {"lower": -1.0, "higher": 1.0}.get(r["regular_when"], 1.0)
                y = i + (j - (len(contrasts) - 1) / 2) * 0.12
                lo, hi = sorted((sign * r["low"], sign * r["high"]))
                base = contrast[: -len(ra.INPUTS)]
                ax.plot([lo, hi], [y, y], color=color(base), lw=1.1, alpha=0.6)
                ax.scatter(sign * r["coef"], y, color=color(base), s=12, label=label(base) if i == 0 else None,
                           marker="o" if r["p_holm"] < 0.05 else "s")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(range(len(primary)), primary, fontsize=7)
        ax.set_title(outcome, loc="left", fontsize=8)
        ax.set_xlabel("points per SD more regular")
    if outcomes:
        axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Insulin and carbohydrates by regularity (D90): full model minus its glucose-only twin "
                 "(circle: Holm p < 0.05)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_h3(outcomes: pd.DataFrame, indices: pd.DataFrame, models: list[str], names: list[str]):
    """H3: what the optimal interval loses against every day, against each primary index (mean over horizons)."""
    d = outcomes[(outcomes["horizon_min"] == 0) & outcomes["model"].isin(models)].dropna(subset=["loss_daily"])
    d = d.merge(indices, on=["cohort", "patient_id"], how="inner")
    names = [n for n in names if n in d and d[n].notna().sum() >= 10]
    if d.empty or not names:
        return None
    fig, axes = model_axes(len(names), 4.4, 3.2, ncols=len(names))
    for ax, name in zip(axes, names):
        for model in models:
            m = d[d["model"] == model].dropna(subset=[name])
            if len(m) < 10:
                continue
            ax.scatter(m[name], 100 * m["loss_daily"], s=6, color=color(model), alpha=0.35, edgecolors="none")
            slope, intercept = np.polyfit(m[name].to_numpy(float), 100 * m["loss_daily"].to_numpy(float), 1)
            x = np.linspace(m[name].min(), m[name].max(), 20)
            ax.plot(x, intercept + slope * x, color=color(model), lw=1.3, label=label(model))
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        regular = INDICES[name][2]
        ax.set_xlabel(f"{name} (more regular: {regular})", fontsize=8)
        ax.set_ylabel("optimal interval - every day (points)")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("H3: do more regular patients lose less with the optimal interval? (unadjusted lines; the adjusted "
                 "test is in associations.csv)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_tertiles(bands: pd.DataFrame, models: list[str], names: list[str]):
    """Gain of refitting and ageing by tertile of each primary index (mean over horizons, full models)."""
    b = bands[(bands["horizon_min"] == 0) & bands["model"].isin(models)]
    panels = [(o, n) for o in ("gain_refit", "ageing") for n in names if not b[(b["outcome"] == o) &
                                                                              (b["index"] == n)].empty]
    fig, axes = model_axes(len(panels), 4.0, 2.8, ncols=len(names) or 1)
    order = ["low", "middle", "high"]
    for ax, (outcome, name) in zip(axes, panels):
        p = b[(b["outcome"] == outcome) & (b["index"] == name)]
        for j, model in enumerate(models):
            m = p[p["model"] == model].set_index("tertile").reindex(order)
            x = np.arange(3) + (j - (len(models) - 1) / 2) * 0.1
            ax.errorbar(x, m["mean"], yerr=[m["mean"] - m["low"], m["high"] - m["mean"]], fmt="o-", ms=3, lw=1,
                        color=color(model), label=label(model), capsize=0)
        ax.set_xticks(range(3), order, fontsize=7)
        ax.set_xlabel(f"tertile of {name} (more regular: {INDICES[name][2]})", fontsize=7)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(ra.OUTCOMES[outcome], loc="left", fontsize=7)
        ax.set_ylabel("points per week" if outcome == "ageing" else "points")
    if panels:
        axes[0].legend(fontsize=5, loc="best")
    fig.suptitle("Outcome by tertile of each primary index within its cohort (mean over horizons; 95 %)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def cell(r) -> str:
    if not np.isfinite(r.get("coef", np.nan)):
        return ""
    value = r["coef_regular"] if np.isfinite(r.get("coef_regular", np.nan)) else r["coef"]
    sign = -1.0 if r.get("regular_when") == "lower" else 1.0
    lo, hi = sorted((sign * r["low"], sign * r["high"])) if np.isfinite(r.get("coef_regular", np.nan)) \
        else (r["low"], r["high"])
    return f"{value:+.2f} ({lo:+.2f} to {hi:+.2f}); p {r['p_holm']:.3f}; n {int(r['patients'])}"


def write_summary(table, by_cohort, bands, fit_rows, outcomes, indices, models, primary, settings, manifest, release,
                  resources) -> str:
    lines = ["# 61 - Q4: regularity and refitting", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"{indices['patient_id'].nunique()} patients with indices (script 60); outcomes on the first "
             f"{settings['outcome_days']} test days of script 50 (release {manifest.get('confirmation_release')}). "
             f"Coefficients in skill points per standard deviation of the index, towards the more regular side when "
             f"the index has one, adjusted for time in range, CV and mean glucose at startup and for the cohort; "
             f"Holm within each outcome, model and horizon (primary indices {primary} together, D94).", ""]
    problems = [f"{r.model}: {r.error}" for r in fit_rows.itertuples() if r.error] if not fit_rows.empty else []
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    lines += ["## Outcomes", ""] + [f"- `{k}`: {v}" for k, v in ra.OUTCOMES.items()] + \
        [f"- `<model>{ra.INPUTS}`: the full model minus its glucose-only twin (D90)", ""]
    if not table.empty:
        t = mean_rows(table)
        h = t[t["primary"] & t["model"].isin(models) & t["outcome"].isin(list(ra.EXPECTED))].copy()
        if not h.empty:
            h["result"] = [cell(r) for _, r in h.iterrows()]
            wide = h.pivot_table(index=["outcome", "index"], columns="model", values="result", aggfunc="first")
            wide = wide[[m for m in models if m in wide.columns]]
            lines += ["## H3 with the primary indices (full models; mean over horizons)", "",
                      "Expected when more regular: `ageing` +, `gain_refit` -, `loss_daily` + (closer to 0). "
                      "Each cell: points per SD more regular (95 % CI); Holm p; patients.", "",
                      markdown_table(wide.reset_index()), ""]
            agree = h[h["agrees"] != ""]
            lines += [f"Holm p < 0.05 in {len(agree)} of {len(h)} primary comparisons: "
                      f"{int((agree['agrees'] == 'yes').sum())} in the direction of H3 and "
                      f"{int((agree['agrees'] == 'no').sum())} against it.", ""]
        s = t[t["primary"] & t["model"].isin(models) & (t["outcome"] == "skill_never")].copy()
        if not s.empty:
            s["result"] = [cell(r) for _, r in s.iterrows()]
            wide = s.pivot_table(index="index", columns="model", values="result", aggfunc="first")
            lines += ["## Skill of the frozen model and regularity (primary indices; mean over horizons)", "",
                      markdown_table(wide[[m for m in models if m in wide.columns]].reset_index()), ""]
        c = t[t["primary"] & t["model"].str.endswith(ra.INPUTS)].copy()
        if not c.empty:
            c["result"] = [cell(r) for _, r in c.iterrows()]
            wide = c.pivot_table(index=["outcome", "index"], columns="model", values="result", aggfunc="first")
            lines += ["## What insulin and carbohydrates add, by regularity (full minus twin; D90)", "",
                      "Positive: the more regular the patient, the more the full model's outcome exceeds its twin's.",
                      "", markdown_table(wide.reset_index()), ""]
        other = t[~t["primary"] & (t["group"] != "adjuster") & (t["p_holm"] < 0.05) &
                  ~t["model"].str.endswith(ra.INPUTS)]
        if not other.empty:
            other = other.assign(result=[cell(r) for _, r in other.iterrows()])
            lines += ["## Other indices with Holm p < 0.05 (mean over horizons)", "",
                      markdown_table(other[["outcome", "model", "index", "group", "result"]]), ""]
        adj = t[(t["group"] == "adjuster") & t["model"].isin(models) & (t["p_holm"] < 0.05)]
        if not adj.empty:
            adj = adj.assign(result=[f"{r['coef']:+.2f} ({r['low']:+.2f} to {r['high']:+.2f}); n {int(r['patients'])}"
                                     for _, r in adj.iterrows()])
            lines += ["## Glycaemic control on its own (points per SD; Holm p < 0.05; mean over horizons)", "",
                      markdown_table(adj[["outcome", "model", "index", "result"]]), ""]
    if not by_cohort.empty:
        b = mean_rows(by_cohort)
        b = b[b["model"].isin(models) & b["outcome"].isin(list(ra.EXPECTED)) & b["coef"].notna()]
        shown = b[b["p_holm"] < 0.05]
        lines += ["## Per cohort (primary indices; full models; mean over horizons; Holm p < 0.05)", "",
                  f"{len(shown)} of {len(b)} cohort associations; every one is in `associations_by_cohort.csv`.", ""]
        if not shown.empty:
            shown = shown.assign(result=[cell(r) for _, r in shown.iterrows()])
            lines += [markdown_table(shown[["cohort", "outcome", "index", "model", "result"]]), ""]
    lines += ["## Checks to look at", "",
              "- `associations.png`: which indices go with which outcome, and whether all models agree.",
              "- `h3_interval.png`: whether more regular patients lose less with the optimal interval.",
              "- `tertiles.png`: the size of the differences between the least and the most regular third.",
              "- `inputs_by_regularity.png`: whether insulin and carbohydrates add more in irregular patients.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
